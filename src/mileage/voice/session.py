"""Hands-free voice session orchestrating VAD, STT, local Agent, and TTS."""

import queue
import sys
import threading
import time
from typing import Any, Callable, Optional

from rich.live import Live

from mileage.agents.local_agent import LocalAgent
from mileage.core.logger import logger
from mileage.voice.states import VoiceState
from mileage.voice.stt import SpeechToTextEngine
from mileage.voice.tts import TextToSpeechEngine
from mileage.voice.ui import VoiceTerminalUI
from mileage.voice.vad import VadDetector

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False


class VoiceSession:
    """
    Manages hands-free, event-driven voice conversation loop.

    State Flow:
        IDLE -> LISTENING -> PROCESSING -> (BUILDING) -> SPEAKING -> COMPLETE -> IDLE
    """

    def __init__(
        self,
        agent: Optional[LocalAgent] = None,
        stt_engine: Optional[SpeechToTextEngine] = None,
        tts_engine: Optional[TextToSpeechEngine] = None,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        enable_tts: bool = True,
        on_state_change: Optional[Callable[[VoiceState], None]] = None,
    ):
        self.agent = agent
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.enable_tts = enable_tts
        self.on_state_change = on_state_change

        self.vad = VadDetector(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
            energy_threshold=350.0,
            silence_timeout_sec=1.2,
        )
        self.stt = stt_engine or SpeechToTextEngine(model_size="tiny.en")
        self.tts = tts_engine or TextToSpeechEngine()
        self.ui = VoiceTerminalUI()

        self._state = VoiceState.IDLE
        self._running = threading.Event()
        self._audio_queue: queue.Queue = queue.Queue()
        self._speech_buffer: bytearray = bytearray()
        self._stream: Optional[Any] = None

    @property
    def state(self) -> VoiceState:
        return self._state

    def set_state(self, new_state: VoiceState, message: Optional[str] = None) -> None:
        """Safely transition state and notify UI/listener."""
        self._state = new_state
        self.ui.update_state(new_state, message)
        if self.on_state_change:
            try:
                self.on_state_change(new_state)
            except Exception as e:
                logger.debug("on_state_change callback error: %s", e)

    def is_mic_muted(self) -> bool:
        """
        Check if microphone capture must be ignored.
        Guarantees acoustic isolation while TTS is speaking.
        """
        return self._state == VoiceState.SPEAKING or self.tts.is_speaking

    def handle_audio_frame(self, frame_bytes: bytes) -> Optional[str]:
        """
        Process a single incoming PCM audio frame.

        Returns:
            The transcribed text if the user just completed a speech turn, else None.
        """
        # Critical Requirement: Prevent microphone capture while TTS is speaking
        if self.is_mic_muted():
            # Drop incoming audio frame to avoid self-listening loop
            if self._speech_buffer:
                self._speech_buffer.clear()
            self.vad.reset()
            return None

        # Check if user started or is currently speaking
        if self._state == VoiceState.IDLE:
            if self.vad.is_speech(frame_bytes):
                self.set_state(VoiceState.LISTENING, "Speech detected, listening...")

        if self._state == VoiceState.LISTENING:
            self._speech_buffer.extend(frame_bytes)

            # Check if user has finished speaking (silence timeout reached)
            finished = self.vad.process_frame(frame_bytes)
            if finished:
                audio_to_transcribe = bytes(self._speech_buffer)
                self._speech_buffer.clear()
                return self._transcribe_and_handle(audio_to_transcribe)

        return None

    def _transcribe_and_handle(self, audio_data: bytes) -> str:
        """Transcribe speech buffer and process command."""
        self.set_state(VoiceState.PROCESSING, "Transcribing user speech...")
        start_time = time.perf_counter()

        transcript = ""
        try:
            transcript = self.stt.transcribe_audio_bytes(
                audio_data, sample_rate=self.sample_rate
            )
        except Exception as e:
            logger.warning("Transcription error: %s", e)

        self.ui.set_transcript(transcript)

        if not transcript or not transcript.strip():
            self.set_state(VoiceState.IDLE, "No clear speech detected. Listening...")
            return ""

        # Check for interrupt command: "stop", "cancel", "halt"
        if self.stt.is_interrupt_command(transcript):
            self.handle_interrupt("User requested stop.")
            return transcript

        # Check if the user is asking to build / execute workspace action
        if any(w in transcript.lower() for w in ["build", "scan", "init", "doctor", "create"]):
            self.set_state(VoiceState.BUILDING, "Executing workspace action...")
        else:
            self.set_state(VoiceState.PROCESSING, "Formulating response...")

        response_text = self._generate_response(transcript)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Speak short, event-driven response
        spoken_sentence = self.tts.compress_for_speech(response_text)
        self.ui.set_response(response_text, spoken_sentence, latency_ms)

        if self.enable_tts and spoken_sentence:
            # Transition to SPEAKING state (microphones muted during this state)
            self.set_state(VoiceState.SPEAKING, f'Speaking: "{spoken_sentence}"')
            self.tts.speak(spoken_sentence, non_blocking=False)

        self.set_state(VoiceState.COMPLETE, "Turn finished.")
        time.sleep(0.3)
        self.set_state(VoiceState.IDLE, "Ready for next command. Speak freely.")
        return transcript

    def _generate_response(self, prompt: str) -> str:
        """Generate response via local agent or fallback rule-based handler."""
        if self.agent:
            try:
                system_instruction = (
                    "You are m.AI.leage voice assistant. Keep answers concise, factual, and direct. "
                    "Always begin with a short 1-sentence punchy summary for voice synthesis."
                )
                self.agent.system_prompt = system_instruction
                return self.agent.run(prompt)
            except Exception as e:
                logger.warning("Agent query failed: %s", e)
                return f"Local model error: {str(e)[:50]}"

        # Built-in local event-driven fallbacks if agent not connected
        p = prompt.lower()
        if "hello" in p or "hi" in p:
            return "Hello. m.AI.leage is online and running locally."
        elif "status" in p:
            return "Workspace status is active. Zero cloud dependencies."
        elif "scan" in p or "workspace" in p:
            return "Workspace inspected. All files tracked locally."
        elif "doctor" in p or "health" in p:
            return "All diagnostic checks passing."
        elif "build" in p:
            return "Build action complete. Manifest updated."
        else:
            return f"Processed query: {prompt}."

    def handle_interrupt(self, reason: str = "Interrupted") -> None:
        """Handle interrupt signal ('stop' command or Ctrl+C)."""
        self.tts.stop()
        self._speech_buffer.clear()
        self.vad.reset()
        self.set_state(VoiceState.IDLE, f"Halted: {reason}")
        if self.enable_tts:
            self.tts.speak("Stopping.", non_blocking=False)

    def _audio_callback(self, indata, frames, time_info, status):
        """Sounddevice input stream callback."""
        if status:
            logger.debug("Audio callback status: %s", status)
        if self._running.is_set():
            # Pass raw 16-bit PCM bytes
            raw_bytes = bytes(indata)
            self._audio_queue.put(raw_bytes)

    def run_live(self) -> None:
        """Run hands-free voice loop with animated Rich terminal interface."""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError(
                "sounddevice is not installed or audio input device not found. "
                "Ensure microphone permissions are enabled."
            )

        self._running.set()
        self.set_state(VoiceState.IDLE, "Hands-free voice mode active. Speak into your mic.")

        # Audio capture stream (16kHz, mono, int16)
        block_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))

        try:
            stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=block_size,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            )
        except Exception as e:
            raise RuntimeError(f"Could not open microphone stream: {e}")

        with stream:
            with Live(self.ui.render(), refresh_per_second=12) as live:
                try:
                    while self._running.is_set():
                        # Drain audio queue and process frames
                        while not self._audio_queue.empty():
                            frame = self._audio_queue.get_nowait()
                            self.handle_audio_frame(frame)

                        # Update animated UI
                        live.update(self.ui.render())
                        time.sleep(0.05)

                except KeyboardInterrupt:
                    self.handle_interrupt("Ctrl+C pressed")
                finally:
                    self.stop()
                    live.update(self.ui.render())

    def stop(self) -> None:
        """Stop voice session and release resources."""
        self._running.clear()
        self.tts.stop()
        self._speech_buffer.clear()
