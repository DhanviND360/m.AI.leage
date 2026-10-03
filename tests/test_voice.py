"""Comprehensive tests for hands-free voice interaction in m.AI.leage."""

import struct
from unittest.mock import MagicMock
import pytest

from mileage.voice.states import VoiceState
from mileage.voice.vad import VadDetector
from mileage.voice.stt import SpeechToTextEngine
from mileage.voice.tts import TextToSpeechEngine
from mileage.voice.session import VoiceSession
from mileage.voice.ui import VoiceTerminalUI


def create_pcm_frame(amplitude: int = 1000, num_samples: int = 480) -> bytes:
    """Helper to generate a synthetic 16-bit PCM audio frame (30ms at 16kHz)."""
    return struct.pack(f"{num_samples}h", *([amplitude] * num_samples))


def test_vad_speech_and_silence_detection():
    detector = VadDetector(
        sample_rate=16000,
        frame_duration_ms=30,
        energy_threshold=300.0,
        silence_timeout_sec=0.1,  # short timeout for test speed
        min_speech_duration_sec=0.06,
    )

    silent_frame = create_pcm_frame(amplitude=50)   # low energy (RMS ~50)
    speech_frame = create_pcm_frame(amplitude=2000) # high energy (RMS 2000)

    # Initial state
    assert not detector.is_speech_active

    # Send silent frames - should not trigger speech
    assert not detector.process_frame(silent_frame)
    assert not detector.is_speech_active

    # Send speech frames to exceed min_speech_duration_sec (2 frames * 30ms = 60ms)
    detector.process_frame(speech_frame)
    detector.process_frame(speech_frame)
    assert detector.is_speech_active

    # Now send silent frames until silence_timeout_sec is reached (4 frames * 30ms = 120ms > 100ms)
    detector.process_frame(silent_frame)
    detector.process_frame(silent_frame)
    detector.process_frame(silent_frame)
    finished = detector.process_frame(silent_frame)

    assert finished, "VAD should trigger end-of-speech after silence timeout"
    assert not detector.is_speech_active, "VAD should reset after speech turns complete"


def test_stt_interrupt_detection():
    assert SpeechToTextEngine.is_interrupt_command("stop")
    assert SpeechToTextEngine.is_interrupt_command("Stop!")
    assert SpeechToTextEngine.is_interrupt_command("cancel")
    assert SpeechToTextEngine.is_interrupt_command("halt")
    assert SpeechToTextEngine.is_interrupt_command("please stop now")
    assert not SpeechToTextEngine.is_interrupt_command("what is the current status")


def test_tts_compression_short_and_punchy():
    long_response = (
        "Here is the breakdown of the workspace. ```python\nprint('hello')\n``` "
        "We have scanned thirty-one files across several subdirectories. "
        "The project is healthy and completely local."
    )
    compressed = TextToSpeechEngine.compress_for_speech(long_response)
    # Must remove markdown code and keep it short
    assert "```" not in compressed
    assert len(compressed.split()) <= 19
    assert compressed.startswith("Here is the breakdown")


def test_mic_muted_during_speaking_prevents_self_listening():
    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_tts = MagicMock(spec=TextToSpeechEngine)
    mock_tts.is_speaking = False

    session = VoiceSession(
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        enable_tts=False,
    )

    speech_frame = create_pcm_frame(amplitude=2000)

    # In IDLE state with TTS silent, mic is active
    assert not session.is_mic_muted()

    # When TTS is speaking, mic MUST be muted
    mock_tts.is_speaking = True
    assert session.is_mic_muted()

    # Sending audio while TTS is speaking should be dropped immediately
    session.handle_audio_frame(speech_frame)
    assert len(session._speech_buffer) == 0, "Frames must be dropped while TTS is speaking"

    # When state is explicitly SPEAKING, mic must also be muted
    mock_tts.is_speaking = False
    session.set_state(VoiceState.SPEAKING)
    assert session.is_mic_muted()
    session.handle_audio_frame(speech_frame)
    assert len(session._speech_buffer) == 0, "Frames must be dropped in SPEAKING state"


def test_complete_listen_understand_respond_cycle():
    """Test the complete end-to-end cycle: listen -> understand -> respond."""
    transitions = []

    def record_transition(state: VoiceState):
        transitions.append(state)

    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_stt.transcribe_audio_bytes.return_value = "how many files are tracked"
    mock_stt.is_interrupt_command.return_value = False

    mock_tts = MagicMock(spec=TextToSpeechEngine)
    mock_tts.is_speaking = False
    mock_tts.compress_for_speech.side_effect = lambda text: TextToSpeechEngine.compress_for_speech(text)

    mock_agent = MagicMock()
    mock_agent.run.return_value = "Workspace contains 31 files. All systems normal."

    session = VoiceSession(
        agent=mock_agent,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        enable_tts=True,
        on_state_change=record_transition,
    )
    # Speed up silence timeout for rapid test
    session.vad.silence_timeout_sec = 0.08
    session.vad.min_speech_duration_sec = 0.03

    assert session.state == VoiceState.IDLE

    # 1. User starts speaking (high energy frames)
    speech_frame = create_pcm_frame(amplitude=2500)
    session.handle_audio_frame(speech_frame)
    session.handle_audio_frame(speech_frame)

    assert session.state == VoiceState.LISTENING
    assert VoiceState.LISTENING in transitions

    # 2. User stops speaking (silence frames until VAD timeout)
    silent_frame = create_pcm_frame(amplitude=50)
    transcript = None
    for _ in range(6):
        res = session.handle_audio_frame(silent_frame)
        if res:
            transcript = res
            break

    # 3. Understand: Verify transcription called and agent ran
    assert transcript == "how many files are tracked"
    mock_stt.transcribe_audio_bytes.assert_called_once()
    mock_agent.run.assert_called_once_with("how many files are tracked")

    # 4. Respond: Verify TTS spoke the compressed response
    mock_tts.speak.assert_called_once()
    spoken_arg = mock_tts.speak.call_args[0][0]
    assert "Workspace contains 31 files" in spoken_arg

    # 5. Check all lifecycle states occurred in correct order
    assert VoiceState.LISTENING in transitions
    assert VoiceState.PROCESSING in transitions
    assert VoiceState.SPEAKING in transitions
    assert VoiceState.COMPLETE in transitions
    assert session.state == VoiceState.IDLE


def test_interrupt_handling():
    mock_stt = MagicMock(spec=SpeechToTextEngine)
    mock_tts = MagicMock(spec=TextToSpeechEngine)

    session = VoiceSession(
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        enable_tts=True,
    )
    session.set_state(VoiceState.LISTENING)
    session._speech_buffer.extend(b"some audio")

    # Trigger interrupt
    session.handle_interrupt("User said stop")

    assert session.state == VoiceState.IDLE
    assert len(session._speech_buffer) == 0
    mock_tts.stop.assert_called_once()
    mock_tts.speak.assert_called_with("Stopping.", non_blocking=False)


def test_ui_rendering():
    ui = VoiceTerminalUI()
    ui.update_state(VoiceState.LISTENING, "Test listening")
    ui.set_transcript("hello world")
    ui.set_response("Full assistant response", "Hello world.", latency_ms=150.0)

    rendered = ui.render()
    assert rendered is not None
