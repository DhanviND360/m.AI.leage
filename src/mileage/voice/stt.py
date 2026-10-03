"""Speech-to-Text engine using local faster-whisper."""

import io
import re
import wave
from typing import Optional
from mileage.core.logger import logger

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False


class SpeechToTextEngine:
    """Local Speech-to-Text transcriber using faster-whisper."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    def _ensure_loaded(self) -> None:
        """Lazy load the Whisper model into memory."""
        if self._model is None:
            if not HAS_FASTER_WHISPER:
                raise RuntimeError(
                    "faster-whisper is not installed. Run 'pip install faster-whisper'."
                )
            logger.info("Loading local Whisper model: %s (%s)", self.model_size, self.compute_type)
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

    def is_available(self) -> bool:
        """Check if faster-whisper is installed."""
        return HAS_FASTER_WHISPER

    def transcribe_audio_bytes(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
    ) -> str:
        """
        Transcribe raw PCM audio bytes to text using local Whisper.
        """
        if not pcm_bytes:
            return ""

        self._ensure_loaded()

        # Wrap PCM into an in-memory WAV buffer
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        wav_buffer.seek(0)

        assert self._model is not None
        segments, info = self._model.transcribe(
            wav_buffer,
            beam_size=1,  # fast greedy decoding
            language="en" if "en" in self.model_size else None,
            vad_filter=True,
        )

        text_parts = [segment.text.strip() for segment in segments]
        transcript = " ".join(text_parts).strip()
        return transcript

    @staticmethod
    def is_interrupt_command(text: str) -> bool:
        """Check if the transcribed phrase is an immediate interrupt/stop request."""
        clean = text.strip().lower()
        interrupt_phrases = {"stop", "cancel", "halt", "quit", "pause", "exit"}
        words = set(re.findall(r"[a-z]+", clean))
        if words & interrupt_phrases:
            return True
        return "shut up" in clean
