"""Voice Activity Detection (VAD) and silence detection."""

import math
import struct
from typing import Optional

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False


class VadDetector:
    """Detects voice activity and silence using WebRTC VAD and RMS energy."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        aggressiveness: int = 2,
        energy_threshold: float = 300.0,
        silence_timeout_sec: float = 1.2,
        min_speech_duration_sec: float = 0.4,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 16-bit = 2 bytes/sample
        self.energy_threshold = energy_threshold
        self.silence_timeout_sec = silence_timeout_sec
        self.min_speech_duration_sec = min_speech_duration_sec

        self.vad = webrtcvad.Vad(aggressiveness) if HAS_WEBRTCVAD else None

        # Tracking state
        self.is_speech_active = False
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.total_frames_processed = 0

    def calculate_rms(self, audio_frame: bytes) -> float:
        """Calculate Root Mean Square (RMS) energy of 16-bit PCM frame."""
        count = len(audio_frame) // 2
        if count == 0:
            return 0.0
        shorts = struct.unpack(f"{count}h", audio_frame)
        sum_squares = sum(s * s for s in shorts)
        return math.sqrt(sum_squares / count)

    def is_speech(self, audio_frame: bytes) -> bool:
        """Check whether a single audio frame contains speech."""
        # 1. Energy check first
        rms = self.calculate_rms(audio_frame)
        if rms < self.energy_threshold:
            return False

        # 2. WebRTC VAD check if available
        if self.vad and len(audio_frame) == self.frame_size:
            try:
                return self.vad.is_speech(audio_frame, self.sample_rate)
            except Exception:
                pass

        # Fallback to energy threshold
        return rms >= self.energy_threshold

    def process_frame(self, audio_frame: bytes) -> bool:
        """
        Process a single audio frame and update speech/silence state.

        Returns:
            True if the user was speaking and has now finished (silence timeout reached).
        """
        self.total_frames_processed += 1
        frame_is_speech = self.is_speech(audio_frame)

        frame_duration = self.frame_duration_ms / 1000.0

        if frame_is_speech:
            self.speech_frames_count += 1
            self.silence_frames_count = 0
            if (self.speech_frames_count * frame_duration) >= self.min_speech_duration_sec:
                self.is_speech_active = True
        else:
            if self.is_speech_active:
                self.silence_frames_count += 1
                silence_duration = self.silence_frames_count * frame_duration
                if silence_duration >= self.silence_timeout_sec:
                    # User was speaking and has now finished!
                    self.reset()
                    return True

        return False

    def reset(self) -> None:
        """Reset speech and silence tracking counters."""
        self.is_speech_active = False
        self.speech_frames_count = 0
        self.silence_frames_count = 0
