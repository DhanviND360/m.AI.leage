"""Lightweight local Text-to-Speech engine with echo cancellation lock."""

import re
import threading
import time
from typing import Any, Optional
from mileage.core.logger import logger

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


class TextToSpeechEngine:
    """Local, offline Text-to-Speech engine using pyttsx3/native synthesis."""

    def __init__(
        self,
        rate: int = 185,
        volume: float = 1.0,
        cooldown_sec: float = 0.25,
    ):
        self.rate = rate
        self.volume = volume
        self.cooldown_sec = cooldown_sec

        self._is_speaking = threading.Event()
        self._interrupted = threading.Event()
        self._lock = threading.Lock()
        self._engine: Optional[Any] = None

    def _get_engine(self):
        """Lazy instantiate the local pyttsx3 TTS engine."""
        if self._engine is None and HAS_PYTTSX3:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                self._engine = engine
            except Exception as e:
                logger.warning("Could not initialize pyttsx3 engine: %s", e)
                self._engine = None
        return self._engine

    @property
    def is_speaking(self) -> bool:
        """Returns True if the TTS engine is currently playing audio."""
        return self._is_speaking.is_set()

    def stop(self) -> None:
        """Immediately interrupt and stop speech output."""
        self._interrupted.set()
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
        self._is_speaking.clear()

    def speak(self, text: str, non_blocking: bool = False) -> None:
        """
        Speak the given text locally.
        Guarantees that microphone capture will observe is_speaking=True during playback.
        """
        shortened = self.compress_for_speech(text)
        if not shortened:
            return

        if non_blocking:
            t = threading.Thread(target=self._speak_sync, args=(shortened,), daemon=True)
            t.start()
        else:
            self._speak_sync(shortened)

    def _speak_sync(self, text: str) -> None:
        with self._lock:
            self._interrupted.clear()
            self._is_speaking.set()
            try:
                engine = self._get_engine()
                if engine and not self._interrupted.is_set():
                    engine.say(text)
                    engine.runAndWait()
                else:
                    # Fallback / simulated timing if audio device / engine unavailable
                    simulated_duration = max(0.5, len(text) / 20.0)
                    time.sleep(simulated_duration)
            except Exception as e:
                logger.debug("TTS playback exception: %s", e)
            finally:
                # Add acoustic cool-down delay to eliminate room reverb / mic self-listening
                if self.cooldown_sec > 0:
                    time.sleep(self.cooldown_sec)
                self._is_speaking.clear()

    @staticmethod
    def compress_for_speech(text: str) -> str:
        """
        Compress response into an extremely short, event-driven spoken sentence.
        Removes markdown code blocks, URLs, and excessive punctuation.
        """
        # Remove code blocks
        clean = re.sub(r"```[\s\S]*?```", " [code block omitted] ", text)
        clean = re.sub(r"`.*?`", "", clean)
        clean = re.sub(r"\[.*?\]\(.*?\)", "", clean)  # remove markdown links
        clean = re.sub(r"[*#_~>]", "", clean)         # remove markdown symbols
        clean = " ".join(clean.split())

        # Grab only the first sentence or first 18 words
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        first_sentence = sentences[0] if sentences else clean

        words = first_sentence.split()
        if len(words) > 18:
            return " ".join(words[:18]) + "."
        return first_sentence.strip()
