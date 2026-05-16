"""Speech recognizer using Google Speech Recognition.

Audio stays in memory — converted to a `speech_recognition.AudioData`
object without writing to disk.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .base import BaseRecognizer
from ..config import RecognitionConfig
from ..types import RecognitionResult


class GoogleRecognizer(BaseRecognizer):
    """Recognizer using Google Speech Recognition (free tier).

    No API key required.  Has rate limits — suitable for
    low-to-moderate volume usage.

    Parameters:
        config: The SDK configuration object.
    """

    def __init__(self, config: RecognitionConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "google"

    async def recognize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        user_id: str,
        user_name: str,
        language: Optional[str] = None,
    ) -> RecognitionResult:
        import speech_recognition as sr

        recognizer = sr.Recognizer()

        # Build an AudioData object in-memory (no temp file)
        raw_wav: bytes = _numpy_to_wav_bytes(audio, sample_rate)
        audio_data = sr.AudioData(raw_wav, sample_rate, 2)  # 2 = 16-bit

        lang_code = language or self._config.google_language

        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(
                None,
                recognizer.recognize_google,
                audio_data,
                None,              # key (None = free tier)
                lang_code,
                False,             # show_all
            )
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError:
            text = ""

        return RecognitionResult(
            user_id=user_id,
            user_name=user_name,
            text=text,
            language=lang_code,
            confidence=1.0 if text else 0.0,
            timestamp=datetime.now(timezone.utc),
            duration_ms=int(len(audio) / sample_rate * 1000),
            recognizer_name=self.name,
        )

    async def close(self) -> None:
        pass  # stateless


def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert int16 numpy audio to WAV bytes in memory."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())
    buf.seek(0)
    # Skip the 44-byte WAV header — speech_recognition expects raw PCM
    return buf.read()[44:]
