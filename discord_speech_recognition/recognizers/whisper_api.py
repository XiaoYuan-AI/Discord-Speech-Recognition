"""Speech recognizer using the OpenAI Whisper API.

Sends audio directly as a file-like object in memory — no temp files.
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .base import BaseRecognizer
from ..config import RecognitionConfig
from ..types import RecognitionResult


class WhisperAPIRecognizer(BaseRecognizer):
    """Recognizer using the OpenAI Whisper API.

    Requires an API key set in :attr:`RecognitionConfig.openai_api_key`.

    Parameters:
        config: The SDK configuration object.
    """

    def __init__(self, config: RecognitionConfig) -> None:
        self._config = config
        self._client = None

    @property
    def name(self) -> str:
        return "whisper_api"

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]

            if not self._config.openai_api_key:
                raise ValueError("openai_api_key is required for WhisperAPIRecognizer")

            self._client = AsyncOpenAI(api_key=self._config.openai_api_key)
        return self._client

    async def recognize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        user_id: str,
        user_name: str,
        language: Optional[str] = None,
    ) -> RecognitionResult:
        client = self._get_client()

        # Build a WAV file in memory
        wav_bytes = _numpy_to_wav_bytes(audio, sample_rate)
        file_obj = io.BytesIO(wav_bytes)
        file_obj.name = "audio.wav"

        lang = language if language and language != "auto" else None

        kwargs: dict = {
            "model": "whisper-1",
            "file": file_obj,
            "response_format": "verbose_json",
        }
        if lang:
            kwargs["language"] = lang

        try:
            transcript = await client.audio.transcriptions.create(**kwargs)
        except Exception:
            return RecognitionResult(
                user_id=user_id,
                user_name=user_name,
                text="",
                language=lang or "unknown",
                confidence=0.0,
                timestamp=datetime.now(timezone.utc),
                duration_ms=int(len(audio) / sample_rate * 1000),
                recognizer_name=self.name,
            )

        text = transcript.text.strip() if transcript.text else ""
        detected_lang = getattr(transcript, "language", lang or "unknown")
        confidence = getattr(transcript, "segments", None)
        conf_val = confidence[0].get("confidence", 0.9) if confidence else 0.9

        return RecognitionResult(
            user_id=user_id,
            user_name=user_name,
            text=text,
            language=detected_lang,
            confidence=conf_val,
            timestamp=datetime.now(timezone.utc),
            duration_ms=int(len(audio) / sample_rate * 1000),
            recognizer_name=self.name,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert int16 numpy audio to WAV bytes in memory."""
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())
    buf.seek(0)
    return buf.read()
