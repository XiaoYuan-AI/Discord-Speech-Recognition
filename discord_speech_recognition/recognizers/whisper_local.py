"""Speech recognizer using a local faster-whisper model.

No temp files — audio is fed directly as a numpy array.
Blocking model calls run in a thread pool to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .base import BaseRecognizer
from ..config import RecognitionConfig
from ..types import RecognitionResult


class LocalWhisperRecognizer(BaseRecognizer):
    """Recognizer using a local Whisper model via `faster-whisper`.

    The model is loaded once (lazily, on first recognition) and reused
    for all subsequent calls.  Model inference runs in a thread pool
    so the asyncio event loop is never blocked.

    Parameters:
        config: The SDK configuration object.
    """

    def __init__(self, config: RecognitionConfig) -> None:
        self._config = config
        self._model = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"whisper_local({self._config.model_size})"

    async def _ensure_model(self):
        """Lazy-load the faster-whisper model on first use (thread-safe)."""
        if self._model is not None:
            return
        loop = asyncio.get_running_loop()

        def _load():
            with self._lock:
                if self._model is not None:
                    return
                from faster_whisper import WhisperModel  # type: ignore[import-untyped]

                self._model = WhisperModel(
                    self._config.model_size,
                    device=self._config.device,
                    compute_type=self._config.compute_type,
                )

        await loop.run_in_executor(None, _load)

    async def recognize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        user_id: str,
        user_name: str,
        language: Optional[str] = None,
    ) -> RecognitionResult:
        await self._ensure_model()

        audio_f32 = audio.astype(np.float32) / 32768.0
        lang = language if language and language != "auto" else None
        model = self._model

        loop = asyncio.get_running_loop()

        # Run the synchronous transcribe call in a thread pool.
        full_text, detected_lang, lang_prob = await loop.run_in_executor(
            None,
            _transcribe_sync,
            model,
            audio_f32,
            lang,
        )

        return RecognitionResult(
            user_id=user_id,
            user_name=user_name,
            text=full_text,
            language=detected_lang,
            confidence=lang_prob,
            timestamp=datetime.now(timezone.utc),
            duration_ms=int(len(audio) / sample_rate * 1000),
            recognizer_name=self.name,
        )

    async def close(self) -> None:
        self._model = None


def _transcribe_sync(model, audio_f32: np.ndarray, lang: Optional[str]):
    """Run faster-whisper transcription synchronously (called in thread pool)."""
    segments, info = model.transcribe(
        audio_f32,
        language=lang,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"threshold": 0.5},
    )
    texts = [seg.text.strip() for seg in segments]
    full_text = " ".join(texts).strip()
    return full_text, info.language, info.language_probability
