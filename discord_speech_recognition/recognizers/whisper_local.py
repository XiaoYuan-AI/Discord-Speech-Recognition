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
        self._detected_languages: dict[str, str] = {}

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
                    cpu_threads=self._config.local_cpu_threads,
                    num_workers=self._config.local_num_workers,
                )

        await loop.run_in_executor(None, _load)

    async def warmup(self) -> None:
        if self._config.preload_local_model:
            await self._ensure_model()

    async def recognize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        user_id: str,
        user_name: str,
        language: Optional[str] = None,
    ) -> RecognitionResult:
        await self._ensure_model()

        audio_f32 = _prepare_audio(
            audio,
            normalize=self._config.normalize_audio,
            target_rms=self._config.target_audio_rms,
            max_gain=self._config.max_audio_gain,
        )
        auto_language = not language or language == "auto"
        lang = None if auto_language else language
        if auto_language and self._config.cache_user_language:
            lang = self._detected_languages.get(user_id)
        model = self._model

        loop = asyncio.get_running_loop()

        # Run the synchronous transcribe call in a thread pool.
        full_text, detected_lang, lang_prob = await loop.run_in_executor(
            None,
            _transcribe_sync,
            model,
            audio_f32,
            lang,
            self._config,
        )

        if (
            auto_language
            and self._config.cache_user_language
            and detected_lang
            and lang_prob >= self._config.language_confidence_threshold
        ):
            self._detected_languages[user_id] = detected_lang

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
        self._detected_languages.clear()


def _prepare_audio(
    audio: np.ndarray,
    *,
    normalize: bool,
    target_rms: float,
    max_gain: float,
) -> np.ndarray:
    audio_f32 = audio.astype(np.float32) / 32768.0
    if len(audio_f32) == 0:
        return audio_f32

    audio_f32 = audio_f32 - float(np.mean(audio_f32))
    if not normalize:
        return audio_f32

    rms = float(np.sqrt(np.mean(audio_f32.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio_f32), initial=0.0))
    if rms <= 1e-6 or peak <= 1e-6:
        return audio_f32

    gain = min(target_rms / rms, 0.98 / peak, max(max_gain, 1.0))
    if gain > 1.0:
        audio_f32 = audio_f32 * gain
    return np.clip(audio_f32, -1.0, 1.0).astype(np.float32, copy=False)


def _transcribe_sync(
    model,
    audio_f32: np.ndarray,
    lang: Optional[str],
    config: RecognitionConfig,
):
    """Run faster-whisper transcription synchronously (called in thread pool).

    The upstream :class:`VoiceReceiver` already RMS-gates speech before it
    ever reaches us, so we explicitly disable faster-whisper's internal
    Silero VAD — otherwise short Discord segments are routinely silenced
    out, which produces a language guess but zero text.
    """
    segments, info = model.transcribe(
        audio_f32,
        language=lang,
        beam_size=max(config.local_beam_size, 1),
        best_of=max(config.local_best_of, 1),
        temperature=config.local_temperature,
        condition_on_previous_text=False,
        vad_filter=False,
        without_timestamps=True,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        initial_prompt=config.local_initial_prompt,
        hotwords=config.local_hotwords,
        language_detection_threshold=config.language_confidence_threshold,
        language_detection_segments=1,
    )
    # `segments` is a generator; force it to materialise.
    texts = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
    full_text = " ".join(texts).strip()
    return full_text, info.language, info.language_probability
