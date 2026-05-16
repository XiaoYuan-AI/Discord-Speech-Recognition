"""Abstract base class for speech recognizers."""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from ..types import RecognitionResult


class BaseRecognizer(abc.ABC):
    """Abstract base for all speech-to-text recognizer backends.

    Subclasses must implement :meth:`recognize`.
    """

    @abc.abstractmethod
    async def recognize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        user_id: str,
        user_name: str,
        language: Optional[str] = None,
    ) -> RecognitionResult:
        """Transcribe an audio segment to text.

        Args:
            audio: Int16 PCM samples as a 1-D numpy array, mono.
            sample_rate: Sample rate of the audio data (Hz).
            user_id: Discord user ID of the speaker.
            user_name: Discord display name of the speaker.
            language: Optional language hint (backend-dependent format).

        Returns:
            A :class:`RecognitionResult` with the transcribed text and metadata.
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name of this recognizer backend."""
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """Release any resources held by this recognizer."""
        ...
