"""Shared types and dataclasses for the Discord Speech Recognition SDK."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RecognitionResult:
    """Result of a speech recognition operation.

    Attributes:
        user_id: Discord user ID who spoke.
        user_name: Discord user display name (resolved asynchronously).
        text: Transcribed text.
        language: Detected/used language code (e.g. "en", "zh").
        confidence: Confidence score (0.0-1.0), if provided by the recognizer.
        timestamp: When the recognition was completed.
        duration_ms: Duration of the audio segment in milliseconds.
        recognizer_name: Name of the recognizer that produced this result.
    """

    user_id: str
    user_name: str
    text: str
    language: str
    confidence: float
    timestamp: datetime
    duration_ms: int
    recognizer_name: str

    @property
    def is_empty(self) -> bool:
        """True if no meaningful speech was detected."""
        return not self.text.strip()


@dataclass
class UserAudioSegment:
    """In-memory audio segment captured from a single user.

    Attributes:
        user_id: Discord user ID (SSRC-resolved).
        user_name: Discord display name of the speaker at capture time.
        pcm_data: Raw PCM int16 samples at `sample_rate` Hz, mono.
        sample_rate: Sample rate of pcm_data (typically 16000 after resampling).
        start_timestamp: When this segment began accumulating.
    """

    user_id: str
    user_name: str
    pcm_data: bytes = field(repr=False)
    sample_rate: int
    start_timestamp: datetime
