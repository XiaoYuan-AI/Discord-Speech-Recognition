"""Recognizer backends for the Discord Speech Recognition SDK."""

from .base import BaseRecognizer
from .google import GoogleRecognizer
from .whisper_api import WhisperAPIRecognizer
from .whisper_local import LocalWhisperRecognizer

__all__ = [
    "BaseRecognizer",
    "GoogleRecognizer",
    "LocalWhisperRecognizer",
    "WhisperAPIRecognizer",
]
