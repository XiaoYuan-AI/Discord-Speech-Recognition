"""Discord Speech Recognition SDK.

Real-time voice-channel speech-to-text for Discord bots.
Supports local AI models (faster-whisper) and cloud services
(Google Speech Recognition, OpenAI Whisper API).

No temp files — all audio processing is done in memory.
"""

from .config import RecognitionConfig, RecognizerType
from .sdk import RecognitionCallback, SpeechRecognitionClient
from .types import RecognitionResult, UserAudioSegment

__all__ = [
    "RecognitionCallback",
    "RecognitionConfig",
    "RecognitionResult",
    "RecognizerType",
    "SpeechRecognitionClient",
    "UserAudioSegment",
]
