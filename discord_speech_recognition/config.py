"""Configuration for the Discord Speech Recognition SDK."""

from dataclasses import dataclass
from typing import Optional, Literal


RecognizerType = Literal["whisper_local", "whisper_api", "google"]


@dataclass
class RecognitionConfig:
    """Configuration for the SpeechRecognitionClient.

    Attributes:
        recognizer: Which recognizer backend to use.
            - ``"whisper_local"``: faster-whisper running on local CPU/GPU.
            - ``"whisper_api"``: OpenAI Whisper API (requires API key).
            - ``"google"``: Google Speech Recognition (free tier, no key needed).
        language: Language code for transcription (e.g. ``"en"``, ``"zh"``, ``"auto"``).
            Set to ``"auto"`` to let the recognizer detect language.
        model_size: Whisper model size when using ``"whisper_local"``.
            One of ``"tiny"``, ``"base"``, ``"small"``, ``"medium"``, ``"large-v3"``.
        device: Device for local Whisper: ``"cpu"``, ``"cuda"``.
        compute_type: Compute type for faster-whisper: ``"int8"``, ``"float16"``, etc.
        preload_local_model: Load the local Whisper model during ``start()`` instead of
            waiting for the first speech segment.
        local_beam_size: Beam size for local Whisper decoding. Higher can improve
            accuracy but costs latency.
        local_best_of: Candidate count for non-beam local Whisper decoding.
        local_temperature: Decoding temperature for local Whisper.
        local_cpu_threads: CPU thread count for faster-whisper. ``0`` lets CTranslate2
            choose automatically.
        local_num_workers: Number of faster-whisper workers for concurrent calls.
        local_initial_prompt: Optional prompt to bias local Whisper vocabulary/style.
        local_hotwords: Optional hotwords to bias local Whisper recognition.
        cache_user_language: Cache high-confidence auto-detected language per user.
        language_confidence_threshold: Minimum language probability for caching.
        openai_api_key: API key for OpenAI Whisper API backend.
        google_language: Language code for Google recognizer (different format,
            e.g. ``"en-US"``, ``"zh-CN"``).

        speech_threshold: RMS energy threshold for VAD speech detection (0.0-1.0).
            Higher values require louder speech.
        silence_threshold: RMS energy threshold below which audio is considered silence.
        min_speech_duration_ms: Minimum speech duration before triggering recognition.
        max_speech_duration_ms: Maximum accumulated speech before forced recognition.
        silence_duration_ms: Silence duration after which the speech segment ends.
        target_sample_rate: Target sample rate for the recognizer (default 16000 Hz).
        normalize_audio: Normalize captured segments before local transcription.
        target_audio_rms: Target RMS level for normalization.
        max_audio_gain: Maximum gain applied during normalization.
    """

    # Recognizer selection
    recognizer: RecognizerType = "whisper_local"

    # Language
    language: str = "auto"

    # Local Whisper settings
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    preload_local_model: bool = True
    local_beam_size: int = 3
    local_best_of: int = 3
    local_temperature: float = 0.0
    local_cpu_threads: int = 0
    local_num_workers: int = 1
    local_initial_prompt: Optional[str] = None
    local_hotwords: Optional[str] = None
    cache_user_language: bool = True
    language_confidence_threshold: float = 0.75

    # OpenAI Whisper API
    openai_api_key: Optional[str] = None

    # Google Speech Recognition
    google_language: str = "en-US"

    # VAD / Segmentation
    speech_threshold: float = 0.015
    silence_threshold: float = 0.008
    min_speech_duration_ms: int = 300
    max_speech_duration_ms: int = 15000
    silence_duration_ms: int = 800

    # Audio processing
    target_sample_rate: int = 16000
    normalize_audio: bool = True
    target_audio_rms: float = 0.08
    max_audio_gain: float = 8.0
