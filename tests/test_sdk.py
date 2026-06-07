import asyncio
from datetime import datetime, timezone

import numpy as np
import pytest

from discord_speech_recognition.config import RecognitionConfig
from discord_speech_recognition.recognizers import (
    GoogleRecognizer,
    LocalWhisperRecognizer,
    WhisperAPIRecognizer,
)
from discord_speech_recognition.sdk import SpeechRecognitionClient, _build_recognizer
from discord_speech_recognition.types import RecognitionResult, UserAudioSegment


def _segment(duration_ms=400, sample_rate=16000):
    samples = np.ones(int(sample_rate * duration_ms / 1000), dtype=np.int16)
    return UserAudioSegment(
        user_id="1",
        user_name="Alice",
        pcm_data=samples.tobytes(),
        sample_rate=sample_rate,
        start_timestamp=datetime.now(timezone.utc),
    )


def _result(text="hello"):
    return RecognitionResult(
        user_id="1",
        user_name="Alice",
        text=text,
        language="en",
        confidence=0.9,
        timestamp=datetime.now(timezone.utc),
        duration_ms=400,
        recognizer_name="fake",
    )


class FakeRecognizer:
    name = "fake"

    def __init__(self, result=None, error=None):
        self.result = result or _result()
        self.error = error
        self.calls = []
        self.closed = False

    async def recognize(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result

    async def close(self):
        self.closed = True


def test_build_recognizer_creates_configured_backend():
    assert isinstance(
        _build_recognizer(RecognitionConfig(recognizer="whisper_local")),
        LocalWhisperRecognizer,
    )
    assert isinstance(
        _build_recognizer(RecognitionConfig(recognizer="whisper_api")),
        WhisperAPIRecognizer,
    )
    assert isinstance(
        _build_recognizer(RecognitionConfig(recognizer="google")),
        GoogleRecognizer,
    )


def test_build_recognizer_rejects_unknown_backend():
    config = RecognitionConfig()
    config.recognizer = "bogus"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unknown recognizer: bogus"):
        _build_recognizer(config)


def test_client_reports_not_listening_before_start():
    config = RecognitionConfig()
    client = SpeechRecognitionClient(config)

    assert client.config is config
    assert client.is_listening is False


def test_on_audio_segment_invokes_recognizer_and_callback_for_non_empty_result():
    received = []

    async def callback(result):
        received.append(result)

    client = SpeechRecognitionClient(RecognitionConfig(language="auto"), callback)
    recognizer = FakeRecognizer()
    client._recognizer = recognizer

    asyncio.run(client._on_audio_segment(_segment()))

    assert received == [recognizer.result]
    assert recognizer.calls[0]["user_id"] == "1"
    assert recognizer.calls[0]["user_name"] == "Alice"
    assert recognizer.calls[0]["sample_rate"] == 16000
    assert recognizer.calls[0]["language"] == "auto"


def test_on_audio_segment_skips_short_segments():
    received = []

    async def callback(result):
        received.append(result)

    client = SpeechRecognitionClient(RecognitionConfig(min_speech_duration_ms=300), callback)
    recognizer = FakeRecognizer()
    client._recognizer = recognizer

    asyncio.run(client._on_audio_segment(_segment(duration_ms=100)))

    assert received == []
    assert recognizer.calls == []


def test_on_audio_segment_suppresses_empty_results_and_recognition_errors():
    received = []

    async def callback(result):
        received.append(result)

    client = SpeechRecognitionClient(RecognitionConfig(), callback)
    client._recognizer = FakeRecognizer(result=_result(text="   "))
    asyncio.run(client._on_audio_segment(_segment()))

    client._recognizer = FakeRecognizer(error=RuntimeError("boom"))
    asyncio.run(client._on_audio_segment(_segment()))

    assert received == []


def test_stop_shuts_down_bot_and_recognizer():
    class FakeBot:
        def __init__(self):
            self.shutdown_called = False

        async def shutdown(self):
            self.shutdown_called = True

    bot = FakeBot()
    recognizer = FakeRecognizer()
    client = SpeechRecognitionClient()
    client._bot = bot
    client._recognizer = recognizer

    asyncio.run(client.stop())

    assert bot.shutdown_called is True
    assert recognizer.closed is True
    assert client._bot is None
    assert client._recognizer is None
