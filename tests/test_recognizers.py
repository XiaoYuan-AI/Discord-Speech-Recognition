import asyncio
import sys
import types
import wave
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest

from discord_speech_recognition.config import RecognitionConfig
from discord_speech_recognition.recognizers.google import (
    GoogleRecognizer,
    _numpy_to_wav_bytes as google_numpy_to_wav_bytes,
)
from discord_speech_recognition.recognizers.whisper_api import (
    WhisperAPIRecognizer,
    _numpy_to_wav_bytes as whisper_numpy_to_wav_bytes,
)
from discord_speech_recognition.recognizers.whisper_local import _transcribe_sync


class FakeWhisperModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        info = SimpleNamespace(language="en", language_probability=0.87)
        segments = iter(
            [
                SimpleNamespace(text=" hello "),
                SimpleNamespace(text=""),
                SimpleNamespace(text="world"),
            ]
        )
        return segments, info


def test_transcribe_sync_uses_fast_settings_and_joins_non_empty_segments():
    model = FakeWhisperModel()
    text, language, confidence = _transcribe_sync(
        model,
        np.zeros(16000, dtype=np.float32),
        "en",
    )

    assert text == "hello world"
    assert language == "en"
    assert confidence == 0.87
    assert model.kwargs == {
        "language": "en",
        "beam_size": 1,
        "condition_on_previous_text": False,
        "vad_filter": False,
        "no_speech_threshold": 0.6,
    }


def test_google_numpy_to_wav_bytes_returns_headerless_pcm():
    samples = np.array([1, -2, 3], dtype=np.int16)

    assert google_numpy_to_wav_bytes(samples, 16000) == samples.tobytes()


def test_whisper_numpy_to_wav_bytes_returns_readable_wav():
    samples = np.array([1, -2, 3], dtype=np.int16)
    wav_bytes = whisper_numpy_to_wav_bytes(samples, 16000)

    with wave.open(BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.readframes(3) == samples.tobytes()


def test_google_recognizer_maps_auto_language_to_google_language(monkeypatch):
    calls = []
    fake_sr = types.ModuleType("speech_recognition")

    class UnknownValueError(Exception):
        pass

    class RequestError(Exception):
        pass

    class AudioData:
        def __init__(self, frame_data, sample_rate, sample_width):
            self.frame_data = frame_data
            self.sample_rate = sample_rate
            self.sample_width = sample_width

    class Recognizer:
        def recognize_google(self, audio_data, key=None, language="en-US", show_all=False):
            calls.append((audio_data.sample_rate, key, language, show_all))
            return "hello"

    fake_sr.UnknownValueError = UnknownValueError
    fake_sr.RequestError = RequestError
    fake_sr.AudioData = AudioData
    fake_sr.Recognizer = Recognizer
    monkeypatch.setitem(sys.modules, "speech_recognition", fake_sr)

    recognizer = GoogleRecognizer(RecognitionConfig(google_language="zh-CN"))
    result = asyncio.run(
        recognizer.recognize(
            np.zeros(16000, dtype=np.int16),
            16000,
            "1",
            "Alice",
            language="auto",
        )
    )

    assert result.text == "hello"
    assert result.language == "zh-CN"
    assert result.confidence == 1.0
    assert calls == [(16000, None, "zh-CN", False)]


def test_google_recognizer_returns_empty_result_on_unknown_value(monkeypatch):
    fake_sr = types.ModuleType("speech_recognition")

    class UnknownValueError(Exception):
        pass

    class RequestError(Exception):
        pass

    class AudioData:
        def __init__(self, frame_data, sample_rate, sample_width):
            pass

    class Recognizer:
        def recognize_google(self, *_args):
            raise UnknownValueError()

    fake_sr.UnknownValueError = UnknownValueError
    fake_sr.RequestError = RequestError
    fake_sr.AudioData = AudioData
    fake_sr.Recognizer = Recognizer
    monkeypatch.setitem(sys.modules, "speech_recognition", fake_sr)

    recognizer = GoogleRecognizer(RecognitionConfig())
    result = asyncio.run(
        recognizer.recognize(
            np.zeros(16000, dtype=np.int16),
            16000,
            "1",
            "Alice",
        )
    )

    assert result.text == ""
    assert result.confidence == 0.0
    assert result.is_empty


def test_whisper_api_missing_api_key_raises_before_importing_optional_package(
    monkeypatch,
):
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    recognizer = WhisperAPIRecognizer(RecognitionConfig(openai_api_key=None))

    with pytest.raises(ValueError, match="openai_api_key is required"):
        recognizer._get_client()


def test_whisper_api_close_closes_cached_client():
    class FakeClient:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    client = FakeClient()
    recognizer = WhisperAPIRecognizer(RecognitionConfig(openai_api_key="key"))
    recognizer._client = client

    asyncio.run(recognizer.close())

    assert client.closed is True
    assert recognizer._client is None
