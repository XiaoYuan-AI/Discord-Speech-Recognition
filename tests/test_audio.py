import numpy as np

from discord_speech_recognition.audio import discord_pcm_to_mono_16k, rms
from discord_speech_recognition.receiver import _discord_pcm_to_mono_16k


def test_discord_pcm_to_mono_16k_returns_empty_int16_array_for_empty_input():
    audio = discord_pcm_to_mono_16k(b"")

    assert audio.dtype == np.int16
    assert audio.size == 0


def test_discord_pcm_to_mono_16k_converts_20ms_stereo_frame_to_320_samples():
    left = np.full(960, 1200, dtype=np.int16)
    right = np.full(960, -1200, dtype=np.int16)
    stereo = np.column_stack([left, right]).reshape(-1)

    audio = discord_pcm_to_mono_16k(stereo.tobytes())

    assert audio.dtype == np.int16
    assert len(audio) == 320
    assert np.max(np.abs(audio), initial=0) == 0


def test_discord_pcm_to_mono_16k_converts_20ms_mono_frame_to_320_samples():
    mono = np.full(960, 800, dtype=np.int16)

    audio = discord_pcm_to_mono_16k(mono.tobytes())

    assert audio.dtype == np.int16
    assert len(audio) == 320
    assert np.max(np.abs(audio), initial=0) > 0


def test_receiver_and_public_audio_helpers_match():
    samples = np.arange(1920, dtype=np.int16)

    assert np.array_equal(
        _discord_pcm_to_mono_16k(samples.tobytes()),
        discord_pcm_to_mono_16k(samples.tobytes()),
    )


def test_rms_is_normalized_to_full_scale():
    frame = np.array([32767, -32768], dtype=np.int16)

    assert 0.99 < rms(frame) <= 1.0
    assert rms(np.array([], dtype=np.int16)) == 0.0
