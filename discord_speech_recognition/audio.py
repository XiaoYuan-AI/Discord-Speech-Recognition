"""Discord audio sink with per-user VAD-based speech segmentation.

No temp files — audio data remains in memory as numpy arrays.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Callable, Awaitable

import numpy as np

import discord

from .config import RecognitionConfig
from .types import UserAudioSegment


# ---------------------------------------------------------------------------
# Per-user ring-buffer with voice-activity state machine
# ---------------------------------------------------------------------------

class _SpeechState:
    """Enum-like VAD states."""
    SILENCE = 0
    SPEECH = 1


class _UserBuffer:
    """Accumulates audio frames for one user and tracks VAD state."""

    __slots__ = (
        "user_id",
        "user_name",
        "_frames",
        "_max_frames",
        "_state",
        "_silence_count",
        "_start_ts",
    )

    def __init__(
        self,
        user_id: str,
        user_name: str,
        max_duration_ms: int,
        frame_duration_ms: int = 20,
    ) -> None:
        self.user_id = user_id
        self.user_name = user_name
        self._frames: list[np.ndarray] = []
        self._max_frames = max(max_duration_ms // frame_duration_ms, 1)
        self._state = _SpeechState.SILENCE
        self._silence_count = 0
        self._start_ts = 0.0

    # -- ingestion ----------------------------------------------------------

    def feed(self, pcm_frame: np.ndarray, is_speech: bool) -> None:
        """Feed a single audio frame (int16 mono, 16kHz, ~20ms)."""
        if self._state == _SpeechState.SILENCE:
            if is_speech:
                self._state = _SpeechState.SPEECH
                self._frames.clear()
                self._silence_count = 0
                self._start_ts = time.monotonic()
                self._frames.append(pcm_frame)
            return

        # In SPEECH state — buffer everything (incl. brief silences)
        self._frames.append(pcm_frame)
        self._silence_count = 0 if is_speech else self._silence_count + 1

        # Enforce ring-buffer cap
        while len(self._frames) > self._max_frames:
            self._frames.pop(0)

    # -- segment readiness --------------------------------------------------

    def is_ready(self, silence_threshold_frames: int) -> bool:
        """True when a speech segment is complete."""
        if self._state != _SpeechState.SPEECH:
            return False
        return self._silence_count >= silence_threshold_frames

    def is_timeout(self, max_duration_ms: int, frame_duration_ms: int = 20) -> bool:
        """True if segment has exceeded the maximum allowed duration."""
        if self._state != _SpeechState.SPEECH:
            return False
        elapsed = (time.monotonic() - self._start_ts) * 1000
        return elapsed >= max_duration_ms

    # -- drain --------------------------------------------------------------

    def drain(self) -> np.ndarray:
        """Return accumulated audio as int16 numpy array and reset state."""
        if not self._frames:
            return np.array([], dtype=np.int16)
        audio = np.concatenate(self._frames)
        self._frames.clear()
        self._state = _SpeechState.SILENCE
        self._silence_count = 0
        return audio

    @property
    def speech_duration_ms(self) -> int:
        if not self._frames:
            return 0
        total_samples = sum(len(f) for f in self._frames)
        return int(total_samples / 16)  # 16000 Hz → ms


# ---------------------------------------------------------------------------
# Audio sink
# ---------------------------------------------------------------------------

class DiscordAudioSink:
    """Receives decoded PCM from Discord voice and emits speech segments.

    Implements the duck-type protocol expected by ``VoiceClient.listen()``:
    ``write(VoiceData)`` and ``cleanup()``.  No base-class dependency on
    any discord.py internal module.

    Parameters:
        config: SDK recognition config (VAD thresholds, sample rate).
        on_segment: Async callback invoked when a user finishes speaking.
            Signature: ``async def(user_audio: UserAudioSegment) -> None``.
    """

    def __init__(
        self,
        config: RecognitionConfig,
        on_segment: Callable[[UserAudioSegment], Awaitable[None]],
        *,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._config = config
        self._on_segment = on_segment
        self._loop = loop
        self._buffers: dict[str, _UserBuffer] = {}

    @property
    def is_listening(self) -> bool:
        """Discord.py checks this to decide whether to keep feeding audio."""
        return True

    # -- audio frame handler ------------------------------------------------

    def write(self, data: discord.VoiceData) -> None:
        """Called by discord.py for every 20ms audio frame."""
        if data.user is None:
            return

        user_id = str(data.user.id)
        user_name = getattr(data.user, "display_name", data.user.name)

        buf = self._buffers.get(user_id)
        if buf is None:
            buf = _UserBuffer(user_id, user_name, self._config.max_speech_duration_ms)
            self._buffers[user_id] = buf
        else:
            buf.user_name = user_name  # keep display name current

        # Discord sends stereo 48kHz int16 PCM → convert to mono 16kHz.
        mono_16k = _discord_pcm_to_mono_16k(data.data)

        # VAD: compute normalised RMS energy.
        rms = _rms(mono_16k)
        is_speech = rms > self._config.speech_threshold

        buf.feed(mono_16k, is_speech)

        # Check segment-completion conditions.
        silence_frames_needed = self._config.silence_duration_ms // 20
        ready = buf.is_ready(silence_frames_needed) or buf.is_timeout(
            self._config.max_speech_duration_ms
        )

        if ready:
            audio = buf.drain()
            if len(audio) == 0:
                return

            segment = UserAudioSegment(
                user_id=user_id,
                user_name=user_name,
                pcm_data=audio.tobytes(),
                sample_rate=self._config.target_sample_rate,
                start_timestamp=datetime.now(timezone.utc),
            )

            # Fire the callback on the event loop (sink runs in voice thread).
            asyncio.run_coroutine_threadsafe(
                self._on_segment(segment), self._loop
            )

    def cleanup(self) -> None:
        """Called when the voice client stops listening."""
        self._buffers.clear()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _discord_pcm_to_mono_16k(data: bytes) -> np.ndarray:
    """Convert Discord stereo 48kHz int16 PCM to mono 16kHz int16.

    Discord delivers 20 ms frames: 960 samples/channel at 48 kHz, stereo.
    After conversion: 320 samples at 16 kHz, mono.
    """
    if not data:
        return np.array([], dtype=np.int16)

    stereo = np.frombuffer(data, dtype=np.int16).reshape(-1, 2)
    # Take left channel — both channels carry identical data for mono sources.
    mono = stereo[:, 0].copy()
    # Decimate 3× (48 → 16 kHz).  Discards content above 8 kHz (safe for voice).
    return mono[::3]


def _rms(frame: np.ndarray) -> float:
    """Root-mean-square amplitude normalised to [0, 1]."""
    if len(frame) == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) / 32768.0)
