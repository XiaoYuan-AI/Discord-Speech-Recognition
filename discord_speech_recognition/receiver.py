"""Custom voice data receiver built directly on the official discord.py.

Uses ``VoiceConnectionState.add_socket_listener`` to receive raw UDP
packets, then handles RTP parsing, decryption, Opus decoding, and VAD
speech segmentation — all without any SDK or fork dependency.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from datetime import datetime, timezone
from typing import Callable, Awaitable

import numpy as np

import discord
from nacl.secret import SecretBox, Aead

from .config import RecognitionConfig
from .types import UserAudioSegment

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RTP / encryption helpers
# ---------------------------------------------------------------------------

_RTP_HEADER_SIZE = 12
_NONCE_SIZE = 24
_POLY1305_TAG_SIZE = 16

# Encryption mode identifiers (as used internally by discord.py)
_MODE_NORMAL = "xsalsa20_poly1305"
_MODE_SUFFIX = "xsalsa20_poly1305_suffix"
_MODE_LITE = "xsalsa20_poly1305_lite"
_MODE_AEAD = "aead_xchacha20_poly1305_rtpsize"


def _parse_rtp_header(packet: memoryview) -> tuple[int, int, int]:
    """Parse an RTP header, returning ``(sequence, timestamp, ssrc)``."""
    seq = struct.unpack_from(">H", packet, 2)[0]
    ts = struct.unpack_from(">I", packet, 4)[0]
    ssrc = struct.unpack_from(">I", packet, 8)[0]
    return seq, ts, ssrc


def _decrypt(
    mode: str,
    header: memoryview,
    encrypted: bytes,
    secret_key: list[int],
) -> bytes | None:
    """Decrypt the RTP payload using the negotiated encryption mode.

    Returns the plaintext Opus bytes, or ``None`` on auth failure.
    """
    key = bytes(secret_key)

    if mode == _MODE_NORMAL:
        box = SecretBox(key)
        nonce = bytearray(_NONCE_SIZE)
        nonce[:12] = header
        try:
            return box.decrypt(encrypted, bytes(nonce))
        except Exception:
            return None

    if mode == _MODE_SUFFIX:
        split = len(encrypted) - _NONCE_SIZE
        message, suffix_nonce = encrypted[:split], encrypted[split:]
        box = SecretBox(key)
        try:
            return box.decrypt(message, suffix_nonce)
        except Exception:
            return None

    if mode == _MODE_LITE:
        split = len(encrypted) - 4
        message, lite_nonce = encrypted[:split], encrypted[split:]
        nonce = bytearray(_NONCE_SIZE)
        nonce[:4] = lite_nonce
        box = SecretBox(key)
        try:
            return box.decrypt(message, bytes(nonce))
        except Exception:
            return None

    if mode == _MODE_AEAD:
        split = len(encrypted) - 4
        message, aead_nonce = encrypted[:split], encrypted[split:]
        nonce = bytearray(_NONCE_SIZE)
        nonce[:4] = aead_nonce
        box = Aead(key)
        try:
            return box.decrypt(message, bytes(header), bytes(nonce))
        except Exception:
            return None

    _log.warning("Unknown encryption mode: %s", mode)
    return None


# ---------------------------------------------------------------------------
# Per-user VAD buffer
# ---------------------------------------------------------------------------

class _SpeechState:
    SILENCE = 0
    SPEECH = 1


class _UserBuffer:
    __slots__ = ("ssrc", "_frames", "_max_frames", "_state", "_silence_count", "_start_ts")

    def __init__(self, ssrc: int, max_duration_ms: int, frame_duration_ms: int = 20) -> None:
        self.ssrc = ssrc
        self._frames: list[np.ndarray] = []
        self._max_frames = max(max_duration_ms // frame_duration_ms, 1)
        self._state = _SpeechState.SILENCE
        self._silence_count = 0
        self._start_ts = 0.0

    def feed(self, pcm_frame: np.ndarray, is_speech: bool) -> None:
        if self._state == _SpeechState.SILENCE:
            if is_speech:
                self._state = _SpeechState.SPEECH
                self._frames.clear()
                self._silence_count = 0
                self._start_ts = time.monotonic()
                self._frames.append(pcm_frame)
            return
        self._frames.append(pcm_frame)
        self._silence_count = 0 if is_speech else self._silence_count + 1
        while len(self._frames) > self._max_frames:
            self._frames.pop(0)

    def is_ready(self, silence_frames: int) -> bool:
        return self._state == _SpeechState.SPEECH and self._silence_count >= silence_frames

    def is_timeout(self, max_ms: int, frame_ms: int = 20) -> bool:
        if self._state != _SpeechState.SPEECH:
            return False
        return (time.monotonic() - self._start_ts) * 1000 >= max_ms

    def drain(self) -> np.ndarray:
        if not self._frames:
            return np.array([], dtype=np.int16)
        audio = np.concatenate(self._frames)
        self._frames.clear()
        self._state = _SpeechState.SILENCE
        self._silence_count = 0
        return audio


# ---------------------------------------------------------------------------
# Voice receiver
# ---------------------------------------------------------------------------

class VoiceReceiver:
    """Receives and processes raw voice UDP packets from a Discord voice channel.

    Registers as a socket listener on the :class:`discord.VoiceClient`'s
    internal connection.  Packets are decrypted, decoded, downsampled, and
    segmented via RMS VAD.  Completed speech segments are delivered to
    *on_segment*.

    Parameters:
        config: VAD and audio processing configuration.
        on_segment: Async callback for completed speech segments.
        loop: The asyncio event loop (for cross-thread dispatch).
    """

    def __init__(
        self,
        config: RecognitionConfig,
        on_segment: Callable[[UserAudioSegment], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._config = config
        self._on_segment = on_segment
        self._loop = loop
        self._voice_client: discord.VoiceClient | None = None
        self._decoder: discord.opus.Decoder | None = None
        self._mode: str = _MODE_NORMAL
        self._secret_key: list[int] = []
        self._buffers: dict[int, _UserBuffer] = {}
        self._ssrc_to_user: dict[int, tuple[str, str]] = {}  # ssrc → (user_id, user_name)

    # -- lifecycle ----------------------------------------------------------

    def attach(self, voice_client: discord.VoiceClient) -> None:
        """Register this receiver with a connected :class:`discord.VoiceClient`.

        Must be called after the voice client is fully connected
        (i.e. after ``channel.connect()`` has returned).
        """
        self._voice_client = voice_client
        self._mode = voice_client.mode
        self._secret_key = voice_client.secret_key
        self._decoder = discord.opus.Decoder()

        # Register our callback on the internal socket reader thread.
        voice_client._connection.add_socket_listener(self._on_raw_packet)

        _log.info(
            "VoiceReceiver attached — mode=%s, ssrc=%s",
            self._mode,
            voice_client.ssrc,
        )

    def detach(self) -> None:
        """Unregister from the voice client."""
        if self._voice_client is not None:
            self._voice_client._connection.remove_socket_listener(self._on_raw_packet)
            self._voice_client = None
        self._decoder = None
        self._buffers.clear()
        self._ssrc_to_user.clear()

    # -- SSRC tracking ------------------------------------------------------

    def register_ssrc(self, ssrc: int, user_id: str, user_name: str) -> None:
        """Map an SSRC to a Discord user (called when SPEAKING events fire)."""
        self._ssrc_to_user[ssrc] = (user_id, user_name)

    # -- packet handler (runs in SocketReader thread) -----------------------

    def _on_raw_packet(self, data: bytes) -> None:
        """Callback invoked by the SocketReader thread for each UDP packet."""
        if len(data) < _RTP_HEADER_SIZE:
            return

        mv = memoryview(data)
        header = mv[:_RTP_HEADER_SIZE]
        encrypted_payload = data[_RTP_HEADER_SIZE:]

        _, _, ssrc = _parse_rtp_header(header)

        # Skip our own packets (the bot's SSRC).
        if self._voice_client is not None and ssrc == self._voice_client.ssrc:
            return

        # Decrypt.
        opus_data = _decrypt(self._mode, header, encrypted_payload, self._secret_key)
        if opus_data is None or len(opus_data) == 0:
            return

        # Decode Opus → stereo 48 kHz int16 PCM.
        try:
            pcm = self._decoder.decode(opus_data)  # type: ignore[union-attr]
        except Exception:
            return

        if not pcm:
            return

        # Convert to mono 16 kHz.
        mono_16k = _discord_pcm_to_mono_16k(pcm)

        # VAD.
        rms = _rms(mono_16k)
        is_speech = rms > self._config.speech_threshold

        # Per-SSRC buffering.
        buf = self._buffers.get(ssrc)
        if buf is None:
            buf = _UserBuffer(ssrc, self._config.max_speech_duration_ms)
            self._buffers[ssrc] = buf
        buf.feed(mono_16k, is_speech)

        silence_frames = self._config.silence_duration_ms // 20
        ready = buf.is_ready(silence_frames) or buf.is_timeout(
            self._config.max_speech_duration_ms
        )

        if ready:
            audio = buf.drain()
            if len(audio) == 0:
                return

            # Resolve user from SSRC mapping (or use SSRC as placeholder).
            user_id, user_name = self._ssrc_to_user.get(
                ssrc, (str(ssrc), f"SSRC_{ssrc}")
            )

            segment = UserAudioSegment(
                user_id=user_id,
                user_name=user_name,
                pcm_data=audio.tobytes(),
                sample_rate=self._config.target_sample_rate,
                start_timestamp=datetime.now(timezone.utc),
            )

            asyncio.run_coroutine_threadsafe(
                self._on_segment(segment), self._loop
            )


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _discord_pcm_to_mono_16k(data: bytes) -> np.ndarray:
    """Convert Discord stereo 48 kHz int16 PCM to mono 16 kHz int16."""
    if not data:
        return np.array([], dtype=np.int16)
    samples = np.frombuffer(data, dtype=np.int16)
    n = len(samples)
    if n >= 1500 and n % 2 == 0:
        mono = samples[::2].copy()
    else:
        mono = samples.copy()
    return mono[::3]


def _rms(frame: np.ndarray) -> float:
    """Normalised RMS amplitude [0, 1]."""
    if len(frame) == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) / 32768.0)
