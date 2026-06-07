"""Custom voice data receiver built directly on the official discord.py.

Uses ``VoiceConnectionState.add_socket_listener`` to receive raw UDP
packets, then handles RTP parsing, decryption, Opus decoding, and VAD
speech segmentation — all without any SDK or fork dependency.
"""

from __future__ import annotations

import asyncio
import ctypes.util
import logging
import os
import struct
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Awaitable

import numpy as np

import discord
from nacl.secret import SecretBox, Aead

from .audio import discord_pcm_to_mono_16k
from .config import RecognitionConfig
from .types import UserAudioSegment

_log = logging.getLogger(__name__)


def _ensure_opus_loaded() -> None:
    """Make sure ``libopus`` is loaded so ``discord.opus.Decoder`` works.

    On macOS the dylib isn't on the default loader path even when installed
    via Homebrew, and on most Linux distros only the SONAME is shipped —
    so ``discord.py``'s auto-loader frequently misses it.  We try a small
    set of well-known locations / names and call :func:`discord.opus.load_opus`
    on the first hit.
    """
    if discord.opus.is_loaded():
        return

    candidates: list[str] = []
    if sys.platform == "darwin":
        candidates += [
            # Homebrew (Apple Silicon, then Intel), then MacPorts.
            "/opt/homebrew/lib/libopus.dylib",
            "/opt/homebrew/lib/libopus.0.dylib",
            "/usr/local/lib/libopus.dylib",
            "/usr/local/lib/libopus.0.dylib",
            "/opt/local/lib/libopus.dylib",
            "libopus.0.dylib",
            "libopus.dylib",
        ]
    elif sys.platform.startswith("linux"):
        candidates += ["libopus.so.0", "libopus.so"]
    elif sys.platform == "win32":
        candidates += ["libopus-0.dll", "opus.dll"]

    # Whatever ctypes thinks is the canonical name on this platform.
    discovered = ctypes.util.find_library("opus")
    if discovered:
        candidates.append(discovered)

    last_err: Exception | None = None
    for name in candidates:
        if not name:
            continue
        if os.sep in name and not os.path.exists(name):
            continue
        try:
            discord.opus.load_opus(name)
            if discord.opus.is_loaded():
                _log.info("Loaded libopus from %s", name)
                return
        except Exception as e:  # OSError, OpusNotLoaded, …
            last_err = e
            continue

    hint = (
        "brew install opus" if sys.platform == "darwin"
        else "apt-get install libopus0" if sys.platform.startswith("linux")
        else "install the Opus codec library"
    )
    raise RuntimeError(
        f"libopus is not installed or could not be loaded. Install it with: {hint}"
    ) from last_err


# ---------------------------------------------------------------------------
# RTP / encryption helpers
# ---------------------------------------------------------------------------

# RTP fixed header is 12 bytes.  A CSRC list or extension may follow before
# the actual payload, so we must inspect the header flags.
_RTP_HEADER_SIZE = 12
_RTP_EXTENSION_HEADER_SIZE = 4  # profile (2) + length-in-words (2)
_NONCE_SIZE = 24

# Encryption mode identifiers (as used internally by discord.py)
_MODE_NORMAL = "xsalsa20_poly1305"
_MODE_SUFFIX = "xsalsa20_poly1305_suffix"
_MODE_LITE = "xsalsa20_poly1305_lite"
_MODE_AEAD = "aead_xchacha20_poly1305_rtpsize"

# RTP header byte‑0 bit masks
_RTP_VERSION_MASK = 0xC0
_RTP_PADDING_MASK = 0x20
_RTP_EXTENSION_MASK = 0x10
_RTP_CSRC_COUNT_MASK = 0x0F

# Expected RTP version for Discord voice packets.
_RTP_VERSION = 2 << 6  # 0x80

# RTCP packets share the same UDP socket and use RTP version 2 too.  We
# distinguish them by byte-1 payload type: RTCP uses 200..204 (SR, RR,
# SDES, BYE, APP) — values that can never appear as an RTP payload type
# because RTP's PT field is only 7 bits (0..127).  Anything ≥ 200 is
# therefore guaranteed RTCP and should be skipped silently.
_RTCP_PT_RANGE = range(200, 205)


def _rtp_unencrypted_prefix_size(packet: memoryview) -> int:
    """Return the size of the unencrypted RTP prefix for ``*_rtpsize`` modes.

    For Discord's ``aead_*_rtpsize`` encryption modes (the only supported
    modes since Nov 2024), the unencrypted header — and therefore the AAD —
    is defined SRTP-style: it includes the 12-byte RTP header, any CSRCs
    AND the 4-byte extension preamble (profile + length) when the X flag
    is set, but NOT the extension data itself.  The extension data and the
    Opus payload are part of the ciphertext.
    """
    if len(packet) < _RTP_HEADER_SIZE:
        return _RTP_HEADER_SIZE

    byte0 = packet[0]
    if (byte0 & _RTP_VERSION_MASK) != _RTP_VERSION:
        return _RTP_HEADER_SIZE

    csrc_count = byte0 & _RTP_CSRC_COUNT_MASK
    offset = _RTP_HEADER_SIZE + csrc_count * 4

    if byte0 & _RTP_EXTENSION_MASK:
        # Include only the 4-byte extension PREAMBLE in the unencrypted
        # prefix.  Extension data follows after decryption.
        offset += _RTP_EXTENSION_HEADER_SIZE

    return offset


def _strip_extension_from_plaintext(
    packet: memoryview, plaintext: bytes
) -> bytes:
    """Strip the (decrypted) RTP extension body from the plaintext.

    In ``*_rtpsize`` modes the extension data sits at the start of the
    ciphertext.  Its length (in 32-bit words) lives in the unencrypted
    extension preamble within the RTP header.
    """
    byte0 = packet[0]
    if not (byte0 & _RTP_EXTENSION_MASK):
        return plaintext

    csrc_count = byte0 & _RTP_CSRC_COUNT_MASK
    preamble_offset = _RTP_HEADER_SIZE + csrc_count * 4
    if len(packet) < preamble_offset + _RTP_EXTENSION_HEADER_SIZE:
        return plaintext

    ext_len_words = struct.unpack_from(">H", packet, preamble_offset + 2)[0]
    ext_bytes = ext_len_words * 4
    if ext_bytes >= len(plaintext):
        return b""
    return plaintext[ext_bytes:]


def _parse_rtp_header(packet: memoryview) -> tuple[int, int, int]:
    """Parse an RTP header, returning ``(sequence, timestamp, ssrc)``."""
    seq = struct.unpack_from(">H", packet, 2)[0]
    ts = struct.unpack_from(">I", packet, 4)[0]
    ssrc = struct.unpack_from(">I", packet, 8)[0]
    return seq, ts, ssrc


def _decrypt(
    mode: str,
    header: bytes,
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
        if split <= 0:
            return None
        message, suffix_nonce = encrypted[:split], encrypted[split:]
        box = SecretBox(key)
        try:
            return box.decrypt(message, suffix_nonce)
        except Exception:
            return None

    if mode == _MODE_LITE:
        split = len(encrypted) - 4
        if split <= 0:
            return None
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
        if split <= 0:
            return None
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
    __slots__ = (
        "ssrc", "_frames", "_max_frames", "_state",
        "_silence_count", "_start_ts",
    )

    def __init__(
        self, ssrc: int, max_duration_ms: int, frame_duration_ms: int = 20
    ) -> None:
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
        return (
            self._state == _SpeechState.SPEECH
            and self._silence_count >= silence_frames
        )

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

    _LOG_EVERY_N = 200  # log a summary every N packets

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
        self._decoders: dict[int, discord.opus.Decoder] = {}
        self._opus_loaded: bool = False
        self._mode: str = _MODE_NORMAL
        self._secret_key: list[int] = []
        self._buffers: dict[int, _UserBuffer] = {}
        self._ssrc_to_user: dict[int, tuple[str, str]] = {}

        # Diagnostic counters (updated from the SocketReader thread).
        self._packet_count: int = 0
        self._rtcp_packet_count: int = 0
        self._own_packet_count: int = 0
        self._decrypt_ok_count: int = 0
        self._decrypt_fail_count: int = 0
        self._decode_fail_count: int = 0
        self._speech_frame_count: int = 0
        self._segment_count: int = 0
        self._payload_offset: int = 0  # set from first packet

    # -- lifecycle ----------------------------------------------------------

    def attach(self, voice_client: discord.VoiceClient) -> None:
        """Register this receiver with a connected :class:`discord.VoiceClient`.

        Must be called after the voice client is fully connected
        (i.e. after ``channel.connect()`` has returned).
        """
        self._voice_client = voice_client
        self._mode = voice_client.mode
        self._secret_key = voice_client.secret_key
        _ensure_opus_loaded()
        self._opus_loaded = True

        # Register our callback on the internal socket reader thread.
        voice_client._connection.add_socket_listener(self._on_raw_packet)

        _log.info(
            "VoiceReceiver attached — mode=%s, ssrc=%s, key_len=%s",
            self._mode,
            voice_client.ssrc,
            len(self._secret_key),
        )

    def detach(self) -> None:
        """Unregister from the voice client."""
        if self._voice_client is not None:
            self._voice_client._connection.remove_socket_listener(
                self._on_raw_packet
            )
            self._voice_client = None
        self._decoders.clear()
        self._buffers.clear()
        self._ssrc_to_user.clear()
        _log.info(
            "VoiceReceiver detached — packets=%d, segments=%d, "
            "decrypt_ok=%d, decrypt_fail=%d, decode_fail=%d",
            self._packet_count,
            self._segment_count,
            self._decrypt_ok_count,
            self._decrypt_fail_count,
            self._decode_fail_count,
        )

    # -- SSRC tracking ------------------------------------------------------

    def register_ssrc(self, ssrc: int, user_id: str, user_name: str) -> None:
        """Map an SSRC to a Discord user (called when SPEAKING events fire)."""
        self._ssrc_to_user[ssrc] = (user_id, user_name)
        _log.debug("SSRC %d → %s (%s)", ssrc, user_name, user_id)

    # -- packet handler (runs in SocketReader thread) -----------------------

    def _on_raw_packet(self, data: bytes) -> None:
        """Callback invoked by the SocketReader thread for each UDP packet."""
        self._packet_count += 1

        if len(data) < _RTP_HEADER_SIZE:
            return

        mv = memoryview(data)

        # Validate RTP version.
        byte0 = mv[0]
        if (byte0 & _RTP_VERSION_MASK) != _RTP_VERSION:
            return  # not a voice RTP packet (e.g. an IP-discovery packet)

        # RTCP packets share this UDP port — drop them before they pollute
        # the decrypt-fail counter.  Their byte-1 PT (200..204) is outside
        # the 7-bit RTP PT space, so the test is unambiguous.
        if mv[1] in _RTCP_PT_RANGE:
            self._rtcp_packet_count += 1
            return

        # Compute the unencrypted-prefix size.  For *_rtpsize modes this
        # is the AAD; the extension body remains inside the ciphertext.
        prefix_size = _rtp_unencrypted_prefix_size(mv)
        if prefix_size >= len(data):
            return
        if self._payload_offset == 0:
            self._payload_offset = prefix_size

        # For AEAD/rtpsize modes the header passed as AAD is the full
        # unencrypted prefix.  For the deprecated xsalsa20 modes only the
        # first 12 bytes are used (they ignore CSRCs/extensions, but those
        # never appear on those legacy paths from Discord).
        if self._mode == _MODE_AEAD:
            header = bytes(mv[:prefix_size])
        else:
            header = bytes(mv[:_RTP_HEADER_SIZE])
        encrypted_payload = data[prefix_size:]

        _, _, ssrc = _parse_rtp_header(mv)

        # Skip our own packets (the bot's SSRC).
        if self._voice_client is not None and ssrc == self._voice_client.ssrc:
            self._own_packet_count += 1
            return

        # Decrypt.
        plaintext = _decrypt(
            self._mode, header, encrypted_payload, self._secret_key
        )
        if plaintext is None:
            self._decrypt_fail_count += 1
            return
        self._decrypt_ok_count += 1

        # In *_rtpsize modes the extension data sits at the start of the
        # plaintext — strip it to get the raw Opus frame.
        opus_data = _strip_extension_from_plaintext(mv, plaintext)

        if len(opus_data) == 0:
            return

        # Decode Opus → stereo 48 kHz int16 PCM, using a per-SSRC decoder
        # so each user's PLC / gain state stays isolated.
        decoder = self._decoders.get(ssrc)
        if decoder is None:
            decoder = discord.opus.Decoder()
            self._decoders[ssrc] = decoder
        try:
            pcm = decoder.decode(opus_data)
        except Exception:
            self._decode_fail_count += 1
            return

        if not pcm:
            return

        # Convert to mono 16 kHz.
        mono_16k = _discord_pcm_to_mono_16k(pcm)

        # VAD.
        rms = _rms(mono_16k)
        is_speech = rms > self._config.speech_threshold
        if is_speech:
            self._speech_frame_count += 1

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

            self._segment_count += 1

            # Resolve user from SSRC mapping (or use SSRC as placeholder).
            user_id, user_name = self._ssrc_to_user.get(
                ssrc, (str(ssrc), f"SSRC_{ssrc}")
            )

            duration_ms = int(len(audio) / 16)  # 16000 Hz → ms

            segment = UserAudioSegment(
                user_id=user_id,
                user_name=user_name,
                pcm_data=audio.tobytes(),
                sample_rate=self._config.target_sample_rate,
                start_timestamp=datetime.now(timezone.utc),
            )

            _log.debug(
                "Segment #%d: ssrc=%d user=%s duration=%dms samples=%d",
                self._segment_count,
                ssrc,
                user_name,
                duration_ms,
                len(audio),
            )

            asyncio.run_coroutine_threadsafe(
                self._on_segment(segment), self._loop
            )

        # Periodic diagnostic summary.
        if self._packet_count % self._LOG_EVERY_N == 0:
            _log.info(
                "VoiceReceiver status: packets=%d(own=%d rtcp=%d) "
                "decrypt_ok=%d decrypt_fail=%d decode_fail=%d "
                "speech_frames=%d segments=%d ssrcs=%d "
                "payload_offset=%d mode=%s",
                self._packet_count,
                self._own_packet_count,
                self._rtcp_packet_count,
                self._decrypt_ok_count,
                self._decrypt_fail_count,
                self._decode_fail_count,
                self._speech_frame_count,
                self._segment_count,
                len(self._buffers),
                self._payload_offset,
                self._mode,
            )


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _discord_pcm_to_mono_16k(data: bytes) -> np.ndarray:
    """Convert Discord stereo 48 kHz int16 PCM to mono 16 kHz int16.

    ``discord.opus.Decoder`` always decodes to two channels (interleaved
    LRLR…), regardless of what the encoder used — so we average L+R into
    a mono channel, then properly resample 48 kHz → 16 kHz with a
    polyphase low-pass filter.  A bare ``[::3]`` decimation aliases all
    audio between 8–24 kHz back into the speech band, which is enough
    distortion to make Whisper produce gibberish / hallucinations.
    """
    return discord_pcm_to_mono_16k(data)


def _rms(frame: np.ndarray) -> float:
    """Normalised RMS amplitude [0, 1]."""
    if len(frame) == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) / 32768.0)
