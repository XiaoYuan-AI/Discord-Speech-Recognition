import asyncio
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from nacl.secret import Aead, SecretBox

from discord_speech_recognition.config import RecognitionConfig
from discord_speech_recognition.receiver import (
    _MODE_AEAD,
    _MODE_LITE,
    _MODE_NORMAL,
    _MODE_SUFFIX,
    _RTP_EXTENSION_MASK,
    _RTP_HEADER_SIZE,
    _UserBuffer,
    _decrypt,
    _parse_rtp_header,
    _rms,
    _rtp_unencrypted_prefix_size,
    _strip_extension_from_plaintext,
    VoiceReceiver,
)


def _rtp_header(byte0=0x80, payload_type=120, seq=7, timestamp=10, ssrc=99):
    return struct.pack(">BBHII", byte0, payload_type, seq, timestamp, ssrc)


def test_rtp_unencrypted_prefix_size_handles_csrc_and_extension_preamble():
    plain = _rtp_header()
    with_csrc = bytes([0x82]) + plain[1:] + b"csrc" + b"CSRC"
    with_extension = bytes([0x80 | _RTP_EXTENSION_MASK]) + plain[1:] + b"\xbe\xde\x00\x02" + b"extdata!"

    assert _rtp_unencrypted_prefix_size(memoryview(plain)) == _RTP_HEADER_SIZE
    assert _rtp_unencrypted_prefix_size(memoryview(with_csrc)) == 20
    assert _rtp_unencrypted_prefix_size(memoryview(with_extension)) == 16


def test_strip_extension_from_plaintext_removes_decrypted_extension_body():
    packet = bytes([0x80 | _RTP_EXTENSION_MASK]) + _rtp_header()[1:] + b"\xbe\xde\x00\x02"

    assert _strip_extension_from_plaintext(memoryview(packet), b"12345678opus") == b"opus"
    assert _strip_extension_from_plaintext(memoryview(packet), b"12345678") == b""
    assert _strip_extension_from_plaintext(memoryview(_rtp_header()), b"opus") == b"opus"


def test_parse_rtp_header_reads_sequence_timestamp_and_ssrc():
    header = _rtp_header(seq=321, timestamp=123456, ssrc=654321)

    assert _parse_rtp_header(memoryview(header)) == (321, 123456, 654321)


@pytest.mark.parametrize("mode", [_MODE_NORMAL, _MODE_SUFFIX, _MODE_LITE, _MODE_AEAD])
def test_decrypt_round_trips_supported_modes(mode):
    key = list(range(32))
    plaintext = b"opus-frame"
    header = _rtp_header()

    if mode == _MODE_NORMAL:
        nonce = bytearray(24)
        nonce[:12] = header
        encrypted = SecretBox(bytes(key)).encrypt(plaintext, bytes(nonce)).ciphertext
    elif mode == _MODE_SUFFIX:
        nonce = bytes(range(24))
        encrypted = SecretBox(bytes(key)).encrypt(plaintext, nonce).ciphertext + nonce
    elif mode == _MODE_LITE:
        lite_nonce = b"\x01\x02\x03\x04"
        nonce = bytearray(24)
        nonce[:4] = lite_nonce
        encrypted = SecretBox(bytes(key)).encrypt(plaintext, bytes(nonce)).ciphertext + lite_nonce
    else:
        aead_nonce = b"\x05\x06\x07\x08"
        nonce = bytearray(24)
        nonce[:4] = aead_nonce
        encrypted = Aead(bytes(key)).encrypt(
            plaintext,
            header,
            bytes(nonce),
        ).ciphertext + aead_nonce

    assert _decrypt(mode, header, encrypted, key) == plaintext


def test_decrypt_returns_none_for_bad_payload_or_unknown_mode():
    key = list(range(32))
    header = _rtp_header()

    assert _decrypt(_MODE_SUFFIX, header, b"too-short", key) is None
    assert _decrypt("unknown", header, b"payload", key) is None


def test_user_buffer_segments_after_speech_then_silence():
    buffer = _UserBuffer(ssrc=123, max_duration_ms=1000)
    speech = np.ones(320, dtype=np.int16)
    silence = np.zeros(320, dtype=np.int16)

    buffer.feed(silence, is_speech=False)
    assert not buffer.is_ready(silence_frames=1)

    buffer.feed(speech, is_speech=True)
    buffer.feed(silence, is_speech=False)

    assert buffer.is_ready(silence_frames=1)
    drained = buffer.drain()
    assert len(drained) == 640
    assert not buffer.is_ready(silence_frames=1)


def test_receiver_ignores_rtcp_without_decrypt_failures():
    async def on_segment(_segment):
        raise AssertionError("RTCP packets must not create segments")

    receiver = VoiceReceiver(RecognitionConfig(), on_segment, asyncio.new_event_loop())
    packet = _rtp_header(payload_type=200) + b"rtcp"

    receiver._on_raw_packet(packet)

    assert receiver._packet_count == 1
    assert receiver._rtcp_packet_count == 1
    assert receiver._decrypt_fail_count == 0


def test_receiver_attach_loads_opus_and_registers_socket_listener(monkeypatch):
    async def on_segment(_segment):
        pass

    receiver = VoiceReceiver(RecognitionConfig(), on_segment, asyncio.new_event_loop())
    listeners = []
    connection = SimpleNamespace(add_socket_listener=listeners.append)
    voice_client = SimpleNamespace(
        mode=_MODE_NORMAL,
        secret_key=list(range(32)),
        ssrc=42,
        _connection=connection,
    )
    monkeypatch.setattr(
        "discord_speech_recognition.receiver._ensure_opus_loaded",
        lambda: None,
    )

    receiver.attach(voice_client)

    assert receiver._opus_loaded is True
    assert listeners == [receiver._on_raw_packet]


def test_receiver_decrypts_dave_payload_with_registered_user():
    import davey

    async def on_segment(_segment):
        pass

    class FakeDaveSession:
        ready = True

        def __init__(self):
            self.calls = []

        def decrypt(self, user_id, media_type, packet):
            self.calls.append((user_id, media_type, packet))
            return b"opus"

    dave_session = FakeDaveSession()
    receiver = VoiceReceiver(RecognitionConfig(), on_segment, asyncio.new_event_loop())
    receiver._voice_client = SimpleNamespace(
        _connection=SimpleNamespace(dave_session=dave_session)
    )
    receiver.register_ssrc(123, "42", "Alice")

    assert receiver._decrypt_dave_if_needed(123, b"encrypted") == b"opus"
    assert dave_session.calls == [(42, davey.MediaType.audio, b"encrypted")]


def test_receiver_skips_dave_payload_until_ssrc_user_mapping_arrives():
    async def on_segment(_segment):
        pass

    receiver = VoiceReceiver(RecognitionConfig(), on_segment, asyncio.new_event_loop())
    receiver._voice_client = SimpleNamespace(
        _connection=SimpleNamespace(dave_session=SimpleNamespace(ready=True))
    )

    assert receiver._decrypt_dave_if_needed(123, b"encrypted") is None
    assert receiver._dave_missing_user_count == 1


def test_receiver_counts_dave_decrypt_failures():
    async def on_segment(_segment):
        pass

    class FakeDaveSession:
        ready = True

        def decrypt(self, *_args):
            raise RuntimeError("bad dave frame")

    receiver = VoiceReceiver(RecognitionConfig(), on_segment, asyncio.new_event_loop())
    receiver._voice_client = SimpleNamespace(
        _connection=SimpleNamespace(dave_session=FakeDaveSession())
    )
    receiver.register_ssrc(123, "42", "Alice")

    assert receiver._decrypt_dave_if_needed(123, b"encrypted") is None
    assert receiver._dave_decrypt_fail_count == 1


def test_rms_matches_public_normalized_contract():
    assert _rms(np.array([0], dtype=np.int16)) == 0.0
    assert 0.49 < _rms(np.array([16384], dtype=np.int16)) < 0.51
