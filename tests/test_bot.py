import asyncio
from types import MethodType, SimpleNamespace

from discord_speech_recognition.bot import VoiceRecognitionBot


class FakeReceiver:
    def __init__(self):
        self.registrations = []

    def register_ssrc(self, ssrc, user_id, user_name):
        self.registrations.append((ssrc, user_id, user_name))


class FakeMember:
    display_name = "Alice"


class FakeGuild:
    def get_member(self, user_id):
        assert user_id == 42
        return FakeMember()


class FakeVoiceWebSocket:
    def __init__(self):
        self.messages = []

    async def received_message(self, msg):
        self.messages.append(msg)
        return "forwarded"


def _new_uninitialized_bot():
    bot = object.__new__(VoiceRecognitionBot)
    bot._voice_client = SimpleNamespace(ws=FakeVoiceWebSocket())
    bot._receiver = FakeReceiver()
    bot._ws_original_received = None
    return bot


def test_hook_voice_ws_registers_speaking_dict_and_forwards_message():
    bot = _new_uninitialized_bot()
    channel = SimpleNamespace(guild=FakeGuild())

    VoiceRecognitionBot._hook_voice_ws(bot, channel)
    result = asyncio.run(
        bot._voice_client.ws.received_message(
            {"op": 5, "d": {"user_id": "42", "ssrc": 1234}}
        )
    )

    assert result == "forwarded"
    assert bot._receiver.registrations == [(1234, "42", "Alice")]
    assert bot._voice_client.ws.messages == [
        {"op": 5, "d": {"user_id": "42", "ssrc": 1234}}
    ]


def test_unhook_voice_ws_restores_original_handler():
    bot = _new_uninitialized_bot()
    channel = SimpleNamespace(guild=FakeGuild())
    original_func = bot._voice_client.ws.received_message.__func__
    original_self = bot._voice_client.ws.received_message.__self__

    VoiceRecognitionBot._hook_voice_ws(bot, channel)
    assert isinstance(bot._voice_client.ws.received_message, MethodType)

    VoiceRecognitionBot._unhook_voice_ws(bot)

    restored = bot._voice_client.ws.received_message
    assert restored.__func__ is original_func
    assert restored.__self__ is original_self
    assert bot._ws_original_received is None
