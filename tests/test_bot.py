import asyncio
from types import SimpleNamespace

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
        self.hook_calls = []

    async def _hook(self, ws, msg):
        self.hook_calls.append((ws, msg))


def _new_uninitialized_bot():
    bot = object.__new__(VoiceRecognitionBot)
    ws = FakeVoiceWebSocket()
    bot._voice_client = SimpleNamespace(ws=ws, _connection=SimpleNamespace(hook=None))
    bot._receiver = FakeReceiver()
    bot._ws_original_hook = None
    bot._ws_original_connection_hook = None
    bot._ws_hook_installed = False
    return bot


def test_hook_voice_ws_registers_speaking_event_and_forwards_to_original_hook():
    bot = _new_uninitialized_bot()
    channel = SimpleNamespace(guild=FakeGuild())

    VoiceRecognitionBot._hook_voice_ws(bot, channel)
    msg = {"op": 5, "d": {"user_id": "42", "ssrc": 1234}}
    asyncio.run(
        bot._voice_client.ws._hook(bot._voice_client.ws, msg)
    )

    assert bot._receiver.registrations == [(1234, "42", "Alice")]
    assert bot._voice_client.ws.hook_calls == [(bot._voice_client.ws, msg)]
    assert bot._voice_client._connection.hook is bot._voice_client.ws._hook


def test_handle_voice_ws_message_registers_client_connect_events():
    receiver = FakeReceiver()
    guild = FakeGuild()

    VoiceRecognitionBot._handle_voice_ws_message(
        receiver,
        guild,
        {"op": 12, "d": {"user_id": "42", "audio_ssrc": "4321"}},
    )
    VoiceRecognitionBot._handle_voice_ws_message(
        receiver,
        guild,
        {
            "op": 11,
            "d": {
                "users": [
                    {"user_id": "42", "audio_ssrc": 9876},
                    {"user_id": "bad", "audio_ssrc": 0},
                ]
            },
        },
    )

    assert receiver.registrations == [
        (4321, "42", "Alice"),
        (9876, "42", "Alice"),
    ]


def test_unhook_voice_ws_restores_original_hooks():
    bot = _new_uninitialized_bot()
    channel = SimpleNamespace(guild=FakeGuild())
    original_hook = bot._voice_client.ws._hook
    original_hook_func = original_hook.__func__
    original_hook_self = original_hook.__self__
    original_connection_hook = object()
    bot._voice_client._connection.hook = original_connection_hook

    VoiceRecognitionBot._hook_voice_ws(bot, channel)
    assert bot._voice_client.ws._hook is not original_hook
    assert bot._voice_client._connection.hook is bot._voice_client.ws._hook

    VoiceRecognitionBot._unhook_voice_ws(bot)

    assert bot._voice_client.ws._hook.__func__ is original_hook_func
    assert bot._voice_client.ws._hook.__self__ is original_hook_self
    assert bot._voice_client._connection.hook is original_connection_hook
    assert bot._ws_original_hook is None
    assert bot._ws_original_connection_hook is None
    assert bot._ws_hook_installed is False
