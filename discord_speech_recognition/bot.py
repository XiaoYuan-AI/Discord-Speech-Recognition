"""Minimal async Discord bot for voice-channel speech recognition.

Uses the official ``discord.py`` library (no fork) with a custom
:class:`~receiver.VoiceReceiver` that handles UDP packet decryption,
Opus decoding, and VAD speech segmentation from scratch.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Awaitable, Any

import discord

from .config import RecognitionConfig
from .receiver import VoiceReceiver
from .types import UserAudioSegment


class VoiceRecognitionBot(discord.Client):
    """A minimal Discord client that joins a voice channel and receives audio.

    This is an internal class — users interact with
    :class:`~sdk.SpeechRecognitionClient` instead.
    """

    def __init__(
        self,
        config: RecognitionConfig,
        on_segment: Callable[[UserAudioSegment], Awaitable[None]],
        token: str,
        channel_id: int,
    ) -> None:
        intents = discord.Intents.default()
        intents.voice_states = True
        super().__init__(intents=intents)
        self._config = config
        self._on_segment = on_segment
        self._token = token
        self._channel_id = channel_id
        self._voice_client: discord.VoiceClient | None = None
        self._receiver: VoiceReceiver | None = None
        self._connected = asyncio.Event()
        self._ws_original_hook: Any = None
        self._ws_original_connection_hook: Any = None
        self._ws_hook_installed = False

    # -- lifecycle ----------------------------------------------------------

    async def on_ready(self) -> None:
        """Called when the bot has successfully logged in."""
        channel = self.get_channel(self._channel_id)
        if channel is None:
            self._connected.set()
            return

        if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
            self._connected.set()
            return

        loop = asyncio.get_running_loop()
        self._receiver = VoiceReceiver(self._config, self._on_segment, loop)
        voice_ws_hook = self._make_voice_ws_hook(self._receiver, channel.guild)

        class _ReceivingVoiceClient(discord.VoiceClient):
            def create_connection_state(voice_self):
                state = super().create_connection_state()
                state.hook = voice_ws_hook
                return state

        try:
            self._voice_client = await channel.connect(cls=_ReceivingVoiceClient)
        except Exception:
            self._connected.set()
            raise

        # Attach our custom voice receiver directly to the VoiceClient.
        self._receiver.attach(self._voice_client)

        self._connected.set()

    async def connect_and_listen(self) -> None:
        """Start the bot and join the voice channel (blocks until connected)."""
        asyncio.create_task(self.start(self._token))
        await self._connected.wait()

    async def shutdown(self) -> None:
        """Detach receiver, disconnect from voice, and close the bot."""
        # Restore the original voice websocket message handler.
        self._unhook_voice_ws()

        if self._receiver is not None:
            self._receiver.detach()
            self._receiver = None
        if self._voice_client is not None and self._voice_client.is_connected():
            await self._voice_client.disconnect()
        await self.close()

    # -- voice websocket SSRC tracking --------------------------------------

    def _hook_voice_ws(self, channel: discord.VoiceChannel | discord.StageChannel) -> None:
        """Register a voice websocket hook to capture SSRC→user mappings.

        The Discord voice gateway sends speaking/client events that contain
        user IDs and SSRCs.  We observe those events through discord.py's
        hook API so reconnects keep receiving mappings.
        """
        if self._voice_client is None or self._receiver is None:
            return

        voice_ws = self._voice_client.ws
        if not hasattr(voice_ws, "_hook"):
            return

        if self._ws_hook_installed:
            self._unhook_voice_ws()

        _original_hook = voice_ws._hook
        connection = getattr(self._voice_client, "_connection", None)
        _original_connection_hook = getattr(connection, "hook", None)
        self._ws_original_hook = _original_hook
        self._ws_original_connection_hook = _original_connection_hook
        _voice_ws_hook = self._make_voice_ws_hook(
            self._receiver,
            channel.guild,
            _original_hook,
        )

        voice_ws._hook = _voice_ws_hook
        if connection is not None:
            # New voice websockets created during a reconnect inherit this hook.
            connection.hook = _voice_ws_hook
        self._ws_hook_installed = True

    def _unhook_voice_ws(self) -> None:
        """Restore the original voice websocket message handler."""
        if self._voice_client is None or not getattr(self, "_ws_hook_installed", False):
            return
        connection = getattr(self._voice_client, "_connection", None)
        if connection is not None:
            connection.hook = self._ws_original_connection_hook
        voice_ws = self._voice_client.ws
        if hasattr(voice_ws, "_hook"):
            voice_ws._hook = self._ws_original_hook
        self._ws_original_hook = None
        self._ws_original_connection_hook = None
        self._ws_hook_installed = False

    def _make_voice_ws_hook(
        self,
        receiver: VoiceReceiver,
        guild: discord.Guild | None,
        original_hook: Any = None,
    ):
        async def _voice_ws_hook(ws_self: Any, msg: dict[str, Any]) -> None:
            # discord.py invokes hooks after its own voice gateway handling.
            self._handle_voice_ws_message(receiver, guild, msg)
            if original_hook is not None and original_hook is not _voice_ws_hook:
                await original_hook(ws_self, msg)

        return _voice_ws_hook

    @staticmethod
    def _handle_voice_ws_message(
        receiver: VoiceReceiver,
        guild: discord.Guild | None,
        msg: dict[str, Any],
    ) -> None:
        if not isinstance(msg, dict):
            return

        op = msg.get("op")
        data = msg.get("d") or {}

        if op == 5:  # SPEAKING
            if not isinstance(data, dict):
                return
            VoiceRecognitionBot._register_voice_user(
                receiver,
                guild,
                user_id=data.get("user_id"),
                ssrc=data.get("ssrc"),
            )
            return

        if op == 12:  # CLIENT_CONNECT
            if not isinstance(data, dict):
                return
            VoiceRecognitionBot._register_voice_user(
                receiver,
                guild,
                user_id=data.get("user_id"),
                ssrc=data.get("audio_ssrc") or data.get("ssrc"),
            )
            return

        if op == 11:  # CLIENTS_CONNECT
            users = data.get("users", []) if isinstance(data, dict) else []
            for user in users:
                if not isinstance(user, dict):
                    continue
                VoiceRecognitionBot._register_voice_user(
                    receiver,
                    guild,
                    user_id=user.get("user_id"),
                    ssrc=user.get("audio_ssrc") or user.get("ssrc"),
                )

    @staticmethod
    def _register_voice_user(
        receiver: VoiceReceiver,
        guild: discord.Guild | None,
        *,
        user_id: Any,
        ssrc: Any,
    ) -> None:
        if user_id is None or ssrc is None:
            return

        user_id_str = str(user_id)
        if not user_id_str:
            return

        try:
            ssrc_int = int(ssrc)
        except (TypeError, ValueError):
            return
        if ssrc_int == 0:
            return

        user_name = user_id_str
        if guild is not None:
            try:
                member = guild.get_member(int(user_id_str))
            except (TypeError, ValueError):
                member = None
            if member is not None:
                user_name = member.display_name

        receiver.register_ssrc(ssrc_int, user_id_str, user_name)
