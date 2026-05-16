"""Minimal async Discord bot for voice-channel speech recognition.

Uses the official ``discord.py`` library (no fork) with a custom
:class:`~receiver.VoiceReceiver` that handles UDP packet decryption,
Opus decoding, and VAD speech segmentation from scratch.
"""

from __future__ import annotations

import asyncio
import json
import types
from typing import Callable, Awaitable, Any, Coroutine

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
        self._ws_original_received: Any = None  # saved for cleanup

    # -- lifecycle ----------------------------------------------------------

    async def on_ready(self) -> None:
        """Called when the bot has successfully logged in."""
        channel = self.get_channel(self._channel_id)
        if channel is None:
            self._connected.set()
            return

        if not isinstance(channel, discord.VoiceChannel):
            self._connected.set()
            return

        try:
            self._voice_client = await channel.connect()
        except Exception:
            self._connected.set()
            raise

        # Attach our custom voice receiver directly to the VoiceClient.
        loop = asyncio.get_running_loop()
        self._receiver = VoiceReceiver(self._config, self._on_segment, loop)
        self._receiver.attach(self._voice_client)

        # Hook the voice websocket to capture SPEAKING events (opcode 5).
        # Discord sends these over the voice websocket to tell us which
        # SSRC maps to which user — this is the only way to get that
        # mapping with official discord.py (no Sink SDK).
        self._hook_voice_ws(channel)

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

    def _hook_voice_ws(self, channel: discord.VoiceChannel) -> None:
        """Monkey-patch the voice websocket to capture SSRC→user mappings.

        The Discord voice gateway sends SPEAKING events (opcode 5) that
        contain ``{user_id, ssrc, speaking}``.  We intercept these to
        resolve SSRCs to real Discord user IDs and display names.
        """
        if self._voice_client is None or self._receiver is None:
            return

        voice_ws = self._voice_client.ws
        if not hasattr(voice_ws, "received_message"):
            return

        # Save the original bound method for later restoration.
        _original_received = voice_ws.received_message
        self._ws_original_received = _original_received
        receiver = self._receiver
        guild = channel.guild

        async def _patched_received(ws_self: Any, msg: str) -> Coroutine[Any, Any, None]:
            # Process SPEAKING events to learn SSRC→user mappings.
            try:
                data = json.loads(msg)
            except (json.JSONDecodeError, TypeError):
                pass
            else:
                if data.get("op") == 5:  # Voice Gateway SPEAKING event
                    d = data.get("d", {})
                    user_id = str(d.get("user_id", ""))
                    ssrc = d.get("ssrc", 0)
                    if user_id and ssrc:
                        # Resolve display name from the guild member cache.
                        user_name = user_id
                        if guild is not None:
                            member = guild.get_member(int(user_id))
                            if member is not None:
                                user_name = member.display_name
                        receiver.register_ssrc(ssrc, user_id, user_name)

            # Always forward to the original handler so discord.py internals
            # (heartbeat ACKs, session descriptions, etc.) continue to work.
            return await _original_received(msg)

        voice_ws.received_message = types.MethodType(_patched_received, voice_ws)

    def _unhook_voice_ws(self) -> None:
        """Restore the original voice websocket message handler."""
        if self._voice_client is None or self._ws_original_received is None:
            return
        voice_ws = self._voice_client.ws
        if hasattr(voice_ws, "received_message"):
            voice_ws.received_message = self._ws_original_received
        self._ws_original_received = None
