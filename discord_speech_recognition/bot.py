"""Minimal async Discord bot for voice-channel speech recognition.

The bot joins a voice channel, starts an :class:`~audio.DiscordAudioSink`,
and pipes received speech segments back to the SDK client.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Awaitable

import discord

from .audio import DiscordAudioSink
from .config import RecognitionConfig
from .types import UserAudioSegment


class VoiceRecognitionBot(discord.Client):
    """A minimal Discord client that joins a voice channel and records.

    This is an internal class — users interact with
    :class:`~sdk.SpeechRecognitionClient` instead.

    Parameters:
        config: The SDK recognition configuration.
        on_segment: Callback invoked for each completed speech segment.
        token: Discord bot token.
        channel_id: ID of the voice channel to join.
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
        self._sink: DiscordAudioSink | None = None
        self._connected = asyncio.Event()

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

        # Start the audio sink and recording.
        loop = asyncio.get_running_loop()
        self._sink = DiscordAudioSink(self._config, self._on_segment, loop=loop)
        self._voice_client.start_recording(self._sink, _noop_callback)
        self._connected.set()

    async def connect_and_listen(self) -> None:
        """Start the bot's asyncio task and join the voice channel.

        Blocks until the voice connection is established (or fails).
        """
        asyncio.create_task(self.start(self._token))
        await self._connected.wait()

    async def shutdown(self) -> None:
        """Gracefully stop recording, disconnect from voice, and close."""
        if self._voice_client is not None and self._voice_client.is_connected():
            self._voice_client.stop_recording()
            await self._voice_client.disconnect()
        await self.close()


def _noop_callback(*_args) -> None:
    """No-op after-recording callback (py-cord requires one)."""
    pass
