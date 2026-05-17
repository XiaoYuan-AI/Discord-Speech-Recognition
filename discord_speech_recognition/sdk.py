"""Main SDK entry point — SpeechRecognitionClient.

Orchestrates the Discord bot, audio sink, and speech recognizer
to provide real-time voice-channel transcription with zero temp files.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable, Optional

import numpy as np

from .bot import VoiceRecognitionBot
from .config import RecognitionConfig
from .recognizers import (
    BaseRecognizer,
    GoogleRecognizer,
    LocalWhisperRecognizer,
    WhisperAPIRecognizer,
)
from .types import RecognitionResult, UserAudioSegment

logger = logging.getLogger(__name__)

RecognitionCallback = Callable[[RecognitionResult], Awaitable[None]]


def _build_recognizer(config: RecognitionConfig) -> BaseRecognizer:
    """Factory: create the appropriate recognizer from config."""
    if config.recognizer == "whisper_local":
        return LocalWhisperRecognizer(config)
    if config.recognizer == "whisper_api":
        return WhisperAPIRecognizer(config)
    if config.recognizer == "google":
        return GoogleRecognizer(config)
    raise ValueError(f"Unknown recognizer: {config.recognizer}")


class SpeechRecognitionClient:
    """Real-time speech recognition for Discord voice channels.

    Typical usage::

        import asyncio
        from discord_speech_recognition import (
            SpeechRecognitionClient,
            RecognitionConfig,
            RecognitionResult,
        )

        async def on_text(result: RecognitionResult) -> None:
            print(f"[{result.user_name}] {result.text}")

        async def main():
            config = RecognitionConfig(recognizer="whisper_local")
            client = SpeechRecognitionClient(config, on_text)
            await client.start("YOUR_BOT_TOKEN", 1234567890)
            await asyncio.sleep(3600)  # run for 1 hour
            await client.stop()

        asyncio.run(main())

    Parameters:
        config: Recognition configuration (model, language, VAD thresholds).
        on_recognition: Async callback invoked each time a user's speech
            is transcribed. Receives a :class:`RecognitionResult`.
    """

    def __init__(
        self,
        config: RecognitionConfig | None = None,
        on_recognition: Optional[RecognitionCallback] = None,
    ) -> None:
        self._config = config or RecognitionConfig()
        self._on_recognition = on_recognition
        self._recognizer: Optional[BaseRecognizer] = None
        self._bot: Optional[VoiceRecognitionBot] = None

    @property
    def config(self) -> RecognitionConfig:
        """The current configuration (read-only snapshot)."""
        return self._config

    @property
    def is_listening(self) -> bool:
        """True if the bot is connected and listening to a voice channel."""
        return self._bot is not None

    # -- public API ---------------------------------------------------------

    async def start(self, token: str, channel_id: int) -> None:
        """Connect to Discord and begin transcribing a voice channel.

        Args:
            token: Discord bot token.
            channel_id: Snowflake ID of the voice channel to join.

        Raises:
            discord.LoginFailure: If the token is invalid.
            RuntimeError: If already started.
        """
        if self._bot is not None:
            raise RuntimeError("Already started — call stop() first.")

        self._recognizer = _build_recognizer(self._config)
        logger.info("Starting speech recognition — recognizer=%s", self._recognizer.name)

        self._bot = VoiceRecognitionBot(
            config=self._config,
            on_segment=self._on_audio_segment,
            token=token,
            channel_id=channel_id,
        )

        await self._bot.connect_and_listen()
        logger.info("Connected to voice channel %s", channel_id)

    async def stop(self) -> None:
        """Disconnect from the voice channel and release resources."""
        if self._bot is None:
            return

        logger.info("Stopping speech recognition…")
        await self._bot.shutdown()
        self._bot = None

        if self._recognizer is not None:
            await self._recognizer.close()
            self._recognizer = None
        logger.info("Stopped.")

    # -- internal -----------------------------------------------------------

    async def _on_audio_segment(self, segment: UserAudioSegment) -> None:
        """Callback from the receiver — transcribe and fire user callback."""
        if self._recognizer is None or self._on_recognition is None:
            return

        audio = np.frombuffer(segment.pcm_data, dtype=np.int16)

        # Skip segments that are too short to contain useful speech.
        duration_ms = len(audio) / segment.sample_rate * 1000
        if duration_ms < self._config.min_speech_duration_ms:
            logger.debug(
                "Skipping short segment from %s: %.0f ms < %d ms",
                segment.user_name,
                duration_ms,
                self._config.min_speech_duration_ms,
            )
            return

        logger.info(
            "Transcribing segment: user=%s duration=%.0fms samples=%d",
            segment.user_name,
            duration_ms,
            len(audio),
        )

        try:
            result = await self._recognizer.recognize(
                audio=audio,
                sample_rate=segment.sample_rate,
                user_id=segment.user_id,
                user_name=segment.user_name,
                language=self._config.language,
            )
        except Exception:
            logger.exception("Recognition failed for segment from %s", segment.user_name)
            return

        if not result.is_empty:
            logger.info(
                "Recognized: %s → \"%s\" (confidence=%.2f)",
                segment.user_name,
                result.text,
                result.confidence,
            )
            await self._on_recognition(result)
        else:
            logger.debug("Recognition produced empty result for %s", segment.user_name)
