"""Example: Real-time Discord voice channel transcription.

This script connects a bot to a voice channel and prints every
spoken sentence to the console in real time.

Prerequisites:
    1. A Discord bot token (from the Discord Developer Portal).
    2. The bot must have the "Voice Connect" and "Use Voice Activity"
       permissions in the target server.
    3. (Optional) A model for the local recognizer.

Setup:
    uv sync
    uv sync --extra local   # for local Whisper support
    uv sync --extra google  # for Google SR support
    uv sync --extra openai  # for OpenAI Whisper API support
    uv sync --extra all     # install everything

Usage:
    python main.py

Environment variables:
    DISCORD_TOKEN   — Discord bot token
    VOICE_CHANNEL   — Discord voice channel ID (snowflake integer)
    OPENAI_API_KEY  — API key for OpenAI Whisper (only if using whisper_api)
"""

import asyncio
import os

from discord_speech_recognition import (
    SpeechRecognitionClient,
    RecognitionConfig,
    RecognitionResult,
)


async def on_recognition(result: RecognitionResult) -> None:
    """Called for every transcribed speech segment."""
    timestamp = result.timestamp.strftime("%H:%M:%S")
    print(f"[{timestamp}] {result.user_name}: {result.text}")


async def main() -> None:
    token = os.environ.get("DISCORD_TOKEN")
    channel_id_str = os.environ.get("VOICE_CHANNEL")

    if not token:
        print("Error: DISCORD_TOKEN environment variable is not set.")
        return
    if not channel_id_str:
        print("Error: VOICE_CHANNEL environment variable is not set.")
        return

    channel_id = int(channel_id_str)

    # --- Configuration ------------------------------------------------
    # Change "whisper_local" to "google" or "whisper_api" as needed.
    config = RecognitionConfig(
        recognizer="whisper_local",
        language="auto",            # auto-detect language
        model_size="base",          # tiny / base / small / medium / large-v3
        device="cpu",               # or "cuda"
        # VAD tuning (adjust if speech is missed or silence is too long):
        speech_threshold=0.015,
        silence_duration_ms=800,
        min_speech_duration_ms=300,
    )

    if config.recognizer == "whisper_api":
        config.openai_api_key = os.environ.get("OPENAI_API_KEY")

    # --- Start transcribing -------------------------------------------
    client = SpeechRecognitionClient(config, on_recognition)
    await client.start(token, channel_id)

    print("Listening… Press Ctrl+C to stop.")
    try:
        # Run until interrupted.
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
