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

Environment variables (or place them in a .env file):
    DISCORD_TOKEN   — Discord bot token
    VOICE_CHANNEL   — Discord voice channel ID (snowflake integer)
    OPENAI_API_KEY  — API key for OpenAI Whisper (only if using whisper_api)
"""

import asyncio
import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configure logging — show info from our package so you can see what's
# happening under the hood.  Set to logging.DEBUG for even more detail.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("discord_speech_recognition").setLevel(logging.INFO)
# Uncomment the next line to see every individual voice packet:
# logging.getLogger("discord_speech_recognition.receiver").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Load .env file if present (no extra dependency required)
# ---------------------------------------------------------------------------
_dotenv_path = Path(__file__).parent / ".env"
if _dotenv_path.exists():
    with open(_dotenv_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip().strip('"').strip("'")
            if _key and _key not in os.environ:
                os.environ[_key] = _val

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
