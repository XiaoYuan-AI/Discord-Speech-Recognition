# Discord Speech Recognition SDK

Real-time voice-channel speech-to-text for Discord bots. Supports local AI
models (faster-whisper) and cloud services -- no temp files, everything
stays in memory.

## Features

- **Real-time transcription** -- when someone speaks in a voice channel, you
  get the text as soon as they finish their sentence.
- **Zero temp files** -- audio data flows as in-memory numpy arrays from
  Discord to the recognizer. Nothing touches disk.
- **Three recognizer backends**:
  - **Local Whisper** (`faster-whisper`) -- runs entirely on your machine.
    Supports GPU (CUDA) or CPU. No API call costs.
  - **Google Speech Recognition** -- free-tier cloud service. No API key
    required, but has rate limits.
  - **OpenAI Whisper API** -- high-quality cloud transcription. Requires an
    API key.
- **Per-user segmentation** -- each speaker's audio is buffered separately
  and recognized with Voice Activity Detection (VAD), so overlapping
  speech is handled correctly.
- **Configurable VAD** -- tune speech/silence thresholds, minimum speech
  duration, and silence timeout.

## Installation

Requires Python 3.13+.

```bash
# Clone and enter the project
cd Discord-Speech-Recognition

# Install base dependencies (discord.py, numpy)
uv sync

# Install recognizer extras (pick one or more):
uv sync --extra local    # faster-whisper (local AI)
uv sync --extra google   # Google Speech Recognition
uv sync --extra openai   # OpenAI Whisper API
uv sync --extra all      # everything
```

## Quick start

### 1. Create a Discord bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Create a new application and add a bot.
3. Copy the **bot token**.
4. Invite the bot to your server with these permissions:
   - `Connect` and `Speak` (voice channel)
   - `Use Voice Activity` (voice channel)

### 2. Run the example

```bash
# Set your credentials
export DISCORD_TOKEN="your-bot-token-here"
export VOICE_CHANNEL="123456789012345678"  # right-click a voice channel -> Copy ID

# Start transcribing (local Whisper -- tiny model for fastest start)
python main.py
```

### 3. Use as a library

```python
import asyncio
from discord_speech_recognition import (
    SpeechRecognitionClient,
    RecognitionConfig,
    RecognitionResult,
)

async def on_text(result: RecognitionResult) -> None:
    print(f"[{result.user_name}] {result.text}")

async def main():
    config = RecognitionConfig(
        recognizer="whisper_local",
        model_size="base",
        language="auto",
    )
    client = SpeechRecognitionClient(config, on_text)
    await client.start("YOUR_BOT_TOKEN", VOICE_CHANNEL_ID)
    await asyncio.sleep(3600)  # listen for 1 hour
    await client.stop()

asyncio.run(main())
```

## Recognizer backends

### Local Whisper (`faster-whisper`)

```python
config = RecognitionConfig(
    recognizer="whisper_local",
    model_size="base",      # tiny | base | small | medium | large-v3
    device="cpu",            # or "cuda"
    compute_type="int8",     # int8 | float16 | float32
    language="auto",         # "en", "zh", "ja", etc. -- or "auto"
)
```

Model sizes and approximate VRAM:

| Size     | VRAM      | Speed (relative) |
|----------|-----------|------------------|
| tiny     | ~1 GB     | fastest          |
| base     | ~1 GB     | fast             |
| small    | ~2 GB     | moderate         |
| medium   | ~5 GB     | slower           |
| large-v3 | ~10 GB    | slowest          |

The model is downloaded automatically on first use from Hugging Face.

### Google Speech Recognition

```python
config = RecognitionConfig(
    recognizer="google",
    google_language="en-US",  # en-US | zh-CN | ja-JP | ...
)
```

Free tier -- no API key needed. Best for low-to-moderate volume. May be
rate-limited by Google.

### OpenAI Whisper API

```python
import os

config = RecognitionConfig(
    recognizer="whisper_api",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    language="en",            # optional language hint
)
```

Requires an OpenAI API key. Charged per minute of audio.

## VAD tuning

The Voice Activity Detector (VAD) controls how aggressively the SDK
segments speech. All values can be tuned via `RecognitionConfig`:

```python
config = RecognitionConfig(
    speech_threshold=0.015,      # RMS above this = speech (0.0-1.0)
    silence_threshold=0.008,     # RMS below this = silence
    min_speech_duration_ms=300,  # ignore utterances shorter than this
    max_speech_duration_ms=15000,# force-split after 15s of continuous speech
    silence_duration_ms=800,     # 800ms of silence ends a segment
)
```

Tuning tips:
- If short words are missed, lower `speech_threshold` and/or
  `min_speech_duration_ms`.
- If background noise triggers false positives, raise `speech_threshold`.
- If sentences are cut too early, increase `silence_duration_ms`.
- If one user speaks continuously for a long time, the segment is
  force-split at `max_speech_duration_ms`.

## API reference

### `SpeechRecognitionClient`

```python
client = SpeechRecognitionClient(config, on_recognition)
await client.start(token, channel_id)  # connect & begin listening
await client.stop()                    # disconnect & cleanup
```

| Method        | Description                                          |
|---------------|------------------------------------------------------|
| `start(t, c)` | Connect bot to voice channel `c` and start listening |
| `stop()`      | Disconnect and release all resources                 |
| `config`      | Read-only snapshot of the current config             |
| `is_listening`| `True` while connected and listening                 |

### `RecognitionResult`

Passed to your `on_recognition` callback:

| Attribute         | Type      | Description                            |
|-------------------|-----------|----------------------------------------|
| `user_id`         | `str`     | Discord user snowflake ID              |
| `user_name`       | `str`     | Discord display name                   |
| `text`            | `str`     | Transcribed text                       |
| `language`        | `str`     | Detected language code                 |
| `confidence`      | `float`   | Confidence score (0.0-1.0)             |
| `timestamp`       | `datetime`| UTC timestamp when recognition finished|
| `duration_ms`     | `int`     | Audio segment length in milliseconds   |
| `recognizer_name` | `str`     | Backend that produced this result      |
| `is_empty`        | `bool`    | `True` if no speech was detected       |

### `RecognitionConfig`

All configuration options:

| Field                   | Type        | Default           | Description                            |
|-------------------------|-------------|-------------------|----------------------------------------|
| `recognizer`            | `str`       | `"whisper_local"` | `whisper_local` / `google` / `whisper_api` |
| `language`              | `str`       | `"auto"`          | Language code or `"auto"`             |
| `model_size`            | `str`       | `"base"`          | Whisper model size (local only)       |
| `device`                | `str`       | `"cpu"`           | `cpu` or `cuda` (local only)          |
| `compute_type`          | `str`       | `"int8"`          | Compute precision (local only)        |
| `openai_api_key`        | `str\|None` | `None`            | OpenAI API key (whisper_api only)     |
| `google_language`       | `str`       | `"en-US"`         | Language for Google recognizer        |
| `speech_threshold`      | `float`     | `0.015`           | VAD speech energy threshold           |
| `silence_threshold`     | `float`     | `0.008`           | VAD silence energy threshold          |
| `min_speech_duration_ms`| `int`       | `300`             | Min utterance duration (ms)           |
| `max_speech_duration_ms`| `int`       | `15000`           | Max continuous speech before split    |
| `silence_duration_ms`   | `int`       | `800`             | Silence before segment ends           |
| `target_sample_rate`    | `int`       | `16000`           | Output sample rate (Hz)               |

## Architecture

The receiver is built directly on the official ``discord.py`` library without
any Sink SDK.  It uses ``VoiceConnectionState.add_socket_listener`` to receive
raw UDP packets, then handles RTP parsing, decryption (4 encryption modes),
Opus decoding, and VAD speech segmentation from scratch.

```
Discord Voice Channel
    |
    v
Raw UDP packets              <- SocketReader thread in discord.py
    |
    v
VoiceReceiver._on_raw_packet()  <- RTP parse, decrypt (NaCl), Opus decode
    |
    v
Mono 16kHz conversion        <- numpy-based, stereo→mono, 48k→16k Hz
    |
    v
RMS VAD per SSRC             <- energy-based voice activity detection
    |
    v
Speech segment               <- accumulated PCM when silence > threshold
    |                            (cross-thread dispatch via run_coroutine_threadsafe)
    v
BaseRecognizer.recognize()   <- thread pool / async
    |
    v
RecognitionResult            <- fired to user callback
```

SSRC-to-user mapping is resolved via the voice websocket's SPEAKING events
(opcode 5), which are intercepted by a lightweight monkey-patch on the
voice websocket's message handler.  This is the only way to get per-user
attribution with official ``discord.py`` (the fork ``py-cord`` is NOT used).

## License

MIT
