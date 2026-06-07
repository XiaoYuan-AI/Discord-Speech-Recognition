# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python package for Discord voice transcription. Source code lives in `discord_speech_recognition/`:

- `bot.py`, `receiver.py`, and `sdk.py` handle Discord connection, voice packet receiving, and public orchestration.
- `audio.py`, `types.py`, and `config.py` contain shared audio utilities, dataclasses, and configuration.
- `recognizers/` contains backend implementations for local Whisper, OpenAI Whisper API, and Google Speech Recognition.
- `main.py` is an executable example for running the bot locally.
- Tests live in `tests/` and mirror the main modules, for example `tests/test_receiver.py`.

## Build, Test, and Development Commands

Use `uv` for dependency management and command execution.

```bash
uv sync --extra local      # install local Whisper support
uv sync --extra all        # install all optional recognizer backends
uv run pytest             # run the full test suite
uv run python main.py     # run the example bot after setting env vars
git diff --check          # catch whitespace errors before committing
```

The example bot expects `DISCORD_TOKEN` and `VOICE_CHANNEL`; `OPENAI_API_KEY` is only needed for `whisper_api`.

## Coding Style & Naming Conventions

Use Python 3.13 syntax and standard 4-space indentation. Prefer type hints on public and cross-module functions. Keep async boundaries explicit: Discord and SDK entry points are async, while blocking recognizer calls should run through an executor. Use `snake_case` for functions and variables, `PascalCase` for classes, and private `_name` helpers for internal implementation details.

Keep changes scoped. Do not refactor unrelated receiver, recognizer, or test code when fixing a narrow bug.

## Testing Guidelines

Tests use pytest. Add or update focused tests for every behavioral change, especially around RTP/decryption, VAD segmentation, recognizer options, and SDK callbacks. Name tests as `test_<behavior>` and place them in the matching `tests/test_*.py` file.

Before submitting, run:

```bash
uv run pytest
git diff --check
```

## Commit & Pull Request Guidelines

Recent history uses concise fixes, and new commits should follow Conventional Commits, for example `fix(voice): receive encrypted Discord audio` or `feat(recognizer): add local whisper tuning`.

Pull requests should include a short summary, test results, and any required runtime configuration changes. Link issues when applicable. For voice or transcription changes, mention the affected backend and whether Discord receive behavior, segmentation, or recognizer output changed.

## Security & Configuration Tips

Never commit tokens, API keys, `.env` files, model caches, or local artifacts such as `.DS_Store`. Keep credentials in environment variables and document any new required settings in `README.md`.
