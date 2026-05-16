"""Audio format conversion helpers (PCM conversion, RMS VAD calculation).

The voice receive and VAD segmentation logic now lives in :mod:`receiver`.
"""

from __future__ import annotations

import numpy as np


def discord_pcm_to_mono_16k(data: bytes) -> np.ndarray:
    """Convert Discord stereo 48 kHz int16 PCM to mono 16 kHz int16.

    Discord delivers 20 ms frames:
    - Stereo:  1920 samples (960 per channel) → 3840 bytes
    - Mono:     960 samples                → 1920 bytes

    Stereo is detected heuristically by sample count.
    """
    if not data:
        return np.array([], dtype=np.int16)

    samples = np.frombuffer(data, dtype=np.int16)
    n = len(samples)

    if n >= 1500 and n % 2 == 0:
        # Interleaved stereo → take left channel
        mono = samples[::2].copy()
    else:
        mono = samples.copy()

    # Decimate 3× (48 → 16 kHz)
    return mono[::3]


def rms(frame: np.ndarray) -> float:
    """Root-mean-square amplitude normalised to [0, 1]."""
    if len(frame) == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) / 32768.0)
