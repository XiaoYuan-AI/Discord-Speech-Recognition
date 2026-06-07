"""Audio format conversion helpers (PCM conversion, RMS VAD calculation).

The voice receive and VAD segmentation logic now lives in :mod:`receiver`.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, resample_poly


# 193-tap Kaiser FIR with cutoff at 7.2 kHz / 24 kHz Nyquist.  Designed
# once at import time and reused for every Discord PCM frame.
_RESAMPLE_FIR_48_TO_16 = firwin(
    numtaps=193, cutoff=7200.0, fs=48000.0, window=("kaiser", 8.6)
).astype(np.float32)


def discord_pcm_to_mono_16k(data: bytes) -> np.ndarray:
    """Convert Discord stereo 48 kHz int16 PCM to mono 16 kHz int16.

    Discord delivers 20 ms frames:
    - Stereo:  1920 samples (960 per channel) → 3840 bytes
    - Mono:     960 samples                → 1920 bytes

    Stereo is detected heuristically by sample count.  Stereo frames are
    mixed down by averaging both channels, then resampled with a low-pass
    polyphase filter to avoid folding high-frequency energy into the
    speech band.
    """
    if not data:
        return np.array([], dtype=np.int16)

    samples = np.frombuffer(data, dtype=np.int16)
    n = len(samples)

    if n >= 1500 and n % 2 == 0:
        stereo = samples.reshape(-1, 2).astype(np.int32)
        mono_48k = ((stereo[:, 0] + stereo[:, 1]) // 2).astype(np.int16)
    else:
        mono_48k = samples.copy()

    resampled = resample_poly(
        mono_48k.astype(np.float32),
        up=1,
        down=3,
        window=_RESAMPLE_FIR_48_TO_16,
    )
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def rms(frame: np.ndarray) -> float:
    """Root-mean-square amplitude normalised to [0, 1]."""
    if len(frame) == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) / 32768.0)
