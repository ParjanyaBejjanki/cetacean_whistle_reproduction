"""S5 input conditioning (INTERNAL to S5; the S4 output itself is never modified).

This is a [REPLACEMENT] design decision required by spec FR-02 ("scaling/normalisation before the detector is
part of the detector's own contract"). It uses a local per-bin noise-floor estimate so that the network sees
level-independent contrast. The same function is used by training (synthetic data) and inference.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def noise_floor_db(band_db: np.ndarray, block_frames: int, median_blocks: int) -> np.ndarray:
    """Local per-bin noise floor: median over blocks of ``block_frames`` frames, then a median across
    ``median_blocks`` neighbouring blocks, linearly interpolated back to frames. Returns (T, F) float32."""
    t, f = band_db.shape
    if t == 0:
        return np.zeros_like(band_db)
    nb = max(1, math.ceil(t / block_frames))
    pad = nb * block_frames - t
    x = band_db
    if pad:
        x = np.pad(x, ((0, pad), (0, 0)), mode="reflect" if pad < t else "edge")
    blocks = np.median(x.reshape(nb, block_frames, f), axis=1)                      # (nb, F)
    smooth = ndimage.median_filter(blocks, size=(median_blocks, 1), mode="nearest")
    u = (np.arange(t) + 0.5) / block_frames - 0.5
    fl = np.floor(u)
    i0 = np.clip(fl.astype(int), 0, nb - 1)
    i1 = np.clip(i0 + 1, 0, nb - 1)
    w = np.clip(u - fl, 0.0, 1.0)[:, None]
    return ((1.0 - w) * smooth[i0] + w * smooth[i1]).astype(np.float32)


def condition(band_db: np.ndarray, cond: dict) -> np.ndarray:
    """(T, F) dB band -> network input, (T, F) float32 ~ O(1): clip(dB - local floor, lo, hi) / scale."""
    floor = noise_floor_db(band_db, int(cond["block_frames"]), int(cond["median_blocks"]))
    x = np.clip(band_db - floor, float(cond["clip_lo_db"]), float(cond["clip_hi_db"])) / float(cond["scale_db"])
    return x.astype(np.float32)
