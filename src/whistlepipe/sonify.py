"""Image -> sound: the paper's own chain (Fig. 2) run on any 128 x 128 whistle image, real or generated.

    image --(threshold 0.05)--> contour --> time-frequency vector --> constant-amplitude NFM signal (Fig. 2(b)-(d))

The image carries the contour SHAPE only: its centre frequency was removed in S8 (paper E-08), so the caller supplies
one (the whistle's own, or one drawn from the training data). The result is a pure tone that follows the contour: no
harmonics, no amplitude envelope, no noise. That is all the model represents.

[EXPLICIT] threshold 0.05 (Fig. 2); constant-amplitude NFM signal (Fig. 2(d)).
[IMPLIED]  grid of S8/S9: column c is centred (c + 1) x 5.333 ms after the image start; row r is (63 - r) x 93.75 Hz
           from the centre frequency (representation.row_to_freq_hz).
[DECISION] per column, the frequency is the intensity-weighted centroid of the contiguous above-threshold run that
           holds the column maximum (the ridge). On real images this equals the plain centroid; on generated images
           it ignores stray pixels above the threshold away from the ridge. Empty interior columns are linearly
           interpolated; the tone spans the first..last active column, one hop around each column centre (as S8).
[DECISION] audio: 48 kHz (96 kHz when the tone would exceed 0.45 x 48 kHz), raised-cosine on/off ramps of the
           S8 duration (edge_ramp_samples at the 12 kHz baseband rate = 2.67 ms), 20 ms of silence either side,
           peak -3 dBFS, 16-bit PCM WAV.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Cfg
from .representation import nfm_grid


@dataclass
class Contour:
    """Frequency offset from the centre frequency for every column from the first to the last active one."""

    cols: np.ndarray          # int, consecutive image columns
    offset_hz: np.ndarray     # float, one per column
    hop_s: float              # seconds per column

    @property
    def duration_s(self) -> float:
        return len(self.cols) * self.hop_s

    def to_json(self) -> dict:
        return {"col0": int(self.cols[0]), "hop_s": self.hop_s, "offset_hz": [round(float(v), 1) for v in self.offset_hz]}


def grid(cfg: Cfg) -> dict:
    p = cfg["paper"]
    g = nfm_grid(cfg)
    return {"bin_hz": float(p["freq_resolution_hz"]), "hop_s": float(p["hop"]) / float(p["fs_target_hz"]),
            "zero_row": int(p["image_height"]) // 2 - 1,
            "ramp_s": int(cfg["decisions"]["s8"]["edge_ramp_samples"]) / g["fs_b"]}


def image_contour(img: np.ndarray, cfg: Cfg, threshold: float = 0.05) -> Contour | None:
    """(128, 128) image (row 0 = highest frequency) -> Contour, or None when no pixel exceeds the threshold."""
    g = grid(cfg)
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    cols, rows_hz = [], []
    for c in range(w):
        col = img[:, c]
        r0 = int(np.argmax(col))
        if col[r0] <= threshold:
            continue
        lo = r0
        while lo > 0 and col[lo - 1] > threshold:
            lo -= 1
        hi = r0
        while hi < h - 1 and col[hi + 1] > threshold:
            hi += 1
        r = np.arange(lo, hi + 1)
        wts = col[lo:hi + 1]
        cols.append(c)
        rows_hz.append(float(((g["zero_row"] - r) * g["bin_hz"] * wts).sum() / wts.sum()))
    if not cols:
        return None
    cols = np.asarray(cols)
    full = np.arange(cols[0], cols[-1] + 1)
    return Contour(full, np.interp(full, cols, np.asarray(rows_hz)), g["hop_s"])


def audio_rate(f_max_hz: float, low: int = 48000, high: int = 96000) -> int:
    return low if f_max_hz <= 0.45 * low else high


def synthesize(contour: Contour, fc_hz: float, cfg: Cfg, fs: int | None = None, slow: float = 1.0,
               pad_s: float = 0.02, peak_dbfs: float = -3.0, low_rate: int = 48000, high_rate: int = 96000) -> tuple[np.ndarray, int]:
    """Constant-amplitude tone following fc + contour. ``slow`` stretches time (same pitch). Returns (float32, fs)."""
    g = grid(cfg)
    f = np.maximum(fc_hz + contour.offset_hz, 20.0)
    fs = int(fs or audio_rate(float(f.max()), low_rate, high_rate))
    hop = contour.hop_s * slow
    dur = len(f) * hop
    n_tone = int(round(dur * fs))
    t = (np.arange(n_tone) + 0.5) / fs
    centres = (np.arange(len(f)) + 0.5) * hop
    inst = np.interp(t, centres, f)                                   # held constant before/after the first/last centre
    phase = 2.0 * np.pi * np.cumsum(inst) / fs
    x = np.cos(phase)
    ramp = min(int(round(g["ramp_s"] * fs)), n_tone // 2)
    if ramp > 1:
        r = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp) / ramp))
        x[:ramp] *= r
        x[-ramp:] *= r[::-1]
    x *= 10.0 ** (peak_dbfs / 20.0) / max(1e-12, float(np.abs(x).max()))
    pad = np.zeros(int(round(pad_s * fs)))
    return np.concatenate([pad, x, pad]).astype(np.float32), fs


def write_wav(path: str | Path, x: np.ndarray, fs: int) -> None:
    sf.write(str(path), np.clip(x, -1.0, 1.0), int(fs), subtype="PCM_16", format="WAV")


def image_to_wav(img: np.ndarray, fc_hz: float, path: str | Path, cfg: Cfg, threshold: float = 0.05, **kw) -> dict | None:
    """Render one image to a WAV; returns what the page needs (contour, rate, duration, frequency range), or None
    for an image with no contour (nothing above the threshold)."""
    c = image_contour(img, cfg, threshold)
    if c is None:
        return None
    x, fs = synthesize(c, fc_hz, cfg, **kw)
    write_wav(path, x, fs)
    f = fc_hz + c.offset_hz
    return {"wav": Path(path).name, "fs": fs, "duration_s": round(c.duration_s, 4), "fc_hz": round(float(fc_hz), 1),
            "f_min_hz": round(float(f.min()), 1), "f_max_hz": round(float(f.max()), 1), "contour": c.to_json()}
