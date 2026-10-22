"""Image -> binary contour -> time-frequency vector -> the 7 whistle features (paper Fig. 2, §IV-B-1, §IV-D-4).

[EXPLICIT] threshold 0.05 (Fig. 2); features L, f_min, f_max, delta f, f_start, f_end, f_c.
[DECISION] per active column the frequency is the mean row index of the above-threshold pixels (binary image);
           rows -> "bin" index b = 128 - row (1-based, low frequency at the bottom, 0 Hz = bin 65);
           L = last - first active column + 1 (samples); f_c = (f_min + f_max) / 2.
Units: L in samples (5.333 ms), frequencies in bins (93.75 Hz).
"""
from __future__ import annotations

import numpy as np
import torch

FEATURES = ("L", "f_min", "f_max", "df", "f_start", "f_end", "f_c")


@torch.no_grad()
def contour_vectors(img: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """(B, 128, 128) or (B, 1, 128, 128) -> (B, 128) frequency in bins per column, NaN where the column is empty."""
    if img.dim() == 4:
        img = img[:, 0]
    m = (img > threshold).to(torch.float32)
    cnt = m.sum(1)                                                    # (B, W)
    rows = torch.arange(img.shape[1], device=img.device, dtype=torch.float32)
    mean_row = (m * rows[None, :, None]).sum(1) / cnt.clamp_min(1)
    b = img.shape[1] - mean_row                                       # bin = 128 - row
    return torch.where(cnt > 0, b, torch.full_like(b, float("nan")))


@torch.no_grad()
def contour_features(img: torch.Tensor, threshold: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Features (B, 7) in FEATURES order and a validity mask (B,) (False when no pixel exceeds the threshold)."""
    v = contour_vectors(img, threshold)
    act = ~torch.isnan(v)
    valid = act.any(1)
    w = v.shape[1]
    first = act.to(torch.int8).argmax(1)
    last = w - 1 - act.flip(1).to(torch.int8).argmax(1)
    big = torch.tensor(1e9, device=v.device)
    fmin = torch.where(act, v, big).min(1).values
    fmax = torch.where(act, v, -big).max(1).values
    ar = torch.arange(v.shape[0], device=v.device)
    fs, fe = v[ar, first], v[ar, last]
    feats = torch.stack([(last - first + 1).float(), fmin, fmax, fmax - fmin, fs, fe, 0.5 * (fmin + fmax)], 1)
    feats[~valid] = float("nan")
    return feats, valid


def features_numpy(images: np.ndarray, threshold: float = 0.05, device: str = "cpu", batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    out, ok = [], []
    for i in range(0, len(images), batch):
        f, v = contour_features(torch.as_tensor(np.asarray(images[i:i + batch], dtype=np.float32), device=device), threshold)
        out.append(f.cpu().numpy())
        ok.append(v.cpu().numpy())
    return np.concatenate(out), np.concatenate(ok)


def satisfies(feats: np.ndarray | torch.Tensor, feature: str, lo: float, hi: float):
    """lo <= feature <= hi (Table 3 conditions are inclusive ranges); NaN rows never satisfy."""
    j = FEATURES.index(feature)
    x = feats[..., j]
    return (x >= lo) & (x <= hi)
