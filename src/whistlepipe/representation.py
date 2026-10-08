"""S7 whistle vectors, S8 centre-frequency removal + constant-amplitude NFM baseband signal, S9 128x128 image.

Paper-defined (spec): vector -> constant-amplitude NFM baseband signal with the centre frequency removed ->
its spectrogram -> 128 x 128 single-channel image with values in [0, 1], centre frequency at the image centre.
Everything else here (centre-frequency definition, time centring, complex baseband, sample rate, edge ramps, magnitude
quantity, per-image max normalisation, orientation, long-track policy) is an IMPLEMENTATION DECISION (config.toml).

Baseband grid is DERIVED from the paper numbers, not chosen: image_height x 93.75 Hz = 12 kHz complex baseband rate;
a 128-sample Hamming window with hop 64 gives the same 10.667 ms / 5.333 ms / 93.75 Hz grid as S4, so one image column
equals one S4 frame and one image row equals one S4 bin (spec E-09/E-11).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .config import Cfg
from .tracking import Track


# ----------------------------------------------------------------------------- S7
@dataclass
class WhistleVector:
    track_id: int
    start_frame: int
    freq_hz: np.ndarray        # one value per consecutive S4 frame
    conf: np.ndarray
    observed_fraction: float

    @property
    def n_frames(self) -> int:
        return len(self.freq_hz)

    @property
    def f_min(self) -> float:
        return float(self.freq_hz.min())

    @property
    def f_max(self) -> float:
        return float(self.freq_hz.max())

    @property
    def mean_conf(self) -> float:
        return float(self.conf[self.conf > 0].mean()) if (self.conf > 0).any() else 0.0


def s7_vectors(tracks: list[Track], cfg: Cfg) -> list[WhistleVector]:
    """S7: one time-frequency vector per track, sampled at the S4 frame step (gaps already linearly filled in S6)."""
    if cfg["decisions"]["s7"]["gap_fill"] != "linear":
        raise NotImplementedError
    out = []
    for t in tracks:
        assert np.all(np.diff(t.frames) == 1), "track frames must be consecutive"
        out.append(WhistleVector(int(t.id), int(t.frames[0]), np.asarray(t.freq_hz, float).copy(), np.asarray(t.conf, float).copy(),
                                 float(t.observed.mean())))
    return out


def vectors_to_npz(vectors: list[WhistleVector]) -> dict:
    lens = np.array([v.n_frames for v in vectors], dtype=int)
    return {
        "track_ids": np.array([v.track_id for v in vectors], dtype=int),
        "start_frames": np.array([v.start_frame for v in vectors], dtype=int),
        "offsets": np.concatenate([[0], np.cumsum(lens)]).astype(int),
        "freq_hz": np.concatenate([v.freq_hz for v in vectors]) if vectors else np.zeros(0),
        "conf": np.concatenate([v.conf for v in vectors]) if vectors else np.zeros(0),
        "observed_fraction": np.array([v.observed_fraction for v in vectors]),
    }


# ----------------------------------------------------------------------------- S8
@dataclass
class NFMResult:
    track_id: int
    start_frame: int
    n_frames: int
    center_hz: float
    delta_hz: np.ndarray       # centre-removed contour (Hz), one value per column used
    col0: int                  # first image column occupied
    baseband: np.ndarray       # complex64, constant amplitude (except edge ramps)


def nfm_grid(cfg: Cfg) -> dict:
    p = cfg["paper"]
    bin_hz = p["freq_resolution_hz"]
    fs_b = p["image_height"] * bin_hz                       # 12 kHz complex baseband
    hop_b = int(round(fs_b * p["hop"] / p["fs_target_hz"]))     # 64
    win_b = int(round(fs_b * p["win_len"] / p["fs_target_hz"]))  # 128
    g = {"fs_b": fs_b, "hop_b": hop_b, "win_b": win_b, "nfft_b": p["image_height"],
         "n_samples": (p["image_width"] - 1) * hop_b + win_b, "bin_hz": bin_hz}
    assert g["win_b"] == g["nfft_b"] and abs(g["fs_b"] / g["nfft_b"] - bin_hz) < 1e-9, "baseband grid inconsistent with the paper grid"
    return g


def s8_baseband(vectors: list[WhistleVector], cfg: Cfg) -> tuple[list[NFMResult], list[dict]]:
    p, d = cfg["paper"], cfg["decisions"]["s8"]
    if d["signal"] != "complex_baseband" or d["center_freq_definition"] != "midrange" or d["time_centering"] != "center":
        raise NotImplementedError("only the configured DECISION variants are implemented")
    g = nfm_grid(cfg)
    W, bin_hz = int(p["image_width"]), g["bin_hz"]
    lo_lim, hi_lim = -(p["image_height"] // 2) * bin_hz, (p["image_height"] // 2 - 1) * bin_hz   # -6000 .. +5906.25 Hz
    ramp = int(d["edge_ramp_samples"])
    out: list[NFMResult] = []
    rejected: list[dict] = []
    for v in vectors:
        L = v.n_frames
        f = v.freq_hz
        if L > W:
            if d["long_track_policy"] == "reject":
                rejected.append({"track_id": v.track_id, "reason": "too_long", "n_frames": L, "start_frame": v.start_frame})
                continue
            raise NotImplementedError("long_track_policy other than 'reject'")
        fc = 0.5 * (float(f.min()) + float(f.max()))
        delta = f - fc
        if delta.min() < lo_lim or delta.max() > hi_lim:
            rejected.append({"track_id": v.track_id, "reason": "too_wide", "bandwidth_hz": float(f.max() - f.min()), "start_frame": v.start_frame})
            continue
        col0 = (W - L) // 2
        n = g["n_samples"]
        centers = (col0 + np.arange(L)) * g["hop_b"] + g["win_b"] // 2        # sample index of each frame centre
        s0, s1 = centers[0] - g["hop_b"] // 2, centers[-1] + g["hop_b"] // 2  # active span: one hop around first/last centre
        k = np.arange(n)
        inst = np.interp(k, centers, delta)                                  # Hz, held constant outside the span
        amp = np.zeros(n)
        amp[max(s0, 0):min(s1, n)] = 1.0
        if ramp > 1:
            r = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
            a0 = max(s0, 0)
            amp[a0:a0 + ramp] *= r[:max(0, min(ramp, n - a0))]
            a1 = min(s1, n)
            amp[a1 - ramp:a1] *= r[::-1][:ramp][-(min(ramp, a1)):] if a1 >= ramp else 1.0
        phase = 2 * np.pi * np.cumsum(inst) / g["fs_b"]
        bb = (amp * np.exp(1j * phase)).astype(np.complex64)
        out.append(NFMResult(v.track_id, v.start_frame, L, fc, delta.copy(), col0, bb))
    return out, rejected


# ----------------------------------------------------------------------------- S9
def s9_image(nfm: NFMResult, cfg: Cfg) -> np.ndarray:
    """Spectrogram (|STFT|, linear magnitude) of the baseband signal -> (128, 128) float32 in [0, 1].
    Row 0 = highest frequency (image displays upright); 0 Hz is on row image_height/2 - 1; column j = S4-like frame j."""
    p, d = cfg["paper"], cfg["decisions"]
    g = nfm_grid(cfg)
    w = sps.get_window("hamming", g["win_b"], fftbins=bool(d["s4"]["window_periodic"]))
    x = nfm.baseband.astype(np.complex128)
    nfr = (len(x) - g["win_b"]) // g["hop_b"] + 1
    assert nfr == p["image_width"], f"{nfr} frames != image_width {p['image_width']}"
    fr = np.lib.stride_tricks.sliding_window_view(x, g["win_b"])[::g["hop_b"]][:nfr]
    mag = np.abs(np.fft.fftshift(np.fft.fft(fr * w, n=g["nfft_b"], axis=1), axes=1))        # (frames, bins asc)
    img = mag.T[::-1]                                                                      # (rows high->low, frames)
    m = img.max()
    img = img / m if m > 0 else img
    out = np.clip(img, p["image_min"], p["image_max"]).astype(np.float32)
    assert out.shape == (p["image_height"], p["image_width"])
    return out


def s9_images(nfms: list[NFMResult], cfg: Cfg) -> np.ndarray:
    p = cfg["paper"]
    if not nfms:
        return np.zeros((0, p["image_height"], p["image_width"]), np.float32)
    return np.stack([s9_image(n, cfg) for n in nfms])


def row_to_freq_hz(row: np.ndarray, cfg: Cfg) -> np.ndarray:
    """Baseband frequency (Hz) of image rows (row 0 = highest)."""
    h = cfg["paper"]["image_height"]
    return ((h - 1 - np.asarray(row)) - h // 2) * cfg["paper"]["freq_resolution_hz"]


def recover_contour(img: np.ndarray, cfg: Cfg, thr: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Inverse step used by the paper (threshold 0.05 on the image -> contour): per column, intensity-weighted
    centroid frequency (Hz) of pixels above ``thr``. Returns (column indices, baseband Hz)."""
    cols, freqs = [], []
    rows = np.arange(img.shape[0])
    for c in range(img.shape[1]):
        m = img[:, c] > thr
        if m.any():
            wts = img[m, c]
            cols.append(c); freqs.append(float((row_to_freq_hz(rows[m], cfg) * wts).sum() / wts.sum()))
    return np.array(cols), np.array(freqs)


def image_meta_rows(stem: str, nfms: list[NFMResult], vectors: list[WhistleVector], hop_s: float) -> list[dict]:
    by_id = {v.track_id: v for v in vectors}
    rows = []
    for i, n in enumerate(nfms):
        v = by_id[n.track_id]
        rows.append({"file": stem, "image_index": i, "track_id": n.track_id, "start_frame": n.start_frame,
                     "start_time_s": round(n.start_frame * hop_s, 4), "n_frames": n.n_frames, "col0": n.col0,
                     "f_center_hz": round(n.center_hz, 2), "f_min_hz": round(v.f_min, 2), "f_max_hz": round(v.f_max, 2),
                     "bandwidth_hz": round(v.f_max - v.f_min, 2), "mean_conf": round(v.mean_conf, 4),
                     "observed_fraction": round(v.observed_fraction, 4)})
    return rows
