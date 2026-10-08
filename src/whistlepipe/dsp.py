"""S2 resampling, S3 high-pass filtering, S4 STFT.

Paper-defined values come from cfg['paper'] (locked). Unspecified details (resampler design, filter
design, window periodicity, output quantity, padding) come from cfg['decisions'] and are IMPLEMENTATION
DECISIONS, not paper evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .audio import AudioSignal
from .config import Cfg


@dataclass
class Spectrogram:
    """S4 output: log-magnitude spectrogram on the paper's grid."""

    db: np.ndarray        # (T, n_bins) float32, 20*log10(|STFT| + eps)
    fs: float             # sampling rate of the analysed signal (Hz)
    win_len: int
    hop: int
    nfft: int
    name: str = ""

    @property
    def n_frames(self) -> int:
        return self.db.shape[0]

    @property
    def n_bins(self) -> int:
        return self.db.shape[1]

    @property
    def bin_hz(self) -> float:
        return self.fs / self.nfft

    @property
    def hop_s(self) -> float:
        return self.hop / self.fs

    def freqs_hz(self) -> np.ndarray:
        return np.arange(self.n_bins) * self.bin_hz

    def frame_center_times_s(self) -> np.ndarray:
        return (np.arange(self.n_frames) * self.hop + self.win_len / 2) / self.fs


# ----------------------------------------------------------------------------- S2
def resample_ratio(fs_in: int, fs_out: int) -> tuple[int, int]:
    g = math.gcd(int(fs_in), int(fs_out))
    return int(fs_out) // g, int(fs_in) // g


def s2_resample(sig: AudioSignal, cfg: Cfg) -> AudioSignal:
    """S2: 500 kHz -> 192 kHz (EXPLICIT). Polyphase FIR design is an IMPLEMENTATION DECISION."""
    fs_out = int(cfg["paper"]["fs_target_hz"])
    if int(sig.fs) == fs_out:
        return AudioSignal(sig.samples.astype(np.float64), sig.fs, sig.name, dict(sig.meta))
    up, down = resample_ratio(int(sig.fs), fs_out)
    d = cfg["decisions"]["s2"]
    if d["method"] != "resample_poly":
        raise NotImplementedError(d["method"])
    y = signal.resample_poly(sig.samples.astype(np.float64), up, down, window=("kaiser", float(d["kaiser_beta"])))
    expected = -(-len(sig.samples) * up // down)  # ceil
    if len(y) != expected:
        raise RuntimeError(f"resample length {len(y)} != expected {expected}")
    return AudioSignal(y, float(fs_out), sig.name, {**sig.meta, "resample_up": up, "resample_down": down})


# ----------------------------------------------------------------------------- S3
def highpass_sos(fs: float, cfg: Cfg) -> np.ndarray:
    d = cfg["decisions"]["s3"]
    if d["filter"] != "butterworth":
        raise NotImplementedError(d["filter"])
    return signal.butter(int(d["order"]), float(cfg["paper"]["hpf_cutoff_hz"]), "highpass", fs=fs, output="sos")


def s3_highpass(sig: AudioSignal, cfg: Cfg) -> AudioSignal:
    """S3: high-pass, passband above 2 kHz (EXPLICIT). Design/phase/order of application = DECISION."""
    sos = highpass_sos(sig.fs, cfg)
    x = signal.sosfiltfilt(sos, sig.samples) if cfg["decisions"]["s3"]["zero_phase"] else signal.sosfilt(sos, sig.samples)
    return AudioSignal(x, sig.fs, sig.name, dict(sig.meta))


# ----------------------------------------------------------------------------- S4
def stft_db(x: np.ndarray, win_len: int, hop: int, nfft: int, periodic: bool = True,
            eps: float = 1e-12, chunk_frames: int = 1024) -> np.ndarray:
    """Hamming STFT -> 20*log10(|X| + eps), shape (T, nfft//2+1) float32. No padding:
    T = (N - win_len)//hop + 1 (0 if N < win_len). Shared by S4 and the synthetic-data generator."""
    n = len(x)
    if n < win_len:
        return np.zeros((0, nfft // 2 + 1), np.float32)
    n_frames = (n - win_len) // hop + 1
    w = signal.get_window("hamming", win_len, fftbins=periodic)
    view = np.lib.stride_tricks.sliding_window_view(x, win_len)[::hop][:n_frames]
    out = np.empty((n_frames, nfft // 2 + 1), np.float32)
    for i in range(0, n_frames, chunk_frames):
        fr = np.asarray(view[i:i + chunk_frames], dtype=np.float64) * w
        mag = np.abs(np.fft.rfft(fr, n=nfft, axis=1))
        out[i:i + chunk_frames] = (20.0 * np.log10(mag + eps)).astype(np.float32)
    return out


def s4_stft(sig: AudioSignal, cfg: Cfg) -> Spectrogram:
    p, d = cfg["paper"], cfg["decisions"]["s4"]
    if p["window"] != "hamming":
        raise NotImplementedError(p["window"])
    if abs(sig.fs - p["fs_target_hz"]) > 1e-6:
        raise ValueError(f"S4 expects fs={p['fs_target_hz']}, got {sig.fs}")
    db = stft_db(sig.samples, int(p["win_len"]), int(p["hop"]), int(p["nfft"]),
                 periodic=bool(d["window_periodic"]), eps=float(d["eps"]))
    return Spectrogram(db=db, fs=sig.fs, win_len=int(p["win_len"]), hop=int(p["hop"]), nfft=int(p["nfft"]), name=sig.name)
