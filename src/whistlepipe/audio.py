"""S0/S1: raw file discovery, read-only loading, channel selection, identity checks.

The raw directory is opened strictly in read mode. Nothing in this module writes anywhere.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Cfg


@dataclass
class AudioSignal:
    """A single-channel signal at a known sampling rate (S1 / S2 / S3 outputs)."""

    samples: np.ndarray            # 1-D, float32 (S1) or float64 (S2/S3)
    fs: float
    name: str                      # raw file name (stem is used for run folders)
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.fs


def list_raw_files(raw_dir: Path) -> list[Path]:
    return sorted(p for p in Path(raw_dir).iterdir() if p.is_file() and p.suffix.lower() == ".flac")


def audio_info(path: Path) -> dict:
    info = sf.info(str(path))
    return {
        "samplerate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "format": info.format,
        "subtype": info.subtype,
        "duration_s": info.duration,
    }


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest_hashes(manifest_csv: Path) -> dict[str, str]:
    """filename -> sha256 from metadata/manifest_363.csv (read-only; used only to verify identity)."""
    if not Path(manifest_csv).exists():
        return {}
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        return {r["filename"]: r["sha256"] for r in csv.DictReader(f)}


def read_channel_int16(path: Path, channel: int, block: int = 5_000_000) -> tuple[np.ndarray, int, int]:
    """Read ONE channel of a (multi-channel) file as int16, in blocks. Returns (samples, fs, n_channels)."""
    parts = []
    with sf.SoundFile(str(path), mode="r") as f:   # read-only
        if not (0 <= channel < f.channels):
            raise ValueError(f"channel {channel} out of range for {f.channels}-channel file {path.name}")
        while True:
            b = f.read(block, dtype="int16", always_2d=True)
            if len(b) == 0:
                break
            parts.append(np.ascontiguousarray(b[:, channel]))
        fs, n_ch = f.samplerate, f.channels
    return np.concatenate(parts) if parts else np.zeros(0, np.int16), fs, n_ch


def s1_select_channel(path: Path, cfg: Cfg) -> AudioSignal:
    """S1: channel handling. UNSPECIFIED in the paper (spec A-01) -> IMPLEMENTATION DECISION in config:
    a single configurable channel, int16 counts scaled to float32 in [-1, 1)."""
    d = cfg["decisions"]["s1"]
    if d["channel_policy"] != "single":
        raise NotImplementedError(f"channel_policy={d['channel_policy']!r} is not implemented")
    x16, fs, n_ch = read_channel_int16(Path(path), int(d["channel_index"]))
    if fs != cfg["paper"]["fs_raw_hz"]:
        raise ValueError(f"{Path(path).name}: fs={fs} != paper fs_raw_hz={cfg['paper']['fs_raw_hz']}")
    if n_ch != cfg["paper"]["n_channels_raw"]:
        raise ValueError(f"{Path(path).name}: {n_ch} channels != paper {cfg['paper']['n_channels_raw']}")
    x = x16.astype(np.float32) / np.float32(d["int16_full_scale"])
    meta = {
        "channel_index": int(d["channel_index"]),
        "n_channels_in_file": n_ch,
        "int16_peak": int(np.abs(x16.astype(np.int32)).max()) if len(x16) else 0,
        "n_full_scale": int((np.abs(x16.astype(np.int32)) >= 32767).sum()),
    }
    return AudioSignal(samples=x, fs=float(fs), name=Path(path).name, meta=meta)


def select_validation_files(files: list[Path], k: int) -> list[Path]:
    """Deterministic, evenly spaced selection: index floor((i + 0.5) * N / k) = midpoint of segment i of k."""
    n = len(files)
    if not (1 <= k <= n):
        raise ValueError(f"k={k} invalid for {n} files")
    idx = [min(n - 1, int((i + 0.5) * n / k)) for i in range(k)]
    return [files[i] for i in idx]
