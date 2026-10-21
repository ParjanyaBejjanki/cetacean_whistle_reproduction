"""S0/S1: raw file discovery, read-only loading, channel selection, identity checks.

The raw directory is opened strictly in read mode. Nothing in this module writes anywhere.
"""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Cfg, project_path, raw_dirs


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


ENCOUNTER_DIR = re.compile(r"encounter_(\d+)_audio")


def encounter_of(path: Path) -> int | None:
    """Encounter id from the raw folder name (encounter_<id>_audio), None for other folders."""
    m = ENCOUNTER_DIR.fullmatch(Path(path).parent.name)
    return int(m.group(1)) if m else None


def files_by_encounter(cfg: Cfg) -> dict[int, list[Path]]:
    """{encounter id: sorted raw files} for every configured raw folder."""
    out: dict[int, list[Path]] = {}
    for d in raw_dirs(cfg):
        m = ENCOUNTER_DIR.fullmatch(d.name)
        if not m:
            raise ValueError(f"raw folder {d} is not named encounter_<id>_audio")
        out[int(m.group(1))] = list_raw_files(d)
    return dict(sorted(out.items()))


def list_all_raw_files(cfg: Cfg) -> list[Path]:
    """Every raw file of every configured raw folder, sorted by file name (= recording time).
    File names must be unique across folders (the stem identifies a file everywhere downstream)."""
    files = [p for d in raw_dirs(cfg) for p in list_raw_files(d)]
    seen: dict[str, Path] = {}
    for p in files:
        if p.name in seen:
            raise ValueError(f"raw file name {p.name} occurs in two folders: {seen[p.name].parent} and {p.parent}")
        seen[p.name] = p
    return sorted(files, key=lambda p: p.name)


def find_raw_file(cfg: Cfg, name: str) -> Path:
    """Raw file by name or stem, searched in every configured raw folder."""
    stem = Path(name).stem
    for d in raw_dirs(cfg):
        p = d / f"{stem}.flac"
        if p.is_file():
            return p
    raise FileNotFoundError(f"{stem}.flac is in none of the raw folders")


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


def load_manifest_rows(manifest_csv: Path) -> dict[str, dict]:
    """filename -> manifest row (read-only). Empty if the manifest does not exist."""
    if not Path(manifest_csv).exists():
        return {}
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


def load_manifest_hashes(manifest_csv: Path) -> dict[str, str]:
    """filename -> sha256 from the raw-file manifest (read-only; used only to verify identity)."""
    return {k: r["sha256"] for k, r in load_manifest_rows(manifest_csv).items()}


def missing_from_manifest(cfg: Cfg, files: list[Path]) -> list[str]:
    """Names of ``files`` that have no row in [paths].manifest_csv (all of them if the manifest does not exist)."""
    rows = load_manifest_rows(project_path(cfg, "manifest_csv"))
    return [p.name for p in files if p.name not in rows]


def read_channel_int16(path: Path, channel: int, block: int = 5_000_000) -> tuple[np.ndarray, int, int]:
    """Read ONE channel of a (multi-channel) file as int16, in blocks. Returns (samples, fs, n_channels).
    A FLAC header states the full length even when the file was cut off (e.g. an interrupted download), so the
    decoded length is checked against it: a truncated or corrupt file raises ValueError instead of being half-used."""
    parts = []
    with sf.SoundFile(str(path), mode="r") as f:   # read-only
        if not (0 <= channel < f.channels):
            raise ValueError(f"channel {channel} out of range for {f.channels}-channel file {path.name}")
        try:
            while True:
                b = f.read(block, dtype="int16", always_2d=True)
                if len(b) == 0:
                    break
                parts.append(np.ascontiguousarray(b[:, channel]))
        except sf.LibsndfileError as e:
            raise ValueError(f"{path.name}: truncated or corrupt file, decoding stopped after "
                             f"{sum(map(len, parts))} of {f.frames} samples ({e})") from e
        fs, n_ch, n_header = f.samplerate, f.channels, f.frames
    x = np.concatenate(parts) if parts else np.zeros(0, np.int16)
    if len(x) != n_header:
        raise ValueError(f"{path.name}: truncated or corrupt file, decoded {len(x)} samples but the header says {n_header}")
    return x, fs, n_ch


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
