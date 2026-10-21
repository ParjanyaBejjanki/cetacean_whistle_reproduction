"""Configuration loading, paper-parameter locking, hashing.

Paper-defined parameters ([paper] section of configs/pipeline.toml) are LOCKED against the values in
resources/PREPROCESSING_SPECIFICATION.md. Changing them requires editing PAPER_LOCK here deliberately,
which makes a silent change impossible.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CFG = PROJECT_ROOT / "configs" / "pipeline.toml"
DEFAULT_DETECTOR_CFG = PROJECT_ROOT / "configs" / "detector.toml"

# Values copied from the specification (sections 4 and 10.2). EXPLICIT or IMPLIED there.
PAPER_LOCK: dict[str, Any] = {
    "fs_raw_hz": 500000,
    "n_channels_raw": 6,
    "fs_target_hz": 192000,
    "hpf_cutoff_hz": 2000,
    "window": "hamming",
    "win_len": 2048,
    "overlap_percent": 50,
    "hop": 1024,
    "nfft": 2048,
    "freq_resolution_hz": 93.75,
    "image_height": 128,
    "image_width": 128,
    "image_channels": 1,
    "image_min": 0.0,
    "image_max": 1.0,
}


class Cfg(dict):
    """dict with attribute access (read-only convenience); nested dicts are wrapped lazily."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self))


def _deep_set(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Overrides like ``decisions.s6.peak_threshold=0.6`` (value parsed as a TOML literal)."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must be key=value, got {item!r}")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw
        _deep_set(cfg, key.strip(), value)
    return cfg


def canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def validate_paper_lock(cfg: dict) -> None:
    paper = cfg.get("paper", {})
    for k, v in PAPER_LOCK.items():
        if k not in paper:
            raise ValueError(f"[paper].{k} missing from config")
        if paper[k] != v:
            raise ValueError(
                f"[paper].{k} = {paper[k]!r} differs from the specification value {v!r}. "
                "Paper-defined parameters must not be changed (spec D-01..D-03)."
            )
    extra = set(paper) - set(PAPER_LOCK)
    if extra:
        raise ValueError(f"unknown [paper] keys (would be an undeclared paper claim): {sorted(extra)}")
    # derived consistency
    if abs(paper["fs_target_hz"] / paper["nfft"] - paper["freq_resolution_hz"]) > 1e-9:
        raise ValueError("fs_target_hz / nfft != freq_resolution_hz")
    if paper["hop"] != paper["win_len"] * (100 - paper["overlap_percent"]) // 100:
        raise ValueError("hop inconsistent with win_len and overlap_percent")


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_pipeline_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Cfg:
    path = Path(path) if path else DEFAULT_PIPELINE_CFG
    raw = load_toml(path)
    apply_overrides(raw, overrides)
    validate_paper_lock(raw)
    raw["_meta"] = {
        "config_path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "overrides": list(overrides or []),
        "effective_hash": canonical_hash({k: v for k, v in raw.items() if k != "_meta"}),
    }
    return Cfg(raw)


def load_detector_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Cfg:
    path = Path(path) if path else DEFAULT_DETECTOR_CFG
    raw = load_toml(path)
    apply_overrides(raw, overrides)
    raw["_meta"] = {
        "config_path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "overrides": list(overrides or []),
        "effective_hash": canonical_hash({k: v for k, v in raw.items() if k != "_meta"}),
    }
    return Cfg(raw)


def processing_hash(cfg: Cfg) -> str:
    """Hash of the settings that determine the processed data: [paper] + [decisions], without the S5 device
    (results are device-independent, tested). Paths, raw-folder lists, storage and dataset options are excluded,
    so adding an encounter folder or moving an output folder keeps an existing dataset compatible."""
    dec = json.loads(json.dumps(cfg["decisions"]))
    dec.get("s5", {}).pop("device", None)
    return canonical_hash({"paper": cfg["paper"], "decisions": dec})


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def project_path(cfg: Cfg, key: str) -> Path:
    """Resolve cfg.paths[key] relative to the project root."""
    return _resolve(cfg["paths"][key])


def raw_dirs(cfg: Cfg) -> list[Path]:
    """The raw-audio folders processed by the pipeline (one per encounter), in config order."""
    return [_resolve(p) for p in cfg["paths"]["raw_dirs"]]


def protected_raw_dirs(cfg: Cfg) -> list[Path]:
    """Every immutable raw folder (pipeline folders + the synthetic-background source): never written to."""
    out: list[Path] = []
    for p in raw_dirs(cfg) + [project_path(cfg, "synth_raw_dir")]:
        if p not in out:
            out.append(p)
    return out


def band_bins(cfg: Cfg) -> tuple[int, int]:
    """S4 bin indices [k_lo, k_hi] (inclusive) of the S5 analysis band."""
    import math

    df = cfg["paper"]["freq_resolution_hz"]
    k_lo = int(math.ceil(cfg["decisions"]["s5"]["band_lo_hz"] / df - 1e-9))
    k_hi = int(math.floor(cfg["decisions"]["s5"]["band_hi_hz"] / df + 1e-9))
    return k_lo, k_hi
