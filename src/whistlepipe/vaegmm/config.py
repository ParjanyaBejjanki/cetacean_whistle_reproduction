"""VAE-GMM configuration: loading, paper-parameter locking, hashing.

The [paper] section of configs/vaegmm.toml is LOCKED against the values in resources/TRAINING_SPECIFICATION.md
(same mechanism as whistlepipe.config.PAPER_LOCK for preprocessing). Deviations live in [deviations] only.
"""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, Cfg, apply_overrides, canonical_hash

DEFAULT_VAEGMM_CFG = PROJECT_ROOT / "configs" / "vaegmm.toml"

_T3 = [(1, "L", 30, 40), (2, "L", 75, 80), (3, "L", 88, 108), (4, "f_min", 34, 55), (5, "f_min", 40, 53),
       (6, "f_max", 62, 66), (7, "f_max", 68, 71), (8, "df", 10, 20), (9, "df", 20, 30), (10, "f_start", 40, 55),
       (11, "f_start", 70, 80), (12, "f_end", 40, 55), (13, "f_end", 70, 80)]

# Values copied from the specification (§3, [EXPLICIT]).
TRAINING_PAPER_LOCK: dict[str, Any] = {
    "latent_dim": 64,
    "latent_dims_table1": [32, 64, 128, 256],
    "beta": 1e-4,
    "betas_table2": [1e-2, 1e-4, 1e-6],
    "lambda_adv": 1.0,
    "lambda_gp": 5.0,
    "n_critic": 3,
    "epochs": 300,
    "lr_vae": 1e-3,
    "lr_critic": 2e-4,
    "critic_adam_beta1": 0.0,
    "critic_adam_beta2": 0.9,
    "batch_per_gpu": 128,
    "n_gpus": 7,
    "rotations_deg": [0, 90, 180, 270],
    "n_components": 10000,
    "n_components_table2": [1000, 5000, 10000, 15000],
    "n_components_fig11": [1000, 3000, 5000, 7000, 10000, 12000, 15000],
    "em_max_iter": 1500,
    "em_tol": 1e-8,
    "contour_threshold": 0.05,
    "fid1_threshold": 0.1,
    "fid_input_size": 224,
    "fid_feature_dim": 2048,
    "component_samples": 500,
    "ldm_timesteps": [250, 500, 1000],
    "ldm_batch": 128,
    "ldm_epochs": 300,
    "ldm_lr": 5e-4,
    "ldm_beta_min": 1e-4,
    "ldm_beta_max": 0.02,
    "table3_conditions": [{"no": n, "feature": f, "lo": lo, "hi": hi} for n, f, lo, hi in _T3],
}


def validate_training_paper_lock(cfg: dict) -> None:
    paper = cfg.get("paper", {})
    for k, v in TRAINING_PAPER_LOCK.items():
        if k not in paper:
            raise ValueError(f"[paper].{k} missing from configs/vaegmm.toml")
        if paper[k] != v:
            raise ValueError(f"[paper].{k} = {paper[k]!r} differs from the specification value {v!r}. "
                             "Paper-defined parameters must not be changed; put deviations in [deviations].")
    extra = set(paper) - set(TRAINING_PAPER_LOCK)
    if extra:
        raise ValueError(f"unknown [paper] keys (would be an undeclared paper claim): {sorted(extra)}")
    dev = cfg.get("deviations", {})
    if dev.get("lr_scaling_rule") == "linear":
        ratio = dev["batch_size"] / (paper["batch_per_gpu"] * paper["n_gpus"])
        for k in ("lr_vae", "lr_critic"):
            if abs(dev[k] - paper[k] * ratio) > 1e-12:
                raise ValueError(f"[deviations].{k} = {dev[k]} is not the linear-scaled value {paper[k] * ratio}")


def load_vaegmm_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Cfg:
    path = Path(path) if path else DEFAULT_VAEGMM_CFG
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    apply_overrides(raw, overrides)
    validate_training_paper_lock(raw)
    raw["_meta"] = {
        "config_path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "overrides": list(overrides or []),
        "effective_hash": canonical_hash({k: v for k, v in raw.items() if k not in ("_meta", "train", "paths")}),
    }
    return Cfg(raw)


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
