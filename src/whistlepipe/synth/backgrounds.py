"""Real-background banks for synthetic training data.

There is NO ground truth for the local files, so backgrounds are real recordings (S1..S3 audio at 192 kHz) that a
purely classical, label-free screen judges unlikely to contain real whistles: windows with persistent narrow-band
ridges are dropped. The screen is imperfect: residual real whistles may remain in the backgrounds (they would be
unlabeled, i.e. treated as negatives). This is recorded as a known limitation and is conservative for false-alarm
measurements.

Backgrounds are drawn only from raw files that belong to NO validation set (val1/val5/val10), and train/val/test
use disjoint files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from .. import audio, dsp
from ..config import Cfg, band_bins, project_path
from ..detector.conditioning import noise_floor_db


def bank_dir(cfg: Cfg) -> Path:
    d = project_path(cfg, "data_dir") / "synthetic" / "backgrounds"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reserved_validation_files(cfg: Cfg) -> set[str]:
    files = audio.list_raw_files(project_path(cfg, "raw_dir"))
    out: set[str] = set()
    for k in (cfg["validation"]["phase1"], cfg["validation"]["phase2"], cfg["validation"]["phase3"]):
        out |= {p.name for p in audio.select_validation_files(files, k)}
    return out


def pick_background_files(cfg: Cfg, dcfg: Cfg) -> dict[str, list[Path]]:
    files = [p for p in audio.list_raw_files(project_path(cfg, "raw_dir")) if p.name not in reserved_validation_files(cfg)]
    rng = np.random.default_rng([int(dcfg["synth"]["seed"]), 1])
    order = rng.permutation(len(files))
    n = dcfg["synth"]["backgrounds"]["n_files"]
    out, i = {}, 0
    for split in ("train", "val", "test"):
        out[split] = [files[j] for j in order[i:i + n[split]]]
        i += n[split]
    return out


def ridge_score(db_above_floor: np.ndarray, screen_db: float, persist: int, frac: float, guard: tuple[int, int]) -> float:
    """Fraction of frames containing a narrow-band persistent ridge (classical, label-free)."""
    m = (db_above_floor > screen_db).astype(np.float32)
    p = ndimage.uniform_filter1d(m, size=persist, axis=0, mode="nearest") >= frac
    wide = np.zeros_like(p)
    for g in range(guard[0], guard[1] + 1):
        wide |= np.roll(p, g, axis=1) | np.roll(p, -g, axis=1)
    return float((p & ~wide).any(axis=1).mean())


def extract_windows(path: Path, cfg: Cfg, dcfg: Cfg, split: str, n_keep: int) -> dict:
    sy = dcfg["synth"]
    n_win = int(round(sy["window_s"] * cfg["paper"]["fs_target_hz"]))
    sig = dsp.s3_highpass(dsp.s2_resample(audio.s1_select_channel(path, cfg), cfg), cfg)
    x = sig.samples
    k_lo, k_hi = band_bins(cfg)
    cond = cfg["decisions"]["s5"]["conditioning"]
    bk = sy["backgrounds"]
    stride = int(cfg["paper"]["fs_target_hz"])        # candidate starts every 1 s
    starts = np.arange(0, len(x) - n_win, stride)
    scores = []
    for s in starts:
        db = dsp.stft_db(x[s:s + n_win], cfg["paper"]["win_len"], cfg["paper"]["hop"], cfg["paper"]["nfft"])[:, k_lo:k_hi + 1]
        above = db - noise_floor_db(db, cond["block_frames"], cond["median_blocks"])
        scores.append(ridge_score(above, bk["screen_db"], bk["screen_persist"], bk["screen_fraction"], tuple(bk["screen_guard_bins"])))
    scores = np.array(scores)
    chosen: list[int] = []
    min_sep = n_win // 2                                  # at most 50 % overlap between kept windows
    for i in np.argsort(scores, kind="stable"):
        if scores[i] > bk["screen_max_score"]:          # absolute cap: busy files contribute fewer (or no) windows
            break
        if all(abs(int(starts[i]) - int(starts[j])) >= min_sep for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= n_keep:
            break
    sel = np.array(sorted(chosen), dtype=int)
    return {"windows": np.stack([x[starts[i]:starts[i] + n_win] for i in sel]).astype(np.float16) if len(sel) else np.zeros((0, n_win), np.float16),
            "start": starts[sel], "score": scores[sel], "score_all": scores}


def build_bank(cfg: Cfg, dcfg: Cfg, split: str, files: list[Path]) -> dict:
    n_keep = int(dcfg["synth"]["backgrounds"]["windows_per_file"][split])
    W, F, S, SC, all_scores = [], [], [], [], []
    for fi, p in enumerate(files):
        r = extract_windows(p, cfg, dcfg, split, n_keep)
        W.append(r["windows"]); F += [fi] * len(r["start"]); S.append(r["start"]); SC.append(r["score"]); all_scores.append(r["score_all"])
        km = r["score"].mean() if len(r["score"]) else float("nan")
        print(f"  [{split}] {p.name}: kept {len(r['start'])} windows, kept-score mean={km:.4f} (all-candidates median={np.median(r['score_all']):.4f})", flush=True)
    d = bank_dir(cfg)
    np.save(d / f"{split}_windows.npy", np.concatenate(W))          # float16, memory-mappable
    np.savez(d / f"{split}.npz", file_idx=np.array(F), start=np.concatenate(S), score=np.concatenate(SC))
    meta = {"split": split, "files": [p.name for p in files], "n_windows": int(sum(len(w) for w in W)),
            "window_samples": int(W[0].shape[1]), "fs_hz": cfg["paper"]["fs_target_hz"], "dtype": "float16",
            "seed": int(dcfg["synth"]["seed"]), "screen": dict(dcfg["synth"]["backgrounds"]),
            "kept_score_mean": float(np.concatenate(SC).mean()), "all_score_median": float(np.median(np.concatenate(all_scores))),
            "note": "label-free classical screen; residual real whistles may remain (unlabeled)."}
    (d / f"{split}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def build_all_banks(cfg: Cfg, dcfg: Cfg) -> None:
    picks = pick_background_files(cfg, dcfg)
    allnames = [p.name for v in picks.values() for p in v]
    assert len(set(allnames)) == len(allnames), "background splits must use disjoint files"
    assert not (set(allnames) & reserved_validation_files(cfg)), "background files must not be validation files"
    for split, files in picks.items():
        print(f"building {split} bank from {len(files)} files", flush=True)
        build_bank(cfg, dcfg, split, files)


def load_bank(cfg: Cfg, split: str) -> dict:
    d = bank_dir(cfg)
    z = np.load(d / f"{split}.npz")
    out = {k: z[k] for k in z.files}
    out["windows"] = np.load(d / f"{split}_windows.npy", mmap_mode="r")
    return out
