"""Torch dataset for on-the-fly synthetic training, and fixed (cached) synthetic validation/test sets."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from ..config import Cfg, project_path
from .backgrounds import load_bank
from .whistles import SynthSampler

SPLIT_ID = {"train": 0, "val": 1, "test": 2}


def make_sampler(cfg: Cfg, dcfg: Cfg, split: str) -> SynthSampler:
    masks = None
    if split == "train" and dcfg["synth"]["labels"].get("ignore_background_ridges", False):
        from .backgrounds import load_ridge_masks

        masks = load_ridge_masks(cfg, dcfg, split)
    return SynthSampler(cfg, dcfg, load_bank(cfg, split)["windows"], SPLIT_ID[split], masks)


def crop_start(seed: int, split: str, index: int, T: int, crop: int) -> int:
    rng = np.random.default_rng([seed, SPLIT_ID[split], 7, int(index)])
    return int(rng.integers(0, max(1, T - crop + 1)))


class SynthTrainDataset(torch.utils.data.Dataset):
    """Deterministic: item i is a pure function of (seed, split, i). The sampler (and the memory-mapped
    background bank) is created lazily inside each DataLoader worker."""

    def __init__(self, cfg: Cfg, dcfg: Cfg, split: str = "train", length: int = 96000):
        self.cfg, self.dcfg, self.split, self.length = cfg, dcfg, split, int(length)
        self.crop = int(dcfg["synth"]["crop_frames"])
        self._sampler: SynthSampler | None = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        if self._sampler is None:
            self._sampler = make_sampler(self.cfg, self.dcfg, self.split)
        s = self._sampler.sample(i)
        a = crop_start(self._sampler.seed, self.split, i, s["x"].shape[0], self.crop)
        sl = slice(a, a + self.crop)
        return (torch.from_numpy(s["x"][sl])[None], torch.from_numpy(s["y"][sl])[None], torch.from_numpy(s["w"][sl])[None])


def eval_set_path(cfg: Cfg, split: str) -> Path:
    d = project_path(cfg, "data_dir") / "synthetic"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"eval_{split}.pkl"


def build_eval_set(cfg: Cfg, dcfg: Cfg, split: str, n: int, force: bool = False) -> dict:
    """Fixed synthetic evaluation set (inputs, targets, weights, ground-truth contours). Cached on disk."""
    path = eval_set_path(cfg, split)
    if path.exists() and not force:
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d["n"] == n and d["seed"] == int(dcfg["synth"]["seed"]):
            return d
    sampler = make_sampler(cfg, dcfg, split)
    crop = int(dcfg["synth"]["crop_frames"])
    X, Y, W, GT, META = [], [], [], [], []
    base = 1_000_000                                  # eval indices are disjoint from training indices (0..steps*batch)
    for i in range(n):
        s = sampler.sample(base + i)
        a = crop_start(sampler.seed, split, base + i, s["x"].shape[0], crop)
        X.append(s["x"][a:a + crop].astype(np.float16)); Y.append(np.rint(s["y"][a:a + crop] * 255).astype(np.uint8))
        W.append(s["w"][a:a + crop].astype(np.uint8))
        gts = []
        for g in s["gt"]:
            m = (g["frames"] >= a) & (g["frames"] < a + crop)
            if m.any():
                gts.append({"wid": g["wid"], "h": g["h"], "frames": g["frames"][m] - a, "f_hz": g["f_hz"][m],
                            "snr_db": g["snr_db"][m], "visible": g["visible"][m]})
        GT.append(gts); META.append(s["meta"])
    d = {"n": n, "seed": int(dcfg["synth"]["seed"]), "split": split, "crop": crop,
         "x": np.stack(X), "y": np.stack(Y), "w": np.stack(W), "gt": GT, "meta": META}
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=4)
    return d


def build_surrogate_negative_set(cfg: Cfg, dcfg: Cfg, split: str, n: int, force: bool = False) -> dict:
    """Whistle-free samples whose background is phase-randomised (no real whistle can remain) + synthetic clicks/sweeps.
    Measures the detector's false-alarm behaviour on noise and clicks without label contamination."""
    path = eval_set_path(cfg, split).with_name(f"eval_{split}_surrogate_neg.pkl")
    if path.exists() and not force:
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d["n"] == n and d["seed"] == int(dcfg["synth"]["seed"]):
            return d
    sampler = make_sampler(cfg, dcfg, split)
    crop = int(dcfg["synth"]["crop_frames"])
    X = []
    for i in range(n):
        s = sampler.sample(2_000_000 + i, surrogate=True, no_whistles=True)
        a = crop_start(sampler.seed, split, 2_000_000 + i, s["x"].shape[0], crop)
        X.append(s["x"][a:a + crop].astype(np.float16))
    d = {"n": n, "seed": int(dcfg["synth"]["seed"]), "split": split, "crop": crop, "x": np.stack(X)}
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=4)
    return d
