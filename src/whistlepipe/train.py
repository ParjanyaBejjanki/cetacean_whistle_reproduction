"""Train the S5 detector on SYNTHETIC data only (real backgrounds + synthetic whistles with known labels).

Nothing here uses detector predictions as labels. Reproducible: seeds, config hashes and the checkpoint SHA-256
are written next to the checkpoint (models/detector/<version>/).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Cfg, band_bins, canonical_hash, project_path
from .detector.ridgenet import RidgeNet, resolve_device, set_deterministic, sha256_file
from .storage import software_versions
from .synth.dataset import SynthTrainDataset, build_eval_set


def loss_fn(logits, y, w, pos_weight: float, dice_weight: float):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    wt = w * (1.0 + (pos_weight - 1.0) * (y > 0.1).float())
    l_bce = (bce * wt).sum() / (w.sum() + 1e-6)
    p = torch.sigmoid(logits)
    inter = (p * y * w).sum()
    dice = 1.0 - (2 * inter + 1.0) / ((p * w).sum() + (y * w).sum() + 1.0)
    return l_bce + dice_weight * dice, l_bce.detach(), dice.detach()


@torch.no_grad()
def validate(model, dev, ev: dict, tau: float = 0.5, bs: int = 32) -> dict:
    model.eval()
    tp = fp = fn = 0
    n = len(ev["x"])
    tot_loss = 0.0
    for i in range(0, n, bs):
        x = torch.from_numpy(ev["x"][i:i + bs].astype(np.float32))[:, None].to(dev)
        y = torch.from_numpy(ev["y"][i:i + bs].astype(np.float32) / 255.0)[:, None].to(dev)
        w = torch.from_numpy(ev["w"][i:i + bs].astype(np.float32))[:, None].to(dev)
        lg = model(x)
        tot_loss += float(loss_fn(lg, y, w, 6.0, 1.0)[0]) * len(x)
        pred = (torch.sigmoid(lg) >= tau)
        pos = y >= 0.3
        m = w > 0
        tp += int((pred & pos & m).sum()); fp += int((pred & ~pos & m).sum()); fn += int((~pred & pos & m).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    model.train()
    return {"val_loss": tot_loss / n, "node_precision": prec, "node_recall": rec,
            "node_f1": 2 * prec * rec / max(1e-9, prec + rec)}


def train_detector(cfg: Cfg, dcfg: Cfg, version: str = "ridgenet-v1") -> Path:
    tr, m = dcfg["train"], dcfg["model"]
    out = project_path(cfg, "models_dir") / "detector" / version
    if (out / "checkpoint.pt").exists():
        raise FileExistsError(f"{out / 'checkpoint.pt'} exists; trained checkpoints are never overwritten. Use --version.")
    out.mkdir(parents=True, exist_ok=True)
    seed = int(dcfg["synth"]["seed"])
    torch.manual_seed(seed); np.random.seed(seed)
    set_deterministic()
    dev = resolve_device(tr["device"])
    band = band_bins(cfg)
    model = RidgeNet(m["channels"], m["dilations"], m.get("block_type", "dense")).to(dev)
    print(f"RidgeNet[{model.block_type}]: {model.n_params()} parameters, {model.macs_per_node()} MACs/node, receptive radius (t,f) = {model.receptive_radius()}, device={dev}", flush=True)

    steps, bs = int(tr["steps"]), int(tr["batch_size"])
    ds = SynthTrainDataset(cfg, dcfg, "train", steps * bs)
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=int(tr["num_workers"]),
                                         persistent_workers=True, prefetch_factor=4, pin_memory=(dev.type == "cuda"))
    val = build_eval_set(cfg, dcfg, "val", int(tr["val_samples"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(tr["lr"]), total_steps=steps, pct_start=0.1)
    best, log = -1.0, []
    t0 = time.time()
    model.train()
    for step, (x, y, w) in enumerate(loader, 1):
        x, y, w = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), w.to(dev, non_blocking=True)
        loss, lb, ld = loss_fn(model(x), y, w, float(tr["pos_weight"]), float(tr["dice_weight"]))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0:
            rec = {"step": step, "loss": float(loss.detach()), "bce": float(lb), "dice": float(ld), "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
            log.append(rec); print(json.dumps(rec), flush=True)
        if step % int(tr["val_every"]) == 0 or step == steps:
            v = validate(model, dev, val)
            log.append({"step": step, **v}); print("VAL", json.dumps(v), flush=True)
            if v["node_f1"] > best:
                best = v["node_f1"]
                torch.save({"model_state": model.state_dict(), "arch": {"channels": m["channels"], "dilations": [list(d) for d in m["dilations"]], "block_type": m.get("block_type", "dense")},
                            "band": list(band), "conditioning": dict(cfg["decisions"]["s5"]["conditioning"]), "version": version,
                            "seed": seed, "metrics": {"step": step, **v}, "train_cfg": dcfg.to_dict()["train"]}, out / "checkpoint.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    info = {"version": version, "checkpoint_sha256": sha256_file(out / "checkpoint.pt"), "best_val_node_f1": best,
            "n_params": model.n_params(), "detector_config_hash": dcfg["_meta"]["effective_hash"],
            "pipeline_config_hash": cfg["_meta"]["effective_hash"], "seed": seed, "steps": steps, "batch_size": bs,
            "train_wall_s": round(time.time() - t0, 1), "software": software_versions(),
            "note": "trained on SYNTHETIC data only; node F1 here is a synthetic-validation number, not a real-data score."}
    (out / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out / "detector_config.json").write_text(json.dumps(dcfg.to_dict(), indent=2), encoding="utf-8")   # full effective config incl. overrides
    print("saved", out, flush=True)
    return out
