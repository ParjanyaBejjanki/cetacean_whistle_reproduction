"""Validation gates G1-G6 (specification §4). Criteria are fixed here BEFORE any result exists.
Each gate writes models/vaegmm/_gates/<gate>_<UTC time>.json with ``passed`` and the measured values.

G1 data      read-only integrity scan of all 180,154 images                         (no training)
G2 numerics  GPU-vs-CPU forward parity, finite losses/gradients on real batches     (no training: one step on a copy)
G3 smoke     2 epochs on 2,048 images; epoch-boundary resume is bit-identical      (TRAINING)
G4 overfit   full objective on 256 images reaches SSE <= 0.3 x all-zero SSE        (TRAINING)
G5 timing    20 warm-up + 200 measured iterations on the full data                 (TRAINING)
G6 features  whistle features of real images agree with their true contours        (no training)
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..detector.ridgenet import resolve_device, set_deterministic
from ..storage import software_versions
from .config import resolve
from .data import Prefetcher, SubsetImages, WhistleImages, epoch_plan
from .features import contour_features
from .losses import critic_loss, vae_loss
from .models import VAE, Critic
from .train import Trainer, run_spec, train_vae

CRITERIA = {
    "G1": "all images finite, within [0, 1], per-image max == 1; counts equal dataset_info.json",
    "G2": "max |GPU - CPU| <= 1e-5 for x_hat, mu, logvar and critic output on 16 real images; all losses and gradients finite",
    "G3": ("2 epochs x 2,048 images: finite; (a) a trainer restored from the epoch-1 checkpoint equals that checkpoint "
           "bit-identically; (b) max |resumed - uninterrupted| <= 10 x max |uninterrupted - repeated uninterrupted| "
           "(bit-identical when the repeat is). Revised 2026-10-22: GPU training with the critic is not bit-reproducible "
           "even without resume (spec §4)"),
    "G4": "256 images, <= 3,000 VAE steps: SSE(z = mu, unrotated) <= 0.3 x SSE of the all-zero image; finite",
    "G5": "recorded only: s/iteration, data-wait fraction, peak GPU memory, projected 300-epoch hours",
    "G6": ">= 95 % of 4,000 real images have |L - n_frames| <= 2 and >= 95 % have |df - bandwidth/93.75 Hz| <= 1.5 bins",
}


def _gates_dir(cfg) -> Path:
    d = resolve(cfg["paths"]["runs_dir"]) / "_gates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record(cfg, gate: str, passed: bool, values: dict) -> dict:
    rec = {"gate": gate, "criterion": CRITERIA[gate], "passed": bool(passed), "values": values,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config_effective_hash": cfg["_meta"]["effective_hash"],
           "software": software_versions()}
    path = _gates_dir(cfg) / f"{gate}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"gate": gate, "passed": rec["passed"], "file": str(path)}), flush=True)
    return rec


def _images(cfg) -> WhistleImages:
    return WhistleImages(resolve(cfg["paths"]["dataset_dir"]))


def gate_data(cfg) -> dict:
    im = _images(cfg)
    s = im.scan()
    ok = s["n_non_finite"] == 0 and s["n_out_of_range"] == 0 and s["n_max_not_1"] == 0 and s["n_images"] == im.info["totals"]["images"]
    return _record(cfg, "G1", ok, s)


def gate_numerics(cfg) -> dict:
    set_deterministic()
    dev = resolve_device(cfg["train"]["device"])
    if dev.type != "cuda":
        return _record(cfg, "G2", False, {"error": "no CUDA device"})
    im = _images(cfg)
    spec = run_spec(cfg, "gate_numerics")
    x = torch.from_numpy(im.gather(np.random.default_rng(1).choice(len(im), 16, replace=False)))[:, None]
    torch.manual_seed(spec["seed"])
    vae_c, cri_c = VAE(spec["latent_dim"]), Critic()
    vae_g, cri_g = copy.deepcopy(vae_c).to(dev), copy.deepcopy(cri_c).to(dev)
    eps = torch.randn(16, spec["latent_dim"])
    with torch.no_grad():
        xc, mc, lc = vae_c(x, eps)
        xg, mg, lg = vae_g(x.to(dev), eps.to(dev))
        dc, dg = cri_c(x), cri_g(x.to(dev))
    diffs = {"x_hat": float((xc - xg.cpu()).abs().max()), "mu": float((mc - mg.cpu()).abs().max()),
             "logvar": float((lc - lg.cpu()).abs().max()), "critic": float((dc - dg.cpu()).abs().max())}
    # one full step (critic + VAE) on real data, losses and gradients must be finite
    xr = x.to(dev)
    with torch.no_grad():
        xf, _, _ = vae_g(xr, eps.to(dev))
    ld, pd = critic_loss(cri_g, xr, xf, torch.rand(16, 1, 1, 1, device=dev), spec["lambda_gp"])
    ld.backward()
    g_crit = all(torch.isfinite(p.grad).all() for p in cri_g.parameters())
    lg_x, mu, lv = vae_g.forward_logits(xr, eps.to(dev))
    lt, parts = vae_loss(lg_x, xr, mu, lv, cri_g.logit(torch.sigmoid(lg_x)), spec["beta"], spec["lambda_adv"], True, spec["recon"])
    lt.backward()
    g_vae = all(torch.isfinite(p.grad).all() for p in vae_g.parameters())
    vals = {"max_abs_diff": diffs, "critic_loss": float(ld.detach()), "vae_loss": float(lt.detach()), **{k: float(v) for k, v in {**pd, **parts}.items()},
            "critic_grads_finite": bool(g_crit), "vae_grads_finite": bool(g_vae)}
    ok = max(diffs.values()) <= 1e-5 and g_crit and g_vae and np.isfinite([vals["critic_loss"], vals["vae_loss"]]).all()
    return _record(cfg, "G2", ok, vals)


def _state_equal(a: dict, b: dict) -> tuple[bool, float]:
    worst = 0.0
    same = True

    def walk(x, y):
        nonlocal worst, same
        if isinstance(x, torch.Tensor):
            if x.shape != y.shape or not torch.equal(x, y):
                same = False
                if x.shape == y.shape and x.is_floating_point():
                    worst = max(worst, float((x.double() - y.double()).abs().max()))
        elif isinstance(x, dict):
            for k in x:
                walk(x[k], y[k])
        elif isinstance(x, (list, tuple)):
            for u, v in zip(x, y):
                walk(u, v)

    walk(a, b)
    return same, worst


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def gate_smoke(cfg) -> dict:
    im = _images(cfg)
    sub = SubsetImages(im, np.sort(np.random.default_rng(3).choice(len(im), 2048, replace=False)))
    root = _gates_dir(cfg) / f"smoke_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    specs = {n: run_spec(cfg, n, epochs=2) for n in ("uninterrupted", "repeat", "resumed")}
    t0 = time.time()
    train_vae(cfg, specs["uninterrupted"], images=sub, out_root=root, write_samples=False, quiet=True)
    train_vae(cfg, specs["repeat"], images=sub, out_root=root, write_samples=False, quiet=True)
    train_vae(cfg, specs["resumed"], images=sub, out_root=root, stop_after_epoch=1, write_samples=False, quiet=True)
    # (a) restoring the epoch-1 checkpoint reproduces it exactly
    ck1 = torch.load(root / "resumed" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    dev = resolve_device(cfg["train"]["device"])
    tr = Trainer(specs["resumed"], sub, dev)
    tr.load(root / "resumed" / "checkpoints" / "last.pt")
    restored = _to_cpu({k: v for k, v in tr.state().items() if k != "spec"})
    ck1.pop("spec")
    restore_exact, _ = _state_equal(ck1, restored)
    # (b) resumed vs uninterrupted, compared with the run-to-run envelope
    train_vae(cfg, specs["resumed"], images=sub, out_root=root, resume=True, write_samples=False, quiet=True)
    fin = {n: torch.load(root / n / "checkpoints" / "final.pt", map_location="cpu", weights_only=False) for n in specs}
    for v in fin.values():
        v.pop("spec")
    same_rep, env = _state_equal(fin["uninterrupted"], fin["repeat"])
    same_res, diff = _state_equal(fin["uninterrupted"], fin["resumed"])
    within = same_res if same_rep else diff <= 10.0 * env
    logs = [json.loads(l) for l in (root / "uninterrupted" / "train_log.jsonl").read_text().splitlines()]
    ep = [r for r in logs if r.get("epoch_summary")]
    vals = {"restore_exact": restore_exact, "run_to_run_identical": same_rep, "run_to_run_max_abs_diff": env,
            "resumed_identical": same_res, "resumed_max_abs_diff": diff, "within_envelope": bool(within),
            "epoch_summaries": ep, "wall_s": round(time.time() - t0, 1), "dir": str(root)}
    return _record(cfg, "G3", restore_exact and within and len(ep) == 2, vals)


@torch.no_grad()
def _sse_ratio(trainer: Trainer, x: torch.Tensor) -> float:
    mu, _ = trainer.vae.encoder(x)
    xr = trainer.vae.decoder(mu)
    return float((xr - x).pow(2).sum() / x.pow(2).sum())


def gate_overfit(cfg, max_steps: int = 3000, check_every: int = 250) -> dict:
    set_deterministic()
    dev = resolve_device(cfg["train"]["device"])
    im = _images(cfg)
    idx = np.sort(np.random.default_rng(4).choice(len(im), 256, replace=False))
    sub = SubsetImages(im, idx)
    x_all = torch.from_numpy(sub.gather(np.arange(256)))[:, None].to(dev)
    tr = Trainer(run_spec(cfg, "gate_overfit"), sub, dev)
    curve = [{"step": 0, "sse_ratio": _sse_ratio(tr, x_all)}]
    epoch, t0, passed = 0, time.time(), False
    while tr.global_step < max_steps:
        epoch += 1
        ep = tr.run_epoch(epoch, depth=4)
        if tr.global_step % check_every == 0 or tr.global_step >= max_steps:
            curve.append({"step": tr.global_step, "sse_ratio": _sse_ratio(tr, x_all),
                          **{k: v for k, v in ep.items() if k in ("rec", "kl", "adv", "loss_d", "d_real", "d_fake", "gp")}})
            print(json.dumps(curve[-1]), flush=True)
            if curve[-1]["sse_ratio"] <= 0.3:
                passed = True
                break
    return _record(cfg, "G4", passed, {"curve": curve, "steps": tr.global_step, "wall_s": round(time.time() - t0, 1)})


def gate_timing(cfg, warmup: int = 20, measured: int = 200) -> dict:
    set_deterministic()
    dev = resolve_device(cfg["train"]["device"])
    im = _images(cfg)
    vals = {}
    for name, use_critic in (("with_critic", True), ("no_critic", False)):
        tr = Trainer(run_spec(cfg, f"gate_timing_{name}", use_critic=use_critic), im, dev)
        vb, cb = epoch_plan(len(im), tr.spec["batch_size"], tr.spec["n_critic"], tr.spec["seed"], 1, use_critic)
        n = warmup + measured
        pf = Prefetcher(im, vb[:n], cb[:n], depth=int(cfg["train"]["prefetch"]), pin=(dev.type == "cuda"))
        g = tr.epoch_generator(1)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_start = wait0 = None
        for i, (cbs, xb) in enumerate(pf):
            if i == warmup:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t_start, wait0 = time.perf_counter(), pf.wait_s
            r = tr.step(cbs, xb, g)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        el = time.perf_counter() - t_start
        spi = el / measured
        steps_epoch = -(-len(im) // tr.spec["batch_size"])
        vals[name] = {"s_per_iteration": spi, "data_wait_fraction": (pf.wait_s - wait0) / el,
                      "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
                      "steps_per_epoch": steps_epoch, "projected_hours_300_epochs": spi * steps_epoch * 300 / 3600,
                      "last_losses": {k: float(v) for k, v in r.items()}}
        print(json.dumps({name: vals[name]}), flush=True)
        pf.close()
    return _record(cfg, "G5", True, vals)


def gate_features(cfg, n: int = 4000) -> dict:
    import pandas as pd

    im = _images(cfg)
    meta = pd.concat([pd.read_csv(d / "images.csv", usecols=["n_frames", "bandwidth_hz"]) for d in im.enc_dirs], ignore_index=True)
    if len(meta) != len(im):
        raise ValueError("images.csv rows differ from image count")
    idx = np.sort(np.random.default_rng(5).choice(len(im), n, replace=False))
    f, ok = contour_features(torch.from_numpy(im.gather(idx)), cfg["paper"]["contour_threshold"])
    f = f.numpy()
    dl = np.abs(f[:, 0] - meta["n_frames"].values[idx])
    dbw = np.abs(f[:, 3] - meta["bandwidth_hz"].values[idx] / cfg["decisions"]["features"]["hz_per_bin"])
    vals = {"n": n, "all_valid": bool(ok.all()), "frac_L_within_2": float((dl <= 2).mean()),
            "frac_df_within_1p5_bins": float((dbw <= 1.5).mean()), "L_abs_err_quantiles_50_95_100": np.quantile(dl, [.5, .95, 1]).tolist(),
            "df_abs_err_quantiles_50_95_100": np.quantile(dbw, [.5, .95, 1]).tolist()}
    passed = vals["all_valid"] and vals["frac_L_within_2"] >= 0.95 and vals["frac_df_within_1p5_bins"] >= 0.95
    return _record(cfg, "G6", passed, vals)


GATES = {"data": gate_data, "numerics": gate_numerics, "smoke": gate_smoke, "overfit": gate_overfit,
         "timing": gate_timing, "features": gate_features}
TRAINING_GATES = {"smoke", "overfit", "timing"}
