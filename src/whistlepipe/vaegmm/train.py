"""VAE + WGAN-GP critic training (specification §3.2-3.4).

Per VAE iteration ([49] Algorithm 1 order): ``n_critic`` critic steps, each on a fresh rotated real batch and the VAE
reconstruction of that batch, then one VAE step on the next batch of the epoch permutation.
Runs live in models/vaegmm/<run_id>/ and are never overwritten (resume continues an interrupted run).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (must precede CUDA init)

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..config import PROJECT_ROOT, canonical_hash
from ..detector.ridgenet import resolve_device, set_deterministic, sha256_file
from ..storage import software_versions
from .config import resolve
from .data import Prefetcher, WhistleImages, epoch_plan, epoch_seed, random_rot90
from .losses import critic_loss, vae_loss
from .models import VAE, Critic, Decoder, n_params, table6_flops


def run_spec(cfg, run_id: str, latent_dim: int | None = None, beta: float | None = None, use_critic: bool = True,
             epochs: int | None = None) -> dict:
    p, dv = cfg["paper"], cfg["deviations"]
    return {"run_id": run_id,
            "latent_dim": int(latent_dim if latent_dim is not None else p["latent_dim"]),
            "beta": float(beta if beta is not None else p["beta"]),
            "use_critic": bool(use_critic),
            "lambda_adv": float(p["lambda_adv"]), "lambda_gp": float(p["lambda_gp"]), "n_critic": int(p["n_critic"]),
            "epochs": int(epochs if epochs is not None else p["epochs"]),
            "batch_size": int(dv["batch_size"]), "lr_vae": float(dv["lr_vae"]), "lr_critic": float(dv["lr_critic"]),
            "critic_betas": [float(p["critic_adam_beta1"]), float(p["critic_adam_beta2"])],
            "vae_betas": [float(b) for b in cfg["decisions"]["optim"]["vae_adam_betas"]],
            "adam_eps": float(cfg["decisions"]["optim"]["adam_eps"]),
            "recon": str(cfg["decisions"]["loss"]["recon"]),
            "tf32": bool(cfg["train"].get("tf32", False)),
            "seed": int(cfg["decisions"]["train"]["seed"])}


def source_hashes() -> dict:
    files = sorted((PROJECT_ROOT / "src" / "whistlepipe" / "vaegmm").glob("*.py")) + [PROJECT_ROOT / "configs" / "vaegmm.toml"]
    return {str(f.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def apply_runtime_flags(cfg, spec: dict) -> dict:
    """After set_deterministic(): optional TF32 (spec, changes numerics slightly) and cuDNN autotuning (cfg[train],
    same maths, different kernel choice). Training with the critic is not bit-reproducible on GPU anyway (gate G3)."""
    if cfg["train"].get("cudnn_benchmark", False):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
    if spec.get("tf32", False):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    return {"tf32": bool(spec.get("tf32", False)), "cudnn_benchmark": bool(cfg["train"].get("cudnn_benchmark", False)),
            "micro_batch": int(cfg["train"].get("micro_batch", 0) or 0), "prefetch": int(cfg["train"]["prefetch"])}


class Trainer:
    def __init__(self, spec: dict, images, device: torch.device, micro_batch: int = 0):
        """``micro_batch`` (0 = whole batch) splits every batch into chunks whose gradients are accumulated with weight
        chunk/batch; all losses are batch means of per-image terms and no layer uses batch statistics, so the update
        equals the whole-batch update (up to float summation order). Memory only; not part of the run spec."""
        self.spec, self.images, self.dev = spec, images, device
        self.micro = int(micro_batch or 0)
        torch.manual_seed(spec["seed"])
        self.vae = VAE(spec["latent_dim"]).to(device)
        self.critic = Critic().to(device) if spec["use_critic"] else None
        self.opt_v = torch.optim.Adam(self.vae.parameters(), lr=spec["lr_vae"], betas=tuple(spec["vae_betas"]), eps=spec["adam_eps"])
        self.opt_c = (torch.optim.Adam(self.critic.parameters(), lr=spec["lr_critic"], betas=tuple(spec["critic_betas"]),
                                       eps=spec["adam_eps"]) if self.critic is not None else None)
        self.epoch = 0          # completed epochs
        self.global_step = 0

    # ------------------------------------------------------------------ one iteration
    def _chunks(self, b: int) -> list[slice]:
        m = self.micro if 0 < self.micro < b else b
        return [slice(i, min(i + m, b)) for i in range(0, b, m)]

    def step(self, critic_batches, vae_batch, g: torch.Generator, measure: bool = False) -> dict:
        s, dev, L = self.spec, self.dev, self.spec["latent_dim"]
        out = {}
        if self.critic is not None:
            for xr in critic_batches:
                xr = xr.to(dev, non_blocking=True)
                b = xr.shape[0]
                # whole-batch random draws first (same order as without micro-batching)
                xr = random_rot90(xr, torch.randint(0, 4, (b,), generator=g, device=dev))
                z_eps = torch.randn(b, L, generator=g, device=dev)
                eps = torch.rand(b, 1, 1, 1, generator=g, device=dev)
                self.opt_c.zero_grad(set_to_none=True)
                acc = {}
                for sl in self._chunks(b):
                    w = (sl.stop - sl.start) / b
                    with torch.no_grad():
                        x_fake = torch.sigmoid(self.vae.forward_logits(xr[sl], z_eps[sl])[0])
                    loss_d, pd = critic_loss(self.critic, xr[sl], x_fake, eps[sl], s["lambda_gp"])
                    (loss_d * w).backward()
                    for k, v in {"loss_d": loss_d.detach(), **pd}.items():
                        acc[k] = acc.get(k, 0.0) + w * v
                self.opt_c.step()
            out.update(acc)
        x = vae_batch.to(dev, non_blocking=True)
        b = x.shape[0]
        x = random_rot90(x, torch.randint(0, 4, (b,), generator=g, device=dev))
        z_eps = torch.randn(b, L, generator=g, device=dev)
        if self.critic is not None:
            self.critic.requires_grad_(False)
        self.opt_v.zero_grad(set_to_none=True)
        acc = {}
        for sl in self._chunks(b):
            w = (sl.stop - sl.start) / b
            logits, mu, logvar = self.vae.forward_logits(x[sl], z_eps[sl])
            logit = self.critic.logit(torch.sigmoid(logits)) if self.critic is not None else None
            loss, parts = vae_loss(logits, x[sl], mu, logvar, logit, s["beta"], s["lambda_adv"], self.critic is not None,
                                   s.get("recon", "bce_sum"))
            (loss * w).backward()
            for k, v in {"loss_vae": loss.detach(), **parts}.items():
                acc[k] = acc.get(k, 0.0) + w * v
        if measure:
            out["gradnorm_vae"] = torch.nn.utils.clip_grad_norm_(self.vae.parameters(), float("inf")).detach()
        self.opt_v.step()
        if self.critic is not None:
            self.critic.requires_grad_(True)
        self.global_step += 1
        out.update(acc)
        return out

    def epoch_generator(self, epoch: int) -> torch.Generator:
        g = torch.Generator(device=self.dev)
        g.manual_seed(epoch_seed(self.spec["seed"], epoch, 2))
        return g

    def run_epoch(self, epoch: int, depth: int = 6, log_every: int = 0, log_fn=None) -> dict:
        """Train epoch ``epoch`` (1-based). Returns the epoch-mean of every logged quantity."""
        s = self.spec
        vae_b, cri_b = epoch_plan(len(self.images), s["batch_size"], s["n_critic"], s["seed"], epoch, self.critic is not None)
        g = self.epoch_generator(epoch)
        pf = Prefetcher(self.images, vae_b, cri_b, depth=depth, pin=(self.dev.type == "cuda"))
        sums, nsum, isum, n_int = {}, 0, {}, 0
        t0 = time.time()
        try:
            for i, (cbs, xb) in enumerate(pf, 1):
                measure = bool(log_every) and i % log_every == 0
                r = self.step(cbs, xb, g, measure=measure)
                for k, v in r.items():
                    if k == "gradnorm_vae":
                        continue
                    sums[k] = sums.get(k, 0.0) + v
                    isum[k] = isum.get(k, 0.0) + v
                nsum += 1
                n_int += 1
                if measure:
                    rec = {"epoch": epoch, "step": self.global_step, **{k: float(v) / n_int for k, v in isum.items()},
                           "gradnorm_vae": float(r["gradnorm_vae"]), "elapsed_s": round(time.time() - t0, 1),
                           "loader_wait_s": round(pf.wait_s, 1)}
                    check_finite(rec)
                    if log_fn:
                        log_fn(rec)
                    isum, n_int = {}, 0
        finally:
            pf.close()
        ep = {k: float(v) / nsum for k, v in sums.items()}
        ep.update({"epoch": epoch, "steps": nsum, "epoch_s": round(time.time() - t0, 1), "loader_wait_s": round(pf.wait_s, 1)})
        check_finite(ep)
        self.epoch = epoch
        return ep

    # ------------------------------------------------------------------ checkpoints
    def state(self) -> dict:
        return {"vae": self.vae.state_dict(), "critic": self.critic.state_dict() if self.critic else None,
                "opt_v": self.opt_v.state_dict(), "opt_c": self.opt_c.state_dict() if self.opt_c else None,
                "epoch": self.epoch, "global_step": self.global_step, "spec": self.spec}

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        # load to CPU: load_state_dict then places model/optimizer tensors exactly as in an uninterrupted run
        # (Adam keeps its 'step' counters on the CPU; mapping them to CUDA would make resumed runs differ structurally)
        st = torch.load(path, map_location="cpu", weights_only=False)
        if st["spec"] != self.spec:
            raise ValueError(f"checkpoint {path} was trained with a different run spec:\n{st['spec']}\nvs\n{self.spec}")
        self.vae.load_state_dict(st["vae"])
        self.opt_v.load_state_dict(st["opt_v"])
        if self.critic is not None:
            self.critic.load_state_dict(st["critic"])
            self.opt_c.load_state_dict(st["opt_c"])
        self.epoch, self.global_step = int(st["epoch"]), int(st["global_step"])


def check_finite(rec: dict) -> None:
    bad = [k for k, v in rec.items() if isinstance(v, float) and not np.isfinite(v)]
    if bad:
        raise FloatingPointError(f"non-finite training quantities {bad}: {rec}")


@torch.no_grad()
def save_sample_grid(trainer: Trainer, fixed_x: torch.Tensor, path: Path) -> None:
    """Top: real (unrotated); middle: reconstruction (z = mu); bottom: decoder(z ~ N(0, I)) (fixed seed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trainer.vae.eval()
    x = fixed_x.to(trainer.dev)
    mu, _ = trainer.vae.encoder(x)
    rec = trainer.vae.decoder(mu)
    g = torch.Generator(device=trainer.dev)
    g.manual_seed(12345)
    prior = trainer.vae.decoder(torch.randn(x.shape[0], trainer.spec["latent_dim"], generator=g, device=trainer.dev))
    trainer.vae.train()
    rows = [x.cpu().numpy()[:, 0], rec.cpu().numpy()[:, 0], prior.cpu().numpy()[:, 0]]
    n = x.shape[0]
    fig, axs = plt.subplots(3, n, figsize=(n * 1.1, 3.5))
    for r, imgs in enumerate(rows):
        for c in range(n):
            axs[r, c].imshow(imgs[c], cmap="gray", vmin=0, vmax=1)
            axs[r, c].axis("off")
    fig.suptitle(f"epoch {trainer.epoch}: real / reconstruction (z=mu) / N(0,I) sample", fontsize=8)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def train_vae(cfg, spec: dict, resume: bool = False, images=None, out_root: Path | None = None,
              stop_after_epoch: int | None = None, write_samples: bool = True, quiet: bool = False) -> Path:
    """Train (or resume) the run described by ``spec``. ``stop_after_epoch`` ends this call early (tests / gates)."""
    out_root = out_root or resolve(cfg["paths"]["runs_dir"])
    out = out_root / spec["run_id"]
    ck = out / "checkpoints"
    if out.exists() and not resume:
        raise FileExistsError(f"{out} exists; runs are never overwritten (use --resume to continue it)")
    ck.mkdir(parents=True, exist_ok=True)
    set_deterministic()
    runtime = apply_runtime_flags(cfg, spec)
    dev = resolve_device(cfg["train"]["device"])
    images = images if images is not None else WhistleImages(resolve(cfg["paths"]["dataset_dir"]))
    trainer = Trainer(spec, images, dev, runtime["micro_batch"])
    if resume and (ck / "last.pt").exists():
        trainer.load(ck / "last.pt")
    log_path = out / "train_log.jsonl"

    def log(rec):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if not quiet:
            print(json.dumps(rec), flush=True)

    info_path = out / "train_info.json"
    if not info_path.exists():
        lineage = images.lineage(out_root / "_dataset_hashes.json") if hasattr(images, "lineage") else {"subset": True}
        info = {"status": "running", "spec": spec, "deviations": cfg.to_dict()["deviations"],
                "config_effective_hash": cfg["_meta"]["effective_hash"], "config": {k: v for k, v in cfg.to_dict().items() if k != "_meta"},
                "dataset": lineage, "source_sha256": source_hashes(), "software": software_versions(), "device": str(dev),
                "runtime": runtime,
                "n_params": {"encoder": n_params(trainer.vae.encoder), "decoder": n_params(trainer.vae.decoder),
                             "critic": n_params(trainer.critic) if trainer.critic else 0},
                "decoder_table6_flops": table6_flops(Decoder(spec["latent_dim"]), (spec["latent_dim"],)),
                "steps_per_epoch": -(-len(images) // spec["batch_size"]), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    fixed = torch.from_numpy(images.gather(np.random.default_rng(0).choice(len(images), 12, replace=False)))[:, None]
    t_start = time.time()
    last_epoch = spec["epochs"] if stop_after_epoch is None else min(spec["epochs"], stop_after_epoch)
    for epoch in range(trainer.epoch + 1, last_epoch + 1):
        ep = trainer.run_epoch(epoch, depth=int(cfg["train"]["prefetch"]), log_every=int(cfg["train"]["log_every"]), log_fn=log)
        log({"epoch_summary": True, **ep})
        trainer.save(ck / "last.pt")
        if epoch % int(cfg["train"]["keep_every"]) == 0 or epoch == spec["epochs"]:
            shutil.copyfile(ck / "last.pt", ck / f"epoch_{epoch:03d}.pt")
        if write_samples and (epoch % int(cfg["train"]["sample_every"]) == 0 or epoch in (1, spec["epochs"])):
            save_sample_grid(trainer, fixed, out / f"samples_epoch_{epoch:03d}.png")
    if trainer.epoch == spec["epochs"]:
        final = ck / "final.pt"
        if not final.exists():
            shutil.copyfile(ck / "last.pt", final)
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info.update({"status": "complete", "final_checkpoint_sha256": sha256_file(final),
                     "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "wall_s_last_session": round(time.time() - t_start, 1)})
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return out


def load_trained_vae(run_dir: Path, device: torch.device, which: str = "final.pt") -> tuple[VAE, dict]:
    st = torch.load(Path(run_dir) / "checkpoints" / which, map_location=device, weights_only=False)
    vae = VAE(st["spec"]["latent_dim"]).to(device)
    vae.load_state_dict(st["vae"])
    vae.eval()
    return vae, st["spec"]


@torch.no_grad()
def encode_means(vae: VAE, images, device: torch.device, batch: int = 1024) -> np.ndarray:
    """Encoder mean mu(x) of every image, UNROTATED (GMM input, specification §3.5)."""
    out = np.empty((len(images), vae.latent_dim), dtype=np.float32)
    for i in range(0, len(images), batch):
        idx = np.arange(i, min(i + batch, len(images)))
        x = torch.from_numpy(images.gather(idx))[:, None].to(device)
        out[idx] = vae.encoder(x)[0].cpu().numpy()
    return out


def canonical_spec_hash(spec: dict) -> str:
    return canonical_hash(spec)
