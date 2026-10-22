"""Training data: the fixed preprocessed dataset, opened READ-ONLY (memory-mapped), plus deterministic batch streams.

180,154 images (11.8 GB float32) do not fit comfortably in RAM next to training, so batches are gathered from the
memory-mapped per-encounter arrays by a background thread. Nothing in preprocessed/ is ever written.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
from pathlib import Path

import numpy as np
import torch

from ..detector.ridgenet import sha256_file


class WhistleImages:
    """All encounter arrays of preprocessed/ behind one global index (encounter order = sorted folder name)."""

    def __init__(self, dataset_dir: Path):
        self.dir = Path(dataset_dir)
        self.info_path = self.dir / "dataset_info.json"
        self.info = json.loads(self.info_path.read_text(encoding="utf-8"))
        self.enc_dirs = sorted(p for p in self.dir.glob("encounter_*") if p.is_dir() and not p.name.endswith(".partial"))
        self.arrays = [np.load(d / "images.npy", mmap_mode="r") for d in self.enc_dirs]
        for d, a in zip(self.enc_dirs, self.arrays):
            if a.dtype != np.float32 or a.shape[1:] != (128, 128):
                raise ValueError(f"{d}/images.npy: unexpected {a.dtype} {a.shape}")
            expected = self.info["encounters"][d.name]["n_images"]
            if a.shape[0] != expected:
                raise ValueError(f"{d}/images.npy has {a.shape[0]} images, dataset_info.json says {expected}")
        self.counts = np.array([a.shape[0] for a in self.arrays], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])
        if int(self.offsets[-1]) != self.info["totals"]["images"]:
            raise ValueError("total image count differs from dataset_info.json")

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def gather(self, idx: np.ndarray) -> np.ndarray:
        """Images for global indices ``idx`` (any order), float32 (n, 128, 128). Reads are sorted per encounter."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((len(idx), 128, 128), dtype=np.float32)
        enc = np.searchsorted(self.offsets, idx, side="right") - 1
        for e in np.unique(enc):
            sel = np.nonzero(enc == e)[0]
            local = idx[sel] - self.offsets[e]
            order = np.argsort(local, kind="stable")
            out[sel[order]] = self.arrays[e][local[order]]
        return out

    def lineage(self, cache_path: Path | None = None) -> dict:
        """dataset identity: dataset_info.json hash + processing/detector hashes + SHA-256 of every images.npy.
        Array hashes are cached by (path, size, mtime) because hashing 11.8 GB takes about a minute."""
        cache = {}
        if cache_path and cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        arrays = {}
        for d in self.enc_dirs:
            p = d / "images.npy"
            st = p.stat()
            key = f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"
            if key not in cache:
                cache[key] = sha256_file(p)
            arrays[d.name] = {"n_images": int(self.arrays[self.enc_dirs.index(d)].shape[0]), "sha256": cache[key]}
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        return {"dataset_dir": str(self.dir), "dataset_info_sha256": sha256_file(self.info_path),
                "processing_config_hash": self.info.get("processing_config_hash"),
                "detector_checkpoint_sha256": self.info.get("detector", {}).get("checkpoint_sha256"),
                "n_images": len(self), "arrays": arrays}

    def scan(self, chunk: int = 2048) -> dict:
        """Integrity scan of every image: finite, within [0, 1], per-image max == 1 (dataset_info claim)."""
        n_bad_finite = n_bad_range = n_max_not_1 = 0
        gmin, gmax = np.inf, -np.inf
        for a in self.arrays:
            for i in range(0, a.shape[0], chunk):
                x = np.asarray(a[i:i + chunk])
                fin = np.isfinite(x).all(axis=(1, 2))
                n_bad_finite += int((~fin).sum())
                mn, mx = x.min(axis=(1, 2)), x.max(axis=(1, 2))
                n_bad_range += int(((mn < 0) | (mx > 1)).sum())
                n_max_not_1 += int((np.abs(mx - 1.0) > 1e-6).sum())
                gmin, gmax = min(gmin, float(mn.min())), max(gmax, float(mx.max()))
        return {"n_images": len(self), "n_non_finite": n_bad_finite, "n_out_of_range": n_bad_range,
                "n_max_not_1": n_max_not_1, "global_min": gmin, "global_max": gmax}


class SubsetImages:
    """A fixed subset of a WhistleImages (for gates); same gather() interface."""

    def __init__(self, base: WhistleImages, indices: np.ndarray):
        self.base, self.indices = base, np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def gather(self, idx: np.ndarray) -> np.ndarray:
        return self.base.gather(self.indices[np.asarray(idx, dtype=np.int64)])


def epoch_seed(seed: int, epoch: int, stream: int) -> int:
    h = hashlib.sha256(f"{seed}|{epoch}|{stream}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def epoch_plan(n: int, batch: int, n_critic: int, seed: int, epoch: int, use_critic: bool = True):
    """Deterministic index plan of one epoch.
    VAE stream: one random permutation in batches (last batch partial) -> defines the epoch.
    Critic stream: independent permutations, ``n_critic`` FULL batches per VAE step ([49] Alg. 1: fresh real samples)."""
    perm = np.random.default_rng(epoch_seed(seed, epoch, 0)).permutation(n)
    vae = [perm[i:i + batch] for i in range(0, n, batch)]
    critic: list[list[np.ndarray]] = [[] for _ in vae]
    if use_critic and n_critic > 0:
        need = len(vae) * n_critic * batch
        rng = np.random.default_rng(epoch_seed(seed, epoch, 1))
        pool = np.concatenate([rng.permutation(n) for _ in range(-(-need // n))])[:need]
        cb = pool.reshape(len(vae), n_critic, batch)
        critic = [list(cb[i]) for i in range(len(vae))]
    return vae, critic


class Prefetcher:
    """Background thread gathering (critic batches, vae batch) for each step into pinned CPU tensors."""

    def __init__(self, images, vae_batches, critic_batches, depth: int = 6, pin: bool = True):
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self.wait_s = 0.0

        def to_t(a):
            t = torch.from_numpy(a)[:, None]
            return t.pin_memory() if pin else t

        def work():
            try:
                for vb, cbs in zip(vae_batches, critic_batches):
                    if self._stop.is_set():
                        return
                    item = ([to_t(images.gather(c)) for c in cbs], to_t(images.gather(vb)))
                    while not self._stop.is_set():
                        try:
                            self.q.put(item, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                self.q.put(None)
            except BaseException as e:   # noqa: BLE001  (surface loader errors in the training thread)
                self.q.put(e)

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def __iter__(self):
        import time
        while True:
            t0 = time.perf_counter()
            item = self.q.get()
            self.wait_s += time.perf_counter() - t0
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self):
        self._stop.set()


def random_rot90(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Rotate each image x[i] (B, 1, H, W) counter-clockwise by k[i] * 90 degrees (H == W). No host sync."""
    rots = torch.stack([torch.rot90(x, r, dims=(2, 3)) for r in range(4)])
    return rots[k, torch.arange(x.shape[0], device=x.device)]
