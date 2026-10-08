"""RidgeNet: the replacement S5 detector (fully convolutional, ~14k parameters) and its inference wrapper.

Design rationale (from the S4 representation, not copied from any paper):
  * a whistle is a thin ridge (about 1-3 bins wide, 20-100+ frames long) that may sweep several bins per frame;
  * a click is a broadband streak, 1-3 frames wide;
  * so the network needs a long, narrow receptive field in time and a moderate one in frequency, at full
    resolution (no pooling, so the output stays aligned with the S4 grid);
  * dilated 3x3 convolutions give a large receptive field with few parameters; residual connections ease training.
Time-dilations (1,2,4,8,1,2) and frequency-dilations (1,1,2,2,1,1) give a receptive field of 39 frames (208 ms)
by 19 bins (1.8 kHz).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import Cfg, band_bins
from ..dsp import Spectrogram
from .conditioning import condition
from .interface import DetectionMap, check_detection_map


class RidgeNet(nn.Module):
    def __init__(self, channels: int = 16, dilations=((1, 1), (2, 1), (4, 2), (8, 2), (1, 1), (2, 1))):
        super().__init__()
        self.channels = channels
        self.dilations = [tuple(d) for d in dilations]
        self.stem = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=d, dilation=d, bias=False),
                          nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
            for d in self.dilations])
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # x: (B, 1, T, F) -> logits (B, 1, T, F)
        h = self.stem(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(h)

    def receptive_radius(self) -> tuple[int, int]:
        """(time, frequency) radius in nodes: the output at (t, k) depends only on inputs within this radius."""
        return 1 + sum(d[0] for d in self.dilations), 1 + sum(d[1] for d in self.dilations)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_deterministic() -> None:
    """Numerics policy for S5: deterministic algorithms and FULL float32 (TF32 off). With TF32 on, CUDA convolutions differ from
    the CPU result by ~1e-2 in confidence, which flips threshold crossings and changes S6 tracks between devices."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class RidgeDetector:
    """S5 implementation (implements ``interface.WhistleDetector``)."""

    name = "ridgenet"

    def __init__(self, model: RidgeNet, version: str, band: tuple[int, int], cond: dict,
                 tile_frames: int = 2048, halo_frames: int = 32, device: str = "auto", meta: dict | None = None):
        set_deterministic()
        self.version = version
        self.band = (int(band[0]), int(band[1]))
        self.cond = dict(cond)
        self.tile_frames, self.halo_frames = int(tile_frames), int(halo_frames)
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.meta = meta or {}
        rt, _ = model.receptive_radius()
        if self.halo_frames < rt:
            raise ValueError(f"halo_frames={self.halo_frames} < receptive-field radius {rt}: tiling would change results")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint(cls, path: Path, cfg: Cfg, device: str | None = None) -> "RidgeDetector":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        arch = ck["arch"]
        model = RidgeNet(arch["channels"], arch["dilations"])
        model.load_state_dict(ck["model_state"])
        s5 = cfg["decisions"]["s5"]
        band = band_bins(cfg)
        if tuple(ck["band"]) != tuple(band):
            raise ValueError(f"checkpoint band {ck['band']} != pipeline band {band}")
        if {k: ck["conditioning"][k] for k in s5["conditioning"] if k in ck["conditioning"]} != {
                k: s5["conditioning"][k] for k in s5["conditioning"] if k in ck["conditioning"]}:
            raise ValueError("checkpoint conditioning differs from pipeline config: retrain or fix the config")
        meta = {"checkpoint": str(path), "checkpoint_sha256": sha256_file(path), "n_params": model.n_params(),
                "receptive_radius_tf": list(model.receptive_radius()), "trained_seed": ck.get("seed"),
                "metrics": ck.get("metrics")}
        return cls(model, ck["version"], band, s5["conditioning"], s5["tile_frames"], s5["halo_frames"],
                   device or s5["device"], meta)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_conditioned(self, x: np.ndarray) -> np.ndarray:
        """x: (T, F) conditioned input -> (T, F) float32 probabilities. Tiled in time with a halo; the
        result is independent of the tiling because halo >= receptive-field radius."""
        t = x.shape[0]
        out = np.empty((t, x.shape[1]), np.float32)
        tile, halo = self.tile_frames, self.halo_frames
        for s in range(0, t, tile):
            e = min(t, s + tile)
            lo, hi = max(0, s - halo), min(t, e + halo)
            inp = torch.from_numpy(np.ascontiguousarray(x[lo:hi])).to(self.device)[None, None]
            p = torch.sigmoid(self.model(inp))[0, 0].float().cpu().numpy()
            out[s:e] = p[s - lo: s - lo + (e - s)]
        return out

    def predict(self, spec: Spectrogram) -> DetectionMap:
        k_lo, k_hi = self.band
        band_db = spec.db[:, k_lo:k_hi + 1]
        conf = self.predict_conditioned(condition(band_db, self.cond)) if spec.n_frames else np.zeros((0, k_hi - k_lo + 1), np.float32)
        conf = np.clip(conf, 0.0, 1.0).astype(np.float32)
        dm = DetectionMap(conf=conf, bin_lo=k_lo, bin_hi=k_hi, n_bins_total=spec.n_bins,
                          detector_name=self.name, detector_version=self.version, meta=dict(self.meta))
        check_detection_map(dm, spec)
        return dm
