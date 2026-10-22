"""Image feature networks for FID (Inception-V3, 2048-d final pooling) and precision/recall ([47]: VGG-16 fc2).

[EXPLICIT] §IV-D-2: 128x128 grayscale -> bicubic resize to 224x224 -> channel replicated to 3 -> Inception V3 2048-d
           final pooling activations. FID1: real AND generated images binarised at 0.1 first; FID2: no thresholding.
[DECISION] torchvision ImageNet weights (IMAGENET1K_V1) for both networks, ImageNet normalisation, VGG-16 feature = fc2
           after ReLU; bicubic resize with the MATLAB imresize convention (Keys kernel a = -0.5, 6 taps, symmetric
           boundary, coordinate map u = x/s + (1 - 1/s)/2, no clamping) because the paper's work used MATLAB R2024a;
           PyTorch's own bicubic (a = -0.75) differs and FID is sensitive to the resize kernel.

Weights are NEVER downloaded implicitly: networks load from local files under models/eval_weights/.
``download_weights()`` fetches them and is only run on the project owner's explicit approval.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHT_URLS = {   # torchvision release files
    "inception_v3": "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def weight_path(root: Path, name: str) -> Path:
    return Path(root) / Path(WEIGHT_URLS[name]).name


def download_weights(root: Path) -> dict:
    """Download both weight files (requires the owner's approval; see specification §5.4)."""
    Path(root).mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in WEIGHT_URLS.items():
        dst = weight_path(root, name)
        if not dst.exists():
            torch.hub.download_url_to_file(url, str(dst), progress=True)
        out[name] = {"path": str(dst), "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
    return out


def _keys(t: torch.Tensor, a: float = -0.5) -> torch.Tensor:
    at = t.abs()
    return torch.where(at <= 1, (a + 2) * at ** 3 - (a + 3) * at ** 2 + 1,
                       torch.where(at < 2, a * at ** 3 - 5 * a * at ** 2 + 8 * a * at - 4 * a, torch.zeros_like(at)))


def imresize_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """(n_out, n_in) bicubic interpolation matrix following MATLAB imresize (upscaling: no antialiasing)."""
    scale = n_out / n_in
    x = torch.arange(1, n_out + 1, dtype=torch.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = torch.floor(u - 2.0)
    taps = left[:, None] + torch.arange(6, dtype=torch.float64)[None]          # P = ceil(4) + 2 = 6
    w = _keys(u[:, None] - taps)
    w = w / w.sum(1, keepdim=True)
    aux = torch.cat([torch.arange(1, n_in + 1), torch.arange(n_in, 0, -1)])     # symmetric boundary
    idx = aux[((taps.long() - 1) % (2 * n_in))] - 1
    m = torch.zeros(n_out, n_in, dtype=torch.float64)
    m.scatter_add_(1, idx, w)
    return m


_RESIZE: dict = {}


def preprocess(x: torch.Tensor, binarize: float | None, size: int = 224) -> torch.Tensor:
    """(B, 1, 128, 128) in [0, 1] -> (B, 3, size, size) ImageNet-normalised (binarise first for FID1)."""
    if binarize is not None:
        x = (x > binarize).to(torch.float32)
    key = (x.shape[-1], size, x.device)
    if key not in _RESIZE:
        _RESIZE[key] = imresize_matrix(x.shape[-1], size).float().to(x.device)
    m = _RESIZE[key]
    x = m @ x.float() @ m.T
    x = x.expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class InceptionPool(nn.Module):
    def __init__(self, weights_file: Path):
        super().__init__()
        from torchvision.models import inception_v3

        # transform_input=True: the torchvision Google weights expect it (it is what weights=IMAGENET1K_V1 sets)
        m = inception_v3(weights=None, aux_logits=True, transform_input=True, init_weights=False)
        m.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
        m.fc = nn.Identity()
        self.net = m.eval()

    def forward(self, x):
        return self.net(x)


class VGGfc2(nn.Module):
    def __init__(self, weights_file: Path):
        super().__init__()
        from torchvision.models import vgg16

        m = vgg16(weights=None)
        m.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
        m.classifier = m.classifier[:5]          # fc1, ReLU, Dropout, fc2, ReLU  -> 4096-d fc2 activations
        self.net = m.eval()

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def extract(net: nn.Module, batches, binarize: float | None, device: torch.device, size: int = 224) -> np.ndarray:
    """Features for an iterable of (B, 1, 128, 128) tensors in [0, 1]."""
    net = net.to(device).eval()
    out = [net(preprocess(b.to(device), binarize, size)).float().cpu().numpy() for b in batches]
    return np.concatenate(out)


def stats(feats: np.ndarray, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Mean and unbiased covariance accumulated in float64 over chunks (features stay float32 in memory)."""
    n, d = feats.shape
    s1 = np.zeros(d)
    for i in range(0, n, chunk):
        s1 += feats[i:i + chunk].astype(np.float64).sum(0)
    mu = s1 / n
    s2 = np.zeros((d, d))
    for i in range(0, n, chunk):
        c = feats[i:i + chunk].astype(np.float64) - mu
        s2 += c.T @ c
    return mu, s2 / (n - 1)
