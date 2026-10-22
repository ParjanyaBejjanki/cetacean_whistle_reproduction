"""Metrics of Tables 1-3 and Fig. 11 (feature-network-independent parts).

* Table 1: per-image SSE ("MSE" column, [IMPLIED]), PSNR = 10 log10(16384 / MSE), Gaussian-window SSIM, latent KL.
* Fréchet distance (FID / FFD), 1-D Wasserstein-1 of eq. 13, improved precision / recall of [47] (k-NN manifolds).
* Fig. 11: exact silhouette (GPU, chunked), Davies-Bouldin and Calinski-Harabasz (scikit-learn).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


# ---------------------------------------------------------------- Table 1
def psnr_from_mse(mse_sum: float, n_pixels: int = 128 * 128, peak: float = 1.0) -> float:
    """PSNR from the mean per-image SSE (Table 1 convention: MSE column = per-image sum of squared errors)."""
    return 10.0 * math.log10(peak ** 2 * n_pixels / mse_sum)


def _gauss_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float64, device=device) - (size - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).float()[None, None]


@torch.no_grad()
def ssim(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03, data_range: float = 1.0) -> torch.Tensor:
    """Per-image SSIM (Wang et al. 2004): 11x11 Gaussian window (sigma 1.5), valid region. x, y: (B, 1, H, W)."""
    w = _gauss_window(11, sigma, x.device)
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    mx, my = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mx ** 2
    syy = F.conv2d(y * y, w) - my ** 2
    sxy = F.conv2d(x * y, w) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


# ---------------------------------------------------------------- distribution distances
def frechet_distance(mu1, s1, mu2, s2) -> float:
    """||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^(1/2))  (Heusel et al. [36]); float64, imaginary residue dropped."""
    mu1, mu2 = np.atleast_1d(np.asarray(mu1, np.float64)), np.atleast_1d(np.asarray(mu2, np.float64))
    s1, s2 = np.atleast_2d(np.asarray(s1, np.float64)), np.atleast_2d(np.asarray(s2, np.float64))
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))


def frechet_from_samples(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return frechet_distance(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))


def wasserstein1_sorted(x: np.ndarray, y: np.ndarray) -> float:
    """eq. 13: (1/N) sum |x_(i) - y_(i)| over order statistics; requires equal sample sizes."""
    x, y = np.sort(np.asarray(x, np.float64)), np.sort(np.asarray(y, np.float64))
    if x.shape != y.shape:
        raise ValueError("eq. 13 needs equal sample sizes")
    return float(np.abs(x - y).mean())


def _sqdist(q: torch.Tensor, r: torch.Tensor, qn: torch.Tensor, rn: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances (float64 result). The dot products run in float64 for low-dimensional inputs
    (exact ties matter for integer-valued whistle features) and in float32 for network features (speed)."""
    dots = (q.double() @ r.double().T) if q.shape[1] <= 64 else (q.float() @ r.float().T).double()
    return (qn[:, None] + rn[None, :] - 2.0 * dots).clamp_min_(0)


@torch.no_grad()
def _kth_nn_radius(a: torch.Tensor, k: int, device, chunk: int = 1024, ref_chunk: int = 16384) -> torch.Tensor:
    """Distance from every row of ``a`` (CPU) to its k-th nearest OTHER row (k+1 smallest incl. self). Chunks go to ``device``."""
    n = a.shape[0]
    an = (a.double() ** 2).sum(1)
    out = torch.empty(n, dtype=torch.float64)
    refs = [(a[j:j + ref_chunk].to(device), an[j:j + ref_chunk].to(device)) for j in range(0, n, ref_chunk)]         if a.numel() * 4 < 2e9 else None
    for i in range(0, n, chunk):
        q, qn = a[i:i + chunk].to(device), an[i:i + chunk].to(device)
        best = None
        for jj, j in enumerate(range(0, n, ref_chunk)):
            r, rn = refs[jj] if refs else (a[j:j + ref_chunk].to(device), an[j:j + ref_chunk].to(device))
            d2 = _sqdist(q, r, qn, rn)
            cand = d2 if best is None else torch.cat([best, d2], 1)
            best = torch.topk(cand, min(k + 1, cand.shape[1]), dim=1, largest=False).values
        out[i:i + chunk] = best[:, k].sqrt().cpu()
    return out


@torch.no_grad()
def _in_manifold(query: torch.Tensor, ref: torch.Tensor, radii: torch.Tensor, device, chunk: int = 1024,
                 ref_chunk: int = 16384) -> torch.Tensor:
    """For each query row: within radius_r of at least one reference row r (distance <= radius)? Inputs on CPU."""
    qn, rn = (query.double() ** 2).sum(1), (ref.double() ** 2).sum(1)
    r2 = radii.double() ** 2
    out = torch.zeros(query.shape[0], dtype=torch.bool)
    refs = [(ref[j:j + ref_chunk].to(device), rn[j:j + ref_chunk].to(device), r2[j:j + ref_chunk].to(device))
            for j in range(0, ref.shape[0], ref_chunk)] if ref.numel() * 4 < 2e9 else None
    for i in range(0, query.shape[0], chunk):
        q, qni = query[i:i + chunk].to(device), qn[i:i + chunk].to(device)
        hit = torch.zeros(q.shape[0], dtype=torch.bool, device=device)
        for jj, j in enumerate(range(0, ref.shape[0], ref_chunk)):
            r, rnj, r2j = refs[jj] if refs else (ref[j:j + ref_chunk].to(device), rn[j:j + ref_chunk].to(device),
                                                 r2[j:j + ref_chunk].to(device))
            d2 = _sqdist(q, r, qni, rnj)
            hit |= (d2 <= r2j[None] * (1 + 1e-12) + 1e-12).any(1)
        out[i:i + chunk] = hit.cpu()
    return out


@torch.no_grad()
def precision_recall(real: np.ndarray, fake: np.ndarray, k: int = 3, device: str = "cpu") -> dict:
    """Improved precision and recall [47]: precision = fraction of fake samples inside the real k-NN manifold,
    recall = fraction of real samples inside the fake k-NN manifold. F1 = 2PR / (P + R)."""
    r = torch.as_tensor(np.asarray(real, np.float32))
    f = torch.as_tensor(np.asarray(fake, np.float32))
    p = float(_in_manifold(f, r, _kth_nn_radius(r, k, device), device).double().mean())
    rc = float(_in_manifold(r, f, _kth_nn_radius(f, k, device), device).double().mean())
    return {"precision": p, "recall": rc, "f1": 0.0 if p + rc == 0 else 2 * p * rc / (p + rc), "k": k,
            "n_real": len(real), "n_fake": len(fake)}


# ---------------------------------------------------------------- Fig. 11
@torch.no_grad()
def silhouette_exact(X: np.ndarray, labels: np.ndarray, device: str = "cpu", chunk: int = 512) -> float:
    """Mean silhouette over all points, Euclidean, exact (scikit-learn convention: s = 0 for singleton clusters)."""
    x = torch.as_tensor(np.asarray(X, np.float32), device=device)
    lab = torch.as_tensor(np.asarray(labels), device=device, dtype=torch.long)
    uniq, lab = torch.unique(lab, return_inverse=True)
    K = len(uniq)
    size = torch.bincount(lab, minlength=K).double()
    xn = (x.double() ** 2).sum(1)
    s_sum = 0.0
    for i in range(0, x.shape[0], chunk):
        q = x[i:i + chunk]
        sums = torch.zeros(q.shape[0], K, dtype=torch.float64, device=device)
        for j in range(0, x.shape[0], 16384):
            # float32 dot products (N^2 = 3.2e10 pairs); the per-cluster sums are accumulated in float64
            d = (xn[i:i + chunk, None] + xn[None, j:j + 16384] - 2.0 * (q @ x[j:j + 16384].T).double()).clamp_min_(0).sqrt()
            sums.index_add_(1, lab[j:j + 16384], d)
        own = lab[i:i + chunk]
        rows = torch.arange(q.shape[0], device=device)
        n_own = size[own]
        a = sums[rows, own] / (n_own - 1).clamp_min(1)
        mean_other = sums / size[None]
        mean_other[rows, own] = float("inf")
        b = mean_other.min(1).values
        s = (b - a) / torch.maximum(a, b)
        s = torch.where(n_own > 1, s, torch.zeros_like(s))
        s_sum += float(torch.nan_to_num(s, nan=0.0).sum())
    return s_sum / x.shape[0]


def clustering_scores(X: np.ndarray, labels: np.ndarray, device: str = "cpu") -> dict:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

    return {"silhouette": silhouette_exact(X, labels, device), "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)), "n_clusters_used": int(len(np.unique(labels)))}
