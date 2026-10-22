"""Component-wise analysis and conditional generation (paper §III-B, eqs. 7-10; §IV-B).

eq. 8   pi~_k = pi_k / sum_{k in K} pi_k                         (conditional component weight)
eq. 10  pi^_k = a_k pi_k / sum_{k in K} a_k pi_k                  (adjusted weight, a_k = acceptance ratio)
Sampler (§III-B steps 1-3 + rejection sampling): C ~ Cat(weights) over K, z ~ N(mu_C, Sigma_C), x = decoder(z);
samples that do not satisfy the condition are discarded until N are accepted.
[DECISION] K = {k : a_k > 0}; a_k from the 500 per-component samples (Fig. 8); Table 3 primary = eq. 10.
"""
from __future__ import annotations

import numpy as np
import torch

from .features import FEATURES, contour_features, satisfies


@torch.no_grad()
def component_features(decoder, gmm, n_per_component: int, generator: torch.Generator, threshold: float = 0.05,
                       batch: int = 4000, progress: bool = False) -> np.ndarray:
    """(K, n_per_component, 7) features of samples drawn from every component (NaN for empty images)."""
    K = gmm.K
    out = np.empty((K, n_per_component, len(FEATURES)), dtype=np.float32)
    comps = np.repeat(np.arange(K), n_per_component)
    for i in range(0, len(comps), batch):
        c = comps[i:i + batch]
        z, _ = gmm.sample(len(c), generator, components=c)
        f, _ = contour_features(decoder(z), threshold)
        out.reshape(-1, len(FEATURES))[i:i + len(c)] = f.cpu().numpy()
        if progress and (i // batch) % 100 == 0:
            print(f"component features {i + len(c)}/{len(comps)}", flush=True)
    return out


def acceptance_ratios(comp_feats: np.ndarray, feature: str, lo: float, hi: float) -> np.ndarray:
    """a_k = fraction of a component's samples satisfying the condition."""
    return satisfies(comp_feats, feature, lo, hi).mean(1)


def weights_eq8(pi: np.ndarray, member: np.ndarray) -> np.ndarray:
    w = np.where(member, pi, 0.0)
    return w / w.sum()


def weights_eq10(pi: np.ndarray, a: np.ndarray, member: np.ndarray) -> np.ndarray:
    w = np.where(member, a * pi, 0.0)
    return w / w.sum()


@torch.no_grad()
def conditional_sample(decoder, gmm, weights: np.ndarray, feature: str, lo: float, hi: float, n_target: int,
                       generator: torch.Generator, threshold: float = 0.05, batch: int = 4000,
                       max_draws: int | None = None, keep_images: int = 0) -> dict:
    """Rejection sampler. Returns accepted features (n_target, 7), acceptance rate, and optionally the first images."""
    max_draws = max_draws or 1000 * n_target
    acc_f, acc_img, draws = [], [], 0
    n_acc = n_img = 0
    while n_acc < n_target:
        if draws >= max_draws:
            raise RuntimeError(f"rejection sampler exceeded {max_draws} draws ({n_acc}/{n_target} accepted)")
        z, _ = gmm.sample(batch, generator, weights=weights)
        x = decoder(z)
        f, _ = contour_features(x, threshold)
        ok = satisfies(f, feature, lo, hi)
        draws += batch
        if bool(ok.any()):
            acc_f.append(f[ok].cpu().numpy())
            if n_img < keep_images:
                acc_img.append(x[ok][:keep_images - n_img, 0].cpu().numpy())
                n_img += len(acc_img[-1])
            n_acc += int(ok.sum())
    feats = np.concatenate(acc_f)[:n_target]
    out = {"features": feats, "draws": draws, "acceptance_rate": n_acc / draws}
    if keep_images:
        out["images"] = np.concatenate(acc_img)[:keep_images]
    return out
