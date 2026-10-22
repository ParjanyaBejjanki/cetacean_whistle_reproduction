"""Full-covariance Gaussian mixture fitted by standard EM (specification §3.5; paper §II-E, eq. 6, App. B).

Scale: N = 180,154 latents, D = 64, K up to 15,000. The N x K responsibility matrix (14 GB in float64) never exists:
the E- and M-steps run over chunks of points x chunks of components on the GPU.

Numerics
* Mahalanobis distances use explicit differences (x - mu_k) and W_k = L_k^-1 (Cholesky factor computed in float64,
  applied in float32): no cancellation from expanding the quadratic form.
* M-step moments are shifted by the current means (single pass, no catastrophic cancellation) and accumulated in float64.
* Log-likelihood accumulated in float64.

[DECISION] items (not in the paper): k-means++ initialisation, scale-relative ridge regularisation, relative
log-likelihood stopping rule, freezing of components with N_k < empty_nk.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

LOG2PI = math.log(2.0 * math.pi)


def kmeanspp_seeds(X: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """Indices of k-means++ seeds (Arthur & Vassilvitskii 2007). Device-independent: the random numbers come from a
    NumPy generator and the squared distances are accumulated in float64 (a seeded torch generator gives different
    streams on CPU and CUDA)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    Xd = X.double()
    idx = torch.empty(k, dtype=torch.long, device=X.device)
    idx[0] = int(rng.integers(n))
    d2 = ((Xd - Xd[idx[0]]) ** 2).sum(1)
    u_all = torch.as_tensor(rng.random(k), dtype=torch.float64, device=X.device)
    for j in range(1, k):
        c = torch.cumsum(d2, 0)
        i = torch.searchsorted(c, (u_all[j] * c[-1]).reshape(1)).clamp_(max=n - 1)
        idx[j] = i[0]
        d2 = torch.minimum(d2, ((Xd - Xd[i]) ** 2).sum(1))
    return idx


def nearest(X: torch.Tensor, C: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Index of the nearest row of C for every row of X (squared Euclidean)."""
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    cn = (C ** 2).sum(1)
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk]
        d = (x ** 2).sum(1, keepdim=True) - 2.0 * x @ C.T + cn[None]
        out[i:i + chunk] = d.argmin(1)
    return out


class GaussianMixtureEM:
    def __init__(self, n_components: int, reg_rel: float = 1e-6, max_iter: int = 1500, tol: float = 1e-8,
                 empty_nk: float = 1.0, seed: int = 0, device: str | torch.device = "cuda",
                 point_chunk: int = 2048, component_chunk: int = 500, verbose: bool = True):
        self.K, self.reg_rel, self.max_iter, self.tol = int(n_components), float(reg_rel), int(max_iter), float(tol)
        self.empty_nk, self.seed, self.dev = float(empty_nk), int(seed), torch.device(device)
        self.pc, self.cc, self.verbose = int(point_chunk), int(component_chunk), verbose
        self.weights_ = self.means_ = self.covariances_ = None     # float64 tensors on device
        self.reg_ = None
        self.history: list[dict] = []
        self.converged_, self.n_iter_ = False, 0

    # ------------------------------------------------------------------ parameters -> evaluation cache
    def _cache(self):
        L, info = torch.linalg.cholesky_ex(self.covariances_)
        if int(info.max()) > 0:
            bad = torch.nonzero(info).flatten()[:10].tolist()
            raise np.linalg.LinAlgError(f"covariance not positive definite for components {bad}")
        eye = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device).expand_as(L)
        W = torch.linalg.solve_triangular(L, eye, upper=False)                   # L^-1
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        with torch.no_grad():
            logw = torch.log(self.weights_.clamp_min(1e-300))
        self._L = L
        self._W32 = W.float().contiguous()
        self._mu32 = self.means_.float().contiguous()
        self._const32 = (logw - 0.5 * logdet - 0.5 * self.means_.shape[1] * LOG2PI).float()

    def _log_joint(self, x: torch.Tensor, ks: slice) -> torch.Tensor:
        """log pi_k + log N(x | mu_k, Sigma_k) for points x (n, D) and components ks -> (n, k)."""
        diff = (x[None, :, :] - self._mu32[ks, None, :])                         # (k, n, D)
        y = torch.bmm(diff, self._W32[ks].transpose(1, 2))                      # rows = W_k (x - mu_k)
        return (self._const32[ks, None] - 0.5 * (y * y).sum(-1)).T              # (n, k)

    def _log_joint_all(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._log_joint(x, slice(s, min(s + self.cc, self.K))) for s in range(0, self.K, self.cc)], 1)

    # ------------------------------------------------------------------ EM
    def _em_pass(self, X: torch.Tensor, accumulate: bool):
        K, D = self.K, X.shape[1]
        ll = torch.zeros((), dtype=torch.float64, device=self.dev)
        if accumulate:
            S0 = torch.zeros(K, dtype=torch.float64, device=self.dev)
            S1 = torch.zeros(K, D, dtype=torch.float64, device=self.dev)
            S2 = torch.zeros(K, D, D, dtype=torch.float64, device=self.dev)
        for i in range(0, X.shape[0], self.pc):
            x = X[i:i + self.pc]
            lj = self._log_joint_all(x)                                          # (n, K)
            lse = torch.logsumexp(lj, 1)
            ll += lse.double().sum()
            if not accumulate:
                continue
            r = torch.exp(lj - lse[:, None])                                     # responsibilities (n, K)
            for s in range(0, K, self.cc):
                ks = slice(s, min(s + self.cc, K))
                rk = r[:, ks].T                                                  # (k, n)
                diff = x[None, :, :] - self._mu32[ks, None, :]                   # (k, n, D), shifted by current means
                S0[ks] += rk.double().sum(1)
                S1[ks] += torch.bmm(rk[:, None, :], diff)[:, 0].double()
                S2[ks] += torch.bmm((diff * rk[:, :, None]).transpose(1, 2), diff).double()
        return (ll, S0, S1, S2) if accumulate else (ll, None, None, None)

    def _m_step(self, S0, S1, S2, n: int):
        D = S1.shape[1]
        ok = S0 >= self.empty_nk
        nk = S0.clamp_min(1e-300)
        d = S1 / nk[:, None]
        cov = S2 / nk[:, None, None] - d[:, :, None] * d[:, None, :]
        cov = 0.5 * (cov + cov.transpose(1, 2)) + self.reg_ * torch.eye(D, dtype=torch.float64, device=self.dev)
        self.means_ = torch.where(ok[:, None], self._mu32.double() + d, self.means_)   # moments were shifted by mu32
        self.covariances_ = torch.where(ok[:, None, None], cov, self.covariances_)
        self.weights_ = S0 / S0.sum()          # = N_k / N; renormalised against float32 rounding of the responsibilities
        return int((~ok).sum())

    def _init_from_seeds(self, X: torch.Tensor):
        n, D = X.shape
        seeds = kmeanspp_seeds(X, self.K, self.seed)
        lab = nearest(X, X[seeds])
        counts = torch.bincount(lab, minlength=self.K).double()
        sums = torch.zeros(self.K, D, dtype=torch.float64, device=self.dev).index_add_(0, lab, X.double())
        means = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], X[seeds].double())
        diff = X.double() - means[lab]
        S2 = torch.zeros(self.K, D, D, dtype=torch.float64, device=self.dev)
        for i in range(0, n, 8192):
            dd = diff[i:i + 8192]
            S2.index_add_(0, lab[i:i + 8192], dd[:, :, None] * dd[:, None, :])
        glob = torch.cov(X.double().T)
        cov = torch.where(counts[:, None, None] > 0, S2 / counts.clamp_min(1)[:, None, None], glob[None])
        self.means_ = means
        self.covariances_ = cov + self.reg_ * torch.eye(D, dtype=torch.float64, device=self.dev)
        self.weights_ = counts.clamp_min(1e-12) / counts.clamp_min(1e-12).sum()
        self.init_info = {"method": "kmeans++ seeds -> nearest-seed assignment -> M-step", "seed": self.seed,
                          "empty_after_assignment": int((counts == 0).sum())}

    def fit(self, X_np: np.ndarray, init_params: dict | None = None, log_path: Path | None = None) -> "GaussianMixtureEM":
        X = torch.as_tensor(np.ascontiguousarray(X_np, dtype=np.float32), device=self.dev)
        n, D = X.shape
        self.n_fit_ = n
        self.reg_ = self.reg_rel * float(X.double().var(0, unbiased=True).mean())
        if init_params is not None:
            self.weights_ = torch.as_tensor(init_params["weights"], dtype=torch.float64, device=self.dev)
            self.means_ = torch.as_tensor(init_params["means"], dtype=torch.float64, device=self.dev)
            self.covariances_ = torch.as_tensor(init_params["covariances"], dtype=torch.float64, device=self.dev)
            self.init_info = {"method": "given"}
        else:
            self._init_from_seeds(X)
        prev = None
        t0 = time.time()
        for it in range(1, self.max_iter + 1):
            self._cache()
            ll, S0, S1, S2 = self._em_pass(X, accumulate=True)
            mean_ll = float(ll) / n
            n_frozen = self._m_step(S0, S1, S2, n)
            rec = {"iter": it, "mean_loglik": mean_ll, "n_frozen": n_frozen, "elapsed_s": round(time.time() - t0, 1)}
            self.history.append(rec)
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            if self.verbose and (it <= 3 or it % 25 == 0):
                print(json.dumps(rec), flush=True)
            self.n_iter_ = it
            if not math.isfinite(mean_ll):
                raise FloatingPointError(f"non-finite log-likelihood at iteration {it}")
            if prev is not None and abs(mean_ll - prev) <= self.tol * abs(mean_ll):
                self.converged_ = True
                break
            prev = mean_ll
        self._cache()
        self.final_mean_loglik_ = float(self._em_pass(X, accumulate=False)[0]) / n
        return self

    # ------------------------------------------------------------------ inference (eqs. 11-12) and sampling
    @torch.no_grad()
    def log_joint(self, X_np: np.ndarray) -> np.ndarray:
        """log pi_k N(x|mu_k, Sigma_k) for every point and component (small inputs only: returns (n, K))."""
        X = torch.as_tensor(np.asarray(X_np, dtype=np.float32), device=self.dev)
        return torch.cat([self._log_joint_all(X[i:i + self.pc]) for i in range(0, len(X), self.pc)]).cpu().numpy()

    @torch.no_grad()
    def predict(self, X_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """MAP component k^ = argmax_k p(C_k | z) (eq. 12) and its responsibility (eq. 11)."""
        X = torch.as_tensor(np.asarray(X_np, dtype=np.float32), device=self.dev)
        lab = np.empty(len(X), dtype=np.int64)
        resp = np.empty(len(X), dtype=np.float64)
        for i in range(0, len(X), self.pc):
            lj = self._log_joint_all(X[i:i + self.pc])
            m, a = lj.max(1)
            lab[i:i + self.pc] = a.cpu().numpy()
            resp[i:i + self.pc] = torch.exp(m - torch.logsumexp(lj, 1)).double().cpu().numpy()
        return lab, resp

    @torch.no_grad()
    def score_samples(self, X_np: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(np.asarray(X_np, dtype=np.float32), device=self.dev)
        return np.concatenate([torch.logsumexp(self._log_joint_all(X[i:i + self.pc]), 1).double().cpu().numpy()
                               for i in range(0, len(X), self.pc)])

    @torch.no_grad()
    def sample(self, n: int, generator: torch.Generator, weights: np.ndarray | None = None,
               components: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """C ~ Cat(weights) (default: pi), z ~ N(mu_C, Sigma_C). ``components`` forces the component of every sample."""
        if components is None:
            w = torch.as_tensor(self.weights_ if weights is None else weights, dtype=torch.float64, device=self.dev)
            comp = torch.multinomial(w / w.sum(), n, replacement=True, generator=generator)
        else:
            comp = torch.as_tensor(components, dtype=torch.long, device=self.dev)
        eps = torch.randn(len(comp), self.means_.shape[1], 1, generator=generator, device=self.dev, dtype=torch.float64)
        z = self.means_[comp] + torch.bmm(self._L[comp], eps)[:, :, 0]
        return z.float(), comp

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path, extra: dict | None = None) -> None:
        np.savez(path, weights=self.weights_.cpu().numpy(), means=self.means_.cpu().numpy(),
                 covariances=self.covariances_.cpu().numpy())
        meta = {"n_components": self.K, "reg_rel": self.reg_rel, "reg_abs": self.reg_, "max_iter": self.max_iter,
                "tol": self.tol, "tol_mode": "relative_mean_loglik", "empty_nk": self.empty_nk, "seed": self.seed,
                "converged": self.converged_, "n_iter": self.n_iter_, "final_mean_loglik": getattr(self, "final_mean_loglik_", None),
                "init": getattr(self, "init_info", None), **(extra or {})}
        Path(str(path).replace(".npz", ".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cuda", **kw) -> "GaussianMixtureEM":
        z = np.load(path)
        meta = json.loads(Path(str(path).replace(".npz", ".json")).read_text(encoding="utf-8"))
        g = cls(meta["n_components"], meta["reg_rel"], meta["max_iter"], meta["tol"], meta["empty_nk"], meta["seed"], device, **kw)
        g.weights_ = torch.as_tensor(z["weights"], device=g.dev)
        g.means_ = torch.as_tensor(z["means"], device=g.dev)
        g.covariances_ = torch.as_tensor(z["covariances"], device=g.dev)
        g.reg_ = meta["reg_abs"]
        g.converged_, g.n_iter_ = meta["converged"], meta["n_iter"]
        g._cache()
        return g

    def diagnostics(self, n_points: int | None = None) -> dict:
        n_points = n_points or getattr(self, "n_fit_", None)
        w = self.weights_.cpu().numpy()
        ev = torch.linalg.eigvalsh(self.covariances_).cpu().numpy()
        out = {"weights_min": float(w.min()), "weights_max": float(w.max()), "eig_min": float(ev.min()),
               "eig_max": float(ev.max()), "reg_abs": self.reg_, "frac_eigs_at_ridge": float((ev < 1.01 * self.reg_).mean())}
        if n_points:
            nk = w * n_points
            d = self.means_.shape[1]
            out.update({"n_points": int(n_points), "nk_quantiles_0_5_50_95_100": np.quantile(nk, [0, .05, .5, .95, 1]).tolist(),
                        "n_components_nk_lt_1": int((nk < 1).sum()), f"n_components_nk_lt_{d + 1}": int((nk < d + 1).sum())})
        return out
