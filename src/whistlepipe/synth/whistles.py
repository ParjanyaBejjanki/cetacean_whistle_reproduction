"""Synthetic whistle / click generator (time domain @ 192 kHz) with exact labels from the construction process.

Pipeline for one sample (deterministic given (seed, split, epoch, index)):
  real background window (S1..S3 audio)  +  synthetic whistles (FM tones with envelope, harmonics)
  +  synthetic clicks / sweeps (negatives)  ->  S4 STFT (the SAME function the pipeline uses)  ->  S5 conditioning.
Labels come from the known instantaneous-frequency contours, never from any detector output:
  y  soft ridge target: Gaussian across frequency around each contour node (value 1 on the ridge)
  w  per-node loss weight: 1 everywhere except "ignore" nodes (weight 0) next to a contour whose local SNR is
     too low to be visible (those nodes are neither forced positive nor negative).
Harmonics of a whistle that fall inside the analysed band are labelled positive too (S6 suppresses them later).
"""
from __future__ import annotations

import numpy as np
from scipy import interpolate, ndimage
from scipy import signal as sps

from .. import dsp
from ..config import Cfg, band_bins
from ..detector.conditioning import condition


class SynthSampler:
    def __init__(self, cfg: Cfg, dcfg: Cfg, windows: np.ndarray, split_id: int, ridge_masks: np.ndarray | None = None):
        self.cfg, self.dcfg = cfg, dcfg
        self.ridge_masks = ridge_masks          # optional cached label-free masks of real tonal structure (one per window)
        p = cfg["paper"]
        self.fs, self.win, self.hop, self.nfft = int(p["fs_target_hz"]), int(p["win_len"]), int(p["hop"]), int(p["nfft"])
        self.bin_hz = self.fs / self.nfft
        self.k_lo, self.k_hi = band_bins(cfg)
        self.F = self.k_hi - self.k_lo + 1
        self.windows = windows
        self.seed = int(dcfg["synth"]["seed"])
        self.split_id = split_id
        self.sumw_half = float(sps.get_window("hamming", self.win, fftbins=True).sum() / 2)
        self.cond = cfg["decisions"]["s5"]["conditioning"]
        self.wc, self.cc, self.lc = dcfg["synth"]["whistles"], dcfg["synth"]["clicks"], dcfg["synth"]["labels"]

    # ------------------------------------------------------------------ helpers
    def _stft_db(self, x: np.ndarray) -> np.ndarray:
        return dsp.stft_db(x, self.win, self.hop, self.nfft)

    # ------------------------------------------------------------------ whistle contours
    def _contour(self, rng, dur: float) -> np.ndarray:
        wc, fs = self.wc, self.fs
        n = int(round(dur * fs))
        lo, hi = wc["f_range_hz"]
        nk = int(rng.integers(wc["n_knots"][0], wc["n_knots"][1] + 1))
        style = rng.choice(["flat", "up", "down", "arch", "valley", "wavy"], p=[0.05, 0.25, 0.20, 0.15, 0.10, 0.25])
        f0 = rng.uniform(3000, 17000) if rng.random() < 0.8 else rng.uniform(lo + 200, hi - 1500)
        scale = rng.uniform(300, 3000)
        mag = np.abs(rng.normal(0, scale, nk - 1))
        if style == "flat":
            d = rng.normal(0, 0.05 * scale, nk - 1)
        elif style == "up":
            d = mag
        elif style == "down":
            d = -mag
        elif style in ("arch", "valley"):
            s = np.where(np.arange(nk - 1) < (nk - 1) / 2, 1.0, -1.0)
            d = mag * s * (1 if style == "arch" else -1)
        else:
            d = mag * rng.choice([-1.0, 1.0], nk - 1)
        vals = f0 + np.concatenate([[0.0], np.cumsum(d)])
        tau = np.sort(np.concatenate([[0.0, dur], rng.uniform(0.05 * dur, 0.95 * dur, nk - 2)])) if nk > 2 else np.array([0.0, dur])
        tau = tau + np.arange(len(tau)) * 1e-6
        f = interpolate.PchipInterpolator(tau, vals)(np.arange(n) / fs)
        if rng.random() < wc["p_ripple"]:
            f = f + rng.uniform(*wc["ripple_depth_hz"]) * np.sin(2 * np.pi * rng.uniform(*wc["ripple_rate_hz"]) * np.arange(n) / fs + rng.uniform(0, 6.28))
        # slope limit (compress deviations about the mean), then fit into the allowed band
        slope = np.abs(np.gradient(f)).max() * fs
        if slope > wc["slope_max_hz_per_s"]:
            f = f.mean() + (f - f.mean()) * wc["slope_max_hz_per_s"] / slope
        span = f.max() - f.min()
        if span > hi - lo - 200:
            f = f.mean() + (f - f.mean()) * (hi - lo - 200) / span
        if f.min() < lo + 100:
            f = f + (lo + 100 - f.min())
        if f.max() > hi - 100:
            f = f - (f.max() - (hi - 100))
        return f

    def _envelope(self, rng, n: int) -> np.ndarray:
        env = sps.windows.tukey(n, rng.uniform(0.15, 0.6))
        t = np.arange(n) / self.fs
        env = env * (1 + rng.uniform(0, 0.3) * np.sin(2 * np.pi * rng.uniform(1, 5) * t + rng.uniform(0, 6.28)))
        return env / max(env.max(), 1e-9)

    def _draw_snr(self, rng) -> float:
        mix = self.wc.get("snr_mix")                      # optional [[weight, lo_db, hi_db], ...] (ridgenet-v2b); absent = v1 rule
        if mix:
            w = np.array([m[0] for m in mix], float)
            i = int(rng.choice(len(mix), p=w / w.sum()))
            return rng.uniform(mix[i][1], mix[i][2])
        lo, hi = self.wc["snr_db"]
        return rng.uniform(lo, 10.0) if rng.random() < 0.5 else rng.uniform(10.0, hi)

    def _make_whistle(self, rng, wid: int, n_total: int, f: np.ndarray | None, start: int | None, L_bg: np.ndarray, dur=None) -> dict:
        wc, fs = self.wc, self.fs
        if f is None:
            d = float(np.exp(rng.uniform(np.log(wc["duration_s"][0]), np.log(wc["duration_s"][1]))))
            f = self._contour(rng, d)
        n = len(f)
        if start is None:
            room = n_total - n - int(0.1 * fs)
            start = int(rng.uniform(int(0.1 * fs), max(int(0.1 * fs) + 1, room))) if room > int(0.1 * fs) else int(0.1 * fs)
        start = int(np.clip(start, 0, max(0, n_total - n)))
        env = self._envelope(rng, n)
        snr = self._draw_snr(rng)
        kbar = int(np.clip(round(f.mean() / self.bin_hz), 0, len(L_bg) - 1))
        amp = 10 ** (snr / 20) * L_bg[kbar] / self.sumw_half
        harm = [(1, 0.0)]
        if rng.random() < wc["p_harmonic2"]:
            harm.append((2, float(rng.uniform(*wc["harmonic_rel_db"]))))
        if rng.random() < wc["p_harmonic3"]:
            harm.append((3, float(rng.uniform(*wc["harmonic_rel_db"])) - 5.0))
        phase = 2 * np.pi * np.cumsum(f) / fs
        s = np.zeros(n)
        comps = []
        for h, rel in harm:
            if h * f.max() > 0.45 * fs:
                continue
            g = amp * 10 ** (rel / 20)
            s += g * env * np.sin(h * phase + rng.uniform(0, 6.28))
            comps.append({"wid": wid, "h": h, "f": h * f, "env": env, "start": start, "snr_nominal_db": snr + rel})
        return {"wid": wid, "start": start, "signal": s, "comps": comps, "snr_db": snr, "dur_s": n / fs, "f_mean": float(f.mean())}

    def _draw_whistles(self, rng, N: int, L_bg: np.ndarray) -> list[dict]:
        wc, fs = self.wc, self.fs
        n_w = int(rng.choice(len(wc["p_n_whistles"]), p=np.array(wc["p_n_whistles"]) / np.sum(wc["p_n_whistles"])))
        sc = wc["p_scenario"]
        scenario = rng.choice(["random", "crossing", "parallel"], p=np.array([sc["random"], sc["crossing"], sc["parallel"]]) / (sc["random"] + sc["crossing"] + sc["parallel"])) if n_w >= 2 else "random"
        out: list[dict] = []
        for i in range(n_w):
            if i == 1 and scenario != "random":
                base = out[0]["comps"][0]
                f1 = base["f"]
                if scenario == "crossing":
                    f2 = 2 * f1.mean() - f1 + rng.uniform(-400, 400)
                else:
                    off = rng.uniform(*wc["parallel_offset_bins"]) * self.bin_hz * rng.choice([-1.0, 1.0])
                    f2 = f1 + off
                lo, hi = wc["f_range_hz"]
                if f2.min() < lo or f2.max() > hi:           # fall back to an independent random whistle
                    out.append(self._make_whistle(rng, i, N, None, None, L_bg))
                else:
                    out.append(self._make_whistle(rng, i, N, f2, base["start"] + int(rng.uniform(-0.08, 0.08) * fs), L_bg))
            else:
                out.append(self._make_whistle(rng, i, N, None, None, L_bg))
        return out

    # ------------------------------------------------------------------ clicks / sweeps (negatives)
    def _clicks(self, rng, N: int, rms: float) -> np.ndarray:
        cc, fs = self.cc, self.fs
        out = np.zeros(N)
        if rng.random() < cc["p_click_train"]:
            for _ in range(int(rng.integers(cc["n_trains"][0], cc["n_trains"][1] + 1))):
                n_c = int(rng.integers(cc["n_clicks"][0], cc["n_clicks"][1] + 1))
                ici = rng.uniform(*cc["ici_ms"]) * 1e-3 * fs
                p0 = rng.uniform(0, N * 0.8)
                fc = min(rng.uniform(*cc["center_khz"]) * 1e3, 0.45 * fs)
                tau = rng.uniform(*cc["decay_ms"]) * 1e-3
                tt = np.arange(int(6 * tau * fs)) / fs
                pulse = np.exp(-tt / tau) * np.sin(2 * np.pi * fc * tt)
                lvl = rng.uniform(*cc["level_db"])
                for j in range(n_c):
                    p = int(p0 + j * ici * (1 + 0.05 * rng.standard_normal()))
                    if 0 <= p < N - len(pulse):
                        out[p:p + len(pulse)] += rms * 10 ** (lvl / 20) * rng.uniform(0.5, 1.5) * pulse
        if rng.random() < cc["p_sweep_train"]:
            d = rng.uniform(3, 15) * 1e-3
            tt = np.arange(int(d * fs)) / fs
            f1, f2 = rng.uniform(5e3, 25e3), rng.uniform(15e3, 60e3)
            sweep = sps.windows.hann(len(tt)) * np.sin(2 * np.pi * (f1 * tt + 0.5 * (f2 - f1) / d * tt ** 2))
            period = int(rng.uniform(0.2, 1.0) * fs)
            lvl = rng.uniform(*cc["level_db"])
            for p in range(int(rng.uniform(0, period)), N - len(sweep), period):
                out[p:p + len(sweep)] += rms * 10 ** (lvl / 20) * sweep
        return out

    # ------------------------------------------------------------------ labels
    def _labels(self, whistles: list[dict], W_mag: np.ndarray, Lb: np.ndarray, T: int):
        lc, F = self.lc, self.F
        y = np.zeros((T, F), np.float32)
        ign = np.zeros((T, F), bool)
        kgrid = np.arange(F)[None, :]
        gt = []
        centers = np.arange(T) * self.hop + self.win // 2
        for wh in whistles:
            for c in wh["comps"]:
                n = len(c["f"])
                idx = centers - c["start"]
                ok = (idx >= 0) & (idx < n)
                t_idx = np.where(ok)[0]
                if not len(t_idx):
                    continue
                act = c["env"][idx[ok]] >= lc["min_envelope"]
                t_idx, fr = t_idx[act], c["f"][idx[ok]][act]
                b = fr / self.bin_hz - self.k_lo
                inb = (b >= -0.5) & (b <= F - 0.5)
                if not inb.any():
                    continue
                t_idx, fr, b = t_idx[inb], fr[inb], b[inb]
                kk = np.clip(np.rint(b).astype(int), 0, F - 1)
                snr = 20 * np.log10(W_mag[t_idx, kk] / Lb[kk] + 1e-12)
                vis = snr >= lc["snr_pos_db"]
                G = np.exp(-0.5 * ((kgrid - b[:, None]) / lc["sigma_bins"]) ** 2).astype(np.float32)
                if vis.any():
                    y[t_idx[vis]] = np.maximum(y[t_idx[vis]], G[vis])
                if (~vis).any():
                    near = np.abs(kgrid - b[~vis][:, None]) <= lc["ignore_radius_bins"]
                    ign[t_idx[~vis]] |= near
                gt.append({"wid": c["wid"], "h": c["h"], "frames": t_idx, "f_hz": fr, "snr_db": snr, "visible": vis})
        w = np.ones((T, F), np.float32)
        w[ign & (y < 0.3)] = 0.0
        return y, w, gt

    # ------------------------------------------------------------------ one sample
    @staticmethod
    def _phase_randomise(x: np.ndarray, rng) -> np.ndarray:
        """Same long-term spectrum, no ridge/click structure (used for CONTAMINATION-FREE negative evaluation only)."""
        X = np.fft.rfft(x)
        ph = rng.uniform(0, 2 * np.pi, len(X))
        ph[0] = 0.0
        if len(x) % 2 == 0:
            ph[-1] = 0.0
        return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x))

    def sample(self, index: int, epoch: int = 0, surrogate: bool = False, no_whistles: bool = False) -> dict:
        rng = np.random.default_rng([self.seed, self.split_id, int(epoch), int(index)])
        widx = int(rng.integers(len(self.windows)))
        bg = self.windows[widx].astype(np.float64)
        rmask = None if self.ridge_masks is None else np.asarray(self.ridge_masks[widx])
        p_rev = float(self.dcfg["synth"]["backgrounds"].get("p_time_reverse", 0.0)) if self.split_id == 0 else 0.0
        if p_rev > 0 and rng.random() < p_rev:            # training-only augmentation (no extra RNG draw when disabled)
            bg = bg[::-1].copy()
            rmask = None if rmask is None else rmask[::-1]   # the line statistic is symmetric under time reversal
        if surrogate:
            bg = self._phase_randomise(bg, rng)
        N = len(bg)
        bg_db_full = self._stft_db(bg)
        B = 10 ** (bg_db_full / 20)
        L_bg = ndimage.uniform_filter1d(np.median(B, axis=0), size=7, mode="nearest")
        whistles = [] if no_whistles else self._draw_whistles(rng, N, L_bg)
        sig_w = np.zeros(N)
        for wh in whistles:
            n = len(wh["signal"])
            sig_w[wh["start"]:wh["start"] + n] += wh["signal"][: N - wh["start"]]
        x = bg + sig_w + self._clicks(rng, N, float(np.sqrt(np.mean(bg ** 2))))
        sl = slice(self.k_lo, self.k_hi + 1)
        mix_db = self._stft_db(x)[:, sl]
        T = mix_db.shape[0]
        W_mag = 10 ** (self._stft_db(sig_w)[:, sl] / 20) if whistles else np.zeros((T, self.F))
        y, w, gt = self._labels(whistles, W_mag, L_bg[sl], T)
        n_bg_ignored = 0
        if self.lc.get("ignore_background_ridges", False) and rmask is not None and not surrogate:
            bgm = rmask[: y.shape[0]] & (y < 0.3)
            w[bgm] = 0.0
            n_bg_ignored = int(bgm.sum())
        return {"x": condition(mix_db, self.cond), "y": y, "w": w, "gt": gt,
                "meta": {"index": int(index), "n_whistles": len(whistles), "snr_db": [round(wh["snr_db"], 1) for wh in whistles],
                         "dur_s": [round(wh["dur_s"], 3) for wh in whistles], "n_bg_ignored": n_bg_ignored}}
