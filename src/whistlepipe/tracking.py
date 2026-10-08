"""S6: replacement post-processing / tracking (converts the S5 confidence map into whistle tracks).

REPLACEMENT BEHAVIOUR: the paper's S6 (its refs [32], [33]) is not available / only partially described. Nothing here
is claimed to reproduce it. Rules and numeric settings are IMPLEMENTATION DECISIONS in cfg['decisions']['s6'].

Algorithm
  1. per-frame ridge peaks: local maxima along frequency with confidence >= peak_threshold, sub-bin refined
     (3-point centroid); frames where too many nodes are active are treated as broadband (clicks) and skipped;
  2. frame-to-frame linking: each active track predicts its next bin by a linear fit of its last observed points;
     peaks are assigned by optimal (Hungarian) assignment inside a gate that grows while the track coasts;
     unmatched tracks coast for up to max_gap_frames (this also carries two tracks through a crossing);
  3. rejection with recorded reasons: too short, mostly interpolated, low confidence, implausibly steep,
     duplicates of a stronger track, harmonics of a lower track.
All steps are deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import Cfg
from .detector.interface import DetectionMap

_BIG = 1e6


@dataclass
class Track:
    id: int
    frames: np.ndarray      # consecutive S4 frame indices first..last
    freq_hz: np.ndarray     # frequency per frame (Hz); linearly interpolated across coasted frames
    conf: np.ndarray        # confidence per frame (0 on coasted frames)
    observed: np.ndarray    # bool, True where a detector peak supported the point


@dataclass
class TrackingResult:
    tracks: list[Track]
    rejected: list[dict]
    stats: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- peaks
def extract_peaks(conf: np.ndarray, c: dict) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    t, f = conf.shape
    tau = float(c["peak_threshold"])
    left = np.pad(conf[:, :-1], ((0, 0), (1, 0)), constant_values=-1.0)
    right = np.pad(conf[:, 1:], ((0, 0), (0, 1)), constant_values=-1.0)
    is_pk = (conf >= tau) & (conf >= left) & (conf > right)
    n_above = (conf >= tau).sum(axis=1)
    broad = n_above > float(c["broadband_frame_fraction"]) * f
    is_pk[broad] = False
    ti, ki = np.nonzero(is_pk)
    a0 = conf[ti, ki]
    am = np.where(ki > 0, conf[ti, np.maximum(ki - 1, 0)], 0.0)
    ap = np.where(ki < f - 1, conf[ti, np.minimum(ki + 1, f - 1)], 0.0)
    kf = ki + (ap - am) / (am + a0 + ap + 1e-9)
    bounds = np.searchsorted(ti, np.arange(t + 1))
    sep = float(c["min_peak_separation_bins"])
    pk_k, pk_c = [], []
    for i in range(t):
        k, v = kf[bounds[i]:bounds[i + 1]], a0[bounds[i]:bounds[i + 1]]
        if len(k) > 1 and np.min(np.diff(k)) < sep:          # merge close peaks, keep the stronger
            keep = []
            for j in np.argsort(-v, kind="stable"):
                if all(abs(k[j] - k[m]) >= sep for m in keep):
                    keep.append(j)
            keep = np.array(sorted(keep))
            k, v = k[keep], v[keep]
        pk_k.append(k); pk_c.append(v)
    return pk_k, pk_c, {"n_peaks": int(sum(len(k) for k in pk_k)), "n_broadband_frames": int(broad.sum())}


# ----------------------------------------------------------------------------- linking
class _Active:
    __slots__ = ("frames", "bins", "confs", "missed", "id")

    def __init__(self, tid: int, t: int, k: float, v: float):
        self.id, self.frames, self.bins, self.confs, self.missed = tid, [t], [k], [v], 0

    def predict(self, t: int, npts: int) -> float:
        n = min(len(self.frames), npts)
        if n == 1:
            return self.bins[-1]
        fr = np.asarray(self.frames[-n:], float)
        b = np.asarray(self.bins[-n:], float)
        fm, bm = fr.mean(), b.mean()
        den = ((fr - fm) ** 2).sum()
        slope = ((fr - fm) * (b - bm)).sum() / den if den > 0 else 0.0
        return bm + slope * (t - fm)


def link_peaks(pk_k: list[np.ndarray], pk_c: list[np.ndarray], c: dict) -> list[_Active]:
    max_gap = int(c["max_gap_frames"])
    npts = int(c["predict_points"])
    gb, gg, gmax = float(c["gate_base_bins"]), float(c["gate_growth_per_miss"]), float(c["gate_max_bins"])
    active: list[_Active] = []
    done: list[_Active] = []
    next_id = 0
    for t in range(len(pk_k)):
        k, v = pk_k[t], pk_c[t]
        matched_tr: set[int] = set()
        matched_pk: set[int] = set()
        if active and len(k):
            pred = np.array([tr.predict(t, npts) for tr in active])
            gate = np.array([min(gmax, gb + gg * tr.missed) for tr in active])
            d = np.abs(k[None, :] - pred[:, None])
            cost = d / gate[:, None] + 0.3 * (1.0 - v[None, :])
            cost[d > gate[:, None]] = _BIG
            ri, ci = linear_sum_assignment(cost)
            for r, cidx in zip(ri, ci):
                if cost[r, cidx] < _BIG:
                    tr = active[r]
                    tr.frames.append(t); tr.bins.append(float(k[cidx])); tr.confs.append(float(v[cidx])); tr.missed = 0
                    matched_tr.add(r); matched_pk.add(cidx)
        still = []
        for i, tr in enumerate(active):
            if i not in matched_tr:
                tr.missed += 1
            (still if tr.missed <= max_gap else done).append(tr)
        active = still
        for j in range(len(k)):
            if j not in matched_pk:
                active.append(_Active(next_id, t, float(k[j]), float(v[j])))
                next_id += 1
    return done + active


# ----------------------------------------------------------------------------- filtering
def _materialise(tr: _Active, bin_lo: int, bin_hz: float) -> Track:
    fr = np.asarray(tr.frames)
    full = np.arange(fr[0], fr[-1] + 1)
    bins = np.interp(full, fr, np.asarray(tr.bins))
    conf = np.zeros(len(full)); conf[fr - fr[0]] = tr.confs
    obs = np.zeros(len(full), bool); obs[fr - fr[0]] = True
    return Track(tr.id, full, (bins + bin_lo) * bin_hz, conf, obs)


def _overlap(a: Track, b: Track):
    lo, hi = max(a.frames[0], b.frames[0]), min(a.frames[-1], b.frames[-1])
    if hi < lo:
        return None
    ia, ib = slice(lo - a.frames[0], hi - a.frames[0] + 1), slice(lo - b.frames[0], hi - b.frames[0] + 1)
    return ia, ib, hi - lo + 1


def filter_tracks(raw: list[Track], c: dict, bin_hz: float) -> tuple[list[Track], list[dict]]:
    rejected: list[dict] = []

    def rej(t: Track, reason: str, **kw):
        rejected.append({"id": int(t.id), "reason": reason, "start_frame": int(t.frames[0]), "span": int(len(t.frames)),
                         "mean_freq_hz": float(t.freq_hz.mean()), **kw})

    keep: list[Track] = []
    for t in raw:
        span, n_obs = len(t.frames), int(t.observed.sum())
        slope = float(np.abs(np.diff(t.freq_hz)).mean() / bin_hz) if span > 1 else 0.0
        if span < int(c["min_track_frames"]):
            rej(t, "too_short")
        elif n_obs / span < float(c["min_observed_fraction"]):
            rej(t, "low_observed_fraction", observed_fraction=n_obs / span)
        elif float(t.conf[t.observed].mean()) < float(c["min_mean_conf"]):
            rej(t, "low_confidence", mean_conf=float(t.conf[t.observed].mean()))
        elif slope > float(c["max_mean_slope_bins_per_frame"]):
            rej(t, "too_steep", slope_bins_per_frame=slope)
        else:
            keep.append(t)

    strength = lambda t: float(t.conf[t.observed].sum())
    keep.sort(key=lambda t: (-strength(t), t.id))

    if c["suppress_duplicates"]:
        kept2: list[Track] = []
        tol, minov = float(c["duplicate_tol_bins"]) * bin_hz, float(c["duplicate_min_overlap"])
        for t in keep:
            dup = None
            for a in kept2:
                o = _overlap(a, t)
                if o is None:
                    continue
                ia, ib, n = o
                if n >= minov * len(t.frames) and np.mean(np.abs(a.freq_hz[ia] - t.freq_hz[ib]) <= tol) >= 0.8:
                    dup = a
                    break
            if dup is None:
                kept2.append(t)
            else:
                rej(t, "duplicate", of=int(dup.id))
        keep = kept2

    if c.get("suppress_synchronous_combs", False) and len(keep) >= int(c["comb_min_tracks"]):
        # >= comb_min_tracks tracks that start within comb_start_tol frames of each other AND have nearly equal spans cannot be
        # independent whistles: they are the lattice of a pulse-train (burst-pulse) or a saturated broadband burst. Found in Phase 3.
        st0 = np.array([t.frames[0] for t in keep]); sp0 = np.array([len(t.frames) for t in keep])
        M = (np.abs(st0[:, None] - st0[None, :]) <= int(c["comb_start_tol_frames"])) &             (np.abs(sp0[:, None] - sp0[None, :]) <= int(c["comb_span_tol_frames"]))
        flag = M.sum(axis=1) >= int(c["comb_min_tracks"])
        kept_c = []
        for t, f in zip(keep, flag):
            if f:
                rej(t, "synchronous_comb", n_synchronous=int(M[keep.index(t)].sum()))
            else:
                kept_c.append(t)
        keep = kept_c

    if c["suppress_harmonics"]:
        orders = [int(h) for h in c["harmonic_orders"]]
        rtol, minov = float(c["harmonic_rel_tol"]), float(c["harmonic_min_overlap"])
        by_freq = sorted(keep, key=lambda t: float(t.freq_hz.mean()))
        harm_ids: dict[int, int] = {}
        for i, b in enumerate(by_freq):
            for a in by_freq[:i]:
                if a.freq_hz.mean() >= b.freq_hz.mean():
                    continue
                o = _overlap(a, b)
                if o is None:
                    continue
                ia, ib, n = o
                if n < minov * len(b.frames):
                    continue
                r = b.freq_hz[ib] / a.freq_hz[ia]
                if any(np.mean(np.abs(r - h) <= rtol * h) >= 0.8 for h in orders):
                    harm_ids[b.id] = a.id
                    break
        kept3 = []
        for t in keep:
            if t.id in harm_ids:
                rej(t, "harmonic", of=int(harm_ids[t.id]))
            else:
                kept3.append(t)
        keep = kept3
    keep.sort(key=lambda t: (int(t.frames[0]), t.id))
    return keep, rejected


# ----------------------------------------------------------------------------- stage entry
def track_confidence_map(conf: np.ndarray, bin_lo: int, cfg: Cfg) -> TrackingResult:
    c = cfg["decisions"]["s6"]
    bin_hz = float(cfg["paper"]["freq_resolution_hz"])
    pk_k, pk_c, pstats = extract_peaks(conf, c)
    raw_active = link_peaks(pk_k, pk_c, c)
    raw = [_materialise(tr, bin_lo, bin_hz) for tr in raw_active if len(tr.frames) >= 2]
    keep, rejected = filter_tracks(raw, c, bin_hz)
    by_reason: dict[str, int] = {}
    for r in rejected:
        by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
    stats = {**pstats, "n_raw_tracks": len(raw), "n_single_peak_tracks_dropped": len(raw_active) - len(raw),
             "rejected_by_reason": by_reason}
    return TrackingResult(keep, rejected, stats)


def run_s6(dm: DetectionMap, cfg: Cfg) -> TrackingResult:
    return track_confidence_map(dm.conf, dm.bin_lo, cfg)
