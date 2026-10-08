"""Evaluation of the S5 detector + S6 tracker on SYNTHETIC data only (ground truth comes from the generator's
construction process). This is evaluation category A (synthetic). It is NOT a real-data precision/recall.

Node level : precision/recall on S5 outputs against the synthetic ridge targets (weight>0 nodes).
Track level: after S6, a ground-truth fundamental is recovered if one track covers >= 60 % of its visible frames
             within 1.5 bins; a track is true if >= 60 % of its frames lie within 1.5 bins of some synthetic contour.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .config import Cfg, band_bins, project_path
from .detector.ridgenet import RidgeDetector
from .synth.dataset import build_eval_set, build_surrogate_negative_set
from .tracking import Track, track_confidence_map

SNR_BUCKETS = [(-5, 5), (5, 10), (10, 15), (15, 40)]
TOL_BINS = 1.5
MIN_EXPECTED_VISIBLE_FRAMES = 15
COVER = 0.6


def node_curve(preds: list[np.ndarray], ev: dict, n_thr: int = 101) -> dict:
    """Node-level PR curve by histogramming predictions (positive: y>=0.3; negative: y<0.1; weight>0 only)."""
    edges = np.linspace(0, 1, n_thr)
    hp = np.zeros(n_thr - 1); hn = np.zeros(n_thr - 1)
    for p, y, w in zip(preds, ev["y"], ev["w"]):
        y = y.astype(np.float32) / 255.0
        m = w > 0
        hp += np.histogram(p[m & (y >= 0.3)], bins=edges)[0]
        hn += np.histogram(p[m & (y < 0.1)], bins=edges)[0]
    tp = np.cumsum(hp[::-1])[::-1]; fp = np.cumsum(hn[::-1])[::-1]
    P = hp.sum()
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P, 1)
    ap = float(np.sum((rec[:-1] - rec[1:]) * prec[:-1]))
    return {"thresholds": edges[:-1].tolist(), "precision": prec.tolist(), "recall": rec.tolist(), "average_precision": ap,
            "n_pos_nodes": int(P), "n_neg_nodes": int(hn.sum())}


def _dense(track_or_gt_frames, f_hz, T):
    a = np.full(T, np.nan)
    a[track_or_gt_frames] = f_hz
    return a


def track_metrics(conf_list: list[np.ndarray], ev: dict, cfg: Cfg, tau: float) -> dict:
    c2 = Cfg(copy.deepcopy(dict(cfg)))
    c2["decisions"]["s6"]["peak_threshold"] = tau
    lo, _ = band_bins(cfg)
    tol = TOL_BINS * 93.75
    expected = matched = n_tracks = true_tracks = harm_leak = frag_sum = 0
    by_snr = {b: [0, 0] for b in SNR_BUCKETS}
    bg_only_frames = 0; bg_only_false = 0
    for conf, gts, meta in zip(conf_list, ev["gt"], ev["meta"]):
        T = conf.shape[0]
        tr = track_confidence_map(conf, lo, c2).tracks
        tdense = [_dense(t.frames, t.freq_hz, T) for t in tr]
        gdense = [(_dense(g["frames"], g["f_hz"], T), g) for g in gts]
        # recall over visible fundamentals
        for gd, g in gdense:
            vis = g["frames"][g["visible"]]
            if g["h"] != 1 or len(vis) < MIN_EXPECTED_VISIBLE_FRAMES:
                continue
            expected += 1
            snr = float(np.median(g["snr_db"][g["visible"]]))
            hits = [float(np.mean(np.abs(td[vis] - gd[vis]) <= tol)) for td in tdense]   # nan compares False
            if hits and max(hits) >= COVER:
                matched += 1
                frag_sum += int(sum(h >= 0.2 for h in hits))
                for b in SNR_BUCKETS:
                    if b[0] <= snr < b[1]:
                        by_snr[b][0] += 1; by_snr[b][1] += 1
            else:
                for b in SNR_BUCKETS:
                    if b[0] <= snr < b[1]:
                        by_snr[b][1] += 1
        # precision over tracks
        for t, td in zip(tr, tdense):
            n_tracks += 1
            ok_any = False; ok_h1 = False
            for gd, g in gdense:
                frac = float(np.mean(np.abs(td[t.frames] - gd[t.frames]) <= tol))
                if frac >= COVER:
                    ok_any = True
                    ok_h1 |= (g["h"] == 1)
            true_tracks += int(ok_any)
            harm_leak += int(ok_any and not ok_h1)
        if meta["n_whistles"] == 0:
            bg_only_frames += T; bg_only_false += len(tr)
    hop_s = 1024 / 192000
    return {"tau": tau, "expected_fundamentals": expected, "recalled": matched,
            "track_recall": matched / max(1, expected), "n_tracks": n_tracks, "true_tracks": true_tracks,
            "track_precision": true_tracks / max(1, n_tracks), "harmonic_tracks_leaked": harm_leak,
            "tracks_per_recalled_whistle": frag_sum / max(1, matched),
            "false_tracks_per_min_on_whistle_free_samples": bg_only_false / max(1e-9, bg_only_frames * hop_s / 60),
            "recall_by_snr_db": {f"[{b[0]},{b[1]})": {"recalled": v[0], "expected": v[1], "recall": v[0] / max(1, v[1])} for b, v in by_snr.items()}}


def evaluate_on_synthetic(cfg: Cfg, dcfg: Cfg, version: str, split: str = "test", calibrate: bool = False) -> dict:
    mdir = project_path(cfg, "models_dir") / "detector" / version
    det = RidgeDetector.from_checkpoint(mdir / "checkpoint.pt", cfg)
    n = int(dcfg["train"]["val_samples" if split == "val" else "test_samples"])
    ev = build_eval_set(cfg, dcfg, split, n)
    conf_list = [det.predict_conditioned(x.astype(np.float32)) for x in ev["x"]]
    out = {"split": split, "n_samples": n, "category": "A: SYNTHETIC evaluation (ground truth from the generator)",
           "detector": {"version": version, **det.meta}, "node": node_curve(conf_list, ev)}
    grid = [round(x, 2) for x in np.arange(0.30, 0.85, 0.05)]
    out["track_level_by_tau"] = [track_metrics(conf_list, ev, cfg, tau) for tau in grid]
    # contamination-free false-alarm rate: phase-randomised real backgrounds + synthetic clicks/sweeps (no real whistle can remain)
    neg = build_surrogate_negative_set(cfg, dcfg, split, 200)
    neg_conf = [det.predict_conditioned(x.astype(np.float32)) for x in neg["x"]]
    neg_min = len(neg_conf) * neg["crop"] * 1024 / 192000 / 60
    lo, _ = band_bins(cfg)
    for m in out["track_level_by_tau"]:
        c2 = Cfg(copy.deepcopy(dict(cfg))); c2["decisions"]["s6"]["peak_threshold"] = m["tau"]
        n_false = sum(len(track_confidence_map(c, lo, c2).tracks) for c in neg_conf)
        m["false_tracks_per_min_surrogate_negative"] = n_false / neg_min
    out["surrogate_negative"] = {"n_samples": len(neg_conf), "minutes": neg_min,
                                 "max_node_confidence": float(max(c.max() for c in neg_conf)),
                                 "note": "phase-randomised backgrounds + synthetic clicks/sweeps; contamination-free"}
    out["note_precision"] = ("track_precision on REAL-background samples is a LOWER BOUND: unlabeled real tonal structure that survives the "
                             "classical background screen counts as a false track. Use false_tracks_per_min_surrogate_negative for the "
                             "detector's own false-alarm behaviour on noise and clicks.")
    cur_tau = float(cfg["decisions"]["s6"]["peak_threshold"])
    out["track_level_at_config_tau"] = next((m for m in out["track_level_by_tau"] if abs(m["tau"] - cur_tau) < 1e-9), None) or track_metrics(conf_list, ev, cfg, cur_tau)
    if calibrate:
        feas = [m for m in out["track_level_by_tau"] if m["false_tracks_per_min_surrogate_negative"] <= 2.0]
        best_rec = max(m["track_recall"] for m in feas)
        near = [m for m in feas if m["track_recall"] >= best_rec - 0.01]
        pick = min(near, key=lambda m: (abs(m["tau"] - 0.5), m["tau"]))
        spread = max(m["track_recall"] for m in out["track_level_by_tau"]) - min(m["track_recall"] for m in out["track_level_by_tau"])
        cal = {"split_used": split, "tau": pick["tau"],
               "criterion": "among thresholds with <= 2 false tracks/min on the contamination-free surrogate negatives and track recall within 0.01 of the best, take the one closest to 0.5",
               "track_recall": pick["track_recall"], "false_tracks_per_min_surrogate_negative": pick["false_tracks_per_min_surrogate_negative"],
               "recall_spread_over_grid": spread,
               "note": "Synthetic validation is nearly flat in tau, so the data do not select a precise value; 0.5 is the natural decision boundary of the BCE-trained confidence. "
                       "Copy into configs/pipeline.toml decisions.s6.peak_threshold ONCE, before the real-data phases. Not tuned on real data and not tuned to any paper count."}
        (mdir / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
        out["calibration"] = cal
        print(json.dumps(cal, indent=2))
    (mdir / f"eval_{split}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    m = out["track_level_at_config_tau"]
    print(f"[{split}] node AP={out['node']['average_precision']:.3f} | tau={m['tau']}: track recall={m['track_recall']:.3f} "
          f"precision={m['track_precision']:.3f} tracks/whistle={m['tracks_per_recalled_whistle']:.2f} "
          f"false/min(real whistle-free bg, contaminated)={m['false_tracks_per_min_on_whistle_free_samples']:.2f} "
          f"false/min(surrogate neg)={m.get('false_tracks_per_min_surrogate_negative', float('nan')):.2f} harm-leak={m['harmonic_tracks_leaked']}")
    return out
