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
    ap = float(np.sum((rec - np.append(rec[1:], 0.0)) * prec))      # step-wise AP: every threshold bin contributes (incl. the highest)
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


# ----------------------------------------------------------------------------- model comparison (synthetic, paired)
def _per_whistle_hits(conf_list, ev, cfg) -> list[tuple[int, int, float, bool]]:
    """(sample, gt index, median SNR, recovered?) for every expected fundamental, using the frozen S6 config."""
    lo, _ = band_bins(cfg)
    tol = TOL_BINS * 93.75
    out = []
    for si, (conf, gts) in enumerate(zip(conf_list, ev["gt"])):
        T = conf.shape[0]
        tdense = [_dense(t.frames, t.freq_hz, T) for t in track_confidence_map(conf, lo, cfg).tracks]
        for gi, g in enumerate(gts):
            vis = g["frames"][g["visible"]]
            if g["h"] != 1 or len(vis) < MIN_EXPECTED_VISIBLE_FRAMES:
                continue
            gd = _dense(g["frames"], g["f_hz"], T)
            hits = [float(np.mean(np.abs(td[vis] - gd[vis]) <= tol)) for td in tdense]
            out.append((si, gi, float(np.median(g["snr_db"][g["visible"]])), bool(hits and max(hits) >= COVER)))
    return out


def _latency(det, spec_db_path: Path, reps: int = 5) -> float | None:
    import time

    import torch

    from .dsp import Spectrogram

    if spec_db_path is None or not Path(spec_db_path).exists():
        return None
    spec = Spectrogram(np.load(spec_db_path), 192000.0, 2048, 1024, 2048, "bench")
    det.predict(spec)
    if det.device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); det.predict(spec)
        if det.device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def compare_on_synthetic(cfg: Cfg, dcfg: Cfg, versions: list[str], split: str = "val", bench_spec: Path | None = None) -> dict:
    """Paired comparison of detector versions on the SAME fixed synthetic set with the SAME frozen S6. versions[0] is the baseline."""
    from scipy.stats import binomtest

    n = int(dcfg["train"]["val_samples" if split == "val" else "test_samples"])
    ev = build_eval_set(cfg, dcfg, split, n)
    neg = build_surrogate_negative_set(cfg, dcfg, split, 200)
    neg_min = len(neg["x"]) * neg["crop"] * 1024 / 192000 / 60
    lo, _ = band_bins(cfg)
    res, hits = {}, {}
    for v in versions:
        ck = project_path(cfg, "models_dir") / "detector" / v / "checkpoint.pt"
        import torch

        arch = torch.load(ck, map_location="cpu", weights_only=False)["arch"]
        from .detector.ridgenet import RidgeNet

        rt = RidgeNet(arch["channels"], arch["dilations"], arch.get("block_type", "dense")).receptive_radius()[0]
        c2 = Cfg(copy.deepcopy(dict(cfg)))
        c2["decisions"]["s5"]["halo_frames"] = max(int(cfg["decisions"]["s5"]["halo_frames"]), int(np.ceil(rt / 16) * 16))
        det = RidgeDetector.from_checkpoint(ck, c2)
        conf_list = [det.predict_conditioned(x.astype(np.float32)) for x in ev["x"]]
        h = _per_whistle_hits(conf_list, ev, cfg)
        hits[v] = h
        nfa = sum(len(track_confidence_map(det.predict_conditioned(x.astype(np.float32)), lo, cfg).tracks) for x in neg["x"])
        by = {}
        for b in SNR_BUCKETS:
            sel = [x[3] for x in h if b[0] <= x[2] < b[1]]
            by[f"[{b[0]},{b[1]})"] = round(float(np.mean(sel)), 3) if sel else None
        gpu = _latency(det, bench_spec)
        cpu_det = RidgeDetector.from_checkpoint(ck, c2, "cpu")
        res[v] = {"n_params": det.meta["n_params"], "macs_per_node": det.meta.get("macs_per_node"), "receptive_radius_tf": det.meta["receptive_radius_tf"],
                  "halo_frames": c2["decisions"]["s5"]["halo_frames"], "node_ap": node_curve(conf_list, ev)["average_precision"],
                  "track_recall": float(np.mean([x[3] for x in h])), "recall_by_snr_db": by,
                  "false_tracks_per_min_surrogate_negative": nfa / neg_min,
                  "s5_seconds_per_60s_file_gpu": gpu, "s5_seconds_per_60s_file_cpu": _latency(cpu_det, bench_spec, reps=3)}
    base = versions[0]
    for v in versions[1:]:
        a = {(x[0], x[1]): x[3] for x in hits[base]}; b = {(x[0], x[1]): x[3] for x in hits[v]}
        wins = sum(1 for k in a if b[k] and not a[k]); losses = sum(1 for k in a if a[k] and not b[k])
        p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
        res[v]["paired_vs_" + base] = {"recovered_only_by_new": wins, "recovered_only_by_base": losses, "mcnemar_exact_p": float(p)}
        for key in ("s5_seconds_per_60s_file_gpu", "s5_seconds_per_60s_file_cpu"):
            if res[v][key] and res[base][key]:
                res[v][key.replace("seconds", "time_change") + "_pct"] = round(100 * (res[v][key] / res[base][key] - 1), 1)
    out = {"split": split, "category": "A: SYNTHETIC (paired, same samples, same frozen S6)", "baseline": base, "results": res}
    p = project_path(cfg, "models_dir") / "detector" / f"compare_{split}_{'_vs_'.join(versions)}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for v in versions:
        r = res[v]
        pv = r.get("paired_vs_" + base, {})
        print(f"{v:14s} params {r['n_params']:6d} RF {r['receptive_radius_tf']} | AP {r['node_ap']:.3f} | recall {r['track_recall']:.3f} by SNR {r['recall_by_snr_db']} | "
              f"FA {r['false_tracks_per_min_surrogate_negative']:.2f}/min | S5 GPU {r['s5_seconds_per_60s_file_gpu'] or float('nan'):.3f}s CPU {r['s5_seconds_per_60s_file_cpu'] or float('nan'):.3f}s"
              + (f" | paired: +{pv['recovered_only_by_new']} / -{pv['recovered_only_by_base']} (p={pv['mcnemar_exact_p']:.2g})" if pv else ""), flush=True)
    return out
