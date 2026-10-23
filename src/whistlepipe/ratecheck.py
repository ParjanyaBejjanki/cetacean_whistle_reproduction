"""Does the detector work on uploads at ordinary sample rates?  (python whistle.py check-rates)

Each real 500 kHz file is processed as the dataset was (S1..S7: the reference tracks), then again from a lower-rate
copy of the same channel (resampled with scipy's polyphase filter, rounded to 16-bit like a real file) entering at
S2 the way an upload does. Tracks are matched one-to-one: time overlap >= half the shorter track and median
|frequency difference| over the overlap <= 2 bins (187.5 Hz) [DECISION]. Reference tracks reaching above 0.45 x the
lower rate cannot survive the lower rate at all; recall is reported with and without them.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import signal

from . import audio
from .config import load_pipeline_config
from .infer import load_inference_config, preprocess, resolve


def match_tracks(ref: list, test: list, max_df_hz: float = 187.5, min_overlap: float = 0.5) -> list[tuple[int, int]]:
    cand = []
    for i, a in enumerate(ref):
        a0, a1 = a.start_frame, a.start_frame + a.n_frames
        for j, b in enumerate(test):
            b0, b1 = b.start_frame, b.start_frame + b.n_frames
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo < min_overlap * min(a.n_frames, b.n_frames):
                continue
            df = np.median(np.abs(a.freq_hz[lo - a0:hi - a0] - b.freq_hz[lo - b0:hi - b0]))
            if df <= max_df_hz:
                cand.append((hi - lo, -df, i, j))
    used_i, used_j, out = set(), set(), []
    for _, _, i, j in sorted(cand, reverse=True):
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            out.append((i, j))
    return out


def check_rates(files: list[Path], rates: list[int], device: str | None = None, progress=None) -> dict:
    from .detector import RidgeDetector

    say = progress or (lambda *a: None)
    pcfg = load_pipeline_config()
    det = RidgeDetector.from_checkpoint(resolve(pcfg["decisions"]["s5"]["checkpoint"]), pcfg, device)
    per_file, tot = [], {r: {"ref": 0, "test": 0, "matched": 0, "ref_reachable": 0, "matched_reachable": 0,
                             "images_ref": 0, "images_test": 0} for r in rates}
    for f in files:
        s1 = audio.s1_select_channel(f, pcfg)
        ref = preprocess(s1, pcfg, det)
        row = {"file": f.name, "ref_tracks": len(ref["vectors"]), "ref_images": len(ref["images"]), "rates": {}}
        for rate in rates:
            g = math.gcd(int(rate), int(s1.fs))
            y = signal.resample_poly(s1.samples.astype(np.float64), rate // g, int(s1.fs) // g)
            y = (np.clip(np.round(y * 32768), -32768, 32767) / 32768).astype(np.float32)
            test = preprocess(audio.AudioSignal(y, float(rate), f.name), pcfg, det)
            m = match_tracks(ref["vectors"], test["vectors"])
            reach = [i for i, v in enumerate(ref["vectors"]) if v.f_max <= 0.45 * rate]
            mi = {i for i, _ in m}
            r = {"test_tracks": len(test["vectors"]), "matched": len(m), "test_images": len(test["images"]),
                 "ref_reachable": len(reach), "matched_reachable": len(mi.intersection(reach))}
            row["rates"][str(rate)] = r
            t = tot[rate]
            t["ref"] += len(ref["vectors"])
            t["test"] += len(test["vectors"])
            t["matched"] += len(m)
            t["ref_reachable"] += len(reach)
            t["matched_reachable"] += r["matched_reachable"]
            t["images_ref"] += len(ref["images"])
            t["images_test"] += len(test["images"])
        per_file.append(row)
        say(json.dumps(row))
    summary = {}
    for rate, t in tot.items():
        summary[str(rate)] = {**t, "recall": round(t["matched"] / max(1, t["ref"]), 4),
                              "recall_reachable": round(t["matched_reachable"] / max(1, t["ref_reachable"]), 4),
                              "precision": round(t["matched"] / max(1, t["test"]), 4)}
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": [f.name for f in files],
           "rates": rates, "matching": {"max_df_hz": 187.5, "min_overlap": 0.5}, "summary": summary, "per_file": per_file}
    d = resolve(load_inference_config()["output"]["out_dir"]) / "_checks"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"rates_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    out["path"] = str(p)
    return out
