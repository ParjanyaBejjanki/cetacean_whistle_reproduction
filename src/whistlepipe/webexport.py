"""Example recordings for the page (python whistle.py export-web): short excerpts of real dataset files plus their
pre-rendered results, so visitors without a hydrophone recording can try the demo, and the page works on a static
host without the API server.

    web/data/examples.json                    index the page reads
    web/data/examples/<id>.flac               the excerpt (channel 0, resampled 500 kHz -> [examples].rate_hz)
    web/data/examples/<id>/result.json ...    the analysis of that excerpt with the model files present at export time

[DECISION] one excerpt per configured encounter: the [examples].seconds window, within one raw file, holding the most
training whistles of 20-128 frames up to 20 (busier windows are hard to read), ties broken by detector confidence.
Re-run after training so the pre-rendered results include the model's output. web/data/ is generated (gitignored).
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
from scipy import signal

from . import audio
from .infer import Engine, resolve

WEB_DATA = resolve("web") / "data"


def pick_window(rows: list[dict], seconds: float, cap: int = 20) -> tuple[str, float, int]:
    by = defaultdict(list)
    for r in rows:
        if 20 <= int(r["n_frames"]) <= 128:
            by[r["source_file"]].append((float(r["start_s"]), float(r["end_s"]), float(r["mean_conf"])))
    best = None
    for f, ws in by.items():
        ws.sort()
        for t, _, _ in ws:
            inside = [w for w in ws if w[0] >= t and w[1] <= t + seconds - 0.3]
            score = (min(len(inside), cap), float(np.mean([w[2] for w in inside])) if inside else 0.0)
            if best is None or score > best[0]:
                best = (score, f, max(0.0, t - 0.25), len(inside))
    return best[1], best[2], best[3]


def export_examples(device: str | None = None, progress=None) -> dict:
    say = progress or (lambda *a: None)
    eng = Engine(device=device)
    c = eng.icfg["examples"]
    root = WEB_DATA / "examples"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    out = []
    for enc in c["encounters"]:
        csv_path = resolve(eng.pcfg["paths"]["dataset_dir"]) / f"encounter_{enc}" / "images.csv"
        with open(csv_path, encoding="utf-8") as fh:
            src, t0, n_in = pick_window(list(csv.DictReader(fh)), float(c["seconds"]))
        s1 = audio.s1_select_channel(audio.find_raw_file(eng.pcfg, src), eng.pcfg)
        fs, rate = int(s1.fs), int(c["rate_hz"])
        x = s1.samples[int(t0 * fs): int((t0 + float(c["seconds"])) * fs)].astype(np.float64)
        g = math.gcd(rate, fs)
        y = signal.resample_poly(x, rate // g, fs // g)
        eid = f"encounter-{enc}"
        wav = root / f"{eid}.flac"
        sf.write(str(wav), np.clip(y, -1, 1), rate, subtype="PCM_16", format="FLAC")
        r = eng.analyze(wav, root / eid, seed=0)
        d = src.split("_")
        when = f"{d[1][:4]}-{d[1][4:6]}-{d[1][6:]} {d[2][:2]}:{d[2][2:4]} UTC"
        out.append({"id": eid, "title": f"Encounter {enc} · {when}", "encounter": enc, "source_file": src,
                    "start_s": round(t0, 3), "seconds": float(c["seconds"]), "fs_hz": rate,
                    "audio": f"data/examples/{eid}.flac", "result": f"data/examples/{eid}/result.json",
                    "n_whistles": len(r["whistles"]), "n_training_whistles_in_window": n_in})
        say(f"{eid}: {src} from {t0:.2f} s, {len(r['whistles'])} whistles")
    index = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": eng.model.status["message"],
             "examples": out}
    (WEB_DATA / "examples.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index
