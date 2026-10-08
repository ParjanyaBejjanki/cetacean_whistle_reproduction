"""Independent verification of stored S1..S4 intermediates (used by the Phase 1 gate and reusable later).

Every check recomputes the quantity by a route different from the pipeline code where possible
(re-reading the raw file with soundfile directly, scipy.signal.stft as the STFT reference, Welch PSDs).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .config import Cfg, project_path
from .storage import load_stage_array


def _welch_db(x, fs, nper=8192):
    f, p = signal.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, window="hann", detrend=False)
    return f, 10 * np.log10(p + 1e-30)


def verify_s1_s4(cfg: Cfg, run_dir: Path, stem: str) -> dict:
    raw = project_path(cfg, "raw_dir") / f"{stem}.flac"
    p = cfg["paper"]
    ch = int(cfg["decisions"]["s1"]["channel_index"])
    R: dict = {"stem": stem, "checks": {}}

    def chk(name, ok, **info):
        R["checks"][name] = {"pass": bool(ok), **info}

    # ---- S1: exact equality with an independent read of the raw channel
    s1, fs1 = load_stage_array(run_dir, stem, "s1")
    with sf.SoundFile(str(raw), "r") as f:
        raw_all = f.read(dtype="int16", always_2d=True)
    s1_i16 = np.rint(s1 * cfg["decisions"]["s1"]["int16_full_scale"]).astype(np.int16)
    chk("s1_equals_raw_channel_exactly", np.array_equal(s1_i16, raw_all[:, ch]), n=len(s1), fs=fs1)
    chk("s1_rate_and_length", fs1 == p["fs_raw_hz"] and len(s1) == raw_all.shape[0] == 30_000_000, fs=fs1, n=len(s1))
    del raw_all

    # ---- S2: length, rate, spectrum preserved below the new Nyquist
    s2 = load_stage_array(run_dir, stem, "s2")
    up, down = p["fs_target_hz"] // math.gcd(p["fs_target_hz"], p["fs_raw_hz"]), p["fs_raw_hz"] // math.gcd(p["fs_target_hz"], p["fs_raw_hz"])
    chk("s2_length", len(s2) == math.ceil(len(s1) * up / down), n=len(s2), ratio=f"{up}/{down}")
    f1, p1 = _welch_db(s1.astype(np.float64), fs1, 16384)
    f2, p2 = _welch_db(s2.astype(np.float64), p["fs_target_hz"], 6144)   # same frequency spacing: 500k/16384 = 192k/6291 ~ 30.5 Hz
    grid = np.arange(1000.0, 90000.0, 500.0)
    d = np.interp(grid, f2, p2) - np.interp(grid, f1, p1)
    chk("s2_spectrum_preserved_1k_90k", np.abs(np.median(d)) < 0.1 and np.percentile(np.abs(d), 95) < 0.6,
        median_db=float(np.median(d)), p95_abs_db=float(np.percentile(np.abs(d), 95)))

    # ---- S3: measured high-pass gain
    s3 = load_stage_array(run_dir, stem, "s3")
    f3, p3 = _welch_db(s3.astype(np.float64), p["fs_target_hz"], 6144)
    g = p3 - p2
    gain = lambda f: float(np.interp(f, f2, g))
    gains = {f"{int(f)}Hz": round(gain(f), 2) for f in (300, 500, 1000, 1500, 2000, 3000, 5000, 10000)}
    chk("s3_passband_flat_above_5k", abs(gain(5000)) < 0.3 and abs(gain(10000)) < 0.1, **gains)
    chk("s3_attenuates_below_cutoff", gain(500) < -30 and gain(1000) < -15, **{"g500": gains["500Hz"], "g1000": gains["1000Hz"]})
    chk("s3_minus_6dB_at_cutoff_decision", -7.5 < gain(2000) < -4.5, g2000=gains["2000Hz"])
    chk("s3_same_length", len(s3) == len(s2))

    # ---- S4: shape, grid, finite, reference STFT on a segment
    s4 = load_stage_array(run_dir, stem, "s4")
    n_exp = (len(s3) - p["win_len"]) // p["hop"] + 1
    chk("s4_shape", s4.shape == (n_exp, p["nfft"] // 2 + 1), shape=list(s4.shape), expected=[n_exp, p["nfft"] // 2 + 1])
    chk("s4_finite", bool(np.isfinite(s4).all()), min=float(s4.min()), max=float(s4.max()))
    a, b = 3000, 3300                                  # frame range checked against scipy
    seg = s3[a * p["hop"]: b * p["hop"] + p["win_len"]].astype(np.float64)
    _, _, Z = signal.stft(seg, p["fs_target_hz"], window="hamming", nperseg=p["win_len"], noverlap=p["win_len"] - p["hop"],
                          nfft=p["nfft"], boundary=None, padded=False, detrend=False)
    ref = 20 * np.log10(np.abs(Z.T) * signal.get_window("hamming", p["win_len"]).sum() + float(cfg["decisions"]["s4"]["eps"]))
    m = min(ref.shape[0], b - a)
    chk("s4_matches_scipy_reference", np.abs(ref[:m] - s4[a:a + m]).max() < 0.05, max_abs_db_diff=float(np.abs(ref[:m] - s4[a:a + m]).max()), frames_compared=int(m))
    chk("s4_grid", abs(p["fs_target_hz"] / p["nfft"] - 93.75) < 1e-12 and abs(p["hop"] / p["fs_target_hz"] * 1e3 - 5.3333333) < 1e-5,
        bin_hz=p["fs_target_hz"] / p["nfft"], hop_ms=p["hop"] / p["fs_target_hz"] * 1e3)
    R["summary"] = {"all_pass": all(v["pass"] for v in R["checks"].values()), "n_checks": len(R["checks"])}
    return R


def channel_survey(cfg: Cfg, stem: str, band=(2000.0, 24000.0)) -> dict:
    """DIAGNOSTIC for the UNSPECIFIED channel choice (spec A-01): per-channel S4 level statistics for one file.
    Shows whether the configured channel is anomalous (dead, saturated, or very different); does not select a channel."""
    from . import audio, dsp

    raw = project_path(cfg, "raw_dir") / f"{stem}.flac"
    out = {}
    for ch in range(cfg["paper"]["n_channels_raw"]):
        c2 = type(cfg)({**cfg, "decisions": {**cfg["decisions"], "s1": {**cfg["decisions"]["s1"], "channel_index": ch}}})
        sig = dsp.s3_highpass(dsp.s2_resample(audio.s1_select_channel(raw, c2), c2), c2)
        db = dsp.s4_stft(sig, c2).db
        k0, k1 = int(math.ceil(band[0] / 93.75)), int(band[1] // 93.75)
        sub = db[:, k0:k1 + 1]
        out[ch] = {"int16_peak": sig.meta["int16_peak"], "median_db_band": round(float(np.median(sub)), 2),
                   "p10_db": round(float(np.percentile(sub, 10)), 2), "p99_db": round(float(np.percentile(sub, 99)), 2),
                   "frac_frames_loud_broadband": round(float((sub > np.median(sub, axis=0) + 15).mean(axis=1).__gt__(0.3).mean()), 4)}
    return out


# ----------------------------------------------------------------------------- run-level gate (Phases 2-4)
def verify_run(run_dir: Path) -> dict:
    """Automated validation of a finished run: lineage completeness, output validity, cross-stage consistency.
    Pure inspection of what is on disk (no recomputation)."""
    import csv
    import json

    run_dir = Path(run_dir)
    R: dict = {"run": run_dir.name, "problems": [], "per_file": {}}

    def bad(msg):
        R["problems"].append(msg)

    man = json.loads((run_dir / "run_manifest.json").read_text())
    summ = json.loads((run_dir / "summary.json").read_text())
    if summ["ok"] + summ["failed"] != man["n_files"]:
        bad(f"ok+failed ({summ['ok'] + summ['failed']}) != n_files ({man['n_files']})")
    if summ["failed"]:
        bad(f"{summ['failed']} failed files: {[f['stem'] for f in summ['failed_files']]}")
    det_shas, cfg_hashes = set(), set()
    n_img_total = 0
    for f in man["files"]:
        stem = f[:-5]
        mp = run_dir / "meta" / f"{stem}.json"
        if not mp.exists():
            if not (run_dir / "failed" / stem / "meta.json").exists():
                bad(f"{stem}: no lineage metadata")
            continue
        m = json.loads(mp.read_text())
        info = {"time_s": m["total_time_s"]}
        if m["status"] != "OK":
            bad(f"{stem}: status {m['status']}")
        if not m["raw"].get("unchanged_after_processing", False):
            bad(f"{stem}: raw file changed during processing!")
        if m["raw"].get("sha256_matches_manifest") is False:
            bad(f"{stem}: sha256 differs from manifest")
        cfg_hashes.add(m["pipeline_config_effective_hash"])
        if m.get("detector"):
            det_shas.add(m["detector"].get("checkpoint_sha256"))
        d = m["stage_diagnostics"]
        if "s9" in d:
            imgs = np.load(run_dir / "final" / stem / "images.npy")
            n = len(imgs)
            n_img_total += n
            info["n_images"] = n
            if imgs.ndim != 3 or imgs.shape[1:] != (128, 128) or imgs.dtype != np.float32:
                bad(f"{stem}: images shape/dtype {imgs.shape} {imgs.dtype}")
            if n:
                if not np.isfinite(imgs).all() or imgs.min() < 0 or imgs.max() > 1:
                    bad(f"{stem}: image values outside [0,1] or non-finite")
                mx = imgs.reshape(n, -1).max(axis=1)
                if np.abs(mx - 1).max() > 1e-5:
                    bad(f"{stem}: an image has max != 1 (per-image max normalisation)")
                if (imgs.reshape(n, -1).max(axis=1) <= 0).any():
                    bad(f"{stem}: blank image")
                rows = list(csv.DictReader(open(run_dir / "final" / stem / "images_meta.csv", encoding="utf-8")))
                if len(rows) != n:
                    bad(f"{stem}: images_meta rows {len(rows)} != images {n}")
            if "s6" in d and "s7" in d and d["s6"]["n_tracks"] != d["s7"]["n_vectors"]:
                bad(f"{stem}: tracks {d['s6']['n_tracks']} != vectors {d['s7']['n_vectors']}")
            if d["s8"]["n_baseband"] + d["s8"]["n_rejected"] != d["s7"]["n_vectors"]:
                bad(f"{stem}: s8 accepted+rejected != vectors")
            if d["s9"]["n_images"] != d["s8"]["n_baseband"]:
                bad(f"{stem}: s9 images != s8 baseband")
        R["per_file"][stem] = info
    if len(cfg_hashes) > 1:
        bad(f"more than one pipeline config hash within the run: {cfg_hashes}")
    if len(det_shas) > 1:
        bad(f"more than one detector checkpoint within the run: {det_shas}")
    R["n_files"], R["n_images_total"], R["config_hashes"], R["detector_checkpoint_sha256"] = man["n_files"], n_img_total, sorted(cfg_hashes), sorted(map(str, det_shas))
    R["pass"] = not R["problems"]
    return R
