"""Pipeline orchestrator: runs S1..S9 per file with diagnostics, lineage and storage policy.

    S1 channel -> S2 resample -> S3 high-pass -> S4 STFT -> S5 detector -> S6 tracking
       -> S7 whistle vectors -> S8 centre-frequency removal + NFM baseband -> S9 128x128 image

Stages are plain functions in audio.py / dsp.py / detector / tracking.py / representation.py. Any contiguous
range can be run (``first``..``last``); starting after S1 loads the previous stage from ``from_run``.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio, dsp
from .config import PROJECT_ROOT, Cfg, band_bins, canonical_hash, project_path
from .storage import RunStore, load_stage_array, load_tracks

STAGE_ORDER = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]


@dataclass
class RunOptions:
    run_id: str
    files: list[Path]
    first: str = "s1"
    last: str = "s9"
    storage_mode: str | None = None
    from_run: Path | None = None
    device: str | None = None
    verify_hash: bool = True
    fail_fast: bool = False
    note: str = ""


# ----------------------------------------------------------------------------- diagnostics
def _stats(a: np.ndarray) -> dict:
    a = np.asarray(a)
    if a.size == 0:
        return {"min": None, "max": None}
    f = a.astype(np.float64) if not np.iscomplexobj(a) else np.abs(a)
    return {"min": float(f.min()), "max": float(f.max()), "mean": float(f.mean())}


def diag_audio(sig) -> dict:
    return {"out_shape": [len(sig.samples)], "fs_hz": sig.fs, "duration_s": round(sig.duration_s, 6),
            "dtype": str(sig.samples.dtype), **_stats(sig.samples)}


def diag_spec(spec) -> dict:
    return {"out_shape": list(spec.db.shape), "fs_hz": spec.fs, "bin_hz": spec.bin_hz,
            "hop_ms": spec.hop_s * 1e3, "dtype": str(spec.db.dtype), **_stats(spec.db),
            "median_db_2to24kHz": float(np.median(spec.db[:, 22:257])) if spec.n_frames else None}


def diag_detection(dm, tau: float) -> dict:
    return {"out_shape": list(dm.conf.shape), "band_bins": [dm.bin_lo, dm.bin_hi], **_stats(dm.conf),
            "n_nodes_above_tau": int((dm.conf >= tau).sum()), "frac_nodes_above_tau": float((dm.conf >= tau).mean()),
            "detector": f"{dm.detector_name}/{dm.detector_version}"}


# ----------------------------------------------------------------------------- loading a previous stage
def _load_prior(prior: str, stem: str, from_run: Path, cfg: Cfg):
    p = cfg["paper"]
    if prior == "s1":
        x, fs = load_stage_array(from_run, stem, "s1")
        return audio.AudioSignal(x, float(fs), stem + ".flac")
    if prior in ("s2", "s3"):
        x = load_stage_array(from_run, stem, prior).astype(np.float64)
        return audio.AudioSignal(x, float(p["fs_target_hz"]), stem + ".flac")
    if prior == "s4":
        db = load_stage_array(from_run, stem, "s4")
        return dsp.Spectrogram(db, float(p["fs_target_hz"]), p["win_len"], p["hop"], p["nfft"], stem + ".flac")
    if prior == "s5":
        from .detector.interface import DetectionMap

        conf = load_stage_array(from_run, stem, "s5")
        lo, hi = band_bins(cfg)
        return DetectionMap(conf, lo, hi, p["nfft"] // 2 + 1, "loaded", "loaded")
    if prior == "s6":
        from .tracking import TrackingResult

        return TrackingResult(load_tracks(Path(from_run) / "intermediate" / stem / "s6_tracks.npz"), [], {})
    raise ValueError(f"cannot resume after {prior}")


# ----------------------------------------------------------------------------- one file
@dataclass
class FileContext:
    stem: str
    path: Path
    outputs: dict = field(default_factory=dict)
    diag: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    stored: dict = field(default_factory=dict)


def _timed(ctx: FileContext, stage: str, fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    ctx.timings[stage] = round(time.perf_counter() - t0, 4)
    return r


def process_file(cfg: Cfg, store: RunStore, opts: RunOptions, path: Path, detector, manifest_hashes: dict) -> dict:
    """Run one file. On failure the exception carries ``ctx_outputs`` (every stage output that existed)."""
    ctx = FileContext(stem=path.stem, path=path)
    try:
        return _process_file(cfg, store, opts, path, detector, manifest_hashes, ctx)
    except Exception as e:
        e.ctx_outputs = ctx.outputs
        e.ctx_diag = ctx.diag
        raise


def _process_file(cfg: Cfg, store: RunStore, opts: RunOptions, path: Path, detector, manifest_hashes: dict,
                  ctx: FileContext) -> dict:
    stem = path.stem
    i0, i1 = STAGE_ORDER.index(opts.first), STAGE_ORDER.index(opts.last)
    tau = float(cfg["decisions"]["s6"]["peak_threshold"])
    raw_stat0 = path.stat()
    raw = {"path": str(path.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in path.parents else str(path),
           "size_bytes": raw_stat0.st_size, "mtime_ns": raw_stat0.st_mtime_ns}
    raw["audio_info"] = audio.audio_info(path)
    if opts.verify_hash:
        t0 = time.perf_counter()
        raw["sha256"] = audio.sha256_file(path)
        exp = manifest_hashes.get(path.name)
        raw["sha256_matches_manifest"] = None if exp is None else (exp == raw["sha256"])
        ctx.timings["verify_sha256"] = round(time.perf_counter() - t0, 3)
        if exp is not None and exp != raw["sha256"]:
            raise RuntimeError(f"raw file hash differs from manifest: {path.name}")

    # resume support
    if i0 > 0:
        prior = STAGE_ORDER[i0 - 1]
        if opts.from_run is None:
            raise ValueError(f"--first {opts.first} needs --from-run (a previous run holding the {prior} intermediate)")
        ctx.outputs[prior] = _load_prior(prior, stem, Path(opts.from_run), cfg)

    for stage in STAGE_ORDER[i0:i1 + 1]:
        o = ctx.outputs
        if stage == "s1":
            o["s1"] = _timed(ctx, "s1", audio.s1_select_channel, path, cfg)
            ctx.diag["s1"] = {**diag_audio(o["s1"]), **o["s1"].meta}
        elif stage == "s2":
            o["s2"] = _timed(ctx, "s2", dsp.s2_resample, o["s1"], cfg)
            ctx.diag["s2"] = {"in_shape": [len(o["s1"].samples)], **diag_audio(o["s2"])}
        elif stage == "s3":
            o["s3"] = _timed(ctx, "s3", dsp.s3_highpass, o["s2"], cfg)
            ctx.diag["s3"] = {"in_shape": [len(o["s2"].samples)], **diag_audio(o["s3"])}
        elif stage == "s4":
            o["s4"] = _timed(ctx, "s4", dsp.s4_stft, o["s3"], cfg)
            ctx.diag["s4"] = {"in_shape": [len(o["s3"].samples)], **diag_spec(o["s4"])}
        elif stage == "s5":
            o["s5"] = _timed(ctx, "s5", detector.predict, o["s4"])
            ctx.diag["s5"] = {"in_shape": list(o["s4"].db.shape), **diag_detection(o["s5"], tau)}
        elif stage == "s6":
            from .tracking import run_s6

            o["s6"] = _timed(ctx, "s6", run_s6, o["s5"], cfg)
            ctx.diag["s6"] = {"in_shape": list(o["s5"].conf.shape), "n_tracks": len(o["s6"].tracks),
                              "n_rejected": len(o["s6"].rejected), **o["s6"].stats}
        elif stage == "s7":
            from .representation import s7_vectors

            o["s7"] = _timed(ctx, "s7", s7_vectors, o["s6"].tracks, cfg)
            ctx.diag["s7"] = {"n_vectors": len(o["s7"]),
                              "frames_total": int(sum(v.n_frames for v in o["s7"]))}
        elif stage == "s8":
            from .representation import s8_baseband

            o["s8"], rej = _timed(ctx, "s8", s8_baseband, o["s7"], cfg)
            o["s8_rejected"] = rej
            by_reason: dict = {}
            for r in rej:
                by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
            ctx.diag["s8"] = {"n_baseband": len(o["s8"]), "n_rejected": len(rej), "rejected_by_reason": by_reason,
                              "baseband_samples": int(o["s8"][0].baseband.shape[0]) if o["s8"] else None}
        elif stage == "s9":
            from .representation import s9_images, image_meta_rows

            imgs = _timed(ctx, "s9", s9_images, o["s8"], cfg)
            o["s9"] = imgs
            ctx.diag["s9"] = {"out_shape": list(imgs.shape), "dtype": str(imgs.dtype),
                              **(_stats(imgs) if imgs.size else {"min": None, "max": None}),
                              "n_images": int(len(imgs))}
        # storage policy
        if stage in ("s1", "s2", "s3", "s4", "s5", "s6", "s8") and store.keep(stage):
            ctx.stored[stage] = store.save_stage(stem, stage, o[stage])

    # final outputs (kept in every mode) when the range reaches S9
    final_desc, rows = [], []
    if "s9" in ctx.outputs:
        from .representation import image_meta_rows, vectors_to_npz

        rows = image_meta_rows(stem, ctx.outputs["s8"], ctx.outputs["s7"], ctx.outputs["s4"].hop_s if "s4" in ctx.outputs else cfg["paper"]["hop"] / cfg["paper"]["fs_target_hz"])
        final_desc = store.save_final(stem, vectors_to_npz(ctx.outputs["s7"]), ctx.outputs["s9"], rows)
    elif "s7" in ctx.outputs:
        from .representation import vectors_to_npz

        base = store.sub("final", stem)
        np.savez_compressed(base / "vectors.npz", **vectors_to_npz(ctx.outputs["s7"]))
        final_desc = [{"path": f"final/{stem}/vectors.npz"}]

    # raw immutability audit
    st1 = path.stat()
    raw["unchanged_after_processing"] = (st1.st_size == raw_stat0.st_size and st1.st_mtime_ns == raw_stat0.st_mtime_ns)

    det_info = None
    if detector is not None and i0 <= STAGE_ORDER.index("s5") <= i1:
        det_info = {"name": detector.name, "version": detector.version, **detector.meta}
    meta = {
        "stem": stem, "encounter_id": audio.encounter_of(path), "status": "OK", "run_id": store.run_id, "storage_mode": store.mode,
        "raw": raw, "pipeline_config_effective_hash": cfg["_meta"]["effective_hash"],
        "stages_run": STAGE_ORDER[i0:i1 + 1], "resumed_from_run": str(opts.from_run) if opts.from_run else None,
        "stage_diagnostics": ctx.diag, "timings_s": ctx.timings, "total_time_s": round(sum(ctx.timings.values()), 3),
        "detector": det_info, "outputs": {"intermediate": ctx.stored, "final": final_desc},
    }
    return {"meta": meta, "ctx": ctx, "rows": rows}


# ----------------------------------------------------------------------------- run
def run_pipeline(cfg: Cfg, opts: RunOptions, project_root: Path = PROJECT_ROOT, det_cfg: Cfg | None = None,
                 runs_dir: Path | None = None, progress=None) -> dict:
    """Run ``opts.files`` into ``<runs_dir>/<run_id>/`` (default runs_dir: [paths].runs_dir).
    ``progress``: optional callable receiving the same progress lines as logs/run.log (e.g. print)."""
    store = RunStore(cfg, opts.run_id, opts.storage_mode, runs_dir=runs_dir)

    def say(msg: str) -> None:
        store.log(msg)
        if progress:
            progress(msg)

    i0, i1 = STAGE_ORDER.index(opts.first), STAGE_ORDER.index(opts.last)
    if i0 > i1:
        raise ValueError("first stage after last stage")
    detector = None
    detector_info = None
    if i0 <= STAGE_ORDER.index("s5") <= i1:
        from .detector import RidgeDetector

        ckp = Path(cfg["decisions"]["s5"]["checkpoint"])
        ck = ckp if ckp.is_absolute() else PROJECT_ROOT / ckp
        detector = RidgeDetector.from_checkpoint(ck, cfg, opts.device)
        detector_info = {"name": detector.name, "version": detector.version, **detector.meta}
    manifest = audio.load_manifest_hashes(project_path(cfg, "manifest_csv"))
    store.write_manifest({
        "options": {"first": opts.first, "last": opts.last, "from_run": str(opts.from_run) if opts.from_run else None,
                    "verify_hash": opts.verify_hash, "note": opts.note},
        "files": [p.name for p in opts.files], "n_files": len(opts.files),
        "detector": detector_info,
        "detector_config_hash": det_cfg["_meta"]["effective_hash"] if det_cfg else None,
        "seed_note": "inference is deterministic; random seeds are used only by synthetic-data generation/training (see models/detector/*/)",
    })
    say(f"run {opts.run_id}: {len(opts.files)} files, stages {opts.first}..{opts.last}, storage={store.mode}")

    summary = {"run_id": opts.run_id, "n_files": len(opts.files), "ok": 0, "failed": 0, "failed_files": [],
               "n_tracks": 0, "n_vectors": 0, "n_images": 0, "track_rejections": {}, "s8_rejections": {},
               "stage_time_s": {}, "files": []}
    all_rows: list[dict] = []
    t_run = time.perf_counter()
    for i, path in enumerate(opts.files, 1):
        stem = path.stem
        try:
            res = process_file(cfg, store, opts, path, detector, manifest)
            meta, ctx = res["meta"], res["ctx"]
            store.write_meta(stem, meta)
            store.append_stats({"stem": stem, "status": "OK", "timings_s": meta["timings_s"], "diag": meta["stage_diagnostics"]})
            all_rows += res["rows"]
            summary["ok"] += 1
            d = ctx.diag
            summary["n_tracks"] += d.get("s6", {}).get("n_tracks", 0)
            summary["n_vectors"] += d.get("s7", {}).get("n_vectors", 0)
            summary["n_images"] += d.get("s9", {}).get("n_images", 0)
            for k, v in d.get("s6", {}).get("rejected_by_reason", {}).items():
                summary["track_rejections"][k] = summary["track_rejections"].get(k, 0) + v
            for k, v in d.get("s8", {}).get("rejected_by_reason", {}).items():
                summary["s8_rejections"][k] = summary["s8_rejections"].get(k, 0) + v
            for k, v in meta["timings_s"].items():
                summary["stage_time_s"][k] = round(summary["stage_time_s"].get(k, 0.0) + v, 3)
            summary["files"].append({"stem": stem, "n_tracks": d.get("s6", {}).get("n_tracks"), "n_images": d.get("s9", {}).get("n_images"),
                                     "time_s": meta["total_time_s"]})
            if store.mode == "debug" and cfg["storage"].get("plots") and all(s in ctx.outputs for s in ("s4", "s5", "s6")):
                from .plotting import plot_overview

                plot_overview(store.sub("diagnostics") / f"{stem}_overview.png", ctx.outputs["s4"], ctx.outputs["s5"],
                              ctx.outputs["s6"], cfg, title=stem)
            say(f"[{i}/{len(opts.files)}] OK  {stem}  {meta['total_time_s']:.1f}s  tracks={d.get('s6', {}).get('n_tracks')} images={d.get('s9', {}).get('n_images')}")
        except Exception as e:   # noqa: BLE001  (file-level isolation; the failure is fully recorded)
            tb = traceback.format_exc()
            summary["failed"] += 1
            summary["failed_files"].append({"stem": stem, "error": repr(e)})
            # on failure keep every intermediate that existed (regardless of the storage policy)
            store.write_failure(stem, getattr(e, "ctx_outputs", {}),
                                {"stem": stem, "encounter_id": audio.encounter_of(path), "run_id": store.run_id, "error": repr(e),
                                 "stage_diagnostics": getattr(e, "ctx_diag", {}),
                                 "config_hash": cfg["_meta"]["effective_hash"]}, tb)
            store.append_stats({"stem": stem, "status": "FAILED", "error": repr(e)})
            say(f"[{i}/{len(opts.files)}] FAILED {stem}: {e!r}")
            if opts.fail_fast:
                raise
    summary["wall_time_s"] = round(time.perf_counter() - t_run, 2)
    p = store.consolidate_images_index(all_rows)
    summary["images_index"] = str(p.relative_to(store.dir)) if p else None
    store.write_json(store.dir / "summary.json", summary)
    say(f"done: ok={summary['ok']} failed={summary['failed']} images={summary['n_images']} wall={summary['wall_time_s']}s")
    return summary

