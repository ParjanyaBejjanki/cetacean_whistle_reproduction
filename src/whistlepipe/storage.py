"""Run storage, data lineage and the storage policy.

Layout of one run (nothing outside runs/<run_id>/ is touched; raw audio is never written):

    runs/<run_id>/
        run_manifest.json        config snapshot + hashes, seeds, versions, detector checkpoint, file list
        summary.json             end-of-run totals (written by the pipeline)
        logs/run.log             human-readable progress
        logs/stage_stats.jsonl   one JSON line per file with per-stage diagnostics
        intermediate/<stem>/     S1..S6 (+S8) outputs, according to the storage policy
        final/<stem>/            vectors.npz (S7), images.npy + images_meta.csv (S9)  -- kept in EVERY mode
        final/images_index.csv   consolidated index of all images in the run
        meta/<stem>.json         lineage: raw file -> config -> stage outputs -> detector -> final output
        failed/<stem>/           traceback + every intermediate that existed when the file failed
        diagnostics/             figures
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .config import PROJECT_ROOT, Cfg, canonical_hash, project_path

STAGES_STORABLE = ("s1", "s2", "s3", "s4", "s5", "s6", "s8")


class RawDataGuardError(RuntimeError):
    pass


def ensure_not_in_raw(path: Path, raw_dir: Path) -> Path:
    """Refuse any path inside the immutable raw-data directory."""
    p, r = Path(path).resolve(), Path(raw_dir).resolve()
    if p == r or r in p.parents:
        raise RawDataGuardError(f"refusing to write inside the immutable raw-data directory: {p}")
    return p


def software_versions() -> dict:
    import numpy
    import scipy

    v = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "soundfile": sf.__version__,
        "libsndfile": sf.__libsndfile_version__,
    }
    try:
        import torch

        v["torch"] = torch.__version__
        v["cuda_available"] = bool(torch.cuda.is_available())
        v["tf32_conv_allowed"] = bool(torch.backends.cudnn.allow_tf32)      # S5 numerics policy: must be False
        v["tf32_matmul_allowed"] = bool(torch.backends.cuda.matmul.allow_tf32)
        v["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
        if torch.cuda.is_available():
            v["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:   # pragma: no cover
        v["torch"] = None
    return v


class RunStore:
    def __init__(self, cfg: Cfg, run_id: str, mode: str | None = None, runs_dir: Path | None = None,
                 create: bool = True):
        self.cfg = cfg
        self.raw_dir = project_path(cfg, "raw_dir")
        self.mode = mode or cfg["storage"]["mode"]
        if self.mode not in ("debug", "normal", "minimal"):
            raise ValueError(f"unknown storage mode {self.mode!r}")
        runs_root = Path(runs_dir) if runs_dir else project_path(cfg, "runs_dir")
        self.dir = ensure_not_in_raw(runs_root / run_id, self.raw_dir)
        self.run_id = run_id
        self.policy = {s: bool(cfg["storage"][self.mode].get(s, False)) for s in STAGES_STORABLE}
        if create:
            if self.dir.exists():
                raise FileExistsError(
                    f"run directory already exists: {self.dir}\n"
                    "Runs are never overwritten. Choose a new --run-id (e.g. append _r2).")
            for sub in ("logs", "intermediate", "final", "meta", "failed", "diagnostics"):
                (self.dir / sub).mkdir(parents=True)

    # ------------------------------------------------------------------ paths
    def sub(self, name: str, stem: str | None = None) -> Path:
        p = self.dir / name / stem if stem else self.dir / name
        p = ensure_not_in_raw(p, self.raw_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def keep(self, stage: str) -> bool:
        return self.policy.get(stage, False)

    # ------------------------------------------------------------------ run-level records
    def write_manifest(self, extra: dict) -> None:
        m = {
            "run_id": self.run_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "storage_mode": self.mode,
            "storage_policy": self.policy,
            "pipeline_config": self.cfg.to_dict(),
            "pipeline_config_effective_hash": self.cfg["_meta"]["effective_hash"],
            "software": software_versions(),
            **extra,
        }
        self._write_json(self.dir / "run_manifest.json", m)

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with open(self.dir / "logs" / "run.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def append_stats(self, record: dict) -> None:
        with open(self.dir / "logs" / "stage_stats.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=_json_default)

    def write_json(self, path: Path, obj: Any) -> None:
        ensure_not_in_raw(path, self.raw_dir)
        self._write_json(path, obj)

    # ------------------------------------------------------------------ stage writers
    def save_stage(self, stem: str, stage: str, obj: Any, failed: bool = False) -> list[dict]:
        """Persist one stage output. Returns descriptors [{path, bytes, shape, dtype}] for the lineage record."""
        base = self.sub("failed", stem) if failed else self.sub("intermediate", stem)
        out: list[dict] = []

        def rec(path: Path, arr=None, **extra):
            d = {"path": str(path.relative_to(self.dir)), "bytes": path.stat().st_size, **extra}
            if arr is not None:
                d.update(shape=list(arr.shape), dtype=str(arr.dtype))
            out.append(d)

        if stage == "s1":
            x16 = np.clip(np.rint(obj.samples * np.float32(self.cfg["decisions"]["s1"]["int16_full_scale"])), -32768, 32767).astype(np.int16)
            p = base / "s1_channel.flac"
            sf.write(str(p), x16, int(obj.fs), subtype="PCM_16", format="FLAC")
            rec(p, x16, fs=obj.fs)
        elif stage in ("s2", "s3"):
            p = base / ("s2_resampled.npy" if stage == "s2" else "s3_highpassed.npy")
            a = obj.samples.astype(np.float32)
            np.save(p, a)
            rec(p, a, fs=obj.fs)
        elif stage == "s4":
            p = base / "s4_spectrogram_db.npy"
            np.save(p, obj.db)
            rec(p, obj.db, fs=obj.fs, win_len=obj.win_len, hop=obj.hop, nfft=obj.nfft)
        elif stage == "s5":
            dtype = np.float32 if (self.mode == "debug" or failed) else np.float16
            a = obj.conf.astype(dtype)
            p = base / "s5_confidence.npy"
            np.save(p, a)
            rec(p, a, bin_lo=obj.bin_lo, bin_hi=obj.bin_hi, detector=obj.detector_version)
        elif stage == "s6":
            p = base / "s6_tracks.npz"
            _save_tracks(p, obj.tracks)
            rec(p, n_tracks=len(obj.tracks))
            pr = base / "s6_rejected.json"
            self._write_json(pr, {"stats": obj.stats, "rejected": obj.rejected[:2000]})
            rec(pr, n_rejected=len(obj.rejected))
        elif stage == "s8":
            p = base / "s8_baseband.npz"
            if obj:
                np.savez_compressed(p, track_ids=np.array([o.track_id for o in obj]),
                                    baseband=np.stack([o.baseband for o in obj]),
                                    center_hz=np.array([o.center_hz for o in obj]),
                                    col0=np.array([o.col0 for o in obj]))
                rec(p, n=len(obj))
        else:
            raise ValueError(stage)
        return out

    def save_final(self, stem: str, vectors_npz: dict, images: np.ndarray, images_meta: list[dict]) -> list[dict]:
        base = self.sub("final", stem)
        out = []
        pv = base / "vectors.npz"
        np.savez_compressed(pv, **vectors_npz)
        out.append({"path": str(pv.relative_to(self.dir)), "bytes": pv.stat().st_size, "n_vectors": int(len(vectors_npz.get("track_ids", [])))})
        pi = base / "images.npy"
        np.save(pi, images.astype(np.float32))
        out.append({"path": str(pi.relative_to(self.dir)), "bytes": pi.stat().st_size,
                    "shape": list(images.shape), "dtype": "float32"})
        pm = base / "images_meta.csv"
        if images_meta:
            with open(pm, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(images_meta[0].keys()))
                w.writeheader()
                w.writerows(images_meta)
            out.append({"path": str(pm.relative_to(self.dir)), "bytes": pm.stat().st_size, "rows": len(images_meta)})
        return out

    def write_meta(self, stem: str, meta: dict, failed: bool = False) -> Path:
        p = (self.sub("failed", stem) / "meta.json") if failed else (self.sub("meta") / f"{stem}.json")
        self.write_json(p, meta)
        return p

    def write_failure(self, stem: str, outputs: dict, meta: dict, tb: str) -> None:
        d = self.sub("failed", stem)
        (d / "traceback.txt").write_text(tb, encoding="utf-8")
        saved = {}
        for st in STAGES_STORABLE:
            if st in outputs and outputs[st] is not None:
                try:
                    saved[st] = self.save_stage(stem, st, outputs[st], failed=True)
                except Exception as e:   # never let diagnostics mask the real failure
                    saved[st] = [{"error": repr(e)}]
        meta = {**meta, "status": "FAILED", "failed_intermediates": saved}
        self.write_meta(stem, meta, failed=True)

    def consolidate_images_index(self, rows: list[dict]) -> Path | None:
        if not rows:
            return None
        p = self.dir / "final" / "images_index.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return p


# ---------------------------------------------------------------------------- (de)serialisation helpers
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _save_tracks(path: Path, tracks: list) -> None:
    if tracks:
        lens = np.array([len(t.frames) for t in tracks])
        np.savez_compressed(
            path, track_ids=np.array([t.id for t in tracks]), offsets=np.concatenate([[0], np.cumsum(lens)]),
            frames=np.concatenate([t.frames for t in tracks]), freq_hz=np.concatenate([t.freq_hz for t in tracks]),
            conf=np.concatenate([t.conf for t in tracks]), observed=np.concatenate([t.observed for t in tracks]))
    else:
        np.savez_compressed(path, track_ids=np.zeros(0, int), offsets=np.zeros(1, int), frames=np.zeros(0, int),
                            freq_hz=np.zeros(0), conf=np.zeros(0), observed=np.zeros(0, bool))


def load_tracks(path: Path) -> list:
    from .tracking import Track

    z = np.load(path)
    out = []
    for i, tid in enumerate(z["track_ids"]):
        a, b = z["offsets"][i], z["offsets"][i + 1]
        out.append(Track(int(tid), z["frames"][a:b], z["freq_hz"][a:b], z["conf"][a:b], z["observed"][a:b]))
    return out


def load_stage_array(run_dir: Path, stem: str, stage: str):
    """Load a stored intermediate (for partial pipeline runs: --from-run)."""
    d = Path(run_dir) / "intermediate" / stem
    if stage == "s1":
        x, fs = sf.read(str(d / "s1_channel.flac"), dtype="float32")
        return x, fs
    if stage in ("s2", "s3"):
        return np.load(d / ("s2_resampled.npy" if stage == "s2" else "s3_highpassed.npy"))
    if stage == "s4":
        return np.load(d / "s4_spectrogram_db.npy")
    if stage == "s5":
        return np.load(d / "s5_confidence.npy").astype(np.float32)
    raise ValueError(stage)
