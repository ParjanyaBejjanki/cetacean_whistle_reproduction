"""Final dataset in a simple layout, one folder per encounter (python whistle.py build-dataset).

    preprocessed/
        README.txt                what every file is
        dataset_info.json         detector, config (+hash), code version, software, per-encounter counts
        encounter_<id>/
            images.npy            (N, 128, 128) float32 in [0, 1]: every whistle image of the encounter
            images.csv            one row per image; row i describes images.npy[i]
            contours.npz          the S7 contours (one per accepted track); images point to them via contour_id
            files.csv             one row per raw file: status, number of images, ...
        _details/                 not needed for training
            logs/                                     encounter_<id>_{run.log, stage_stats.jsonl, run_manifest.json, summary.json, entry.json}
            per_file/encounter_<id>/<stem>.json       lineage of each raw file
            failed/encounter_<id>/<stem>/             traceback + intermediates of a failed file (only if one failed)
            detector_output/encounter_<id>/<stem>/    S5 confidence (float16) + S6 tracks ([dataset].keep_detector_output)

How it is built: each encounter runs through the normal pipeline into a temporary run
(_details/work/encounter_<id>/), is packed into the four files above, and the temporary run is removed.
The encounter folder is written as encounter_<id>.partial/ and renamed last, so it exists only when complete:
an interrupted build leaves finished encounters valid and the next build redoes only the missing ones.
A complete encounter is never rebuilt silently: if its raw folder gained or lost files, the build stops.
One dataset = one processing config ([paper] + [decisions]) and one detector checkpoint (anything else is refused);
paths and the list of raw folders are not part of that identity, so encounters can be added later.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import audio
from .config import PROJECT_ROOT, Cfg, processing_hash, project_path, protected_raw_dirs
from .manifest import EXPECTED_FAILURE_STATUSES
from .pipeline import RunOptions, run_pipeline
from .storage import _json_default, code_version, ensure_not_in_raw, software_versions

ENCOUNTER_FILES = ("images.npy", "images.csv", "contours.npz", "files.csv")
DETAIL_SUBDIRS = ("logs", "per_file", "failed", "detector_output")
IMAGE_COLUMNS = ["index", "encounter", "source_file", "index_in_file", "contour_id", "start_s", "end_s", "duration_ms",
                 "n_frames", "f_min_hz", "f_max_hz", "center_hz", "bandwidth_hz", "mean_conf", "observed_fraction",
                 "image_col0"]
FILE_COLUMNS = ["file", "encounter", "status", "n_images", "first_image_index", "n_contours", "n_too_long", "n_too_wide",
                "duration_s", "manifest_status", "detector_background_split", "sha256_ok", "raw_unchanged", "time_s", "error"]
_YES_NO = {True: "yes", False: "NO", None: ""}


def enc_name(eid: int) -> str:
    return f"encounter_{eid}"


def _rel(p: Path) -> str:
    return p.relative_to(PROJECT_ROOT).as_posix() if PROJECT_ROOT in p.parents else str(p)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _remove(p: Path, root: Path) -> None:
    """Delete something this module created (only ever inside the dataset folder)."""
    if Path(root).resolve() not in Path(p).resolve().parents:
        raise RuntimeError(f"refusing to delete outside {root}: {p}")
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


def _leftovers(root: Path, eid: int) -> list[Path]:
    """Everything an interrupted build of one encounter may have left behind."""
    n, d = enc_name(eid), root / "_details"
    return [root / f"{n}.partial", d / "work" / n, d / "per_file" / n, d / "failed" / n, d / "detector_output" / n,
            *sorted((d / "logs").glob(f"{n}_*"))]


def _checkpoint(cfg: Cfg) -> Path:
    p = Path(cfg["decisions"]["s5"]["checkpoint"])
    return p if p.is_absolute() else PROJECT_ROOT / p


def _identity(cfg: Cfg) -> dict:
    return {"processing_config_hash": processing_hash(cfg),
            "detector_checkpoint_sha256": audio.sha256_file(_checkpoint(cfg))}


def complete_entries(root: Path) -> dict[int, dict]:
    """Build records of the encounters whose folder is complete."""
    out = {}
    for p in sorted((Path(root) / "_details" / "logs").glob("encounter_*_entry.json")):
        e = json.loads(p.read_text(encoding="utf-8"))
        if (Path(root) / enc_name(e["encounter"])).is_dir():
            out[e["encounter"]] = e
    return dict(sorted(out.items()))


def raw_changes(enc_dir: Path, files: list[Path]) -> str | None:
    """None if the encounter folder was built from exactly ``files``; otherwise what was added/removed since."""
    built = {r["file"] for r in _read_csv(enc_dir / "files.csv")}
    now = {p.stem for p in files}
    if built == now:
        return None
    added, removed = sorted(now - built), sorted(built - now)
    return (f"{enc_dir.name}: built from {len(built)} raw files, the raw folder now has {len(now)} "
            f"(added {len(added)}: {added[:3]}, removed {len(removed)}: {removed[:3]})")


def background_splits(cfg: Cfg) -> dict[str, str]:
    """raw file name -> 'train' / 'val' / 'test' for the files whose quiet windows are background banks of the
    synthetic detector data (train = seen in detector training; val/test = detector validation/test only)."""
    d = project_path(cfg, "data_dir") / "synthetic" / "backgrounds"
    out = {}
    for split in ("train", "val", "test"):
        p = d / f"{split}.json"
        if p.exists():
            for name in json.loads(p.read_text(encoding="utf-8"))["files"]:
                out[Path(name).name] = split
    return out


# ----------------------------------------------------------------------------- build
def build_dataset(cfg: Cfg, det_cfg: Cfg | None = None, encounters: list[int] | None = None,
                  device: str | None = None, log=print) -> dict:
    root = project_path(cfg, "dataset_dir")
    ensure_not_in_raw(root, protected_raw_dirs(cfg))
    by_enc = audio.files_by_encounter(cfg)
    todo = list(dict.fromkeys(encounters)) if encounters else list(by_enc)
    unknown = [e for e in todo if e not in by_enc]
    if unknown:
        raise ValueError(f"encounters {unknown} are not in [paths].raw_dirs (configured: {list(by_enc)})")
    empty = [e for e in todo if not by_enc[e]]
    if empty:
        raise ValueError(f"the raw folders of encounters {empty} contain no .flac files")
    changed = [msg for e in todo if (root / enc_name(e)).is_dir() and (msg := raw_changes(root / enc_name(e), by_enc[e]))]
    if changed:
        raise RuntimeError("raw folders changed since these encounters were built:\n  " + "\n  ".join(changed) +
                           f"\nDelete those folders in {root} to rebuild them (their _details are cleaned automatically).")
    manifest_csv = project_path(cfg, "manifest_csv")
    missing = audio.missing_from_manifest(cfg, [p for e in todo for p in by_enc[e]])
    if missing:
        raise RuntimeError(f"{len(missing)} raw files are not in {manifest_csv} (e.g. {missing[:3]}): run "
                           "`python whistle.py build-manifest` (it never overwrites: delete the old manifest first if it exists)")
    manifest = audio.load_manifest_rows(manifest_csv)
    ident = _identity(cfg)
    for e in complete_entries(root).values():
        if e["identity"] != ident:
            raise RuntimeError(f"{root} holds {enc_name(e['encounter'])} built with another processing config/detector "
                               f"({e['identity']} != {ident}). Use another [paths].dataset_dir or remove {root} deliberately.")
    for d in DETAIL_SUBDIRS:
        (root / "_details" / d).mkdir(parents=True, exist_ok=True)

    bg = background_splits(cfg)
    for eid in todo:
        n = enc_name(eid)
        if (root / n).is_dir():
            log(f"{n}: already complete, skipped")
            continue
        for p in _leftovers(root, eid):
            if p.exists():
                log(f"{n}: removing leftover of an interrupted build: {p.relative_to(root)}")
                _remove(p, root)
        _build_encounter(cfg, det_cfg, root, eid, by_enc[eid], manifest, bg, ident, device, log)
        write_dataset_info(cfg, root)
    work = root / "_details" / "work"
    if work.is_dir() and not any(work.iterdir()):
        work.rmdir()
    write_readme(root)
    return write_dataset_info(cfg, root)


def _build_encounter(cfg: Cfg, det_cfg, root: Path, eid: int, files: list[Path], manifest: dict, bg: dict,
                     ident: dict, device, log) -> None:
    n = enc_name(eid)
    keep = bool(cfg["dataset"]["keep_detector_output"])
    work_root = root / "_details" / "work"
    log(f"{n}: processing {len(files)} files")
    t0 = time.perf_counter()
    summary = run_pipeline(cfg, RunOptions(run_id=n, files=files, storage_mode="normal" if keep else "minimal",
                                           device=device, note=f"dataset {n}"), det_cfg=det_cfg, runs_dir=work_root,
                           progress=lambda m: log(f"{n} {m}"))
    work = work_root / n
    detector = json.loads((work / "run_manifest.json").read_text(encoding="utf-8"))["detector"]
    part = root / f"{n}.partial"
    part.mkdir()
    packed = _pack(cfg, work, part, eid, files, manifest, bg)
    _move_details(work, root, n, packed, keep)
    _write_json(root / "_details" / "logs" / f"{n}_entry.json", {
        "encounter": eid, "identity": ident, "detector": detector, "code": code_version(),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(files), "files_ok": summary["ok"], "files_failed": summary["failed"],
        "failed_files": [f["stem"] for f in summary["failed_files"]], "unexpected_failures": packed["unexpected_failures"],
        "n_images": packed["n_images"], "n_contours": packed["n_contours"], "n_tracks": summary["n_tracks"],
        "track_rejections": summary["track_rejections"], "s8_rejections": summary["s8_rejections"],
        "build_time_s": round(time.perf_counter() - t0, 1)})
    _remove(work, root)
    os.replace(part, root / n)                     # the encounter folder appears only now, complete
    log(f"{n}: done, {summary['ok']}/{len(files)} files OK, {packed['n_images']} images"
        + (f", FAILED: {[f['stem'] for f in summary['failed_files']]}" if summary["failed"] else ""))


def _pack(cfg: Cfg, work: Path, part: Path, eid: int, files: list[Path], manifest: dict, bg: dict) -> dict:
    """Per-file outputs of the work run -> the four encounter files (images are streamed, never all in RAM)."""
    p = cfg["paper"]
    hop_s = p["hop"] / p["fs_target_hz"]
    metas = {}
    for f in files:
        ok = work / "meta" / f"{f.stem}.json"
        metas[f.stem] = json.loads((ok if ok.exists() else work / "failed" / f.stem / "meta.json").read_text(encoding="utf-8"))
    n_img = sum(m["stage_diagnostics"]["s9"]["n_images"] for m in metas.values() if m["status"] == "OK")
    images = np.lib.format.open_memmap(part / "images.npy", mode="w+", dtype=np.float32,
                                       shape=(n_img, p["image_height"], p["image_width"]))
    img_rows, file_rows, first = [], [], {}
    con: dict[str, list] = {k: [] for k in ("source_file", "track_id", "start_frame", "n_frames", "freq_hz", "conf",
                                            "observed_fraction", "image_index")}
    i = c = 0
    for f in files:
        s, m = f.stem, metas[f.stem]
        k = nv = 0
        s8_rej: dict = {}
        if m["status"] == "OK":
            fin = work / "final" / s
            v = np.load(fin / "vectors.npz")
            nv = len(v["track_ids"])
            cid = {int(t): c + j for j, t in enumerate(v["track_ids"])}
            image_of = np.full(nv, -1, np.int64)
            imgs = np.load(fin / "images.npy")
            k = len(imgs)
            images[i:i + k] = imgs
            rows = _read_csv(fin / "images_meta.csv") if k else []
            if len(rows) != k:
                raise RuntimeError(f"{s}: images_meta rows {len(rows)} != images {k}")
            for r in rows:
                j, nf, sf0 = int(r["image_index"]), int(r["n_frames"]), int(r["start_frame"])
                ci = cid[int(r["track_id"])]
                image_of[ci - c] = i + j
                img_rows.append({"index": i + j, "encounter": eid, "source_file": s, "index_in_file": j, "contour_id": ci,
                                 "start_s": round(sf0 * hop_s, 4), "end_s": round((sf0 + nf) * hop_s, 4),
                                 "duration_ms": round(nf * hop_s * 1e3, 2), "n_frames": nf,
                                 "f_min_hz": r["f_min_hz"], "f_max_hz": r["f_max_hz"], "center_hz": r["f_center_hz"],
                                 "bandwidth_hz": r["bandwidth_hz"], "mean_conf": r["mean_conf"],
                                 "observed_fraction": r["observed_fraction"], "image_col0": r["col0"]})
            con["source_file"] += [s] * nv
            con["track_id"].append(v["track_ids"])
            con["start_frame"].append(v["start_frames"])
            con["n_frames"].append(np.diff(v["offsets"]))
            con["freq_hz"].append(v["freq_hz"])
            con["conf"].append(v["conf"])
            con["observed_fraction"].append(v["observed_fraction"])
            con["image_index"].append(image_of)
            s8_rej = m["stage_diagnostics"]["s8"]["rejected_by_reason"]
        raw, mrow = m.get("raw", {}), manifest.get(f.name, {})
        first[s] = i if k else None
        file_rows.append({"file": s, "encounter": eid, "status": m["status"], "n_images": k,
                          "first_image_index": i if k else "", "n_contours": nv,
                          "n_too_long": s8_rej.get("too_long", 0), "n_too_wide": s8_rej.get("too_wide", 0),
                          "duration_s": mrow.get("duration_seconds", ""), "manifest_status": mrow.get("validation_status", ""),
                          "detector_background_split": bg.get(f.name, ""),
                          "sha256_ok": _YES_NO[raw.get("sha256_matches_manifest")],
                          "raw_unchanged": _YES_NO[raw.get("unchanged_after_processing")],
                          "time_s": m.get("total_time_s", ""), "error": m.get("error", "")})
        i += k
        c += nv
    images.flush()
    del images

    def cat(key, dtype):
        return np.concatenate(con[key]).astype(dtype) if con[key] else np.zeros(0, dtype)

    lens = cat("n_frames", np.int64)
    start = cat("start_frame", np.int64)
    image_index = cat("image_index", np.int64)
    # S8 checks length first, so a contour without image is too long if it exceeds the image width, else too wide
    no_image = np.where(image_index >= 0, "", np.where(lens > p["image_width"], "too_long", "too_wide"))
    np.savez_compressed(part / "contours.npz",
                        source_file=np.array(con["source_file"], dtype=str), track_id=cat("track_id", np.int64),
                        start_frame=start, start_s=np.round(start * hop_s, 4), n_frames=lens,
                        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                        freq_hz=cat("freq_hz", np.float32), conf=cat("conf", np.float32),
                        observed_fraction=cat("observed_fraction", np.float32), image_index=image_index,
                        no_image_reason=no_image.astype(str), frame_s=np.float64(hop_s), encounter=np.int64(eid))
    _write_csv(part / "images.csv", IMAGE_COLUMNS, img_rows)
    _write_csv(part / "files.csv", FILE_COLUMNS, file_rows)
    unexpected = [r["file"] for r in file_rows if r["status"] != "OK" and r["manifest_status"] not in EXPECTED_FAILURE_STATUSES]
    return {"n_images": n_img, "n_contours": c, "metas": metas, "first_image_index": first, "unexpected_failures": unexpected}


def _move_details(work: Path, root: Path, n: str, packed: dict, keep: bool) -> None:
    d = root / "_details"
    for src, dst in (("logs/run.log", "run.log"), ("logs/stage_stats.jsonl", "stage_stats.jsonl"),
                     ("run_manifest.json", "run_manifest.json"), ("summary.json", "summary.json")):
        if (work / src).exists():
            os.replace(work / src, d / "logs" / f"{n}_{dst}")
    if any((work / "failed").iterdir()):
        os.replace(work / "failed", d / "failed" / n)
    if keep and any((work / "intermediate").iterdir()):
        os.replace(work / "intermediate", d / "detector_output" / n)
    per_file = d / "per_file" / n
    per_file.mkdir(parents=True)
    for stem, m in packed["metas"].items():
        if m["status"] == "OK":
            m["outputs"] = {"images": f"{n}/images.npy", "first_image_index": packed["first_image_index"][stem],
                            "n_images": m["stage_diagnostics"]["s9"]["n_images"], "contours": f"{n}/contours.npz",
                            "detector_output": f"_details/detector_output/{n}/{stem}/" if keep else None}
        else:
            m["outputs"] = {"failed": f"_details/failed/{n}/{stem}/"}
        _write_json(per_file / f"{stem}.json", m)


# ----------------------------------------------------------------------------- dataset-level records
def write_dataset_info(cfg: Cfg, root: Path) -> dict:
    from .representation import row_to_freq_hz

    entries = complete_entries(root)
    p = cfg["paper"]
    hop_ms = p["hop"] / p["fs_target_hz"] * 1e3
    first = next(iter(entries.values()), None)
    manifest_csv = project_path(cfg, "manifest_csv")
    tot = lambda key: sum(e[key] for e in entries.values())   # noqa: E731
    info = {
        "description": "Whistle images, one folder per encounter (see README.txt). Encounter ids are the DCLDE 2022 "
                       "site-1705 encounter numbers (metadata/whale_encounters_1705.csv).",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "totals": {"encounters": len(entries), "files": tot("n_files"), "files_ok": tot("files_ok"),
                   "files_failed": tot("files_failed"),
                   "files_failed_unexpected": sum(len(e["unexpected_failures"]) for e in entries.values()),
                   "images": tot("n_images"), "contours": tot("n_contours")},
        "encounters": {enc_name(k): {x: e[x] for x in ("n_files", "files_ok", "files_failed", "failed_files",
                                                       "unexpected_failures", "n_images", "n_contours", "n_tracks",
                                                       "track_rejections", "s8_rejections", "built_utc", "build_time_s",
                                                       "code")} for k, e in entries.items()},
        "image_format": {
            "array": f"images.npy: (N, {p['image_height']}, {p['image_width']}) float32, values 0..1, every image has max 1",
            "rows": f"frequency relative to the whistle centre frequency, {p['freq_resolution_hz']} Hz per row: "
                    f"row 0 = {row_to_freq_hz(0, cfg):+.2f} Hz (top), row {p['image_height'] // 2 - 1} = 0 Hz, "
                    f"row {p['image_height'] - 1} = {row_to_freq_hz(p['image_height'] - 1, cfg):+.2f} Hz",
            "columns": f"time, {hop_ms:.4f} ms per column ({p['image_width'] * hop_ms:.1f} ms in total), whistle centred; "
                       "images.csv image_col0 = first column of the whistle",
            "absolute_frequency": "images.csv center_hz (and f_min_hz / f_max_hz)",
        },
        "detector": first["detector"] if first else None,
        "processing_config_hash": first["identity"]["processing_config_hash"] if first else None,
        "manifest": {"path": _rel(manifest_csv), "sha256": audio.sha256_file(manifest_csv) if manifest_csv.exists() else None},
        "pipeline_config": cfg.to_dict(),
        "software": software_versions(),
    }
    _write_json(Path(root) / "dataset_info.json", info)
    return info


README_TEXT = """\
PREPROCESSED WHISTLE DATASET
============================

One folder per encounter. Every encounter folder holds the same 4 files:

  encounter_<id>/images.npy    ALL whistle images of the encounter: array (N, 128, 128), float32, values 0..1
  encounter_<id>/images.csv    one row per image, same order (row i describes images.npy[i])
  encounter_<id>/contours.npz  the whistle contours (time + frequency) the images were made from
  encounter_<id>/files.csv     one row per raw audio file: OK/FAILED, number of images, ...

  dataset_info.json            how the data was made (detector, settings, code version, date) + counts per encounter
  _details/                    NOT needed for training: logs, per-file records, failures, detector output

Image layout
  rows    = frequency relative to the whistle's centre frequency, 93.75 Hz per row
            (row 0 = top = highest, 0 Hz on row 63); the absolute centre frequency is images.csv center_hz
  columns = time, 5.333 ms per column (683 ms in total); the whistle is centred in time

Times
  The analysis frames are 10.667 ms long and start every 5.333 ms (frame k starts at k x 5.333 ms in the file).
  start_s = start of the whistle's first frame, end_s = start_s + duration_ms (accurate to about +/-10 ms).

images.csv columns
  index              row number in images.npy
  encounter          encounter id
  source_file        raw audio file the whistle came from
  index_in_file      image number within that file
  contour_id         entry in contours.npz
  start_s, end_s     position of the whistle in the source file (seconds, see Times)
  duration_ms        whistle length
  n_frames           whistle length in 5.333 ms frames
  f_min_hz, f_max_hz, center_hz, bandwidth_hz   frequency range of the whistle (absolute, Hz)
  mean_conf, observed_fraction                  detector confidence and fraction of directly detected frames
  image_col0         first image column holding the whistle

contours.npz (one entry per contour)
  freq_hz[offsets[k]:offsets[k+1]]   frequency (Hz) of contour k, one value per 5.333 ms frame
  image_index        row in images.npy, or -1 if the contour got no image:
  no_image_reason    "too_long" (longer than 128 frames = 683 ms) or "too_wide" (wider than the 12 kHz image band)
  source_file, track_id, start_frame, start_s, n_frames, conf, observed_fraction

files.csv columns
  file, encounter, status (OK/FAILED), n_images, first_image_index, n_contours,
  n_too_long, n_too_wide       contours that got no image (see contours.npz no_image_reason)
  duration_s                   from the file header
  manifest_status              VALID / SHORT / SUSPECT_TRUNCATED / INVALID / UNREADABLE (metadata/manifest_all.csv);
                               files that are not VALID/SHORT are expected to fail (e.g. a cut-off download)
  detector_background_split    train / val / test if quiet parts of this file were used as background in the
                               detector's synthetic data (train = seen during detector training), else empty
  sha256_ok, raw_unchanged, time_s, error

Loading
  import numpy as np, pandas as pd
  X    = np.load("preprocessed/encounter_191/images.npy", mmap_mode="r")   # not read into RAM at once
  info = pd.read_csv("preprocessed/encounter_191/images.csv")             # info.iloc[i] describes X[i]

Built by: python whistle.py build-dataset      Checked by: python whistle.py validate-dataset
"""


def write_readme(root: Path) -> None:
    (Path(root) / "README.txt").write_text(README_TEXT, encoding="utf-8")


# ----------------------------------------------------------------------------- validation
def validate_dataset(root: Path, cfg: Cfg | None = None, chunk: int = 4096) -> dict:
    """Automated checks of a built dataset (pure inspection of what is on disk). With ``cfg`` it also checks that
    every configured encounter is built and that each encounter's files.csv matches its raw folder today."""
    root = Path(root)
    R: dict = {"dataset": str(root), "problems": [], "expected_failures": [], "encounters": {}}
    bad = R["problems"].append
    if not root.is_dir():
        bad(f"{root} does not exist")
        R["pass"] = False
        return R
    leftovers = [q.name for q in root.glob("*.partial")] + (["_details/work"] if (root / "_details" / "work").exists() else [])
    if leftovers:
        bad(f"leftovers of an interrupted build: {leftovers} (run build-dataset again)")
    entries = complete_entries(root)
    if len({json.dumps(e["identity"], sort_keys=True) for e in entries.values()}) > 1:
        bad("encounters were built with different processing configs/detectors")
    by_enc = audio.files_by_encounter(cfg) if cfg is not None else {}
    for eid in by_enc:
        if not (root / enc_name(eid)).is_dir():
            bad(f"{enc_name(eid)}: configured but not built")
    for d in sorted(q for q in root.glob("encounter_*") if q.is_dir() and not q.name.endswith(".partial")):
        try:
            eid = int(d.name.split("_", 1)[1])
        except ValueError:
            bad(f"{d.name}: unexpected folder")
            continue
        missing = [f for f in ENCOUNTER_FILES if not (d / f).exists()]
        if missing:
            bad(f"{d.name}: missing {missing}")
            continue
        if eid not in entries:
            bad(f"{d.name}: no build record (_details/logs/{d.name}_entry.json)")
        X = np.load(d / "images.npy", mmap_mode="r")
        n = len(X)
        if X.ndim != 3 or X.shape[1:] != (128, 128) or X.dtype != np.float32:
            bad(f"{d.name}: images {X.shape} {X.dtype}, expected (N, 128, 128) float32")
        for a in range(0, n, chunk):
            b = np.asarray(X[a:a + chunk])
            if not np.isfinite(b).all() or b.min() < 0 or b.max() > 1:
                bad(f"{d.name}: image values outside [0,1] or non-finite (images {a}..{a + len(b) - 1})")
                break
            if np.abs(b.reshape(len(b), -1).max(axis=1) - 1).max() > 1e-5:
                bad(f"{d.name}: an image has max != 1 (images {a}..{a + len(b) - 1})")
                break
        del X
        rows = _read_csv(d / "images.csv")
        if len(rows) != n:
            bad(f"{d.name}: images.csv rows {len(rows)} != images {n}")
        if [int(r["index"]) for r in rows] != list(range(len(rows))):
            bad(f"{d.name}: images.csv index is not 0..N-1 in order")
        if any(int(r["encounter"]) != eid for r in rows):
            bad(f"{d.name}: images.csv holds rows of another encounter")
        C = np.load(d / "contours.npz")
        nc = len(C["track_id"])
        if len(C["offsets"]) != nc + 1 or int(C["offsets"][-1]) != len(C["freq_hz"]):
            bad(f"{d.name}: contours.npz offsets inconsistent")
        cid = np.array([int(r["contour_id"]) for r in rows], np.int64)
        if len(cid) and (cid.min() < 0 or cid.max() >= nc or (C["image_index"][cid] != np.arange(len(cid))).any()):
            bad(f"{d.name}: images.csv contour_id and contours.npz image_index disagree")
        if int((C["image_index"] >= 0).sum()) != n:
            bad(f"{d.name}: contours with an image {int((C['image_index'] >= 0).sum())} != images {n}")
        files = _read_csv(d / "files.csv")
        if sum(int(f["n_images"]) for f in files) != n:
            bad(f"{d.name}: files.csv n_images sum != images {n}")
        no_img = {r: int((C["no_image_reason"] == r).sum()) for r in ("too_long", "too_wide")}
        if no_img != {r: sum(int(f[f"n_{r}"]) for f in files) for r in no_img}:
            bad(f"{d.name}: contours without image {no_img} disagree with files.csv n_too_long / n_too_wide")
        C.close()
        failed = [f for f in files if f["status"] != "OK"]
        R["expected_failures"] += [f"{d.name}/{f['file']} ({f['manifest_status']}): {f['error']}"
                                   for f in failed if f["manifest_status"] in EXPECTED_FAILURE_STATUSES]
        unexpected = [f["file"] for f in failed if f["manifest_status"] not in EXPECTED_FAILURE_STATUSES]
        if unexpected:
            bad(f"{d.name}: {len(unexpected)} files failed unexpectedly: {unexpected[:5]}")
        if any(f["sha256_ok"] == "NO" for f in files):
            bad(f"{d.name}: a raw file hash differs from the manifest")
        if any(f["raw_unchanged"] == "NO" for f in files):
            bad(f"{d.name}: a raw file changed during processing!")
        if eid in entries and entries[eid]["n_images"] != n:
            bad(f"{d.name}: build record says {entries[eid]['n_images']} images, folder has {n}")
        if eid in by_enc and (msg := raw_changes(d, by_enc[eid])):
            bad(msg)
        R["encounters"][d.name] = {"files": len(files), "failed": len(failed), "images": n, "contours": nc}
    R["n_images_total"] = sum(e["images"] for e in R["encounters"].values())
    R["pass"] = not R["problems"]
    return R
