"""Raw-file manifest for every encounter folder: metadata/manifest_all.csv (python whistle.py build-manifest).

One row per raw file: encounter, SHA-256, size, audio header, timestamp from the file name, and a status:
    VALID              500 kHz, 6 channels, full expected duration
    SHORT              correct format, header says shorter than [dataset].expected_file_duration_s (still processed)
    SUSPECT_TRUNCATED  the file is far smaller than its header length implies (< half the median bytes per sample):
                       a FLAC header keeps the full length when a file is cut off, e.g. by an interrupted download
    INVALID            wrong sampling rate or channel count
    UNREADABLE         the header cannot be read
Only the header is read here (decoding all files would take about as long as preprocessing); S1 checks the decoded
length of every file and fails a truncated one, so SUSPECT_TRUNCATED / INVALID / UNREADABLE files end up FAILED
(expected failures, listed separately by validate-dataset).
Encounter-191 hashes are cross-checked against the original manifest_363.csv; any difference aborts.
The manifest is written once and never overwritten (delete it deliberately to rebuild, e.g. after adding raw files).
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import audio
from .config import PROJECT_ROOT, Cfg, project_path

COLUMNS = ["index", "encounter_id", "filename", "relative_path", "sha256", "file_size_bytes", "sample_rate_hz",
           "channel_count", "frame_count", "duration_seconds", "timestamp_utc", "validation_status",
           "matches_manifest_363"]
EXPECTED_FAILURE_STATUSES = ("SUSPECT_TRUNCATED", "INVALID", "UNREADABLE")
SUSPECT_SIZE_FRACTION = 0.5     # [DECISION] bytes per sample below this fraction of the median -> SUSPECT_TRUNCATED


def timestamp_from_name(name: str) -> str:
    """1705_20170929_225047_062.flac -> 2017-09-29T22:50:47Z ('' if the name does not follow the pattern)."""
    parts = Path(name).stem.split("_")
    try:
        return datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S").strftime("%Y-%m-%dT%H:%M:%SZ")
    except (IndexError, ValueError):
        return ""


def header_status(info: dict, cfg: Cfg) -> str:
    p = cfg["paper"]
    if info["samplerate"] != p["fs_raw_hz"] or info["channels"] != p["n_channels_raw"]:
        return "INVALID"
    if info["frames"] < round(float(cfg["dataset"]["expected_file_duration_s"]) * p["fs_raw_hz"]):
        return "SHORT"
    return "VALID"


def flag_suspect_sizes(rows: list[dict]) -> None:
    """Mark VALID/SHORT rows whose bytes per sample are far below the median as SUSPECT_TRUNCATED (in place)."""
    ok = [r for r in rows if r["validation_status"] in ("VALID", "SHORT") and r["frame_count"]]
    if not ok:
        return
    bps = {id(r): r["file_size_bytes"] / (r["frame_count"] * r["channel_count"]) for r in ok}
    med = float(np.median(list(bps.values())))
    for r in ok:
        if bps[id(r)] < SUSPECT_SIZE_FRACTION * med:
            r["validation_status"] = "SUSPECT_TRUNCATED"


def build_manifest(cfg: Cfg, log=print) -> Path:
    out = project_path(cfg, "manifest_csv")
    if out.exists():
        raise FileExistsError(f"{out} already exists and is never overwritten (delete it deliberately to rebuild)")
    old = audio.load_manifest_hashes(project_path(cfg, "manifest_363_csv"))
    files = audio.list_all_raw_files(cfg)
    rows, t0 = [], time.perf_counter()
    for i, p in enumerate(files):
        sha = audio.sha256_file(p)
        if p.name in old and old[p.name] != sha:
            raise RuntimeError(f"{p.name}: SHA-256 differs from manifest_363.csv ({sha} != {old[p.name]})")
        row = {"index": i, "encounter_id": audio.encounter_of(p), "filename": p.name,
               "relative_path": p.relative_to(PROJECT_ROOT).as_posix() if PROJECT_ROOT in p.parents else str(p),
               "sha256": sha, "file_size_bytes": p.stat().st_size, "timestamp_utc": timestamp_from_name(p.name),
               "matches_manifest_363": "yes" if p.name in old else ""}
        try:
            info = audio.audio_info(p)
            row.update(sample_rate_hz=info["samplerate"], channel_count=info["channels"], frame_count=info["frames"],
                       duration_seconds=round(info["frames"] / info["samplerate"], 6),
                       validation_status=header_status(info, cfg))
        except Exception as e:   # noqa: BLE001  (recorded; the file will fail at S1)
            log(f"  {p.name}: header unreadable: {e!r}")
            row.update(sample_rate_hz="", channel_count="", frame_count="", duration_seconds="", validation_status="UNREADABLE")
        rows.append(row)
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            log(f"[{i + 1}/{len(files)}] hashed  ({time.perf_counter() - t0:.0f} s)")
    missing_old = sorted(set(old) - {r["filename"] for r in rows})
    if missing_old:
        raise RuntimeError(f"{len(missing_old)} files of manifest_363.csv are missing from the raw folders, e.g. {missing_old[:3]}")
    flag_suspect_sizes(rows)
    tmp = out.with_name(out.name + ".partial")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["validation_status"]] = by_status.get(r["validation_status"], 0) + 1
    log(f"wrote {out} ({len(rows)} files: {by_status})")
    for r in rows:
        if r["validation_status"] != "VALID":
            log(f"  {r['validation_status']}: {r['relative_path']} ({r['file_size_bytes'] / 1e6:.1f} MB, header "
                f"{r['duration_seconds']} s, {r['sample_rate_hz']} Hz, {r['channel_count']} ch)")
    return out
