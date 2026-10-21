"""Multi-encounter raw folders, manifest, and the final per-encounter dataset layout (preprocessed/)."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from whistlepipe import audio
from whistlepipe.config import band_bins
from whistlepipe.dataset import ENCOUNTER_FILES, FILE_COLUMNS, build_dataset, validate_dataset
from whistlepipe.detector import RidgeNet
from whistlepipe.detector.ridgenet import RidgeDetector
from whistlepipe.manifest import build_manifest
from whistlepipe.storage import RawDataGuardError

LONG_RIDGE_FILE = "1705_20170101_000100_000.flac"     # the stand-in S5 draws a ridge longer than 128 frames here


def _write_flac(path: Path, seconds: float = 1.0):
    fs = 500000
    t = np.arange(int(fs * seconds)) / fs
    x = np.stack([0.2 * np.sin(2 * np.pi * (11000 + 2000 * c) * t) for c in range(6)], axis=1)
    sf.write(str(path), (x * 32767).astype(np.int16), fs, subtype="PCM_16", format="FLAC")


def _write_truncated_flac(path: Path):
    """A FLAC cut to 1/8 of its bytes (like an interrupted download): the header still states the full length."""
    tmp = path.with_suffix(".tmp")
    _write_flac(tmp)
    b = tmp.read_bytes()
    tmp.unlink()
    path.write_bytes(b[:len(b) // 8])


def _ridge_predict(self, spec):
    """Stand-in S5: one clean ridge per file, 100 frames (-> image), or 150 frames (-> too long) in LONG_RIDGE_FILE."""
    from whistlepipe.detector.interface import DetectionMap

    lo, hi = self.band
    conf = np.zeros((spec.n_frames, hi - lo + 1), np.float32)
    k = np.arange(conf.shape[1])[None, :]
    n = 150 if spec.name == LONG_RIDGE_FILE else 100
    t = np.arange(20, 20 + n)
    conf[t] = 0.95 * np.exp(-0.5 * ((k - (60 + 30 * (t - 20) / (n - 1))[:, None]) / 0.7) ** 2)
    return DetectionMap(conf, lo, hi, spec.n_bins, self.name, self.version, dict(self.meta))


@pytest.fixture()
def enc_env(cfg, tmp_path, monkeypatch):
    """Two encounter folders (901: 2 files, 902: 1 file), a manifest, a random-weight checkpoint, ridge stand-in S5,
    and a background bank listing one 902 file as a 'train' background."""
    files = {901: ["1705_20170101_000000_000.flac", LONG_RIDGE_FILE], 902: ["1705_20170202_000000_000.flac"]}
    dirs = []
    for eid, names in files.items():
        d = tmp_path / f"encounter_{eid}_audio"
        d.mkdir()
        for nm in names:
            _write_flac(d / nm)
        dirs.append(str(d))
    torch.manual_seed(0)
    m = RidgeNet(16)
    ck = tmp_path / "ck.pt"
    torch.save({"model_state": m.state_dict(), "arch": {"channels": 16, "dilations": [list(x) for x in m.dilations]},
                "band": list(band_bins(cfg)), "conditioning": dict(cfg["decisions"]["s5"]["conditioning"]),
                "version": "test-random", "seed": 0, "metrics": {}}, ck)
    bank = tmp_path / "data" / "synthetic" / "backgrounds"
    bank.mkdir(parents=True)
    (bank / "train.json").write_text(json.dumps({"split": "train", "files": ["1705_20170202_000000_000.flac"]}))
    cfg["paths"].update(raw_dirs=dirs, synth_raw_dir=dirs[0], runs_dir=str(tmp_path / "checks"),
                        dataset_dir=str(tmp_path / "preprocessed"), manifest_csv=str(tmp_path / "manifest_all.csv"),
                        manifest_363_csv=str(tmp_path / "no_363.csv"), data_dir=str(tmp_path / "data"))
    cfg["decisions"]["s5"].update(checkpoint=str(ck), device="cpu")
    cfg["dataset"]["expected_file_duration_s"] = 1.0
    monkeypatch.setattr(RidgeDetector, "predict", _ridge_predict)
    build_manifest(cfg, log=quiet)
    return cfg, tmp_path


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def _rebuild_manifest(cfg):
    Path(cfg["paths"]["manifest_csv"]).unlink()
    build_manifest(cfg, log=quiet)


def quiet(*a):
    pass


# ------------------------------------------------------------------ raw folders + manifest
def test_encounter_discovery(enc_env):
    cfg, _ = enc_env
    by = audio.files_by_encounter(cfg)
    assert list(by) == [901, 902] and [len(v) for v in by.values()] == [2, 1]
    allf = audio.list_all_raw_files(cfg)
    assert [p.name for p in allf] == sorted(p.name for p in allf) and len(allf) == 3
    assert audio.encounter_of(allf[-1]) == 902 and audio.find_raw_file(cfg, allf[0].stem) == allf[0]


def test_duplicate_file_name_across_encounters_refused(enc_env):
    cfg, tmp = enc_env
    _write_flac(tmp / "encounter_902_audio" / "1705_20170101_000000_000.flac")
    with pytest.raises(ValueError, match="two folders"):
        audio.list_all_raw_files(cfg)


def test_manifest_rows_status_and_no_overwrite(enc_env):
    cfg, tmp = enc_env
    rows = _csv(cfg["paths"]["manifest_csv"])
    assert [int(r["encounter_id"]) for r in rows] == [901, 901, 902]
    assert all(r["validation_status"] == "VALID" and r["timestamp_utc"].startswith("2017-0") for r in rows)
    assert rows[0]["sha256"] == _sha(tmp / "encounter_901_audio" / rows[0]["filename"])
    with pytest.raises(FileExistsError):
        build_manifest(cfg, log=quiet)
    cfg["dataset"]["expected_file_duration_s"] = 60.0
    _rebuild_manifest(cfg)
    assert {r["validation_status"] for r in _csv(cfg["paths"]["manifest_csv"])} == {"SHORT"}


def test_manifest_cross_check_against_363(enc_env):
    cfg, tmp = enc_env
    Path(cfg["paths"]["manifest_csv"]).unlink()
    old = tmp / "m363.csv"
    old.write_text(f"index,filename,sha256\n0,1705_20170101_000000_000.flac,{'0' * 64}\n")
    cfg["paths"]["manifest_363_csv"] = str(old)
    with pytest.raises(RuntimeError, match="differs from manifest_363"):
        build_manifest(cfg, log=quiet)


# ------------------------------------------------------------------ truncated raw files
def test_s1_refuses_a_truncated_file(cfg, tmp_path):
    p = tmp_path / "1705_20170101_000000_000.flac"
    _write_truncated_flac(p)
    assert sf.info(str(p)).frames == 500000                                 # the header still claims the full length
    with pytest.raises(ValueError, match="truncated or corrupt"):
        audio.s1_select_channel(p, cfg)


def test_truncated_file_flagged_in_manifest_and_expected_failure(enc_env):
    cfg, tmp = enc_env
    bad = tmp / "encounter_902_audio" / "1705_20170202_000100_000.flac"
    _write_truncated_flac(bad)
    _rebuild_manifest(cfg)
    status = {r["filename"]: r["validation_status"] for r in _csv(cfg["paths"]["manifest_csv"])}
    assert status[bad.name] == "SUSPECT_TRUNCATED" and list(status.values()).count("VALID") == 3
    info = build_dataset(cfg, log=quiet)
    assert info["totals"]["files_failed"] == 1 and info["totals"]["files_failed_unexpected"] == 0
    row = [f for f in _csv(tmp / "preprocessed" / "encounter_902" / "files.csv") if f["file"] == bad.stem][0]
    assert row["status"] == "FAILED" and row["manifest_status"] == "SUSPECT_TRUNCATED" and "truncated" in row["error"]
    r = validate_dataset(tmp / "preprocessed", cfg)
    assert r["pass"], r["problems"]
    assert len(r["expected_failures"]) == 1 and bad.stem in r["expected_failures"][0]


# ------------------------------------------------------------------ dataset layout
def test_dataset_layout_and_contents(enc_env):
    cfg, tmp = enc_env
    raw_before = {p: (_sha(p), p.stat().st_mtime_ns) for p in audio.list_all_raw_files(cfg)}
    lines = []
    info = build_dataset(cfg, log=lines.append)
    root = tmp / "preprocessed"
    assert {p.name for p in root.iterdir()} == {"README.txt", "dataset_info.json", "encounter_901", "encounter_902", "_details"}
    assert {p.name for p in (root / "_details").iterdir()} == {"logs", "per_file", "failed", "detector_output"}   # work/ removed
    for eid, n_files, n_img in ((901, 2, 1), (902, 1, 1)):
        d = root / f"encounter_{eid}"
        assert sorted(p.name for p in d.iterdir()) == sorted(ENCOUNTER_FILES)
        X = np.load(d / "images.npy")
        assert X.shape == (n_img, 128, 128) and X.dtype == np.float32 and np.allclose(X.reshape(n_img, -1).max(1), 1)
        rows = _csv(d / "images.csv")
        assert [int(r["index"]) for r in rows] == list(range(n_img)) and {r["encounter"] for r in rows} == {str(eid)}
        assert float(rows[0]["duration_ms"]) == pytest.approx(100 * 1024 / 192, abs=0.01)
        files = _csv(d / "files.csv")
        assert list(files[0]) == FILE_COLUMNS and len(files) == n_files
        assert all(f["status"] == "OK" and f["sha256_ok"] == "yes" and f["raw_unchanged"] == "yes" for f in files)
        C = np.load(d / "contours.npz")
        assert list(C["source_file"]) == [f["file"] for f in files] and C["offsets"][-1] == len(C["freq_hz"])
        for f in files:
            meta = json.loads((root / "_details" / "per_file" / f"encounter_{eid}" / f"{f['file']}.json").read_text())
            assert meta["encounter_id"] == eid and meta["outputs"]["images"] == f"encounter_{eid}/images.npy"
            assert (root / "_details" / "detector_output" / f"encounter_{eid}" / f["file"] / "s6_tracks.npz").exists()
        for suffix in ("run.log", "stage_stats.jsonl", "run_manifest.json", "summary.json", "entry.json"):
            assert (root / "_details" / "logs" / f"encounter_{eid}_{suffix}").exists()
    # 901: the long ridge (150 frames) got no image and is counted as too long, not as too wide
    f901 = {f["file"]: f for f in _csv(root / "encounter_901" / "files.csv")}
    long_ = f901[Path(LONG_RIDGE_FILE).stem]
    assert (long_["n_images"], long_["n_contours"], long_["n_too_long"], long_["n_too_wide"]) == ("0", "1", "1", "0")
    assert long_["first_image_index"] == "" and f901["1705_20170101_000000_000"]["first_image_index"] == "0"
    C = np.load(root / "encounter_901" / "contours.npz")
    assert list(C["image_index"]) == [0, -1] and list(C["no_image_reason"]) == ["", "too_long"]
    # detector-background marking
    assert _csv(root / "encounter_902" / "files.csv")[0]["detector_background_split"] == "train"
    assert {f["detector_background_split"] for f in f901.values()} == {""}
    assert info["totals"] == {"encounters": 2, "files": 3, "files_ok": 3, "files_failed": 0, "files_failed_unexpected": 0,
                              "images": 2, "contours": 3}
    assert info["detector"]["version"] == "test-random" and len(info["processing_config_hash"]) == 64
    assert set(info["encounters"]["encounter_901"]["code"]) == {"git_commit", "git_uncommitted_code_changes"}
    assert any("[1/2] OK" in m and m.startswith("encounter_901") for m in lines)        # per-file progress reaches the log
    r = validate_dataset(root, cfg)
    assert r["pass"], r["problems"]
    assert {p: (_sha(p), p.stat().st_mtime_ns) for p in raw_before} == raw_before                 # raw untouched


def test_detector_output_can_be_dropped(enc_env):
    cfg, tmp = enc_env
    cfg["dataset"]["keep_detector_output"] = False
    build_dataset(cfg, log=quiet)
    assert not any((tmp / "preprocessed" / "_details" / "detector_output").iterdir())


def test_resume_skips_complete_and_redoes_interrupted(enc_env):
    cfg, tmp = enc_env
    root = tmp / "preprocessed"
    build_dataset(cfg, encounters=[901], log=quiet)
    assert (root / "encounter_901").is_dir() and not (root / "encounter_902").exists()
    assert not validate_dataset(root, cfg)["pass"]                                  # 902 configured but not built
    t901 = (root / "encounter_901" / "images.npy").stat().st_mtime_ns
    (root / "encounter_902.partial").mkdir()                                         # simulate a crash in 902
    (root / "_details" / "work" / "encounter_902").mkdir(parents=True)
    info = build_dataset(cfg, log=quiet)
    assert info["totals"]["encounters"] == 2 and not (root / "encounter_902.partial").exists()
    assert (root / "encounter_901" / "images.npy").stat().st_mtime_ns == t901       # finished encounter not rebuilt
    assert validate_dataset(root, cfg)["pass"]


def test_adding_an_encounter_later_keeps_the_dataset_compatible(enc_env):
    cfg, tmp = enc_env
    root = tmp / "preprocessed"
    build_dataset(cfg, log=quiet)
    t901 = (root / "encounter_901" / "images.npy").stat().st_mtime_ns
    d = tmp / "encounter_903_audio"
    d.mkdir()
    _write_flac(d / "1705_20170303_000000_000.flac")
    cfg["paths"]["raw_dirs"] = cfg["paths"]["raw_dirs"] + [str(d)]                 # changes paths, not the processing
    cfg["decisions"]["s5"]["device"] = "auto"                                       # device is not part of the identity
    _rebuild_manifest(cfg)
    info = build_dataset(cfg, log=quiet)
    assert info["totals"]["encounters"] == 3 and (root / "encounter_901" / "images.npy").stat().st_mtime_ns == t901
    assert validate_dataset(root, cfg)["pass"]


def test_raw_folder_changed_after_build_is_detected(enc_env):
    cfg, tmp = enc_env
    root = tmp / "preprocessed"
    build_dataset(cfg, log=quiet)
    (tmp / "encounter_901_audio" / LONG_RIDGE_FILE).unlink()
    with pytest.raises(RuntimeError, match="raw folders changed"):
        build_dataset(cfg, log=quiet)
    r = validate_dataset(root, cfg)
    assert not r["pass"] and any("raw folder now has 1" in p for p in r["problems"])


def test_mixing_processing_configs_in_one_dataset_refused(enc_env):
    cfg, _ = enc_env
    build_dataset(cfg, encounters=[901], log=quiet)
    cfg["decisions"]["s6"]["peak_threshold"] = 0.6
    with pytest.raises(RuntimeError, match="another processing config/detector"):
        build_dataset(cfg, encounters=[902], log=quiet)


def test_files_missing_from_manifest_refused(enc_env):
    cfg, tmp = enc_env
    _write_flac(tmp / "encounter_902_audio" / "1705_20170202_000100_000.flac")
    with pytest.raises(RuntimeError, match="build-manifest"):
        build_dataset(cfg, log=quiet)
    assert audio.missing_from_manifest(cfg, audio.list_all_raw_files(cfg)) == ["1705_20170202_000100_000.flac"]


def test_unexpected_failure_is_recorded_not_packed(enc_env, monkeypatch):
    from whistlepipe import dsp

    cfg, tmp = enc_env
    real = dsp.s3_highpass
    bad = "1705_20170101_000000_000.flac"

    def boom(sig, c):
        if sig.name == bad:
            raise RuntimeError("injected failure")
        return real(sig, c)

    monkeypatch.setattr(dsp, "s3_highpass", boom)
    info = build_dataset(cfg, encounters=[901], log=quiet)
    root = tmp / "preprocessed"
    files = _csv(root / "encounter_901" / "files.csv")
    assert [f["status"] for f in files] == ["FAILED", "OK"] and "injected failure" in files[0]["error"]
    assert np.load(root / "encounter_901" / "images.npy").shape[0] == 0                # the other file has no image
    assert info["totals"]["files_failed"] == 1 and info["totals"]["files_failed_unexpected"] == 1
    assert (root / "_details" / "failed" / "encounter_901" / Path(bad).stem / "traceback.txt").exists()
    r = validate_dataset(root)
    assert not r["pass"] and any("failed unexpectedly" in p for p in r["problems"])


def test_dataset_inside_raw_refused(enc_env):
    cfg, tmp = enc_env
    cfg["paths"]["dataset_dir"] = str(tmp / "encounter_901_audio" / "out")
    with pytest.raises(RawDataGuardError):
        build_dataset(cfg, log=quiet)


def test_corrupted_dataset_detected(enc_env):
    cfg, tmp = enc_env
    build_dataset(cfg, log=quiet)
    p = tmp / "preprocessed" / "encounter_902" / "images.npy"
    X = np.load(p)
    X[0] *= 3
    np.save(p, X)
    r = validate_dataset(tmp / "preprocessed")
    assert not r["pass"] and any("outside [0,1]" in q for q in r["problems"])


def test_check_runs_cannot_process_all_files():
    from whistlepipe.cli import main

    with pytest.raises(SystemExit):
        main(["run", "--run-id", "x", "--set", "all"])
