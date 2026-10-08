"""Orchestrator: lineage, storage policy, failure retention, no-overwrite, raw immutability, resume."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from whistlepipe import dsp
from whistlepipe.config import band_bins
from whistlepipe.detector import RidgeNet
from whistlepipe.pipeline import RunOptions, run_pipeline


@pytest.fixture()
def env(cfg, fake_raw, tmp_path):
    """cfg pointing at tmp raw/runs dirs + a random-weight detector checkpoint (pipeline plumbing needs no training)."""
    torch.manual_seed(0)
    m = RidgeNet(16)
    ck = tmp_path / "ck.pt"
    torch.save({"model_state": m.state_dict(), "arch": {"channels": 16, "dilations": [list(d) for d in m.dilations]},
                "band": list(band_bins(cfg)), "conditioning": dict(cfg["decisions"]["s5"]["conditioning"]),
                "version": "test-random", "seed": 0, "metrics": {}}, ck)
    cfg["decisions"]["s5"]["checkpoint"] = str(ck)
    cfg["decisions"]["s5"]["device"] = "cpu"
    cfg["storage"]["plots"] = False
    cfg["paths"]["manifest_csv"] = str(tmp_path / "no_manifest.csv")
    return cfg, fake_raw


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def opts(files, run_id, **kw):
    return RunOptions(run_id=run_id, files=files, verify_hash=kw.pop("verify_hash", True), **kw)


def test_full_run_lineage_and_layout(env):
    cfg, raw = env
    before = (sha(raw), raw.stat().st_mtime_ns)
    s = run_pipeline(cfg, opts([raw], "r_debug", storage_mode="debug"))
    assert s["ok"] == 1 and s["failed"] == 0
    d = Path(cfg["paths"]["runs_dir"]) / "r_debug"
    for sub in ("logs", "intermediate", "final", "meta", "failed", "diagnostics"):
        assert (d / sub).is_dir()
    stem = raw.stem
    for f in ("s1_channel.flac", "s2_resampled.npy", "s3_highpassed.npy", "s4_spectrogram_db.npy", "s5_confidence.npy", "s6_tracks.npz"):
        assert (d / "intermediate" / stem / f).exists(), f
    meta = json.loads((d / "meta" / f"{stem}.json").read_text())
    assert meta["status"] == "OK" and meta["raw"]["sha256"] == before[0] and meta["raw"]["unchanged_after_processing"] is True
    assert meta["pipeline_config_effective_hash"] == cfg["_meta"]["effective_hash"]
    assert meta["detector"]["version"] == "test-random" and len(meta["detector"]["checkpoint_sha256"]) == 64
    assert set(meta["stage_diagnostics"]) >= {"s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"}
    assert meta["stage_diagnostics"]["s4"]["out_shape"] == [186, 1025] and meta["stage_diagnostics"]["s2"]["fs_hz"] == 192000
    assert all(k in meta["timings_s"] for k in ("s1", "s2", "s3", "s4", "s5", "s6"))
    man = json.loads((d / "run_manifest.json").read_text())
    assert man["pipeline_config"]["paper"]["nfft"] == 2048 and "numpy" in man["software"] and man["detector"]["version"] == "test-random"
    imgs = np.load(d / "final" / stem / "images.npy")
    assert imgs.ndim == 3 and imgs.shape[1:] == (128, 128) and imgs.dtype == np.float32
    if len(imgs):
        assert imgs.min() >= 0 and imgs.max() <= 1
    assert (d / "final" / stem / "vectors.npz").exists() and (d / "summary.json").exists() and (d / "logs" / "stage_stats.jsonl").exists()
    assert (sha(raw), raw.stat().st_mtime_ns) == before                         # raw untouched


@pytest.mark.parametrize("mode,expect", [("debug", {"s1", "s2", "s3", "s4", "s5", "s6"}), ("normal", {"s5", "s6"}), ("minimal", set())])
def test_storage_policy_by_mode(env, mode, expect):
    cfg, raw = env
    run_pipeline(cfg, opts([raw], f"r_{mode}", storage_mode=mode))
    d = Path(cfg["paths"]["runs_dir"]) / f"r_{mode}"
    names = {"s1": "s1_channel.flac", "s2": "s2_resampled.npy", "s3": "s3_highpassed.npy", "s4": "s4_spectrogram_db.npy", "s5": "s5_confidence.npy", "s6": "s6_tracks.npz"}
    present = {k for k, f in names.items() if (d / "intermediate" / raw.stem / f).exists()}
    assert present == expect
    assert (d / "final" / raw.stem / "images.npy").exists() and (d / "meta" / f"{raw.stem}.json").exists()   # final + lineage in EVERY mode


def test_runs_never_overwrite_and_coexist(env):
    cfg, raw = env
    run_pipeline(cfg, opts([raw], "phaseA", storage_mode="minimal"))
    run_pipeline(cfg, opts([raw], "phaseB", storage_mode="minimal"))
    with pytest.raises(FileExistsError):
        run_pipeline(cfg, opts([raw], "phaseA", storage_mode="minimal"))
    runs = Path(cfg["paths"]["runs_dir"])
    assert (runs / "phaseA" / "summary.json").exists() and (runs / "phaseB" / "summary.json").exists()


def test_failure_keeps_intermediates_even_in_minimal_mode(env, monkeypatch, tmp_path):
    cfg, raw = env
    raw2 = raw.parent / "1705_20170101_000100_000.flac"
    sf.write(str(raw2), sf.read(str(raw), dtype="int16")[0], 500000, subtype="PCM_16", format="FLAC")
    real = dsp.s3_highpass

    def boom(sig, c):
        if sig.name == raw.name:
            raise RuntimeError("injected failure")
        return real(sig, c)

    monkeypatch.setattr(dsp, "s3_highpass", boom)
    s = run_pipeline(cfg, opts([raw, raw2], "r_fail", storage_mode="minimal"))
    assert s["failed"] == 1 and s["ok"] == 1 and s["failed_files"][0]["stem"] == raw.stem
    d = Path(cfg["paths"]["runs_dir"]) / "r_fail"
    fd = d / "failed" / raw.stem
    assert (fd / "traceback.txt").exists() and "injected failure" in (fd / "traceback.txt").read_text()
    assert (fd / "s1_channel.flac").exists() and (fd / "s2_resampled.npy").exists()            # intermediates up to the failing stage
    assert not (fd / "s3_highpassed.npy").exists()
    meta = json.loads((fd / "meta.json").read_text())
    assert meta["status"] == "FAILED" and "injected failure" in meta["error"] and "s1" in meta["stage_diagnostics"]
    assert (d / "meta" / f"{raw2.stem}.json").exists()                                           # the other file still processed


def test_manifest_hash_mismatch_is_a_failure(env, tmp_path):
    cfg, raw = env
    mf = tmp_path / "m.csv"
    mf.write_text(f"index,filename,sha256\n0,{raw.name},{'0' * 64}\n")
    cfg["paths"]["manifest_csv"] = str(mf)
    s = run_pipeline(cfg, opts([raw], "r_badhash", storage_mode="minimal"))
    assert s["failed"] == 1 and "hash differs" in s["failed_files"][0]["error"]
    mf.write_text(f"index,filename,sha256\n0,{raw.name},{sha(raw)}\n")
    s = run_pipeline(cfg, opts([raw], "r_goodhash", storage_mode="minimal"))
    assert s["ok"] == 1
    meta = json.loads((Path(cfg["paths"]["runs_dir"]) / "r_goodhash" / "meta" / f"{raw.stem}.json").read_text())
    assert meta["raw"]["sha256_matches_manifest"] is True


def test_stage_range_and_resume_equal_direct_run(env):
    cfg, raw = env
    run_pipeline(cfg, opts([raw], "A_s1_s4", storage_mode="debug", last="s4"))
    run_pipeline(cfg, opts([raw], "B_s5_s6", storage_mode="debug", first="s5", last="s6", from_run=Path(cfg["paths"]["runs_dir"]) / "A_s1_s4"))
    run_pipeline(cfg, opts([raw], "C_direct", storage_mode="debug", last="s6"))
    runs = Path(cfg["paths"]["runs_dir"])
    b = np.load(runs / "B_s5_s6" / "intermediate" / raw.stem / "s5_confidence.npy")
    c = np.load(runs / "C_direct" / "intermediate" / raw.stem / "s5_confidence.npy")
    assert np.allclose(b, c, atol=1e-6)
    assert not (runs / "A_s1_s4" / "intermediate" / raw.stem / "s5_confidence.npy").exists()     # stage range respected


def test_resume_requires_from_run(env):
    cfg, raw = env
    s = run_pipeline(cfg, opts([raw], "r_bad", first="s5", last="s6", storage_mode="minimal"))
    assert s["failed"] == 1


def test_run_manifest_records_options_and_overrides(env):
    cfg, raw = env
    run_pipeline(cfg, opts([raw], "r_man", storage_mode="minimal", note="unit test"))
    man = json.loads((Path(cfg["paths"]["runs_dir"]) / "r_man" / "run_manifest.json").read_text())
    assert man["options"]["note"] == "unit test" and man["storage_mode"] == "minimal" and man["n_files"] == 1


def _ridge_predict(self, spec):
    """Stand-in S5 returning one clean 100-frame ridge (keeps S6-S9 exercised without a trained model)."""
    from whistlepipe.detector.interface import DetectionMap

    lo, hi = self.band
    conf = np.zeros((spec.n_frames, hi - lo + 1), np.float32)
    k = np.arange(conf.shape[1])[None, :]
    t = np.arange(20, 120)
    kc = 60 + 30 * (t - 20) / 99
    conf[20:120] = 0.95 * np.exp(-0.5 * ((k - kc[:, None]) / 0.7) ** 2)
    return DetectionMap(conf, lo, hi, spec.n_bins, self.name, self.version, dict(self.meta))


def test_end_to_end_with_ridge_and_run_validator(env, monkeypatch):
    from whistlepipe.checks import verify_run
    from whistlepipe.detector.ridgenet import RidgeDetector

    monkeypatch.setattr(RidgeDetector, "predict", _ridge_predict)
    cfg, raw = env
    s = run_pipeline(cfg, opts([raw], "r_v", storage_mode="normal"))
    assert s["ok"] == 1 and s["n_tracks"] == 1 and s["n_vectors"] == 1 and s["n_images"] == 1
    d = Path(cfg["paths"]["runs_dir"]) / "r_v"
    r = verify_run(d)
    assert r["pass"], r["problems"]
    imgs_path = d / "final" / raw.stem / "images.npy"
    imgs = np.load(imgs_path)
    assert imgs.shape == (1, 128, 128) and abs(float(imgs.max()) - 1) < 1e-6
    from whistlepipe.representation import recover_contour

    cols, f = recover_contour(imgs[0], cfg)
    # ridge sweeps 30 bins over 100 frames: 40 frames apart => 30*93.75*40/99 = 1136 Hz
    assert f[len(f) // 2 + 20] - f[len(f) // 2 - 20] == pytest.approx(30 * 93.75 * 40 / 99, rel=0.1)
    imgs[0] *= 3.0                                                      # corrupt: breaks the [0, 1] range / max == 1 rule
    np.save(imgs_path, imgs)
    r2 = verify_run(d)
    assert not r2["pass"] and any("outside [0,1]" in p or "max != 1" in p for p in r2["problems"])
