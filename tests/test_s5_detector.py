"""S5 contract, determinism, tiling invariance, conditioning, checkpoint safety (random-weight model: no training needed)."""
import ast
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from whistlepipe import dsp
from whistlepipe.config import band_bins
from whistlepipe.detector import DetectionMap, RidgeDetector, RidgeNet, check_detection_map
from whistlepipe.detector.conditioning import condition, noise_floor_db


def random_spec(n_frames=300, seed=0):
    rng = np.random.default_rng(seed)
    db = (-25 + 4 * rng.standard_normal((n_frames, 1025))).astype(np.float32)
    db[100:180, 100] += 25          # a ridge
    return dsp.Spectrogram(db, 192000.0, 2048, 1024, 2048, "x")


def make_detector(cfg, seed=0, tile=64, halo=32, device="cpu"):
    torch.manual_seed(seed)
    s5 = cfg["decisions"]["s5"]
    return RidgeDetector(RidgeNet(16), "test-random", band_bins(cfg), s5["conditioning"], tile, halo, device)


def test_model_is_lightweight_and_rf_matches_design():
    m = RidgeNet(16)
    assert m.n_params() < 50_000
    assert m.receptive_radius() == (19, 9)


def test_band_bins_from_config(cfg):
    lo, hi = band_bins(cfg)
    assert (lo, hi) == (22, 256) and lo * 93.75 >= 2000 and hi * 93.75 <= 24000


def test_conditioning_is_level_invariant(cfg):
    spec = random_spec()
    lo, hi = band_bins(cfg)
    c = cfg["decisions"]["s5"]["conditioning"]
    a = condition(spec.db[:, lo:hi + 1], c)
    b = condition(spec.db[:, lo:hi + 1] + 30.0, c)
    assert np.allclose(a, b, atol=1e-4)
    assert a.dtype == np.float32 and a.min() >= c["clip_lo_db"] / c["scale_db"] - 1e-6 and a.max() <= c["clip_hi_db"] / c["scale_db"] + 1e-6


def test_noise_floor_tracks_slow_level_changes():
    rng = np.random.default_rng(1)
    t = np.arange(1000)[:, None]
    db = -30 + 0.01 * t + rng.standard_normal((1000, 8))
    fl = noise_floor_db(db.astype(np.float32), 64, 7)
    assert np.abs(fl - (-30 + 0.01 * t)).mean() < 1.0


def test_contract_shape_dtype_range(cfg):
    spec = random_spec()
    dm = make_detector(cfg).predict(spec)
    check_detection_map(dm, spec)
    lo, hi = band_bins(cfg)
    assert dm.conf.shape == (300, hi - lo + 1) and dm.conf.dtype == np.float32
    assert dm.to_full().shape == (300, 1025) and np.all(dm.to_full()[:, :lo] == 0)
    assert 0.0 <= dm.conf.min() and dm.conf.max() <= 1.0 and np.isfinite(dm.conf).all()


def test_contract_violations_are_caught(cfg):
    spec = random_spec()
    dm = make_detector(cfg).predict(spec)
    bad = DetectionMap(dm.conf[:-1], dm.bin_lo, dm.bin_hi, dm.n_bins_total, "x", "x")
    with pytest.raises(AssertionError):
        check_detection_map(bad, spec)
    bad = DetectionMap(dm.conf * 2.0, dm.bin_lo, dm.bin_hi, dm.n_bins_total, "x", "x")
    with pytest.raises(AssertionError):
        check_detection_map(bad, spec)


def test_inference_is_deterministic(cfg):
    spec = random_spec()
    det = make_detector(cfg)
    assert np.array_equal(det.predict(spec).conf, det.predict(spec).conf)
    assert np.array_equal(det.predict(spec).conf, make_detector(cfg).predict(spec).conf)   # same weights -> same output


def test_output_independent_of_tiling(cfg):
    spec = random_spec(400)
    full = make_detector(cfg, tile=100000, halo=32).predict(spec).conf
    for tile in (37, 64, 128):
        assert np.allclose(make_detector(cfg, tile=tile, halo=32).predict(spec).conf, full, atol=1e-5)


def test_halo_smaller_than_receptive_field_is_refused(cfg):
    with pytest.raises(ValueError):
        make_detector(cfg, halo=10)


def test_empty_spectrogram(cfg):
    spec = dsp.Spectrogram(np.zeros((0, 1025), np.float32), 192000.0, 2048, 1024, 2048, "x")
    assert make_detector(cfg).predict(spec).conf.shape[0] == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cpu_and_gpu_agree(cfg):
    spec = random_spec(300)
    cpu = make_detector(cfg, device="cpu").predict(spec).conf
    gpu = make_detector(cfg, device="cuda").predict(spec).conf
    assert np.allclose(cpu, gpu, atol=1e-4)


def _save_ckpt(path, cfg, cond=None, band=None):
    m = RidgeNet(16)
    torch.save({"model_state": m.state_dict(), "arch": {"channels": 16, "dilations": [list(d) for d in m.dilations]},
                "band": list(band or band_bins(cfg)), "conditioning": cond or dict(cfg["decisions"]["s5"]["conditioning"]),
                "version": "t", "seed": 0, "metrics": {}}, path)


def test_checkpoint_roundtrip_and_mismatch_refused(cfg, tmp_path):
    p = tmp_path / "c.pt"
    _save_ckpt(p, cfg)
    det = RidgeDetector.from_checkpoint(p, cfg, "cpu")
    assert det.version == "t" and len(det.meta["checkpoint_sha256"]) == 64
    bad_cond = dict(cfg["decisions"]["s5"]["conditioning"]); bad_cond["scale_db"] = 99.0
    _save_ckpt(p, cfg, cond=bad_cond)
    with pytest.raises(ValueError):
        RidgeDetector.from_checkpoint(p, cfg, "cpu")
    _save_ckpt(p, cfg, band=(10, 200))
    with pytest.raises(ValueError):
        RidgeDetector.from_checkpoint(p, cfg, "cpu")


def test_detector_does_not_depend_on_synth_or_tracking():
    """S5 must be replaceable independently of training-data generation and S6."""
    src = Path(__file__).resolve().parents[1] / "src" / "whistlepipe" / "detector"
    for f in src.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "synth" not in node.module and "tracking" not in node.module and "representation" not in node.module


def test_numerics_policy_full_float32_and_deterministic(cfg):
    """TF32 would make CUDA results differ from CPU by ~1e-2 and flip S6 tracks between devices."""
    make_detector(cfg)
    assert torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.are_deterministic_algorithms_enabled()
    from whistlepipe.storage import software_versions
    v = software_versions()
    assert v["tf32_conv_allowed"] is False and v["deterministic_algorithms"] is True


def test_separable_block_type_contract_and_cost_model(cfg):
    torch.manual_seed(0)
    m = RidgeNet(16, [(1, 1), (2, 2), (4, 4)], "separable")
    assert m.receptive_radius() == (8, 8) and m.macs_per_node() == 9 * 16 + 3 * (16 * 9 + 16 * 16) + 16
    s5 = cfg["decisions"]["s5"]
    spec = random_spec(300)
    full = RidgeDetector(m, "t", band_bins(cfg), s5["conditioning"], 100000, 32, "cpu").predict(spec)
    check_detection_map(full, spec)
    tiled = RidgeDetector(m, "t", band_bins(cfg), s5["conditioning"], 50, 32, "cpu").predict(spec)
    assert np.allclose(full.conf, tiled.conf, atol=1e-5)


def test_v1_checkpoint_without_block_type_loads_as_dense(cfg, tmp_path):
    p = tmp_path / "old.pt"
    _save_ckpt(p, cfg)                                       # arch has no "block_type" key, like ridgenet-v1
    assert RidgeDetector.from_checkpoint(p, cfg, "cpu").model.block_type == "dense"
