"""Synthetic training-data generator: determinism, label correctness, independence from any detector."""
import ast
from pathlib import Path

import numpy as np
import pytest

from whistlepipe import audio
from whistlepipe.config import PROJECT_ROOT, band_bins
from whistlepipe.synth import backgrounds
from whistlepipe.synth.whistles import SynthSampler


@pytest.fixture(scope="module")
def sampler(base_cfg, det_cfg):
    rng = np.random.default_rng(0)
    windows = (0.003 * rng.standard_normal((4, int(3.0 * 192000)))).astype(np.float16)   # white-noise stand-in backgrounds
    return SynthSampler(base_cfg, det_cfg, windows, split_id=0)


def test_samples_are_deterministic(sampler):
    a, b, c = sampler.sample(5), sampler.sample(5), sampler.sample(6)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["y"], b["y"]) and np.array_equal(a["w"], b["w"])
    assert not np.array_equal(a["x"], c["x"])


def test_sample_shapes_and_ranges(sampler, base_cfg):
    s = sampler.sample(1)
    lo, hi = band_bins(base_cfg)
    assert s["x"].shape == s["y"].shape == s["w"].shape == ((int(3.0 * 192000) - 2048) // 1024 + 1, hi - lo + 1)   # 561 frames
    assert s["y"].min() >= 0 and s["y"].max() <= 1 and set(np.unique(s["w"])) <= {0.0, 1.0}


def _first_with(sampler, pred, limit=200):
    for i in range(limit):
        s = sampler.sample(i)
        if pred(s):
            return s
    pytest.skip("no suitable sample found")


def test_no_whistle_sample_has_empty_target(sampler):
    s = _first_with(sampler, lambda s: s["meta"]["n_whistles"] == 0)
    assert s["y"].max() == 0 and (s["w"] == 1).all() and not s["gt"]


def test_labels_sit_on_the_actual_ridge(sampler, base_cfg):
    """For a strong single whistle the labelled contour must coincide with the spectrogram ridge (frame timing + bin mapping)."""
    s = _first_with(sampler, lambda s: s["meta"]["n_whistles"] == 1 and s["meta"]["snr_db"][0] > 18 and s["meta"]["dur_s"][0] > 0.3)
    g = [g for g in s["gt"] if g["h"] == 1][0]
    lo, _ = band_bins(base_cfg)
    b = g["f_hz"] / 93.75 - lo
    t = g["frames"]
    peak_bin = s["x"][t].argmax(axis=1)
    assert np.mean(np.abs(peak_bin - b) <= 2.0) > 0.85
    vis = g["visible"]
    assert vis.mean() > 0.8
    ys = s["y"][t[vis], np.clip(np.rint(b[vis]).astype(int), 0, s["y"].shape[1] - 1)]
    assert (ys > 0.8).mean() > 0.95        # target at the rounded bin is >= exp(-0.5*(0.5/0.8)^2) = 0.82
    assert (s["w"][t[vis], np.clip(np.rint(b[vis]).astype(int), 0, s["y"].shape[1] - 1)] == 1).all()


def test_invisible_parts_are_ignored_not_negative(sampler, base_cfg):
    s = _first_with(sampler, lambda s: any((~g["visible"]).any() and (g["h"] == 1) for g in s["gt"]))
    lo, _ = band_bins(base_cfg)
    g = [g for g in s["gt"] if (~g["visible"]).any()][0]
    t = g["frames"][~g["visible"]]
    k = np.clip(np.rint(g["f_hz"][~g["visible"]] / 93.75 - lo).astype(int), 0, s["y"].shape[1] - 1)
    covered_by_other = s["y"][t, k] > 0.3
    assert (s["w"][t, k][~covered_by_other] == 0).all()


def test_generator_never_imports_a_detector():
    for f in (PROJECT_ROOT / "src" / "whistlepipe" / "synth").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "ridgenet" not in node.module and "tracking" not in node.module
            if isinstance(node, ast.ImportFrom) and node.module == "detector" and any(a.name.lower().startswith("ridge") for a in node.names):
                pytest.fail("synth must not import detector models")


def test_background_splits_disjoint_and_exclude_validation_files(base_cfg, det_cfg):
    if not (PROJECT_ROOT / "encounter_191_audio").exists():
        pytest.skip("raw data not present")
    picks = backgrounds.pick_background_files(base_cfg, det_cfg)
    names = [p.name for v in picks.values() for p in v]
    assert len(names) == len(set(names))
    assert not (set(names) & backgrounds.reserved_validation_files(base_cfg))
    assert backgrounds.reserved_validation_files(base_cfg) >= {p.name for p in audio.select_validation_files(
        audio.list_raw_files(PROJECT_ROOT / "encounter_191_audio"), 1)}


def test_ridge_screen_flags_persistent_narrowband_but_not_clicks():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((300, 100))
    tonal = base.copy(); tonal[50:150, 40] += 30           # persistent narrow ridge
    clicks = base.copy(); clicks[::20, :] += 30            # broadband 1-frame clicks
    args = (8.0, 9, 0.7, (3, 5))
    assert backgrounds.ridge_score(tonal, *args) > 0.2
    assert backgrounds.ridge_score(clicks, *args) < 0.01
    assert backgrounds.ridge_score(base, *args) < 0.01


def _cfg_with(det_cfg, **synth_updates):
    import copy

    from whistlepipe.config import Cfg

    d = Cfg(copy.deepcopy(dict(det_cfg)))
    for path, val in synth_updates.items():
        sec, key = path.split("__")
        d["synth"][sec][key] = val
    return d


def test_snr_mix_option_respects_intervals(base_cfg, det_cfg):
    rng = np.random.default_rng(0)
    windows = (0.003 * rng.standard_normal((2, int(3.0 * 192000)))).astype(np.float16)
    s = SynthSampler(base_cfg, _cfg_with(det_cfg, whistles__snr_mix=[[1.0, 4.0, 6.0]]), windows, split_id=0)
    snrs = [x for i in range(40) for x in s.sample(i)["meta"]["snr_db"]]
    assert snrs and all(4.0 <= v <= 6.0 for v in snrs)


def test_time_reverse_applies_to_training_split_only(base_cfg, det_cfg):
    rng = np.random.default_rng(0)
    windows = (0.003 * rng.standard_normal((2, int(3.0 * 192000)))).astype(np.float16)
    on = _cfg_with(det_cfg, backgrounds__p_time_reverse=1.0)
    off = _cfg_with(det_cfg, backgrounds__p_time_reverse=0.0)
    assert not np.array_equal(SynthSampler(base_cfg, on, windows, 0).sample(3)["x"], SynthSampler(base_cfg, off, windows, 0).sample(3)["x"])
    assert np.array_equal(SynthSampler(base_cfg, on, windows, 1).sample(3)["x"], SynthSampler(base_cfg, off, windows, 1).sample(3)["x"])   # val untouched


def test_defaults_add_no_random_draws(base_cfg, det_cfg):
    """With the new options absent the generator must reproduce the ridgenet-v1 stream exactly."""
    rng = np.random.default_rng(0)
    windows = (0.003 * rng.standard_normal((2, int(3.0 * 192000)))).astype(np.float16)
    plain = _cfg_with(det_cfg)
    plain["synth"]["backgrounds"].pop("p_time_reverse", None); plain["synth"]["whistles"].pop("snr_mix", None)
    zero = _cfg_with(det_cfg, backgrounds__p_time_reverse=0.0)
    assert np.array_equal(SynthSampler(base_cfg, plain, windows, 0).sample(7)["x"], SynthSampler(base_cfg, zero, windows, 0).sample(7)["x"])
