"""Config lock, S1 (audio/channel), S2 (resample), S3 (high-pass), S4 (STFT grid), storage safety."""
import hashlib
import math

import numpy as np
import pytest
from scipy import signal

from conftest import CHANNEL_TONES_HZ, real_first_file
from whistlepipe import audio, dsp
from whistlepipe.config import load_pipeline_config, validate_paper_lock
from whistlepipe.storage import RawDataGuardError, RunStore, ensure_not_in_raw


def tone(f, fs, n, a=0.1):
    return a * np.sin(2 * np.pi * f * np.arange(n) / fs)


def peak_amp(x, fs, f):
    """Amplitude of a sinusoid at f via projection."""
    t = np.arange(len(x)) / fs
    return 2 * abs(np.mean(x * np.exp(-2j * np.pi * f * t)))


# ------------------------------------------------------------------ config lock
def test_paper_values_locked(base_cfg):
    assert base_cfg["paper"]["fs_target_hz"] == 192000 and base_cfg["paper"]["nfft"] == 2048


@pytest.mark.parametrize("key,val", [("fs_target_hz", 96000), ("win_len", 1024), ("hop", 512), ("hpf_cutoff_hz", 1000), ("image_width", 64)])
def test_paper_values_cannot_be_overridden(key, val):
    with pytest.raises(ValueError):
        load_pipeline_config(overrides=[f"paper.{key}={val}"])


def test_unknown_paper_key_rejected(base_cfg):
    bad = {**base_cfg.to_dict()}
    bad["paper"] = {**bad["paper"], "invented": 1}
    with pytest.raises(ValueError):
        validate_paper_lock(bad)


def test_override_recorded_in_hash():
    a = load_pipeline_config()
    b = load_pipeline_config(overrides=["decisions.s6.peak_threshold=0.7"])
    assert a["_meta"]["effective_hash"] != b["_meta"]["effective_hash"]
    assert b["decisions"]["s6"]["peak_threshold"] == 0.7


# ------------------------------------------------------------------ S1
def test_s1_selects_requested_channel(cfg, fake_raw):
    for ch in (0, 3, 5):
        cfg["decisions"]["s1"]["channel_index"] = ch
        sig = audio.s1_select_channel(fake_raw, cfg)
        assert sig.fs == 500000 and sig.samples.dtype == np.float32 and len(sig.samples) == 500000
        assert abs(peak_amp(sig.samples, sig.fs, CHANNEL_TONES_HZ[ch]) - 0.2) < 2e-3
        for other, f in enumerate(CHANNEL_TONES_HZ):
            if other != ch:
                assert peak_amp(sig.samples, sig.fs, f) < 2e-3
        assert sig.meta["channel_index"] == ch


def test_s1_scaling_full_scale(cfg, fake_raw):
    sig = audio.s1_select_channel(fake_raw, cfg)
    assert np.abs(sig.samples).max() <= 1.0


def test_s1_does_not_modify_raw(cfg, fake_raw):
    before = (hashlib.sha256(fake_raw.read_bytes()).hexdigest(), fake_raw.stat().st_mtime_ns)
    audio.s1_select_channel(fake_raw, cfg)
    after = (hashlib.sha256(fake_raw.read_bytes()).hexdigest(), fake_raw.stat().st_mtime_ns)
    assert before == after


def test_s1_rejects_bad_channel(cfg, fake_raw):
    cfg["decisions"]["s1"]["channel_index"] = 6
    with pytest.raises(ValueError):
        audio.s1_select_channel(fake_raw, cfg)


def test_validation_set_selection_is_deterministic_and_spread():
    files = [f"f{i:03d}" for i in range(363)]
    a, b = audio.select_validation_files(files, 5), audio.select_validation_files(files, 5)
    assert a == b and len(set(a)) == 5 and a == sorted(a)
    assert audio.select_validation_files(files, 1) == ["f181"]


# ------------------------------------------------------------------ S2
def test_s2_length_rate_and_tone(cfg):
    fs, n = 500000, 500000
    sig = audio.AudioSignal(tone(10000.0, fs, n).astype(np.float32), fs, "x.flac")
    out = dsp.s2_resample(sig, cfg)
    assert out.fs == 192000 and len(out.samples) == 192000
    assert (out.meta["resample_up"], out.meta["resample_down"]) == (48, 125)
    mid = out.samples[20000:-20000]
    assert abs(peak_amp(mid, 192000, 10000.0) - 0.1) < 1e-3


def test_s2_antialiasing_above_new_nyquist(cfg):
    fs, n = 500000, 500000
    sig = audio.AudioSignal(tone(150000.0, fs, n).astype(np.float32), fs, "x.flac")   # above 96 kHz
    out = dsp.s2_resample(sig, cfg)
    assert np.sqrt(np.mean(out.samples[20000:-20000] ** 2)) < 0.1 / np.sqrt(2) * 10 ** (-60 / 20)


def test_s2_full_file_length_matches_spec(cfg):
    up, down = dsp.resample_ratio(500000, 192000)
    assert math.ceil(30_000_000 * up / down) == 11_520_000


# ------------------------------------------------------------------ S3
def test_s3_highpass_response(cfg):
    fs, n = 192000, 192000 * 2
    gains = {}
    for f in (500.0, 1000.0, 2000.0, 5000.0, 20000.0):
        sig = audio.AudioSignal(tone(f, fs, n), fs, "x")
        y = dsp.s3_highpass(sig, cfg)
        gains[f] = 20 * np.log10(peak_amp(y.samples[fs // 2:-fs // 2], fs, f) / 0.1)
    assert abs(gains[5000.0]) < 0.1 and abs(gains[20000.0]) < 0.1          # passband
    assert gains[500.0] < -40                                              # stopband
    assert gains[1000.0] < -20
    assert -7.0 < gains[2000.0] < -5.0                                     # zero-phase, 4th order: -6 dB at the corner (DECISION)


def test_s3_zero_phase_no_delay(cfg):
    fs = 192000
    x = np.zeros(fs)
    x[fs // 2] = 1.0
    y = dsp.s3_highpass(audio.AudioSignal(x, fs, "x"), cfg).samples
    assert np.argmax(np.abs(y)) == fs // 2


# ------------------------------------------------------------------ S4
def test_s4_grid_and_frame_count(cfg):
    n = 192000 * 3
    spec = dsp.s4_stft(audio.AudioSignal(tone(9000.0, 192000, n), 192000, "x"), cfg)
    assert spec.db.shape == ((n - 2048) // 1024 + 1, 1025) and spec.db.dtype == np.float32
    assert spec.bin_hz == 93.75 and spec.hop_s == pytest.approx(1024 / 192000) and spec.hop_s * 1e3 == pytest.approx(5.3333, abs=1e-3)
    assert spec.freqs_hz()[1] == 93.75 and spec.freqs_hz()[-1] == 96000.0
    assert spec.frame_center_times_s()[0] == pytest.approx(1024 / 192000)


def test_s4_full_file_frame_count_matches_spec(cfg):
    assert (11_520_000 - 2048) // 1024 + 1 == 11249


@pytest.mark.parametrize("k0", [22, 64, 100, 256, 500])
def test_s4_bin_centred_tone_peaks_in_its_bin(cfg, k0):
    f = k0 * 93.75
    spec = dsp.s4_stft(audio.AudioSignal(tone(f, 192000, 192000), 192000, "x"), cfg)
    assert (spec.db.argmax(axis=1) == k0).all()
    # Hamming main lobe: neighbours ~ -7.4 dB (0.426) below the peak for a bin-centred tone
    mid = spec.db[spec.n_frames // 2]
    assert mid[k0] - mid[k0 + 1] == pytest.approx(20 * np.log10(1 / 0.4259), abs=0.3)


def test_s4_matches_scipy_reference(cfg):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(192000)
    spec = dsp.s4_stft(audio.AudioSignal(x, 192000, "x"), cfg)
    _, _, Z = signal.stft(x, 192000, window="hamming", nperseg=2048, noverlap=1024, nfft=2048,
                          boundary=None, padded=False, detrend=False)
    ref_db = 20 * np.log10(np.abs(Z.T) * signal.get_window("hamming", 2048).sum() + 1e-12)
    assert ref_db.shape == spec.db.shape
    assert np.abs(ref_db - spec.db).max() < 1e-3


def test_s4_temporal_localisation(cfg):
    x = np.zeros(192000)
    x[96000] = 1.0           # impulse
    db = dsp.s4_stft(audio.AudioSignal(x, 192000, "x"), cfg).db
    hit = np.where(db[:, 100] > db[:, 100].max() - 20)[0]
    assert all(h * 1024 <= 96000 < h * 1024 + 2048 for h in hit) and len(hit) == 2   # the two frames covering the sample


def test_s4_rejects_wrong_rate(cfg):
    with pytest.raises(ValueError):
        dsp.s4_stft(audio.AudioSignal(np.zeros(5000), 96000, "x"), cfg)


# ------------------------------------------------------------------ storage safety
def test_raw_guard_blocks_writes_inside_raw(cfg, fake_raw):
    raw = fake_raw.parent
    with pytest.raises(RawDataGuardError):
        ensure_not_in_raw(raw / "anything.npy", raw)
    cfg["paths"]["runs_dir"] = str(raw / "runs")           # a run directory inside raw must be refused
    with pytest.raises(RawDataGuardError):
        RunStore(cfg, "r1")


def test_run_never_overwritten(cfg, fake_raw):
    RunStore(cfg, "r1")
    with pytest.raises(FileExistsError):
        RunStore(cfg, "r1")


def test_storage_modes_policy(cfg, fake_raw):
    assert RunStore(cfg, "a", "debug").policy["s4"] is True
    assert RunStore(cfg, "b", "normal").policy["s4"] is False and RunStore(cfg, "c", "normal").policy["s5"] is True
    assert not any(RunStore(cfg, "d", "minimal").policy.values())
    with pytest.raises(ValueError):
        RunStore(cfg, "e", "bogus")


def test_s1_flac_roundtrip_is_exact(cfg, fake_raw):
    sig = audio.s1_select_channel(fake_raw, cfg)
    store = RunStore(cfg, "rt", "debug")
    store.save_stage("stem", "s1", sig)
    from whistlepipe.storage import load_stage_array

    x, fs = load_stage_array(store.dir, "stem", "s1")
    assert fs == 500000 and np.array_equal(x, sig.samples)


# ------------------------------------------------------------------ real data (read-only)
@pytest.mark.realdata
def test_real_file_s1_to_s4_shapes(cfg):
    p = real_first_file()
    if p is None:
        pytest.skip("raw data not present")
    s1 = audio.s1_select_channel(p, cfg)
    assert len(s1.samples) == 30_000_000 and s1.fs == 500000
    s2 = dsp.s2_resample(s1, cfg)
    assert len(s2.samples) == 11_520_000
    s4 = dsp.s4_stft(dsp.s3_highpass(s2, cfg), cfg)
    assert s4.db.shape == (11249, 1025)
