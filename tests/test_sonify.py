"""sonify.py: image -> contour -> tone, checked against known contours and through the project's own S2-S9 chain."""
import csv

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from whistlepipe import dsp, sonify
from whistlepipe.audio import AudioSignal
from whistlepipe.config import PROJECT_ROOT
from whistlepipe.representation import WhistleVector, s8_baseband, s9_image

BIN = 93.75


def make_image(cfg, freq_hz):
    v = WhistleVector(0, 0, np.asarray(freq_hz, float), np.ones(len(freq_hz)), 1.0)
    nfms, rej = s8_baseband([v], cfg)
    assert not rej
    return s9_image(nfms[0], cfg), nfms[0]


def contour_of_audio(x, fs, cfg):
    """The tone run back through S2-S4, then the spectral peak per frame -> (frame centre times, absolute Hz)."""
    s4 = dsp.s4_stft(dsp.s3_highpass(dsp.s2_resample(AudioSignal(x.astype(np.float32), float(fs), "rt"), cfg), cfg), cfg)
    db = s4.db
    k_lo, k_hi = int(2000 / BIN), int(24000 / BIN)
    band = db[:, k_lo:k_hi + 1]
    pk = band.argmax(1)
    peak = band.max(1)
    on = np.nonzero(peak > peak.max() - 20)[0]
    t0, t1 = on[0], on[-1]
    k = np.clip(pk[t0:t1 + 1], 1, band.shape[1] - 2)
    r = np.arange(t0, t1 + 1)
    a, b, c = band[r, k - 1], band[r, k], band[r, k + 1]
    delta = 0.5 * (a - c) / np.where(np.abs(a - 2 * b + c) > 1e-9, a - 2 * b + c, 1e-9)
    return s4.frame_center_times_s()[r], (k_lo + k + delta) * BIN


def offsets_on_common_columns(c1, c2):
    common, i1, i2 = np.intersect1d(c1.cols, c2.cols, return_indices=True)
    return c1.offset_hz[i1], c2.offset_hz[i2]


def test_contour_recovers_known_shape(cfg):
    n = 70
    f = 9000 + 1800 * np.sin(np.linspace(0, 2 * np.pi, n))
    img, nfm = make_image(cfg, f)
    c = sonify.image_contour(img, cfg)
    assert c.cols[0] == nfm.col0 and len(c.cols) == n
    err = np.abs(c.offset_hz - nfm.delta_hz)[2:-2]          # ramps blur the first/last column
    assert err.max() < 0.35 * BIN, err.max()


def test_ridge_ignores_stray_pixels(cfg):
    img, nfm = make_image(cfg, np.linspace(7000, 9000, 50))
    noisy = img.copy()
    noisy[5, nfm.col0 + 10: nfm.col0 + 20] = 0.3              # a stray blob far above the ridge
    a, b = sonify.image_contour(img, cfg), sonify.image_contour(noisy, cfg)
    assert np.allclose(a.offset_hz, b.offset_hz)


def test_empty_image_has_no_contour(cfg, tmp_path):
    assert sonify.image_contour(np.zeros((128, 128)), cfg) is None
    assert sonify.image_to_wav(np.full((128, 128), 0.01), 9000, tmp_path / "x.wav", cfg) is None


def test_tone_follows_contour_and_level(cfg, tmp_path):
    f = np.linspace(6000, 11000, 60)
    img, _ = make_image(cfg, f)
    c = sonify.image_contour(img, cfg)
    fc = 0.5 * (f.min() + f.max())
    meta = sonify.image_to_wav(img, fc, tmp_path / "w.wav", cfg)
    x, fs = sf.read(tmp_path / "w.wav")
    assert fs == 48000 and meta["fs"] == 48000
    assert abs(20 * np.log10(np.abs(x).max()) + 3.0) < 0.05          # -3 dBFS (16-bit rounding)
    pad = int(0.02 * fs)
    assert np.all(x[:pad] == 0) and np.all(x[-pad:] == 0)
    assert len(x) == pytest.approx(2 * pad + len(c.cols) * c.hop_s * fs, abs=2)
    tone = x[pad:-pad]
    inst = np.diff(np.unwrap(np.angle(signal.hilbert(tone)))) * fs / (2 * np.pi)
    t = (np.arange(len(inst)) + 1) / fs
    want = np.interp(t, (np.arange(len(c.cols)) + 0.5) * c.hop_s, fc + c.offset_hz)
    mid = slice(int(0.05 * len(inst)), int(0.95 * len(inst)))
    assert np.median(np.abs(inst[mid] - want[mid])) < 5.0


def test_slow_keeps_pitch_and_doubles_length(cfg):
    img, _ = make_image(cfg, np.full(40, 8000.0))
    c = sonify.image_contour(img, cfg)
    x1, fs = sonify.synthesize(c, 8000.0, cfg, pad_s=0)
    x2, _ = sonify.synthesize(c, 8000.0, cfg, pad_s=0, slow=2.0)
    assert len(x2) == pytest.approx(2 * len(x1), abs=2)
    for x in (x1, x2):
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        assert abs(np.argmax(spec) * fs / len(x) - 8000.0) < 30


def test_high_tones_use_96k(cfg):
    img, _ = make_image(cfg, np.linspace(20000, 23000, 30))
    c = sonify.image_contour(img, cfg)
    _, fs = sonify.synthesize(c, 21500.0, cfg)
    assert fs == 96000


def round_trip(img, fc, cfg, pad_s=0.02):
    """image -> tone -> S2-S4 -> peak contour. Returns the absolute-frequency error of the heard tone against the
    image's contour (interior frames, Hz), and the image rebuilt from the heard contour by S8/S9."""
    c0 = sonify.image_contour(img, cfg)
    x, fs = sonify.synthesize(c0, fc, cfg, pad_s=pad_s)
    t, f = contour_of_audio(x, fs, cfg)
    want = np.interp(t, pad_s + (np.arange(len(c0.cols)) + 0.5) * c0.hop_s, fc + c0.offset_hz)
    err = np.abs(f - want)[2:-2]
    img2, _ = make_image(cfg, f)
    return err, c0, img2


def test_round_trip_through_pipeline(cfg):
    f = 10000 + 2500 * np.sin(np.linspace(-1.2, 2.4, 90)) + np.linspace(0, 800, 90)
    img, _ = make_image(cfg, f)
    err, c0, img2 = round_trip(img, 0.5 * (f.min() + f.max()), cfg)
    # the heard pitch follows the image's contour to a fraction of one 93.75 Hz row
    assert np.median(err) < 0.25 * BIN and np.quantile(err, 0.95) < 0.5 * BIN, (np.median(err), err.max())
    # and the image rebuilt from the heard tone has the same shape. S8 re-centres it in time (a length change of a
    # frame moves it by a column) and in frequency (midrange), so compare at the best column shift, up to a constant.
    c1 = sonify.image_contour(img2, cfg)
    assert abs(len(c1.cols) - len(c0.cols)) <= 2
    best = []
    for s in range(-2, 3):
        a, b = offsets_on_common_columns(c0, type(c1)(c1.cols - s, c1.offset_hz, c1.hop_s))
        best.append(np.median(np.abs((a - b) - np.median(a - b))))
    assert min(best) < 0.25 * BIN, best


def _real_images(n):
    d = PROJECT_ROOT / "preprocessed" / "encounter_191"
    if not (d / "images.npy").exists():
        return None
    arr = np.load(d / "images.npy", mmap_mode="r")
    rows = list(csv.DictReader(open(d / "images.csv", encoding="utf-8")))
    idx = np.random.default_rng(7).choice(len(rows), n, replace=False)
    return [(np.asarray(arr[i]), float(rows[i]["center_hz"]), int(rows[i]["n_frames"])) for i in idx]


def test_round_trip_real_training_images(cfg):
    real = _real_images(25)
    if real is None:
        pytest.skip("preprocessed dataset not present")
    med = []
    for img, fc, n_frames in real:
        err, _, _ = round_trip(img, fc, cfg)
        if len(err):
            med.append(np.median(err))
    med = np.asarray(med)
    # per image: median absolute pitch error of the heard tone against the image's contour
    assert np.median(med) < 0.25 * BIN and np.quantile(med, 0.9) < 0.5 * BIN, med
