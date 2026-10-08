"""S6 tracking behaviour, S7 vectors, S8 centre removal + NFM, S9 image (all on small synthetic fixtures)."""
import numpy as np
import pytest

from whistlepipe import representation as rep
from whistlepipe.config import band_bins
from whistlepipe.tracking import Track, filter_tracks, track_confidence_map

BIN_LO = 22


def ridge_map(T=400, F=235, ridges=(), sigma=0.7, amp=0.95):
    """Confidence map with Gaussian ridges. ridge = (t0, t1, k_start, k_end) linear in band bins."""
    c = np.zeros((T, F), np.float32)
    k = np.arange(F)[None, :]
    for (t0, t1, ka, kb) in ridges:
        t = np.arange(t0, t1)
        kc = ka + (kb - ka) * (t - t0) / max(1, t1 - t0 - 1)
        c[t0:t1] = np.maximum(c[t0:t1], amp * np.exp(-0.5 * ((k - kc[:, None]) / sigma) ** 2))
    return c


def tracks_of(conf, cfg):
    return track_confidence_map(conf, BIN_LO, cfg)


def test_single_ridge_one_track_accurate(cfg):
    r = tracks_of(ridge_map(ridges=[(50, 150, 60, 90)]), cfg)
    assert len(r.tracks) == 1
    t = r.tracks[0]
    assert t.frames[0] == 50 and t.frames[-1] == 149 and len(t.frames) == 100
    expected_hz = (BIN_LO + 60 + 30 * (t.frames - 50) / 99) * 93.75
    assert np.abs(t.freq_hz - expected_hz).max() < 0.25 * 93.75       # sub-bin accuracy


def test_nothing_below_threshold(cfg):
    assert tracks_of(ridge_map(ridges=[(50, 150, 60, 90)], amp=0.3), cfg).tracks == []


def test_two_simultaneous_ridges_stay_separate(cfg):
    r = tracks_of(ridge_map(ridges=[(50, 200, 60, 80), (60, 190, 100, 90)]), cfg)
    assert len(r.tracks) == 2
    means = sorted(float(t.freq_hz.mean()) / 93.75 for t in r.tracks)
    assert means[1] - means[0] > 15


def test_crossing_ridges_keep_identity(cfg):
    r = tracks_of(ridge_map(ridges=[(50, 250, 60, 120), (50, 250, 120, 60)]), cfg)
    assert len(r.tracks) == 2
    slopes = sorted(float(t.freq_hz[-1] - t.freq_hz[0]) for t in r.tracks)
    assert slopes[0] < -40 * 93.75 and slopes[1] > 40 * 93.75       # one rising, one falling: not swapped at the crossing


def test_gap_is_bridged_and_flagged(cfg):
    c = ridge_map(ridges=[(50, 200, 70, 90)])
    c[100:106] = 0
    r = tracks_of(c, cfg)
    assert len(r.tracks) == 1
    t = r.tracks[0]
    assert (~t.observed).sum() == 6 and len(t.frames) == 150 and t.conf[~t.observed].max() == 0


def test_long_gap_splits_track(cfg):
    c = ridge_map(ridges=[(50, 200, 70, 90)])
    c[100:125] = 0
    assert len(tracks_of(c, cfg).tracks) == 2


def test_broadband_click_frame_is_skipped_and_bridged(cfg):
    c = ridge_map(ridges=[(50, 200, 70, 90)])
    c[120:122, :] = 0.9                                   # click: every node high
    r = tracks_of(c, cfg)
    assert r.stats["n_broadband_frames"] == 2
    assert len(r.tracks) == 1 and len(r.tracks[0].frames) == 150


def test_short_blip_rejected_with_reason(cfg):
    r = tracks_of(ridge_map(ridges=[(50, 58, 70, 72)]), cfg)
    assert r.tracks == [] and any(x["reason"] == "too_short" for x in r.rejected)


def test_harmonic_is_suppressed(cfg):
    fund = (50, 200, 40, 60)
    harm = (50, 200, 2 * (BIN_LO + 40) - BIN_LO, 2 * (BIN_LO + 60) - BIN_LO)        # exactly 2x in Hz
    r = tracks_of(ridge_map(ridges=[fund, harm]), cfg)
    assert len(r.tracks) == 1 and r.tracks[0].freq_hz.mean() / 93.75 < 100
    assert any(x["reason"] == "harmonic" for x in r.rejected)
    cfg["decisions"]["s6"]["suppress_harmonics"] = False
    assert len(tracks_of(ridge_map(ridges=[fund, harm]), cfg).tracks) == 2


def test_unrelated_whistle_at_non_integer_ratio_is_not_a_harmonic(cfg):
    r = tracks_of(ridge_map(ridges=[(50, 200, 40, 60), (50, 200, 110, 140)]), cfg)
    assert len(r.tracks) == 2


def test_duplicate_tracks_removed(cfg):
    f = np.arange(100, 200)
    mk = lambda i, off: Track(i, f, (BIN_LO + 60 + 0.1 * (f - 100) + off) * 93.75, np.full(100, 0.9), np.ones(100, bool))
    keep, rej = filter_tracks([mk(0, 0.0), mk(1, 0.5)], cfg["decisions"]["s6"], 93.75)
    assert len(keep) == 1 and rej[0]["reason"] == "duplicate"


def test_synchronous_comb_is_rejected_but_small_groups_and_unsynchronised_tracks_survive(cfg):
    # 8 short parallel ridges starting together with equal spans, at non-harmonic spacing (pulse-train lattice)
    comb = [(60, 76, 8 + 25 * i, 10 + 25 * i) for i in range(8)]
    r = tracks_of(ridge_map(ridges=comb, F=235), cfg)
    assert r.tracks == [] and sum(q["reason"] == "synchronous_comb" for q in r.rejected) >= 6
    # only 4 synchronous ridges: below comb_min_tracks -> kept (they are separate whistles)
    r = tracks_of(ridge_map(ridges=comb[:4]), cfg)
    assert len(r.tracks) == 4
    # same 8 ridges but with clearly different starts/spans -> not a comb
    varied = [(60 + 15 * i, 60 + 15 * i + 20 + 7 * i, 8 + 25 * i, 10 + 25 * i) for i in range(8)]
    assert len(tracks_of(ridge_map(T=500, ridges=varied), cfg).tracks) == 8
    cfg["decisions"]["s6"]["suppress_synchronous_combs"] = False
    assert len(tracks_of(ridge_map(ridges=comb), cfg).tracks) == 8


def test_tracking_is_deterministic(cfg):
    c = ridge_map(ridges=[(50, 250, 60, 120), (50, 250, 120, 60), (80, 160, 30, 30)])
    a, b = tracks_of(c, cfg), tracks_of(c, cfg)
    assert [(t.id, t.frames[0], len(t.frames)) for t in a.tracks] == [(t.id, t.frames[0], len(t.frames)) for t in b.tracks]
    assert all(np.array_equal(x.freq_hz, y.freq_hz) for x, y in zip(a.tracks, b.tracks))


# ------------------------------------------------------------------ S7 / S8 / S9
def make_vec(L=80, f0=9000.0, f1=11000.0, start=100, tid=0):
    f = np.linspace(f0, f1, L)
    return rep.WhistleVector(tid, start, f, np.full(L, 0.9), 1.0)


def test_s7_preserves_track(cfg):
    t = Track(7, np.arange(10, 60), np.linspace(5000, 6000, 50), np.full(50, 0.8), np.ones(50, bool))
    v, = rep.s7_vectors([t], cfg)
    assert v.track_id == 7 and v.start_frame == 10 and v.n_frames == 50 and np.array_equal(v.freq_hz, t.freq_hz)
    assert rep.vectors_to_npz([v])["offsets"].tolist() == [0, 50]


def test_baseband_grid_is_derived_from_paper_numbers(cfg):
    g = rep.nfm_grid(cfg)
    assert (g["fs_b"], g["hop_b"], g["win_b"], g["nfft_b"], g["n_samples"]) == (12000.0, 64, 128, 128, 8256)


def test_s8_centre_removal_and_placement(cfg):
    (n,), rej = rep.s8_baseband([make_vec(80, 9000, 11000)], cfg)
    assert not rej
    assert n.center_hz == pytest.approx(10000.0) and n.delta_hz.min() == pytest.approx(-n.delta_hz.max())
    assert n.col0 == (128 - 80) // 2 == 24
    assert n.baseband.dtype == np.complex64 and n.baseband.shape == (8256,)
    mid = np.abs(n.baseband[24 * 64 + 100: (24 + 80) * 64 - 100])
    assert np.allclose(mid, 1.0, atol=1e-6)                                   # constant amplitude


def test_s8_instantaneous_frequency_matches_contour(cfg):
    (n,), _ = rep.s8_baseband([make_vec(80, 9000, 11000)], cfg)
    ph = np.unwrap(np.angle(n.baseband.astype(np.complex128)))
    inst = np.diff(ph) * 12000.0 / (2 * np.pi)
    centers = (n.col0 + np.arange(80)) * 64 + 64
    got = np.interp(centers[5:-5], np.arange(len(inst)) + 0.5, inst)
    assert np.abs(got - n.delta_hz[5:-5]).max() < 2.0


def test_s8_rejects_too_long_and_too_wide(cfg):
    out, rej = rep.s8_baseband([make_vec(129), make_vec(60, 3000, 16000, tid=1), make_vec(128, tid=2)], cfg)
    reasons = {r["track_id"]: r["reason"] for r in rej}
    assert reasons == {0: "too_long", 1: "too_wide"} and [o.track_id for o in out] == [2]


def test_s9_shape_dtype_range_and_centre_row(cfg):
    flat = rep.WhistleVector(0, 0, np.full(60, 12345.0), np.full(60, 0.9), 1.0)
    (n,), _ = rep.s8_baseband([flat], cfg)
    img = rep.s9_image(n, cfg)
    assert img.shape == (128, 128) and img.dtype == np.float32 and img.min() >= 0 and img.max() == pytest.approx(1.0)
    assert (img[:, 64] .argmax() == 63) and (img[:, 40].argmax() == 63)         # delta=0 Hz -> row 63 (0-indexed), columns inside the whistle
    assert img[:, 5].max() < 0.2                                                # empty region is dark


def test_s9_ridge_follows_contour_and_threshold_recovers_it(cfg):
    (n,), _ = rep.s8_baseband([make_vec(90, 7000, 12000)], cfg)
    img = rep.s9_image(n, cfg)
    cols, f = rep.recover_contour(img, cfg, thr=0.05)
    inside = (cols >= n.col0 + 4) & (cols < n.col0 + 90 - 4)
    expect = np.interp(cols[inside] - n.col0, np.arange(90), n.delta_hz)
    assert np.abs(f[inside] - expect).max() < 0.5 * 93.75
    thickness = (img[:, n.col0 + 45] > 0.05).sum()
    assert 2 <= thickness <= 6
    assert cols.min() >= n.col0 - 2 and cols.max() <= n.col0 + 90 + 1          # no splatter far outside the whistle


def test_s9_rising_whistle_is_upright(cfg):
    (n,), _ = rep.s8_baseband([make_vec(100, 8000, 12000)], cfg)
    img = rep.s9_image(n, cfg)
    r_start, r_end = img[:, n.col0 + 3].argmax(), img[:, n.col0 + 96].argmax()
    assert r_end < r_start                                                      # rises to the right = smaller row index


def test_s9_batch_and_empty(cfg):
    assert rep.s9_images([], cfg).shape == (0, 128, 128)
    nf, _ = rep.s8_baseband([make_vec(50, tid=i) for i in range(3)], cfg)
    imgs = rep.s9_images(nf, cfg)
    assert imgs.shape == (3, 128, 128) and imgs.dtype == np.float32 and 0 <= imgs.min() and imgs.max() <= 1.0


def test_chain_conf_map_to_images(cfg):
    conf = ridge_map(ridges=[(50, 150, 60, 90), (70, 140, 120, 110)])
    r = tracks_of(conf, cfg)
    vecs = rep.s7_vectors(r.tracks, cfg)
    nf, rej = rep.s8_baseband(vecs, cfg)
    imgs = rep.s9_images(nf, cfg)
    assert len(r.tracks) == len(vecs) == len(nf) == len(imgs) == 2 and not rej
