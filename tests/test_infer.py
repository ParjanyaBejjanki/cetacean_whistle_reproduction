"""Inference engine: upload reading, the S2-S9 path, and the model step with randomly initialised files saved in the
exact formats training writes (so a real final.pt / gmm.npz dropped into place takes the same code path)."""
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from whistlepipe import audio, infer
from whistlepipe.infer import Engine, load_inference_config, model_paths
from whistlepipe.vaegmm.config import load_vaegmm_config
from whistlepipe.vaegmm.gmm import GaussianMixtureEM
from whistlepipe.vaegmm.models import VAE

WHISTLES = [(0.4, 0.35, lambda u: 7000 + 2500 * u), (1.3, 0.45, lambda u: 11000 - 3000 * u),
            (2.2, 0.5, lambda u: 9000 + 1500 * np.sin(2 * np.pi * u)), (3.1, 0.3, lambda u: 6000 + 800 * u ** 2)]


def synthetic_recording(path, fs=96000, dur=4.0, channels=1, seed=0):
    """Four frequency-modulated whistles (known times and ranges) in white noise."""
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    x = 0.003 * rng.standard_normal(n)
    for t0, d, f in WHISTLES:
        if t0 + d > dur:
            continue
        k = int(d * fs)
        u = np.arange(k) / k
        x[int(t0 * fs):int(t0 * fs) + k] += 0.05 * np.sin(np.pi * u) ** 0.3 * np.sin(2 * np.pi * np.cumsum(f(u)) / fs)
    data = np.stack([x] + [0.003 * rng.standard_normal(n) for _ in range(channels - 1)], 1) if channels > 1 else x
    sf.write(str(path), data.astype(np.float32), fs, subtype="PCM_16")
    return path


RUN = "TEST_L64_random"
K = 5


def write_random_model(runs_dir, gmm=True, comp_feats=True):
    """final.pt and gmm_K5/gmm.npz (+ gmm.json, component_features.npy) in the formats of train.py / gmm.py."""
    torch.manual_seed(0)
    vae = VAE(64)
    ck = runs_dir / RUN / "checkpoints"
    ck.mkdir(parents=True)
    torch.save({"vae": vae.state_dict(), "critic": None, "epoch": 300, "global_step": 0,
                "spec": {"run_id": RUN, "latent_dim": 64, "beta": 1e-4, "use_critic": True, "epochs": 300}}, ck / "final.pt")
    if not gmm:
        return
    g = GaussianMixtureEM(K, device="cpu", verbose=False)
    rng = np.random.default_rng(0)
    g.weights_ = torch.as_tensor(np.full(K, 1.0 / K))
    g.means_ = torch.as_tensor(rng.normal(size=(K, 64)))
    g.covariances_ = torch.as_tensor(np.stack([np.eye(64) * 0.5] * K))
    g.reg_, g.converged_, g.n_iter_ = 1e-6, True, 1
    gd = runs_dir / RUN / f"gmm_K{K}"
    gd.mkdir()
    g.save(gd / "gmm.npz")
    if comp_feats:
        from whistlepipe.vaegmm.conditional import component_features

        g = GaussianMixtureEM.load(gd / "gmm.npz", "cpu", verbose=False)
        np.save(gd / "component_features.npy", component_features(vae.decoder.eval(), g, 10, torch.Generator().manual_seed(0)))


def make_engine(tmp_path, **kw):
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    if kw:
        write_random_model(runs, **kw)
    icfg = load_inference_config(overrides=[f'model.run_id="{RUN}"', f"model.n_components={K}"])
    vcfg = load_vaegmm_config(overrides=[f"paths.runs_dir='{runs.as_posix()}'"])
    return Engine(icfg=icfg, vcfg=vcfg, device="cpu")


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    return synthetic_recording(tmp_path_factory.mktemp("rec") / "synthetic.wav")


# ----------------------------------------------------------------------------- where the trained files go
def test_model_paths_match_the_training_layout():
    from whistlepipe.vaegmm.evaluate import gmm_dir
    from whistlepipe.vaegmm.config import resolve

    icfg, vcfg = load_inference_config(), load_vaegmm_config()
    p = model_paths(icfg, vcfg)
    rid, k = icfg["model"]["run_id"], icfg["model"]["n_components"]
    assert p["vae"] == resolve(vcfg["paths"]["runs_dir"]) / rid / "checkpoints" / "final.pt"   # train.py: out/"checkpoints"/"final.pt"
    assert p["gmm"] == gmm_dir(vcfg, rid, k) / "gmm.npz"                                        # evaluate.fit_gmm
    from whistlepipe.vaegmm.cli import EXPERIMENTS

    assert rid == EXPERIMENTS["V1"]["run_id"] and k == vcfg["paper"]["n_components"]


# ----------------------------------------------------------------------------- upload reading
def test_read_upload_any_rate_and_channel(tmp_path):
    icfg = load_inference_config()
    p = synthetic_recording(tmp_path / "st.flac", fs=48000, dur=1.0, channels=2)
    s = audio.read_upload(p, 1, icfg)
    assert s.fs == 48000 and s.samples.ndim == 1 and s.meta["n_channels"] == 2 and s.meta["channel_index"] == 1
    assert s.meta["warnings"] == [] and s.samples.dtype == np.float32
    for fs in (32000, 44100):     # below 48 kHz the top of the 2-24 kHz detector band is missing: said, not refused
        low = synthetic_recording(tmp_path / f"low{fs}.wav", fs=fs, dur=1.0)
        assert "missing" in audio.read_upload(low, 0, icfg).meta["warnings"][0]


@pytest.mark.parametrize("case,msg", [("rate", "below"), ("channel", "channel"), ("long", "long"), ("type", "unsupported"),
                                      ("junk", "readable")])
def test_read_upload_rejects(tmp_path, case, msg):
    icfg = load_inference_config()
    if case == "rate":
        p = synthetic_recording(tmp_path / "a.wav", fs=22050, dur=1.0)
    elif case == "channel":
        p = synthetic_recording(tmp_path / "a.wav", fs=48000, dur=1.0)
    elif case == "long":
        p = tmp_path / "a.wav"
        sf.write(str(p), np.zeros(48000 * 121, np.int16), 48000)
    elif case == "type":
        p = tmp_path / "a.txt"
        p.write_text("x")
    else:
        p = tmp_path / "a.wav"
        p.write_bytes(b"not audio at all")
    with pytest.raises(audio.InputError, match=msg):
        audio.read_upload(p, 3 if case == "channel" else 0, icfg)


# ----------------------------------------------------------------------------- analysis
def _check_detected(r):
    assert len(r["whistles"]) == len(WHISTLES)
    for w, (t0, d, f) in zip(r["whistles"], WHISTLES):
        assert abs(w["start_s"] - t0) < 0.05 and abs(w["duration_ms"] / 1000 - d) < 0.06
        fr = f(np.linspace(0, 1, 200))
        assert abs(w["f_min_hz"] - fr.min()) < 300 and abs(w["f_max_hz"] - fr.max()) < 300


def test_analysis_without_model_files(tmp_path, rec):
    eng = make_engine(tmp_path)
    st = eng.status()
    assert st["model"]["ready"] == {"recreate": False, "same_type": False, "generate": False, "conditional": False}
    assert "not trained yet" in st["model"]["message"] and len(st["conditions"]) == 13
    out = tmp_path / "job"
    r = eng.analyze(rec, out)
    _check_detected(r)
    for w in r["whistles"]:
        assert "recreation" not in w and "samples" not in w
        for f in (w["input"]["png"], w["input"]["sound"]["wav"], w["recording"]["wav"]):
            assert (out / f).exists()
        assert abs(w["input"]["sound"]["fc_hz"] - w["fc_hz"]) < 0.1
    saved = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert saved["whistles"] == r["whistles"] and (out / "overview.png").exists() and (out / "overview_annotated.png").exists()
    assert saved["pipeline"]["matches_training_data"] in (True, None)
    with pytest.raises(audio.InputError, match="not trained"):
        eng.generate(tmp_path / "g")


def test_analysis_with_random_model_in_the_real_formats(tmp_path, rec):
    eng = make_engine(tmp_path, gmm=True, comp_feats=True)
    assert all(eng.model.status["ready"].values()), eng.model.status
    out = tmp_path / "job"
    r = eng.analyze(rec, out, n_samples=3, seed=5)
    _check_detected(r)
    assert r["samples_method"] == "gmm"
    for w in r["whistles"]:
        rc = w["recreation"]
        assert (out / rc["png"]).exists() and np.isfinite(rc["psnr_db"])
        assert 0 <= w["type"]["component"] < K and 0 < w["type"]["posterior"] <= 1
        assert len(w["samples"]) == 3 and all((out / s["png"]).exists() for s in w["samples"])
    # same seed -> same samples
    r2 = eng.analyze(rec, tmp_path / "job2", n_samples=3, seed=5)
    a = np.asarray(__import__("PIL.Image").Image.open(out / "w000_s0.png"))
    b = np.asarray(__import__("PIL.Image").Image.open(tmp_path / "job2" / "w000_s0.png"))
    assert np.array_equal(a, b) and r2["whistles"][0]["type"] == r["whistles"][0]["type"]


def test_vae_without_gmm_uses_the_encoder_spread(tmp_path, rec):
    eng = make_engine(tmp_path, gmm=False)
    st = eng.model.status
    assert st["ready"]["recreate"] and not st["ready"]["same_type"] and "fit-gmm" in st["message"]
    r = eng.analyze(rec, tmp_path / "job", n_samples=2)
    assert r["samples_method"] == "encoder_posterior"
    assert all("type" not in w and len(w["samples"]) == 2 for w in r["whistles"])


def test_mismatched_gmm_is_reported_not_fatal(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_random_model(runs, gmm=True, comp_feats=False)
    torch.save({"vae": VAE(32).state_dict(), "spec": {"latent_dim": 32}}, runs / RUN / "checkpoints" / "final.pt")
    eng = make_engine(tmp_path)
    assert eng.model.status["ready"]["recreate"] and not eng.model.status["ready"]["same_type"]
    assert "latent size" in eng.model.status["gmm"]["error"]


# ----------------------------------------------------------------------------- generation
def test_generate_unconditional_and_conditional(tmp_path):
    eng = make_engine(tmp_path, gmm=True, comp_feats=True)
    r = eng.generate(tmp_path / "g", n=3, seed=1)
    assert r["n"] == 3 and all(0 <= it["component"] < K for it in r["items"])
    assert all((tmp_path / "g" / it["png"]).exists() for it in r["items"])
    fixed = eng.generate(tmp_path / "g2", n=2, fc_hz=12000.0)
    assert all(it["fc_hz"] == 12000.0 for it in fixed["items"])
    # a random decoder fills the whole image, so its contours last 128 columns: only "long" conditions can be met
    cf = eng.model.comp_feats
    long_ok = [c for c in eng.conditions() if c["feature"] == "L" and (cf[..., 0] >= c["lo"]).any() and (cf[..., 0] <= c["hi"]).any()]
    with pytest.raises(audio.InputError):
        eng.generate(tmp_path / "g3", n=2, condition=1)
    with pytest.raises(audio.InputError, match="1-13"):
        eng.generate(tmp_path / "g4", n=2, condition=14)
    assert isinstance(long_ok, list)


def test_condition_labels_in_listener_units(tmp_path):
    eng = make_engine(tmp_path)
    c = {x["no"]: x["label"] for x in eng.conditions()}
    assert c[1] == "duration 160-213 ms"                        # 30-40 samples x 5.333 ms
    assert c[4] == "lowest pitch -2906 to -938 Hz from centre"  # bins 34-55, 0 Hz = bin 65 (-937.5 rounds to -938)
    assert c[8] == "pitch range +938 to +1875 Hz"               # 10-20 bins x 93.75 Hz


def test_recording_clip_from_a_500k_file_is_playable(tmp_path):
    icfg = load_inference_config()
    fs = 500000
    t = np.arange(fs) / fs
    sig = audio.AudioSignal((0.1 * np.sin(2 * np.pi * 9000 * t)).astype(np.float32), fs, "x")
    x, out_fs, band = infer.recording_clip(sig, 0.3, 0.6, 8500, 9500, icfg)
    assert out_fs == 48000 and band == [7500.0, 10500.0]
    assert len(x) == pytest.approx((0.3 + 2 * 0.05) * 48000, abs=50)
    assert abs(20 * np.log10(np.abs(x).max()) + 3) < 0.01
