"""Inference: a recording -> its whistles -> the model's images -> sound (web demo and ``python whistle.py infer``).

    upload (read_upload: any rate >= 32 kHz, one channel)
      -> S2..S9 exactly as configs/pipeline.toml (the training data's settings; processing hash checked)
      -> VAE encoder mean -> decoder                         = the recreation
      -> GMM eqs. 11-12 on the encoder mean                  = the whistle's type (MAP component)
      -> z ~ N(mu_k, Sigma_k) of that component -> decoder   = new whistles of the same type
      -> sonify.py on every image at the whistle's own centre frequency, plus the recording excerpt itself

The trained files are read from where the training commands write them (configs/inference.toml [model]):
    models/vaegmm/<run_id>/checkpoints/final.pt                 python whistle.py train-vae --experiment V1
    models/vaegmm/<run_id>/gmm_K<K>/gmm.npz (+ gmm.json)        python whistle.py fit-gmm --experiment V1
    models/vaegmm/<run_id>/gmm_K<K>/component_features.npy      python whistle.py vae-eval components --experiment V1
Missing files are reported (``python whistle.py model-status``); everything on the input side still runs.

[DECISION] when the VAE is present but the GMM is not, "new whistles" fall back to samples of the encoder's own
posterior z = mu + sigma * eps (a standard VAE operation, not the paper's generation method; labelled as such).
"""
from __future__ import annotations

import csv
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import signal

from . import audio, dsp, sonify
from .config import PROJECT_ROOT, Cfg, apply_overrides, canonical_hash, load_pipeline_config, load_toml, processing_hash
from .detector.ridgenet import resolve_device, sha256_file
from .representation import image_meta_rows, s7_vectors, s8_baseband, s9_images
from .tracking import run_s6

DEFAULT_INFERENCE_CFG = PROJECT_ROOT / "configs" / "inference.toml"
FEATURE_LABELS = {"L": "duration", "f_min": "lowest pitch", "f_max": "highest pitch", "df": "pitch range",
                  "f_start": "start pitch", "f_end": "end pitch", "f_c": "centre"}


def load_inference_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Cfg:
    path = Path(path) if path else DEFAULT_INFERENCE_CFG
    raw = load_toml(path)
    apply_overrides(raw, overrides)
    raw["_meta"] = {"config_path": str(path), "overrides": list(overrides or []),
                    "effective_hash": canonical_hash({k: v for k, v in raw.items() if k != "_meta"})}
    return Cfg(raw)


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def new_job_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:6]


# ----------------------------------------------------------------------------- the trained files
def model_paths(icfg: Cfg, vcfg: Cfg) -> dict[str, Path]:
    """Same layout as vaegmm/train.py (checkpoints/) and vaegmm/evaluate.gmm_dir (gmm_K<K>[_tag]/); tested."""
    m = icfg["model"]
    run = resolve(vcfg["paths"]["runs_dir"]) / m["run_id"]
    gd = run / f"gmm_K{int(m['n_components'])}{('_' + m['gmm_tag']) if m['gmm_tag'] else ''}"
    return {"run_dir": run, "vae": run / "checkpoints" / m["checkpoint"], "gmm": gd / "gmm.npz",
            "component_features": gd / "component_features.npy"}


@dataclass
class Model:
    status: dict
    vae: object | None = None
    gmm: object | None = None
    comp_feats: np.ndarray | None = None

    @property
    def latent_dim(self) -> int | None:
        return self.vae.latent_dim if self.vae is not None else None


def load_model(icfg: Cfg, vcfg: Cfg, device: torch.device) -> Model:
    from .vaegmm.gmm import GaussianMixtureEM
    from .vaegmm.train import load_trained_vae

    p = model_paths(icfg, vcfg)
    st = {k: {"path": rel(p[k]), "present": p[k].exists(), "loaded": False, "error": None}
          for k in ("vae", "gmm", "component_features")}
    st["run_id"] = icfg["model"]["run_id"]
    st["n_components"] = int(icfg["model"]["n_components"])
    m = Model(status=st)
    if st["vae"]["present"]:
        try:
            m.vae, spec = load_trained_vae(p["run_dir"], device, icfg["model"]["checkpoint"])
            st["vae"].update(loaded=True, latent_dim=m.vae.latent_dim, sha256=sha256_file(p["vae"]),
                             spec={k: spec[k] for k in ("latent_dim", "beta", "use_critic", "epochs") if k in spec})
        except Exception as e:  # noqa: BLE001  (reported to the page, never fatal)
            st["vae"]["error"] = repr(e)
    if st["gmm"]["present"]:
        try:
            dg = vcfg["decisions"]["gmm"]
            g = GaussianMixtureEM.load(p["gmm"], device, point_chunk=dg["point_chunk"], component_chunk=dg["component_chunk"], verbose=False)
            if m.vae is not None and g.means_.shape[1] != m.vae.latent_dim:
                raise ValueError(f"GMM latent size {g.means_.shape[1]} != VAE latent size {m.vae.latent_dim}")
            m.gmm = g
            st["gmm"].update(loaded=True, n_components=g.K, sha256=sha256_file(p["gmm"]))
        except Exception as e:  # noqa: BLE001
            st["gmm"]["error"] = repr(e)
    if st["component_features"]["present"] and m.gmm is not None:
        try:
            cf = np.load(p["component_features"])
            if cf.shape[0] != m.gmm.K:
                raise ValueError(f"component_features has {cf.shape[0]} components, the GMM {m.gmm.K}")
            m.comp_feats = cf
            st["component_features"]["loaded"] = True
        except Exception as e:  # noqa: BLE001
            st["component_features"]["error"] = repr(e)
    ok_v, ok_g = m.vae is not None, m.vae is not None and m.gmm is not None
    st["ready"] = {"recreate": ok_v, "same_type": ok_g, "generate": ok_g, "conditional": ok_g and m.comp_feats is not None}
    if not ok_v:
        st["message"] = (f"Model not trained yet: the recreation appears once {st['vae']['path']} exists "
                         f"(python whistle.py train-vae --experiment V1).")
    elif not ok_g:
        st["message"] = (f"VAE loaded; whistle types and generation appear once {st['gmm']['path']} exists "
                         f"(python whistle.py fit-gmm --experiment V1). New whistles meanwhile come from the encoder's own spread.")
    else:
        st["message"] = f"Model {st['run_id']} loaded (K = {m.gmm.K})."
    for k in ("vae", "gmm", "component_features"):
        if st[k]["error"]:
            st["message"] += f" {k} failed to load: {st[k]['error']}"
    return m


def training_centre_frequencies(pcfg: Cfg) -> np.ndarray:
    """Centre frequencies (Hz) of every training whistle, from preprocessed/encounter_*/images.csv."""
    d = resolve(pcfg["paths"]["dataset_dir"])
    out = []
    for f in sorted(d.glob("encounter_*/images.csv")):
        with open(f, encoding="utf-8") as fh:
            out += [float(r["center_hz"]) for r in csv.DictReader(fh)]
    return np.asarray(out, dtype=np.float64)


# ----------------------------------------------------------------------------- pieces of one analysis
def preprocess(sig: audio.AudioSignal, pcfg: Cfg, detector) -> dict:
    """S2..S9 on one signal, exactly as the dataset was built (pipeline._process_file, without storage)."""
    t, tm = time.perf_counter(), {}

    def lap(k):
        nonlocal t
        now = time.perf_counter()
        tm[k] = round(now - t, 3)
        t = now

    s3 = dsp.s3_highpass(dsp.s2_resample(sig, pcfg), pcfg)
    lap("s2_s3")
    s4 = dsp.s4_stft(s3, pcfg)
    lap("s4")
    dm = detector.predict(s4)
    lap("s5")
    s6 = run_s6(dm, pcfg)
    vec = s7_vectors(s6.tracks, pcfg)
    nfms, rejected = s8_baseband(vec, pcfg)
    images = s9_images(nfms, pcfg)
    rows = image_meta_rows(sig.name, nfms, vec, s4.hop_s)
    lap("s6_s9")
    return {"s4": s4, "vectors": vec, "nfms": nfms, "rejected": rejected, "images": images, "rows": rows,
            "n_tracks": len(s6.tracks), "timings_s": tm}


@torch.no_grad()
def run_model(model: Model, images: np.ndarray, n_samples: int, seed: int, device: torch.device, batch: int = 256) -> dict:
    """Recreation (decoder of the encoder mean), MAP component, and n_samples new whistles per input image."""
    if model.vae is None or not len(images):
        return {}
    vae = model.vae
    mus, lvs, recs = [], [], []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.ascontiguousarray(images[i:i + batch], dtype=np.float32))[:, None].to(device)
        mu, lv = vae.encoder(x)
        mus.append(mu)
        lvs.append(lv)
        recs.append(vae.decoder(mu)[:, 0].cpu().numpy())
    mu, lv, recon = torch.cat(mus), torch.cat(lvs), np.concatenate(recs)
    out = {"recon": recon, "samples": None, "method": None}
    n = len(images)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    if model.gmm is not None:
        lab, resp = model.gmm.predict(mu.cpu().numpy())
        out.update(component=lab, posterior=resp, weight=model.gmm.weights_.cpu().numpy()[lab])
        if n_samples:
            z, _ = model.gmm.sample(n * n_samples, gen, components=np.repeat(lab, n_samples))
            out.update(samples=_decode(vae, z, batch).reshape(n, n_samples, 128, 128), method="gmm")
    elif n_samples:
        eps = torch.randn(n, n_samples, mu.shape[1], generator=gen, device=device)
        z = (mu[:, None] + torch.exp(0.5 * lv)[:, None] * eps).reshape(-1, mu.shape[1])
        out.update(samples=_decode(vae, z, batch).reshape(n, n_samples, 128, 128), method="encoder_posterior")
    return out


def _decode(vae, z: torch.Tensor, batch: int) -> np.ndarray:
    return np.concatenate([vae.decoder(z[i:i + batch])[:, 0].cpu().numpy() for i in range(0, len(z), batch)])


def recording_clip(sig: audio.AudioSignal, start_s: float, end_s: float, f_lo: float, f_hi: float, icfg: Cfg) -> tuple[np.ndarray, int, list]:
    """The whistle as recorded: [start - margin, end + margin], band-passed around it, resampled to a playable rate."""
    o = icfg["output"]
    fs = int(sig.fs)
    m, guard = float(o["clip_margin_s"]), 0.1                  # guard: filter on a longer span, then cut (no edge transients)
    a, b = max(0, int((start_s - m - guard) * fs)), min(len(sig.samples), int(math.ceil((end_s + m + guard) * fs)))
    lo = max(500.0, f_lo - float(o["clip_band_margin_hz"]))
    hi = min(0.45 * fs, f_hi + float(o["clip_band_margin_hz"]))
    x = sig.samples[a:b].astype(np.float64)
    if hi > lo * 1.05:
        x = signal.sosfiltfilt(signal.butter(4, [lo, hi], "bandpass", fs=fs, output="sos"), x)
    c0 = int((start_s - m) * fs) - a
    x = x[max(0, c0): max(0, c0) + int((end_s - start_s + 2 * m) * fs)]
    out = sonify.audio_rate(hi, int(o["audio_rate_hz"]), int(o["high_audio_rate_hz"]))
    if out != fs:
        g = math.gcd(out, fs)
        x = signal.resample_poly(x, out // g, fs // g)
    fade = min(int(0.005 * out), len(x) // 2)
    if fade > 1:
        r = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade))
        x[:fade] *= r
        x[-fade:] *= r[::-1]
    x *= 10 ** (-3 / 20) / max(1e-12, float(np.abs(x).max()))
    return x.astype(np.float32), out, [round(lo, 1), round(hi, 1)]


def save_png(path: Path, img: np.ndarray) -> str:
    from PIL import Image

    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8), mode="L").save(path)
    return path.name


# Spectrogram palette of the page ("shallows": quiet = pale sea foam, loud = deep water). web/lab.js holds the same
# stops and colours overview.png and the 128 x 128 images in the browser; the PNGs themselves stay plain levels.
SHALLOWS = [(0.0, "#eef7f8"), (0.18, "#c8e2e9"), (0.4, "#7fb6c9"), (0.65, "#2a6e96"), (0.85, "#0d3a5c"), (1.0, "#041a2e")]


def overview_png(path: Path, s4, pcfg: Cfg, boxes: list[dict], rejected: list[dict], max_cols: int = 2400) -> dict:
    """The recording's spectrogram over the detector band, in dB above the local noise floor. Writes ``path`` (the
    bare image as 8-bit levels, 3..30 dB above the floor -> 0..255, full bleed: the page colours it and draws axes and
    clickable boxes over it) and ``<stem>_annotated.png`` (coloured, with axes, kept whistles boxed and numbered,
    rejected ones dashed) for reading results without the page."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    shallows = LinearSegmentedColormap.from_list("shallows", SHALLOWS)
    from .detector.conditioning import noise_floor_db

    k_lo, k_hi = pcfg["decisions"]["s5"]["band_lo_hz"], pcfg["decisions"]["s5"]["band_hi_hz"]
    bins = np.arange(int(k_lo // s4.bin_hz), int(k_hi // s4.bin_hz) + 1)
    db = s4.db[:, bins]
    c = pcfg["decisions"]["s5"]["conditioning"]
    db = db - noise_floor_db(db, int(c["block_frames"]), int(c["median_blocks"]))    # dB above the local floor, as S5 sees it
    step = max(1, math.ceil(len(db) / max_cols))
    if step > 1:
        db = db[: len(db) // step * step].reshape(-1, step, db.shape[1]).max(1)
    dur = s4.n_frames * s4.hop_s
    from PIL import Image

    lev = np.clip((db.T[::-1] - 3) / 27, 0, 1)
    Image.fromarray((lev * 255).round().astype(np.uint8), mode="L").save(path)
    annotated = path.with_name(path.stem + "_annotated.png")
    fig = plt.figure(figsize=(16, 4.2), dpi=100, facecolor="#eef7f8")
    ax = fig.add_axes([0.045, 0.14, 0.945, 0.8], facecolor="#eef7f8")
    ax.imshow(db.T, origin="lower", aspect="auto", cmap=shallows, vmin=3, vmax=30,
              extent=[0, dur, bins[0] * s4.bin_hz / 1000, bins[-1] * s4.bin_hz / 1000], interpolation="nearest")
    for b in boxes:
        ax.add_patch(plt.Rectangle((b["start_s"], b["f_min_hz"] / 1000 - 0.3), b["end_s"] - b["start_s"],
                                   (b["f_max_hz"] - b["f_min_hz"]) / 1000 + 0.6, fill=False, ec="#0b2f4d", lw=1.2))
        ax.text(b["start_s"], b["f_max_hz"] / 1000 + 0.5, str(b["i"] + 1), color="#0b2f4d", fontsize=10, fontweight="bold")
    for r in rejected:
        if r.get("f_min_hz") is not None:
            ax.add_patch(plt.Rectangle((r["start_s"], r["f_min_hz"] / 1000 - 0.25), r["end_s"] - r["start_s"],
                                       (r["f_max_hz"] - r["f_min_hz"]) / 1000 + 0.5, fill=False, ec="#b5762f", lw=1.0, ls="--"))
    ax.set_xlabel("time (s)", color="#24435a")
    ax.set_ylabel("kHz", color="#24435a")
    ax.tick_params(colors="#24435a", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#9fc2cf")
    fig.savefig(annotated, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"png": path.name, "annotated_png": annotated.name, "t1_s": round(dur, 3),
            "f_lo_hz": float(bins[0] * s4.bin_hz), "f_hi_hz": float(bins[-1] * s4.bin_hz)}


def contour_error_hz(a: np.ndarray, b: np.ndarray, cfg: Cfg, thr: float) -> float | None:
    ca, cb = sonify.image_contour(a, cfg, thr), sonify.image_contour(b, cfg, thr)
    if ca is None or cb is None:
        return None
    _, ia, ib = np.intersect1d(ca.cols, cb.cols, return_indices=True)
    return round(float(np.abs(ca.offset_hz[ia] - cb.offset_hz[ib]).mean()), 1) if len(ia) else None


# ----------------------------------------------------------------------------- engine
class Engine:
    """Loads the detector, the trained files and the training centre frequencies once; runs one job at a time."""

    def __init__(self, icfg: Cfg | None = None, pcfg: Cfg | None = None, vcfg: Cfg | None = None, device: str | None = None):
        from .detector import RidgeDetector
        from .vaegmm.config import load_vaegmm_config

        self.icfg = icfg or load_inference_config()
        self.pcfg = pcfg or load_pipeline_config()
        self.vcfg = vcfg or load_vaegmm_config()
        self.device = resolve_device(device or self.icfg["model"]["device"])
        self.detector = RidgeDetector.from_checkpoint(resolve(self.pcfg["decisions"]["s5"]["checkpoint"]), self.pcfg, str(self.device))
        self.model = load_model(self.icfg, self.vcfg, self.device)
        self.fc_pool = training_centre_frequencies(self.pcfg)
        self.processing_hash = processing_hash(self.pcfg)
        info = resolve(self.pcfg["paths"]["dataset_dir"]) / "dataset_info.json"
        self.training_hash = json.loads(info.read_text(encoding="utf-8")).get("processing_config_hash") if info.exists() else None
        self.lock = threading.Lock()

    def reload_model(self) -> dict:
        with self.lock:
            self.model = load_model(self.icfg, self.vcfg, self.device)
        return self.model.status

    def status(self) -> dict:
        c = self.icfg["input"]
        return {"model": self.model.status, "device": str(self.device),
                "detector": f"{self.detector.name} {self.detector.version}",
                "processing_hash": self.processing_hash, "matches_training_data": (self.processing_hash == self.training_hash)
                if self.training_hash else None,
                "limits": {"formats": list(c["formats"]), "min_sample_rate_hz": c["min_sample_rate_hz"],
                           "max_duration_s": c["max_duration_s"], "max_upload_mb": c["max_upload_mb"],
                           "n_samples": self.icfg["output"]["n_samples"], "max_generate": self.icfg["generate"]["max_n"]},
                "conditions": self.conditions(), "n_training_whistles": int(len(self.fc_pool))}

    def conditions(self) -> list[dict]:
        """The paper's 13 Table 3 conditions in listener's units (ms; Hz relative to the whistle's centre)."""
        fz = self.vcfg["decisions"]["features"]
        ms, hz = float(fz["ms_per_sample"]), float(fz["hz_per_bin"])
        zero_bin = int(self.pcfg["paper"]["image_height"]) // 2 + 1          # bin = 128 - row; 0 Hz on row 63 -> bin 65
        out = []
        for c in self.vcfg["paper"]["table3_conditions"]:
            if c["feature"] == "L":
                lo, hi, unit = c["lo"] * ms, c["hi"] * ms, "ms"
            elif c["feature"] == "df":
                lo, hi, unit = c["lo"] * hz, c["hi"] * hz, "Hz"
            else:
                lo, hi, unit = (c["lo"] - zero_bin) * hz, (c["hi"] - zero_bin) * hz, "Hz from centre"
            out.append({**c, "label": f"{FEATURE_LABELS[c['feature']]} {lo:+.0f} to {hi:+.0f} {unit}" if unit != "ms"
                        else f"duration {lo:.0f}-{hi:.0f} ms"})
        return out

    def pick_fc(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if len(self.fc_pool):
            return rng.choice(self.fc_pool, n)
        return np.full(n, float(self.icfg["generate"]["fallback_fc_hz"]))

    # ------------------------------------------------------------------ analysis of a recording
    def analyze(self, path: str | Path, out_dir: str | Path, channel: int | None = None, n_samples: int | None = None,
                seed: int = 0, name: str | None = None, progress=None) -> dict:
        say = progress or (lambda stage, frac=None: None)
        o, thr = self.icfg["output"], float(self.icfg["output"]["contour_threshold"])
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        say("reading")
        ch = int(self.icfg["input"]["default_channel"] if channel is None else channel)
        sig = audio.read_upload(Path(path), ch, self.icfg, name)
        say("detecting")
        pre = preprocess(sig, self.pcfg, self.detector)
        order = np.argsort([r["start_time_s"] for r in pre["rows"]], kind="stable")
        rows = [pre["rows"][i] for i in order]
        images = pre["images"][order] if len(order) else pre["images"]
        hop_s, win_s = pre["s4"].hop_s, pre["s4"].win_len / pre["s4"].fs
        say("model")
        n_s = int(o["n_samples"] if n_samples is None else n_samples)
        mo = run_model(self.model, images, n_s, seed, self.device)
        say("sound")
        whistles = []
        for i, r in enumerate(rows):
            if progress and i % 5 == 0:
                say("sound", i / max(1, len(rows)))
            start, end = r["start_time_s"], r["start_time_s"] + r["n_frames"] * hop_s + win_s
            fc = r["f_center_hz"]
            w = {"i": i, "start_s": round(start, 4), "end_s": round(end, 4), "duration_ms": round(r["n_frames"] * hop_s * 1000, 1),
                 "n_frames": r["n_frames"], "fc_hz": fc, "f_min_hz": r["f_min_hz"], "f_max_hz": r["f_max_hz"],
                 "bandwidth_hz": r["bandwidth_hz"], "mean_conf": r["mean_conf"], "rendered": i < int(o["max_rendered"])}
            tag = f"w{i:03d}"
            w["input"] = {"png": save_png(out / f"{tag}_input.png", images[i])}
            if w["rendered"]:
                w["input"]["sound"] = sonify.image_to_wav(images[i], fc, out / f"{tag}_input.wav", self.pcfg, thr)
                x, fs, band = recording_clip(sig, start, end, r["f_min_hz"], r["f_max_hz"], self.icfg)
                sonify.write_wav(out / f"{tag}_recording.wav", x, fs)
                w["recording"] = {"wav": f"{tag}_recording.wav", "fs": fs, "band_hz": band}
            if mo:
                rec = mo["recon"][i]
                sse = float(((rec - images[i]) ** 2).sum())
                w["recreation"] = {"png": save_png(out / f"{tag}_recreation.png", rec),
                                   "psnr_db": round(10 * math.log10(images[i].size / max(sse, 1e-12)), 2), "sse": round(sse, 3),
                                   "contour_error_hz": contour_error_hz(images[i], rec, self.pcfg, thr)}
                if w["rendered"]:
                    w["recreation"]["sound"] = sonify.image_to_wav(rec, fc, out / f"{tag}_recreation.wav", self.pcfg, thr)
                if "component" in mo:
                    w["type"] = {"component": int(mo["component"][i]), "posterior": round(float(mo["posterior"][i]), 4),
                                 "weight": float(mo["weight"][i])}
                if mo["samples"] is not None:
                    w["samples"] = []
                    for j, s in enumerate(mo["samples"][i]):
                        e = {"png": save_png(out / f"{tag}_s{j}.png", s)}
                        if w["rendered"]:
                            e["sound"] = sonify.image_to_wav(s, fc, out / f"{tag}_s{j}.wav", self.pcfg, thr)
                        w["samples"].append(e)
                    w["samples_method"] = mo["method"]
            whistles.append(w)
        # S8 rejections (too long / too wide for a 128 x 128 image), placed from their S7 vectors
        vec = {v.track_id: v for v in pre["vectors"]}
        rejected = []
        for r in sorted(pre["rejected"], key=lambda r: r["start_frame"]):
            v = vec[r["track_id"]]
            start = v.start_frame * hop_s
            d = {"start_s": round(start, 4), "end_s": round(start + v.n_frames * hop_s + win_s, 4), "reason": r["reason"],
                 "f_min_hz": round(v.f_min, 1), "f_max_hz": round(v.f_max, 1)}
            d["detail"] = (f"{v.n_frames * hop_s * 1000:.0f} ms long; an image holds "
                           f"{int(self.pcfg['paper']['image_width']) * hop_s * 1000:.0f} ms") if r["reason"] == "too_long" else \
                          f"spans {v.f_max - v.f_min:.0f} Hz; an image holds about 12 kHz"
            rejected.append(d)
        say("overview")
        ov = overview_png(out / "overview.png", pre["s4"], self.pcfg, whistles, rejected)
        result = {
            "kind": "analysis", "id": out.name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input": {"name": sig.name, **{k: sig.meta[k] for k in ("fs_in_hz", "n_channels", "channel_index", "duration_s", "warnings")}},
            "pipeline": {"processing_hash": self.processing_hash, "matches_training_data":
                         (self.processing_hash == self.training_hash) if self.training_hash else None,
                         "detector": f"{self.detector.name} {self.detector.version}", "n_tracks": pre["n_tracks"]},
            "model": self.model.status, "n_samples": n_s, "seed": seed, "samples_method": mo.get("method") if mo else None,
            "overview": ov, "whistles": whistles, "rejected": rejected,
            "timings_s": {**pre["timings_s"], "total": round(time.perf_counter() - t0, 2)},
        }
        (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        say("done", 1.0)
        return result

    # ------------------------------------------------------------------ generation without a recording
    @torch.no_grad()
    def generate(self, out_dir: str | Path, n: int = 8, seed: int = 0, condition: int | None = None,
                 fc_hz: float | None = None, progress=None) -> dict:
        from .vaegmm.conditional import acceptance_ratios, conditional_sample, weights_eq8, weights_eq10
        from .vaegmm.features import FEATURES, contour_features

        say = progress or (lambda stage, frac=None: None)
        g, thr = self.icfg["generate"], float(self.icfg["output"]["contour_threshold"])
        m = self.model
        if not m.status["ready"]["generate"]:
            raise audio.InputError(m.status["message"])
        n = int(max(1, min(int(g["max_n"]), n)))
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        rng = np.random.default_rng(int(seed))
        say("model")
        cond, comps, rate = None, None, None
        if condition:
            if m.comp_feats is None:
                raise audio.InputError("Conditional generation needs component_features.npy "
                                       "(python whistle.py vae-eval components --experiment V1).")
            cond = next((c for c in self.conditions() if c["no"] == int(condition)), None)
            if cond is None:
                raise audio.InputError(f"no Table 3 condition {condition} (1-13)")
            a = acceptance_ratios(m.comp_feats, cond["feature"], cond["lo"], cond["hi"])
            member = a > 0
            if not member.any():
                raise audio.InputError(f"no component produces whistles with {cond['label']}")
            pi = m.gmm.weights_.cpu().numpy()
            w = weights_eq10(pi, a, member) if g["weights"] == "eq10" else weights_eq8(pi, member)
            try:
                r = conditional_sample(m.vae.decoder, m.gmm, w, cond["feature"], cond["lo"], cond["hi"], n, gen,
                                       thr, batch=min(4000, 64 * n), max_draws=int(g["max_draws"]), keep_images=n)
            except RuntimeError as e:
                raise audio.InputError(f"too few samples met {cond['label']}: {e}") from e
            images, rate = r["images"], round(float(r["acceptance_rate"]), 5)
        else:
            z, c = m.gmm.sample(n, gen)
            images, comps = m.vae.decoder(z)[:, 0].cpu().numpy(), c.cpu().numpy()
        feats, _ = contour_features(torch.from_numpy(np.ascontiguousarray(images)), thr)
        fcs = np.full(len(images), float(fc_hz)) if fc_hz else self.pick_fc(rng, len(images))
        say("sound")
        items = []
        for i, img in enumerate(images):
            e = {"i": i, "png": save_png(out / f"g{i:03d}.png", img), "fc_hz": round(float(fcs[i]), 1),
                 "component": int(comps[i]) if comps is not None else None,
                 "features": {k: (None if not np.isfinite(v) else round(float(v), 2)) for k, v in zip(FEATURES, feats[i].tolist())}}
            e["sound"] = sonify.image_to_wav(img, fcs[i], out / f"g{i:03d}.wav", self.pcfg, thr)
            items.append(e)
        result = {"kind": "generation", "id": out.name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "model": m.status, "n": len(items), "seed": seed, "condition": cond, "acceptance_rate": rate,
                  "weights": g["weights"] if cond else "pi", "fc": "fixed" if fc_hz else "drawn from the training whistles",
                  "items": items}
        (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        say("done", 1.0)
        return result
