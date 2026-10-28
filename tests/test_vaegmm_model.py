"""VAE-GMM: paper lock, architecture (Tables 6-8), losses (eqs. 3-5), learning-rate verification, data layer, resume.
CPU only, tiny synthetic data; nothing here needs the real dataset or trains a real model."""
import copy
import json

import numpy as np
import pytest
import torch

from whistlepipe.vaegmm.config import TRAINING_PAPER_LOCK, load_vaegmm_config, validate_training_paper_lock
from whistlepipe.vaegmm.data import SubsetImages, WhistleImages, epoch_plan, random_rot90
from whistlepipe.vaegmm.losses import adv_neg_log_d, critic_loss, kl_std_normal, recon_bce, recon_sse, vae_loss
from whistlepipe.vaegmm.models import VAE, Critic, Decoder, Encoder, n_params, table6_flops
from whistlepipe.vaegmm.train import Trainer, run_spec, train_vae


@pytest.fixture(scope="module")
def vcfg():
    return load_vaegmm_config()


def fresh_cfg(vcfg):
    return type(vcfg)(copy.deepcopy(dict(vcfg)))


# ---------------------------------------------------------------- configuration
def test_paper_lock_matches_config_and_rejects_changes(vcfg):
    raw = vcfg.to_dict()
    validate_training_paper_lock(raw)
    bad = copy.deepcopy(raw)
    bad["paper"]["lambda_gp"] = 10.0
    with pytest.raises(ValueError, match="lambda_gp"):
        validate_training_paper_lock(bad)
    extra = copy.deepcopy(raw)
    extra["paper"]["dropout"] = 0.1
    with pytest.raises(ValueError, match="undeclared"):
        validate_training_paper_lock(extra)


def test_linear_lr_deviation_is_exactly_128_over_896(vcfg):
    d, p = vcfg["deviations"], vcfg["paper"]
    assert d["batch_size"] == 128 and p["batch_per_gpu"] * p["n_gpus"] == 896
    assert d["lr_vae"] == pytest.approx(1e-3 * 128 / 896, rel=1e-12)
    assert d["lr_critic"] == pytest.approx(2e-4 * 128 / 896, rel=1e-12)
    raw = vcfg.to_dict()
    raw["deviations"]["lr_vae"] = 1e-3
    with pytest.raises(ValueError, match="linear-scaled"):
        validate_training_paper_lock(raw)


def test_table3_conditions_locked():
    conds = TRAINING_PAPER_LOCK["table3_conditions"]
    assert len(conds) == 13 and conds[3] == {"no": 4, "feature": "f_min", "lo": 34, "hi": 55}
    assert conds[4] == {"no": 5, "feature": "f_min", "lo": 40, "hi": 53}       # Fig. 12 text example Z(4, 5)


# ---------------------------------------------------------------- architecture (Tables 6-8)
def test_encoder_decoder_shapes_follow_table7():
    enc, dec = Encoder(64), Decoder(64)
    x = torch.zeros(2, 1, 128, 128)
    sizes = []
    h = x
    for c in enc.convs:
        h = c(h)
        sizes.append(tuple(h.shape[1:]))
    assert sizes == [(32, 64, 64), (32, 32, 32), (64, 16, 16), (64, 8, 8)]
    mu, lv = enc(x)
    assert mu.shape == (2, 64) and lv.shape == (2, 64)
    h = dec.proj(mu).view(-1, 64, 8, 8)
    sizes = []
    for t in dec.tconvs:
        h = t(h)
        sizes.append(tuple(h.shape[1:]))
    assert sizes == [(64, 16, 16), (64, 32, 32), (32, 64, 64), (32, 128, 128)]
    y = dec(mu).detach()
    assert y.shape == (2, 1, 128, 128) and float(y.min()) >= 0 and float(y.max()) <= 1


def test_decoder_reproduces_table6_params_and_flops():
    dec = Decoder(64)
    assert n_params(dec) == 368_097                       # Table 6: 368k
    assert table6_flops(dec, (64,)) == 557_318_144         # Table 6: 557M


def test_critic_follows_table8_and_outputs_probability():
    c = Critic()
    x = torch.rand(3, 1, 128, 128)
    h = torch.nn.functional.leaky_relu(c.c1(x), 0.2)
    sizes = [tuple(h.shape[1:])]
    for conv, norm in zip(c.blocks, c.norms):
        h = torch.nn.functional.leaky_relu(norm(conv(h)), 0.2)
        sizes.append(tuple(h.shape[1:]))
    assert sizes == [(32, 64, 64), (32, 32, 32), (64, 16, 16), (64, 8, 8)]
    d = c(x)
    assert d.shape == (3,) and bool(((d > 0) & (d < 1)).all())
    assert n_params(c) == 184_609
    assert torch.allclose(c(x), torch.sigmoid(c.logit(x)))


def test_glorot_init_and_zero_bias():
    v = VAE(64)
    w = v.encoder.convs[1].weight.detach()
    bound = (6.0 / (9 * 32 + 9 * 32)) ** 0.5
    assert float(w.abs().max()) <= bound and float(v.encoder.convs[1].bias.detach().abs().max()) == 0.0


# ---------------------------------------------------------------- losses (eqs. 3-5)
def test_kl_closed_form_matches_monte_carlo():
    torch.manual_seed(0)
    mu = torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float64)
    lv = torch.tensor([[-0.3, 0.4, -1.2]], dtype=torch.float64)
    z = mu + torch.exp(0.5 * lv) * torch.randn(400_000, 3, dtype=torch.float64)
    logq = (-0.5 * ((z - mu) ** 2 / lv.exp() + lv + np.log(2 * np.pi))).sum(1)
    logp = (-0.5 * (z ** 2 + np.log(2 * np.pi))).sum(1)
    assert float(kl_std_normal(mu, lv)) == pytest.approx(float((logq - logp).mean()), abs=0.01)


def test_recon_is_per_image_sum_and_losses_are_batch_means():
    x = torch.zeros(2, 1, 4, 4)
    logits = torch.zeros(2, 1, 4, 4)                         # sigmoid = 0.5
    assert torch.allclose(recon_sse(torch.sigmoid(logits), x), torch.tensor([4.0, 4.0]))
    mu, lv = torch.zeros(2, 3), torch.zeros(2, 3)
    total, parts = vae_loss(logits, x, mu, lv, torch.zeros(2), beta=1e-4, lambda_adv=1.0, recon="sse_sum")
    assert float(parts["rec"]) == 4.0 and float(parts["adv"]) == pytest.approx(np.log(2))
    assert float(total) == pytest.approx(4.0 + np.log(2))
    total, parts = vae_loss(logits, x, mu, lv, torch.zeros(2), beta=1e-4, lambda_adv=1.0)       # default: bce_sum
    assert float(parts["rec"]) == pytest.approx(16 * np.log(2))                              # 16 pixels x ln 2
    with pytest.raises(ValueError):
        vae_loss(logits, x, mu, lv, torch.zeros(2), 1e-4, 1.0, recon="l1")


def test_bce_reconstruction_matches_cross_entropy_and_has_no_saturation():
    torch.manual_seed(0)
    x = torch.rand(3, 1, 8, 8, dtype=torch.float64)
    lg = 3 * torch.randn(3, 1, 8, 8, dtype=torch.float64)
    ref = torch.nn.functional.binary_cross_entropy(torch.sigmoid(lg), x, reduction="none").flatten(1).sum(1)
    assert torch.allclose(recon_bce(lg, x), ref, rtol=1e-10)
    # a whistle pixel (x = 1) predicted at logit -40: squared error has ~no gradient, cross-entropy does
    lg1 = torch.tensor([[[[-40.0]]]], dtype=torch.float64, requires_grad=True)
    one = torch.ones_like(lg1)
    (g_sse,) = torch.autograd.grad(recon_sse(torch.sigmoid(lg1), one).sum(), lg1)
    (g_bce,) = torch.autograd.grad(recon_bce(lg1, one).sum(), lg1)
    assert abs(float(g_sse)) < 1e-15 and float(g_bce) == pytest.approx(-1.0)


def test_neg_log_d_equals_log_sigmoid_and_stays_finite():
    a = torch.linspace(-15, 15, 31, dtype=torch.float64)
    assert torch.allclose(adv_neg_log_d(a), -torch.log(torch.sigmoid(a)))
    far = torch.tensor([-200.0])                     # sigmoid underflows to 0 in float32 -> log would give inf
    assert torch.sigmoid(far) == 0 and torch.isfinite(adv_neg_log_d(far)).all()


class _LinCritic(torch.nn.Module):
    """D(x) = sigmoid(w . x + b): grad_x D = sigmoid'(a) w is known analytically."""

    def __init__(self, n):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(n, dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

    def logit(self, x):
        return x.flatten(1) @ self.w + self.b

    def forward(self, x):
        return torch.sigmoid(self.logit(x))


def test_gradient_penalty_is_on_the_post_sigmoid_output():
    torch.manual_seed(1)
    c = _LinCritic(16)
    xr, xf = torch.rand(5, 1, 4, 4, dtype=torch.float64), torch.rand(5, 1, 4, 4, dtype=torch.float64)
    eps = torch.rand(5, 1, 1, 1, dtype=torch.float64)
    loss, parts = critic_loss(c, xr, xf, eps, lambda_gp=5.0)
    xt = eps * xr + (1 - eps) * xf
    s = torch.sigmoid(c.logit(xt))
    gnorm = s * (1 - s) * c.w.norm()
    expect = c(xf).mean() - c(xr).mean() + 5.0 * ((gnorm - 1) ** 2).mean()
    assert float(loss) == pytest.approx(float(expect), rel=1e-12)
    assert float(parts["gnorm"]) == pytest.approx(float(gnorm.mean()), rel=1e-12)


def test_critic_loss_parameter_gradient_matches_finite_differences():
    torch.manual_seed(2)
    c = _LinCritic(16)
    xr, xf = torch.rand(4, 1, 4, 4, dtype=torch.float64), torch.rand(4, 1, 4, 4, dtype=torch.float64)
    eps = torch.rand(4, 1, 1, 1, dtype=torch.float64)
    loss, _ = critic_loss(c, xr, xf, eps, 5.0)
    (gw,) = torch.autograd.grad(loss, [c.w])
    h = 1e-6
    for i in (0, 7, 15):
        with torch.no_grad():
            c.w[i] += h
        lp = float(critic_loss(c, xr, xf, eps, 5.0)[0])
        with torch.no_grad():
            c.w[i] -= 2 * h
        lm = float(critic_loss(c, xr, xf, eps, 5.0)[0])
        with torch.no_grad():
            c.w[i] += h
        assert float(gw[i]) == pytest.approx((lp - lm) / (2 * h), rel=1e-5, abs=1e-8)


# ---------------------------------------------------------------- learning-rate verification (specification §3.3)
def test_accumulated_microbatch_gradient_equals_large_batch_gradient():
    """Batch-mean reduction: mean of 7 micro-batch gradients (128 each in the paper) == gradient of the 896 batch."""
    torch.manual_seed(3)
    vae, cri = VAE(8).double(), Critic().double()
    x = torch.rand(14, 1, 128, 128, dtype=torch.float64) * (torch.rand(14, 1, 128, 128, dtype=torch.float64) > 0.98)
    eps = torch.randn(14, 8, dtype=torch.float64)

    def grads(xb, eb):
        vae.zero_grad()
        lg, mu, lv = vae.forward_logits(xb, eb)
        loss, _ = vae_loss(lg, xb, mu, lv, cri.logit(torch.sigmoid(lg)), 1e-4, 1.0)
        loss.backward()
        return [p.grad.clone() for p in vae.parameters()]

    full = grads(x, eps)
    parts = [grads(x[i:i + 2], eps[i:i + 2]) for i in range(0, 14, 2)]
    for j, g in enumerate(full):
        assert torch.allclose(g, sum(p[j] for p in parts) / 7, rtol=1e-9, atol=1e-12)


def test_adam_is_invariant_to_gradient_scale_so_sum_vs_mean_aggregation_does_not_matter():
    torch.manual_seed(4)
    p1 = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64))
    p2 = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64))
    o1, o2 = torch.optim.Adam([p1], lr=1e-3), torch.optim.Adam([p2], lr=1e-3)
    for _ in range(50):
        g = torch.randn(5, dtype=torch.float64)
        p1.grad, p2.grad = g.clone(), 7.0 * g          # 7 GPUs summed instead of averaged
        o1.step()
        o2.step()
    assert torch.allclose(p1, p2, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------- augmentation and data
def test_random_rot90_matches_torch_rot90_per_image():
    x = torch.rand(4, 1, 6, 6)
    k = torch.tensor([0, 1, 2, 3])
    y = random_rot90(x, k)
    for i in range(4):
        assert torch.equal(y[i], torch.rot90(x[i], int(k[i]), dims=(1, 2)))


def test_rotation_draws_are_uniform():
    g = torch.Generator().manual_seed(0)
    k = torch.randint(0, 4, (40_000,), generator=g)
    frac = torch.bincount(k, minlength=4).double() / len(k)
    assert torch.allclose(frac, torch.full((4,), 0.25, dtype=torch.float64), atol=0.01)


def test_epoch_plan_covers_every_image_once_and_is_deterministic():
    vb, cb = epoch_plan(1000, 128, 3, seed=7, epoch=1)
    allv = np.concatenate(vb)
    assert sorted(allv.tolist()) == list(range(1000)) and len(vb) == 8 and len(vb[-1]) == 1000 - 7 * 128
    assert all(len(c) == 3 and all(len(b) == 128 for b in c) for c in cb)
    vb2, cb2 = epoch_plan(1000, 128, 3, seed=7, epoch=1)
    assert all(np.array_equal(a, b) for a, b in zip(vb, vb2)) and np.array_equal(cb[2][1], cb2[2][1])
    vb3, _ = epoch_plan(1000, 128, 3, seed=7, epoch=2)
    assert not np.array_equal(vb[0], vb3[0])
    _, none = epoch_plan(1000, 128, 3, seed=7, epoch=1, use_critic=False)
    assert all(c == [] for c in none)


def make_fake_dataset(root, counts=(20, 12), seed=0):
    rng = np.random.default_rng(seed)
    info = {"encounters": {}, "totals": {"images": int(sum(counts))}, "processing_config_hash": "fake",
            "detector": {"checkpoint_sha256": "fake"}}
    for e, n in enumerate(counts, start=1):
        d = root / f"encounter_{e:03d}"
        d.mkdir(parents=True)
        x = np.zeros((n, 128, 128), np.float32)
        for i in range(n):
            r0, c0, ln = rng.integers(40, 80), rng.integers(10, 60), rng.integers(12, 60)
            for t in range(ln):
                r = int(r0 + 6 * np.sin(t / 9))
                x[i, r - 1:r + 2, c0 + t] = [0.4, 1.0, 0.4]
        np.save(d / "images.npy", x)
        info["encounters"][d.name] = {"n_images": n}
    (root / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


def test_loader_gathers_in_requested_order_and_is_read_only(tmp_path):
    root = make_fake_dataset(tmp_path / "pre")
    im = WhistleImages(root)
    assert len(im) == 32
    ref = np.concatenate([np.load(root / "encounter_001" / "images.npy"), np.load(root / "encounter_002" / "images.npy")])
    idx = np.array([31, 0, 25, 3, 19, 20])
    assert np.array_equal(im.gather(idx), ref[idx])
    with pytest.raises(ValueError):
        im.arrays[0][0, 0, 0] = 1.0                 # memory-mapped read-only
    s = im.scan()
    assert s["n_non_finite"] == 0 and s["n_out_of_range"] == 0 and s["n_max_not_1"] == 0
    sub = SubsetImages(im, np.array([5, 30]))
    assert np.array_equal(sub.gather(np.array([1, 0])), ref[[30, 5]])


def test_loader_rejects_count_mismatch(tmp_path):
    root = make_fake_dataset(tmp_path / "pre")
    info = json.loads((root / "dataset_info.json").read_text())
    info["encounters"]["encounter_002"]["n_images"] = 13
    (root / "dataset_info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="dataset_info"):
        WhistleImages(root)


# ---------------------------------------------------------------- training mechanics (CPU, tiny)
def test_trainer_step_updates_both_networks_and_keeps_critic_frozen_in_vae_step(vcfg, tmp_path):
    im = WhistleImages(make_fake_dataset(tmp_path / "pre"))
    spec = {**run_spec(vcfg, "t"), "batch_size": 4, "latent_dim": 8}
    tr = Trainer(spec, im, torch.device("cpu"))
    v0 = [p.clone() for p in tr.vae.parameters()]
    c0 = [p.clone() for p in tr.critic.parameters()]
    vb, cb = epoch_plan(len(im), 4, 3, spec["seed"], 1)
    to_t = lambda a: torch.from_numpy(im.gather(a))[:, None]
    r = tr.step([to_t(c) for c in cb[0]], to_t(vb[0]), torch.Generator().manual_seed(0), measure=True)
    assert all(torch.isfinite(v).all() for v in r.values())
    assert any(not torch.equal(a, b) for a, b in zip(v0, tr.vae.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(c0, tr.critic.parameters()))
    assert all(p.requires_grad for p in tr.critic.parameters())
    assert tr.opt_c.state_dict()["param_groups"][0]["betas"] == (0.0, 0.9)


def test_resume_at_epoch_boundary_is_bit_identical(vcfg, tmp_path):
    cfg = fresh_cfg(vcfg)
    cfg["train"]["device"] = "cpu"
    im = WhistleImages(make_fake_dataset(tmp_path / "pre"))
    base = {**run_spec(cfg, "a", epochs=2), "batch_size": 8, "latent_dim": 8}
    out = tmp_path / "runs"
    train_vae(cfg, base, images=im, out_root=out, write_samples=False, quiet=True)
    sb = {**base, "run_id": "b"}
    train_vae(cfg, sb, images=im, out_root=out, stop_after_epoch=1, write_samples=False, quiet=True)
    assert not (out / "b" / "checkpoints" / "final.pt").exists()
    train_vae(cfg, sb, images=im, out_root=out, resume=True, write_samples=False, quiet=True)
    a = torch.load(out / "a" / "checkpoints" / "final.pt", weights_only=False)
    b = torch.load(out / "b" / "checkpoints" / "final.pt", weights_only=False)
    for k in ("vae", "critic"):
        for n in a[k]:
            assert torch.equal(a[k][n], b[k][n]), (k, n)
    info = json.loads((out / "a" / "train_info.json").read_text())
    assert info["status"] == "complete" and info["spec"]["batch_size"] == 8 and "source_sha256" in info
    with pytest.raises(FileExistsError):
        train_vae(cfg, base, images=im, out_root=out, write_samples=False, quiet=True)


def test_no_critic_run_has_no_adversarial_term(vcfg, tmp_path):
    im = WhistleImages(make_fake_dataset(tmp_path / "pre"))
    spec = {**run_spec(vcfg, "nc", beta=1.0, use_critic=False), "batch_size": 4, "latent_dim": 8}
    tr = Trainer(spec, im, torch.device("cpu"))
    assert tr.critic is None and tr.opt_c is None
    vb, cb = epoch_plan(len(im), 4, 3, spec["seed"], 1, use_critic=False)
    r = tr.step(cb[0], torch.from_numpy(im.gather(vb[0]))[:, None], torch.Generator().manual_seed(0))
    assert "adv" not in r and "loss_d" not in r


def test_micro_batched_step_equals_whole_batch_step(vcfg, tmp_path):
    """Gradient accumulation over chunks (memory only) gives the same update as the whole batch."""
    im = WhistleImages(make_fake_dataset(tmp_path / "pre"))
    spec = {**run_spec(vcfg, "mb"), "batch_size": 8, "latent_dim": 8}
    vb, cb = epoch_plan(len(im), 8, 3, spec["seed"], 1)
    to_t = lambda a: torch.from_numpy(im.gather(a))[:, None]
    runs = []
    for micro in (0, 3):
        tr = Trainer(spec, im, torch.device("cpu"), micro_batch=micro)
        r = tr.step([to_t(c) for c in cb[0]], to_t(vb[0]), torch.Generator().manual_seed(0))
        # .grad holds the accumulated gradient of the last critic step and of the VAE step
        grads = {n: [p.grad.clone() for p in m.parameters()] for n, m in (("vae", tr.vae), ("critic", tr.critic))}
        runs.append((grads, r))
    (ga, ra), (gb, rb) = runs
    for net in ("vae", "critic"):
        for x, y in zip(ga[net], gb[net]):
            # float32 summation order only (a bias gradient sums ~5e5 pixel terms of mixed sign)
            assert float((x - y).abs().max()) <= 1e-3 * float(x.abs().max()) + 1e-7, net
    for k in ("loss_vae", "rec", "kl", "adv", "loss_d", "gp"):
        assert float(ra[k]) == pytest.approx(float(rb[k]), rel=1e-5, abs=1e-6), k


def test_run_spec_records_loss_and_precision(vcfg):
    spec = run_spec(vcfg, "x")
    assert spec["recon"] == "bce_sum" and spec["tf32"] is False
