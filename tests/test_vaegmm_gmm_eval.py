"""VAE-GMM: EM vs scikit-learn, MAP / sampling, whistle features, conditional-generation equations, metrics, LDM, CLI.
CPU only, small synthetic data."""
import itertools
import json

import numpy as np
import pytest
import torch
from scipy.stats import wasserstein_distance

from whistlepipe.vaegmm.conditional import acceptance_ratios, conditional_sample, weights_eq8, weights_eq10
from whistlepipe.vaegmm.features import FEATURES, contour_features, satisfies
from whistlepipe.vaegmm.fid import imresize_matrix, preprocess
from whistlepipe.vaegmm.gmm import GaussianMixtureEM, kmeanspp_seeds
from whistlepipe.vaegmm.ldm import LinearSchedule, UNet, ldm_loss, ldm_sample
from whistlepipe.vaegmm.metrics import (frechet_distance, frechet_from_samples, precision_recall, psnr_from_mse,
                                        silhouette_exact, ssim, wasserstein1_sorted)


def blobs(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0, 0], [6, 0, 1, 0], [0, 7, -2, 3]], float)
    A = [np.diag([1, .5, .3, .2]), np.array([[1, .6, 0, 0], [0, .4, 0, 0], [0, 0, .5, .1], [0, 0, 0, .3]]), np.eye(4) * .7]
    lab = rng.integers(0, 3, n)
    X = np.stack([centres[l] + A[l] @ rng.standard_normal(4) for l in lab]).astype(np.float32)
    return X, lab


# ---------------------------------------------------------------- GMM
def test_em_matches_sklearn_from_identical_initialisation():
    from sklearn.mixture import GaussianMixture

    X, _ = blobs()
    ours = GaussianMixtureEM(3, reg_rel=1e-6, max_iter=1, tol=0.0, seed=1, device="cpu", verbose=False)
    ours.fit(X)                                                    # 1 iteration from k-means++ init, just to get an init
    init = {"weights": ours.weights_.numpy(), "means": ours.means_.numpy(), "covariances": ours.covariances_.numpy()}
    a = GaussianMixtureEM(3, reg_rel=1e-6, max_iter=20, tol=0.0, device="cpu", verbose=False).fit(X, init_params=init)
    reg_abs = a.reg_
    sk = GaussianMixture(3, covariance_type="full", reg_covar=reg_abs, max_iter=20, tol=0.0, weights_init=init["weights"],
                         means_init=init["means"], precisions_init=np.linalg.inv(init["covariances"]), init_params="random_from_data")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk.fit(X.astype(np.float64))
    assert np.allclose(a.weights_.numpy(), sk.weights_, atol=1e-6)
    assert np.allclose(a.means_.numpy(), sk.means_, atol=1e-4)
    assert np.allclose(a.covariances_.numpy(), sk.covariances_, atol=1e-4)
    assert a.final_mean_loglik_ == pytest.approx(sk.score(X.astype(np.float64)), abs=1e-4)


def test_em_loglik_is_monotone_and_converges_with_tolerance():
    X, _ = blobs(seed=2)
    g = GaussianMixtureEM(3, max_iter=500, tol=1e-8, seed=0, device="cpu", verbose=False).fit(X)
    ll = [h["mean_loglik"] for h in g.history]
    assert all(b >= a - 1e-6 for a, b in zip(ll, ll[1:]))
    assert g.converged_ and g.n_iter_ < 500
    assert np.isclose(g.weights_.sum().item(), 1.0)


def test_map_prediction_recovers_clusters_and_sampling_follows_components():
    X, lab = blobs(seed=3)
    g = GaussianMixtureEM(3, seed=0, device="cpu", verbose=False, max_iter=200).fit(X)
    pred, resp = g.predict(X)
    lj = g.log_joint(X)
    assert np.array_equal(pred, lj.argmax(1)) and bool((resp > 0).all() and (resp <= 1 + 1e-9).all())
    # cluster agreement up to label permutation
    agree = max(np.mean(pred == np.array(q)[lab]) for q in itertools.permutations(range(3)))
    assert agree > 0.97
    gen = torch.Generator().manual_seed(0)
    z, comp = g.sample(30000, gen, components=np.full(30000, 1))
    assert torch.allclose(z.double().mean(0), g.means_[1], atol=0.05)
    assert torch.allclose(torch.cov(z.double().T), g.covariances_[1], atol=0.05)
    z2, comp2 = g.sample(20000, gen)
    frac = np.bincount(comp2.numpy(), minlength=3) / 20000
    assert np.allclose(frac, g.weights_.numpy(), atol=0.02)


def test_ridge_keeps_degenerate_components_positive_definite_and_save_load(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8)).astype(np.float32)        # 40 points, 10 components: < D+1 points each
    g = GaussianMixtureEM(10, reg_rel=1e-6, max_iter=30, tol=0.0, seed=0, device="cpu", verbose=False).fit(X)
    ev = torch.linalg.eigvalsh(g.covariances_)
    assert float(ev.min()) > 0
    d = g.diagnostics()
    assert d["n_components_nk_lt_9"] >= 1 and d["frac_eigs_at_ridge"] > 0
    g.save(tmp_path / "gmm.npz", {"note": "test"})
    h = GaussianMixtureEM.load(tmp_path / "gmm.npz", "cpu", verbose=False)
    assert np.allclose(h.score_samples(X), g.score_samples(X))
    assert json.loads((tmp_path / "gmm.json").read_text())["note"] == "test"


def test_kmeanspp_seeds_are_distinct_and_deterministic():
    X, _ = blobs(500)
    t = torch.from_numpy(X)
    s1, s2 = kmeanspp_seeds(t, 20, 5), kmeanspp_seeds(t, 20, 5)
    assert torch.equal(s1, s2) and len(set(s1.tolist())) == 20


# ---------------------------------------------------------------- whistle features (Fig. 2, Table 3)
def draw(rows_of_cols: dict, width=1, value=1.0):
    img = torch.zeros(1, 128, 128)
    for c, r in rows_of_cols.items():
        img[0, r - width:r + width + 1, c] = value
    return img


def test_features_of_a_known_contour():
    # columns 40..79 (L = 40); row goes 70 -> 51 then stays; bin = 128 - row
    rows = {c: 70 - min(c - 40, 19) for c in range(40, 80)}
    f, ok = contour_features(draw(rows))
    f = dict(zip(FEATURES, f[0].tolist()))
    assert bool(ok[0]) and f["L"] == 40
    assert f["f_start"] == 128 - 70 and f["f_end"] == 128 - 51
    assert f["f_min"] == 58 and f["f_max"] == 77 and f["df"] == 19 and f["f_c"] == 67.5


def test_threshold_005_and_empty_images():
    rows = {c: 64 for c in range(10, 20)}
    faint = draw(rows, value=0.04)
    f, ok = contour_features(faint, 0.05)
    assert not bool(ok[0]) and torch.isnan(f).all()
    f, ok = contour_features(faint, 0.03)
    assert bool(ok[0]) and f[0, 0] == 10 and f[0, 1] == 64      # row 64 -> bin 64; 0 Hz (row 63) -> bin 65


def test_condition_check_is_inclusive_and_nan_safe():
    feats = np.array([[30, 0, 0, 0, 0, 0, 0], [40, 0, 0, 0, 0, 0, 0], [41, 0, 0, 0, 0, 0, 0], [np.nan] * 7], float)
    assert satisfies(feats, "L", 30, 40).tolist() == [True, True, False, False]


# ---------------------------------------------------------------- conditional generation (eqs. 7-10)
def test_conditional_weights_follow_eq8_and_eq10():
    pi = np.array([0.5, 0.3, 0.2])
    a = np.array([0.0, 0.5, 1.0])
    member = a > 0
    w8, w10 = weights_eq8(pi, member), weights_eq10(pi, a, member)
    assert np.allclose(w8, [0, 0.6, 0.4]) and np.allclose(w10, [0, 0.15 / 0.35, 0.2 / 0.35])


def test_acceptance_ratio_and_rejection_sampler_only_return_accepted():
    cf = np.zeros((2, 4, 7))
    cf[0, :, 0] = [30, 35, 50, 60]
    cf[1, :, 0] = [31, 32, 33, 34]
    assert np.allclose(acceptance_ratios(cf, "L", 30, 40), [0.5, 1.0])

    class Dec:                       # decoder whose image length is |z0| columns
        def __call__(self, z):
            n = z.shape[0]
            img = torch.zeros(n, 1, 128, 128)
            for i in range(n):
                L = int(min(120, max(1, round(abs(float(z[i, 0]))))))
                img[i, 0, 60, 4:4 + L] = 1.0
            return img

    X, _ = blobs(200)
    g = GaussianMixtureEM(2, device="cpu", verbose=False, max_iter=5).fit(np.c_[X[:, :1] * 10 + 40, X[:, 1:2]])
    s = conditional_sample(Dec(), g, g.weights_.numpy(), "L", 30, 40, 50, torch.Generator().manual_seed(0), batch=64, keep_images=5)
    assert s["features"].shape == (50, 7) and satisfies(s["features"], "L", 30, 40).all()
    assert 0 < s["acceptance_rate"] <= 1 and s["images"].shape == (5, 128, 128)


# ---------------------------------------------------------------- metrics
def test_psnr_convention_reproduces_table1():
    assert psnr_from_mse(0.5778) == pytest.approx(44.5264, abs=1e-3)      # paper 44.5274 (rounding of MSE)
    assert psnr_from_mse(0.3861) == pytest.approx(46.2772, abs=1e-3)


def test_ssim_identity_and_ordering():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 32, 32)
    assert torch.allclose(ssim(x, x), torch.ones(2), atol=1e-6)
    assert float(ssim(x, x + 0.05 * torch.randn_like(x)).mean()) > float(ssim(x, torch.rand_like(x)).mean())


def test_wasserstein_eq13_matches_scipy_for_equal_sizes():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 500), rng.normal(0.3, 1.2, 500)
    assert wasserstein1_sorted(a, b) == pytest.approx(wasserstein_distance(a, b), rel=1e-12)


def test_frechet_distance_analytic():
    assert frechet_distance([0.0], [[1.0]], [0.0], [[1.0]]) == pytest.approx(0.0, abs=1e-10)
    assert frechet_distance([1.0], [[4.0]], [3.0], [[1.0]]) == pytest.approx(4.0 + 1.0, rel=1e-9)   # (dmu)^2 + (s1 - s2)^2
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4000, 3))
    assert frechet_from_samples(a, a) == pytest.approx(0.0, abs=1e-8)


def test_precision_recall_limits():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((300, 5))
    pr = precision_recall(a, a.copy(), k=3)
    assert pr["precision"] == 1.0 and pr["recall"] == 1.0 and pr["f1"] == 1.0
    far = precision_recall(a, a + 100.0, k=3)
    assert far["precision"] == 0.0 and far["recall"] == 0.0
    half = precision_recall(a, np.r_[a[:150], a[:150] + 100.0], k=3)
    assert half["precision"] == pytest.approx(0.5, abs=0.02)


def test_silhouette_matches_sklearn():
    from sklearn.metrics import silhouette_score

    X, lab = blobs(600, seed=4)
    lab = lab.copy()
    lab[0] = 7                                                    # a singleton cluster (s = 0 convention)
    assert silhouette_exact(X, lab) == pytest.approx(silhouette_score(X, lab), abs=1e-5)


# ---------------------------------------------------------------- LDM (Table 9) and FID preprocessing
def test_ldm_unet_shapes_loss_and_sampling():
    torch.manual_seed(0)
    m = UNet()
    sched = LinearSchedule(10, 1e-4, 0.02, "cpu")
    z0 = torch.randn(4, 1, 8, 8)
    assert m(z0, torch.tensor([0, 3, 5, 9])).shape == (4, 1, 8, 8)
    loss = ldm_loss(m, sched, z0, torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
    assert ldm_sample(m, sched, 3, torch.Generator().manual_seed(1), "cpu").shape == (3, 1, 8, 8)
    assert float(sched.beta[0]) == pytest.approx(1e-4) and float(sched.beta[-1]) == pytest.approx(0.02)


def test_fid_preprocessing_binarisation_resize_and_channels():
    x = torch.zeros(2, 1, 128, 128)
    x[:, :, 60:62, 30:90] = 0.08
    x[:, :, 64:66, 30:90] = 0.5
    p_raw = preprocess(x, None)
    p_bin = preprocess(x, 0.1)
    assert p_raw.shape == (2, 3, 224, 224)
    assert torch.allclose(p_raw[:, 0] * 0.229 + 0.485, p_raw[:, 2] * 0.225 + 0.406, atol=1e-6)     # channel replicated
    raw01 = p_raw[:, 1] * 0.224 + 0.456
    bin01 = p_bin[:, 1] * 0.224 + 0.456
    assert float(raw01.max()) < 0.7 and float(bin01.max()) > 0.95          # bicubic overshoot (no clamping, as MATLAB)
    assert float(bin01[:, :, 0:40].abs().max()) < 1e-6                      # empty region stays empty


def test_imresize_matrix_matlab_convention():
    m = imresize_matrix(128, 224)
    assert m.shape == (224, 128) and torch.allclose(m.sum(1), torch.ones(224, dtype=torch.float64))
    ramp = torch.arange(128, dtype=torch.float64)
    up = m @ ramp
    # a linear ramp is reproduced exactly by the cubic kernel in the interior: u = x/s + (1 - 1/s)/2 (1-based)
    x = torch.arange(1, 225, dtype=torch.float64)
    u = x * 128 / 224 + 0.5 * (1 - 128 / 224) - 1
    assert torch.allclose(up[4:-4], u[4:-4], atol=1e-9)
    assert torch.allclose(imresize_matrix(16, 16), torch.eye(16, dtype=torch.float64), atol=1e-12)


def test_cli_registers_vaegmm_commands(capsys):
    from whistlepipe.cli import main

    assert main(["vae-experiments"]) == 0
    out = capsys.readouterr().out
    assert "V1_L64_b1e-4_critic" in out and "V8_L64_b1e-4_nocritic" in out
    assert main(["fetch-eval-weights"]) == 1                       # refuses without --approved


# ---------------------------------------------------------------- GPU paths (same results as CPU)
@pytest.mark.gpu
def test_gmm_and_metrics_gpu_match_cpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    X, lab = blobs(2000, seed=6)
    kw = dict(reg_rel=1e-6, max_iter=15, tol=0.0, seed=0, verbose=False, point_chunk=256, component_chunk=2)
    c = GaussianMixtureEM(3, device="cpu", **kw).fit(X)
    g = GaussianMixtureEM(3, device="cuda", **kw).fit(X)
    assert np.allclose(c.means_.numpy(), g.means_.cpu().numpy(), atol=1e-4)
    assert np.allclose(c.covariances_.numpy(), g.covariances_.cpu().numpy(), atol=1e-4)
    assert np.array_equal(c.predict(X)[0], g.predict(X)[0])
    a = np.random.default_rng(0).standard_normal((700, 7))
    b = a + 0.3 * np.random.default_rng(1).standard_normal((700, 7))
    assert precision_recall(a, b, 3, "cpu") == precision_recall(a, b, 3, "cuda")
    assert silhouette_exact(X, lab, "cuda") == pytest.approx(silhouette_exact(X, lab, "cpu"), abs=1e-5)
    img = torch.zeros(4, 1, 128, 128)
    img[:, 0, 50:53, 20:70] = 1.0
    assert torch.equal(contour_features(img)[0], contour_features(img.cuda())[0].cpu())
