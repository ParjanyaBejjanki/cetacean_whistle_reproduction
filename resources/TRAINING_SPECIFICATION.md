# TRAINING AND EVALUATION SPECIFICATION — VAE-GMM

**Project:** reproduction of *"A Generative Model for Cetacean Whistles Using Variational Autoencoder and Mixture of Gaussians"* (Park et al., IEEE Access 14, 2026, pp. 27877–27894).
**Created:** 2026-10-22. **Scope:** everything after preprocessing. The dataset is fixed and is **not** re-processed.
**Companion files:** `configs/vaegmm.toml` (the `[paper]` section is locked by `src/whistlepipe/vaegmm/config.py`) and `src/whistlepipe/vaegmm/`.

## 0. Conventions

| Tag | Meaning |
|---|---|
| **[EXPLICIT]** | Stated by the paper. Location is given as section / equation / table / appendix and journal page. |
| **[IMPLIED]** | Derivable from stated facts. The derivation is shown. |
| **[DECISION]** | The paper is silent. Chosen here and recorded with its reason. **Not paper evidence.** |
| **[DEVIATION]** | Differs from a value the paper states. Every result must be reported with it. |
| **[CONFLICT]** | The paper's own statements are mutually inconsistent. The conflict is documented and resolved *against the paper's text*. |

Other sources are used only where the paper explicitly defers to them:
- [49] Gulrajani et al., WGAN-GP: Algorithm 1.
- [47] Kynkäänniemi et al.: improved precision/recall.
- [35] Bishop, PRML: EM.

The paper's numbers are **reference values, never tuning targets**.

## 1. Discrepancy report

This audit covers the plan of 2026-10-21 and the brief of 2026-10-22, checked against eqs. 3–13, Tables 1–9 and Appendices A–D.

| # | Item | Earlier plan / brief | What the paper says | Correction |
|---|---|---|---|---|
| 1 | Adversarial weight | called γ | eq. 4: L = L_rec + βL_KL + **λ**L_adv; App. A: "WGAN-GP weight λ was set to 1"; gradient penalty λ_GP = 5 (eq. 5, App. A) | Named `lambda_adv` = 1, separate from `lambda_gp` = 5 |
| 2 | Critic output / loss family | not analysed | Table 8 output layer **sigmoid**; eq. 5 "standard WGAN-GP loss" on D(·) = critic output; eq. 4 L_adv = −E log D(x̂) | **[CONFLICT]**, resolved literally (§3.2). No substitution of a standard GAN or WGAN-GP form |
| 3 | Learning rates | plan kept 1e-3 / 2e-4 at batch 128; brief asks for linear scaling | 1e-3 / 2e-4 at aggregate batch 896 (App. A) | Linear-scaled **[DEVIATION]**: 1.4286e-4 / 2.8571e-5, after verifying aggregation and reduction (§3.3) |
| 4 | "Standard VAE" (Table 4 row 1) | plan: β = 1e-4, no critic | §IV-D-1 defines "standard VAEs where β = 1" | Primary: β = 1, no critic, z ~ N(0, I) **[IMPLIED, weak]**. Supplementary at zero training cost: row-3 VAE (β = 1e-4) sampled from N(0, I) |
| 5 | EM "probability tolerance 1e-8" | plan also used it as a MATLAB posterior cut-off | App. B: "probability tolerance for early stopping was set to 10⁻⁸"; nothing else | Posterior cut-off **removed** (not stated). The stopping rule is the relative change of the mean log-likelihood ≤ 1e-8, the convergence check of [35] §9.2.2 **[DECISION]** |
| 6 | Precision/Recall feature network | plan: Inception features | "quantified following the method proposed in [47]"; [47] uses VGG-16 fc2 features, k = 3 | VGG-16 fc2, k = 3. Weights download pending approval |
| 7 | Number of VAE trainings | plan: 7; brief: 6 for Tables 1–2 | Table 1: L ∈ {32, 64, 128, 256}, β = 1e-4; Table 2: L = 64, β ∈ {1e-2, 1e-4, 1e-6} | Brief's **6** confirmed. Ablations add **2** (item 4 and row 3). Baselines are counted separately (§3.8) |
| 8 | Fig. 11 grid | — | K ∈ {1k, 3k, 5k, 7k, 10k, 12k, 15k}; L and β not stated | L = 64, β = 1e-4 **[IMPLIED]**: §IV-D-1 "primarily used the VAE model with latent dimension 64 … for performance evaluation" |
| 9 | Thresholds | — | 0.05 to obtain the contour (Fig. 2); 0.1 binarisation of real **and** generated images for FID₁; FID₂ unthresholded (§IV-D-2) | Three separate settings. P/R input is not stated: both variants computed, unthresholded is primary **[DECISION]** |
| 10 | Conditional weights | plan: eq. 10 only | Two weightings: π̃ (eq. 8), used in the three-step sampler with Cat(π̃); π̌ ∝ a_kπ_k (eqs. 9–10) with rejection sampling | Both implemented. Table 3 primary = eqs. 9–10 + rejection **[DECISION]**; eq. 8 variant reported too |
| 11 | Table 3 vs Fig. 12 | — | Fig. 12 diagonal = ln(Table 3 FFD) for all 13 conditions (max abs. difference 0.02; natural log). Text example Z(4,5): conditions 4 (34 ≤ f_min ≤ 55) and 5 (40 ≤ f_min ≤ 53) match Table 3 | Consistent. Two notes: (a) the text maps real→x-axis, but the figure has real data on the **y-axis (rows)**, so we follow the figure; (b) the Fig. 9 caption says 74–80 samples but Table 3 row 2 says 75–80, so Table 3 is used |
| 12 | Table 1 "MSE" | — | PSNR = 10·log10(16384 / MSE) holds within 0.0014 dB for all four rows | MSE = per-image **sum** of squared errors; PSNR peak = 1 **[IMPLIED]** |
| 13 | Architecture check | — | Table 6: VAE-GMM generator 368k params, 557M FLOPs | Tables 7–8 with "same" padding give exactly 368,097 params and 557.3 MFLOPs (2 × MAC, transposed conv counted at output size). This is a unit test |
| 14 | Table 9 (LDM) | — | ConvBlock 3 "3/2/256" yet output stays 2×2; ConvBlock 5 "3/1/128" yet output 8×8×64; listed layers give 1.96M conv params / 87.4 MFLOPs vs Table 6's 2.9M / 95.02M per step | **[CONFLICT]**, documented; resolved when the baselines are built (§3.8). About 0.9M params sit in layers Table 9 does not list (time embedding, GroupNorm) |
| 15 | F1 consistency | — | F1 = 2PR/(P+R) reproduces every printed F1 in Tables 2–5; WGAN-GP row: 70.69 computed vs 70.70 printed (rounding) | No action |
| 16 | GMM data per component | — | 228,673 / 10,000 = 22.9 images per component in the paper; ours 180,154 / 10,000 = 18.0. A full 64-D sample covariance needs ≥ 65 points | Regularisation is **required** but unstated **[DECISION]** (§3.5) |
| 17 | Table 4 row 2 = Table 5 row "WGAN-GP" | — | Identical numbers (13.82 / 202.51 / 61.47 / 83.17 / 70.70) | Same model, trained once |
| 18 | Reconstruction loss (added 2026-10-22, after gate G4 failed) | squared error per image | −log p(x\|z), form not stated | Squared error on the sigmoid output saturated on the sparse images: whistle-pixel logits ≈ −40, sigmoid′ ≈ 1e-8, blank outputs, at both learning rates tried. **Bernoulli NLL (binary cross-entropy from logits)** does not saturate and learned in a diagnostic run [DECISION, revised] |
| 19 | Gate G3 criterion (added 2026-10-22) | bit-identical resume | — | Two identical GPU runs with the critic already differ by ≈ 1e-6 after one step (kernel-level float effects; runs without the critic are bit-identical). Criterion revised after this observation: exact restore of the checkpoint, and a resumed-vs-uninterrupted difference ≤ 10 × the run-to-run difference |

## 2. Fixed input

`preprocessed/encounter_<id>/images.npy`: 8 encounters, **180,154 images** (N, 128, 128), float32 in [0, 1], every image max = 1.
- Row 0 = highest frequency, 0 Hz on row 63; 93.75 Hz per row, 5.333 ms per column.
- Built with processing config `78f1bb1b…` and detector `ridgenet-v2c` (`c00d66a0…`).

Files are opened read-only (memory-mapped). There is no filtering, and the paper describes none.
- The 37 % of images shorter than 30 frames stay in.
- So do the 56 detector-background files.

There is no train/test split: the paper evaluates against "the training data" (Abstract; §IV). Every run records:
- the `dataset_info.json` SHA-256 and the processing/detector hashes;
- the SHA-256 of each `images.npy`;
- the git commit, plus SHA-256 of every source file used.

## 3. Corrected specification

### 3.1 Architecture [EXPLICIT Tables 7–8, App. A-A]

| Net | Layers |
|---|---|
| Encoder | Conv 3×3 s2 32 + swish → Conv 3×3 s2 32 + swish → Conv 3×3 s2 64 + swish → Conv 3×3 s2 64 + swish (8×8×64) → FC 4096 → 2L → sampling → z ∈ ℝᴸ |
| Decoder | FC L → 4096, reshape 8×8×64 → TConv 3×3 s2 64 + swish → TConv 3×3 s2 64 + swish → TConv 3×3 s2 32 + swish → TConv 3×3 s2 32 + swish (128×128×32) → Conv 3×3 s1 1 + sigmoid |
| Critic | Conv 5×5 s2 32 + LeakyReLU 0.2 → [Conv 5×5 s2 (32, 64, 64) + LayerNorm + LeakyReLU 0.2] × 3 (8×8×64) → Conv 8×8 s1 1 + sigmoid → scalar |

[IMPLIED]
- "Same" padding, from the output sizes.
- No activation after the encoder FC or the decoder projection (none listed).
- The parameter/FLOP check in item 13.

[DECISION]
- FC output split: the first L units are μ, the last L are log σ².
- PyTorch transposed conv: k = 3, s = 2, p = 1, output_padding = 1. This gives the exact 2× size; MATLAB's crop placement may differ by one pixel.
- The output layer is a 3×3 conv, which is equivalent to a stride-1 transposed conv.
- LayerNorm normalises over C×H×W per image with per-channel scale and offset (`GroupNorm(1, C)`).
- Weights are Glorot-uniform and biases zero. This is the default of the paper's framework, MATLAB R2024a (App. A-B).

### 3.2 Objective

**[EXPLICIT] eqs. 3–5, p. 27882**
- **VAE (eq. 4):** L = L_rec + β·L_KL + λ·L_adv, with L_KL = KL(q(z|x) ‖ N(0, I)) and L_adv = −E_{z~q(z|x)}[log D(p(x|z))].
- **Critic (eq. 5):** L_D = E_{x̂~P_VAE}[D(x̂)] − E_{x~P_r}[D(x)] + λ_GP·E_x̃[(‖∇_x̃ D(x̃)‖₂ − 1)²], with x̃ = εx + (1−ε)x̂ and ε ~ U(0, 1).

**[CONFLICT] Sigmoid critic vs WGAN-GP vs −log D**
- WGAN-GP [49] assumes an **unbounded** critic. Then E D(x) − E D(x̂) estimates the Wasserstein-1 distance and the gradient penalty enforces the Lipschitz constraint.
- The paper's critic ends in a **sigmoid** (Table 8). So D ∈ (0, 1), the critic gap is bounded in (−1, 1), and it is not a W₁ estimate. The penalty on the post-sigmoid output can only be met where ‖∇a‖ ≥ 4, since σ′ ≤ ¼ (a = pre-sigmoid logit).
- The generator term −log D is the non-saturating GAN form; the paper motivates it with [34]. It **requires** D ∈ (0, 1), so it is consistent with the sigmoid.
- The paper therefore deliberately combines (i) a WGAN-GP critic objective, (ii) a sigmoid-bounded critic and (iii) a log-based generator loss.

**Resolution (literal, no substitution)**
- D(·) is the sigmoid output, as eq. 5 says ("D(·) denotes the output of the critic network").
- L_D is computed on post-sigmoid values, and the gradient penalty uses ∇_x̃ of the post-sigmoid output.
- L_adv = −mean log D(x̂).

Not used: an unbounded critic, the −E D(x̂) generator loss, and BCE discriminator losses.

The only numerical safeguard is computing log D as `logsigmoid(a)`, an identical function. It avoids log 0 when σ(a) underflows in fp32 (a < −88).

**Losses [IMPLIED]**
- The expectations in eqs. 3–5 are batch means of per-image terms.
- L_rec = −log p(x|z) of a whole image is a **sum over pixels**; L_KL is the analytic KL **summed over the L dimensions**.

**[DECISION]**
- L_rec = per-image sum of binary cross-entropy, computed from the decoder logits (Bernoulli likelihood). Revised from squared error after gate G4 (item 18); `decisions.loss.recon` keeps both selectable. Table 1's MSE remains the per-image squared error (item 12).
- One reparameterised z per image.
- In eqs. 4 and 5, x̂ = decoder output for z ~ q(z|x) of the **same** real batch, i.e. the "VAE outputs" (Fig. 4: reconstruction branch into the critic).

### 3.3 Optimisation

| Item | Paper | Used here |
|---|---|---|
| Optimiser | Adam, both nets [EXPLICIT] | Adam |
| VAE learning rate | 1e-3 at aggregate batch 896 [EXPLICIT] | **1e-3 × B/896 [DEVIATION unless B = 896]**: 3e-3 at B = 2688 (Colab A100 40 GB), 6e-3 at B = 5376 (80 GB), 1.4286e-4 at B = 128 (laptop gates) |
| Critic learning rate | 2e-4 at aggregate batch 896 [EXPLICIT] | **2e-4 × B/896**: 6e-4 at B = 2688, 1.2e-3 at B = 5376, 2.8571e-5 at B = 128 |
| Critic Adam β₁, β₂ | 0, 0.9 [EXPLICIT] | 0, 0.9 |
| VAE Adam β₁, β₂, ε | not stated | 0.9, 0.999, 1e-8 [DECISION] (Adam defaults) |
| LR schedule | not stated | constant [DECISION] |
| Batch | 128/GPU × 7 = 896 [EXPLICIT] | **One GPU. Main run on a Colab A100: B = 2688 (40 GB) or 5376 (80 GB) [DEVIATION]; fallback B = 896 with the paper's learning rates (no deviation). Laptop gates: B = 128.** Set per run by override (`colab/train_V1_colab.ipynb`), recorded in `train_info.json`. Optional gradient accumulation (`train.micro_batch`) gives the same update with less memory |
| Critic steps | 3 per VAE iteration [EXPLICIT] | 3; critic steps precede each VAE step, each on a **fresh** real batch ([49] Alg. 1) [DECISION] |
| Epochs | 300 [EXPLICIT] | 300 = 1,408 VAE steps/epoch (last batch 58 images) = 422,400 VAE + 1,267,200 critic updates |
| Model kept | not stated | final epoch; no early stopping or model selection [DECISION] |
| Numerics | not stated | fp32, TF32 off, deterministic algorithms, fixed seeds (repo policy) [DECISION] |

**Verification behind the learning-rate deviation** (unit tests in `tests/test_vaegmm_*.py`):
1. *Loss reduction.* Every loss is a batch mean of per-image terms. So the accumulated gradient of 7 micro-batches of 128 equals the gradient of one 896 batch, and our 128-batch gradient is an unbiased estimate of the paper's gradient with ≈ 7× the variance.
2. *Aggregation across the 7 GPUs* (mean or sum; App. A does not say). Adam's update −lr·m̂/(√v̂ + ε) is invariant to a constant gradient scale (up to ε). So mean versus sum does not change the paper's effective step, and the scaling question is only about noise and number of updates.
3. *What the linear rule preserves:* learning rate × updates per epoch. On our data that is 1e-3 × 202 = 0.202 and 1.4286e-4 × 1,408 = 0.201.

The linear rule is an SGD heuristic. The documented alternative for adaptive methods, the square-root rule (Malladi et al., NeurIPS 2022), gives 3.78e-4 / 7.56e-5 but would also change Adam's β₁ and β₂. It is **not** used.

The linear values are **experimental starting values**. They are kept unless the validation gates show instability: non-finite losses, no decrease, or critic saturation. There is **no learning-rate search**, and nothing is tuned against published scores.

### 3.4 Data and augmentation

- **[EXPLICIT] App. A-B:** each image is randomly rotated by 0°, 90°, 180° or 270° with equal probability.
- **[DECISION]**
  - Rotation is drawn independently per image and per use.
  - The rotated image is the VAE input, the reconstruction target and the critic's real sample.
  - An epoch is one pass of a seeded random permutation for the VAE updates; the critic batches come from an independent seeded stream.

### 3.5 GMM [EXPLICIT §II-E eq. 6, App. B]

**Stated by the paper**
- Mixing weights π_k, mean vectors μ_k and covariance **matrices** Σ_k (full).
- Standard EM.
- Input: the encoder's **latent mean** outputs.
- At most 1,500 iterations; tolerance 1e-8.

**[DECISION]**

| Item | Choice |
|---|---|
| Input | μ(x) of all 180,154 images, **unrotated** (rotation is a training-time augmentation; Fig. 7 samples are unrotated), final VAE checkpoint |
| Initialisation | k-means++ seeding on the latent means (seeded) → hard assignment to the nearest seed → one M-step |
| Stopping | \|ℓ_t − ℓ_{t−1}\| ≤ 1e-8·\|ℓ_t\|, ℓ = mean log-likelihood per point accumulated in fp64 ([35] §9.2.2), or 1,500 iterations. Because 1e-8 is below fp32 per-point resolution, runs are expected to approach 1,500 iterations; the outcome is recorded |
| Covariance regularisation | Σ_k ← Σ_k + r·I at every M-step, r = 1e-6 × mean per-dimension variance of the latent means (scale-relative; needed by item 16). Sensitivity at ×0.1 / ×10 is reported as a robustness check, not used for selection |
| Degenerate components | A component with effective count N_k < 1 keeps its previous μ_k and Σ_k; π_k = N_k/N. Never re-seeded; K stays fixed. Counted every iteration |
| Computation | GPU, chunked over points × components. Explicit differences (x − μ_k), Mahalanobis distance via L_k⁻¹ (Cholesky in fp64, applied in fp32), log-sum-exp per point. M-step moments are shifted by the previous μ_k (single pass, no cancellation) and accumulated in fp64. Validated against scikit-learn `GaussianMixture` (float64) from identical initialisation |

**Reported diagnostics:** the log-likelihood curve; the N_k histogram; the number of components with N_k < 1 and N_k < 65; covariance eigenvalue ranges.

### 3.6 Generation and inference [EXPLICIT §III, eqs. 7–12]

- **Unconditional:** C ~ Cat(π), z ~ N(μ_C, Σ_C), x̂ = decoder output (sigmoid mean) [DECISION: no pixel noise].
- **Per-component analysis (§IV-B-1, Fig. 8):** 500 samples per component → 7 features.
- **Acceptance ratio:** a_k = the fraction of those 500 that meet the condition.
- **Conditional set:** 𝒦 = {k : a_k > 0} [DECISION].
- **eq. 8:** π̃_k = π_k / Σ_{𝒦} π_k.
- **eq. 10:** π̌_k = a_kπ_k / Σ_{𝒦} a_kπ_k.
- **Sampler:** C ~ Cat(weights) over 𝒦, z ~ N(μ_C, Σ_C), decode, reject samples that fail the condition (§III-B, "rejection sampling [35]"), repeat until N are accepted.
- **MAP clustering:** p(C|z) by eq. 11 and k̂ = argmax by eq. 12.

**Whistle features** (§IV-D-4 step 3, "as illustrated in Fig. 2(c)"):
1. Binarise at **0.05**.
2. [DECISION] Per column with at least one active pixel, take the mean row index of the active pixels. Rows are converted to "bin" index b = 128 − row (1-based, low frequency at the bottom, as plotted; 0 Hz = bin 65, consistent with Fig. 8's centre-frequency histograms at 64–66). Columns become sample index c + 1.
3. Compute the features:
   - L = last − first active column + 1 [samples];
   - f_min and f_max over the vector; Δf = f_max − f_min;
   - f_start, f_end = the first and last vector value;
   - f_c = (f_min + f_max)/2, the same definition as preprocessing S8 [DECISION].

Unit conversion [IMPLIED, Table 3 errors]: 5.333 ms per sample, 93.75 Hz per bin. The extractor is validated on real images against their true contours in `images.csv`.

### 3.7 Evaluation

| Paper | Grid | Procedure |
|---|---|---|
| **Table 1** | L ∈ {32, 64, 128, 256}, β = 1e-4, with critic; no GMM | Reconstructions of the training images (unrotated, z = μ [DECISION]). MSE = mean per-image SSE; PSNR = 10·log10(16384/MSE) [IMPLIED]; SSIM with a Gaussian window σ = 1.5, K = (0.01, 0.03), range 1 [DECISION]; KL = mean per-image KL summed over dims |
| **Table 2** | L = 64: K ∈ {1k, 5k, 10k, 15k} at β = 1e-4; β ∈ {1e-2, 1e-4, 1e-6} at K = 10k | **FID₁:** real and generated binarised at 0.1. **FID₂:** unthresholded. Inception-V3 2048-d final-pool features; bicubic 128→224; channel ×3 [EXPLICIT]; torchvision ImageNet weights, ImageNet normalisation, bicubic with the MATLAB `imresize` kernel (Keys a = −0.5, symmetric boundary, no clamping; PyTorch's bicubic uses a = −0.75) [DECISION]. **P/R** per [47] (VGG-16 fc2, k = 3); F1 = 2PR/(P+R). Reference set = all training images; N_gen = N_real = 180,154, fixed seeds [DECISION] |
| **Fig. 11** | K ∈ {1k, 3k, 5k, 7k, 10k, 12k, 15k}, L = 64, β = 1e-4 | Silhouette (exact, Euclidean, latent means), Davies–Bouldin, Calinski–Harabasz on the MAP labels (eq. 12) of all training latents [DECISION: latent-mean space] |
| **Table 3** | L = 64, K = 10k, β = 1e-4 [EXPLICIT]; 13 conditions | X = all real images meeting the condition (N = \|X\|); Y = N conditional samples. W₁ by eq. 13 on the conditioned feature (converted to ms/Hz); FFD = Fréchet distance of the 7-D features; P/R/F1 on the 7-D features with k = 3, raw units [DECISION] |
| **Fig. 12** | same | 13×13 matrix Z(i, j) = ln FFD(real condition i, generated condition j); rows = real, as in the figure |
| **Table 4** | L = 64 | (1) standard VAE, β = 1, no critic, z ~ N(0, I) [+ supplementary row-3 VAE with N(0, I)]; (2) WGAN-GP baseline; (3) VAE-GMM without critic, β = 1e-4, K = 10k; (4) the main model. Metrics as Table 2 |
| **Table 5** | — | WhistleGAN (DCGAN [20], [48], retrained from scratch with label smoothing and label flipping); WGAN-GP [49]; LDM T ∈ {250, 500, 1000} (Table 9, App. D: batch 128, 300 epochs, lr 5e-4, linear β schedule 1e-4 → 0.02, U-Net on the 8×8×1 latent of the β = 1e-4 VAE). Generator architectures for WhistleGAN and WGAN-GP are **not given**: Table 6's 7.7M / 3.79G and 5.1M / 1.71G are the only constraints, so these two baselines will be [DECISION] designs labelled as such |
| **Table 6** | — | Parameters and FLOPs of the generating networks only (decoder for VAE-GMM), with the paper's convention (item 13). Latency: 250 trials, first 50 warm-up, 0.5 s settling; B = 1, 16, 32; GPU. Hardware differs, so latencies are not comparable |
| Figs. 7–10, 13–14 | L = 64, K = 10k | Sample grids, 20-contour overlays and 500-sample histograms per component, 100-sample conditional overlays, MAP overlays (100 samples from k̂) |

### 3.8 Experiment matrix (identical configurations are trained once and reused)

| Unique VAE training | Used by |
|---|---|
| V1 L64 β1e-4, critic (**main**) | Table 1 (L64), Table 2 (all K rows; β1e-4 row), Fig. 11, Table 3, Fig. 12, Table 4 (4), Tables 5–6, Figs. 7–10, LDM latents |
| V2–V4 L32 / L128 / L256 β1e-4, critic | Table 1 |
| V5–V6 L64 β1e-2 / β1e-6, critic | Table 2 |
| V7 L64 β1 (standard VAE), no critic | Table 4 (1) |
| V8 L64 β1e-4, no critic | Table 4 (3) (+ supplementary for row 1) |

| Group | Count |
|---|---|
| VAE trainings | 6 for Tables 1–2 + 2 ablation = **8** |
| GMM fits | V1 × 7 K values (the union of the Table 2 and Fig. 11 grids) + V5, V6, V8 at K = 10k = **10** |
| Baseline trainings (separate) | WGAN-GP (Table 4 (2) = Table 5) **1**, WhistleGAN **1**, LDM **3** (one per T: the β schedule is defined over T [DECISION]) |

**Order:**
1. Unit tests and gates.
2. V1 → its GMM fits → evaluation of V1.
3. V2–V8.
4. Baselines.

Sweeps never start before V1 has finished and been evaluated.

### 3.9 Runtime options (no effect on the method)

| Option | Effect |
|---|---|
| `train.micro_batch` | gradient accumulation over chunks (same update, less GPU memory) |
| `train.cudnn_benchmark` | cuDNN autotuning: same maths, faster kernels, not bit-deterministic |
| `train.tf32` | TF32 tensor cores (Ampere and newer): slightly reduced internal precision. Stored in the run spec; a [DEVIATION] when used |

### 3.10 Implementation status (2026-10-22)

| Part | Status |
|---|---|
| VAE, critic, losses, training loop, resume, lineage | implemented, unit-tested (CPU) |
| GMM (EM, MAP, sampling), whistle features, conditional sampling, all metrics, FID/P-R pipeline, Figs. 7–10, Table 6 timing | implemented, unit-tested. GMM and metrics are also tested on the GPU against the CPU. k-means++ seeding is device-independent (NumPy random numbers, float64 distances) |
| LDM baseline (Table 9) | implemented, unit-tested (shapes, loss, sampler) |
| WhistleGAN and WGAN-GP baselines | **not implemented**: the paper gives no generator architecture (only Table 6 sizes). Needs either ref. [20] for WhistleGAN or an owner-approved [DECISION] design |
| Feature-network weights | download code present (`fetch-eval-weights --approved`); **not downloaded** (approval pending) |
| Measured cost | EM at K = 10k on 180,154 × 64 latents: ≈ 17 s per iteration on the RTX 5060 Laptop (memory-bound), so ≤ ≈ 7 h per 1,500-iteration fit. VAE training time is measured by gate G5 |

## 4. Validation gates (all must pass before the main run)

| Gate | Pre-declared pass criterion |
|---|---|
| G0 unit tests | `pytest` green. Covers: architecture shapes and Table 6 counts; KL closed form vs Monte Carlo; logsigmoid ≡ log∘sigmoid; gradient penalty by finite differences (float64); accumulation equivalence; Adam scale invariance; the rotation group; loader read-only behaviour and coverage; EM vs scikit-learn; features on synthetic contours; W₁ vs scipy; Fréchet vs analytic Gaussians; P/R on identical/disjoint sets; equation-level conditional weights |
| G1 data | All 180,154 images finite, in [0, 1], max = 1; counts equal `dataset_info.json` |
| G2 numerics | GPU-vs-CPU forward parity of VAE and critic (max abs. diff ≤ 1e-5); finite losses and gradients on real batches |
| G3 smoke | 2 epochs on a 2,048-image subset. (a) A trainer restored from the epoch-1 checkpoint equals it bit-identically; (b) max \|resumed − uninterrupted\| ≤ 10 × max \|uninterrupted − repeated uninterrupted\| (bit-identical when the repeat is). Revised 2026-10-22 (item 19) |
| G4 overfit | Full objective, batch 128, the run's learning rates, 256 fixed images. Within 3,000 VAE steps the reconstruction SSE (z = μ, unrotated) drops to ≤ 0.3 × the SSE of the all-zero image; all losses finite |
| G4 outcome (2026-10-28) | Cross-entropy loss: ratio 0.923 → **0.416 at step 3,000** (criterion **not met** as measured; monotone, accelerating decrease, last drop 0.098 per 250 steps). **Accepted by the project owner** without a longer run, on the extrapolation that ≤ 0.3 is reached at about 3,300–3,500 steps (not measured). Records: `G4_20261028T054038Z.json` (measured), `G4_owner_acceptance_*.json` (decision) |
| G5 timing | 200 measured iterations on the full data after 20 warm-up: seconds/iteration, data-wait fraction, peak GPU memory, projected 300-epoch hours. Recorded, not thresholded |
| G6 features | On ≥ 2,000 real images: \|L − n_frames\| ≤ 2 for ≥ 95 % and \|Δf − bandwidth_hz/93.75\| ≤ 1.5 bins for ≥ 95 % |

## 5. Integrity rules

1. `[paper]` values are locked in code, and every deviation is listed in `[deviations]` and in each run's `train_info.json`.
2. Choices tagged [DECISION] are made before results exist and are not revised to improve agreement with the paper.
3. Nothing is reported as run unless its output files exist. Absolute FID values are not comparable with the paper because the data, the S5/S6 replacement, the feature-network weights and the framework all differ. Comparisons are of **trends**:
   - MSE ↓ and KL ↑ with L;
   - FID₁ ↓ with K;
   - β = 1e-4 best;
   - ablation order (1) < (3) < (4);
   - Silhouette ↑ with K;
   - Fig. 12 diagonal < off-diagonal.
4. The Inception-V3 and VGG-16 weights are downloaded only after the project owner approves.
