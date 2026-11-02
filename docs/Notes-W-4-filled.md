# Week 4 Project Logbook: Training Experiments & Evaluation — filled in

**Project:** reproduction of Park et al., "A Generative Model for Cetacean Whistles Using Variational Autoencoder and Mixture of Gaussians", *IEEE Access* 14 (2026). The paper is called **VAE-GMM** below. The data are false killer whale recordings from NOAA DCLDE 2022.
**Template:** `Notes-W-4.pdf`, transcribed in [Notes-W-4.md](Notes-W-4.md).
**Filled in:** 2026-11-01, from the files in this repository.
**Results table:** [Notes-W-4-results.csv](Notes-W-4-results.csv) holds the measured results and the paper's target values, one metric per row, each with its source file.

**Two models are covered:**

| Model | Role in the project | Status |
|---|---|---|
| **VAE-GMM** | The paper's model (main run `V1_L64_b1e-4_critic`) | Training was stopped at **epoch 6 of 300**. No test results exist yet. |
| **RidgeNet** | The replacement whistle detector, stage S5 | The tuning, ablation and test experiments are finished. Most of the measured numbers in this logbook come from here. |

**Status tags used throughout:**

| Tag | Meaning |
|---|---|
| ✅ done | Has a result file in the repo |
| ⏳ pending | Planned but not run yet |
| ✗ not run | Not part of the agreed scope |

**Rule followed:** nothing is reported as run unless its output file exists (`resources/TRAINING_SPECIFICATION.md` §5).

> **Scope change (2026-11-02):** after this logbook was filled in, training moved to encounter 191 only (363 files, 63,490 images), 150 epochs, K = 3,000. This logbook records Week 4 as it was; see `resources/TRAINING_SPECIFICATION.md` §2.1.

> **V1 run state is as of the local copy.** Google Drive (`G:`) is not mounted on this machine. The V1 numbers therefore come from `models/vaegmm/V1_L64_b1e-4_critic/`: the epoch-6 checkpoint, plus a log that ends at epoch 7, step 212, on 2026-10-29. If the run was resumed on Colab after that, newer logs exist on Drive.

---

## I. Hyperparameter Tuning & Dataset Experiments
*(To be completed during the initial tuning phase)*

### Hyperparameter Grid / Random Search Matrix

**No grid or random search was run, for either model. This was deliberate.**
- **VAE-GMM:** the specification fixes every hyperparameter to the paper's value ("There is no learning-rate search, and nothing is tuned against published scores", spec §3.3).
  - The only change is the batch size, because training runs on one GPU instead of the paper's seven.
  - The learning rates are scaled linearly with the batch size: lr × B / 896.
- **RidgeNet:** one training recipe is used for every version (v1, v2a–v2e and 5-24). The versions differ in data and architecture, not in optimiser settings.

#### *Learning Rates tested:*

**VAE-GMM**

| VAE lr / critic lr | Batch | Where it ran | Status |
|---|---|---|---|
| 1.4286e-4 / 2.8571e-5 | 128 | Laptop gates G3–G5 (RTX 5060 Laptop), 2026-10-22 and 2026-10-28 | ✅ ran |
| **6e-3 / 1.2e-3** | **5376** | **V1 main run, Colab A100-SXM4-80GB, started 2026-10-29** | ✅ running; stopped at epoch 6 |
| 1e-3 / 2e-4 (the paper's values) | 896 | Fallback run `V1_L64_b1e-4_critic_B896` | ✗ configured, not run |
| 3e-3 / 6e-4 | 2688 | Colab A100 40 GB setting | ✗ configured, not run |

- **Fixed for all VAE-GMM runs:** Adam with betas (0.9, 0.999) and eps 1e-8 for the VAE; betas (0, 0.9) for the critic (the paper's values); constant learning rate.
- **Check on the linear rule:** it keeps lr × (updates per epoch) the same as the paper's:
  - Paper: 1e-3 × 202 = 0.202
  - B = 128: 1.4286e-4 × 1,408 = 0.201
  - V1: 6e-3 × 34 = **0.204**

**RidgeNet:** lr **3e-3** with weight decay 1e-4 for every version. Not swept.

#### *Batch Sizes tested:*

- **VAE-GMM:**
  - **128:** gates only, 1,408 steps per epoch.
  - **5376:** V1, 34 steps per epoch, 10,200 VAE updates and 30,600 critic updates over 300 epochs.
  - **896 and 2688:** configured, not run.
  - Paper: 128 per GPU × 7 GPUs = 896.
- **RidgeNet:** **16**, for 6,000 steps, in every version.

#### *Loss weights* (the Section III checklist asks for these, but the template has no field for them)

**VAE-GMM**
- β = 1e-4, λ_adv = 1, λ_GP = 5, n_critic = 3, all the paper's values. None of them were varied.
- The one loss experiment was the **form of the reconstruction loss**, tested in gate G4 (256 images, batch 128, up to 3,000 steps). The pass mark is SSE ratio ≤ 0.3.

| Reconstruction loss | SSE ratio at step 250 | SSE ratio at step 3,000 | Outcome |
|---|---|---|---|
| Squared error (2026-10-22) | 1.0025 | **1.0001** | Collapsed to a blank output; failed |
| **Per-image binary cross-entropy (BCE) sum (2026-10-28)** | 0.923 | **0.416** | Missed the 0.3 mark, but **accepted by the project owner** on the trend. The 0.3 mark was extrapolated to be reached at about 3,300–3,500 steps; this was not measured. |

**RidgeNet:** pos_weight = 6.0 and dice_weight = 1.0. Not swept.

#### *Best Configuration Identified:*

These were not chosen by a search. They are the paper's settings plus the batch-size change described above.

**VAE-GMM V1**

| Setting | Value |
|---|---|
| Latent size L | 64 |
| β | 1e-4 |
| Critic | on (WGAN-GP, sigmoid output), 3 critic steps per VAE step |
| Batch | 5,376 |
| Learning rates (VAE / critic) | 6e-3 / 1.2e-3 |
| Epochs | 300 |
| Reconstruction loss | BCE sum |
| Precision | fp32, TF32 off |
| Seed | 20261022 |
| Model size | encoder 589,408 / decoder 368,097 / critic 184,609 parameters. The decoder's 557.3 MFLOPs match the paper's Table 6. |
| Source | `models/vaegmm/V1_L64_b1e-4_critic/train_info.json` |

**RidgeNet:** **ridgenet-v2c**, adopted for the dataset.
- 14,209 parameters, 16 channels, 6 dense dilated blocks.
- Trained with the background-label fix, the SNR mix 25 / 50 / 25 %, crossing and parallel whistle scenes, and time-reversed backgrounds.
- 6,000 steps at batch 16, lr 3e-3.
- Source: `models/detector/ridgenet-v2c/detector_config.json`.

### Dataset Ratio & Augmentation Hypotheses

#### *50% vs 100% Data Impact:*

**✗ Not run for either model.**
- V1 trains on 100 % of the data: 180,154 images from 8 encounters and 1,415 files.
- The agreed scope is one VAE-GMM run (V1) and one GMM at K = 10,000. A 50 % run is not in that scope.

#### *Augmentation ON vs OFF Impact:*

**VAE-GMM: ✗ not run.** The paper's augmentation is always on: a random rotation of 0°, 90°, 180° or 270°, drawn independently for each image every time it is used.

**RidgeNet: ✅ the closest equivalent was run** (synthetic validation; paired whistle-by-whistle comparison).
- **The change:** ridgenet-v2b is v1 plus three training-data changes:
  - more training whistles at low and mid SNR (the "SNR emphasis")
  - extra crossing whistle scenes
  - time-reversed backgrounds
- **Overall effect:**
  - Recall 0.737 → 0.739.
  - Paired comparison: +14 / −13 whistles, p = 1. In other words, **no measurable effect**.
- **5–10 dB SNR band:** recall 0.554 → 0.571.
- **Side effect:** the versions trained with these changes (v2b, v2c, v2e) lose 3–4 points of recall on clear whistles (≥ 10 dB).
  - The cause is time reversal, the crossing scenes, or both. The two have not been separated.

### Ablation Study Breakdown (Removing 1 key component at a time)

**VAE-GMM ablations: ⏳ not run.**
- These would be the paper's Table 4: a standard VAE (run V7) and VAE-GMM without the critic (run V8).
- The specification says no sweep starts before V1 has finished and been evaluated.

**RidgeNet ablations: ✅ run.** Each is compared with the adopted v2c on synthetic validation, using paired comparisons:

#### *Component 1 removed → Metric change:*

**Background-label fix removed** (v2c → v2b).
- **What the fix does:** it gives zero loss weight to background pixels that look like real faint whistles, instead of teaching the model that they are "not a whistle".
- **Recall:** 0.760 → 0.739.
- **5–10 dB recall:** 0.631 → 0.571.
- **Effect of the fix alone:** **+24 / −11 whistles, p = 0.041**. This is the only data change with a significant effect.

#### *Component 2 removed → Metric change:*

**SNR emphasis removed:** v2c → v2e, which keeps the label fix but uses v1's SNR balance.
- **Recall:** 0.760 → 0.755.
- **5–10 dB recall:** 0.631 → 0.625.
- **Paired, v2e vs v2c:** +10 / −13, p = 0.68. **No significant effect.**

#### *Component 3* (the template has only two slots; the checklist asks for 3 variants)

**Wider context added:** v2c → v2d. This one adds something rather than removing it.
- **Recall:** 0.760 → 0.759.
- **5–10 dB recall:** 0.631 → 0.619.
- **Paired, v2d vs v2c:** +16 / −17, p = 1.
- **Cost:** about 20 % more S5 time on the GPU and 60 % more on the CPU.
- **Result:** rejected.

**Earlier variant:** v2a, which has wider context on v1, was **worse** than v1:
- Recall 0.710 vs 0.737.
- Paired: +15 / −32, p = 0.019.

**Sources:** README section "S5 detector experiments" and `models/detector/compare_*.json`.

---

## II. Test Results Collection & Deep Analysis
*(To be completed after running the best checkpoint on the held-out test set)*

### Evaluation Metrics Computed (vs. Paper's Baseline Table)

**Metric 1 (F1, detector on real annotated data, DCLDE 2011):**
- **Your Result: 65.2 %** (P 53.6 %, R 84.9 %)
- **Paper Result: 87.2 %** (P 88.8 %, R 85.8 %), from Li et al. 2023, their best result

**Metric 2 (FID₁, VAE-GMM at L = 64, β = 1e-4, K = 10k):**
- **Your Result: ⏳ pending**
- **Paper Result: 2.68** (Table 2)

**VAE-GMM against the paper** (row L = 64 of Table 1; row L = 64, β = 1e-4, K = 10k of Table 2)

| Metric | Our V1 | Paper | Notes |
|---|---|---|---|
| MSE (per-image sum of squared errors) | ⏳ | 0.5778 | Table 1 |
| PSNR [dB] | ⏳ | 44.5274 | Table 1 |
| SSIM | ⏳ | 0.9861 | Table 1 |
| KL (latent) | ⏳ | 43.1929 | Table 1 |
| FID₁ / FID₂ | ⏳ | 2.68 / 68.74 | Table 2. Also blocked because the Inception-V3 / VGG-16 weights have not been downloaded yet (waiting for approval). |
| Recall / Precision / F1 [%] | ⏳ | 87.21 / 91.99 / 89.54 | Table 2 |

**Interim training indicator (not a test metric):**
- The reconstruction loss is a per-image BCE sum. At the end of epoch 6 it was **425.8**.
- Two reference levels put that number in context:

| Reference level | BCE sum |
|---|---|
| Always outputting the average image | 449.2 |
| A perfect reconstruction | 151.9 |

- So V1 had covered about **8 %** of the distance from "average image" to "perfect" after 6 of 300 epochs.

**The paper's ablation table** (Table 4). We have not reproduced it yet. These are the targets for V7 and V8:

| Model | FID₁ | FID₂ | Recall [%] | Precision [%] | F1 [%] |
|---|---|---|---|---|---|
| Standard VAE (L = 64) | 91.83 | 502.41 | 23.70 | 37.51 | 29.05 |
| WGAN-GP | 13.82 | 202.51 | 61.47 | 83.17 | 70.70 |
| VAE-GMM without WGAN-GP (L = 64, K = 10k) | 7.37 | 182.84 | 80.58 | 92.96 | 86.33 |
| VAE-GMM with WGAN-GP (L = 64, K = 10k) | 2.68 | 68.74 | 87.21 | 91.99 | 89.54 |

**Detector on the DCLDE 2011 benchmark** (✅ done)
- **Model:** ridgenet-5-24 (a 5–24 kHz variant of v2c), threshold τ = 0.4.
- **Test data:** the 9 test files from Li et al. 2023, scored with their protocol.

| Variant | P [%] | R [%] | F1 [%] |
|---|---|---|---|
| **Main:** harmonics kept, detections ≥ 150 ms, 5–24 kHz, averaged over species | **53.6** | **84.9** | **65.2** |
| Scored against all annotated whistles (`--full-band`) | — | — | 62.1 |
| No minimum detection length | — | — | 46.5 |
| Li et al. 2023, best (5–50 kHz, their own setup; context only) | 88.8 | 85.8 | 87.2 |

**Detector on the synthetic held-out test** (✅ done, run once)

| | v1 | **v2c (adopted)** | 5-24 (τ = 0.4) |
|---|---|---|---|
| Track recall | 0.704 | **0.729** | 0.629 |
| Recall at < 5 dB SNR | 0.000 | 0.051 | 0.086 |
| Recall at 5–10 dB | 0.519 | 0.647 | 0.632 |
| Recall at 10–15 dB | 0.942 | 0.905 | 0.885 |
| Recall at ≥ 15 dB | 0.994 | 0.966 | 0.980 |
| False tracks per minute on contamination-free negatives (phase-randomised backgrounds + synthetic clicks) | 0.00 | 0.00 | 0.00 |

- **v2c vs v1, paired:** +28 / −14 whistles, McNemar p = 0.044.
- **5-24 model precision:** 0.576. This is only a lower bound: unlabeled real whistles in the backgrounds count as false detections.

### Performance Gap Hypothesis (If your results differ from the paper, why?)

**Detector on DCLDE 2011: recall matches, precision does not** (84.9 vs 85.8 % recall; 53.6 vs 88.8 % precision). Possible causes:

1. **Never saw DCLDE data.** The detector is tiny (about 14k parameters) and was trained only on synthetic whistles laid over 2017 false killer whale backgrounds.
   - Non-whistle sounds that never appear in those backgrounds are likely to trigger detections.
2. **Two files have very low precision:**
   - Precision is 29 % on `Qx-Tt-SCI0608-N1-060814-121518`.
   - Precision is 32 % on `Qx-Dc-CC0411-TAT11-CH2-041114-154040-s`. This is a CalCOFI recording, and the corpus README says it contains echosounder pings.
   - Not yet examined in detail (TODO §4).
3. **The setups differ, so this is not a like-for-like comparison:**
   - We use 5–24 kHz; Li et al. use 5–50 kHz. About 25 % of the annotated whistles fall outside our band and are excluded.
   - We keep harmonics.
   - We apply a 150 ms minimum detection length. This is Silbido's default and was not tuned; without it, F1 drops to 46.5.
4. **Missing fair baseline:** Silbido's graph search, scored with our rules, has not been run yet (TODO §1).

**VAE-GMM: no gap can be measured yet.** When the results come in, these differences are expected to matter:
1. **Different data:**
   - We have 180,154 images from 8 DCLDE 2022 encounters.
   - The paper has 228,673 vectors from 2,486 files.
   - Our detector and tracker (S5 and S6) are new designs, because the paper's are not available.
2. **Larger batch:** 5,376 vs 896, with linearly scaled learning rates.
   - This gives 6× fewer updates per epoch. The linear rule is a heuristic designed for SGD, not for Adam.
3. **BCE reconstruction loss:** chosen after the squared-error loss collapsed in gate G4. The paper does not state which loss it used.
4. **Feature-network weights:** our FID and P/R use torchvision's weights, which may not match the paper's. Absolute FID values are therefore **not comparable**; only trends are, as spec §5 says.

### Systematic Failure Case Analysis (Group Review)

These findings come from the run records. **No group review session has been held yet.**

#### *Pattern found in failed test samples:*

**Detector**
- **Low SNR:** recall collapses below 5 dB local SNR (5–9 %). It is about 63–65 % at 5–10 dB.
- **Clear whistles:** recall on whistles ≥ 10 dB dropped 3–4 points once the data changes were added.
- **Dense crossings:** in dense crossing scenes, S6 can split one whistle into fragments or swap which whistle is which.
- **Pulse trains:** the synchronous-comb rule threw away a real whistle that started at the same moment as a pulse train. This happened once in 15 files.
- **DCLDE 2011:** two recordings have very low precision (29 % and 32 %), and one of them has echosounder pings.

**VAE-GMM V1 at epoch 6**
- **Same output for every input:** every reconstruction is the same "+" blob, i.e. close to the average image.
- **Loss trajectory:** the BCE sum stayed near the average-image level (449) through epoch 5 and reached 425.8 at epoch 6.
  - The same plateau-then-breakout shape appeared in gate G4. There the loss sat around 450–520 for about 1,250 steps, then fell to 256 by step 3,000.
- **Gradient spike:** at step 66 (epoch 2) the gradient norm reached 1.59 × 10⁶. The epoch-3 average KL rose to 25,272. Training recovered by epoch 4 (rec 472.9).

#### *Are there data quality issues?*

**Yes. These are known and documented:**

| Issue | Detail |
|---|---|
| Many short fragments | **37.3 %** of images are shorter than 30 frames, because the minimum track length is 12 frames. The median whistle is 40 frames (213 ms). |
| Long whistles removed | **5.5 %** of tracks (10,458) were longer than 128 frames and were rejected rather than cropped. |
| Unbalanced encounters | Encounter 191 supplies 63,490 images (35 %); encounter 416 supplies 4,183 (2 %). |
| Imperfect backgrounds | The detector's training backgrounds come only from encounter 191, which is dense with whistles. Clean background windows are scarce, and unlabeled real whistles remain in them (this is why the label fix exists). |
| No ground truth | There are no hand labels for the 2017 data, so no precision or recall on real 2017 data is known. |
| One channel | Only channel 0 (chosen arbitrarily) is used. 2 of the 1,415 files produce no images. |

---

## III. Practical Implementation & Verification Checklist
(Hands-on execution log for Week 4 based on your workflow plan)

- [ ] **Hyperparameter Tuning:** Conducted grid/random search varying learning rate, batch size, and loss weights; logged every run to identify the best configuration.
  - **Status: not done as written, by design.** No search was run.
  - What was run: gates G3–G5 at B = 128, and the loss-form test in G4 (squared error vs BCE). All of it is logged in `models/vaegmm/_gates/`.
- [ ] **Dataset & Augmentation Experiments:** Trained models using 50% vs 100% data ratios and tested with augmentation ON vs OFF to measure model data-hunger.
  - **Status: not done for VAE-GMM.**
  - The detector equivalent (v1 vs v2b, more varied training data) is done and showed no effect.
- [ ] **Best Config Training & Ablation:** Re-trained the model with the top hyperparameter setup for full epochs (50-100); ran 3 ablation variants and built a results table matching the paper's ablation section.
  - **Status: in progress.**
  - V1 has done 6 of its 300 epochs. Note that the paper trains for 300 epochs, not 50–100.
  - The VAE-GMM ablations (V7, V8 → Table 4) have not started.
  - The 3 detector variants are done (v2b, v2e, v2d against v2c).
- [ ] **Test Results Collection:** Ran the top checkpoint on the held-out test set; computed all relevant metrics and prepared a results summary CSV.
  - **Status: detector done; VAE-GMM pending.**
  - Results summary CSV: [Notes-W-4-results.csv](Notes-W-4-results.csv), 135 rows, one metric per row. The VAE-GMM rows hold the paper's values; our values stay blank until V1 has been evaluated.
  - Detector results: the synthetic test (`eval_test.json`, `compare_test_*.json`) and DCLDE 2011.
  - The DCLDE 2011 score files are still only on Drive (TODO §2).
- [ ] **Model Internal Visualizations:** Generated metric bar charts, confusion matrices, or before/after spectrograms; visualized internals like feature maps, attention maps, or latent space t-SNEs for 5 easy and 5 hard samples.
  - **Status: not done.**
  - What exists: the V1 epoch-1 sample grid (`samples_epoch_001.png`) and the detector diagnostics figures in `preprocessing_checks/<run_id>/diagnostics/`.
  - No t-SNE plots and no 5 easy / 5 hard samples yet.
  - The paper's figures (Figs. 7–12, including the Fig. 12 confusion matrix) are implemented in `vae-eval figures` and `vae-eval conditional`. They need a trained V1 and a fitted GMM.

---

## IV. Implementation Notes & Code Snippets

### Experiment / Evaluation Insights

1. **Linear learning-rate scaling keeps the paper's lr × updates per epoch** (0.204 vs 0.202).
   - However, no gate tested the V1 setting itself (B = 5376). It is watched through its loss curves and sample grids, with the paper's B = 896 as the fallback.
   - **Rule:** if rec is still above about 420 at epoch 20, switch to the fallback.
2. **The squared-error loss collapses to a blank output on these sparse images.** Most pixels are 0, so outputting all zeros is already a good squared-error solution.
   - BCE sum fixed this: the G4 SSE ratio went from 1.0001 to 0.416.
3. **The critic dominates training cost.**
   - Gate G5, at B = 128 on the laptop: 0.565 s per iteration with the critic, 0.066 s without, i.e. **8.6× slower**.
   - Projected time for 300 epochs: 66 h with the critic, 7.7 h without.
4. **GPU training with the critic is not bit-reproducible** (G3: run-to-run maximum difference 12.6).
   - So the resume test compares against the run-to-run spread, not against exact equality.
5. **Single comparison runs are noisy, so differences need paired tests.**
   - The detector timings of identical networks varied from −34 % to +0 %.
   - Accuracy differences are therefore judged with paired McNemar tests on fixed sets, not with one aggregate number.
6. **Fewer multiply-adds did not mean faster.** Depthwise-separable detector designs were 1.3–1.7× slower on the GPU and 4–6× slower on the CPU, despite fewer multiply-adds.
7. **TF32 changed results between devices.** It made the detector's confidences on GPU and CPU differ by up to 7e-3, which changed the S6 tracks.
   - TF32 is off in every run.

### Core Evaluation Script Snippet

Table 1 reconstruction metrics, from `src/whistlepipe/vaegmm/evaluate.py` lines 99–118:

```python
@torch.no_grad()
def eval_recon(cfg, run_id: str, force: bool = False) -> Path:
    d = _run(cfg, run_id)
    dev, images = _ctx(cfg)
    vae, spec = load_trained_vae(d, dev)
    sse = ss = kl = 0.0
    n = 0
    for x in _image_batches(images):
        x = x.to(dev)
        mu, logvar = vae.encoder(x)
        xr = vae.decoder(mu)                                   # [DECISION] z = mu
        sse += float(recon_sse(xr, x).double().sum())
        ss += float(ssim(xr, x).double().sum())
        kl += float(kl_std_normal(mu, logvar).double().sum())
        n += len(x)
    mse = sse / n
    return _write(d / "eval" / "table1_recon.json", {
        "run_id": run_id, "latent_dim": spec["latent_dim"], "beta": spec["beta"], "n_images": n,
        "MSE_per_image_sse": mse, "PSNR_dB": psnr_from_mse(mse), "SSIM": ss / n, "KL_latent": kl / n,
        "convention": "MSE = mean per-image sum of squared errors; PSNR = 10 log10(16384/MSE); z = mu; unrotated training images"}, force)
```

The evaluation commands, in order, once V1 has finished:

```bash
python whistle.py fit-gmm --experiment V1 --K 10000
python whistle.py vae-eval recon --experiment V1
python whistle.py fetch-eval-weights --approved
python whistle.py vae-eval gen --experiment V1 --K 10000
python whistle.py vae-eval figures --experiment V1
```

- `fit-gmm` takes about 7 h on the laptop.
- `vae-eval recon` produces Table 1.
- `fetch-eval-weights` downloads the FID and P/R feature weights. Run it only after approval.
- `vae-eval gen` produces Table 2: FID₁, FID₂, P / R / F1.
- `vae-eval figures` produces Figs. 7–10.

---

## Sources

| Fact | Source |
|---|---|
| V1 settings, model size, dataset hashes | `models/vaegmm/V1_L64_b1e-4_critic/train_info.json` |
| V1 losses per epoch, gradient spike | `models/vaegmm/V1_L64_b1e-4_critic/train_log.jsonl` |
| Gates G3, G4 (both losses), G5, owner acceptance | `models/vaegmm/_gates/*.json` |
| Optimisation rules, experiment matrix, gates | `resources/TRAINING_SPECIFICATION.md` §3.3, §3.8, §4, §5 |
| Paper Tables 1, 2, 4 | `resources/VAE-GMM Cetacean Whistles.pdf`, pages 10 and 13 |
| Dataset counts, data limitations, detector experiments | `README.md` |
| Detector training settings | `models/detector/*/train_info.json`, `detector_config.json` |
| Detector held-out test | `models/detector/compare_test_ridgenet-v1_vs_ridgenet-v2c.json`, `mod_detector_5_24/models/detector/ridgenet-5-24/eval_test.json` |
| DCLDE 2011 benchmark | `mod_detector_5_24/TODO.md` and `README.md`. The score files are on Drive, not in the repo. |
