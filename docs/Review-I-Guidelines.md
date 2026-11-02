# Review-I: Generative Model for Cetacean Whistles (VAE-GMM reproduction)

Our answers to every Review-I criterion (Weeks 1–4, **70 marks**), with the numbers, evidence and reasons to present.

**Base paper:** G.-H. Park et al., "A Generative Model for Cetacean Whistles Using Variational Autoencoder and Mixture of Gaussians", *IEEE Access* 14 (2026), pp. 27877–27894, DOI 10.1109/ACCESS.2026.3664765.

**Our data:** NOAA DCLDE 2022 recordings of false killer whales (HICEAS 2017 survey, ship 1705).

**Status on 1 Nov 2026:**
- The full dataset is built: 180,154 whistle images. Since 2026-11-02, training uses encounter 191 only (363 files, 63,490 images), for 150 epochs, with K = 3,000.
- Every pre-training check is done.
- The main model (V1) has trained 6 of its 300 epochs on a Colab A100.
- The replacement whistle detector is finished and tested, including a real-data benchmark.

**Where the evidence is:**

| Kind | Location |
|---|---|
| Figures | `docs/figures/` |
| Week 4 detail | [Notes-W-4-filled.md](Notes-W-4-filled.md) |
| All results in one table | [Notes-W-4-results.csv](Notes-W-4-results.csv) |
| Specifications | `resources/` |

---

## The 15-minute presentation

| Minutes | Criteria | Show | Key message |
|---|---|---|---|
| **01–02** | 1A, 1B | Problem/goal slide; the 3 enhancement cards; the 8-week plan table | We reproduce a generative model of whistles and add three enhancements: a realism loss, learned conditioning, and three-axis evaluation. |
| **03–06** | 2A–2D, 1C, 1D | Dataset table; waveform figure; S1–S9 pipeline table; image statistics | 1,415 files, 23.6 h, checked by hash → 180,154 images. The paper has no split, and we explain why we don't invent one. The rotation augmentation is the paper's. |
| **07–10** | 3A–3D | Architecture table; loss equations; gate results; learning-curve figure | We verified the paper's model before training (parameter count matches the paper exactly; the gates caught a loss collapse). V1 is training: 6 of 300 epochs logged, no NaN. |
| **11–13** | 4A–4D | Detector experiment table; DCLDE 2011 results; paper target tables | Experiments use paired significance tests. Real-data detector F1 is 65.2 % (recall 84.9 % ≈ the published 85.8 %). VAE-GMM targets are set. |
| **14–15** | 4D, demo, Q&A | Live: `python whistle.py plot-waveform`, `pytest`, repo tour; failure cases | 187 automated tests; the known failure patterns and their causes. |

---

# Week 1: Orientation, Acoustics & Literature Breakdown (15 marks)

## 1A. Problem statement & research clarity (3 marks)

**Problem**

Cetacean whistles are frequency-modulated tones, and their shapes vary a great deal. The paper wants a **compact, tractable generative model** of these shapes for *biomimetic underwater communication*: man-made signals that sound like real whistles.

**The paper's goals (from its abstract)**
1. Learn the probability distribution of whistle contours from spectrogram images, using a **VAE**.
2. Model the latent space with a **Gaussian mixture (GMM)**, giving an explicit multimodal density.
3. Use that density for **unsupervised clustering** (whistle types, by MAP inference) and **conditional generation** (whistles with chosen features).
4. Improve the fine detail of the output with a **WGAN-GP critic**.

**Inputs and outputs**

| | What |
|---|---|
| Raw input | 60 s recordings, 6 channels at 500 kHz (FLAC) |
| Model input | 128 × 128 single-channel whistle image in [0, 1]. Rows are frequency (93.75 Hz each, centred on the whistle); columns are time (5.333 ms each). |
| Model outputs | (1) **New whistle images**: pick a GMM component, sample z, decode. (2) A **whistle-type label** for any whistle (MAP over the 10,000 components). (3) **Conditional samples** that meet a feature condition, such as length 30–40 frames or f_min in bins 34–55 (13 conditions in the paper's Table 3). |
| The paper's headline result | FID 2.68, recall 87.21 %, precision 91.99 %, F1 89.54 % (L = 64, K = 10,000) |

**Our selected enhancements**

| | 01 Biological Realism Loss | 02 Learned Conditional Generation | 03 Three-Axis Evaluation |
|---|---|---|---|
| **Idea** | Add acoustic-feature constraints to training, to penalise biologically implausible whistle contours | A learned conditioning mechanism that generates contours matching target features | Measure **novelty** (generated whistles differ from the training samples; no near-duplicates), **diversity** (coverage and variation) and **fidelity** (FID and complementary metrics) |
| **Details** | Features: duration, min/max frequency, frequency sweep, bandwidth | Conditioning on user-specified whistle characteristics or feature targets | Fidelity: FID₁, FID₂, precision. Diversity: recall, coverage. Novelty: nearest-neighbour distance to the training set. |
| **Evaluation** | Does plausibility improve without hurting fidelity or diversity? | Compared with the paper's component-based conditional generation, for control accuracy and quality | Applied to V1 and to both enhancements |
| **What the paper lacks** | No feature term in its loss (eq. 4 is reconstruction + KL + adversarial) | Its conditioning picks GMM components, then throws away samples that miss the condition (eqs. 8–10) | No novelty or memorisation check |
| **Starting point in our code** | `vaegmm/features.py` (7 features, not yet differentiable) | `vaegmm/conditional.py` (the paper's method, as the baseline) | `vaegmm/evaluate.py` (FID, P/R) |

**Design defaults [DECISION]**
- "Sweep" means end frequency minus start frequency.
- "Plausible" means within the 1st–99th percentile of the real data for each feature (see 2A).
- Enhancement 01 fine-tunes V1, with a matched control run.
- Enhancement 02 learns a conditional prior on the frozen V1 latents.
- The novelty threshold is the 1st percentile of real-to-real nearest-neighbour distances.

## 1B. Literature survey & timelines (3 marks)

### Key prior works (5–8)

*To be filled in by the team.*

| # | Work (authors, year, venue) | Contribution | How we use it |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |

### 8-week SDLC milestone plan

**Model: iterative and incremental.**
- First, the base reproduction goes through requirements → design → implementation → testing.
- Then each enhancement is added as its own increment, with a short design → build → test cycle, all measured with the same evaluation.

**Calendar [DECISION]:** Monday-to-Sunday weeks. Week 4 = 26 Oct – 1 Nov 2026, so Week 1 starts 5 Oct and Week 8 ends 29 Nov.

| Week | Dates | Phase | Milestone | Content | Evidence | Status |
|---|---|---|---|---|---|---|
| 1 | 5–11 Oct | Requirements & analysis | **M1** Paper broken down; preprocessing spec; repo and environment set up | S1–S9 specification with the paper evidence; first pipeline code; detector v1 trained | commit `02f445c` (8 Oct); `resources/PREPROCESSING_SPECIFICATION.md` | ✅ |
| 2 | 12–18 Oct | Data | **M2** Raw data verified; preprocessing validated | Detector threshold calibrated (10 Oct file); check runs on 1 / 5 / 10 / 363 files; detector variants v2a–v2e | README validation record | ✅ (no commits this week) |
| 3 | 19–25 Oct | Design & implementation | **M3** Dataset built; model, loss and evaluation code; gates | 180,154 images from 8 encounters; manifest of 1,415 files; VAE-GMM code and spec; gates G1, G2, G5, G6; inference and web demo | `8cf3ed4`, `2cf1fc1`, `f6827ce`, `f23674d` (21–24 Oct) | ✅ The dataset build slipped from Week 2. |
| 4 | 26 Oct – 1 Nov | Training & testing | **M4** Main training started; experiments; Review-I | BCE loss fix; gates G3 and G4; V1 on Colab (6 epochs); 5–24 kHz detector; DCLDE 2011 benchmark | `5925df3`, `63e11ae`, `f388805` (28–31 Oct) | 🟡 V1 partly trained |
| 5 | 2–8 Nov | Implementation & testing | **M5** Base reproduction done; enhancement 03 built | • Train V1 on encounter 191 (150 epochs, ~5.3 h on an A100).<br>• Fit the GMM at K = 3,000 (at most ~45 min).<br>• Paper Tables 1, 2, 3 and 6 and Figs. 7–12.<br>• Three-axis evaluation.<br>• t-SNE of the latents. | — | Planned |
| 6 | 9–15 Nov | Increment 1 | **M6** Enhancement 01 measured | • Differentiable features.<br>• Plausibility ranges.<br>• V1 fine-tuned with the realism loss, plus a control without it (~5 h each, 50 epochs). | — | Planned |
| 7 | 16–22 Nov | Increment 2 | **M7** Enhancement 02 compared with the paper | • Learned conditional prior.<br>• The paper's 13 conditions: control accuracy, W₁, FFD, P/R/F1 against eqs. 8–10. | — | Planned |
| 8 | 23–29 Nov | Integration & delivery | **M8** Final evaluation and review | • Three-axis evaluation of all models.<br>• Ablation table.<br>• Update the web demo.<br>• Report, slides, code freeze. | — | Planned |

**Why this plan**
1. **V1 is the critical path.** The GMM, every paper metric, both enhancement baselines and the t-SNE need the trained V1, so it comes first in Week 5.
2. **Enhancement 03 comes first** because it needs no training and becomes the yardstick for 01. Enhancement 01's test is "without compromising fidelity or diversity", and that is measured with 03.
3. **01 comes before 02** because 01 builds the differentiable features (duration, frequencies, sweep, bandwidth) that 02 conditions on.
4. **The compute budget sets the design.** One full 300-epoch run costs about 30 h on an A100 (about 350–360 Colab compute units). So 01 is a fine-tune plus a control (~10 h), and 02 trains only a small prior. This also keeps the agreed scope of one main VAE training and one GMM.
5. **Matched controls make the ablations fair.** They separate the effect of the new loss from the effect of extra training, and they supply the on/off ablation variants.
6. **Week 8 is a buffer.** Two things have already slipped (the dataset by a week; V1 was stopped at epoch 6), so Week 8 holds only integration and delivery work.

**Risks and fallbacks**

| Risk | Sign it is happening | Fallback |
|---|---|---|
| V1 doesn't converge at batch 5,376 | rec still above about 420 at epoch 20 | The paper's own setting (B = 896, lr 1e-3 / 2e-4); Weeks 6–7 move back by one week |
| Colab compute units run out | Balance below what the remaining epochs need | Use the 40 GB setting (B = 2,688), or cut the fine-tunes to 25 epochs |
| No approval to download the FID / P/R weights | Not approved by mid-Week 5 | Report Tables 1 and 3 first; FID moves to Week 6 |
| The realism loss hurts fidelity | FID or recall worse than the control | Lower its weight and report the trade-off curve as the result |

## 1C. Environment & audio pipeline setup (4 marks)

**Environment**

| | Laptop (preprocessing, gates, GMM, evaluation) | Colab (main training) |
|---|---|---|
| Hardware | NVIDIA RTX 5060 Laptop GPU, Windows 11 | NVIDIA A100-SXM4-80GB (5–24 kHz detector trained on a T4) |
| Python | 3.11.0 | 3.13.16 |
| Libraries | numpy 2.4.4, scipy 1.17.1, soundfile 0.13.1 (libsndfile 1.2.2), torch 2.14.0+cu130 | numpy 2.1.3, scipy 1.16.3, torch 2.11.0+cu130 |
| Setup | `pip install -r requirements.txt` (pinned lower bounds), torch from the CUDA index | `colab/train_V1_colab.ipynb` with a code bundle whose hashes are checked against the laptop copy |
| Tests | **187 automated tests** (`python -m pytest`), all passing | — |

**Audio libraries: soundfile + scipy instead of librosa / torchaudio**
- Every stage has to match the paper's chain exactly and be checked against an independent reference.
- S1 has to read one channel of a 6-channel 500 kHz FLAC in blocks and detect truncated files.
- S2 and S3 need exactly specified filters (polyphase Kaiser FIR; zero-phase Butterworth).
- soundfile and scipy give that control directly. librosa and torchaudio would add a layer whose defaults we would have to undo.

**Loading, waveform and resampling**: `python whistle.py plot-waveform` → `docs/figures/waveform_1705_20170930_015147_062.png`

| Step | What it does | Check |
|---|---|---|
| **S1 load** | Decode the FLAC, take channel 0, scale int16 → float32 in [−1, 1). Compare the decoded length with the header (a truncated file is rejected). | Output equals the raw channel bit for bit (12/12 checks on the gate file). |
| **Waveform** | Whole 60 s file (30,000,000 samples), a 2 ms zoom, and the spectra | Figure panels 1–3 |
| **S2 resample** | 500 kHz → **192 kHz**, polyphase FIR with a Kaiser window, ratio 48/125 → 11,520,000 samples | Spectrum preserved: the median S2 − S1 difference over 2–80 kHz is **+0.01 dB**; the anti-aliasing roll-off sits just below 96 kHz. |
| S3 high-pass | 4th-order Butterworth, zero-phase, 2 kHz | Measured −6.01 dB at 2 kHz, ≈ 0 dB above 5 kHz |

What the figure also shows: the file has bursts that reach full scale (clipping, S1 peak 0.99988). After resampling the peak reaches 1.227; that is the normal overshoot when a clipped signal is band-limited.

## 1D. Feature extraction & exploration (5 marks)

**STFT (S4), the paper's time-frequency grid**
- nfft = window = 2048, hop 1024 (50 %), periodic Hamming window, output in dB, no padding.
- At 192 kHz this gives **93.75 Hz × 5.333 ms** cells.
- A 60 s file becomes an 11,249 × 1,025 spectrogram.
- It matches scipy within 0.03 dB.

**From spectrogram to the model's features**

| Stage | What it does |
|---|---|
| S5 | RidgeNet (14,209 parameters) gives, for every time-frequency cell from 2 to 24 kHz, the confidence that a whistle passes through it |
| S6 | Tracker: threshold 0.5, gaps up to 12 frames bridged, tracks shorter than 12 frames dropped, harmonics and click combs removed |
| S7 | Contour vector: frequency (Hz) per 5.333 ms frame |
| S8 | Remove the centre frequency, re-synthesise as a constant-amplitude tone at complex baseband (12 kHz band) |
| S9 | 128 × 128 image: linear magnitude, normalised to max 1, 0 Hz on row 63 |

**Mel and CQT: not used.**
- The paper's images are on a **linear 93.75 Hz grid**, centred on each whistle's own frequency.
- A Mel or CQT axis would change what the model learns, so it would no longer be a reproduction.
- The 2-D whistle image *is* the paper's feature.

**Similarity matrices and t-SNE: planned for Week 5.**
- They need the trained encoder: a t-SNE of the 63,490 latent means coloured by GMM component, and nearest-neighbour similarity.
- They are part of enhancement 03's novelty axis.
- What exists now is the per-whistle feature statistics in 2A.

---

# Week 2: Dataset Collection, Splits & Preprocessing (15 marks)

## 2A. Dataset identification & integrity (3 marks)

**Source**
- NOAA *DCLDE 2022*, ship 1705 (R/V Lasker), HICEAS 2017.
- Public bucket: `gs://noaa-passive-bioacoustic/dclde/2022/dclde_2022_1705/audio/`.
- The encounter list comes from `metadata/whale_encounters_1705.csv`.

**Licences**
- The recordings are public NOAA data, but the data licence still has to be written into the README.
- The paper is CC BY-NC-ND 4.0.
- The project's own licence hasn't been chosen yet.

**Integrity checks** (the equivalent of `dataset_stats.py`; the same checks run as commands and leave records)
- `python whistle.py build-manifest`: SHA-256 of every raw file → `metadata/manifest_all.csv`. It is written once and never overwritten.
- `build-dataset` refuses any file that isn't in the manifest.
- S1 compares each file's decoded length with its header (a cut-off download fails).
- `python whistle.py validate-dataset`: automated checks of the final dataset.

### Raw recordings

| Item | Value |
|---|---|
| Files | **1,415** FLAC files (8 encounters, 60 s each) |
| Manifest status | **1,414 VALID, 1 SHORT** (shorter header, processed normally). 0 truncated, 0 invalid, 0 unreadable. |
| Format | 500 kHz, 6 channels |
| Total | **23.57 h of audio, 269.8 GB** |
| Integrity | All 1,415 SHA-256 values matched during processing; every raw file was unchanged afterwards |
| Recorded | 29 Sep – 30 Nov 2017 |

| Encounter | Files | Hours | Recorded (UTC) | Images | Share of images |
|---|---|---|---|---|---|
| 191 | 363 | 6.05 | 29–30 Sep 2017 | 63,490 | 35.2 % |
| 214 | 124 | 2.07 | 9–10 Oct 2017 | 6,677 | 3.7 % |
| 276 | 120 | 1.99 | 1 Nov 2017 | 14,151 | 7.9 % |
| 279 | 145 | 2.42 | 2–3 Nov 2017 | 33,517 | 18.6 % |
| 280 | 104 | 1.73 | 3 Nov 2017 | 15,803 | 8.8 % |
| 308 | 162 | 2.70 | 7 Nov 2017 | 21,894 | 12.2 % |
| 331 | 331 | 5.52 | 17–18 Nov 2017 | 20,439 | 11.3 % |
| 416 | 66 | 1.10 | 30 Nov 2017 | 4,183 | 2.3 % |
| **Total** | **1,415** | **23.57** | | **180,154** | 100 % |

### Processing outcome

| Item | Value |
|---|---|
| Files processed | **1,415 / 1,415 OK**, 0 failed |
| Contours after tracking | 190,612 |
| Rejected as longer than 128 frames (683 ms) | 10,458 (5.5 %) |
| Rejected as wider than the 12 kHz image band | 0 |
| **Images** | **180,154** |
| Images per file | mean 127.3, median 95, max 1,003; 2 files had none |
| Build | 21 Oct 2026, commit `8cf3ed4`, detector `ridgenet-v2c`, 71.5 min (3.0 s per file) |

### Whistle statistics

| Feature | Mean | Min | 5 % | Median | 95 % | Max |
|---|---|---|---|---|---|---|
| Duration [ms] | 254.4 | 64.0 | 69.3 | **213.3** | 570.7 | 682.7 |
| Length [frames] | 48 | 12 | 13 | **40** | 107 | 128 |
| f_min [Hz] | 6,769 | 2,063 | 3,631 | 6,326 | 11,877 | 23,957 |
| f_max [Hz] | 7,699 | 2,077 | 4,274 | 7,265 | 12,938 | 23,999 |
| Centre frequency [Hz] | 7,234 | 2,070 | 4,008 | **6,774** | 12,375 | 23,978 |
| Bandwidth [Hz] | 929 | 2 | 103 | **652** | 2,714 | 11,172 |
| Detector confidence | 0.964 | 0.567 | 0.869 | 0.980 | 0.997 | 1.000 |

37.3 % of images are shorter than 30 frames (160 ms).

### Image statistics (all 180,154 images read)

| Item | Value |
|---|---|
| Tensor | (N, 128, 128) float32, 64 KiB per image, 11.8 GB in total |
| Values | All finite, in [0, 1]; **every image has a maximum of exactly 1** |
| Pixels above 0.05 (the paper's contour threshold) | **1.07 %**: on average 176 of 16,384 pixels per image (median 148, max 546) |
| Mean pixel | 0.00614 |

**Takeaway:** the images are about 99 % empty. This explains why a squared-error loss collapsed in testing, and why we use cross-entropy (see 3C).

## 2B. Data leakage prevention & splits (4 marks)

**The paper specifies no train/val/test split, so we don't invent one.**
- The paper trains on all whistle images, and evaluates generation **against the training data**: FID and precision/recall against the training set (Abstract; §IV).
- For a generative model, the training distribution *is* the reference that generated samples are compared with.
- Our specifications record this explicitly: "do not infer a split from anything" (PREPROCESSING_SPECIFICATION A-18 and D-12; TRAINING_SPECIFICATION §2).

**Fixed seeds everywhere**
- V1 training: 20261022.
- Detector and synthetic data: 20261008.
- GMM initialisation and evaluation sampling: fixed seeds, recorded in each run's `train_info.json` together with the git commit and the SHA-256 of every input file.

**Where we do have splits (the detector), they are leak-free**
- The detector's synthetic training, validation and test sets use **disjoint recording files**: 24 / 16 / 16 background files.
- "Recording file" plays the role of "speaker".
- These 56 files are marked in `files.csv`, so they can be excluded from any analysis.
- Validation and test sets are fixed. Each detector version was compared whistle by whistle on the same samples, and the test set was used **once**.

## 2C. Preprocessing & augmentation pipeline (5 marks)

**The complete pipeline (S1–S9), run by `python whistle.py build-dataset`**

| Stage | Operation | Setting |
|---|---|---|
| S1 | Load, channel 0, int16 → float32 | Truncation check |
| S2 | Resample 500 → 192 kHz | Polyphase Kaiser FIR, 48/125 |
| S3 | High-pass | 2 kHz, 4th-order Butterworth, zero-phase |
| S4 | STFT | 2048 / 1024, Hamming, dB → 93.75 Hz × 5.333 ms |
| S5 | Whistle detector (RidgeNet) | 2–24 kHz confidence map |
| S6 | Tracker | Threshold 0.5; gap 12 frames; minimum 12 frames; removes harmonics and click combs |
| S7 | Contour vector | Hz per frame |
| S8 | Centre-frequency removal → constant-amplitude baseband tone | 12 kHz complex baseband, 2.7 ms edge ramps |
| S9 | Image | 128 × 128, linear magnitude, **normalised to max 1** |

**Saved output** (one folder per encounter in `preprocessed/`)

| File | Contents |
|---|---|
| `images.npy` | (N, 128, 128) float32 |
| `images.csv` | One metadata row per image |
| `contours.npz` | The contours |
| `files.csv` | One row per raw file |

**Records**
- `dataset_info.json` holds the config hash, the detector hash, the software versions and the counts.
- Every file's lineage, from raw SHA-256 to output, is kept.
- Check runs on 1 / 5 / 10 files and one file per encounter passed automated validation before the full build.

**Augmentation: we use the paper's own**
- **Random rotation by 0°, 90°, 180° or 270°**, chosen independently for each image every time it is used (paper App. A-B).
- The rotated image is the VAE input, its reconstruction target, and the critic's "real" sample.
- SpecAugment and pitch shift are not in the paper, and they would distort the contour shapes the model is meant to learn.
- **Noise augmentation** is used where it fits: the detector trains on synthetic whistles mixed into real background noise at controlled SNR (−2 to 24 dB), with clicks, crossings and time-reversed backgrounds.

## 2D. DataLoader optimisation (3 marks)

**Throughput**

| Measurement | Value | Meets ≥ 10 batches/s? |
|---|---|---|
| Gate G5, laptop, batch 128, VAE step without the critic | **15.2 batches/s** | ✅ |
| Gate G5, same, with the critic (3 critic steps + gradient penalty per VAE step) | 1.8 batches/s | Limited by GPU work |
| Time the GPU waited for data (G5) | **0.003 %** | The loader is not the bottleneck |
| V1 on the A100, batch 5,376 | 10.75 s per step (≈ 500 images/s); loader wait **0.3 %** | Compute-bound |

**How the loader works**
- Memory-mapped `images.npy`, opened read-only.
- 2 batches prefetched.
- Rotation applied on the fly.
- Optional gradient accumulation (`micro_batch`) for GPUs with less memory, giving the same update.

**Shapes and values (gate G1)**
- All 180,154 images: 0 non-finite, 0 out of range, 0 with max ≠ 1.
- Counts equal `dataset_info.json`.
- Tensors are (B, 1, 128, 128) into the encoder and (B, 64) in the latent space.

---

# Week 3: Architectural Analysis & Baseline Implementation (20 marks)

## 3A. Architectural breakdown (4 marks)

**Data flow**

```
image x (1×128×128)
  ─► Encoder ─► μ, log σ² (64 each) ─► z = μ + σ·ε ─► Decoder ─► x̂ (1×128×128, sigmoid)
                                                          │
                              real x (rotated) ──► Critic ◄┘  (sigmoid score)
After training:  μ(x) of all 63,490 images (unrotated) ─► GMM, K = 3,000, full covariance (EM)
Generate:        component k ~ π ─► z ~ N(μ_k, Σ_k) ─► Decoder ─► new whistle
```

**Component specification** (the paper's Tables 7–8; "same" padding)

| Network | Layers | Output shape | Parameters |
|---|---|---|---|
| **Encoder** | 4 × [Conv 3×3, stride 2, + swish] with 32, 32, 64, 64 channels → FC 4096 → 2L | 128² → 64² → 32² → 16² → 8×8×64 → μ, log σ² ∈ ℝ⁶⁴ | **589,408** |
| **Decoder** | FC L → 4096, reshape 8×8×64 → 4 × [TConv 3×3, stride 2, + swish] with 64, 64, 32, 32 channels → Conv 3×3 → sigmoid | ℝ⁶⁴ → 8×8×64 → … → 128×128×32 → 1×128×128 | **368,097** |
| **Critic** | Conv 5×5 s2 32 + LeakyReLU 0.2 → 3 × [Conv 5×5 s2 + LayerNorm + LeakyReLU] with 32, 64, 64 channels → Conv 8×8 → sigmoid | 128² → … → 8×8×64 → scalar | **184,609** |

**Match with the paper:** the decoder's **368,097 parameters and 557.3 MFLOPs** equal the paper's Table 6 (368k / 557M), and a unit test checks this.

## 3B. Hyperparameters & loss formulation (4 marks)

**Losses**
- **VAE (eq. 4):** L = L_rec + β·L_KL + λ·L_adv.
  - L_rec: per-image sum of binary cross-entropy (Bernoulli likelihood).
  - L_KL: KL(q(z|x) ‖ N(0, I)), summed over the 64 dimensions.
  - L_adv: −E[log D(x̂)].
- **Critic (eq. 5, WGAN-GP):** L_D = E[D(x̂)] − E[D(x)] + λ_GP·E[(‖∇D(x̃)‖₂ − 1)²], where x̃ is a random mix of a real and a reconstructed image.
- **Auxiliary losses:**
  - the KL term (weight β = 1e-4, which is small);
  - the adversarial term (λ = 1);
  - the gradient penalty (λ_GP = 5).
- **A conflict in the paper, documented and resolved:** the paper uses a *sigmoid* critic with a WGAN-GP objective. We follow it literally: D is the sigmoid output, and log D is computed as `logsigmoid` for numerical safety.

**Hyperparameters**

| | Paper | Ours (V1) |
|---|---|---|
| Latent size L | 64 | 64 |
| β, λ_adv, λ_GP, critic steps per VAE step | 1e-4, 1, 5, 3 | same |
| Optimiser | Adam (critic β₁ = 0, β₂ = 0.9) | Adam: VAE (0.9, 0.999, ε 1e-8) [DECISION]; critic (0, 0.9) |
| Learning rate (VAE / critic) | 1e-3 / 2e-4 at batch 896 (7 GPUs × 128) | **6e-3 / 1.2e-3 at batch 5,376** (one A100). Scaled linearly, which keeps lr × updates per epoch at 0.204 vs the paper's 0.202. |
| LR scheduler | not stated | **Constant** [DECISION] |
| Epochs | 300 | 300 |
| Initialisation | MATLAB defaults | Glorot-uniform, zero biases (same as MATLAB) |
| Augmentation | random 90° rotations | same |

**Mixed precision: deliberately off.**
- Everything runs in fp32 with TF32 disabled.
- On our GPU, TF32 changed the detector's output by up to 7e-3 between GPU and CPU, which changed the tracked whistles.
- For a reproduction, matching results across machines is worth more than the speed-up.

## 3C. Baseline smoke test & verification (6 marks)

**We tested the paper's real model instead of a toy 3-layer network.**
- A toy network would show that *a* network trains.
- These gates show that *the paper's* model trains correctly, and one of them found a real bug before the expensive run.
- All gates ran on the laptop at batch 128.

| Gate | Test | Result |
|---|---|---|
| **G0 unit tests** | Architecture shapes and the Table 6 counts; KL vs Monte Carlo; gradient penalty vs finite differences; gradient accumulation; EM vs scikit-learn; features; metrics | ✅ 187 tests pass |
| **G1 data** | All images finite, in [0, 1], max = 1; counts | ✅ 180,154 / 0 problems |
| **G2 numerics** | GPU vs CPU forward pass on real images; finite losses and gradients | ✅ max difference 6e-8 (decoder), 4e-7 (critic); all gradients finite |
| **G3 smoke + resume** | 2 epochs on 2,048 images; restoring from a checkpoint and continuing | ✅ Restore is bit-exact. GPU training with the critic is not bit-reproducible (run-to-run difference 12.6), so resume is checked against that spread. |
| **G4 overfit (loss goes down)** | 256 images, up to 3,000 steps; the error must fall to ≤ 0.3 × that of a blank image | **Squared-error loss: collapsed to a blank output (ratio stuck at 1.0001) → found and fixed.** With BCE: 0.923 → **0.416** at step 3,000, falling faster each time. The 0.3 mark wasn't reached in the steps allowed; the project owner accepted the result on that trend. |
| **G5 timing** | Speed, data wait, memory | 0.56 s per iteration with the critic (66 h projected for 300 epochs on the laptop → moved to an A100); 1.7 GB peak memory |
| **G6 features** | Feature extractor on 4,000 real images | ✅ 100 % within tolerance for length and bandwidth |

**Parameter count:** `n_params()` gives encoder 589,408, decoder 368,097, critic 184,609. They are recorded in `train_info.json` and unit-tested against the paper's Table 6. This is the same information torchsummary would give, plus a check against the paper.

## 3D. First full training run & checkpointing (6 marks)

**Run V1_L64_b1e-4_critic** (figure: `docs/figures/v1_learning_curves.png`)

| Item | Value |
|---|---|
| Started | 29 Oct 2026, Colab A100-SXM4-80GB, commit `5925df3` (clean); data hashes match the laptop copy exactly |
| Speed | 34 steps per epoch, **365.5 s per epoch** (first epoch 396.6 s); loader wait 0.3 % |
| Progress | **6 epochs (204 steps)** before it was stopped. The remaining 294 epochs are Week 5 milestone M5 (≈ 30 h). |
| Logging | `train_log.jsonl`: every 8 steps, plus a summary per epoch (losses, critic outputs, gradient norm, timing); the notebook plots them |
| NaN / stability | **Automatic check:** every logged record and every epoch summary goes through `check_finite()` (`vaegmm/train.py`), which stops training with an error on any NaN or inf. No NaN or inf so far. One gradient spike at step 66 (norm 1.6 × 10⁶; KL briefly up to 77k); it recovered by epoch 4. |
| Checkpoints | `last.pt` every epoch: both networks, both optimiser states, epoch and step. The data order is derived from the seed and the epoch number, so a resumed run continues the same sequence (gate G3). Also a kept copy every 25 epochs, and a sample grid every 10 epochs. |
| Model kept | The **final** epoch. The paper has no validation set and describes no model selection, so a "best" epoch would need a split it doesn't have. |

**Learning curve (mean per epoch)**

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Reconstruction loss (BCE sum per image) | 1833.2 | 570.5 | 780.2 | 472.9 | 450.1 | **425.8** |
| KL | 791 | 338 | 25,272 | 3,212 | 5,534 | 2,193 |
| Critic D(real) / D(reconstruction) | 0.61 / 0.38 | 0.62 / 0.36 | 0.74 / 0.29 | 0.74 / 0.25 | 0.81 / 0.23 | 0.84 / 0.21 |

**Reading the curve:**
- 449.2 is the loss of always outputting the average image; 151.9 is a perfect reconstruction.
- At epoch 6, V1 has just dropped below the "average image" level.
- In the laptop overfit test the same pattern appeared: a long plateau near 450, then a fast fall (256 by step 3,000). So this is expected early behaviour.

---

# Week 4: Training Experiments, Ablation & Test Evaluation (20 marks)

Full detail: [Notes-W-4-filled.md](Notes-W-4-filled.md). Every number: [Notes-W-4-results.csv](Notes-W-4-results.csv) (135 rows).

## 4A. Hyperparameter tuning (5 marks)

**For VAE-GMM: no search, on purpose.**
- A reproduction must use the paper's hyperparameters.
- Tuning them against the paper's published scores would bias the comparison (TRAINING_SPECIFICATION §5).

**What was explored and logged**

| Experiment | Settings | Outcome |
|---|---|---|
| Batch size and learning rate (linear scaling) | B = 128 (lr 1.43e-4 / 2.86e-5) in the gates; **B = 5,376 (6e-3 / 1.2e-3) in V1**; B = 896 (the paper's) and 2,688 configured as fallbacks | lr × updates per epoch kept at 0.201–0.204 (paper 0.202) |
| Reconstruction loss | Squared error vs binary cross-entropy (G4) | Squared error collapses (1.0001); **BCE learns (0.416)** |
| Loss weights | β = 1e-4, λ = 1, λ_GP = 5 | Fixed to the paper's values |
| Detector design (6 versions, all logged) | Context width, training-data mix, label fix, SNR balance | **v2c adopted**: recall 0.737 → 0.760 on validation |

## 4B. Dataset & augmentation impact (4 marks)

| Experiment | Result |
|---|---|
| 50 % vs 100 % data | Not run as a comparison. Since 2026-11-02, V1 uses encounter 191 only (35 % of the images: 63,490); the agreed scope is one main training. |
| Augmentation ON vs OFF (detector: v1 vs v2b, with SNR emphasis, crossings and time-reversed backgrounds) | Recall 0.737 → 0.739; paired +14 / −13 whistles, **p = 1 → no effect**. Recall on clear whistles (≥ 10 dB) dropped 3–4 points. |
| VAE-GMM augmentation (90° rotations) | Always on, as in the paper |

## 4C. Ablation study (5 marks)

**Three detector ablations**, each on the same fixed validation whistles and judged with a paired McNemar test:

| Variant (vs the adopted v2c) | Recall | 5–10 dB recall | Paired result | Verdict |
|---|---|---|---|---|
| v2c (full) | **0.760** | **0.631** | — | adopted |
| 1. Background-label fix removed (v2b) | 0.739 | 0.571 | v2c +24 / −11, **p = 0.041** | the fix matters |
| 2. SNR emphasis removed (v2e) | 0.755 | 0.625 | +10 / −13, p = 0.68 | no effect |
| 3. Wider context added (v2d) | 0.759 | 0.619 | +16 / −17, p = 1 | no gain; 20 % slower on the GPU → rejected |

**VAE-GMM ablation targets:** the paper's Table 4, to be run after V1 is evaluated. The enhancements' on/off runs (Weeks 6–8) also serve as ablations.

| Model | FID₁ | FID₂ | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Standard VAE (L = 64) | 91.83 | 502.41 | 23.70 | 37.51 | 29.05 |
| WGAN-GP | 13.82 | 202.51 | 61.47 | 83.17 | 70.70 |
| VAE-GMM without critic | 7.37 | 182.84 | 80.58 | 92.96 | 86.33 |
| **VAE-GMM with critic (main model)** | **2.68** | **68.74** | **87.21** | 91.99 | **89.54** |

## 4D. Test results & failure analysis (6 marks)

**Detector on real annotated data**: DCLDE 2011, the 9 test files from Li et al. 2023, scored with their protocol.

| | Precision | Recall | F1 |
|---|---|---|---|
| **Ours** (ridgenet-5-24, 5–24 kHz, harmonics kept, detections ≥ 150 ms) | 53.6 % | **84.9 %** | 65.2 % |
| Li et al. 2023 (best; 5–50 kHz; their setup, so context only) | 88.8 % | 85.8 % | 87.2 % |

Variants: scored against all annotated whistles, F1 62.1 %; with no minimum detection length, F1 46.5 %.

**Detector on the held-out synthetic test (used once)**
- v1 → v2c recall: **0.704 → 0.729**. Paired +28 / −14, **p = 0.044**.
- 5–10 dB recall: 0.519 → 0.647.
- 0 false tracks per minute on clean negatives.

**VAE-GMM against the paper (targets; ours pending V1)**

| Table | Paper value |
|---|---|
| Table 1 (L = 64) | MSE 0.5778, PSNR 44.53 dB, SSIM 0.9861, KL 43.19 |
| Table 2 (L = 64, β = 1e-4, K = 10k) | FID₁ 2.68, FID₂ 68.74, R 87.21 %, P 91.99 %, F1 89.54 % |

"Held-out" here means the paper's protocol: compared against the training distribution (see 2B).

**Why the detector's real-data precision is lower**
1. It is a ~14k-parameter model trained only on synthetic whistles over 2017 false killer whale backgrounds. It never saw DCLDE 2011 data.
2. Two files have very low precision: 29 %, and 32 % on a recording with echosounder pings.
3. The setups differ: our 5–24 kHz band excludes about 25 % of the annotated whistles, and Li et al. use 5–50 kHz.

**Failure patterns found**

| Where | Pattern | Cause / next step |
|---|---|---|
| Detector | Recall collapses below 5 dB local SNR (5–9 %) | A limit of the 14k-parameter detector |
| Detector | −3 to −4 points on clear whistles after the data changes | Time reversal or crossing scenes (not yet separated) |
| Tracker | Fragments or swapped identities in dense crossings; the comb rule once removed a real whistle (1 of 15 files) | Replacement-tracker rules |
| DCLDE 2011 | False positives on pings and impulsive noise | Inspect the false positives on development files only |
| V1 at epoch 6 | Every reconstruction is the same "+" blob (close to the average image) | Expected plateau: the overfit test broke out of the same plateau after ~1,250 steps |
| Data | 37 % of images shorter than 30 frames; 5.5 % of long whistles rejected; encounter 191 = 35 % of the data | Known; `images.csv` allows filtering without re-processing |

---

## Still to confirm before the review

| Item | Status |
|---|---|
| Review date, team members and who presents which slot | Not known yet |
| Prior works (1B) | Team to fill in |
| NOAA data licence | To be written into the README |
