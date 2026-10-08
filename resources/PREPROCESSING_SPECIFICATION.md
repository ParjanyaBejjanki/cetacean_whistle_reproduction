# PREPROCESSING SPECIFICATION

**Project:** Reproduction of *"A Generative Model for Cetacean Whistles Using Variational Autoencoder and Mixture of Gaussians"* (G.-H. Park et al., IEEE Access, vol. 14, 2026, pp. 27877–27894, DOI 10.1109/ACCESS.2026.3664765)
**Document status:** Authoritative preprocessing specification — **no preprocessing code exists yet**
**Created:** 2026-10-08
**Location note:** this folder was originally spelled `resouces/` on disk; the project owner has since renamed it `resources/`. Path strings elsewhere in this document that read `resouces/` refer to this same folder (text left unchanged for traceability).
**Revision 2 (2026-10-08):** added §11 (reference-detector analysis and replacement requirements) and a cross-reference on A-20. **No confirmed parameter in §§1–10 was changed.**

---

## 0. How to read this document

### 0.1 Evidence classes (required by the task)

| Class | Meaning |
|---|---|
| **EXPLICIT** | Directly stated by the paper (text, caption, table, or appendix). |
| **IMPLIED** | Not stated verbatim, but reliably derivable from stated facts (arithmetic, figure axes, table units). The derivation is always shown. |
| **UNSPECIFIED** | The paper does not provide enough information. **Nothing is guessed.** Each UNSPECIFIED item appears again in §8 and, where it blocks coding, in §10. |

### 0.2 Sources and how they are cited

| Tag | Source | Role |
|---|---|---|
| **[P]** | `resouces/VAE-GMM Cetacean Wistles.pdf` (the paper; 18 PDF pages). Cited as section + journal page (e.g. "§II-C, p. 27881"). | **Primary and only authority** for the pipeline. |
| **[R32]** | `resouces/reference_32_mca_cfar.md` — a Markdown copy of the paper's reference [32] (Park et al., "A Design of MCA-CFAR Detector for Extraction of Cetacean Whistle Contours", KICS 2024). Authors match ref. [32] in [P]. | **Secondary.** Describes a detector the paper cites but does not parameterise. Its values are *not* evidence of what [P] used (see §3.6, §8). Provenance of the Markdown transcription itself is unverified. |
| **[DS]** | `metadata/Dataset_DCLDE_Oahu.pdf`, `metadata/DCLDE2020_DetectionData.xlsx`, `metadata/whale_encounters_1705.csv`, `metadata/manifest_363.csv` | Dataset description only (§2, §7). Never used as evidence for the paper's pipeline. |
| **[INS]** | Read-only inspection of `encounter_191_audio/` performed for this document (2026-10-08). | Dataset facts only. |

Detail for each paper citation is in §6 (evidence IDs `E-01` … `E-17`).

Sources used **only** in §11 (defined in §11.0): **[L30]** Li et al. 2020 (arXiv:2005.08894), **[31]** and **[33]** (the papers numbered 31 and 33 in [P]'s reference list; not obtained), **[RW1]** (a related paper that [P] does not cite).

### 0.3 Not paper evidence

`metadata/audio_activity_scores.csv` (columns `activity`, `whistle_snr_db`, `click_crest_factor`, `rms_energy`) is **not** part of the paper's methodology and its generating procedure is undocumented. It must not be used as a quality-control or selection rule (§9, D-11).

---

## 1. Objective

Establish, **before any code is written**, the exact preprocessing specification the paper uses to turn raw towed-array recordings of false killer whale (*Pseudorca crassidens*) sounds into the **128 × 128 single-channel whistle images (values 0–1)** that train the VAE-GMM, and record precisely which parts of that specification the paper does **not** supply.

The reproduction applies the paper's methodology unchanged to the 363 locally available files. The smaller file count is treated purely as a dataset-size difference (§2.2).

---

## 2. Paper dataset vs our dataset

### 2.1 Comparison

| Attribute | Paper [P] | Our dataset [INS][DS] | Match? |
|---|---|---|---|
| Source collection | NOAA DCLDE 2022 raw passive acoustic data, HICEAS survey, July–Nov 2017 (E-01) | Same collection: `gs://noaa-passive-bioacoustic/dclde/2022/dclde_2022_1705/audio/`, file prefix `1705` (R/V Lasker), dates 2017-09-29/30 | ✅ |
| Target species | False killer whale (E-02) | Encounter 191 = species code 33 = *False killer whale* (verified in the original workbook `DCLDE2020_DetectionData.xlsx`, sheet `1705_OdontoceteDetections`, AcousticDetectionNumber 191, 2017-09-29 22:49:48 → 2017-09-30 04:53:44 UTC) | ✅ |
| Number of audio files | **2,486** files selected as containing false killer whale whistles (E-02) | **363** files | ❌ size difference only |
| File duration | "generally" 60 s (E-02) | Exactly 60.000 s for all 363 | ✅ |
| Hydrophone channels per file | 6 (E-02) | 6 in all 363 | ✅ |
| Sampling rate | 500 kHz (E-02) | 500,000 Hz in all 363 | ✅ |
| Container / bit depth | **Not stated** | FLAC, PCM_16 (16-bit) in all 363. Original DCLDE release was 16-bit WAV with a lossless FLAC version also published [DS] | n/a (paper silent) |
| Whistle vectors obtained | **228,673** from 2,486 files (≈ 91.98 per file) (E-06) | Unknown until pipeline is run | n/a |
| How files were judged to contain whistles | **Not stated** | Not applicable — the 363 files are *every consecutive 60-s file of one encounter*, not a per-file whistle-presence selection | ⚠ difference in selection (documented, not corrected) |

### 2.2 Dataset-size difference — treatment

* The paper does **not** state, imply, or condition any preprocessing step on the number of files. Every stage in §3 operates independently per file (stages S0–S4), per detected whistle (S5–S8), or per image (S9).
* **Therefore no preprocessing parameter, threshold, STFT setting, frequency range, or detector setting changes because 363 < 2,486.**
* The paper's counts (2,486 files, 228,673 vectors) are *reference facts about the paper*, **not targets**. They must not be used to tune thresholds so that a yield of ≈ 92 vectors/file is reproduced (§9, D-12).

### 2.3 Inference about where the 2,486 files likely came from (NOT paper-stated)

From the original DCLDE workbook: ship 1705 has 8 false-killer-whale encounters totalling 1,415.0 min and ship 1706 has 5 totalling 1,048.3 min — **2,463.3 min across 13 encounters**. Adding the partial-minute files at each encounter boundary would bring this close to 2,486 files. This is consistent with the paper having used all files overlapping all false-killer-whale encounters of both ships, in which case **our 363 files are one encounter out of 13 (≈ 14.6 % of the paper's file count)**. This is an inference from public metadata; the paper does not say it. It matters only as context: the data are from the same population, one encounter.

---

## 3. Exact preprocessing pipeline

### 3.1 Pipeline overview

```
S0  Raw file: 60 s × 6 ch @ 500 kHz
 │
S1  Channel handling ........................ UNSPECIFIED
 │
S2  Resample 500 kHz → 192 kHz
 │
S3  High-pass filter, passband above 2 kHz ... (order vs S2/S4 UNSPECIFIED)
 │
S4  STFT spectrogram: Hamming, 2048 pt, 50 % overlap
 │
S5  Whistle detection: neural network [31] ... method cited, NOT provided
 │
S6  Post-processing to individual tracks: [32],[33] ... cited, [33] NOT provided,
 │                                                       [32] only partially described
S7  One time–frequency vector per whistle      (228,673 vectors in the paper)
 │
S8  Remove centre frequency → constant-amplitude NFM baseband signal
 │
S9  Spectrogram of that signal → 128 × 128 grayscale image, values in [0, 1]
 │
 ▼
Final dataset = training input to VAE (128 × 128 × 1)
```

The stage order S2 → S4 → S5–S9 is stated by the paper (E-03, E-05, E-06). The position of S3 relative to S2/S4 is **not** stated (§8, A-04).

### 3.2 Stage inputs and outputs

See §5.1 (kept there so that the section numbering follows the requested outline).

### 3.3 Operations the paper does NOT describe

The paper describes **no** background/noise normalisation (e.g. whitening, median subtraction, per-band normalisation), **no** click removal, **no** explicit per-file or per-whistle quality-control or rejection rule, **no** train/validation/test split, and **no** audio segmentation other than processing each 60-s file. These are recorded as UNSPECIFIED/"not described" and must not be added (§9).

### 3.4 Operations that are downstream of dataset construction (not preprocessing)

| Operation | Where | Why it is not preprocessing |
|---|---|---|
| Random rotation of each image by 0°, 90°, 180°, 270° (equal probability) | Appendix A, p. 27892 (E-13) | Training-time augmentation of the already-built dataset |
| Threshold 0.05 on a *generated* image → binary contour → time–frequency vector → NFM signal | Fig. 2 caption, p. 27880 (E-09) | Inverse step used after generation; useful only because it constrains what pixel values mean |
| Threshold 0.1 (FID₁) / none (FID₂); bicubic resize 128→224; replicate to 3 channels | §IV-D-2, p. 27887 (E-14) | Evaluation only (Inception V3 input) |

### 3.5 Model-input representation (final output of preprocessing)

128 × 128 × 1, grayscale, intensities in [0, 1], whistle regions close to 1, almost-empty background, centre frequency of every whistle normalised to the image centre (E-06, E-07, E-08, E-12, E-15). The VAE input layer is 128×128×1 and the decoder output uses a sigmoid (Table 7, p. 27890), consistent with a [0, 1] range.

### 3.6 Secondary evidence: the MCA-CFAR detector in [R32]

The paper cites [32] only as part of a post-processing step that turns detection outputs into individual whistle tracks (E-05). [R32] describes:

* A **1-D MCA-CFAR detector along the frequency axis** (reference cells along frequency so that broadband clicks are treated as noise).
* The noise estimate discards large-magnitude reference cells via a **step-size**.
* **Two MCA-CFAR thresholds** with step-sizes d₁, d₂ are computed and combined by a per-cell minimum: **T(n) = min{T₁(n), T₂(n)}**.
* Demonstration values (on a *different* NOAA dataset, DOI 10.25921/PF0H-SQ72, not the DCLDE 2022 DOI 10.25921/e12p-gj65 used by [P]):

| Detector in [R32] Figs. 1–2 | Reference cells | Guard cells | Step-size | P_fa |
|---|---|---|---|---|
| CA-CFAR (baseline) | 22 | 6 | — | 10⁻⁶ |
| MCA-CFAR (baseline) | 24 | not given | 14 | not given |
| Proposed (two-threshold MCA-CFAR) | (24, 82) | not given | (14, 66) | not given |

**Status of these values:** EXPLICIT *in [R32]*; **applicability to [P]'s pipeline is UNCONFIRMED.** [P] does not state that these values, or the detector at all as a stand-alone stage, were used; it does not give the STFT settings under which [R32]'s values were tuned; and [R32] does not give its own STFT settings, threshold scale factor, or P_fa for the proposed detector. See A-07, D-05.

---

## 4. Detailed parameter table

Evidence column format: **CLASS** — evidence ID(s) (see §6). "—" in *Unit* means not applicable.

| Stage | Operation | Parameter | Value | Unit | Evidence |
|------|-----------|-----------|-------|------|----------|
| S0 | Source data | Dataset | NOAA DCLDE 2022 raw passive acoustic data (HICEAS 2017) | — | **EXPLICIT** — E-01 |
| S0 | File selection | Target | Files containing false killer whale whistles | — | **EXPLICIT** — E-02 |
| S0 | File selection | Number of files | 2,486 | files | **EXPLICIT** — E-02 |
| S0 | File selection | Criterion/procedure by which a file "contains whistles" | — | — | **UNSPECIFIED** |
| S0 | Audio input | File duration | ≈ 60 ("generally") | s | **EXPLICIT** — E-02 |
| S0 | Audio input | Hydrophones (channels) per file | 6 (towed array) | channels | **EXPLICIT** — E-02 |
| S0 | Audio input | Native sampling rate | 500 | kHz | **EXPLICIT** — E-02 |
| S0 | Audio input | File container / codec | — | — | **UNSPECIFIED** (paper silent; our files are FLAC, §7) |
| S0 | Audio input | Bit depth / integer→float scaling | — | — | **UNSPECIFIED** (paper silent; our files are int16) |
| S1 | Channel handling | Which channel(s) feed the spectrogram | — | — | **UNSPECIFIED** — E-17 |
| S1 | Channel handling | Combination across channels (select one / sum / average / beamform / process each) | — | — | **UNSPECIFIED** — E-17 |
| S1 | Channel handling | Removal of duplicate whistles seen on several channels | — | — | **UNSPECIFIED** — E-17 |
| S2 | Resampling | Target sampling rate | 192 | kHz | **EXPLICIT** — E-03 |
| S2 | Resampling | Rate ratio | 192/500 = 48/125 | — | **IMPLIED** — E-02, E-03 (arithmetic) |
| S2 | Resampling | Nyquist frequency after resampling | 96 | kHz | **IMPLIED** — E-03 (192/2) |
| S2 | Resampling | Algorithm, anti-alias filter, delay compensation | — | — | **UNSPECIFIED** |
| S3 | Filtering | Operation | High-pass filter, stated purpose: suppress low-frequency ocean noise | — | **EXPLICIT** — E-04 |
| S3 | Filtering | Cut-off / passband start | 2 (passband above 2 kHz) | kHz | **EXPLICIT** — E-04 |
| S3 | Filtering | Meaning of "2 kHz" (−3 dB point, −6 dB point, stopband edge, …) | — | — | **UNSPECIFIED** |
| S3 | Filtering | Filter family (FIR/IIR), order, ripple, stop-band attenuation, zero-phase vs causal | — | — | **UNSPECIFIED** |
| S3 | Filtering | Sampling rate at which it is applied (500 vs 192 kHz) and order relative to S2 and S4 | — | — | **UNSPECIFIED** |
| S4 | STFT | Window type | Hamming | — | **EXPLICIT** — E-03 |
| S4 | STFT | Window length | 2,048 (at 192 kHz) | samples | **EXPLICIT** — E-03 |
| S4 | STFT | Window duration | 2,048/192,000 = 10.667 | ms | **IMPLIED** — E-03 |
| S4 | STFT | Overlap | 50 | % | **EXPLICIT** — E-03 |
| S4 | STFT | Hop size | 1,024 | samples | **IMPLIED** — E-03 (50 % of 2,048) |
| S4 | STFT | Hop duration | 1,024/192,000 = 5.333 | ms | **IMPLIED** — E-03 |
| S4 | STFT | Time resolution (as stated) | ≈ 5.3 (equals hop duration, not window duration) | ms | **EXPLICIT** value; interpretation **IMPLIED** — E-03 |
| S4 | STFT | Frequency resolution | 93.75 | Hz | **EXPLICIT** — E-03 |
| S4 | STFT | FFT size | 2,048 (192,000/2,048 = 93.75 Hz ⇒ bin spacing; no zero-padding) | points | **IMPLIED** — E-03 |
| S4 | STFT | Window variant (symmetric vs periodic), window normalisation | — | — | **UNSPECIFIED** |
| S4 | STFT | Frame alignment / edge padding / first-frame time | — | — | **UNSPECIFIED** |
| S4 | STFT | Spectrogram quantity (magnitude, power, dB), one-sided vs two-sided | — | — | **UNSPECIFIED** (Fig. 3 shows a colour-mapped spectrogram; scale not stated) |
| S4 | Frequency range | Lower bound for analysis | 2 (only via the high-pass) | kHz | **IMPLIED** — E-04 |
| S4 | Frequency range | Upper bound for analysis | — (up to 96 kHz available; Fig. 3 *displays* ≈ 2–23 kHz) | — | **UNSPECIFIED** — E-10 |
| S4 | Noise / background | Any background estimation, whitening, per-band normalisation, click suppression applied to the spectrogram | none described | — | **UNSPECIFIED** (not described) — E-17 |
| S4 | Software | Tool used for preprocessing | — (MATLAB R2024a is stated only for VAE training) | — | **UNSPECIFIED** — E-13 |
| S5 | Whistle detection | Method | Neural-network-based whistle extraction, ref. [31] | — | **EXPLICIT** (citation only) — E-05 |
| S5 | Whistle detection | Architecture, weights, training data, input representation, output form, decision threshold | — | — | **UNSPECIFIED** ([31] not in project) |
| S5 | Peak detection | Local-maximum / peak picking | — | — | **UNSPECIFIED** |
| S5 | Thresholds | Any detection threshold for this pipeline | — | — | **UNSPECIFIED** |
| S6 | Post-processing | Method that converts detection outputs into individual whistle tracks | Refs. [32] (MCA-CFAR) and [33] (detection-and-tracking) | — | **EXPLICIT** (citation only) — E-05 |
| S6 | Post-processing | How [32]/[33] are combined with [31]'s output | — | — | **UNSPECIFIED** |
| S6 | CFAR parameters | Reference cells, guard cells, step-size(s), P_fa / threshold scale *as used for this paper* | — | — | **UNSPECIFIED** (values in [R32] §3.6 are not confirmed for [P]) — E-16 |
| S6 | Contour linking / tracking | Association rule, gap tolerance, minimum track length, merging/splitting of crossing whistles | — | — | **UNSPECIFIED** ([33] not in project) |
| S7 | Contour extraction | Output unit | One time–frequency vector per individual whistle | — | **EXPLICIT** — E-06 |
| S7 | Contour extraction | Vectors obtained | 228,673 (from 2,486 files) | vectors | **EXPLICIT** — E-06 |
| S7 | Contour extraction | Vector time step | = STFT frame step (5.333 ms); "length" counted in samples | ms | **IMPLIED** — E-09, E-11 (weak: figure axes and table units) |
| S7 | Contour extraction | Frequency precision of vector (bin-quantised vs sub-bin) | — | — | **UNSPECIFIED** |
| S7 | Quality control | Minimum/maximum duration, bandwidth, SNR; any rejection rule | none described | — | **UNSPECIFIED** (not described) — E-17 |
| S7 | Segmentation / cropping | Handling of whistles longer than the 128-column image (> 128 frames ≈ 682.7 ms) | — | — | **UNSPECIFIED** (Table 3 shows real whistles up to ≥ 108 samples — E-11) |
| S7 | Segmentation / cropping | Segmentation of the 60-s audio | none described (whole file processed) | — | **UNSPECIFIED** (not described) |
| S8 | Centre-frequency removal | Operation | Whistle expressed as baseband contour with centre frequency removed | — | **EXPLICIT** — E-07, E-08 |
| S8 | Centre-frequency removal | Result | Centre frequency of every whistle normalised to image centre | — | **EXPLICIT** — E-08 |
| S8 | Centre-frequency removal | Definition of the centre frequency f_c (mean, midpoint of f_min/f_max, median, …) | — | — | **UNSPECIFIED** |
| S8 | Centring in time | Whistle placed at horizontal centre of image | — | — | **IMPLIED** (visual only: Figs. 2, 3; text silent) — E-09, E-10 |
| S8 | Baseband signal | Signal model | Constant-amplitude nonlinear-FM (NFM) baseband signal | — | **EXPLICIT** — E-06 |
| S8 | Baseband signal | Sampling rate, interpolation of the vector to signal rate, phase integration, real vs complex signal | — | — | **UNSPECIFIED** (Fig. 2 shows a two-sided ±6 kHz axis with one contour, but the paper does not say real vs complex) |
| S9 | Spectrogram / image | Source of image | Spectrogram of the NFM baseband signal | — | **EXPLICIT** — E-06 |
| S9 | Spectrogram / image | Window, hop, FFT size, magnitude scale of *this* second spectrogram | — | — | **UNSPECIFIED** (axes of Fig. 2 are consistent with the S4 grid, 93.75 Hz × 5.333 ms — **IMPLIED, weak**) — E-09 |
| S9 | Image dimensions | Height × width | 128 × 128 | pixels | **EXPLICIT** — E-06, E-12 |
| S9 | Image dimensions | Channels | 1 (grayscale) | — | **EXPLICIT** — E-06, E-12 |
| S9 | Image grid ↔ physics | Frequency extent of 128 rows | 128 × 93.75 = 12.0 (≈ ±6) | kHz | **IMPLIED** — E-09 (Fig. 2 axis ≈ −6…+6 kHz) |
| S9 | Image grid ↔ physics | Time extent of 128 columns | 128 × 5.333 = 682.7 (≈ ±0.34) | ms (s) | **IMPLIED** — E-09 (Fig. 2 axis ≈ −0.34…+0.34 s) |
| S9 | Image grid ↔ physics | Features measured in "bin" (frequency) and "sample" (time) units | 1 bin = 1 image row, 1 sample = 1 image column | — | **IMPLIED** — E-11 |
| S9 | Spectrogram → 128×128 | Direct STFT grid vs crop vs resize / interpolation | — | — | **UNSPECIFIED** |
| S9 | Normalisation | Value range | 0 to 1 | — | **EXPLICIT** — E-07 |
| S9 | Normalisation | Whistle region value / background value | ≈ 1 / ≈ 0 ("sparse": majority of pixels are zero) | — | **EXPLICIT** — E-07, E-15 |
| S9 | Normalisation | Normalisation rule (per-image max, fixed reference, dB mapping, clipping) | — | — | **UNSPECIFIED** |
| S9 | Image orientation | Time on horizontal axis, frequency on vertical axis, low frequency at bottom in plots | — | — | **IMPLIED** — E-09 (array row order **UNSPECIFIED**) |
| S9 | Image storage | Data type (uint8 / float), file format, on-disk layout | — | — | **UNSPECIFIED** |
| Input | Final model input | Tensor shape | 128 × 128 × 1 | — | **EXPLICIT** — E-12 |
| Input | Final model input | Output activation of decoder | sigmoid (⇒ [0, 1]) | — | **EXPLICIT** — E-12 |
| Input | Dataset split | Train / validation / test partition | — | — | **UNSPECIFIED** — E-17 |
| Training (not preprocessing) | Augmentation | Random rotation | 0°, 90°, 180°, 270°, equal probability | degrees | **EXPLICIT** — E-13 |

---

## 5. Input / output of every stage

### 5.1 Stage inputs and outputs

| Stage | Input | Output | Status |
|---|---|---|---|
| **S0 Acquisition** | One `.flac` file (60 s, 6 ch, 500 kHz, int16) | Sample array, shape (30,000,000 × 6) in our data | Defined; amplitude scaling UNSPECIFIED |
| **S1 Channel handling** | S0 array (6 channels) | One or more single-channel signals | **UNSPECIFIED** |
| **S2 Resample** | Signal @ 500 kHz | Signal @ 192 kHz (60 s → 11,520,000 samples per channel; Nyquist 96 kHz) | Rate EXPLICIT; method UNSPECIFIED |
| **S3 High-pass** | Signal | Signal with energy below 2 kHz suppressed | Cut-off EXPLICIT; design UNSPECIFIED |
| **S4 STFT** | Filtered/resampled signal | Spectrogram on a 93.75 Hz × 5.33 ms grid | Window/overlap EXPLICIT; scaling UNSPECIFIED |
| **S5 Detection** | Spectrogram (representation fed to the network UNSPECIFIED) | Detection output (form UNSPECIFIED) | **Method cited only [31]** |
| **S6 Post-processing** | Detection output | Individual whistle *tracks* | **Methods cited only [32],[33]** |
| **S7 Contour** | Tracks | One time–frequency vector per whistle | Defined by name only |
| **S8 Baseband NFM** | One time–frequency vector | Constant-amplitude NFM baseband signal (centre frequency removed) | Concept EXPLICIT; construction UNSPECIFIED |
| **S9 Image** | Baseband NFM signal | 128 × 128 × 1 image, values 0–1, whistle ≈ 1, background ≈ 0 | Size/range EXPLICIT; construction UNSPECIFIED |

### 5.2 Parameters used downstream that constrain the image definition

These are not preprocessing parameters but determine how a preprocessed image must be interpretable:

| Item | Value | Evidence |
|---|---|---|
| Binarisation of a generated image to recover a contour | pixel > 0.05 | **EXPLICIT** — E-09 |
| Binarisation for FID₁ (evaluation) | pixel > 0.1 | **EXPLICIT** — E-14 |
| Frequency features (f_min, f_max, Δf, f_start, f_end) | measured in image-row ("bin") index (128 rows); whistle centre frequency normalised to the image centre | **IMPLIED** — E-08, E-11 |
| Duration feature L | measured in image-column ("sample") count | **IMPLIED** — E-11 |

---

## 6. Paper evidence for each important parameter

All items are paraphrased from [P]. "p." = journal page; the PDF page is in parentheses.

| ID | Location | Evidence (paraphrased) |
|---|---|---|
| **E-01** | §II-C, p. 27881 (PDF 5) | Training data come from the NOAA DCLDE 2022 raw passive acoustic data (ref. [29]): recordings from a multichannel towed hydrophone array during the HICEAS survey, July–November 2017. |
| **E-02** | §II-C, p. 27881 | The authors selected 2,486 files containing false killer whale whistles. A file generally holds 60 s of signal from six hydrophones in a towed array; sampling rate is 500 kHz. |
| **E-03** | §II-C, p. 27881 | The spectrogram is generated after resampling to 192 kHz, using a 2,048-point Hamming window and 50 % overlap. The resulting time and frequency resolutions are about 5.3 ms and 93.75 Hz. (Cross-check: 192,000/2,048 = 93.75 Hz; 1,024/192,000 = 5.333 ms.) |
| **E-04** | §II-C, p. 27881 | To suppress low-frequency ocean noise a high-pass filter with a passband above 2 kHz was applied. No filter design details are given. |
| **E-05** | §II-C, p. 27881 | A neural-network whistle-extraction method (ref. [31]) was applied to detect individual whistle contours; the extraction was combined with a post-processing algorithm (refs. [32], [33]) that converts detection outputs into individual whistle tracks. Per the reference list, [31] and [33] are 2024 conference papers (KICS and KIMST, South Korea) that are not supplied in the project; [32] is supplied as [R32]. |
| **E-06** | §II-C, p. 27881; Fig. 3 | The extracted time–frequency vectors were turned into individual images: 228,673 vectors from the 2,486 files. Each vector generated a constant-amplitude NFM baseband signal and its spectrogram, which was further transformed into a 128 × 128 grayscale image. The Fig. 3 caption describes the same chain (vector → constant-amplitude NLFM signal → spectrogram → image) and states that the centre frequency is removed in the resulting image. |
| **E-07** | §II-B p. 27881; Fig. 1 caption p. 27880 | Each whistle is a single image of a baseband contour with the centre frequency removed. Data used are normalised to intensities between 0 and 1, whistle regions close to 1 (Fig. 1 is displayed with inverted intensity for visibility only). |
| **E-08** | §IV-D-4 p. 27888; Appendix C p. 27893; §V p. 27892 | The centre frequency of all whistles is normalised to the centre of the image during dataset preprocessing; therefore it is excluded from conditioning. The model learns only the contour shape with centre frequency removed; training with the centre frequency retained (contour free to move vertically) was found ineffective. |
| **E-09** | Fig. 2 and caption, p. 27880 | A whistle image shown with axes of roughly −0.34…+0.34 s (time) and −6…+6 kHz (frequency); a simple 0.05 threshold gives the contour (panel b), a time–frequency vector (c), then a constant-amplitude NFM signal (d). The generated whistle sits near the horizontal centre; pixel blocks are visible. *The ±0.34 s and ±6 kHz extents are read visually from the plot axes; they agree with 128 × 5.333 ms = 682.7 ms and 128 × 93.75 Hz = 12 kHz.* |
| **E-10** | Fig. 3, p. 27881 | Illustration of the dataset-building chain: raw signal → spectrogram (a 3-s excerpt, 42–45 s, displayed ≈ 2–23 kHz) → colour-coded whistle contours → six 128 × 128 images with white contour on black, contours near the image centre. The display range is a plotting choice; the text does not state it as an analysis limit. |
| **E-11** | Table 3 & §IV-D-4, p. 27888; Figs. 13–14, p. 27891 | Whistle duration conditions are given in *samples* (30–40, 75–80, 88–108), with errors in ms; frequency conditions are given in *bin* index (e.g. f_min 34–55, f_max 62–71 and 80–90, Δf 10–30, f_start/f_end 40–80, bin indices up to ≈ 94), with errors in Hz. Features were obtained by converting each image into a time–frequency representation (as in Fig. 2c). The conversion factors from sample→ms and bin→Hz are not stated. |
| **E-12** | Tables 7 and 8, p. 27890 | VAE and critic input size 128 × 128 × 1; VAE output layer uses a sigmoid. |
| **E-13** | Appendix A, p. 27892 | VAE training used MATLAB R2024a on seven V100 GPUs. During training each image is randomly rotated by 0°, 90°, 180° or 270° with equal probability (to avoid over-focus on horizontal lines). |
| **E-14** | §IV-D-2, p. 27887 | FID₁ binarises real and generated images at 0.1 before Inception V3; FID₂ feeds the grayscale images directly. 128 × 128 images are resized to 224 × 224 by bicubic interpolation and the channel replicated to 3. Evaluation only. |
| **E-15** | §IV-D-6, p. 27889 | The whistle image data are sparse; the majority of pixel values are zero. |
| **E-16** | [R32] §II–III | See §3.6. Describes 1-D MCA-CFAR along frequency, two step-sizes, T(n) = min{T₁(n), T₂(n)}; demonstration values for a different NOAA dataset. |
| **E-17** | Whole paper (full read) | **Absence evidence.** The paper contains no statement on: which of the six channels (or how channels are combined) is used; the upper analysis frequency; background/noise normalisation; click handling; per-file or per-whistle rejection rules; the definition of centre frequency; how images are normalised to [0, 1]; the parameters of the second (baseband) spectrogram; the train/validation/test split. |

---

## 7. Dataset compatibility observations

All measurements below were made read-only on `encounter_191_audio/`; no raw file was opened for writing, renamed, moved, converted, or modified.

### 7.1 Measured facts [INS]

| Check | Result |
|---|---|
| Files present | **363** (all `.flac`; no other file types, no sub-folders) |
| Container / codec | FLAC, PCM_16 (16-bit) — 363/363 |
| Sampling rate | 500,000 Hz — 363/363 |
| Channels | 6 — 363/363 |
| Frames per file | 30,000,000 — 363/363 |
| Duration per file | 60.000 s — 363/363 (total 21,780 s = 6.05 h) |
| File size | 189.2–200.7 MB each (mean 189.9 MB), **68.9 GB total** (64.2 GiB) |
| File-name pattern | `1705_YYYYMMDD_HHMMSS_fff.flac` — 363/363 conform; all prefix `1705` (R/V Lasker) |
| Time span (UTC) | First file starts 2017-09-29 22:50:47; last file starts 2017-09-30 04:52:47 (ends 04:53:47) |
| Continuity | **0 gaps**: every file starts exactly 60 s after the previous one (ms suffix is `062` for 360 files and `063` for 3) |
| Unreadable / corrupted | **None found.** Every file opened and was decoded **in full** (read-only): 363/363 decoded to exactly 30,000,000 frames × 6 channels with no decoder error. (The SHA-256 values in the manifest were not recomputed.) |
| Manifest cross-check | `manifest_363.csv` (363 rows, all `VALID`) matches the live files on file list, size, sample rate, channel count and frame count — 0 mismatches |
| Silent files / dead channels | None. Across all 363 files × 6 channels the smallest per-channel peak is 4,365 int16 counts; no silent file and no dead channel |
| Full-scale (clipping) samples | 85 of 363 files contain at least one int16 full-scale sample (absolute value ≥ 32,767); 771 samples in total (≈ 1.2 × 10⁻⁸ of all samples). Worst file: `1705_20170930_014847_062.flac` (359 samples). Not a compatibility problem; recorded because the paper does not mention clipping and describes no handling for it (none is to be added — D-04) |

### 7.2 Compatibility with the paper's pipeline

| Pipeline requirement from [P] | Our data | Verdict |
|---|---|---|
| 500 kHz native rate | 500 kHz | ✅ compatible |
| 6 channels per file | 6 | ✅ compatible (channel handling remains UNSPECIFIED by the paper) |
| ≈ 60-s files | exactly 60 s | ✅ compatible |
| Resample to 192 kHz | Rate ratio 48/125 is an integer-ratio conversion; Nyquist 96 kHz < original 250 kHz | ✅ applicable (algorithm UNSPECIFIED) |
| High-pass above 2 kHz | The hydrophones already have a hardware high-pass at 1.6 kHz [DS] (−10 dB at 1 kHz), below the paper's 2 kHz edge | ✅ applicable; no conflict found |
| Container/format | Paper silent; FLAC is a lossless variant that the data provider publishes alongside the original 16-bit WAV [DS] | ✅ no incompatibility found |
| Species | False killer whale (encounter 191) | ✅ matches |

**No incompatibility between our files and the paper's preprocessing specification was found.**

### 7.3 Differences from the paper's dataset (documented, not corrected)

1. **File count:** 363 vs 2,486 (§2.2). Size difference only.
2. **Selection basis:** the paper picked files containing false killer whale whistles; ours are all consecutive files of one encounter, so some files may contain few or no whistles. The paper gives no criterion for judging this, so none is applied (D-11). `audio_activity_scores.csv` labels 42 of the 363 files "LOW (Ambient)" by an undocumented method; this is not paper evidence.
3. **Coverage:** one vessel (1705), one encounter (191), ≈ 6 h contiguous; the paper (inferred, §2.3) used 13 encounters from two vessels.
4. **Encounter start:** the encounter began 22:49:48 UTC but the first local file starts 22:50:47; no `22:49:xx` file is present in the local set. Whether such a file exists in the source archive was not checked.
5. **Reference counts:** the paper's 228,673 vectors cannot be compared to ours until the pipeline exists.

---

## 8. Ambiguities and missing information in the paper

Each item states what is missing and why it matters. None has been resolved by assumption.

| ID | Stage | Missing information | Impact |
|---|---|---|---|
| **A-01** | S1 | Which hydrophone channel(s) are used, or how six channels are combined; whether whistles are extracted per channel and de-duplicated. | Changes both the signal and the number of whistle vectors per file. **Blocking.** |
| **A-02** | S0/S2 | Integer→float scaling; resampler algorithm; anti-alias filter; delay handling. | Affects absolute levels and spectral edge behaviour. |
| **A-03** | S3 | Filter family, order, corner definition, phase behaviour, sampling rate at which it runs. | Affects noise floor and low-frequency whistle onset near 2 kHz. |
| **A-04** | S3/S4 | Order of high-pass relative to resampling and to the STFT (the paper's text lists it after the spectrogram sentence but does not state an order). | Different orders give different edge behaviour. |
| **A-05** | S4 | Spectrogram quantity (magnitude/power/dB), window variant, normalisation, padding, one-sided spectrum. | Determines the detector input. |
| **A-06** | S4 | Upper analysis-frequency bound (only the 2 kHz lower edge is given; Fig. 3 displays ≈ 2–23 kHz, which is a plot range). | Could change which whistles are retained. |
| **A-07** | S5–S6 | **The detector is a cited neural network [31] plus cited post-processing [32],[33].** The network (architecture, weights, training data), its input/output form, thresholds, and the tracking rules are not given; [31] and [33] are not in the project. [R32] gives a CFAR design but not its settings for this study, and it is unclear whether MCA-CFAR is applied to the spectrogram, to the network output, or as a track-forming stage. | **Blocking.** The paper's core extraction step cannot be reproduced from the supplied material. |
| **A-08** | S6 | Contour-linking rules: association metric, maximum gap, minimum duration, handling of overlapping/crossing whistles and harmonics. | Defines what counts as "one whistle". |
| **A-09** | S7 | Quality-control / rejection rules (duration, bandwidth, SNR, minimum track length). None are described. | Affects dataset content. |
| **A-10** | S7/S9 | Handling of whistles longer than 128 frames (≈ 682.7 ms): crop, reject, or rescale. Table 3 shows real whistles up to at least 108 samples. | Determines dataset content for long whistles. |
| **A-11** | S8 | Definition of the centre frequency f_c used for removal (mean, mid-range, median, weighted) and exact vertical placement (which row is "the centre"). | Directly determines the vertical alignment of every image. |
| **A-12** | S8 | Time centring (inferred from figures only) and exact column placement for even-length contours. | Alignment of each image. |
| **A-13** | S8 | How a frame-rate time–frequency vector is turned into a continuous constant-amplitude NFM signal (signal sampling rate, interpolation, phase accumulation, real vs complex). | Determines the spectral spread (line width) of the contour in the image. |
| **A-14** | S9 | Parameters of the *second* spectrogram (window, hop, FFT, scale) and whether the image is a direct STFT grid, a crop, or a resize. (Fig. 2 axes are consistent with the S4 grid, but this is not stated.) | Determines pixel-level image appearance. |
| **A-15** | S9 | Normalisation rule that produces [0, 1] with whistle ≈ 1 (per-image max, fixed scale, clipping, dB). | Determines intensity statistics the VAE sees. |
| **A-16** | S9 | Image dtype, orientation in the array (row 0 = high or low frequency), storage format. | Needed for consistent encoding/decoding. |
| **A-17** | S0 | Criterion for deciding that a file contains whistles (basis of the 2,486-file selection). | Dataset-selection comparability. |
| **A-18** | Dataset | No train/validation/test split is described for the 228,673 images. | Outside preprocessing but needed before training/evaluation. |
| **A-19** | Table 3 | Conversion factors between the "sample"/"bin" units and ms/Hz are not stated (the paper's ms and Hz errors are consistent with 5.33 ms/sample and 93.75 Hz/bin but this is not confirmed). | Only affects later feature analysis. |
| **A-20** | Refs | Ref. [30] (Li et al., "Learning deep models from synthetic data for extracting dolphin whistle contours", arXiv:2005.08894) appears in the reference list but is not cited in the body text. It is topically related to [31]; the paper never links them. | Not to be assumed to be the detector. **Update (Revision 2):** a later project brief equated this paper with ref. [31]; that is a contradiction with [P]'s reference list, documented (not resolved) in §11.0. |

---

## 9. Decisions that must NOT be made without evidence

The following must not be chosen, changed, or "improved" without paper evidence or an explicit, recorded decision by the project owner. Items D-01–D-04 are **fixed by the paper** (EXPLICIT, or IMPLIED where marked in §4) and must not be altered; the remainder are **unsupported by the paper** and must not be silently filled in by code.

### 9.1 Fixed by the paper — do not change

| ID | Rule |
|---|---|
| **D-01** | Do not change the resampling target (192 kHz), window (Hamming, 2,048 samples), overlap (50 %), FFT size (2,048) or high-pass edge (2 kHz) — **including because the dataset is smaller**. |
| **D-02** | Do not restrict or widen the frequency range, add a band-pass, or change the 2 kHz edge because of dataset size or convenience. |
| **D-03** | Do not change the image size (128 × 128), channel count (1) or value range ([0, 1]). |
| **D-04** | Do not add preprocessing the paper does not describe (e.g. denoising, spectral whitening, click removal, data augmentation in the dataset, mirror/flip copies, SNR filters, resolution changes). |

### 9.2 Unsupported by the paper — not to be guessed

| ID | Rule |
|---|---|
| **D-05** | Do not assume the [R32] CFAR values (22/6/10⁻⁶, 24/14, (24,82)/(14,66)) are the settings used by [P]. They were demonstrated on a different dataset with unknown STFT settings and unknown P_fa/scale for the proposed detector. |
| **D-06** | Do not substitute a different whistle detector/tracker and call it a faithful reproduction. Any substitute is a **documented deviation** to be decided explicitly (§10). |
| **D-07** | Do not choose a channel, or combine channels, without an explicit recorded decision (A-01). |
| **D-08** | Do not pick the high-pass design, resampler, spectrogram scaling, or normalisation rule as if they were paper-derived. If a choice is unavoidable it must be recorded as an *implementation assumption* with its rationale. |
| **D-09** | Do not define f_c, time-centring, long-whistle handling, or the NFM construction by convention without recording it as an assumption (A-10 to A-13). |
| **D-10** | Do not tune thresholds or detector settings to hit the paper's yield (228,673 vectors / ≈ 91.98 per file) or to fit 363 files. |
| **D-11** | Do not use `audio_activity_scores.csv` (or any SNR / click-factor / "activity" label) to include, exclude, or weight files; the paper uses no such rule. |
| **D-12** | Do not infer a train/validation/test split from anything; the paper gives none (A-18). |
| **D-13** | Do not modify, rename, move, convert, delete, or overwrite anything in `encounter_191_audio/`. All derived data goes elsewhere. |

---

## 10. READY FOR IMPLEMENTATION

**Overall status: NOT READY to implement the full pipeline.** Stages S0, S2, S3, S4 and the image-format end of the pipeline (S9) are specified well enough to code. The detection/tracking core (S5–S6) and several construction details (S1, S8, S9) are not specified and require decisions first.

### 10.1 Confirmed preprocessing stages

| Stage | Status |
|---|---|
| S0 Load 6-ch 60-s files @ 500 kHz | Confirmed (dataset verified) |
| S2 Resample 500 → 192 kHz | Confirmed (method open) |
| S3 High-pass, passband above 2 kHz | Confirmed (design and order open) |
| S4 STFT, Hamming 2,048, 50 % overlap | Confirmed (scaling open) |
| S5 Neural-network whistle detection [31] | **Stage exists; not implementable from supplied material** |
| S6 Post-processing to individual tracks [32],[33] | **Stage exists; not implementable from supplied material** |
| S7 One time–frequency vector per whistle | Confirmed as output |
| S8 Centre-frequency removal + constant-amplitude NFM baseband signal | Confirmed concept; construction open |
| S9 Spectrogram → 128 × 128 grayscale image in [0, 1] | Confirmed format; construction open |

### 10.2 Confirmed parameters

**EXPLICIT:** raw rate 500 kHz; 6 channels; ≈ 60 s files; resample to 192 kHz; Hamming window, 2,048 points; 50 % overlap; frequency resolution 93.75 Hz; time resolution ≈ 5.3 ms; high-pass passband above 2 kHz; image 128 × 128 × 1 with intensities in [0, 1] (whistle ≈ 1); centre frequency normalised to image centre; 228,673 vectors from 2,486 files; contour threshold 0.05 (generated images).

**IMPLIED (derived):** FFT size 2,048; hop 1,024 samples (5.333 ms); window 10.667 ms; rate ratio 48/125; Nyquist 96 kHz; 128 columns ≈ 682.7 ms and 128 rows ≈ 12 kHz (≈ ±6 kHz) as read from Fig. 2; whistle centred in time (figures only); row = "bin", column = "sample".

**Dataset facts (verified):** 363 FLAC, 16-bit, 500 kHz, 6 ch, 60.000 s, contiguous, same source and species as the paper; no incompatibility found.

### 10.3 Unresolved ambiguities

A-01 … A-20 (§8). The blocking ones are **A-01** (channel handling), **A-07/A-08** (detection and tracking), **A-11** (centre-frequency definition), **A-13/A-14/A-15** (how the NFM signal and its spectrogram become a normalised 128×128 image).

### 10.4 Information that must be decided before coding

| # | Needed before coding | Options (not recommendations) |
|---|---|---|
| 1 | **Whistle detector/tracker (S5–S6).** The paper's method relies on [31] and [33]. | (a) Obtain [31]/[33] (or the authors' code/weights) and reproduce faithfully; (b) adopt a substitute and record it as a **documented deviation from the paper**. |
| 2 | **Channel handling (S1).** | Record the chosen policy and its justification as an assumption; the paper offers none. |
| 3 | **Definitions that the paper leaves open (S8–S9):** f_c, time/row/column placement, NFM construction, second-spectrogram settings, [0, 1] normalisation, handling of whistles > 128 frames. | Each must be written down as an explicit assumption before any code is generated. |
| 4 | **Design choices of generic DSP operations (S2–S4):** resampler, high-pass design/order, spectrogram scale. | Record as implementation assumptions; they are not paper-derived. |
| 5 | **Treatment of [R32].** | Decide whether it is a permitted source of parameters (default per this document: no — D-05). |
| 6 | **Train/validation/test split policy (later stage).** | Decide separately; outside this document's scope. |
| 7 | **Output location for derived data.** | Must be outside `encounter_191_audio/`. |

*No code was written, no raw data modified, and no report other than this file was created in the project.*

---

## 11. Reference Detector Analysis and Replacement Requirements

**Scope and rules.** Analysis only. No neural network is designed, proposed, trained or implemented here, and no architecture is recommended. §11.0–§11.7 report what the sources say about the **original** method, as far as it can be known. §11.8, §11.9 and the closing box concern a **future replacement** and are labelled as such, so the paper's method and our proposed replacement are never conflated. Evidence labels are those of §0.1 (EXPLICIT / IMPLIED / UNSPECIFIED); every statement carries its source tag. "Applicability to [31]/[P]" is **UNKNOWN** for every [L30] statement unless stated otherwise.

### 11.0 Source identification, access status, and one contradiction (read first)

| Item | Finding |
|---|---|
| **Contradiction** | The task brief named *"Learning Deep Models from Synthetic Data for Extracting Dolphin Whistle Contours"* as reference [31]. In [P]'s reference list that title is **[30]** (P. Li et al., arXiv:2005.08894; IJCNN 2020). Reference **[31]** is Park, Ahn, Kim, Kim, Lee, *"Implementation of a neural network for cetacean whistle frequency detection based on data synthesis"*, KICS Winter Conf., Gangwon, 2024, pp. 690–691 (2 pages). [P]'s body text cites [31] (§II-C) but **never cites [30]** (see A-20). |
| **Treatment** | **Not resolved by assumption.** The two are kept separate throughout: **[L30]** = Li et al. (obtained, read in full); **[31]** = Park et al. (not obtained). Whether [31] builds on, follows, or is unrelated to [L30] is **UNSPECIFIED** (shared theme "synthetic/synthesised training data"; different author groups; no link stated in [P]). |
| **[L30] access** | Obtained from arXiv (PDF, 10 pp.) and read completely. Stored only in the session scratchpad, **not** in the project. |
| **[31] access** | **Not obtained.** English- and Korean-title web searches, the KICS 2024-winter programme page, and the authors' ResearchGate profile (blocked by bot protection; not circumvented) yielded no accessible copy. |
| **[33] access** | **Not obtained** (same attempts). Park, I. S. Kim, S. G. Lee, D. H. Lee, *"Detection and tracking-based dolphin whistle extraction method"*, Korea Inst. Mil. Sci. Technol. Annual Conf., Jeju, 2024, pp. 382–383 (2 pages). |
| **[R32]** | Supplied in the project (§0.2); partial description of a CFAR detector only. |
| **[RW1]** | Kim, Seol, Kim, **Park (G.-H.)**, Chung, *"Multiscale Attention-Based Whistle Segmentation for Biomimetic Communication"*, Electronics 15(10):2147, 2026. **Not cited by [P]** and it cites neither [31] nor [33]. Found while searching. Same data family (DCLDE 2022, false killer whale). Used in this section **only** as non-evidence context (§11.6c). |

**Consequence.** Everything in §11.1.2 describes [L30]. It must not be read as a description of [31].

### 11.1 Reference [31] analysis

#### 11.1.1 What is actually known about [31] itself

| Fact | Label | Source |
|---|---|---|
| Bibliographic data: title, authors, venue, year (2024), pages 690–691 | **EXPLICIT** | [P] reference list |
| [P] applied "a neural network-based whistle extraction method [31]" to detect individual whistle contours | **EXPLICIT** | [P] §II-C, p. 27881 (E-05) |
| [31] is a neural network that detects whistle frequencies and is trained using synthesised data | **IMPLIED** (from its title only) | [P] reference list |
| Whether [31]'s network output is a 2-D time–frequency map, a per-frame frequency value, or something else | **UNSPECIFIED** (the title says "frequency detection", [L30] says "contour/confidence map") | — |
| Input representation, patch size, channels, normalisation, output shape/meaning, threshold, loss, training data, synthesis recipe, metrics, performance | **UNSPECIFIED** (document not obtained) | — |

#### 11.1.2 [L30] — the 22 requested items (data-synthesis neural whistle detector by Li et al.)

Label applies to [L30]; source location is [L30] section/page. Architecture is deliberately **not** transcribed (only functional properties are recorded).

| # | Item | Finding in [L30] | Label | Loc. |
|---|---|---|---|---|
| 1 | Input to the network | A patch cut from a spectrogram. At test time the whole spectrogram is partitioned into patches and each patch is fed to the network. | **EXPLICIT** | §I p.2; §III p.3; §III-B p.4 |
| 2 | Input representation | Log-magnitude spectrogram; dynamic range limited to floor 0 and ceiling 6 (stated as 0–120 dB rel., uncalibrated, not adjusted for bin width), then normalised to [0, 1]. Unit = 20 dB per count (6 ↔ 120 dB). | **EXPLICIT** (clip, normalise); **IMPLIED** (20 dB/unit; log of magnitude) | §III-A p.4 |
| 3 | Spectrogram / STFT | Audio at 192 kHz (16/24-bit standardised to 16-bit); 8 ms Hamming frames ("125 Hz resolution"); hop 2 ms; band-limited to 5–50 kHz = 361 bins. Derived: 8 ms = 1,536 samples; 2 ms = 384 samples (75 % overlap); FFT 1,536 points (192,000/1,536 = 125 Hz); (50−5)/0.125 + 1 = 361. | **EXPLICIT** (stated values); **IMPLIED** (samples, FFT size, overlap) | §III-A p.3–4 |
| 4 | Input patch dimensions | 100 ms × 6.25 kHz = 50 × 50 time–frequency bins | **EXPLICIT** | §III-A p.4; Fig. 2 |
| 5 | Input channels | Not stated. Only one input quantity (normalised log-magnitude) is described. | **IMPLIED, weak** (one channel); channel count itself **UNSPECIFIED** | — |
| 6 | Patch extraction | Training: spectrogram tiled into patch-sized regions overlapping by 50 %; split into whistle-positive (74,112) and whistle-negative groups; an equal number of negatives chosen at random (148,224 total). Test: non-overlapping patches. Definition of "positive" (minimum whistle content), and handling of partial patches at spectrogram edges: not stated. | **EXPLICIT** (tiling, counts); **UNSPECIFIED** (positive criterion, edges) | §III-A p.4; §III-B p.4 |
| 7 | Overlap / stride | Training 50 % (stride 25 bins); test 0 % (stride 50 bins). | **EXPLICIT** (overlap); **IMPLIED** (stride) | §III-A, §III-B |
| 8 | Input normalisation | Fixed global mapping: clip to [0, 6] then scale to [0, 1]. No per-patch or per-file normalisation is described. Synthetic-blending amplitude α ∈ [0.03, 0.23] corresponds to "24 dB" (0.2 × 120 dB), consistent with 1.0 = 120 dB. Whether the blended patch is re-clipped to [0, 1]: not stated. | **EXPLICIT** (global mapping); **IMPLIED** (no per-patch normalisation; 1.0 = 120 dB); **UNSPECIFIED** (re-clipping) | §III-A, §III-C p.4 |
| 9 | Network output | A pointwise 2-D **confidence map**; each element is the likelihood that the corresponding time–frequency point belongs to a whistle contour. Elsewhere described as predicting "whistle energy" (Fig. 2, §III-B) and, in the abstract, "time–frequency peaks associated with whistles". | **EXPLICIT** | Abstract p.1; §I p.2; Fig. 2 |
| 10 | Output dimensions | 50 × 50 per patch (labels are 50 × 50). Stride 1, zero-padding, no pooling, "predictions on individual time–frequency nodes" ⇒ output same size as input patch (one output channel assumed). Reassembled map has the same size as the original spectrogram. | **EXPLICIT** (design statements, reassembled size); **IMPLIED** (per-patch size, one channel) | §III p.3; §III-B; §III-E p.6 |
| 11 | Meaning of each value | Whistle-contour presence at that node. Training target for synthetic data: binary mask Y (1 at whistle) convolved with a random Gaussian filter G ⇒ a **blurred** contour. For real labelled data (variant WGT) labels are 1 in every node containing a whistle, else 0. Gaussian width/range and whether WGT labels are blurred: not stated. | **EXPLICIT**; **UNSPECIFIED** (G parameters; WGT blurring) | §III-B, §III-C p.4 |
| 12 | Probability / score / mask? | Called a "confidence map" and "the probability that each time–frequency node is part of a whistle contour". Trained with a regression-type Charbonnier loss (not cross-entropy); output activation function and calibration not stated. Values treated as lying between 0 and 1 (thresholds 0–1 are applied). | **EXPLICIT** (terminology); **IMPLIED** (range); **UNSPECIFIED** (activation, calibration) | §III-E p.6; §IV-A p.6–7 |
| 13 | Output thresholding | τ = 0.5 for identifying whistle energy before post-processing; precision–recall of the maps uses 30 thresholds between 0 and 1; the recall-guided sampling indicator also thresholds at 0.5. | **EXPLICIT** | §III-D p.5; §III-E; §IV-A |
| 14 | Combining overlapping predictions | At test time patches do not overlap, so no merging rule is needed or described. The reassembly ("assembled … into a larger map") has no stated blending rule. The 50 % training overlap is a sampling choice, not a prediction-merging scheme. | **IMPLIED** (none needed); **UNSPECIFIED** (any blending) | §I p.2; §III-B |
| 15 | Output → whistle contour | Thresholded map (τ = 0.5) replaces the spectrogram as input to an existing graph-search peak-processing extractor (Roch et al. 2011, [L30]'s ref. 8): at each time step whistle energy initiates a new graph or extends terminating nodes of existing graphs when the new peak lies on a trajectory predicted by a low-order polynomial fit near the node; graphs bridge gaps up to a user-set time interval; a peak added to two graphs merges them; graphs not extended within a set time are closed; at crossing points forward/backward polynomial-fit predictions decide which branches are joined. The peak-picking rule applied to the thresholded map, polynomial order, gap interval and closing time are not given (deferred to ref. 8). | **EXPLICIT** (procedure); **UNSPECIFIED** (peak rule and numeric settings) | §III-E p.6 |
| 16 | Training data | Subset of DCLDE 2011 delphinid data: 20 analyst-annotated sequences (14 training, 6 test); taxa: common dolphins (*Delphinus* spp.) and bottlenose dolphins; 192 kHz source. 7,161 training and 911 test whistle contours. 148,224 training patches (74,112 positive + 74,112 negative). Five variants: **WGT** (all real annotated patches), **EdgeGT** (analyst edges of natural images, BSDS500), **EdgeCanny** (Canny edges of BSDS500 images, no human annotation), **µWGT** (contours of a random 6.25 % = 4,632 positive patches, no audio containing real whistles used), **µWGT-RG** (µWGT + recall-guided learning). | **EXPLICIT** | §III-A p.3–4; §IV-A p.6 |
| 17 | Synthetic-data generation | Whistle-like "primitive shapes" injected into whistle-absent 100 ms × 6.25 kHz patches taken from the data's own whistle-negative patches. Blend: **X′ = X + α(Y∗G)** with X background patch, Y binary mask, G random Gaussian blur, α ~ U[0.03, 0.23]; resulting SNR depends on background and α. Recall-guided adaptive synthesis (µWGT-RG): stage 1 trains on the 6.25 % subset with equal negatives; stage 2, each epoch, computes per-sample recall R(z) of the thresholded (0.5) confidence maps on validation data and resamples positives with probability ∝ (1 − R) and negatives uniformly, synthesising new samples each epoch. | **EXPLICIT**; **UNSPECIFIED** (G parameters; number of synthesised samples per epoch) | §III-C p.4–5; §III-D p.5 |
| 18 | Loss function | Charbonnier loss: sqrt(‖ŷ − y‖₂² + ε), ε = 1×10⁻³. Per-element reduction/weighting not stated. | **EXPLICIT**; **UNSPECIFIED** (reduction, class weighting) | §IV-A p.6–7 |
| 19 | Training procedure | Adam; 600,000 iterations; learning rate 0.001, ×0.1 every 250,000 iterations; Kaiming-normal initialisation; µWGT-RG initialised from the final µWGT network; PyTorch; ≈ 8 h on an NVIDIA GTX 1080 Ti. Batch size, Adam betas/weight decay, validation-based model selection or early stopping: not stated. §III-B says "stochastic gradient descent" (inconsistency, §11.1.3). | **EXPLICIT**; **UNSPECIFIED** (batch size etc.) | §IV-A, §IV-B p.6–7 |
| 20 | Evaluation metrics | (a) Confidence-map quality: BSDS500 benchmark protocol — precision/recall with ground truth thinned to 1 pixel, predictions binarised at 30 thresholds in (0, 1), default tool parameters. (b) Whistle-contour quality after the full extraction system: recall, precision, F1 against analyst contours using the matching rules of ref. 8 — overlap in time; mean frequency deviation < 350 Hz; analyst whistle ≥ 150 ms and SNR ≥ 10 dB over ≥ one third of the whistle; expected-but-short/low-SNR detections discarded; unmatched detections counted as false positives; evaluated at a single operating point (that of ref. 8 / DCLDE 2011). (c) Speed. | **EXPLICIT**; **UNSPECIFIED** (BSDS distance tolerance; definition of the "single operating point") | §IV-A p.6 |
| 21 | Reported performance | Contour extraction on 911 test whistles (354 bottlenose, 557 common dolphin): see table below. Speed: ≈ 10 ms network prediction + 300 ms graph search per 1 s of audio on a GTX 1080 Ti (≈ 3× faster than real time). Map-level PR curves are given as plots only (Fig. 6), no tabulated values. | **EXPLICIT** | Table I p.7; §IV-B p.7 |
| 22 | Assumptions needed by downstream contour extraction | The extractor is **peak-based** and time-sequential: it needs, for every time frame, thresholded whistle nodes at spectrogram-node resolution that it can treat as peaks; whistle energy must be separated from clicks (clicks appear as vertical lines in the input and are absent in the map); crossings must survive in the map so the graph search can resolve them; gaps in detections are tolerated up to a set interval. | **IMPLIED** (from the procedure in #15 and §IV-B discussion) | §III-E; §IV-B p.8 |

**[L30] Table I (911 test whistles) — reported as published:**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Graph-search only (baseline, no network) | 0.634 | 0.633 | 0.634 |
| WGT (all real labels) | 0.956 | 0.822 | 0.884 |
| EdgeGT | 0.913 | 0.638 | 0.751 |
| EdgeCanny | 0.900 | 0.603 | 0.722 |
| µWGT | 0.900 | 0.673 | 0.770 |
| µWGT-RG | 0.924 | 0.693 | 0.792 |

Cross-checks: 0.792 − 0.634 = 0.158 (the "≈ 25 %" of the abstract); 0.884 − 0.634 = 0.250; 0.722 − 0.634 = 0.088 (≈ 14 %). [L30] notes its baseline scores are lower than in ref. 8 because a difficult subset was used.

#### 11.1.3 Internal inconsistencies in [L30] (documented, not resolved)

| ID | Inconsistency |
|---|---|
| I-1 | Filter / receptive-field figures do not agree: §III-B gives filters "no larger than 10 ms by 375 Hz" and a receptive field of 46 ms × 2.875 kHz; §III-E gives a first-layer filter of 50 ms × 3.125 kHz and later filters of "25 ms by 0.625 Hz". With filter size 5 on a 2 ms / 125 Hz grid a first-layer filter would span 10 ms × 625 Hz; 50 ms × 3.125 kHz equals 25 × 25 nodes (the apparent total receptive field). "0.625 Hz" is presumably 0.625 kHz. |
| I-2 | Optimiser: §III-B says stochastic gradient descent; §IV-A says Adam. |
| I-3 | "µWGT-GT" appears in §IV-B but is not defined; the listed variants are WGT, EdgeGT, EdgeCanny, µWGT, µWGT-RG. |
| I-4 | The output is called "likelihood/probability" (§I, §III-E), "whistle energy" (Fig. 2, §III-B) and "time–frequency peaks" (abstract). |

### 11.2 Reference [33] analysis

**Status: the document was not obtained. Every item below is UNSPECIFIED for [33] itself.** The right-hand column gives indirect context and is **not** a description of [33].

| # | Item | [33] | Indirect context (not [33]) |
|---|---|---|---|
| 1 | Input [33] expects | **UNSPECIFIED** | [P] says [32],[33] convert "detection outputs" into individual whistle tracks (E-05) — **IMPLIED**: the input is a detection output of S5 (and/or of [32]); its form is unknown. |
| 2 | Representation | **UNSPECIFIED** | Title ("detection and tracking-based") suggests tracking in the time–frequency plane — **IMPLIED, weak**. |
| 3 | How detections become contours | **UNSPECIFIED** | [L30] uses a graph search with polynomial trajectory prediction (§11.1.2 #15) — an analogue only; no evidence [33] works that way. |
| 4 | Tracking / linking rules | **UNSPECIFIED** | — |
| 5 | Thresholds | **UNSPECIFIED** | [R32] gives CFAR demo values (§3.6) that are not confirmed for [P] (D-05). |
| 6 | Gap handling | **UNSPECIFIED** | [L30]: gaps bridged up to a user-set interval (value not given). |
| 7 | Minimum / maximum track constraints | **UNSPECIFIED** | [P] gives none (A-09). |
| 8 | Overlapping whistles | **UNSPECIFIED** | [R32] states multi-whistle detection as a design goal of its detector (**EXPLICIT [R32]**); [L30] resolves crossings in its extractor. Neither is [33]. |
| 9 | Harmonics / noise | **UNSPECIFIED** | [R32] **EXPLICIT**: reference cells along the frequency axis so broadband clicks are treated as noise; harmonics not mentioned. |
| 10 | Output representation | **EXPLICIT** (via [P]): individual whistle tracks, then one time–frequency vector per whistle (S7) | [P] §II-C (E-05, E-06). |
| 11 | Assumptions about detector output | **UNSPECIFIED** | — |

### 11.3 S5 detector input requirements

| Requirement | Value | Label | Source |
|---|---|---|---|
| Signal domain | A spectrogram (time–frequency image) derived from the audio; Fig. 3 shows the spectrogram directly feeding contour extraction. Other input forms are not excluded by the text. | **IMPLIED** | [P] Fig. 3, §II-C |
| Upstream processing | S2–S4 exactly as in §4: 192 kHz, high-pass above 2 kHz, Hamming 2,048, 50 % overlap ⇒ grid 93.75 Hz × 5.333 ms | **EXPLICIT / IMPLIED** (see §4) | [P] E-03, E-04 |
| Frequency span given to the detector | Lower edge 2 kHz via the high-pass; upper edge not stated | **IMPLIED** (lower) / **UNSPECIFIED** (upper) | [P]; A-06 |
| Spectrogram quantity, scaling, clipping, normalisation | Not stated | **UNSPECIFIED** | [P]; A-05 |
| Patching / tiling / context window | Not stated | **UNSPECIFIED** | [P], [31] not obtained |
| Channel(s) | Not stated | **UNSPECIFIED** | [P]; A-01 |
| Input extent | A whole 60-s file is converted to a spectrogram; extent handling by the detector not stated | **IMPLIED** (whole files processed) / **UNSPECIFIED** | [P] §II-C |
| Precedent (not [P]) | [L30]: log-magnitude, clipped to [0, 6], scaled to [0, 1], 50 × 50 patches on a 2 ms × 125 Hz grid, 5–50 kHz | **EXPLICIT [L30]** | §11.1.2 #2–#8 |
| Grid mismatch [P] vs [L30] | [P] hop 5.333 ms vs 2 ms; [P] bin spacing 93.75 Hz vs 125 Hz. [L30]'s patch (100 ms × 6.25 kHz) would be 18.75 frames × 66.7 bins on [P]'s grid (non-integer); [L30]'s 5–50 kHz band would be 480 bins on [P]'s grid. Arithmetic only; **no transfer of [L30] values to [P] is implied**. | **IMPLIED** (arithmetic) | derived |

### 11.4 S5 detector output requirements

| Requirement | Value | Label | Source |
|---|---|---|---|
| Output consumed by S6 | "Detection outputs" that S6 converts to individual whistle tracks | **EXPLICIT** (name only) | [P] E-05 |
| Output form in [P] | Not stated | **UNSPECIFIED** | [P], [31] not obtained |
| Needed by S7 | Tracks must yield a time–frequency vector per whistle on the S4 grid (measured in "samples" and "bins" in [P]'s analysis) | **IMPLIED** | [P] E-11 |
| Output form in [L30] (precedent) | Dense per-node confidence map, same size as input spectrogram, one channel, continuous values within [0, 1] | **EXPLICIT / IMPLIED [L30]** | §11.1.2 #9–#12 |
| Output meaning in [L30] | Per-node whistle-contour presence; trained against Gaussian-blurred contour masks | **EXPLICIT [L30]** | §11.1.2 #11 |
| Output form of the CFAR in [R32] | Per-cell detection decisions along the frequency axis, shown as detection images | **IMPLIED, weak [R32]** | [R32] §III |
| Threshold in [P] | Not stated | **UNSPECIFIED** | [P] (A-07) |
| Threshold in [L30] | τ = 0.5 (fixed operating point); sweep of 30 values for map-level PR | **EXPLICIT [L30]** | §11.1.2 #13 |

### 11.5 S5 → S6 interface

Only fields supported by a source are filled. Fields marked **NOT FILLED** are not supported by any obtained source and are intentionally left open.

```
INPUT (S4 → S5)
  Input ........................ spectrogram of the S4 stage (grid 93.75 Hz × 5.333 ms)        [IMPLIED, P Fig.3 + §II-C]
  Input representation ......... NOT FILLED (quantity, scale, normalisation)                   [P: UNSPECIFIED; A-05]
  Input channels ............... NOT FILLED                                                    [P: UNSPECIFIED; A-01]
  Frequency span ............... >= 2 kHz; upper edge NOT FILLED                               [P: IMPLIED/UNSPECIFIED; A-06]

OUTPUT (S5 → S6)
  Output ....................... "detection output" convertible into individual whistle tracks [EXPLICIT name only, P E-05]
  Output form (P / [31]) ....... NOT FILLED                                                    [P, 31: UNSPECIFIED]
  Output shape ................. T frames × F bins of the S4 spectrogram, one value per node   [EXPLICIT precedent L30; IMPLIED need]
  Output value meaning ......... confidence that the node lies on a whistle contour            [EXPLICIT precedent L30 only]
  Output value range ........... continuous, within [0, 1]                                     [IMPLIED precedent L30 only]
  Required resolution .......... not coarser than the S4 grid in time or frequency             [IMPLIED, P E-11 + L30 #10]
  Thresholding ................. one global threshold → node-level detections; value NOT FILLED for P   [L30: τ = 0.5]
  Ridge convention ............. NOT FILLED (1-node line vs blurred band; peak picking per frame) [L30 defers to ref. 8; P: UNSPECIFIED]

EXPECTED DOWNSTREAM (S6 → S7)
  S6 ........................... time-sequential linking of detected nodes into tracks; gaps bridged; crossings resolved   [EXPLICIT for L30's extractor; P: UNSPECIFIED]
  Click handling ............... broadband vertical energy must not become whistle tracks     [EXPLICIT L30 + R32; IMPLIED for P (Fig. 3 stripes)]
  Track output ................. one track per whistle → one time–frequency vector (S7)      [EXPLICIT, P]
  S6 numeric rules ............. NOT FILLED (gap, minimum duration, association metric, CFAR settings)   [P/33: UNSPECIFIED; D-05]
```

### 11.6 What Park et al. actually specify

#### (a) Explicit statements in [P] about the detector chain

| Statement | Label | Source |
|---|---|---|
| A neural-network-based extraction method [31] "was applied" to detect individual whistle contours | **EXPLICIT** | §II-C (E-05) |
| The extraction was combined with a post-processing algorithm ([32],[33]) that converts detection outputs into individual whistle tracks | **EXPLICIT** | §II-C (E-05) |
| Spectrogram: resampled to 192 kHz, Hamming 2,048, 50 % overlap; resolutions ≈ 5.3 ms and 93.75 Hz; high-pass above 2 kHz | **EXPLICIT** | §II-C (E-03, E-04) |
| 228,673 time–frequency vectors from 2,486 files | **EXPLICIT** | §II-C (E-06) |
| Fig. 3 shows spectrogram → colour-coded whistle contours → images. The spectrogram contains vertical broadband stripes consistent with clicks; [P] does not label them in the caption, but its Introduction states that clicks are broadband pulse signals | **EXPLICIT** (figure chain); **IMPLIED** (stripes = clicks) | Fig. 3 (E-10); [P] §I |

#### (b) Not specified by [P]

Everything in §11.3–§11.5 marked UNSPECIFIED, plus: the detector's architecture, weights, training data and synthesis recipe; whether [31] was used unchanged from its own publication; which stage (the network, [32], or [33]) applies thresholds; how [32] and [33] interact with [31]; track-level constraints; quality control; the relation of [31] to [L30].

Three readings of the chain remain open and **cannot be decided from [P] or [R32]**: **(A)** network output → CFAR used as a thresholding/peak stage → tracking; **(B)** network output → tracking directly, with CFAR used elsewhere; **(C)** CFAR provides a second detection output that is tracked alongside the network's.

#### (c) Comparison [P] vs [L30], and mismatches (documented, not resolved)

| Aspect | [P] | [L30] | Note |
|---|---|---|---|
| Identity | Cites [31] in text; lists [30] without citing it | — | §11.0 contradiction; link between [31] and [L30] UNKNOWN |
| Species / data | False killer whale, DCLDE 2022 (HICEAS, 500 kHz → 192 kHz) | Common and bottlenose dolphins, DCLDE 2011 (192 kHz) | Different domain |
| STFT | Hamming 2,048 @ 192 kHz (10.667 ms), hop 5.333 ms, 93.75 Hz | Hamming 8 ms, hop 2 ms, 125 Hz | Different grid; [P]'s S4 stays fixed (D-01) |
| Frequency band | ≥ 2 kHz; upper unstated | 5–50 kHz | — |
| Post-processing | [32] + [33] | Graph search of Roch et al. 2011 | Different downstream |
| Network output | "Detection outputs" | Per-node confidence map | Form for [31] UNKNOWN; "frequency detection" in [31]'s title leaves a per-frame formulation possible |
| Training data | Not stated | Annotated DCLDE 2011 contours + synthetic injection | UNKNOWN for [31] |
| Performance | Not reported for the detector | F1 0.634 → 0.792 (best synthesis) / 0.884 (supervised) | No detector performance in [P] |
| **Other source [RW1] (not cited by [P])** | — | — | A later paper co-authored by G.-H. Park uses a pixel-wise **segmentation** formulation on DCLDE 2022 false-killer-whale spectrograms: 512 × 512 single-channel images, input scaled to [−1, 1], ground truth in [0, 1], prediction thresholded at 0.5, **STFT Hamming 4,096, 97.56 % overlap, FFT 49,152** — differs from [P]'s S4 (2,048 / 50 %). It cannot be [31] (published 2026, cites neither [31] nor [33]); **do not use its values as [P]'s.** |

#### (d) Reproducibility ledger

| Component | Reproducible from available information? | Basis |
|---|---|---|
| S4 spectrogram | **Yes**, with scaling/padding choices open | §4 |
| Detector [31] (weights, architecture, data) | **No** | Not obtained; [P] gives no details |
| A functionally analogous detector in the style of [L30] | **Partly describable** (input, patching, loss, training protocol, evaluation) on a different grid/species; no code or weights obtained; code availability not stated in [L30] | §11.1.2 |
| Post-processing [32] (MCA-CFAR) | **Partly** (principle, demo values; no P_fa/scale/STFT) | §3.6 |
| Tracking [33] | **No** | Not obtained |
| [L30]'s graph search (ref. 8) | **Procedure described; numeric settings not** (ref. 8 not read) | §11.1.2 #15 |
| S7 vectors identical to the paper's | **No** | Depends on all of the above |

### 11.7 Remaining unknowns

| ID | Unknown | Relates to |
|---|---|---|
| U-01 | Content of [31] (input, patching, output form, threshold, training data, synthesis, loss, metrics, performance) | A-07 |
| U-02 | Content of [33] (tracking/linking rules, thresholds, gap handling, constraints, overlap and harmonic handling, assumed detector output) | A-08 |
| U-03 | Relationship between [31] and [L30] (follows, builds on, unrelated) | A-20 |
| U-04 | Which of readings (A)/(B)/(C) in §11.6(b) describes the real chain; where CFAR [32] sits | A-07 |
| U-05 | Whether [31]'s network sees the S4 grid, a different STFT, or a different representation | A-05 |
| U-06 | Frequency span fed to the detector (upper bound) | A-06 |
| U-07 | Ridge convention expected by S6 (thin line vs blurred band; peak-per-frame) | A-08 |
| U-08 | S6 numeric rules: gap tolerance, minimum/maximum duration, association metric, CFAR P_fa/scale | A-08, A-09 |
| U-09 | Treatment of the six channels at S4–S6 | A-01 |
| U-10 | **No contour-level ground truth exists in the project** (the DCLDE workbook has encounter-level start/end times only). [L30] evaluates against analyst contours from DCLDE 2011, which are not in the project. | evaluation |
| U-11 | Definition of "comparable to [31]" when [31] itself can neither be run nor read | §11.9 |
| U-12 | [L30] internal inconsistencies I-1 … I-4 and numeric settings deferred to its ref. 8 | §11.1.3 |

### 11.8 Functional requirements for a replacement detector (FUTURE REPLACEMENT — not the paper's method)

Functional only; no architecture is implied. **Basis** says where each requirement comes from: a source with its label, or **PROJECT** (derived from this specification's own rules, not from the references).

| ID | Requirement | Basis |
|---|---|---|
| FR-01 | Must consume the S4 spectrogram exactly as defined in §4 (192 kHz, high-pass above 2 kHz, Hamming 2,048, 50 % overlap, grid 93.75 Hz × 5.333 ms); must not require changes to S2–S4. | **EXPLICIT/IMPLIED** [P] §4; **PROJECT** D-01, D-02 |
| FR-02 | Any scaling, clipping, normalisation or tiling applied to the spectrogram before the detector is part of the detector's own contract and must be defined and recorded as an implementation assumption (A-05 is open). | [P] **UNSPECIFIED**; [L30] precedent |
| FR-03 | Must cover the whole analysis band the paper leaves open: from 2 kHz upward; upper limit to be decided and recorded, not silently narrowed. | [P] **IMPLIED** (lower) / **UNSPECIFIED** (upper) |
| FR-04 | Must output a dense time–frequency map with one value per S4 node, covering the same extent as the input, for files of arbitrary length (whole 60-s files). | [L30] **EXPLICIT** precedent; **IMPLIED** need from S7 on the S4 grid |
| FR-05 | Output semantics: per-node confidence that the node lies on a whistle contour — not a per-patch, per-file or click/noise class label. | [L30] **EXPLICIT** precedent; [P] **UNSPECIFIED** |
| FR-06 | Output values continuous within [0, 1]; a single global threshold converts the map to node-level detections. The threshold is set by evaluation (§11.9), never to match the paper's yield (D-10). | [L30] **EXPLICIT** (τ = 0.5) precedent; **PROJECT** D-10 |
| FR-07 | Output resolution not coarser than the S4 grid in time or frequency. | **IMPLIED** ([P] E-11; [L30] #10) |
| FR-08 | The output's ridge convention (thin line vs blurred band; how a per-frame peak is taken) must be fixed jointly with S6 and recorded, because S6 is itself unspecified. | [L30] **EXPLICIT** (peak-based consumer); [P] **UNSPECIFIED** |
| FR-09 | Must not respond to broadband impulsive clicks (vertical stripes). | [L30] **EXPLICIT**; [R32] **EXPLICIT**; vertical stripes consistent with clicks are visible in [P] Fig. 3 (**IMPLIED**); applicability to [31] **UNKNOWN** |
| FR-10 | Simultaneous whistles must remain separately detectable and crossings must not be suppressed, so that S6 can resolve them. | [R32] **EXPLICIT**; [L30] **EXPLICIT** (crossings); applicability to [31] **UNKNOWN** |
| FR-11 | Output for a file must not depend on the tiling used to process it, beyond a documented and tested tolerance (reassembly rule is unspecified in [L30]). | **PROJECT** (derived from whole-file processing in [P]) |
| FR-12 | Must work on false killer whale whistles in DCLDE 2022 data at the S4 grid; behaviour on [L30]'s dolphin data/grid is not a requirement. | [P] **EXPLICIT** data |
| FR-13 | Must accept one single-channel spectrogram per call; the channel policy (A-01) is decided outside the detector. | [L30] **IMPLIED, weak**; **PROJECT** |
| FR-14 | Deterministic for identical input and parameters; all detector-side assumptions logged. | **PROJECT** |
| FR-15 | "Lightweight" and low-latency are project goals for the next phase; no obtained source requires them for the detector. ([L30] reports ≈ 10 ms of network time per 1 s of audio on a GTX 1080 Ti as a reference point only; [P] Table 6 concerns the generator, not the detector.) | **PROJECT** goal; [L30] reference point |

### 11.9 What must be evaluated when comparing our replacement against [31] (FUTURE REPLACEMENT)

[31] cannot be run or read (U-01, U-11), so every comparison below is against a **stated proxy**, and the proxy must be named in the evaluation protocol.

| ID | Evaluation | Basis |
|---|---|---|
| V-01 | **Map-level precision–recall**: ground-truth contours thinned to 1 pixel; maps binarised at 30 thresholds in (0, 1); report the curve and the point at 0.5. | [L30] **EXPLICIT**; BSDS tolerance **UNSPECIFIED** |
| V-02 | **Contour-level precision / recall / F1 with the same S6 applied to both candidate and reference maps** (matching rule of [L30]/ref. 8: time overlap; mean frequency deviation < 350 Hz; analyst whistle ≥ 150 ms and SNR ≥ 10 dB over ≥ one third; unmatched detections count as false positives; single operating point). Applicability of these dolphin-derived criteria to false killer whales on the S4 grid is **UNCONFIRMED**. | [L30] **EXPLICIT** |
| V-03 | **Comparison target**: the only published numbers are [L30]'s (Table I: baseline F1 0.634; best synthesis 0.792; fully supervised 0.884). They come from a different dataset, grid and post-processor, so they are **proxy references only**; a same-data, same-S6 reference is needed for a real comparison, and its choice is an open decision. | [L30] **EXPLICIT**; **PROJECT** |
| V-04 | **Click robustness**: false responses on click-dominated frames; [L30] gives only qualitative evidence, so a metric must be defined. | [L30] **EXPLICIT** (qualitative); metric **UNSPECIFIED** |
| V-05 | **Overlapping / crossing whistles**: detection and preservation of crossings; metric to be defined ([L30] qualitative). | [L30], [R32] **EXPLICIT**; metric **UNSPECIFIED** |
| V-06 | **Contour continuity / fragmentation and missed whistles** after thresholding (e.g. counts of whistles split into several connected components, or with no prediction). [RW1] defines such counts; adopting them is a project decision. | [RW1] (not cited by [P]); **PROJECT** |
| V-07 | **Threshold sensitivity** (F1 versus threshold). | [L30] sweeps 30 thresholds; **PROJECT** |
| V-08 | **Annotation / data efficiency** if synthetic or limited-label training is used: performance versus annotation budget ([L30] Fig. 5: rises with training-data fraction 6.25 → 100 %, largest gain near 25 %; values plotted, not tabulated). | [L30] **EXPLICIT** (qualitative) |
| V-09 | **Runtime** measured with a stated convention ([L30]: ≈ 10 ms network + 300 ms graph search per 1 s audio on a GTX 1080 Ti; hardware will differ). | [L30] **EXPLICIT** |
| V-10 | **Pipeline-level descriptive check** of S7 vectors (count per file, duration and bandwidth distributions) against the paper's reported facts (E-06, E-11), **never as a tuning target** (D-10). | [P]; **PROJECT** D-10 |
| V-11 | **Ground truth**: no contour-level annotations exist in the project (U-10); the source of annotations for V-01/V-02 must be decided before any evaluation. | **PROJECT** |

### REPLACEMENT DETECTOR DESIGN INPUT

**1. Required input.** The S4 spectrogram exactly as specified in §4 (192 kHz, high-pass above 2 kHz, Hamming 2,048, 50 % overlap, 93.75 Hz × 5.333 ms grid), single channel per call. Spectrogram scaling/normalisation, upper frequency limit and channel policy are **not** fixed by the sources and must be decided and recorded (A-01, A-05, A-06).

**2. Required output.** A dense time–frequency map aligned one-to-one with the S4 grid (T frames × F bins), covering the full file, continuous values within [0, 1], convertible with a single global threshold into node-level detections.

**3. Required output semantics.** Per-node confidence that the node lies on a whistle contour (not a patch/file label, not a click or noise class). Source precedent: [L30] (EXPLICIT); [P]/[31] form UNSPECIFIED.

**4. Required resolution.** Not coarser than 5.333 ms in time and 93.75 Hz in frequency; no loss of resolution relative to the S4 grid. Sub-bin precision is not specified anywhere.

**5. Required downstream compatibility.** Output must be usable by S6 for time-sequential linking of nodes into tracks (gap bridging, crossings), reject broadband clicks, keep simultaneous whistles separable, and ultimately yield one time–frequency vector per whistle (S7). The ridge convention must be agreed with S6, whose numeric rules are unknown (U-07, U-08).

**6. Training/evaluation requirements inherited from the references.** *Precedent from [L30], not a mandate:* training from whistle-absent background patches with injected whistle-like shapes (blend X′ = X + α(Y∗G), α ~ U[0.03, 0.23]) or from annotated patches; per-node regression target (blurred contour mask) with a Charbonnier-type loss; a fixed 0.5 operating threshold; evaluation by map-level PR (30 thresholds, thinned ground truth) and contour-level precision/recall/F1 after the same graph-search-type post-processing, with the matching rules in §11.9 V-02; runtime reporting. The comparison target must be an explicitly named proxy (V-03), and ground-truth annotations must first be sourced (U-10).

**7. Unknowns that still need to be resolved.** U-01 … U-12 (§11.7); above all the content of [31] and [33], whether readings (A)/(B)/(C) apply, the S6 rules, the ridge convention, the upper frequency limit, the channel policy, and the source of ground-truth contours.

**8. What can legitimately be changed by our proposed replacement.** The internal model design and size, internal tiling/context strategy, loss and optimiser, training-data construction (including synthetic generation), model-internal normalisation, and the method of choosing the operating threshold — provided the input/output contract in items 1–5 is respected. **What cannot change:** S2–S4 parameters (D-01, D-02), the S4 grid alignment of the output, the 128 × 128 image definition downstream (D-03), and the rule that tuning to the paper's yield is forbidden (D-10). Any replacement is a **documented deviation** from the paper's method (D-06), to be named as such wherever results are reported.

---

*End of Revision 2. No network was designed or implemented, no preprocessing code was written, and no raw data was modified. The only project file changed is this document.*
