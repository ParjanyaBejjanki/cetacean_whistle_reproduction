# whistlepipe — a reproduction of VAE-GMM for cetacean whistles

Reproduces Park et al., "A Generative Model for Cetacean Whistles Using Variational Autoencoder and Mixture of
Gaussians", *IEEE Access* 14 (2026), pp. 27877–27894 (DOI 10.1109/ACCESS.2026.3664765), on false killer whale whistles
from NOAA's DCLDE 2022 data: raw recordings -> whistle images -> VAE-GMM training and evaluation -> inference and a
web demo.

Preprocessing implements S1–S9 of `resources/PREPROCESSING_SPECIFICATION.md` (the paper evidence for every stage; nothing
from it is repeated here beyond what is needed to run and extend the code). The VAE-GMM training, GMM fitting and
evaluation (`src/whistlepipe/vaegmm/`) follow `resources/TRAINING_SPECIFICATION.md` (see "VAE-GMM" below).

**Status (2026-11-03):** the dataset is built (180,154 images from 8 encounters, see "Dataset record"). The main model
V1 trains on encounter 191 only (363 files, 63,490 images), planned for 150 epochs, with K = 3,000; the other encounters
stay on disk, unused (`resources/TRAINING_SPECIFICATION.md` §2.1). Its log ends in epoch 102; the last finished epoch is 101
(rec 171.2), saved as `models/vaegmm/V1_L64_b1e-4_critic_E191/checkpoints/last.pt`. There is no `final.pt`, GMM fit or evaluation result yet.

> **Research-integrity statement.** The paper's S5 detector (its reference [31]) and S6 tracker (refs [32], [33]) are
> unavailable. This project does **not** reproduce them. S5 and S6 here are **newly designed replacements** that satisfy the
> functional requirements in spec §11. They are not claimed to be equivalent to the originals. Li et al. 2020 ("[L30]") is
> **not** reference [31]; it was used only as methodological inspiration (injecting known shapes into whistle-free
> backgrounds; per-node confidence maps) and none of its numbers were adopted.

## Evidence tags (used in `configs/*.toml`, code docstrings and this file)
`[EXPLICIT]` / `[IMPLIED]` / `[UNSPECIFIED]` come from the specification. **`[DECISION]`** = implementation decision made
here because the paper leaves it open (never paper evidence). **`[REPLACEMENT]`** = part of the new S5/S6 design.
The `[paper]` section of `configs/pipeline.toml` is locked: `config.py` refuses to run if any value differs from the spec.

## Layout
```
encounter_<id>_audio/  RAW, immutable, one folder per encounter (8 folders, 1,415 files). Opened read-only; storage.py
                       refuses any write inside them. Encounter 191 also feeds the synthetic backgrounds (synth_raw_dir).
metadata/  resources/  provided inputs and the specification; manifest_all.csv = SHA-256 of every raw file (build-manifest)
configs/pipeline.toml  paper-defined [paper] + implementation decisions [decisions.*] + storage + validation sets
configs/detector.toml  S5 architecture, synthetic-data generator, training  (all [REPLACEMENT])
configs/vaegmm.toml    VAE-GMM training, GMM fitting and evaluation ([paper] section locked by vaegmm/config.py)
src/whistlepipe/       audio.py(S1) dsp.py(S2-S4) detector/(S5) tracking.py(S6) representation.py(S7-S9)
                       pipeline.py(orchestrator) storage.py(lineage/policy) dataset.py(final layout) manifest.py
                       checks.py plotting.py cli.py
                       synth/(synthetic training data)  train.py  evaluate.py  vaegmm/(VAE-GMM)
                       infer.py(inference engine) sonify.py(image -> sound) server.py(web API) webexport.py ratecheck.py
configs/inference.toml inference / web demo settings (not part of any training or processing hash)
web/                   the project webpage (scroll film + "Try it"); web/media4k/*.mp4 and web/data/ are not in git
colab/                 Colab notebook for the main VAE training + make_colab_bundle.py (its output colab/dist/ is not in git)
tests/                 pytest, small synthetic fixtures; none needs the full dataset
models/detector/<ver>/ checkpoint.pt, train_info.json, calibration.json, eval_*.json
models/vaegmm/<run_id>/  VAE runs: train_info.json, train_log.jsonl, sample grids (checkpoints/ not in git); _gates/ = gate records
data/synthetic/        background banks (real S3 audio windows) and cached synthetic val/test sets
preprocessed/          FINAL dataset, one folder per encounter: see "Dataset layout"
preprocessing_checks/<run_id>/   small check runs with full debug detail (never overwritten): see "Storage and lineage"
```

## Setup
* **Python 3.11 or newer** (`tomllib`), then `pip install -r requirements.txt`. For a CUDA GPU, install `torch` and
  `torchvision` from the PyTorch index for your CUDA version first. Versions in use: Python 3.11.0, numpy 2.4.4,
  scipy 1.17.1, soundfile 0.13.1 (libsndfile 1.2.2), torch 2.14.0+cu130 (dataset build, `preprocessed/dataset_info.json`);
  Python 3.13, torch 2.11 on Colab (V1 `train_info.json`).
* **Raw audio** (not in git, about 190 MB per file): NOAA DCLDE 2022, ship 1705 (R/V Lasker),
  `gs://noaa-passive-bioacoustic/dclde/2022/dclde_2022_1705/audio/`. `metadata/whale_encounters_1705.csv` gives each
  encounter's first and last file prefix; the false killer whale encounters used here are the eight in
  `configs/pipeline.toml` `[paths].raw_dirs`. Put each encounter's files in `encounter_<id>_audio/` in the project root, then
  run `build-manifest` once.
* **Licence:** none has been chosen yet. The paper PDF in `resources/` is CC BY-NC-ND 4.0 (its first page).

## Commands (from the project root)
```bash
python -m pytest                                   # all automated tests
python whistle.py list-sets                        # validation sets (val1/val5/val10 from encounter 191, per_encounter) + files per encounter
python whistle.py build-backgrounds                # real-background banks for synthetic training (S1-S3 on 56 non-validation files)
python whistle.py build-ridge-masks                # label-free masks of real faint whistles in the training backgrounds; needed
                                                   #   before training with synth.labels.ignore_background_ridges = true (v2c, v2e)
python whistle.py train --version ridgenet-v1      # train S5 on SYNTHETIC data (checkpoints are never overwritten; --resume continues
                                                   #   after an interruption; full settings of v2c-v2e: models/detector/<ver>/detector_config.json)
python whistle.py eval-synth --split val --calibrate   # synthetic validation + threshold calibration
python whistle.py eval-synth --split test              # synthetic test
python whistle.py build-manifest                   # once: hash every raw file -> metadata/manifest_all.csv (never overwritten;
                                                   #   check runs and build-dataset refuse files that are not in it)
python whistle.py run --run-id v2c_check1_1file --set val1 --last s4 --storage debug     # check runs -> preprocessing_checks/
python whistle.py run --run-id v2c_check2_val5 --set val5 --storage debug
python whistle.py run --run-id v2c_check3_per_encounter --set per_encounter --storage debug
python whistle.py validate-run --run-id v2c_check3_per_encounter   # automated gate: lineage, output validity, cross-stage counts
python whistle.py report --run-id v2c_check3_per_encounter         # compact per-file table
python whistle.py plot-waveform                    # S1 waveform of the val1 file, S1/S2 samples overlaid, S1 vs S2 spectra
                                                   #   -> docs/figures/waveform_<stem>.png (--file <stem>, --zoom-s, --zoom-ms)
python whistle.py build-dataset                    # FINAL dataset -> preprocessed/ (only after the checks pass; resumable)
python whistle.py build-dataset --encounters 191 214   # a subset of encounters
python whistle.py validate-dataset                 # automated checks of preprocessed/
```
Any stage range: `--first s5 --last s6 --from-run preprocessing_checks/<earlier run>` (needs that run to hold the previous stage's
intermediate). `--override key=value` changes a config value for one run and is recorded in the run manifest.

## Dataset layout (`preprocessed/`, built by `build-dataset`)
```
preprocessed/
  README.txt  dataset_info.json          what every file is; detector, config + hash, software, counts per encounter
  encounter_<id>/images.npy              (N, 128, 128) float32 in [0, 1]: all whistle images of the encounter
  encounter_<id>/images.csv              one row per image (row i <-> images.npy[i]): source file, time, frequencies, contour_id
  encounter_<id>/contours.npz            S7 contours (Hz per 5.333 ms frame); image_index = -1 + no_image_reason too_long/too_wide
  encounter_<id>/files.csv               one row per raw file: OK/FAILED, n_images, first_image_index, n_too_long/n_too_wide,
                                         manifest status, detector_background_split, hash check
  _details/                              not needed for training:
      logs/encounter_<id>_*              run log, per-file stage stats, run manifest, summary, build record
      per_file/encounter_<id>/<stem>.json   lineage of each raw file (same content as meta/<stem>.json below)
      failed/encounter_<id>/<stem>/      traceback + intermediates of a failed file
      detector_output/encounter_<id>/<stem>/   S5 confidence (float16) + S6 tracks ([dataset].keep_detector_output)
```
Each encounter is run through the normal pipeline into a temporary run (`_details/work/`), packed into the four files and the
temporary run is deleted. The folder is written as `encounter_<id>.partial/` and renamed last, so it exists only when complete;
re-running `build-dataset` skips complete encounters and redoes interrupted ones; it stops if a complete encounter's raw folder
gained or lost files since it was built (delete that encounter folder to rebuild it). One dataset = one processing config
(`[paper]` + `[decisions]`, without the S5 device) and one detector checkpoint (anything else is refused); paths and the list of raw
folders are not part of it, so encounters can be added later. Every raw file must be in `manifest_all.csv` before a build starts.
* **Raw-file status** (`manifest_all.csv`): VALID, SHORT (header shorter than 60 s; processed), SUSPECT_TRUNCATED (far fewer bytes
  than the header length implies, e.g. a cut-off download: a FLAC header keeps the full length), INVALID, UNREADABLE. S1 checks the
  decoded length of every file against its header, so truncated files FAIL with a clear error; failures of non-VALID/SHORT files are
  *expected* (listed by `validate-dataset`, not counted as problems).
* **Detector-background files:** 56 files of encounter 191 provided quiet windows for the detector's synthetic data (24 train,
  16 val, 16 test; `data/synthetic/backgrounds/*.json`). They are processed like every other file and marked in
  `files.csv` `detector_background_split`, so they can be checked or excluded.
* **Code version:** every run manifest, build record and `dataset_info.json` records the git commit and whether `src/`, `configs/`
  or `whistle.py` had uncommitted changes (`git_uncommitted_code_changes`); commit before a final build.

## Storage and lineage (check runs)
`preprocessing_checks/<run_id>/` contains `run_manifest.json` (full config snapshot + hashes, software versions, detector checkpoint
SHA-256, file list, options), `summary.json`, `logs/`, `intermediate/<stem>/`, `final/<stem>/`, `meta/<stem>.json`,
`failed/<stem>/`, `diagnostics/`.
* **Lineage** `meta/<stem>.json`: raw path/size/mtime/**SHA-256 (verified against `metadata/manifest_all.csv`)** → config
  hash → per-stage diagnostics (shapes, rate, duration, range, counts, rejections, timings) → detector name/version/
  checkpoint hash → stored outputs. Each file also records `unchanged_after_processing` (raw immutability audit).
* **Policy** (`[storage]`): `debug` keeps S1–S6 (+S8) intermediates and plots; `normal` keeps S5/S6; `minimal` keeps none.
  S7 vectors, S9 images and lineage are kept in **every** mode. A **failed** file always keeps every intermediate that
  existed, plus the traceback, under `failed/<stem>/`.
* Runs are isolated by `--run-id`; an existing run is never overwritten.

## S5 contract (`detector/interface.py`)
Input: S4 log-magnitude spectrogram `(T, 1025)` float32 dB on the 93.75 Hz × 5.333 ms grid. Output: `DetectionMap.conf`
`(T, F)` float32 in [0, 1], one value per S4 node for bins `[bin_lo, bin_hi]` = 2–24 kHz (`[DECISION]` for the upper edge);
semantics = confidence that the node lies on a whistle contour (fundamental or harmonic). No thresholding in S5. Output is
independent of the internal tiling (tested), deterministic, and CPU/GPU-consistent (tested; TF32 is disabled because it changed S6 tracks between devices). Replace the detector by
implementing `WhistleDetector.predict(spec) -> DetectionMap`; nothing else changes.

**RidgeNet** (`detector/ridgenet.py`, ~14k parameters): fully convolutional, no pooling, dilated 3×3 blocks; receptive field
±19 frames × ±9 bins. Input conditioning (local per-bin noise-floor subtraction, clip, scale; `detector/conditioning.py`) is
*internal to S5* and shared by training and inference; the S4 output is never modified.

## Synthetic training (no real ground truth is used or fabricated)
Real S3 audio windows (3 s) from raw files that belong to **no** validation set serve as backgrounds, screened by a purely
classical label-free ridge score (windows with persistent narrow-band ridges are dropped). Synthetic whistles (random
contours incl. crossings/parallel pairs, 0–5 simultaneous, harmonics, variable SNR) and synthetic clicks/sweeps are added in
the time domain and passed through the *same* S4 function. Labels (Gaussian ridge targets; nodes of invisible whistle parts
are ignored, not forced negative) come only from the known contours. The generator never imports a detector (tested).
Train/val/test use disjoint background files. **Known limitations:** encounter 191 is whistle-dense, so clean windows are
scarce (see `data/synthetic/backgrounds/*.json`); residual weak real whistles may remain in backgrounds as unlabeled
negatives; synthetic metrics therefore describe the synthetic task only.

## Evaluation categories (kept separate)
* **A — synthetic** (`eval-synth`): node PR and track-level recall/precision against the generator's ground truth.
* **B — real-data engineering validation** (`validate-run`, `report`, diagnostics figures): consistency, counts, rejections,
  timing, visual inspection. **No real-data precision/recall/F1 is computed or claimed.**
* **C — manually annotated real data**: not available; not implemented.
The paper's counts (228,673 vectors, ≈ 91.98 per file) are **not** optimisation targets.

## Implementation decisions that are NOT paper evidence (full list: `[DECISION]`/`[REPLACEMENT]` in `configs/pipeline.toml`)
S1 channel 0 (arbitrary; a six-channel survey showed no anomalous channel) · S2 polyphase Kaiser FIR 48/125 · S3 4th-order
Butterworth, zero-phase, after S2 (−6 dB at 2 kHz) · S4 periodic Hamming, dB output, no padding · S5 band 2–24 kHz and all of its
internals · S6 entire tracker and its settings · S8 centre frequency = (f_min+f_max)/2, whistle centred in time, complex
12 kHz baseband derived from the paper grid, 2.7 ms edge ramps, tracks > 128 frames **rejected** (counted, not cropped) ·
S9 linear magnitude, per-image max normalisation, row 0 = highest frequency (0 Hz on row 63).

## Dataset record (ridgenet-v2c, 8 encounters: the training data)
Built 2026-10-21 by `build-dataset` into `preprocessed/` (`dataset_info.json`): git commit `8cf3ed4` with no uncommitted code
changes, processing config `78f1bb1b…`, detector `ridgenet-v2c` (`c00d66a0…`), manifest `d20f4b38…`. Before the build, the check
runs `v2c_check1_1file` (S1–S4), `v2c_check2_val5` (5 files, 766 images) and `v2c_check3_per_encounter` (one file per encounter,
1,174 tracks, 1,098 images) finished with every file OK, and `validate-run` passes on both full check runs (0 problems, one
detector checkpoint; re-run 2026-10-31).

| Encounter | Files | Tracks after S6 | Rejected in S8 (> 128 frames) | Images |
|---|---|---|---|---|
| 191 | 363 | 67,236 | 3,746 | 63,490 |
| 214 | 124 | 6,825 | 148 | 6,677 |
| 276 | 120 | 14,653 | 502 | 14,151 |
| 279 | 145 | 36,171 | 2,654 | 33,517 |
| 280 | 104 | 16,972 | 1,169 | 15,803 |
| 308 | 162 | 23,599 | 1,705 | 21,894 |
| 331 | 331 | 20,903 | 464 | 20,439 |
| 416 | 66 | 4,253 | 70 | 4,183 |
| **Total** | **1,415** | **190,612** | **10,458 (5.5 %)** | **180,154** |

1,415/1,415 files OK, 0 failed. Images per file: mean 127.3, median 95, max 1,003 (2 files have none). S6 rejections: too short
71,546, harmonic 19,962, low observed fraction 7,375, synchronous comb 329, duplicate 32, too steep 1. Median whistle = 40 frames
(213 ms); 37.3 % of images are shorter than 30 frames. Time: 71.5 min in total, 3.0 s per file (RTX 5060 Laptop GPU). The S6
settings are those frozen in the v1 work below (threshold 0.5, not recalibrated for v2c). The paper's 228,673 vectors from 2,486
files are not a target and not comparable (different S5/S6, different selection of files).

## Validation record of ridgenet-v1 (encounter 191 only; superseded)
The v1 run folders (`runs/…`) were deleted on 2026-10-21 to make room for the v2c multi-encounter dataset; the numbers below are
the record of that work. The v2c dataset that replaced it is recorded above.

| Run | Files | What happened |
|---|---|---|
| `phase1_1file` | 1 (`1705_20170930_015147_062`) | S1–S4 gate. 12/12 independent checks pass: S1 equals the raw channel bit-for-bit; S2 spectrum preserved (median −0.07 dB); S3 measured gain −6.01 dB at 2 kHz, ≈0 dB above 5 kHz; S4 matches scipy within 0.03 dB. A six-channel survey showed no anomalous channel. |
| `phase2_5files`, `phase3_10files` | 5, 10 | First full S1–S9 passes. Automated gate passed, but **visual inspection found a pulse-train artefact** (17 near-identical short tracks in one burst). → S6 rule `synchronous_comb` added. |
| `…_r2` | 5, 10 | Repeated after the rule. Artefact removed (23 → 6 tracks in that window), gate passed. |
| `…_r3` | 5, 10 | Repeated after an **S5 numerics fix**: CUDA TF32 made confidences differ from CPU by up to 7e-3 and changed S6 tracks between devices; TF32 is now off (difference 5e-6, identical tracks). Gate passed. |
| `phase4_full363` | 363 | First full run; **superseded** (pre-numerics-fix). |
| **`phase4_full363_r2`** | **363** | **Final.** 363/363 OK, 0 failed, one config hash (`19aaca0b…`), one detector checkpoint (`58798cb4…`), all 363 raw SHA-256 equal to the manifest, raw files unchanged. |

Configuration was frozen after the synthetic validation (S6 gate 3 bins, gap 12 frames, threshold 0.5) and not changed between
the real-data phases except for the two documented fixes above, each followed by repeating the affected gates.

**Final run numbers:** 61,326 tracks → **57,990 images** (mean 159.8, median 106 per file; max 926). S6 rejections: too short
29,848, low observed fraction 4,486, harmonic 7,782, synchronous comb 699, duplicate 10, too steep 3. S8: 3,336 tracks rejected as
longer than 128 frames (5.4 %). Median whistle = 36 frames (192 ms); 40.9 % of images are shorter than 30 frames. Time: **3.2 s per
file** (19.3 min total): S1 1.31 s (FLAC decode of the 6-channel file) is the largest part, S2 0.36, S3 0.14, S4 0.24, S5 0.19 (GPU;
≈0.65 CPU), S6 0.39, S7–S9 0.07, SHA-256 check 0.13. Disk: 5.8 GB for the run in `normal` mode.
The paper's counts are not targets and are not comparable (different S5/S6, different selection of files).

**Synthetic evaluation (category A; `models/detector/ridgenet-v1/eval_*.json`):** track recall of visible synthetic fundamentals
0.737 (validation, where S6 gate/gap were chosen) and **0.704 (held-out test)**; by median ridge SNR: ≥10 dB 94–99 %, 5–10 dB 52–55 %,
<5 dB ≈ 0 %. 1.02–1.04 tracks per recovered whistle. **0.00 false tracks/min on 4.6 min of contamination-free negatives** (phase-
randomised real backgrounds + synthetic clicks/sweeps). Precision against labels on real-background samples (0.60–0.62) is only a lower
bound because unlabeled real whistles in those backgrounds count as false.

## S5 detector experiments (synthetic category A; they led to `ridgenet-v2c`, which the pipeline uses)
Goal: better detection with at most **+25 % S5 time** (budget defined by measured latency, not parameter count). Measured on a real
60 s spectrogram: depthwise-separable designs are *slower* despite fewer multiply-adds (GPU 1.3–1.7×, CPU 4–6×); 18 channels doubles
CPU time; so the feasible family is 16 channels, 6–7 dense blocks. All candidates: same synthetic stream/seed, same frozen S6,
**paired** whistle-by-whistle comparison on fixed sets (`python whistle.py compare-synth ...`, results in `models/detector/compare_*.json`).

| Version | Change vs v1 | Recall val | 5–10 dB val | Paired vs v1 (val) | S5 time |
|---|---|---|---|---|---|
| v1 | — | 0.737 | 0.554 | — | 1.0× |
| v2a | wider context (7 blocks, ±35 × ±23) | 0.710 | 0.482 | +15 / −32, p = 0.019 (**worse**) | GPU −14 / +5 %, CPU −38 / +17 % (two runs) |
| v2b | data only: SNR emphasis, crossings, time-reversed backgrounds (trained from a separate session; evaluated here) | 0.739 | 0.571 | +14 / −13, p = 1 (no effect) | 1.0× (v1's network) |
| **v2c** | v2b + **background-label fix** | **0.760** | **0.631** | **+30 / −16, p = 0.054** | **1.0× (v1's network)** |
| v2d | v2c + wider context | 0.759 | 0.619 | +23 / −10; vs v2c +16 / −17, p = 1 | GPU +20 %, CPU +60 % (one run) |
| v2e | label fix with v1's SNR balance | 0.755 | 0.625 | +24 / −13; vs v2c +10 / −13, p = 0.68 | 1.0× (v1's network) |

S5 time = change against v1 per 60-s file, as stored in `compare_*.json`. v2b, v2c and v2e are v1's network (14,209 weights)
with other training data, yet their stored times range from −34 % to +0 % on the CPU: one comparison run's timing is noise of
that size, so only v2d's slowdown is clearly real.

* **Background-label fix** (`synth.labels.ignore_background_ridges`): a label-free classical statistic (median dB-above-floor along
  25-frame lines, max over 13 slopes; threshold calibrated on phase-randomised noise) marks probable *real* faint whistles in the
  training backgrounds; those nodes get loss weight 0 instead of being taught as negatives. Masks 2.2 % of background nodes, ~70 % of
  which is real structure (0.65 % noise loss). Isolated effect, v2c vs v2b: **+24 / −11, p = 0.041**.
* **Held-out test, v1 vs v2c (run once):** recall **0.704 → 0.729**, 5–10 dB **0.519 → 0.647**, paired **+28 / −14, p = 0.044**,
  0.00 false tracks/min on contamination-free negatives. Cost: clear whistles (≥ 10 dB) drop 3–4 points; the same dip appears in
  v2b/v2e, so it comes from time reversal and/or the extra crossing scenes (not yet separated), not from the label fix or SNR mix.
* **Real data (category B, no ground truth), 10 Phase-3 files:** v2c keeps 82–100 % of v1's tracks and adds 13 % more; the extra tracks
  inspected in a quiet file sit on faint visible ridges. S5 time unchanged (0.23 → 0.21 s); S6 0.62 → 0.66 s (more tracks to link).
* **v2c is adopted** (`[decisions.s5].checkpoint`) for the multi-encounter dataset; the S6 threshold 0.5 was kept (not
  recalibrated for v2c). The check runs and the dataset build were re-run with it (see "Dataset record").

## Known limitations (observed, not hidden)
* **No real-data accuracy is known.** Only synthetic metrics and engineering validation exist; do not read them as recall/precision on
  real whistles. The S5 detector is a replacement and is **not** claimed equivalent to the paper's.
* Encounter 191, the only source of detector backgrounds, is whistle-dense, so clean background windows are scarce (middle of the
  encounter unusable); synthetic validation therefore covers quiet-to-moderate backgrounds, and dense scenes were judged only by
  inspection (the v1 figures of that inspection were deleted with the run folders on 2026-10-21; the v2c check runs keep theirs in
  `preprocessing_checks/<run_id>/diagnostics/`).
* S5 is nearly blind below about 5 dB local SNR (synthetic test recall 0 % for v1, 5 % for v2c); S6 can fragment or swap
  identities in very dense crossings.
* The comb rule can discard a real whistle that starts at the same instant as a pulse-train lattice (observed once in 15 files).
* `min_track_frames = 12` admits many short fragments (37 % of the dataset's images < 30 frames) and tracks > 128 frames are
  rejected. Both are replacement decisions; `images.csv` carries `n_frames`, so a dataset-level filter can be applied later **without**
  re-running preprocessing — but that is a deliberate downstream decision, not something tuned to the paper.
* Single channel (index 0, arbitrary); channel fusion is not implemented.

## VAE-GMM (training, GMM, evaluation)
Authoritative: `resources/TRAINING_SPECIFICATION.md` (discrepancy report, corrected specification, decisions, deviations,
gates); configuration `configs/vaegmm.toml` (its `[paper]` section is locked in `vaegmm/config.py`). Deviations from
stated paper values: the batch size on one GPU (paper 7 x 128 = 896) with learning rates scaled linearly, lr x B / 896, and
the reduced scope below (spec §2.1).
* **Main run V1** (Colab A100 80 GB): encounter 191 only, 150 epochs, **B = 5772, lr 6.442e-3 (VAE) / 1.288e-3 (critic)**, 11 steps
  per epoch, projected about 5.3 h. It also uses cuDNN autotuning (same maths, not bit-deterministic; spec §3.9). GMM at K = 3,000.
  Its fallback run `V1_L64_b1e-4_critic_E191_B896` uses the paper's batch and learning rates. The earlier all-data run
  `V1_L64_b1e-4_critic` (B = 5376) stopped at epoch 7 and is not used.
* **Laptop gates and the `configs/vaegmm.toml` default:** B = 128, 1.4286e-4 / 2.8571e-5.

Every run's actual values are in its `train_info.json` (`spec`, `deviations`, `runtime`).
```bash
python whistle.py vae-experiments                         # experiment matrix V1-V8 (spec §3.8)
python whistle.py vae-gates                               # gates that do not train: G1 data, G2 numerics, G6 features
python whistle.py vae-gates --gate all --allow-training   # + G3 smoke, G4 overfit, G5 timing (short trainings)
python whistle.py train-vae --experiment V1               # main model, 150 epochs on encounter 191 (--resume continues an interrupted run)
# Colab A100: colab/train_V1_colab.ipynb (code bundle: python colab/make_colab_bundle.py)
python whistle.py fit-gmm --experiment V1 --K 3000        # encode unrotated means, full-covariance EM (at most about 45 min)
python whistle.py vae-eval recon --experiment V1          # Table 1
python whistle.py fetch-eval-weights --approved           # Inception-V3 + VGG-16 weights (only after approval)
python whistle.py vae-eval gen --experiment V1 --K 10000  # Table 2 (FID1, FID2, P/R/F1); --source prior for Table 4 (1)
python whistle.py vae-eval cluster --experiment V1        # Fig. 11 (needs the 7 K fits)
python whistle.py vae-eval components --experiment V1     # 500 samples per component (Fig. 8)
python whistle.py vae-eval conditional --experiment V1    # Table 3 + Fig. 12
python whistle.py vae-eval cost --experiment V1           # Table 6
python whistle.py vae-eval figures --experiment V1        # Figs. 7-10
python whistle.py train-ldm --experiment V1 --T 250 500 1000   # Table 5 LDM baseline
```
Outputs: `models/vaegmm/<run_id>/` (`train_info.json` with full lineage, `train_log.jsonl`, `checkpoints/`, `latents_mu.npy`,
`gmm_K<k>/`, `eval/`, `figures/`); gate records in `models/vaegmm/_gates/`. Runs and GMM fits are never overwritten.

## Inference and the web demo
A recording goes through S2-S9 exactly as the training data did (`configs/pipeline.toml`; the processing hash is checked
against `preprocessed/dataset_info.json`), then through the trained model, and every image is turned back into sound by
the paper's own Fig. 2 chain (threshold 0.05 -> contour -> constant-amplitude tone) at the whistle's own centre frequency
(`sonify.py`). Settings: `configs/inference.toml`. Uploads: WAV/FLAC/OGG/MP3, any rate >= 32 kHz (below 48 kHz the top of
the 2-24 kHz detector band is missing and the result says so), any number of channels (one is analysed: `--channel` or
the page's Channel box, default 0), up to 120 s.
```bash
python whistle.py model-status              # are the trained files where inference looks for them?
python whistle.py infer recording.wav       # -> runs/infer/<id>/: result.json, overview.png, per-whistle PNG + WAV
python whistle.py generate --n 8            # new whistles from the GMM; --condition 1..13 = Table 3, --fc 9000 fixes the pitch
python whistle.py serve                     # the page + API on http://127.0.0.1:8000 ("Try it" section)
python whistle.py export-web                # example excerpts + pre-rendered results -> web/data/ (static hosting)
python whistle.py check-rates               # detector on 96 / 48 kHz copies of real files vs the 500 kHz originals
                                            #   (--rates 96000 48000 44100 adds 44.1 kHz)
```
The trained files are read from where the training commands write them; nothing else needs to change after training
(`serve` also offers `POST /api/reload-model`):

| File | Written by | Enables |
|---|---|---|
| `models/vaegmm/V1_L64_b1e-4_critic_E191/checkpoints/last.pt` | `train-vae --experiment V1` (every epoch; equals `final.pt` at the end) | recreation (decoder of the encoder mean); a mid-training checkpoint is labelled as such |
| `models/vaegmm/V1_L64_b1e-4_critic_E191/gmm_K3000/gmm.npz` + `gmm.json` | `fit-gmm --experiment V1` | whistle type (eqs. 11-12), new whistles of that type, generation |
| `models/vaegmm/V1_L64_b1e-4_critic_E191/gmm_K3000/component_features.npy` | `vae-eval components --experiment V1` | generation under a Table 3 condition (eqs. 9-10 + rejection) |

Until the files exist, the input side (detection, images, the recording and its contour as sound) works and the model
panels say what is missing. With the VAE but no GMM, "new whistles" come from the encoder's own spread (a standard VAE
step, labelled as not the paper's method). Checked so far: on 25 real training images, the tone run back through S2-S4
follows the image's contour to a median of 0.6 Hz per image (worst image 1.1 Hz; re-measured 2026-10-31 with the round trip
of tests/test_sonify.py, which itself only requires < 23 Hz, a quarter of a row); the detector on 96 / 48 / 44.1 kHz copies
of 8 real files (one per encounter, 1,174 tracks) finds 99.8 / 99.4 / 99.7 % of the 500 kHz tracks at 99.7 / 99.5 / 99.1 %
precision (2026-10-23, `python whistle.py check-rates --rates 96000 48000 44100`; record in `runs/infer/_checks/`).
