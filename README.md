# whistlepipe — preprocessing implementation

Implements S1–S9 of `resources/PREPROCESSING_SPECIFICATION.md` (the authoritative specification; nothing from it is
repeated here beyond what is needed to run and extend the code). The VAE/GMM is **not** part of this project.

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
encounter_191_audio/   RAW, immutable. Opened read-only; storage.py refuses any write inside it.
metadata/  resources/  provided inputs and the specification
configs/pipeline.toml  paper-defined [paper] + implementation decisions [decisions.*] + storage + validation sets
configs/detector.toml  S5 architecture, synthetic-data generator, training  (all [REPLACEMENT])
src/whistlepipe/       audio.py(S1) dsp.py(S2-S4) detector/(S5) tracking.py(S6) representation.py(S7-S9)
                       pipeline.py(orchestrator) storage.py(lineage/policy) checks.py plotting.py cli.py
                       synth/(synthetic training data)  train.py  evaluate.py
tests/                 pytest, small synthetic fixtures; none needs the full dataset
models/detector/<ver>/ checkpoint.pt, train_info.json, calibration.json, eval_*.json
data/synthetic/        background banks (real S3 audio windows) and cached synthetic val/test sets
runs/<run_id>/         one self-contained folder per run (never overwritten): see "Storage and lineage"
```

## Commands (from the project root)
```bash
python -m pytest                                   # all automated tests
python whistle.py list-sets                        # deterministic validation file sets (val1 / val5 / val10 / all)
python whistle.py build-backgrounds                # real-background banks for synthetic training (S1-S3 on 56 non-validation files)
python whistle.py train --version ridgenet-v1      # train S5 on SYNTHETIC data (checkpoints are never overwritten)
python whistle.py eval-synth --split val --calibrate   # synthetic validation + threshold calibration
python whistle.py eval-synth --split test              # synthetic test
python whistle.py run --run-id phase1_1file  --set val1  --last s4 --storage debug
python whistle.py run --run-id phase2_5files --set val5  --storage debug
python whistle.py run --run-id phase3_10files --set val10 --storage debug
python whistle.py validate-run --run-id phase3_10files   # automated gate: lineage, output validity, cross-stage counts
python whistle.py report --run-id phase3_10files         # compact per-file table
python whistle.py run --run-id phase4_full363_r2 --set all --storage normal     # only after phases 1-3 pass (see Validation record)
```
Any stage range: `--first s5 --last s6 --from-run runs/<earlier run>` (needs that run to hold the previous stage's
intermediate). `--override key=value` changes a config value for one run and is recorded in the run manifest.

## Storage and lineage
`runs/<run_id>/` contains `run_manifest.json` (full config snapshot + hashes, software versions, detector checkpoint
SHA-256, file list, options), `summary.json`, `logs/`, `intermediate/<stem>/`, `final/<stem>/`, `meta/<stem>.json`,
`failed/<stem>/`, `diagnostics/`.
* **Lineage** `meta/<stem>.json`: raw path/size/mtime/**SHA-256 (verified against `metadata/manifest_363.csv`)** → config
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
Train/val/test use disjoint background files. **Known limitations:** the encounter is whistle-dense, so clean windows are
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

## Validation record (runs are kept for lineage; the FINAL dataset is `runs/phase4_full363_r2/`)
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

## Known limitations (observed, not hidden)
* **No real-data accuracy is known.** Only synthetic metrics and engineering validation exist; do not read them as recall/precision on
  real whistles. The S5 detector is a replacement and is **not** claimed equivalent to the paper's.
* The encounter is whistle-dense, so clean background windows are scarce (middle of the encounter unusable); synthetic validation
  therefore covers quiet-to-moderate backgrounds, and dense scenes were judged only by inspection (figures in `runs/*/diagnostics/`).
* S5 is blind below about 5 dB local SNR; S6 can fragment or swap identities in very dense crossings.
* The comb rule can discard a real whistle that starts at the same instant as a pulse-train lattice (observed once in 15 files).
* `min_track_frames = 12` admits many short fragments (41 % of images < 30 frames) and tracks > 128 frames are rejected. Both are
  replacement decisions; `final/images_index.csv` carries `n_frames`, so a dataset-level filter can be applied later **without**
  re-running preprocessing — but that is a deliberate downstream decision, not something tuned to the paper.
* Single channel (index 0, arbitrary); channel fusion is not implemented.
