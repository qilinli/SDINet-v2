# SDINet-v2 — Claude Code Guide

## GPU environment notes

**TITAN V GPUs (device IDs 0, 1, 3, 4)** require `LD_PRELOAD=/tmp/nvml_stub.so` to avoid an NVML symbol error (`nvmlDeviceGetNvLinkRemoteDeviceType`) at backward pass. Always prefix commands with it when using these devices:

```
CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=/tmp/nvml_stub.so python train.py ...
```

## Architecture reference

See **`MODEL_ARCHITECTURE.md`** for annotated diagrams of the v1, DR, and C heads — input/output shapes, sensor independence, importance weighting, and softmax clarification. (B shares DR's architecture with mean-pooling replacing learned attention.)

## Research framing

See **`RESEARCH.md`** for the evolving research problem statement, central thesis, dataset roles, and open questions. Update it as the research discussion develops.

## What this repo is

Structural Damage Identification (SHM) neural network. Sensor accelerometer data → damage location + severity. Four model heads:

- **v1** (`Midn`): single global damage scalar + softmax location (original Engineering Structures paper)
- **Approach C** (`MidnC`): DETR-style slot prediction with Hungarian matching
- **Approach DR** (`MidnDR`): per-location regression with MIL sensor attention
- **Baseline B** (`PlainDR`): plain per-location regression (mean-pool over sensors, no learned attention) — isolates MIL contribution

Current development focus: **fault-robust SDI across the sim→lab→field trio** — evaluating all model heads (v1, C, DR, B) plus C-head variants (C+fh with fault detection head, C+fh+sb with structural affinity bias in cross-attention routing) under controlled sensor fault injection. The structural affinity bias (`R_bias`) injects the known sensor layout as a routing prior in attention logit space, guiding each damage slot to attend toward sensors near its predicted location without modifying sensor feature embeddings. A spatial self-attention approach (`SensorSpatialLayer`) was tried and abandoned — it contaminated clean sensor representations (see Domain Insight 21).

## Collaboration workflow

**Discussion-first, implementation-second.** When discussing research direction or architecture, stay at framework level (what, why, trade-offs). Do NOT produce implementation plans (file changes, code sketches, CLI flags) until the user explicitly asks to proceed with implementation. This prevents wasting context on plans that may change.

**Run commands always on one line.** When providing any command to run (training, evaluation, etc.), always give it as a single line so it can be copied and pasted directly. Never use multi-line `\` continuations.

## Entry points

| Script | Purpose |
|--------|---------|
| `python train.py --model v1 --dataset 7story [--epochs N]` | **Primary**: train any model on any dataset |
| `python train.py --model c  --dataset qatar  [--epochs N]` | e.g. Approach-C on Qatar |
| `python train.py --model dr --dataset tower  [--epochs N]` | e.g. DR on Tower |
| `python train.py --model c  --dataset lumo   [--epochs N]` | e.g. Approach-C on LUMO (field) |
| `python train.py --model b  --dataset lumo   [--epochs N]` | e.g. Baseline B (plain regression) on LUMO |
| `python train_all.py --dataset 7story [--models v1 c dr b] [--epochs N]` | Train multiple models sequentially |
| `python evaluate.py --dataset 7story --v1 <ckpt> --c <ckpt> --dr <ckpt> --b <ckpt>` | Side-by-side metric table, saves JSON+CSV |
| `python inspect_predictions.py --model c --dataset tower [--ckpt <ckpt>]` | Prediction visualisation plots |

### Dataset registry

`lib/datasets.py` — `DatasetConfig` registry. To add a dataset: (1) write a `_loader_<name>` wrapper, (2) add a `DatasetConfig(...)` entry to `DATASETS`, (3) add one `elif` in `train.py`'s `_build_dl_kwargs` (~3 lines). Nothing else changes.

## Key library files

| File | Role |
|------|------|
| `lib/midn.py` | `Midn` (v1 head), `MidnC` (C head), `MidnDR` (DR head), `PlainDR` (baseline B head) |
| `lib/model.py` | `ModelConfig`, `ModelConfigC`, `ModelConfigDR`, `build_model()`, `build_criterion_c()`; `load_model_from_checkpoint`, `load_model_c_from_checkpoint`, `load_model_dr_from_checkpoint` |
| `lib/faults.py` | `apply_signal_aug`, `apply_fault_aug`, `inject_faults_batch`, `FAULT_TYPES` — unified augmentation for all datasets |
| `lib/losses.py` | `SetCriterion` (DETR-style Hungarian matching loss for C) |
| `lib/metrics.py` | **Canonical metrics**: `evaluate_all_v1` (v1), `evaluate_all_c` (C), `evaluate_all_dr` (DR), `distributed_map_v1/c/dr`, `map_mse`, `top_k_recall`, `average_precision`, `severity_mae_at_detected`, `f1_from_counts` |
| `lib/training.py` | `do_training` (v1), `do_training_c` (C), `do_training_dr` (DR) |
| `lib/data_7story.py` | `get_7story_dataloaders(["single","double"], ...)` — 7-story frame loader; `load_real_test_tensors`, `DEFAULT_BENCHMARK`, `TWO_DAMAGE_BENCHMARK` — real .mat benchmark access |
| `lib/data_tower.py` | `get_tower_dataloaders(...)` — tower dataset; `TOWER_TIME_LEN=400`, `TOWER_N_SENSORS=6`, `TOWER_N_LOCATIONS=4` |
| `lib/data_qatar.py` | `get_qatar_dataloaders(...)`, `get_qatar_double_test_dataloader(...)` — Qatar SHM Benchmark; `QATAR_FS=1024`, `QATAR_N_SENSORS=30`, `QATAR_N_LOCATIONS=30`; `row_sensors(r)`, `col_sensors(c)`, `mask_sensors(x, indices)` for fault evaluation |
| `lib/data_lumo.py` | `get_lumo_dataloaders(...)` — LUMO field SHM benchmark; `LUMO_FS=1651.61`, `LUMO_N_SENSORS=18`, `LUMO_N_LOCATIONS=3`; binary labels, scenarios = DAM{3,4,6}_{010,111} |
| `lib/visualization.py` | `plot_training_results(...)` — saves loss/mse/metric curves |

## Metrics convention

**All metrics are defined in `lib/metrics.py` and must be used consistently everywhere.**

- `map_mse` — distributed damage map MSE in **[0, 1]** space (primary comparison metric, valid for all heads)
- `top_k_recall` — fraction of true damaged locations in top-K predictions (K = true # damages per sample)
- `ap` — average precision (area under PR curve), threshold-free
- `f1` / `precision` / `recall` — presence detection at threshold
- `severity_mae` — MAE at correctly detected locations only
- `mean_k_pred` / `mean_k_true` — predicted vs true number of damaged locations

For v1: `evaluate_all_v1(dmg_pred, loc_pred, y_norm)` — uses `distributed_map_v1` + softmax ranking
For C:  `evaluate_all_c(loc_logits, severity, y_norm)` — uses `distributed_map_c` + slot ranking
For DR: `evaluate_all_dr(pred, y_norm)` — uses `distributed_map_dr` + direct score ranking

**Labels in normalized space**: `y_norm = raw_damage / 0.15 - 1 ∈ [-1, 1]`. Undamaged = -1, max damaged = +1.

## Signal preprocessing (consistent across all datasets)

All datasets share a two-stage preprocessing pipeline implemented via `scipy.signal.decimate` (FIR, zero-phase) and `lib.preprocessing.normalize_rms`:

1. **FIR anti-alias decimation** — low-pass FIR filter at the new Nyquist frequency, then downsample. Removes high-frequency content that would otherwise alias into the structural response band (<50 Hz). Zero-phase (forward+backward) to avoid inter-channel temporal misalignment. Factor varies by dataset raw Fs.

2. **Global RMS normalization** — divide all sensors by the scalar RMS computed across all sensors and time steps jointly (`ref_channel=None`). Preserves cross-sensor amplitude ratios (the damage signature) while providing excitation-amplitude invariance. Global (all-sensor) reference chosen over single-sensor reference for fault robustness: a faulted reference sensor would corrupt all channels, whereas global RMS degrades gracefully (~1/S perturbation per faulted sensor).

| Dataset | Raw Fs | Decimation | Effective Fs | Nyquist | T | RMS ref |
|---------|--------|-----------|-------------|---------|---|---------|
| 7-story | 1000 Hz | truncate T=500 (zero-padded tail) | 1000 Hz | 500 Hz | 500 | global |
| Qatar | 1024 Hz | 4× | 256 Hz | 128 Hz | 512 | global |
| LUMO | 1652 Hz | 4× | 413 Hz | 207 Hz | 512 | global |

## Data

### 7-story frame (simulated, L=70)
- Root: `data/7-story-frame/safetensors/unc=0/{single,double}/`
- Single-damage: K=1 per sample; Double-damage: K=2 per sample
- Combined single+double: ~28k train / 6k val / 6k test
- Real benchmark: `data/7-story-frame/Testing_SingleEAcc9Sensor0.5sec.mat` (physical units, 1 sample)
- Preprocessing: FIR anti-alias decimation 2× (1000→500 Hz, T=500) + global RMS normalization
- Augmentation: via `lib/faults.py` (see **Data augmentation** section below)
- Structured mask groups: 7 beam + 7 left-col + 7 right-col (3 sensors each) = 21 groups (`build_struct_masks_7story`)

### Tower dataset (physical experiment, L=4)
- Root: `data/tower/`
- 34 recordings, 9 damage states (healthy + DS1–DS8), excitation: EQ / WN / sine
- Input: `X (N, 1, 400, 6)` — 2s windows, 6 accelerometers, normalised by C1 RMS
- Labels: `Y (N, 4)` — `[B6, B1, connection, floor3]` severity ∈ {0, 0.5, 1.0}
- Labels normalised: `y_norm = Y*2−1 ∈ [−1, 1]` (undamaged=−1, max damaged=+1)
- Split: recording-level, stratified by damage category (~15% val, ~15% test)
- Augmentation: via `lib/faults.py` (see **Data augmentation** section below)
- Structured mask groups: 3 floor-level pairs (`build_struct_masks_tower`)
- Use `--dataset tower` in `train.py` / `train_all.py`; `--root` overrides default path

### Qatar SHM Benchmark (physical experiment, L=30)
- Root: `data/Qatar/`
- Build NPZ cache (run once, ~35 min): `python data/Qatar/build_dataset.py`
- Processed output: `data/Qatar/processed/` (default)
- 30 sensors, 30 joints, 1024 Hz, white noise excitation
- Labels: binary {0,1} → y_norm {-1,+1}; no severity grading
- Dataset A (31 files: 1 undamaged + 30 single-damage) → train only
- Dataset B (31 files, same scenarios, independent run) → val (first 50%) / test (last 50%)
- Double Damage (5 files, real double-damage) → 4 included in training by default; j23+j24 (case_id=4) always held out for test
- Windowing: `window_size` (default 2048=2s) + `overlap` (default 0.5) in data loader
- Preprocessing: FIR anti-alias decimation 4× (1024→256 Hz, T=512) + global RMS normalization
- Augmentation: via `lib/faults.py` (see **Data augmentation** section below)
- Structured mask groups: 6 rows (5 sensors) + 5 columns (6 sensors) = 11 groups (`build_struct_masks_qatar`)
- Use `--dataset qatar` in `train.py` / `train_all.py`; `--root` overrides default path
- `--window-size` and `--overlap` flags on `train.py` to vary parametrically

### LUMO field benchmark (long-term ambient SHM, L=3)
- Root: `data/LUMO/`
- **Field / long-term SHM** benchmark completing the *synthetic → lab → field* trio (7-story=sim, Qatar=lab, LUMO=field)
- Leibniz Uni Hannover outdoor lattice tower (9 m), ambient wind excitation — unlike the forced/shaker 7-story and lab white-noise Qatar
- 22 raw channels per 10-min .mat: 18 accel (9 levels × xy, used) + 3 base strain (near-zero variance, dropped) + 1 temp (not used under parity framing)
- **Effective sensor suite: 18 accelerometers**, Fs=1651.61 Hz, modal energy in 2–20 Hz band
- 12 scenario folders = 6 × `*_Healthy` interleaved with 6 damage scenarios: `DAM{3,4,6}_{010,111}` — 010 = only leg-2↔3 bracing removed, 111 = all 3 bracings removed
- **L=3 instrumented damage positions**: DAM3 / DAM4 / DAM6 (DAM1/2/5 do not exist as experimental scenarios). Labels binary `{-1,+1}^3`: +1 iff any bracing at that position is unbolted; 010 and 111 patterns collapse to a single "damaged" label (parity framing, severity out of scope)
- Windowing: `window_size` (default 2048) + `overlap` (default 0.5) + `downsample` (default 4) → `T=512`
- Preprocessing: FIR anti-alias decimation 4× (1652→413 Hz, T=512) applied before windowing + global RMS normalization
- Split: recording-level stratified across {healthy, DAM3, DAM4, DAM6} so every position appears in train/val/test; 5 recordings per damage folder → 3/1/1 per folder
- Augmentation: via `lib/faults.py` (see **Data augmentation** section below)
- Structured mask groups: 9 level pairs (2 sensors) + 2 axis groups (9 sensors) = 11 groups (`build_struct_masks_lumo`)
- Structural affinity: DAM3→ML5+ML6 (ch 8-11), DAM4→ML6+ML7 (ch 10-13), DAM6→ML8+ML9 (ch 14-17); ML6 shared between DAM3/DAM4 (`build_structural_affinity_lumo`)
- Use `--dataset lumo` in `train.py` / `train_all.py`; `--root` overrides default path
- `--window-size`, `--overlap`, `--downsample` flags share the Qatar CLI and work unchanged

## Training output

- Checkpoints: `states/<run_name>/<model>-combined-<uuid>.pt`
- Calibration: `states/<run_name>/<model>-combined-<uuid>.json` (alongside checkpoint)
- Plots: `saved_results/<run_name>/<model>/` — loss curves, val_map_mse, val_acc
- Per-epoch progress bar: `train_loss | val_loss | val_map_mse | val_top_k_recall` (v1) or `val_f1` (C/DR)
- Final eval after training: test split(s) + real benchmark (7-story only) + double-damage test (Qatar only)

`<run_name>` = `<dataset>` by default, or `<dataset>-<tag>` when `--tag` is used (e.g. `qatar-fault`).

## Optimizer / scheduler

| Head | Optimizer | LR | Scheduler | Notes |
|------|-----------|-----|-----------|-------|
| v1, DR, B | AdamW | 5e-4 | OneCycleLR (per-step) | Original tuning from v1 |
| C (all variants) | AdamW | 1e-4 | CosineAnnealingLR (to 0) | DETR-style; OneCycleLR's warmup ramp destabilised fault head loss balance |

Shared: weight_decay=1e-2, betas=(0.9, 0.999). Original DETR uses weight_decay=1e-4; 1e-2 is fine for our small model — revisit if overfitting observed.

## Key train.py flags

| Flag | Default | Notes |
|------|---------|-------|
| `--tag <str>` | `""` | Suffix for output dirs — keeps experimental runs separate from baseline |
| `--p-hard <float>` | `0.0` | Per-sensor probability of hard fault (zero-out) during training |
| `--p-soft <float>` | `0.0` | Per-sensor probability of soft fault during training |
| `--p-struct-mask <float>` | `0.0` | Per-window probability of structured group masking |
| `--use-fault-head` | `False` | C-head: enable per-sensor binary fault classifier |
| `--use-structural-bias` | `False` | C-head: learnable location-sensor affinity bias in cross-attention |
| `--no-obj-weight` | `0.1` | C-head: down-weight for ∅ class in location CE |
| `--sev-weight` | `0.0`* | C-head severity loss weight (`*` Qatar registry default) |
| `--num-slots` | `5` | C-head: K_max detection slots |

## Evaluation output

`evaluate.py` prints a metric table **and** saves results automatically:
- `saved_results/<dataset>/eval_<timestamp>.json` — full nested dict (NaN → null)
- `saved_results/<dataset>/eval_<timestamp>.csv` — same table as printed

Use `--out <stem>` to override the filename (e.g. `--out saved_results/qatar-fault/eval_fault_c-fh`).

`inspect_predictions.py` saves plots to `saved_results/<dataset>/<model>/` (consistent with training output).
Use `--out-dir` to override for non-standard paths.

## Data augmentation

All augmentation is centralised in **`lib/faults.py`** and applied identically across all four datasets via `_AugDataset` wrappers in each data loader. Val/test loaders use raw `TensorDataset` with no augmentation.

### Signal augmentation (always on, every training sample)
`apply_signal_aug(x)`:
1. Global amplitude scaling: `x *= U(0.8, 1.2)`
2. Per-sensor Gaussian noise: `x += N(0, 0.05 * rms_per_sensor)`

### Sensor fault augmentation (opt-in via `--p-hard`, `--p-soft`, `--p-struct-mask`)
`apply_fault_aug(x, n_sensors, struct_masks, p_hard, p_struct, p_soft)`:

Three independent mechanisms applied per sample in order:
1. **Per-sensor hard fault** (`--p-hard`): each sensor independently zeroed with probability `p_hard`. Number of faults is a natural Binomial(S, p_hard) draw.
2. **Structured masking** (`--p-struct-mask`): per-window gate — with probability `p_struct`, one random pre-defined sensor group is zeroed entirely. Models correlated failures (e.g. a floor's wiring harness).
3. **Per-sensor soft fault** (`--p-soft`): each non-hard-faulted sensor independently gets a random soft fault with probability `p_soft`. Six types: gain, bias, gain_bias, noise, stuck, partial — all RMS-relative.

When any fault flag > 0, the data loader returns `(x, y, y_fault)` triplets; otherwise `(x, y)` pairs. `y_fault ∈ {0,1}^S` is the per-sensor fault ground truth, used by the C-head fault detection head (`--use-fault-head`) for BCE supervision. v1/DR discard `y_fault` but still train on faulted inputs for implicit robustness.

### Evaluation-time fault injection
`inject_faults_batch(x, fault_type, n_faulted, rng)` in `lib/faults.py` — injects exactly one fault type with exactly `n_faulted` sensors per sample, seeded for reproducibility. Used by `evaluate_fault.py` for 7-story (per-batch, independent samples).

`_inject_recording(x, sensor_idx, fault_type)` in `evaluate_fault.py` — vectorized recording-consistent injection for LUMO/Qatar. Same sensors faulted across all windows in a recording, one random fault parameter draw per recording (matches physical reality: a faulty sensor stays faulty for the whole measurement session).

## Fault-robust experiment protocol

Consistent protocol applied identically across 7-story, Qatar, and LUMO.

### Training
- **Augmentation**: `--p-hard 0.3 --p-soft 0.3 --p-struct-mask 0.3`
- **Epochs**: 200
- **Tag**: `--tag fault` → checkpoints in `states/<dataset>-fault/`
- **Models**: B, v1, C, C+fh (`--use-fault-head`), C+fh+sb (`--use-fault-head --use-structural-bias`)
  - LUMO: C+fh+sb now supported — structural affinity derived from readme.pdf Figure 1 (DAM3→ML5+ML6, DAM4→ML6+ML7, DAM6→ML8+ML9)
  - DR: not yet trained for fault experiments

### Evaluation (`evaluate_fault.py`)
- **Fault types**: 7 types — hard, gain, bias, gain_bias, noise, stuck, partial
- **Fault ratio**: `--fault-ratio 0.0 0.2 0.5 0.8` (fraction of sensors faulted)
  - 7-story (S=65): nf = {0, 13, 32, 52}
  - Qatar (S=30): nf = {0, 6, 15, 24}
  - LUMO (S=18): nf = {0, 4, 9, 14}
- **Repeats**: `--n-repeats 3` (3 random fault sensor selections per condition)
- **Injection paradigm**:
  - 7-story: per-sample (each simulation sample is independent)
  - Qatar / LUMO: per-recording (fault persists across all windows of a measurement session)
- **Output**: `saved_results/<dataset>-fault/eval_fault_{b,v1,c,c-fh,c-fh-sb}.{json,csv}`
- **7-story and Qatar reports**: single-damage and double-damage subsets combined in one CSV with `subset` column

### Example commands
```
python train.py --model c --dataset qatar --epochs 200 --tag fault --p-hard 0.3 --p-soft 0.3 --p-struct-mask 0.3 --use-fault-head
python evaluate_fault.py --dataset qatar --c states/qatar-fault/c-fh-<uuid>.pt --c-label C+fh --out saved_results/qatar-fault/eval_fault_c-fh
```

## Qatar double-damage handling

Qatar now defaults to including double-damage in training (like 7-story), with no special flags needed:
- 4 double-damage recordings (j03+j26, j07+j14, j13+j23, j21+j25) → included in training
- 1 double-damage recording (j23+j24, case_id=4) → always held out for test (`QATAR_HELD_OUT_DOUBLE = 4` in `lib/datasets.py`)

This is hardcoded — `--held-out-double` and `--split-double` flags have been removed.

### Historical LOO and split experiments (prior to default change)
- LOO (`--held-out-double`): mean F1=0.439 (range 0.055–0.740) — high variance due to spatial specificity (fold 4 = j23+j24 scored 0.740 because j23 overlaps with training fold j13+j23).
- Split-½ (`--split-double`): F1=0.9996 — near-perfect, confirming spatial specificity as root cause (see Insight 18).

**Key finding**: C-head achieves near-perfect double-damage detection when it has seen the specific joint pair during training.

## Checkpoints in states/

- `states/qatar/` — baseline Qatar models (v1, C, DR trained on single-damage only)
- `states/qatar-dd-split/` — Qatar v1, C, DR trained with first½ of all 5 double-damage recordings
- `states/qatar-fault/` — B, v1, C, C+fh, C+fh+sb with fault aug (200 epochs) — **stale: trained on single-damage only + old optimizer; needs retraining with double-damage default + CosineAnnealingLR for C-variants**
- `states/7story/` — **empty, needs retraining** (old checkpoints trained on broken FIR-decimated zero-padded data, deleted)
- `states/7story-fault/` — **empty, needs retraining** (same)
- `states/lumo/` — baseline LUMO models (v1, C, DR; 18 accel sensors; single-damage binary labels); eval: `saved_results/lumo/eval_20260410_021906.{json,csv}`
- `states/lumo-fault/` — B, v1 trained with fault aug (200 epochs, valid); C, C+fh, C+fh+sb **stale: need retraining with pos_weight removed + CosineAnnealingLR**; eval: `saved_results/lumo-fault/eval_fault_{b,v1,c,c-fh}.{json,csv}`
- `states/lumo-lumo-nobj05/` — exploratory: LUMO C-head with `--no-obj-weight 0.05`
- `states/lumo-lumo-slots2/` — exploratory: LUMO C-head with `--num-slots 2`
- `states/tower/` — tower models

## Experiment results (7-story frame)

Stale — preprocessing changed (reverted FIR decimation on zero-padded data, added RMS normalization). Needs retraining and re-evaluation.

## Experiment results (Qatar)

Confirmed results only. Numbers are on the **B-split test set** (single-damage, hard fault sweep).

### Baseline (`states/qatar/`) — single-damage training, no fault aug
Eval: `saved_results/qatar/eval_20260329_053938.json` (SDI) + `eval_fault_baseline_{v1,c,dr}.json` (fault robustness)

| Model | SDI F1 (K=1) | SDI F1 (K=2) | SDI F1 @ n_fault=10 |
|-------|-------------|-------------|----------------------|
| v1    | 0.975       | 0.357       | 0.973                |
| C     | 0.992       | 0.300       | 0.985                |
| DR    | 0.991       | 0.387       | 0.989                |

All models robust to hard faults at the SDI level (fault_f1 not applicable — no fault head).

### Double-damage (`states/qatar-dd-split/`)

Results from `train.py` terminal output (checkpoints in `states/qatar-dd-split/`).

| Run | Model | K=1 F1 | K=2 F1 | K=2 top-k recall | K=2 AP | Notes |
|-----|-------|--------|--------|-----------------|--------|-------|
| LOO | C | — | 0.439 mean (0.055–0.740) | — | — | high variance by joint pair |
| Split-½ | C | 0.992 | 0.9996 | — | — | seen joint pairs → near-perfect |
| Split-½ | v1 | 0.978 | 0.667 | 0.552 | 0.630 | **architectural ceiling**: single argmax → precision=1.0, recall=0.5 |
| Split-½ | DR | 0.993 | **0.995** | 0.998 | 1.000 | near-perfect; DR regresses per-location so handles K=2 naturally |

Key insight: DR with split-½ training matches C on double-damage (F1≈1.0). v1 is architecturally capped at F1=0.667 (single prediction head). The C vs DR gap on double-damage disappears when both have seen the target joint pairs — the advantage of C's slot architecture matters most in the zero-shot generalisation regime.

### Fault-aware (`states/qatar-fault/`)
Training: `--p-hard 0.3 --p-soft 0.3 --p-struct-mask 0.3`, 200 epochs. Models: B, v1, C, C+fh, C+fh+sb.
Eval: `saved_results/qatar-fault/eval_fault_{b,v1,c,c-fh,c-fh-sb}.{json,csv}` — pending retraining and re-eval with consistent fault-ratio sweep.

## Experiment results (LUMO)

Eval file: `saved_results/lumo/eval_20260410_065940.json`.

LUMO is the **field / long-term ambient SHM** benchmark (wind-excited outdoor lattice tower, 18 accel sensors, L=3 binary). Completes the *synthetic (7-story) → lab (Qatar) → field (LUMO)* trio under identical parity protocol (same augmentation, same metric suite, same model code).

### Baseline (`states/lumo/`) — single-damage, binary labels, parity framing

| Model | F1    | precision | recall | top_k_recall | map_mse | ap    | severity_mae | mean_k_pred |
|-------|-------|-----------|--------|-------------|---------|-------|-------------|-------------|
| v1    | 0.943 | 0.924     | 0.963  | 0.973       | 0.018   | 0.845 | 0.055       | 0.625       |
| C     | 0.946 | 0.965     | 0.929  | 0.982       | nan     | nan   | 0.356       | 0.578       |
| DR    | 0.972 | 0.972     | 0.972  | 0.977       | 0.014   | 0.994 | 0.058       | 0.600       |

(`mean_k_true = 0.60` for all; `map_mse`/`ap` nan for C is a known binary-label artefact in `lib/metrics.py`.)

**v1 calibration note**: For small L (e.g. L=3), `max(softmax) ≥ 1/L` always, making the standard beta gate ineffective — v1 cannot predict K=0. Fixed by adding a `dmg_gate` sweep on the global damage scalar during calibration (`dmg_gate=-0.8` for LUMO). This fix also marginally improves Qatar v1 (F1 0.975→0.989).

### Sim → lab → field comparison (single-damage F1)

| Model | 7-story (sim) | Qatar (lab) | LUMO (field) |
|-------|--------------|-------------|-------------|
| v1    | 0.981–0.984  | 0.975       | **0.943**   |
| C     | 0.994–0.999  | 0.992       | **0.946**   |
| DR    | 0.924–0.961  | 0.991       | **0.972**   |

Key findings:
- **DR is most ambient-robust** (F1=0.972, barely degrades from lab). Per-location regression with no obligate "how many damages" prior handles the "is anything wrong at all?" question cleanly.
- **C degrades modestly** (−0.046 from Qatar). The DETR ∅-class slot absorbs healthy-window ambiguity, but not perfectly.
- **v1 competitive after calibration fix** (F1=0.943, −0.032 from Qatar). With the dmg_gate fix, v1 correctly predicts K=0 for most healthy windows (`mean_k_pred=0.625` vs 0.600 true). The remaining gap vs DR/C is from the softmax-location head's forced allocation.
- **All three heads localise damage correctly when it exists** (`top_k_recall ≈ 0.97–0.98`). The gap is entirely about *whether* to call damage, not *where*.
- **Ranking on LUMO**: DR > C ≈ v1. Under ambient field conditions, per-location regression is the most natural fit; the C-head's slot machinery is overhead when K≤1.

### Fault-aware (`states/lumo-fault/`)
Eval: `saved_results/lumo-fault/eval_fault_{b,v1,c,c-fh}.{json,csv}`
Training: `--p-hard 0.3 --p-soft 0.3 --p-struct-mask 0.3`, 200 epochs. Fault-ratio sweep: `{0.0, 0.1, 0.33, 0.5, 0.67, 0.8}` → nf={0,2,6,9,12,14}. *(Protocol now uses `{0.0, 0.2, 0.5, 0.8}` — these results from original sweep.)*

**Mean SDI F1 by fault type (mean across nf={2,6,9,12,14}):**

| Model | hard  | gain  | bias  | gain_bias | noise | stuck | partial | MEAN  |
|-------|-------|-------|-------|-----------|-------|-------|---------|-------|
| B     | 0.927 | 0.941 | 0.933 | 0.930     | 0.932 | 0.928 | 0.933   | 0.932 |
| v1    | 0.943 | 0.966 | 0.963 | 0.963     | 0.960 | 0.940 | 0.959   | 0.956 |
| C     | 0.811 | 0.837 | 0.837 | 0.826     | 0.808 | 0.805 | 0.825   | 0.821 |
| C+fh  | 0.905 | 0.929 | 0.909 | 0.916     | 0.917 | 0.904 | 0.909   | 0.913 |

Key findings:
- **Plain C collapses on LUMO** (F1=0.845 clean, 0.821 mean faulted) — DETR slot machinery is excessive for L=3 binary.
- **C+fh rescues C** (0.845→0.935 clean, +0.092) — fault head provides strong regularization even on clean data.
- **v1 most fault-robust** (mean 0.956), consistent with 7-story.
- **Fault detection works on LUMO** — C+fh achieves fault F1≈0.65–0.70 at nf=6 (vs ≈0.0 on 7-story). Small S=18 makes per-sensor anomaly detection tractable.
