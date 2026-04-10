# SDINet-v2 — Claude Code Guide

## GPU environment notes

**TITAN V GPUs (device IDs 0, 1, 3, 4)** require `LD_PRELOAD=/tmp/nvml_stub.so` to avoid an NVML symbol error (`nvmlDeviceGetNvLinkRemoteDeviceType`) at backward pass. Always prefix commands with it when using these devices:

```
CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=/tmp/nvml_stub.so python train.py ...
```

## Architecture reference

See **`MODEL_ARCHITECTURE.md`** for annotated diagrams of all three heads (v1, DR, C) — input/output shapes, sensor independence, importance weighting, and softmax clarification.

## Research framing

See **`RESEARCH.md`** for the evolving research problem statement, central thesis, dataset roles, and open questions. Update it as the research discussion develops.

## What this repo is

Structural Damage Identification (SHM) neural network. Sensor accelerometer data → damage location + severity. Three model heads:

- **v1** (`Midn`): single global damage scalar + softmax location (original Engineering Structures paper)
- **Approach C** (`MidnC`): DETR-style slot prediction with Hungarian matching
- **Approach DR** (`MidnDR`): per-location regression with MIL sensor attention
- **Baseline B** (`PlainDR`): plain per-location regression (mean-pool over sensors, no learned attention) — isolates MIL contribution

Current development focus: **sensor-fault-aware SDI** — extending the C-head (MidnC) with a sensor spatial reasoning layer that leverages known sensor layout to distinguish structural damage signatures (spatially coherent) from sensor faults (spatially incoherent), without any external fault oracle.

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

## Data

### 7-story frame (simulated, L=70)
- Root: `data/7-story-frame/safetensors/unc=0/{single,double}/`
- Single-damage: K=1 per sample; Double-damage: K=2 per sample
- Combined single+double: ~28k train / 6k val / 6k test
- Real benchmark: `data/7-story-frame/Testing_SingleEAcc9Sensor0.5sec.mat` (physical units, 1 sample)
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
- Double Damage (5 files, real double-damage) → by default test only; can be added to training via `--held-out-double` or `--split-double`
- Windowing: `window_size` (default 2048=2s) + `overlap` (default 0.5) in data loader
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
- Windowing: `window_size` (default 2048) + `overlap` (default 0.5) + `downsample` (default 4) — identical to Qatar defaults → `T=512`
- Split: recording-level stratified across {healthy, DAM3, DAM4, DAM6} so every position appears in train/val/test; 5 recordings per damage folder → 3/1/1 per folder
- Augmentation: via `lib/faults.py` (see **Data augmentation** section below)
- Structured mask groups: 9 level pairs (2 sensors) + 2 axis groups (9 sensors) = 11 groups (`build_struct_masks_lumo`)
- Use `--dataset lumo` in `train.py` / `train_all.py`; `--root` overrides default path
- `--window-size`, `--overlap`, `--downsample` flags share the Qatar CLI and work unchanged

## Training output

- Checkpoints: `states/<run_name>/<model>-combined-<uuid>.pt`
- Calibration: `states/<run_name>/<model>-combined-<uuid>.json` (alongside checkpoint)
- Plots: `saved_results/<run_name>/<model>/` — loss curves, val_map_mse, val_acc
- Per-epoch progress bar: `train_loss | val_loss | val_map_mse | val_top_k_recall` (v1) or `val_f1` (C/DR)
- Final eval after training: test split(s) + real benchmark (7-story only) + double-damage test (Qatar only)

`<run_name>` = `<dataset>` by default, or `<dataset>-<tag>` when `--tag` is used (e.g. `qatar-pmix`).

## Key train.py flags

| Flag | Default | Notes |
|------|---------|-------|
| `--tag <str>` | `""` | Suffix for output dirs — keeps experimental runs separate from baseline |
| `--p-hard <float>` | `0.0` | Per-sensor probability of hard fault (zero-out) during training |
| `--p-soft <float>` | `0.0` | Per-sensor probability of soft fault during training |
| `--p-struct-mask <float>` | `0.0` | Per-window probability of structured group masking |
| `--use-fault-head` | `False` | C-head: enable per-sensor binary fault classifier |
| `--use-structural-bias` | `False` | C-head: learnable location-sensor affinity bias in cross-attention |
| `--held-out-double <0-4>` | `None` | Qatar only: hold out one double-damage recording for test, add other 4 to training |
| `--split-double` | `False` | Qatar only: use first½ of all 5 double-damage recordings for training, second½ for test. Mutually exclusive with `--held-out-double` |
| `--no-obj-weight` | `0.1` | C-head: down-weight for ∅ class in location CE |
| `--sev-weight` | `0.0`* | C-head severity loss weight (`*` Qatar registry default) |
| `--num-slots` | `5` | C-head: K_max detection slots |

## Evaluation output

`evaluate.py` prints a metric table **and** saves results automatically:
- `saved_results/<dataset>/eval_<timestamp>.json` — full nested dict (NaN → null)
- `saved_results/<dataset>/eval_<timestamp>.csv` — same table as printed

Use `--out <stem>` to override the filename (e.g. `--out saved_results/qatar/eval_pmix`).

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
`inject_faults_batch(x, fault_type, n_faulted, rng)` in `lib/faults.py` — injects exactly one fault type with exactly `n_faulted` sensors per sample, seeded for reproducibility. Used by `evaluate_fault.py` to sweep controlled (fault_type × n_faulted) grids.

## Multi-damage training modes (Qatar)

Two modes for incorporating double-damage data, controlled by flags on `train.py` / `train_all.py` / `train_loo.py`:

### LOO with real double-damage (`--held-out-double <0-4>`)
Hold out one double-damage recording for test, add the other 4 to training.
Use `train_loo.py` to run all 5 folds automatically.
Result: mean F1=0.439 (range 0.055–0.740) — high variance due to spatial specificity.

### All-5 time split (`--split-double`)
First 50% of each of the 5 double-damage recordings → training. Second 50% → test.
Mutually exclusive with `--held-out-double`.
Result: F1=0.9996 — near-perfect, confirming spatial specificity as root cause (see Insight 18).

**Key finding**: C-head achieves near-perfect double-damage detection when it has seen the specific joint pair during training.

## Checkpoints in states/

- `states/qatar/` — baseline Qatar models (v1, C, DR trained on single-damage only)
- `states/qatar-dd/` — Qatar C LOO models (4 real double-damage recordings in training, 1 held out per fold)
- `states/qatar-dd-split/` — Qatar v1, C, DR trained with first½ of all 5 double-damage recordings
- `states/qatar-fault/` — v1, C, DR trained with sensor fault augmentation; C-head eval: `saved_results/qatar-fault/eval_fault.{json,csv}`
- `states/qatar-fault-sb/` — C-head with structural-bias attention (`--use-structural-bias`); attention maps saved; eval: `saved_results/qatar-fault-sb/eval_fault.{json,csv}`
- `states/qatar-spatial-fault/` — C-head with spatial reasoning layer (`--use-spatial-layer`); eval: `saved_results/qatar-spatial-fault/eval_fault.{json,csv}`
- `states/7story/` — baseline 7-story models (v1, C, DR; full S=65 sensors; single+double combined training)
- `states/7story-sparse/` — sparse-sensor 7-story models (v1, C, DR; S=9 sensors matching physical benchmark)
- `states/7story-sparse-single/` — v1 only, sparse S=9, single-damage training only (no eval on record)
- `states/7story-fault/` — v1, C, DR trained with fault aug; eval: `saved_results/7story-fault/eval_fault_{v1,dr,c}.{json,csv}`
- `states/7story-fault-fh/` — C+fault-head (`--use-fault-head`); eval: `saved_results/7story-fault-fh/eval_fault.{json,csv}`
- `states/7story-fault-sr/` — C+spatial-layer+fault-head (`--use-fault-head --use-spatial-layer`); eval: `saved_results/7story-fault-sr/eval_fault.{json,csv}`
- `states/7story-fault-sb/` — C+fault-head+structural-bias (`--use-fault-head --use-structural-bias`); eval: `saved_results/7story-fault-sb/eval_fault.{json,csv}`
- `states/lumo/` — baseline LUMO models (v1, C, DR; 18 accel sensors; single-damage binary labels); eval: `saved_results/lumo/eval_20260410_021906.{json,csv}`
- `states/tower/` — tower models

## Experiment results (7-story frame)

Eval files: `saved_results/7story/eval_20260402_023724.json` (full S=65), `saved_results/7story-sparse/eval_20260402_123221.json` (sparse S=9).

### Simulated test set (single and double damage, clean)

| Model | Config | Single F1 | Single top-k | Double F1 | Double top-k |
|-------|--------|-----------|-------------|-----------|-------------|
| v1    | full   | 0.981     | 0.999       | 0.704     | 0.663       |
| C     | full   | 0.999     | 1.000       | 0.942     | 0.939       |
| DR    | full   | 0.924     | 0.924       | 0.849     | 0.837       |
| v1    | sparse | 0.984     | 0.998       | 0.688     | 0.659       |
| C     | sparse | 0.994     | 0.997       | 0.927     | 0.925       |
| DR    | sparse | 0.961     | 0.989       | 0.871     | 0.889       |

### Real physical benchmark (`Testing_SingleEAcc9Sensor0.5sec.mat`)

| Model | Config | real-1dmg F1 | real-1dmg top-k | real-2dmg F1 | real-2dmg top-k |
|-------|--------|-------------|----------------|-------------|----------------|
| v1    | full   | 0.000       | 0.000          | 0.000       | 0.000          |
| C     | full   | 0.000       | 0.000          | 0.000       | 0.000          |
| DR    | full   | 0.167       | 0.000          | 0.000       | 0.000          |
| v1    | sparse | 0.000       | 0.000          | 0.200       | 0.000          |
| C     | sparse | 0.000       | 0.000          | 0.000       | 0.000          |
| DR    | sparse | 0.400       | 1.000          | 0.133       | 0.000          |

Real benchmark performance is very poor across all models — significant sim-to-real gap. DR sparse is the best but still weak. C completely fails the real benchmark despite near-perfect simulation performance.

### Fault-aware (`states/7story-fault/`, `states/7story-fault-fh/`, `states/7story-fault-sb/`)
Eval: `saved_results/7story-fault/eval_fault_{v1,dr,c}.{json,csv}`, `saved_results/7story-fault-fh/eval_fault.{json,csv}`, `saved_results/7story-fault-sb/eval_fault.{json,csv}`
Training: fault aug (200 epochs, old flag interface; now replaced by `--p-hard` / `--p-soft`). All 6 variants complete.

**SDI F1 — hard faults, single damage:**

| Model    | clean | nf=1  | nf=5  | nf=10 | nf=15 |
|----------|-------|-------|-------|-------|-------|
| v1       | 0.984 | 0.984 | 0.981 | 0.977 | 0.970 |
| DR       | 0.944 | 0.944 | 0.941 | 0.935 | 0.929 |
| C        | 0.999 | 0.994 | 0.974 | 0.945 | 0.918 |
| C+fh     | 0.998 | 0.993 | 0.972 | 0.942 | 0.908 |
| C+sr+fh  | 0.999 | 0.991 | 0.962 | 0.925 | 0.885 |
| C+fh+sb  | 0.999 | 0.997 | 0.989 | 0.975 | 0.958 |

**Mean SDI F1 averaged across all 7 fault types × nf∈{1,3,5,10,15}:**

| Model    | single | double | mean   |
|----------|--------|--------|--------|
| v1       | 0.9640 | 0.6960 | 0.8300 |
| DR       | 0.8308 | 0.7863 | 0.8085 |
| C        | 0.9591 | 0.8866 | 0.9229 |
| C+fh     | 0.9617 | 0.9015 | 0.9316 |
| C+sr+fh  | 0.9461 | 0.9002 | 0.9232 |
| C+fh+sb  | 0.9693 | 0.8917 | 0.9305 |

**Fault detection F1 (fault head output) @ nf=5, single damage:**

| Model    | hard  | gain  | bias  | gain_bias | noise | stuck | partial |
|----------|-------|-------|-------|-----------|-------|-------|---------|
| C+fh     | 0.000 | 0.087 | 0.156 | 0.091     | 0.059 | 0.000 | 0.078   |
| C+sr+fh  | 0.008 | 0.049 | 0.045 | 0.038     | 0.003 | 0.010 | 0.045   |
| C+fh+sb  | 0.155 | 0.136 | 0.131 | 0.133     | 0.132 | 0.142 | 0.126   |

Key findings:
- **C+fh+sb is best for single-damage SDI robustness** (highest avg F1 across all fault types/levels).
- **C+fh is marginally best overall** (better on double damage), essentially tied with C+fh+sb.
- **C+sr+fh (spatial layer) hurts single-damage robustness** relative to plain C+fh — the spatial self-attention does not help on 7-story's complex 65-sensor layout (contrast: spatial layer also underperformed on Qatar).
- **DR collapses on bias and noise faults** (F1 drops to 0.57–0.61 at nf=10) — vulnerable to soft faults.
- **Fault detection fails on 7-story** (contrast with Qatar fault F1 ≈ 1.0): the fault head does not reliably identify faulty sensors. C+fh+sb shows marginal improvement but all variants are far from useful. Likely cause: 65-sensor complex layout makes per-sensor anomaly detection harder from memory embeddings alone.
- v1 is fault-robust for SDI but architecturally capped at double-damage F1 ≈ 0.70.

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

### Double-damage (`states/qatar-dd/`, `states/qatar-dd-split/`)

Results from `train.py` terminal output (not saved to JSON — checkpoints in `states/qatar-dd-split/`).

| Run | Model | K=1 F1 | K=2 F1 | K=2 top-k recall | K=2 AP | Notes |
|-----|-------|--------|--------|-----------------|--------|-------|
| LOO (`--held-out-double`) | C | — | 0.439 mean (0.055–0.740) | — | — | high variance by joint pair |
| Split-½ (`--split-double`) | C | 0.992 | 0.9996 | — | — | seen joint pairs → near-perfect |
| Split-½ (`--split-double`) | v1 | 0.978 | 0.667 | 0.552 | 0.630 | **architectural ceiling**: single argmax → precision=1.0, recall=0.5 |
| Split-½ (`--split-double`) | DR | 0.993 | **0.995** | 0.998 | 1.000 | near-perfect; DR regresses per-location so handles K=2 naturally |

Key insight: DR with split-½ training matches C on double-damage (F1≈1.0). v1 is architecturally capped at F1=0.667 (single prediction head). The C vs DR gap on double-damage disappears when both have seen the target joint pairs — the advantage of C's slot architecture matters most in the zero-shot generalisation regime.

### Fault-aware (`states/qatar-fault/`, `states/qatar-fault-sb/`, `states/qatar-spatial-fault/`)
Eval: `eval_fault.json` in each corresponding `saved_results/` folder (fault sweep: hard, 0–15 sensors)

| Run | Model | SDI F1 (clean) | SDI F1 @ n_fault=10 | Fault det. F1 @ n_fault=1 |
|-----|-------|---------------|----------------------|--------------------------|
| `qatar-fault` | C | 0.992 | 0.986 | ≈1.0 |
| `qatar-fault-sb` | C + structural bias | 0.994 | 0.986 | ≈1.0 |
| `qatar-spatial-fault` | C + spatial layer | 0.952 | 0.911 | ≈1.0 |

Note: `qatar-fault` also has v1 and DR checkpoints but fault eval JSON covers C-head only. v1/DR fault robustness of the *baseline* models is in `saved_results/qatar/eval_fault_baseline_{v1,dr}.json`.

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
