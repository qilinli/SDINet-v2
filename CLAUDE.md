# SDINet-v2 — Claude Code Guide

## Architecture reference

See **`MODEL_ARCHITECTURE.md`** for annotated diagrams of all three heads (v1, DR, C) — input/output shapes, sensor independence, importance weighting, and softmax clarification.

## Research framing

See **`RESEARCH.md`** for the evolving research problem statement, central thesis, dataset roles, and open questions. Update it as the research discussion develops.

## What this repo is

Structural Damage Identification (SHM) neural network. Sensor accelerometer data → damage location + severity. Three model heads:

- **v1** (`Midn`): single global damage scalar + softmax location (original Engineering Structures paper)
- **Approach C** (`MidnC`): DETR-style slot prediction with Hungarian matching
- **Approach DR** (`MidnDR`): direct per-location regression baseline

Current development focus: **v1, DR, C on the Qatar SHM Benchmark** — single-damage localisation, multi-damage generalisation, sensor fault tolerance.

## Entry points

| Script | Purpose |
|--------|---------|
| `python train.py --model v1 --dataset 7story [--epochs N]` | **Primary**: train any model on any dataset |
| `python train.py --model c  --dataset qatar  [--epochs N]` | e.g. Approach-C on Qatar |
| `python train.py --model dr --dataset tower  [--epochs N]` | e.g. DR on Tower |
| `python train_all.py --dataset 7story [--models v1 c dr] [--epochs N]` | Train multiple models sequentially |
| `python evaluate.py --dataset 7story --v1 <ckpt> --c <ckpt> --dr <ckpt>` | Side-by-side metric table, saves JSON+CSV |
| `python inspect_predictions.py --model c --dataset tower [--ckpt <ckpt>]` | Prediction visualisation plots |
| `python sweep_c_threshold.py --c <ckpt>` | Sweep `is_obj` threshold for C-head on Qatar double-damage |

### Dataset registry

`lib/datasets.py` — `DatasetConfig` registry. To add a dataset: (1) write a `_loader_<name>` wrapper, (2) add a `DatasetConfig(...)` entry to `DATASETS`, (3) add one `elif` in `train.py`'s `_build_dl_kwargs` (~3 lines). Nothing else changes.

## Key library files

| File | Role |
|------|------|
| `lib/midn.py` | `Midn` (v1 head), `MidnC` (C head), `MidnDR` (DR head) |
| `lib/model.py` | `ModelConfig`, `ModelConfigC`, `ModelConfigDR`, `build_model()`, `build_criterion_c()`; `load_model_from_checkpoint`, `load_model_c_from_checkpoint`, `load_model_dr_from_checkpoint` |
| `lib/losses.py` | `SetCriterion` (DETR-style Hungarian matching loss for C) |
| `lib/metrics.py` | **Canonical metrics**: `evaluate_all_v1` (v1), `evaluate_all_c` (C), `evaluate_all_dr` (DR), `distributed_map_v1/c/dr`, `map_mse`, `top_k_recall`, `average_precision`, `severity_mae_at_detected`, `f1_from_counts` |
| `lib/training.py` | `do_training` (v1), `do_training_c` (C), `do_training_dr` (DR) |
| `lib/data_safetensors.py` | `get_combined_dataloaders(["single","double"], ...)` — safetensors loader; `load_real_test_tensors`, `DEFAULT_BENCHMARK`, `TWO_DAMAGE_BENCHMARK` — real .mat benchmark access |
| `lib/data_tower.py` | `get_tower_dataloaders(...)` — tower dataset; `TOWER_TIME_LEN=400`, `TOWER_N_SENSORS=6`, `TOWER_N_LOCATIONS=4` |
| `lib/data_qatar.py` | `get_qatar_dataloaders(...)`, `get_qatar_double_test_dataloader(...)` — Qatar SHM Benchmark; `QATAR_FS=1024`, `QATAR_N_SENSORS=30`, `QATAR_N_LOCATIONS=30`; `row_sensors(r)`, `col_sensors(c)`, `mask_sensors(x, indices)` for fault evaluation |
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

### Tower dataset (physical experiment, L=4)
- Root: `data/tower/`
- 34 recordings, 9 damage states (healthy + DS1–DS8), excitation: EQ / WN / sine
- Input: `X (N, 1, 400, 6)` — 2s windows, 6 accelerometers, normalised by C1 RMS
- Labels: `Y (N, 4)` — `[B6, B1, connection, floor3]` severity ∈ {0, 0.5, 1.0}
- Labels normalised: `y_norm = Y*2−1 ∈ [−1, 1]` (undamaged=−1, max damaged=+1)
- Split: recording-level, stratified by damage category (~15% val, ~15% test)
- Augmentation: amplitude scaling ×U(0.8,1.2) + 5%-RMS Gaussian noise (train only)
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
- Augmentation: amplitude scaling + Gaussian noise + random channel masking + structured row/col masking + synthetic double-damage mixing (see `--p-mix`)
- Use `--dataset qatar` in `train.py` / `train_all.py`; `--root` overrides default path
- `--window-size` and `--overlap` flags on `train.py` to vary parametrically

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
| `--p-mix <float>` | `0.0` | Qatar only: fraction of training windows replaced by synthetic K=2 samples |
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

## Multi-damage training modes (Qatar)

Three modes for incorporating double-damage data, controlled by flags on `train.py` / `train_all.py` / `train_loo.py`:

### Synthetic mixing only (`--p-mix`)
Each training window has `p_mix` probability of replacement by a synthetic K=2 sample:
1. Partner drawn from K=1 windows of a *different* recording (K=2 windows excluded from pool to prevent K=3).
2. `x_mix = (x_i + x_j) / √2` — energy-preserving sum.
3. `y_mix = max(y_i, y_j)` — label union.
Result: C-head double-damage F1 improves from 0.317 → 0.508 (200 ep, p_mix=0.5).

### LOO with real double-damage (`--held-out-double <0-4>`)
Hold out one double-damage recording for test, add the other 4 to training alongside `--p-mix`.
Use `train_loo.py` to run all 5 folds automatically.
Result: mean F1=0.439 (range 0.055–0.740) — high variance due to spatial specificity.

### All-5 time split (`--split-double`)
First 50% of each of the 5 double-damage recordings → training. Second 50% → test.
Mutually exclusive with `--held-out-double`.
Result: F1=0.9996 — near-perfect, confirming spatial specificity as root cause (see Insight 18).

**Key finding**: C-head achieves near-perfect double-damage detection when it has seen the specific joint pair during training. Synthetic augmentation alone cannot substitute for real recordings of the target damage combinations.

## Checkpoints in states/

- `states/qatar/` — baseline Qatar models (v1, C, DR trained on single-damage only)
- `states/qatar-pmix/` — Qatar C model trained with synthetic double-damage mixing (p_mix=0.5, 200ep)
- `states/qatar-dd/` — Qatar C LOO models (4 real double-damage recordings in training, 1 held out per fold)
- `states/qatar-dd-split/` — Qatar C trained with first½ of all 5 double-damage recordings
- `states/7story/` — 7-story frame models
- `states/tower/` — tower models
