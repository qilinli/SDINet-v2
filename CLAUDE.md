# SDINet-v2 — Claude Code Guide

## What this repo is

Structural Damage Identification (SHM) neural network. Sensor accelerometer data → damage location + severity across 70 structural locations. Two model heads implemented:

- **v1** (`Midn`): single global damage scalar + softmax location (original Engineering Structures paper)
- **Approach B** (`MidnB`): per-location presence + severity decomposition (current development focus)

## Entry points

| Script | Purpose |
|--------|---------|
| `python main.py [--epochs N]` | Train v1 on single+double combined data |
| `python main_b.py [--epochs N]` | Train B on single+double combined data |
| `python train_all.py [--models v1 b] [--epochs N]` | Run both sequentially |
| `python evaluate.py --v1 <ckpt> --b <ckpt>` | Side-by-side metric table for both models |
| `python -m lib.testing --v1 <ckpt>` | Quick real-benchmark eval for v1 |
| `python -m lib.testing --b <ckpt>` | Quick real-benchmark eval for B |

## Key library files

| File | Role |
|------|------|
| `lib/midn.py` | `Midn` (v1 head) and `MidnB` (B head) |
| `lib/model.py` | `ModelConfig`, `ModelConfigB`, `build_model()`, `build_criterion_b()` |
| `lib/losses.py` | `PresenceSeverityLoss` (BCE presence + masked severity MSE) |
| `lib/metrics.py` | **Canonical metrics**: `evaluate_all` (B), `evaluate_all_v1` (v1), `distributed_map_v1/b`, `map_mse`, `top_k_recall`, `average_precision`, `severity_mae_at_detected`, `f1_from_counts` |
| `lib/training.py` | `do_training` (v1), `do_training_b` (B), `val_one_epoch`, `val_one_epoch_b` |
| `lib/data_safetensors.py` | `get_combined_dataloaders(["single","double"], ...)` — main loader; `get_dataloaders(subset, ...)` for individual subsets |
| `lib/testing.py` | `do_real_test` / `do_real_test_b` (real .mat benchmark), `eval_on_loader_v1` / `eval_on_loader_b` (dataset eval), checkpoint loaders |
| `lib/visualization.py` | `plot_training_results(...)` — saves loss/mse/metric curves |

## Metrics convention

**All metrics are defined in `lib/metrics.py` and must be used consistently everywhere.**

- `map_mse` — distributed damage map MSE in **[0, 1]** space (primary comparison metric, valid for both heads)
- `top_k_recall` — fraction of true damaged locations in top-K predictions (K = true # damages per sample)
- `ap` — average precision (area under PR curve), threshold-free
- `f1` / `precision` / `recall` — presence detection at sigmoid > 0.5
- `severity_mae` — MAE at correctly detected locations only
- `mean_k_pred` / `mean_k_true` — predicted vs true number of damaged locations

For v1: `evaluate_all_v1(dmg_pred, loc_pred, y_norm)` — uses `distributed_map_v1` + softmax ranking
For B:  `evaluate_all(presence_logits, severity, y_norm)` — uses `distributed_map_b` + sigmoid ranking

**Labels in normalized space**: `y_norm = raw_damage / 0.15 - 1 ∈ [-1, 1]`. Undamaged = -1, max damaged = +1.
**Severity target** (B head): `(y_norm + 1) / 2 ∈ [0, 1]`

## Data

- Root: `data/safetensors/unc=0/{single,double}/`
- Single-damage: K=1 per sample; Double-damage: K=2 per sample
- Both models now trained on combined single+double (28k train / 6k val / 6k test)
- Real benchmark: `data/Testing_SingleEAcc9Sensor0.5sec.mat` (physical units, 1 sample)

## Training output

- Checkpoints: `states/v1-combined-<uuid>.pt`, `states/b-combined-<uuid>.pt`
- Plots: `saved_results/` (v1), `saved_results_b/` (B) — loss curves, val_map_mse, val_top_k_recall / val_F1
- Per-epoch progress bar: `train_loss | val_loss | val_map_mse | val_top_k_recall` (v1) or `val_f1` (B)
- Final eval after training: separate single/double test splits + real benchmark

## Checkpoints in states/

- `states/single-damage-sparse-f8a04590-....pt` — v1 trained on single only (old)
- `states/multi-damage-B-6cb80aae-....pt` — B trained on double only (old)

## `pos_weight` for B head

Rule of thumb: `(L - K_mean) / K_mean` where L=70.
- Single only: 69.0
- Double only: 34.0
- Combined (K_mean=1.5): **45.7** (current default in `RunConfigB`)
