# SDINet-v2 — Claude Code Guide

## GPU environment notes

**TITAN V GPUs (device IDs 0, 1, 3, 4)** require `LD_PRELOAD=/tmp/nvml_stub.so` to avoid an NVML symbol error (`nvmlDeviceGetNvLinkRemoteDeviceType`) at backward pass. Always prefix commands with it when using these devices:

```
CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=/tmp/nvml_stub.so python train.py ...
```

## Architecture reference

See **`docs/MODEL_ARCHITECTURE.md`** for annotated diagrams of the v1, DR, and C heads — input/output shapes, sensor independence, importance weighting, and softmax clarification. (B shares DR's architecture with mean-pooling replacing learned attention.)

## Research framing

See **`docs/RESEARCH.md`** for the evolving research problem statement, central thesis, dataset roles, and open questions. Update it as the research discussion develops.

Domain-ML insights accumulated across the project are in **`docs/DOMAIN_INSIGHTS.md`** — append new ones as the research develops.

Deferred architectural work items for the iT+C proposal are in **`docs/future_plans.md`** (F1 sample-level gate — superseded by `no_obj_weight=0.5`, kept as fallback; F2 deep supervision; F3 R_bias freeze+drift logging; F4 anchor queries; F5 multi-scale conv tokenizer redesign preserving time sequence). Consult before proposing new architectural work.

## What this repo is

Structural Damage Identification (SHM) neural network. Sensor accelerometer data → damage location + severity. Four model heads:

- **v1** (`Midn`): single global damage scalar + softmax location (original Engineering Structures paper)
- **Approach C** (`MidnC`): DETR-style slot prediction with Hungarian matching
- **Approach DR** (`MidnDR`): per-location regression with MIL sensor attention
- **Baseline B** (`PlainDR`): plain per-location regression (mean-pool over sensors, no learned attention) — isolates MIL contribution

Current development focus: **fault-robust SDI on 7story-fault-k0 (single + double + undamaged on `unc=0` and `unc=1`)**, with the proposed configuration being iT+C+fh+sb with **pre-encoder fault head** (`--fault-head-location encoder`), **`no_obj_weight=0.5`** (up from DETR default 0.1), and **`--norm-method none`** (consistency across v1, B, and C on this protocol — `none` is within noise of `mean` for the iT backbone per Insight 30 revised). Canonical auto-label: `c-nn-it4-pfh-sb-nw5`. This config **ties v1 on K=1 F1** (0.962 vs 0.962), **wins K=2 F1** (0.883 grand, +0.24 over v1's 0.646), and **wins K=0 sample-FAR** (0.010 grand vs B's 0.167 vs v1's 0.507). The two tuning changes compose cleanly: nw=0.5 fixes the K=0 false-alarm problem by giving the ∅ class adequate gradient (Insight 32); pre-encoder fault head releases encoder capacity from gradient-path competition with damage supervision (Insight 33). Abandoned / deferred: structural affinity bias (`R_bias`) is retained from the legacy baseline but not the star; spatial self-attention (`SensorSpatialLayer`, Insight 21) remained off; multi-scale conv tokenizer failed to converge at 200 epochs under our training budget and is deferred (Insight 35, `docs/future_plans.md` F5). ASCE-hammer replication with the same config (`c-nn-it4-pfh-sb-nw5-s6`) confirms the architectural claims on a second dataset, though absolute F1 there sits in a 2-fold geometric symmetry ceiling (≈0.65 K=1) caused by same-wall-span equivalence in the 16-sensor ASCE layout.

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

2. **Global RMS normalization** — divide all sensors by the scalar RMS computed across all sensors and time steps jointly (`ref_channel=None`). Preserves cross-sensor amplitude ratios (the damage signature) while providing excitation-amplitude invariance. Global (all-sensor) reference chosen over single-sensor reference for fault robustness: a faulted reference sensor would corrupt all channels, whereas global RMS degrades gracefully (~1/S perturbation per faulted sensor). **CLI default is `none`** for both `train.py` and `evaluate_fault.py`. v1 and B match their original-paper preprocessing under the default (softmax and mean-pool heads absorb amplitude variation internally; removing the shared RMS denominator also eliminates a small fault-contamination path for B at extreme fault ratios). **C-variants (DenseNet backbone) must pass `--norm-method mean` explicitly** — the fault-aware head (`fh`+`sb`) exploits mean-RMS as an end-to-end fault-contrast cue and any alternative (none, final-LN) strictly regresses. For the iTransformer backbone (`--encoder-type itransformer`), norm choice is approximately neutral (Δ within ±0.005 at every fault cell) because the per-sensor Linear tokeniser preserves amplitude and learns the same fault contrast end-to-end. See Insight 30 for the backbone-dependent scope. The `median` option was removed from the codebase after the DenseNet norm-ablation showed it regressed at every cell.

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

### C-head checkpoint naming (head-to-toe)

Auto-label for any C-head (`--model c`) run without an explicit `--label`, walking the data path in order:

```
c-<norm>[-<tokenizer>]-<encoder><depth>[-<posemb>][-<faulthead>][-<bias>][-s<K>d<D>][-<readout>]
```

| stage | always shown | non-default markers |
|-------|---|---|
| `<norm>` | yes | `nm` (mean) / `nn` (none) |
| `<tokenizer>` | no — omit for linear | `msc` (multi-scale conv, Phase B1) |
| `<encoder><depth>` | yes | `it<D>` (iTransformer + depth) or `dn` (DenseNet) |
| `<posemb>` | no — omit for `learned` | `pen` (none), `per` (rope) |
| `<faulthead>` | no — omit if disabled | `fh` (decoder-located, legacy) / `pfh` (pre-encoder, Phase B2) |
| `<bias>` | no — omit if off | `sb` (structural affinity) |
| `s<K>` | no — omit if default (5) | e.g. `s3` |
| `d<D>` | no — omit if default (2) | e.g. `d4` (decoder depth, not to be confused with `it<D>` encoder depth) |
| `nw<X>` | no — omit if default (0.1) | e.g. `nw5` (no_obj_weight=0.5), `nw3` (=0.3). `X = int(value × 10)` |
| `<readout>` | no — omit if direct | `mil` |

Examples:
- `c-nm-it2-fh-sb` = iT+C d=2 baseline (mean norm, linear tokenizer, learned PE, decoder fault head, structural bias).
- `c-nm-it4-fh-sb` = iT+C d=4.
- `c-nn-it2-fh-sb` = iT+C d=2 with no-norm (no-norm ablation).
- `c-nm-it2-per-fh-sb` = iT+C d=2 with rope PE.
- `c-nm-msc-it4-pfh-sb` = **Phase B full**: multi-scale conv tokenizer + pre-encoder fault head at d=4.
- `c-nm-dn-fh-sb` = DenseNet C+fh+sb.

Existing checkpoints keep their legacy names (`c-fh-sb-it-d4`, `c-fh-sb`, etc.) — renaming would break references in saved eval CSVs. The new scheme applies only to runs going forward.

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
- `states/tmp/7story-fault/` — B, v1, C, C+fh, C+sb, C+fh+sb with fault aug (200 epochs, K=0-naive); eval: `saved_results/tmp/7story-fault/eval_fault_{b,v1,c,c-fh,c-sb,c-fh-sb}.{json,csv}`. C+fh+sb wins on K=2 across all fault levels; v1 still wins K=1 under faults. fh+sb shows synergistic interaction (Insight 28). Superseded by the K=0-aware `states/7story-fault-k0/` cohort for the current proposal; retained under `tmp/` for historical comparison.
- **Deleted checkpoint cohorts** (numerical evidence preserved in `saved_results/` CSVs/JSONs; all were dead-end configurations):
  - `7story-fault-ablate{50,200}` — ABC ablation (fault-gate, freeze-rbias+layer0, aux-loss); all three regressed vs base at both budgets. Insight 27.
  - `7story-fault-sweep50` — Round-1 10-variant 50-epoch HP sweep. Winner: n20 (no-obj-weight 0.2).
  - `7story-fault-sweep100` — Round-2 5-variant 100-epoch HP sweep + cross-ckpt comparison that motivated removing EMA (see `saved_results/tmp/7story-fault-sweep100/cross_checkpoint_comparison.md`). Winner: s3 (num_slots=3).
  - `7story-fault-n20-200ep`, `7story-fault-s3-200ep` — 200-ep confirmations of both sweep winners. Both null (s3 stronger: K1_nf52 −0.029, K2_nf52 −0.032). n=2 confirmation for Insight 29.
  - `7story-fault-norm-{median,none,median-mil}` — norm ablation on c-fh-sb (200 epochs each). Mean RMS strictly dominates median and no-norm at every cell; median+MIL collapses at high nf. Insight 30. Eval CSVs under `saved_results/tmp/7story-fault-norm-{median,none,median-mil}/`.
- `states/lumo/` — baseline LUMO models (v1, C, DR; 18 accel sensors; single-damage binary labels); eval: `saved_results/lumo/eval_20260410_021906.{json,csv}`
- `states/lumo-fault/` — B, v1 trained with fault aug (200 epochs, valid); C, C+fh, C+fh+sb **stale: need retraining with pos_weight removed + CosineAnnealingLR**; eval: `saved_results/lumo-fault/eval_fault_{b,v1,c,c-fh}.{json,csv}`
- `states/lumo-lumo-nobj05/` — exploratory: LUMO C-head with `--no-obj-weight 0.05`
- `states/lumo-lumo-slots2/` — exploratory: LUMO C-head with `--num-slots 2`
- `states/tower/` — tower models
- `states/asce-fault/` — impact-hammer B, v1, C, C+fh, C+fh+sb trained with fault aug (200 epochs each). **Geometry-capped**: ASCE's 4-sensor-per-floor mid-edge layout × symmetric-mass rigid-floor × diagonal-centre impulse produces a 4-fold identifiability equivalence on brace labels (8 4-groups across story × axis). All heads observe this ceiling — B/v1 K=1 F1 ≈ 0.35, recall = 1.0, precision ≈ 0.25, mean_k_pred ≈ 4. See Insight 31. Relative-comparison metric only; absolute F1 is not comparable to 7-story/Qatar/LUMO.
- `states/tmp/7story-fault-itransformer/` — C+fh+sb with iTransformer encoder replacing DenseNet (200 epochs fp32; fp16 diverges at ep 88). Matches DenseNet within 0.005 at every fault cell for nf ≤ 32; fault-detection F1 slightly higher. Eval: `saved_results/tmp/7story-fault-itransformer/eval_fault_c-fh-sb-it.{json,csv}`. Real-benchmark test is skipped (9-sensor .mat vs 65-sensor iT tokeniser — architectural mismatch). Superseded by the K=0-aware cohort; retained under `tmp/` for history.

## Experiment results (7-story frame)

Baseline retraining after preprocessing fix pending. Fault-robust results below are current.

### Fault-aware K=0-naive (HISTORICAL — `states/tmp/7story-fault/`, 200 epochs, `--p-hard 0.3 --p-soft 0.3 --p-struct-mask 0.3`)

Eval: `saved_results/tmp/7story-fault/eval_fault_{v1,b,c,c-fh,c-sb,c-fh-sb}.{json,csv}`. Fault-ratio sweep `{0.0, 0.2, 0.5, 0.8}` → nf={0, 13, 32, 52} on a 65-sensor layout, 3 repeats per condition, mean F1 across 7 fault types. Training used `--subsets single double` (no undamaged). Superseded by the K=0-aware cohort in `states/7story-fault-k0/`.

**Single-damage (K=1):**

| Model     | clean | nf=13 | nf=32 | nf=52 | fault_F1@32 |
|-----------|-------|-------|-------|-------|-------------|
| v1        | **0.974** | **0.972** | **0.966** | **0.941** | —       |
| B         | 0.923 | 0.920 | 0.879 | 0.463 | —       |
| C         | 0.927 | 0.922 | 0.911 | 0.849 | —       |
| C+fh      | 0.920 | 0.918 | 0.909 | 0.855 | 0.978   |
| C+sb      | 0.916 | 0.910 | 0.898 | 0.828 | —       |
| C+fh+sb   | 0.939 | 0.935 | 0.926 | 0.872 | 0.977   |

**Double-damage (K=2):**

| Model     | clean | nf=13 | nf=32 | nf=52 |
|-----------|-------|-------|-------|-------|
| v1        | 0.648 | 0.650 | 0.649 | 0.646 |
| B         | 0.846 | 0.849 | 0.834 | 0.531 |
| C         | 0.864 | 0.861 | 0.852 | 0.806 |
| C+fh      | 0.859 | 0.856 | 0.848 | 0.801 |
| C+sb      | 0.854 | 0.849 | 0.838 | 0.782 |
| C+fh+sb   | **0.873** | **0.869** | **0.862** | **0.817** |

Key findings:
- **v1 is the K=1 king under faults** (F1=0.941 at nf=52) — softmax-importance implicitly down-weights faulted sensors and there's no wrong-slot penalty at K=1. But v1 is architecturally capped at K=2 (single argmax → recall ≤ 0.5, F1 ≈ 0.65).
- **C+fh+sb is the K=2 king across all fault ratios** (F1=0.817 at nf=52 vs v1's 0.646). This is the headline result for the C architecture on 7-story-fault — it retains multi-damage capability and is fault-robust.
- **fh and sb are synergistic, not additive** (see Insight 28). Individually each is neutral-to-negative (fh alone +0.006, sb alone −0.021 at K=1 nf=52). Combined they gain +0.023 — ~3× the sum of individual effects, positive where neither is. Mechanism: fh encodes fault state in feature space; sb routes attention in logit space. Orthogonal injection spaces compose multiplicatively via soft cross-attention (≈ spatial_proximity × health). Putting both priors in the same space (as in the ABC ablation's fault-gate variant) breaks the synergy.
- **B collapses at nf=52** (F1=0.463) — mean-pool has no mechanism to down-weight faulted sensors implicitly.

### Normalization ablation on C+fh+sb (argmax decoder, 200 epochs each)

Compares global-mean RMS (baseline), per-window-median RMS, no normalization, and median+MIL. Mean F1 across 7 fault types by nf.

**K=1:**

| method      | clean | nf=13 | nf=32 | nf=52 |
|-------------|-------|-------|-------|-------|
| mean (base) | **0.939** | **0.935** | **0.926** | **0.872** |
| median      | 0.915 | 0.911 | 0.903 | 0.837 |
| no norm     | 0.910 | 0.907 | 0.895 | 0.826 |
| median+MIL  | 0.848 | 0.846 | 0.826 | 0.667 |

**K=2:**

| method      | clean | nf=13 | nf=32 | nf=52 |
|-------------|-------|-------|-------|-------|
| mean (base) | **0.873** | **0.869** | **0.862** | **0.817** |
| median      | 0.857 | 0.854 | 0.843 | 0.787 |
| no norm     | 0.854 | 0.850 | 0.839 | 0.783 |
| median+MIL  | 0.812 | 0.806 | 0.784 | 0.666 |

Mean RMS wins every cell. At K=1 nf=52, median loses to mean on all 7 fault types (largest gaps on bias-family: bias −0.049, gain_bias −0.043). Under fault-aware training the mean normalizer itself carries fault-contrast signal — a gain/bias-faulted sensor inflates the global RMS and suppresses clean channels, producing a pattern the fault head and slot cross-attention learn to exploit. Robust (median) and no-norm alternatives erase this cue. See Insight 30.

### Encoder ablation — iTransformer replacing DenseNet backbone on C+fh+sb (200 epochs, fp32)

`states/tmp/7story-fault-itransformer/c-fh-sb-it-<uuid>-best200.pt`. Eval: `saved_results/tmp/7story-fault-itransformer/eval_fault_c-fh-sb-it.{json,csv}`. Hungarian-matched DETR slot head + fault head + structural bias, identical to base C+fh+sb except the encoder is replaced by an iTransformer (per-sensor Linear(T, D) tokenisation + depth-2 Transformer self-attention across the sensor dimension).

| Subset | Model              | clean | nf=13 | nf=32 | nf=52 | top_k_recall (clean) | fault_detect @ nf=32 |
|--------|--------------------|-------|-------|-------|-------|---------------------|----------------------|
| K=1    | C+fh+sb (DenseNet) | 0.939 | 0.935 | 0.926 | **0.872** | 0.98+               | 0.977                |
| K=1    | C+fh+sb-iT         | **0.944** | **0.939** | **0.929** | 0.860 | **0.984**           | **0.984**            |
| K=2    | C+fh+sb (DenseNet) | 0.873 | 0.869 | 0.862 | **0.817** | 0.88+               | 0.977                |
| K=2    | C+fh+sb-iT         | **0.878** | **0.876** | **0.867** | 0.812 | 0.877               | **0.983**            |

iTransformer matches DenseNet within 0.005 at every cell for nf ≤ 32 (slightly wins clean + moderate-fault on both K=1 and K=2), and trails by ~0.005–0.012 at the extreme nf=52 tail. Fault-detection F1 is strictly higher (+0.006–0.007) — per-sensor tokenisation directly exposes the per-sensor health classifier to sensor-level features. iTransformer **required fp32 training** — its self-attention across 65 sensors saturates fp16 exp() past ep 88 under fault-aware training and diverges catastrophically (val_top_k_recall 0.910 → 0.031 in one epoch). `train.py` auto-gates `--encoder-type itransformer` to fp32; DenseNet C remains on fp16 (no softmax over 65 elements, so no overflow path). The post-training real-benchmark step is skipped for iT because the 9-sensor real `.mat` cannot be fed to a model whose tokeniser is fixed at n_sensors=65.

#### iTransformer sub-ablations (C+fh+sb-iT base, 200 epochs fp32 each)

Mean F1 across 7 fault types at each nf cell. Baseline (d=2) = learned absolute pos-emb + global-mean RMS norm.

| Variant                | K=1 clean | K=1 nf=13 | K=1 nf=32 | K=1 nf=52 | K=2 clean | K=2 nf=13 | K=2 nf=32 | K=2 nf=52 | fault_F1@32 |
|------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-------------|
| baseline (d=2)         | 0.944     | 0.939     | 0.929     | 0.860     | 0.878     | 0.876     | 0.867     | 0.812     | **0.984**   |
| **no-pe** (drop pos-emb) | **0.947** | **0.944** | **0.933** | **0.868** | 0.873     | 0.870     | 0.863     | 0.809     | 0.983       |
| **no-norm** (drop RMS) | 0.940     | 0.938     | 0.929     | 0.862     | 0.878     | 0.875     | 0.867     | 0.817     | 0.983       |
| **rope** (RoPE pos-emb) | **0.947** | **0.944** | 0.932     | 0.863     | 0.881     | 0.878     | 0.869     | 0.812     | 0.963       |
| **d=4 deeper** (4 layers) | 0.941  | 0.937     | 0.926     | 0.855     | **0.884** | **0.882** | **0.873** | **0.818** | **0.984**   |

Checkpoints: `states/tmp/7story-fault-it-{no-pe,no-norm,rope,d4}/`. Evals: `saved_results/tmp/7story-fault-it-{no-pe,no-norm,rope,d4}/eval_fault_*.{json,csv}`.

Findings:
- **All 4 variants within ±0.006 F1 of baseline at every cell** — the iT architecture is remarkably insensitive to PE choice, norm choice, and depth in the 2–4 range. Consistent with Insight 30 revised: amplitude-preserving tokenisers absorb the preprocessing choice, and sensor-dim attention over 65 tokens does not depend heavily on explicit position cues.
- **d=4 wins K=2 across the board** (+0.006 clean/nf=13, +0.006 nf=52). Deeper sensor-attention helps multi-damage slot routing specifically; K=1 cost is −0.003 to −0.005. Best K=2 variant overall.
- **no-pe wins K=1 marginally** (+0.003–0.008 across cells). Absolute learned position embeddings over an unordered 65-sensor set add small routing noise; removing them slightly tightens K=1 argmax without hurting K=2 much.
- **no-norm ≈ baseline** (within 0.005 every cell, K=2 nf=52 actually +0.005). Directly contradicts DenseNet norm-ablation expectation (Insight 30): iT's Linear(T, D) tokeniser preserves per-sensor amplitude and learns the fault-contrast end-to-end, making preprocessing choice redundant.
- **rope is neutral on F1 but costs fault_detect** (0.963 vs baseline 0.984). Relative-position bias applied to 65 physically-unordered sensor tokens doesn't match the 1D-sequence assumption RoPE was designed for; fault-head degrades the most because per-sensor health prediction loses the stable absolute-position signal.

### Phase B + K=0 training + no_obj_weight=0.5 — current proposal headline (`states/7story-fault-k0/`)

Adds three independent changes to the `C+fh+sb-iT d=4` baseline, evaluated on the **K=0-aware test set** (single + double + undamaged test splits; 30 held-out healthy samples from both `unc=0` and `unc=1`):

1. **K=0 data inclusion in training**: undamaged subsets from `unc=0` and `unc=1` added to the default `--subsets single double undamaged` flow.
2. **Phase B2**: fault head moved from MidnC (post-encoder) to iTransformerEncoder (pre-encoder, reading raw tokeniser output before cross-sensor self-attention). CLI: `--fault-head-location encoder`.
3. **`no_obj_weight=0.5`**: up from DETR default 0.1. CLI: `--no-obj-weight 0.5`.

Trained as: `train.py --model c --dataset 7story --epochs 200 --tag fault-k0 --p-hard 0.3 --p-soft 0.3 --p-struct-mask 0.3 --use-fault-head --use-structural-bias --encoder-type itransformer --encoder-num-layers 4 --fault-head-location encoder --no-obj-weight 0.5` → auto-label `c-nn-it4-pfh-sb-nw5` (norm=none is the CLI default). A `norm-method=mean` variant was trained first (`c-nm-it4-pfh-sb-nw5`) and matches within ±0.005 at every fault cell — Insight #30 revised. The `nn` variant is the canonical proposal because it keeps all three models (v1, B, C) on the same preprocessing without measurable cost.

Baselines v1 and B are also on `--norm-method none` (the default) so all three models in the 7story-fault-k0 comparison share identical preprocessing.

**K=0 healthy-sample FAR** (fraction of 30 held-out undamaged samples flagging ≥1 location; ↓ better):

| model | nf=0 | nf=13 | nf=32 | nf=52 | grand |
|---|---|---|---|---|---|
| v1-k0 | 0.500 | 0.513 | 0.500 | 0.508 | 0.507 |
| B-k0 | **0.000** | **0.000** | **0.000** | 0.524 | 0.167 |
| iT+C K=0-aware (legacy, nw=0.1) | 0.500 | 0.500 | 0.494 | 0.483 | 0.492 |
| B2 only (linear+pfh, nw=0.1, nm) | 0.500 | 0.394 | 0.379 | 0.410 | 0.399 |
| Proposal (nm, nw=0.5) | **0.000** | **0.000** | 0.006 | 0.062 | 0.022 |
| **Proposal (nn, nw=0.5, canonical)** | **0.000** | **0.000** | **0.000** | **0.030** | **0.010** |

**K=1 F1**:

| model | nf=0 | nf=13 | nf=32 | nf=52 | grand |
|---|---|---|---|---|---|
| v1-k0 | 0.973 | 0.972 | 0.968 | **0.944** | 0.962 |
| iT+C K=0-aware (legacy) | 0.920 | 0.915 | 0.901 | 0.830 | 0.884 |
| B2 only (nw=0.1, nm) | 0.953 | 0.950 | 0.942 | 0.865 | 0.921 |
| Proposal (nm, nw=0.5) | **0.983** | **0.982** | **0.979** | 0.938 | **0.967** |
| **Proposal (nn, nw=0.5, canonical)** | 0.979 | 0.977 | 0.974 | 0.933 | 0.962 |

**K=2 F1**:

| model | nf=0 | nf=13 | nf=32 | nf=52 | grand |
|---|---|---|---|---|---|
| v1-k0 | 0.647 | 0.648 | 0.649 | 0.643 | 0.646 |
| B-k0 | 0.844 | 0.845 | 0.823 | 0.512 | 0.732 |
| iT+C K=0-aware (legacy) | 0.858 | 0.855 | 0.844 | 0.786 | 0.829 |
| B2 only (nw=0.1, nm) | 0.890 | 0.887 | 0.878 | 0.823 | 0.864 |
| Proposal (nm, nw=0.5) | **0.906** | **0.905** | **0.899** | **0.852** | **0.886** |
| **Proposal (nn, nw=0.5, canonical)** | 0.902 | 0.900 | 0.895 | 0.847 | 0.883 |

Key findings:
- **Proposal ties v1 on K=1 (0.962 vs 0.962)** and **wins K=2 by +0.24 grand** (0.883 vs 0.646). The slot head no longer underperforms softmax on single-damage once the ∅ class is properly supervised — Insight #34.
- **K=0 FAR drops from 50% to ~1%** across fault levels on the canonical `nn` config. nw=0.5 provides the ∅-class gradient that K=0 inclusion alone (at nw=0.1) cannot deliver — see Insight 32.
- **Moving the fault head pre-encoder improves damaged-task F1** (+0.04 grand, B2 vs legacy) by ending gradient-path competition between damage and fault objectives on the shared encoder. Costs fault_detect F1 (0.92 → 0.77) — acceptable since damage detection is primary. See Insight 33.
- **norm=none vs norm=mean**: within ±0.005 on all 7story cells (Insight #30 revised). `nn` is the canonical proposal because v1/B run on `--norm-method none` (the CLI default) and using `mean` for C-only creates an inconsistency with no measurable benefit on the iT backbone.
- **msc (multi-scale conv) tokenizer deferred**: all msc variants at dec=2 failed to converge in 200 epochs; dec=4 partially recovered but remained below the linear-tokenizer baseline. Training-budget mismatch (Insight 35, `docs/future_plans.md` F5). Linear tokenizer retained as the proposal backbone.

Checkpoints in `states/7story-fault-k0/`:
- **Canonical proposal**: `c-nn-it4-pfh-sb-nw5-<uuid>-best200.pt`
- Alt proposal (`norm=mean`): `c-nm-it4-pfh-sb-nw5-<uuid>-best180.pt` (within-noise equivalent; kept as consistency evidence).
- Baselines in this folder: `v1-<uuid>-best190.pt`, `b-<uuid>-best170.pt`, `c-fh-sb-it-<uuid>-best120.pt` (legacy K=0-aware iT+C), plus 4 Phase B ablations (B1, B2, B full, B full+dec=4) for the 2×2 tokenizer × fault-head-location grid.
Evals: `saved_results/7story-fault-k0/eval_fault_*.{json,csv}`. The master `eval_fault_c-nn-it4-pfh-sb-nw5.{csv,json}` is the proposal reference.

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

## Experiment results (ASCE — impact hammer, fault-aware)

Geometry-capped absolute F1 (see Insight 31). All heads observe the same 4-fold identifiability ceiling from symmetric-mass × 4-mid-edge-sensors-per-floor × diagonal roof-centre impulse. Absolute F1 ≈ 0.40 at nf=0 is a dataset property, not a model property — use relative degradation as the metric. Fault-ratio sweep `{0.0, 0.2, 0.5, 0.8}` → nf={0,3,8,13} on 16-sensor layout, 3 repeats, mean F1 across 7 fault types (hard/gain/bias/gain_bias/noise/stuck/partial). `states/asce-fault/`. Eval: `saved_results/asce-fault/eval_fault_{b,v1,c,c-fh}.{json,csv}`.

| Model    | clean  | nf=3   | nf=8   | nf=13  | top_k_recall (clean) | fault_detect F1 @ nf=8 |
|----------|--------|--------|--------|--------|---------------------|------------------------|
| B        | 0.423  | 0.421  | 0.392  | 0.326  | 0.337               | —                      |
| v1       | 0.413  | 0.411  | 0.406  | 0.386  | 0.324               | —                      |
| C        | 0.381  | 0.380  | 0.378  | 0.362  | 0.312               | —                      |
| C+fh     | 0.381  | 0.381  | 0.379  | 0.367  | 0.309               | 0.990                  |
| C+fh+sb  | 0.378  | 0.376  | 0.376  | 0.366  | 0.307               | **0.992**              |

Key findings:
- **4-group ceiling confirmed across all five heads** — clean top_k_recall ≈ 0.31–0.34 (≈ 1/4 expected from geometry), F1 ≈ 0.38–0.42. Architecture-independent.
- **v1 is again K=1 king under faults**: degrades only 0.027 from clean to nf=13 vs B's 0.097. Consistent with 7-story / Qatar / LUMO pattern.
- **B collapses under faults** (Δ=−0.097 at nf=13), just like on 7-story — mean-pool lacks fault-rejection machinery.
- **Both C+fh and C+fh+sb achieve near-perfect fault detection on ASCE** (fault F1 = 0.990 and 0.992) — with only 16 sensors, per-sensor fault-presence classification is easy, beating even the 65-sensor 7-story case (~0.98).
- **C-variants plateau around 0.37–0.38 with smallest fault-induced drop** (Δ ≈ −0.015 to −0.019 across C/C+fh/C+fh+sb at nf=13). The slot/matching machinery trades absolute F1 for robustness; within the observability ceiling they are essentially indistinguishable (Δ between C-variants < 0.01).
- **Structural affinity bias (sb) does not help on ASCE** — the 4-corner ambiguity is about *which of 4 equally-coupled braces* is damaged, not *which of L locations is closest to a sensor*, so the affinity prior has no discriminative signal to inject. Consistent with Insight 28: sb requires a well-posed observability problem to add value.

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

## Excitation paradigm: impact testing vs shaker-based monitoring

The three benchmarks in the sim→lab→field trio (7-story, Qatar, LUMO) all use **deterministic or effectively repeated excitation** — forced shaker bursts, fixed-profile hammer impacts, or common ambient wind loading across scenarios. The time-domain response of a single window is itself a damage-discriminative signature: the impulse response rings out, modal content is in the envelope and decay, and a time-domain CNN can learn damage→response directly because the excitation pattern is shared across scenarios.

The original ASCE benchmark pipeline used **per-scenario independent white-noise forcing** (one unique seed per .mat in `cal_resp.m`, so every scenario had a statistically identical but realization-unique force). Under that protocol, a single time-domain window from a monitoring session contains no scenario-specific modal content — only a stationary random sample from a process whose second-order statistics encode the damage. Modal information lives in PSDs / FRFs / transmissibilities, not in the raw time series. Feeding per-scenario-unique time series to a CNN then creates a memorization channel (time-pattern fingerprint → scenario index → label) that evaporates at val/test time.

**Decision.** ASCE is being regenerated with impact-style forcing (single shared excitation profile across scenarios; damage state is the only inter-scenario variable) so it slots into the same time-domain CNN pipeline as 7-story. Results and checkpoints from the shaker-era run are discarded.
