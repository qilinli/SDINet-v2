# Model Architecture Reference — SDINet-v2

---

## v1 Head (Midn)

### Overview

```
Input: (B, 1, T=500, S)
  │   B = batch, 1 = channel, T = time steps, S = sensors
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  DenseNet Backbone                                      │
│  Conv2d kernels: (20,1) stem, (5,1) dense, (2,1) pool  │
│  ──────────────────────────────────────────────────     │
│  Each kernel has width=1  →  sensors NEVER interact     │
│  Operates along time axis only                          │
└─────────────────────────────────────────────────────────┘
  │
  │  (B, C, T', S)   C channels, T' = reduced time
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  Neck                                                   │
│  Flatten(dim 1&2): (B, C·T', S)                        │
│  Conv1d(C·T' → 768, kernel=1) + ReLU                   │
│  Conv1d(768  → 768, kernel=1) + ReLU                   │
│  ──────────────────────────────────────────────────     │
│  kernel=1  →  sensors STILL never interact              │
└─────────────────────────────────────────────────────────┘
  │
  │  (B, 768, S)   one 768-dim feature vector per sensor
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  Midn Head                                              │
│  Conv1d(768 → 2·(L+1), kernel=1)                       │
└──────────────────────────┬──────────────────────────────┘
                           │
            (B, 2·(L+1), S)  split in half along channel
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
   prediction branch                 importance branch
   (B, L+1, S)                       (B, L+1, S)
           │                               │
           │                               │  × temperature
           │                               │  dropout (train)
           │                               │  softmax over S ←── NOTE: over sensors, not locations
           │                               │
           │                        (B, L+1, S)  each row sums to 1 over S
           │                               │
    split channel 0 vs 1:L          split channel 0 vs 1:L
           │                               │
    ┌──────┴──────┐               ┌────────┴────────┐
    ▼             ▼               ▼                 ▼
  tanh          (none)         dmg_imp           loc_imp
(B, 1, S)    (B, L, S)        (B, 1, S)         (B, L, S)
dmg_pred     loc_pred     sensor weights     sensor weights
                          for damage         for each location
```

### Aggregation over sensors

```
dmg_pred  (B, 1, S)          loc_pred  (B, L, S)
     ×                              ×
dmg_imp   (B, 1, S)          loc_imp   (B, L, S)
     │                              │
     └──── .sum(dim=S) ─────────────┘
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
  dmg_score (B, 1)      loc_score (B, L)
  ∈ (-1, +1)            raw logits, unbounded

  "is there damage?"    "which location?"
  tanh-weighted avg     importance-weighted avg
  over sensors          over sensors, per location
```

### Post-processing for evaluation

```
dmg_score (B, 1)            loc_score (B, L)
     │                            │
     │  (x+1)/2                   │  softmax over L   ← post-hoc, NOT in model
     ▼                            ▼
dmg_scale (B, 1) ∈ (0,1)    loc_probs (B, L) ∈ (0,1), sums to 1
     │                            │
     └──────── × ─────────────────┘
                    │
                    ▼
          distributed_map (B, L) ∈ (0,1)
          = dmg_scale × softmax(loc_score)
          used for: map_mse, bar chart visualization

For location detection:
  ├── top-k(loc_score)          top-K locations by raw score (evaluation)
  └── ratio threshold           loc_probs[l] > α × max AND dmg_scale > β  (calibration)
```

### Key design insight: sensor independence

```
Sensor 0  ──[DenseNet]──[Neck]──┐
Sensor 1  ──[DenseNet]──[Neck]──┤
Sensor 2  ──[DenseNet]──[Neck]──┤──▶ (B, 768, S)
   ...                          │
Sensor S  ──[DenseNet]──[Neck]──┘
                                         each sensor processed in isolation
                                         ────────────────────────────────
                                         sensors only interact in the Midn
                                         head via the importance-weighted sum
```

The importance weights (softmax over S) determine which sensor contributes most to:
- the global damage score (one weight per sensor: `dmg_imp`)
- each location's score (one weight per sensor per location: `loc_imp[l]`)

---

## DR Head (MidnDR)

Same backbone and neck as v1. Different head:

```
(B, 768, S)
     │
     ▼
┌──────────────────────────────────────┐
│  Two parallel Conv1d(768→L, kernel=1) │
└──────────────┬───────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
importance branch      prediction branch
(B, L, S)              (B, L, S)
× temperature           sigmoid
dropout (train)
softmax over S ──▶ (B, L, S) weights
    │                     │
    └─── × ──── .sum(S) ──┘
                 │
                 ▼
           pred (B, L) ∈ (0,1)
           direct severity per location
           no separate dmg scalar
```

No post-hoc softmax over locations needed — pred is sigmoid-activated and serves directly
as both the damage map and the presence ranking score.

### DR evaluation and calibration

```
model output: pred (B, L) ∈ (0,1)   ← same tensor used everywhere, no transformation

Training loss:
  MSE mode:  F.mse_loss(pred, (y_norm+1)/2)          both in [0,1]
  BCE mode:  F.binary_cross_entropy(pred, target)     both in [0,1]

Calibration:  ratio threshold on raw pred values
  pred_pres = (pred > α × max(pred)) AND (max(pred) > β)
  α, β swept on validation set to maximise F1

Evaluation metrics (evaluate_all_dr):
  distributed_map = pred                              identity — pred IS the map
  top-k recall    = top-k(pred[b])                   rank by pred value directly
  AP              = average_precision(pred, y_pres)  pred as probability scores
  severity_mae    = |pred[tp_locs] − y_sev[tp_locs]| both in [0,1]
  F1              = ratio threshold (same rule as calibration)

Inspection (run_dr / _run_real_inference):
  pred = model(x)    ← returned as-is, no further transformation
```

---

## C Head (MidnC)

Same backbone and neck. Head is DETR-style:

```
(B, 768, S)
     │
     permute
     ▼
memory (B, S, 768)   ← S sensor feature vectors as transformer memory
     │
     │    learnable slot queries (K_max, 768) → expand → (B, K_max, 768)
     │                                                          │
     └──────────── TransformerDecoder (cross-attention) ────────┘
                          │
                    slots (B, K_max, 768)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    Linear(768, L+1)            Linear(768,1) + Sigmoid
    loc_logits (B, K, L+1)      severity (B, K) ∈ (0,1)
    RAW logits                  sigmoid-activated
    last class = ∅ (no-object)
```

### C training loss (SetCriterion, Hungarian matching)

```
For each sample b:
  1. Hungarian matching (on detached outputs):
       cost[k,g] = −log P(slot k → gt_loc g) + sev_weight × |sev[k] − gt_sev[g]|
       where P comes from softmax(loc_logits[k])   ← softmax applied here for cost
       → assigns each GT damage to one slot; unmatched slots → ∅ class

  2. Location CE loss  (all K_max slots):
       F.cross_entropy(loc_logits, loc_targets, weight)
       ← F.cross_entropy applies softmax internally; loc_logits stay raw

  3. Severity MSE loss (matched slots only):
       F.mse_loss(severity[matched], gt_sev[matched])
       ← both in [0,1]; severity is sigmoid-activated, gt_sev = (y_norm+1)/2
```

### C slot decoding (same function used in calibration AND evaluation)

```
_c_slot_decode(loc_logits):                      ← single function, used everywhere

  argmax    = loc_logits.argmax(dim=-1)           (B, K)   over L+1 classes
  active[k] = argmax[k] ≠ L                      hard binary: no threshold needed
  pred_loc  = loc_logits[..., :L].argmax(dim=-1)  (B, K)   best of L real locations
  is_obj    = 1 − softmax(loc_logits)[..., L]     (B, K)   P(not ∅), used for ranking
```

### C evaluation and calibration

```
Calibration (calibrate_c_obj_threshold):
  uses _c_slot_decode → no free threshold to tune
  reports mean_k_pred vs mean_k_true for diagnostics only
  returns {} empty dict (pure argmax, nothing to calibrate)

Evaluation metrics (evaluate_all_c):
  uses _c_slot_decode
  F1 / precision / recall  from set comparison:  pred_set = {pred_loc[k] : active[k]}
  top-k recall             rank slots by is_obj, take top-k, check locations
  severity_mae             at TP locations only, highest-confidence active slot
  map_mse = NaN            discrete set → no soft map → not computable
  AP      = NaN            no per-location probability score → not computable

Inspection (run_c / _run_real_inference):
  soft distributed map for VISUALIZATION only (not used in any metric):
    is_obj = 1 − softmax(loc_logits)[..., ∅]
    scale  = is_obj × severity
    map[l] = Σ_k  scale[k] × softmax(loc_logits[k])[:L][l]    → (B, L)
```

### Why map_mse / AP are NaN for C

The C-head makes a **discrete set prediction** — each slot either fires (active) or doesn't.
There is no continuous per-location score to compute AP over, and no soft map to compute
MSE against. The soft map above is only produced in `run_c` for the bar chart; it is not
fed into any evaluation metric. Use F1 and top-k recall to evaluate C-head performance.

Unlike v1/DR, the C-head is the first point where sensors interact with each other
(via cross-attention in the TransformerDecoder). v1 and DR sensors are independent
all the way through; C-head slots attend over all S sensor memory vectors jointly.

---

## Shared softmax clarification

| Where | Softmax over | Purpose |
|-------|-------------|---------|
| Midn / MidnDR importance branch | **Sensors (S)** | Attention weights: which sensor to trust for each output |
| MidnC slot decoder | **Sensors (S)** | Cross-attention over sensor memory (inside TransformerDecoder) |
| `evaluate_all_v1` / `distributed_map_v1` | **Locations (L)** | Post-hoc: convert raw loc_score to a probability distribution for metrics |
| MidnC loc_logits decode | **Locations (L+1)** | Slot classification: which location does this slot predict |

The location-level softmax for v1 is **not part of the model** — it is applied in the
evaluation and visualization layer only.
