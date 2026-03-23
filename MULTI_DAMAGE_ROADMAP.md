# SDINet-v2 Multi-Damage Roadmap

## Background

v1 (baseline, Engineering Structures publication) predicts one global damage scalar `(B, 1)` + one location vector `(B, 70)`. All locations share the same magnitude — this is the single-damage bottleneck.

v2 objectives:
1. Handle single and multiple simultaneous damages in one unified model
2. Improve backbone/preprocessing with SOTA components (transformer, time-series encoders, new losses)
3. Evaluate on new datasets

---

## Multi-Damage Head: Design Options

Brainstormed approaches (ranked by publication relevance for Engineering Structures):

### A — Dense Severity Map *(Minimal change)*
Output `(B, 70)` severity per location. Train with regression + L1 sparsity penalty.
- **Pro:** Minimal change, naturally handles any K.
- **Con:** Sparsity is soft (λ is a tunable nuisance); nothing prevents diffuse activations.

### B — Presence + Severity Decomposition ✅ *Implemented*
Two heads sharing one importance: binary presence `(B, 70)` via weighted BCE, and severity regression `(B, 70)` via sigmoid. Final output = `sigmoid(presence) × severity`.
- **Pro:** Structurally enforces sparsity without a penalty parameter. Clean decomposition ("is there damage?" vs. "how much?") — highly interpretable for engineers.
- **Con:** BCE class imbalance (most locations are 0); handled via `pos_weight` (default 34–69).

### C — DETR-Style Set Prediction *(High novelty, high complexity)*
Predict K_max damage instances via transformer decoder + Hungarian matching loss.
- **Pro:** Elegant variable-K handling, no threshold or sparsity penalty needed. Novel in SHM.
- **Con:** Much more complex to train; overkill if K ≤ 5. Requires significantly more data.

### D — Graph-Informed Spatial Refinement *(Strong engineering angle)*
After MIL sensor aggregation, apply a GNN over a structural topology graph (70 location nodes) to refine the severity map.
- **Pro:** Physically motivated — structurally connected locations have correlated damage. Resonates well with structural engineering reviewers.
- **Con:** Requires structural graph definition (adjacency of the 70 locations). Structural topology confirmed available.

---

## Implementation Status

### Completed

| Component | File(s) | Notes |
|-----------|---------|-------|
| `MidnB` head | `lib/midn.py` | Shared importance, 3 Conv1d branches |
| `ModelConfigB` | `lib/model.py` | Plug-and-play with `build_model()` |
| `PresenceSeverityLoss` | `lib/losses.py` | Weighted BCE + masked severity MSE |
| B training loop | `lib/training.py` | `train/val_one_epoch_b`, `do_training_b` |
| Evaluation suite | `lib/metrics.py` | See below |
| Entry point | `main_b.py` | Parallel to `main.py` |

### Evaluation suite (`lib/metrics.py`)

| Metric | Function | Role |
|--------|----------|------|
| Distributed map MSE | `map_mse` | **Primary** — comparable across v1 and B |
| Top-K recall | `top_k_recall` | Localisation quality, threshold-free |
| Severity MAE at hits | `severity_mae_at_detected` | Severity accuracy, decoupled from localisation |
| Average Precision | `average_precision` | Detection quality, PR-curve area, threshold-free |
| F1 / P / R | `presence_f1_stats` | Reference at threshold=0.5 |
| Full suite | `evaluate_all` | Returns dict of all above; stratify by K |

---

## Datasets

| Dataset | Status | K damages | Notes |
|---------|--------|-----------|-------|
| 70-location frame (simulated) | Available | 1–2 | Current benchmark; unc=0 and unc=1 splits |
| Additional real datasets | Planned | 1–5 | Simulate multi-damage; topology available |

---

## Next Steps

### Step 1 — Train and compare B vs v1 baseline  *(immediate)*
- Run `python main.py` (v1) and `python main_b.py` (B) on the same split
- Use `evaluate_all` on the test set, stratified by K:
  ```python
  results_k1 = evaluate_all(pres[k1], sev[k1], y[k1])
  results_k2 = evaluate_all(pres[k2], sev[k2], y[k2])
  ```
- Expected story: B ≈ v1 for K=1, B >> v1 for K=2 on `map_mse` and `top_k_recall`
- Tune `presence_pos_weight` (34 for double, 69 for single)

### Step 2 — Ablation: pos_weight and severity_weight  *(after Step 1)*
- Grid over `pos_weight ∈ {20, 34, 50, 69}` on the double-damage split
- Grid over `severity_weight ∈ {0.5, 1.0, 2.0}`
- Report `map_mse` and `ap` per setting

### Step 3 — Add Approach D (GNN spatial refinement)  *(after Step 2)*
Now that topology is confirmed available:
- Define adjacency matrix for the 70-location frame (beam/column connectivity)
- Add `GNNRefinement` module: GCN or GraphSAGE over location nodes
- Insert after the B-head: `refined_map = gnn(distributed_map_b, adjacency)`
- New config `ModelConfigBD` extending `ModelConfigB`
- Compare B vs B+D on the same test set — D should improve localisation when
  adjacent locations are co-damaged

### Step 4 — Sensor-failure robustness extension  *(parallel to Step 3)*
Extend beyond random dropping (already in v1):
- Whole-sensor zero masking (dead sensors)
- Contiguous block dropout (regional hardware failure)
- Sensor corruption (noise spikes, drift, gain bias)
- Optional: pass binary sensor-availability mask to the head

### Step 5 — Additional real datasets  *(after Steps 1–3)*
- Simulate K=1–5 damages for new structural datasets
- Apply `evaluate_all` with K-stratified breakdown
- Report generalisation: does B trained on 70-location frame transfer?

### Step 6 — Backbone upgrade  *(optional, later)*
- Replace DenseNet with a time-series transformer (e.g. PatchTST, TimesNet)
- Keep MIL neck and B-head unchanged — isolates backbone contribution
- Compare on same test splits as Steps 1–3

---

## Sensor-Failure Robustness (preserve from v1)

Continue and extend failure simulation during training:
- Random sensor dropping (already in v1)
- Whole-sensor zero masking (dead sensors)
- Contiguous sensor block dropout (regional hardware failure)
- Sensor corruption (noise spikes, drift, gain bias)

Optional: pass a binary sensor-availability mask to the head so the model can distinguish missing sensors from true zero signal.

---

## Paper Narrative (Engineering Structures)

Three-level architecture story:
1. *Sensor level* → MIL handles sensor failure (from v1, preserved)
2. *Location level* → presence/severity heads replace single-damage bottleneck (B-head)
3. *Spatial level* → GNN refinement using structural topology (D-head, optional)

Key result to show: single model handles K=1 and K=2 without degradation, where v1 fails for K=2.
