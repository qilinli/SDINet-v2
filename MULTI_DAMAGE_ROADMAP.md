# Multi-Damage + Sensor-Failure-Robust SDINet Roadmap

## Goal

Upgrade the current single-damage pipeline to:

1. predict multiple simultaneous damages naturally, and
2. remain robust when sensors fail or degrade.


## Current Limitation (Why Change)

Current `Midn` predicts:

- one global damage scalar `(B, 1)`, and
- one location score vector `(B, 70)`.

This is strong for single-damage scenarios, but multi-damage cases are forced into a compromise because all locations share one global magnitude.


## Proposed Model Change (Unified v2)

Keep:

- backbone (`SDIDenseNet`)
- neck (`Flatten + Conv1d + ReLU + Conv1d + ReLU`)

Replace the head with a single **multi-damage map head** for all training and inference.

### Multi-Damage Head (sensor-attention style)

Input from neck: `z` with shape `(B, E, S)` where:

- `B`: batch
- `E`: embedding channels
- `S`: sensors

Head outputs:

- `pred = Conv1d(E, 70, 1)` -> `(B, 70, S)` (per-location, per-sensor severity evidence)
- `imp  = Conv1d(E, 70, 1)` -> `(B, 70, S)` (per-location sensor importance logits)
- `imp  = softmax(imp, dim=-1)` (normalize over sensors only)
- `severity_map = (pred * imp).sum(-1)` -> `(B, 70)` (final multi-location severity prediction)

Optional auxiliary branch:

- `presence_logit = Conv1d(E, 70, 1)` + reduction -> `(B, 70)` for damaged/not-damaged classification per location.

This preserves sensor-attention robustness while removing the single-global-scalar bottleneck.
There is no single/double model switch in v2; one architecture is used for any number of damages `k`.


## Training Objective (for sparse multi-damage labels)

Assume target `y` has shape `(B, 70)` (normalized severity at each location).

Recommended loss:

- `L_reg`: `SmoothL1Loss(severity_map, y)` (or MSE)
- `L_sparse`: `mean(abs(severity_map))` to encourage sparse activations
- Optional `L_bce`: `BCEWithLogitsLoss(presence_logit, (y > threshold).float())`

Total:

- `L = L_reg + lambda1 * L_sparse + lambda2 * L_bce`


## Sensor-Failure Robustness Strategy

Continue and extend failure simulation during training:

- random sensor dropping (already aligned with existing strategy),
- whole-sensor zero masking (dead sensors),
- contiguous sensor block dropout (regional hardware failure),
- random sensor corruption (noise spikes, drift, gain bias).

Optional improvement:

- pass a binary sensor-availability mask to the head so the model can distinguish missing sensors from true zero signal.


## Multi-Phase Migration Plan (Unified Pipeline)

### Phase 1 - Architecture (minimal, clean)

1. Update `lib/midn.py` to output only location-wise severity map behavior:
   - `pred`, `imp`, and weighted sensor reduction to `(B, 70)`.
2. Update `lib/model.py` to use `out_channels=70` and remove single-damage assumptions in head configuration.
3. Keep backbone (`SDIDenseNet`) and neck unchanged.

### Phase 2 - Training Objective

1. Update `lib/training.py` to supervise the full target map directly:
   - remove `max` damage + class-index target decomposition.
2. Use map regression as the primary objective:
   - `L_reg = SmoothL1(severity_map, y)`.
3. Add sparsity control:
   - `L_sparse = mean(abs(severity_map))`,
   - `L = L_reg + lambda1 * L_sparse` (start with small `lambda1`).

### Phase 3 - Validation and Testing

1. Evaluate map predictions directly (no `scalar * softmax(location)` reconstruction).
2. Replace single-location metrics with multi-damage metrics:
   - map MSE/MAE,
   - top-k location hit rate,
   - precision/recall on active-damage locations.
3. Keep sensor-subset robustness validation, but compute errors on final map outputs.

### Phase 4 - Robustness Expansion

1. Keep random sensor dropping.
2. Add:
   - whole-sensor zero masking,
   - contiguous sensor block dropout,
   - sensor corruption (spikes, drift, gain bias).
3. Optional: pass sensor availability mask to distinguish missing sensors from true zero signal.


## Versioning Decision

- v2 is intentionally a unified multi-damage pipeline.
- Backward compatibility with v1 checkpoints/scripts is not a design requirement inside this branch.
- If legacy comparison is needed, keep v1 in a separate repository/branch and compare at experiment level.
