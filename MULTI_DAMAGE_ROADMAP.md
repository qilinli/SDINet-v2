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


## Proposed Model Change

Keep:

- backbone (`SDIDenseNet`)
- neck (`Flatten + Conv1d + ReLU + Conv1d + ReLU`)

Replace the head with a **multi-damage map head**.

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


## Minimal Migration Plan

1. Add a new head class in `lib/midn.py` (for example, `MultiDamageMidn`) that returns `(B, 70)` severity maps.
2. Add head-mode config in `lib/model.py` (for example, `"single"` vs `"multi"`), keeping `"single"` for backward compatibility.
3. Update `lib/training.py`:
   - for multi mode, train against full location target map directly (not `max` + class index only).
4. Update `lib/testing.py`:
   - for multi mode, evaluate predicted `severity_map` directly against location-wise target,
   - avoid converting from `scalar * softmax(location)` in multi mode.
5. Keep current validation for single mode; add multi-mode metrics:
   - map MSE/MAE,
   - top-k location hit rate,
   - precision/recall on active-damage locations.


## Backward Compatibility

- Keep current single-damage head path intact to avoid breaking existing checkpoints and scripts.
- Introduce multi-damage behavior behind a config switch.
