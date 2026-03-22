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

### B — Presence + Severity Decomposition *(Recommended head)*
Two heads: binary presence `(B, 70)` via BCE, and severity regression `(B, 70)`. Final output = presence × severity.
- **Pro:** Structurally enforces sparsity without a penalty parameter. Clean decomposition ("is there damage?" vs. "how much?") — highly interpretable for engineers.
- **Con:** BCE class imbalance (most locations are 0); needs focal loss or positive weighting.

### C — DETR-Style Set Prediction *(High novelty, high complexity)*
Predict K_max damage instances via transformer decoder + Hungarian matching loss.
- **Pro:** Elegant variable-K handling, no threshold or sparsity penalty needed. Novel in SHM.
- **Con:** Much more complex to train; overkill if K is small (1–2 damages).

### D — Graph-Informed Spatial Refinement *(Strong engineering angle)*
After MIL sensor aggregation, apply a GNN over a structural topology graph (70 location nodes) to refine the severity map.
- **Pro:** Physically motivated — structurally connected locations have correlated damage. Resonates well with structural engineering reviewers.
- **Con:** Requires structural graph definition (adjacency of the 70 locations).

---

## Current Recommendation

**B + D combination:**
- Use **Approach B** as the prediction head (presence + severity)
- Add **Approach D** as an optional spatial refinement module if structural topology is available
- Keep **MIL sensor aggregation** unchanged (core v1 contribution)

This gives a clean three-level narrative:
1. *Sensor level* → MIL handles sensor failure (from v1)
2. *Feature level* → improved backbone (DenseNet → transformer/SOTA)
3. *Location level* → presence/severity heads + optional GNN refinement

**Open questions before deciding:**
- How many simultaneous damages does the dataset have? (If always 1–2, B alone is sufficient)
- Is structural topology (adjacency of the 70 locations) available? (Determines if D is viable)
- Is the multi-damage dataset already available or needs to be generated?

---

## Sensor-Failure Robustness (preserve from v1)

Continue and extend failure simulation during training:
- Random sensor dropping (already in v1)
- Whole-sensor zero masking (dead sensors)
- Contiguous sensor block dropout (regional hardware failure)
- Sensor corruption (noise spikes, drift, gain bias)

Optional: pass a binary sensor-availability mask to the head so the model can distinguish missing sensors from true zero signal.
