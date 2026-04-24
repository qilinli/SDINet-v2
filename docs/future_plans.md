# Future plans — iT+C architecture

Deferred architectural work for the iT+C proposal. Items are kept here (not in CLAUDE memory) so they survive across sessions and are visible in source control.

Each item has: what to build, why it matters, expected leverage, rough effort.

---

## F1 — Sample-level "any damage" gate — **SUPERSEDED by `no_obj_weight=0.5` (retain as fallback)**

**Original motivation** (as of 2026-04-22): K=0 healthy samples were triggering 50% sample-level FAR on the iT+C head because the DETR ∅ class was under-supervised. A sample-level binary gate was proposed as an architectural fix — a tiny global head that would suppress all slots when it decides "nothing is happening."

**What changed** (2026-04-24): raising `no_obj_weight` from 0.1 to 0.5 (a pure loss-hyperparameter change, no architecture work) reduced K=0 grand sample_far from 0.492 to 0.022 — solving the problem that F1 was meant to solve, without adding a new head. The winning config `c-nm-it4-pfh-sb-nw5` achieves near-zero K=0 FAR at clean / moderate fault and 0.062 even at 80% sensor faults. See Insight 32.

**Status.** Kept on this list as a fallback lever. If a future dataset or excitation regime resurfaces the K=0 FAR problem and `no_obj_weight` tuning alone doesn't close it, F1's sample-level gate remains the obvious next architectural fix. Low urgency at this point — the simplest fix worked.

```python
# Sketch if revived:
self.global_head = nn.Linear(embed_dim, 1)
global_p = sigmoid(self.global_head(sensor_features.mean(dim=1)))
# inference: if global_p < τ: return K=0 (suppress all slots)
```

**Effort if revived.** ~1 day — forward path change + BCE term + threshold calibration.

---

## F2 — Deep supervision (`aux_loss`) if decoder depth ≥ 3

**What.** Apply the SetCriterion loss at every intermediate decoder layer, not just the final one. Code path already exists (`aux_loss=True`); currently off because CLAUDE.md Insight 27 showed it regressed at 50 epochs with decoder d=2.

**Why it matters.**
- With the current default `num_decoder_layers=2`, only the final layer gets loss signal — intermediate layer update is purely through backprop. Two layers is shallow enough that this is fine.
- If we ever push the **decoder** to d ≥ 3 (e.g. to improve K=3+ slot specialization), deep supervision becomes important. DETR itself uses aux-loss specifically for making deep slot decoders trainable.

**Expected gain.** Only relevant *conditional on decoder deepening*. Enables stable training at decoder d ∈ {3, 4, 6}.

**Effort.** ~1 day — already implemented; just needs retesting under the right conditions.

**Trigger.** Revisit if / when decoder depth is increased. Not actionable at current `num_decoder_layers=2`.

---

## F3 — R_bias: freeze by default + drift logging

**What.**
- Change the default to `freeze_r_bias=True` so the physics-derived sensor-location adjacency stays at its init.
- Regardless of freeze/unfreeze, log `(R_bias - R_bias_init).norm() / R_bias_init.norm()` per epoch so drift is visible in training curves.

**Why it matters.**
- Currently `self.R_bias = nn.Parameter(structural_affinity.clone())` is learnable. Over 200 epochs it drifts from the physics prior; after training we no longer know what sensor-location relationship the model is using.
- The paper's "structural affinity bias" claim is weakened if the final R_bias is not actually structural-affinity-shaped any more.
- Frozen R_bias = the claim becomes *defensible* by construction: we injected the physics prior, didn't let gradients erode it, and the model still works.

**Expected gain.** Paper defensibility, not a numerical win. May cost 0.005–0.010 on grand F1 (forgoing task-specific R_bias optimization) — acceptable trade for a cleaner claim.

**Effort.** ~2 hours — flip the default, add a logging hook in the training loop, rerun one iT+C d=4 experiment to quantify the cost.

---

## F4 — Anchor queries (location-seeded slots)

**What.** Replace input-agnostic learnable queries (`nn.Parameter(K, D)`) with queries that embed an explicit structural location prior — in the spirit of DAB-DETR / Conditional-DETR but adapted to our discrete L=70 location set.

Two concrete variants to explore:

**Variant A (simple):** fix K anchor locations (e.g. `anchors = [10, 25, 35, 50, 60]` for K=5, roughly evenly spread over L=70). Each slot's query = learnable content vector + `loc_embedding(anchors[k])`. Anchor stays fixed; only content is refined.

**Variant B (DAB-style):** anchor is learnable per slot, expressed as a distribution over L locations. At each decoder layer, anchor probs are refined (slot "homes in" on its chosen location). Query at layer i = content_i + `loc_embedding(anchor_probs_i @ loc_matrix)`.

**Why it matters.**
- Vanilla DETR queries start from identical random vectors — slots must discover specialization during training, which is slow and fragile with few slots (K=5) over many locations (L=70).
- Anchor queries give each slot a "starting zip code." Convergence is faster; specialization is interpretable; slots are more likely to cover different spatial regions naturally.
- Potential novel contribution for the paper: "structural-location-anchored slot queries for SHM."

**Expected gain.** Faster convergence (fewer epochs to plateau); better K=2 slot specialization; possibly +0.01–0.02 on K=2 F1 and better slot-diversity stats.

**Effort.** ~2 days — new query construction, layer-wise anchor refinement for Variant B, careful ablation (Variant A vs B vs baseline).

**Risk.** Only justified by pilot — single-seed ablation on 7story-fault-k0 to decide whether to pursue. Novel-contribution potential is high if Variant B works, low if only Variant A.

---

---

## F5 — Multi-scale conv tokenizer redesign (preserving time sequence into iT)

**What.** Replace the `Linear(T, D)` per-sensor tokenizer with a multi-scale dilated-conv tokenizer that **preserves the time sequence** rather than pooling it to a scalar. Concretely, each sensor's T=500 series would produce a short sequence `(T', D)` that enters the iT encoder as (B, S × T', D) tokens — the encoder now attends across both sensors AND coarse time steps.

```python
class MultiScaleConvTokenizer(nn.Module):
    def __init__(self, time_len, embed_dim, dilations=(1,2,4,8), stride=2, seq_len=32, ...):
        # Parallel dilated convs with MODEST stride (2, not 4)
        # Output: (B*S, branch_dim, T') for each branch
        # Concat branches: (B*S, total_branch_dim, T')
        # Project + interpolate to seq_len: (B*S, embed_dim, seq_len)
        # Reshape: (B, S*seq_len, embed_dim) — encoder attends over both dims
```

**Why it matters.**
- The Phase-B msc variant (stride=4 + AdaptiveAvgPool1d(1)) compressed all 500 time samples into 4 scalars per sensor. This killed the time-resolution information the iT encoder could have attended over. Training failed at the shared 200-epoch budget (see Insight 35).
- The *intended* benefit of a multi-scale conv tokenizer is to give the encoder *time*-structured features (not just scalar-per-sensor) at multiple receptive-field scales. Implementing this properly requires changing what enters the encoder from `(B, S, D)` to `(B, S × T', D)` — a bigger architectural change than the Phase B drop-in tokenizer attempt.

**Expected gain (speculative).** If the redesign succeeds, iT encoder sees sub-window temporal structure (ringdown patterns, modal decay rates, partial-fault amplitude modulations) directly. Could close the remaining fault_detect gap (0.77 → ~0.92) for Phase B, since partial faults become visible as a distinct time-signature.

**Effort.** ~3–5 days:
- Tokenizer rewrite (preserve T' dim, choose stride + seq_len)
- iT encoder adjusted to handle S × T' tokens (sequence length grows from 65 to ~65 × 8 = 520)
- Training-budget tuning (may need longer schedule or warm-up)
- Ablation: compare vs linear tokenizer on the proposal benchmark

**Risk.** High — this is research, not tuning. The 65-token attention that works well at S=65 scales quadratically at S × T' = 520. Memory and training-time go up. May require attention patterns (windowed or hierarchical) rather than full attention.

**Trigger.** Revisit only if the linear-tokenizer proposal lands and reviewers ask for multi-scale temporal features explicitly, or if deployment data shows the linear tokenizer missing specific fault patterns that a time-aware tokenizer could catch.

---

## Sequencing note

F3 is low-effort and does not require retraining anything critical — can be done opportunistically.

F4 is the most paper-valuable single item (novel contribution candidate), but higher effort and higher risk.

F5 is speculative and big-effort; defer unless the proposal arm stalls.

F2 is conditional on an independent decision (decoder depth) and doesn't unlock on its own.

F1 is now a fallback (superseded by `no_obj_weight=0.5`); revive only if a future setting reopens the K=0 FAR problem.

Suggested order after the proposal arm: **F3 → F4 → F2 → F5 → (F1 if needed)**.
