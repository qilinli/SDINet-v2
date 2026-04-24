# Research Framing — SDINet-v2

This document records the evolving conceptual framing of the research. It is a living document — update it as the ideas develop.

---

## Problem Setting

**Vibration-based structural damage identification from a single forced-vibration test.**

A structure is excited (white noise, impact, or sweep), accelerations are recorded at S sensor locations for a short window (0.5–2 s), and from that single snapshot the model identifies which structural components are damaged and how severely. This is a **snapshot identification problem**, not long-term monitoring or anomaly detection — there is no temporal drift signal and no baseline comparison across days or weeks.

### Why "forced-vibration testing," not "structural health monitoring"

Long-term SHM infers damage from slow changes in ambient vibration statistics over months. Forced-vibration testing is a discrete inspection event: a team deploys a sensor array, excites the structure, and needs a damage report from that single test. The timescales, signal types, and operational constraints are different. Our models take one measurement window and produce one damage map.

---

## The Sensor Fault Problem

In field and laboratory deployments, **some sensors may be unreliable during the test**:
- Cables disconnect → sensor reads near-zero
- Connectors corrode → intermittent signal
- Sensors fall off or are misplaced → constant offset or stuck value
- Data acquisition glitches → clipped or saturated channels

Current vibration-based identification methods either (a) assume all sensors are healthy, or (b) pre-process faults externally — detect faults first, mask them, then run identification. Both approaches require knowing which sensors failed before the model runs.

### The core ambiguity

From raw sensor signals alone, two physically different events produce similar observations:

| Event | Observation |
|-------|-------------|
| Sensor near a damaged joint | Anomalously **high** response relative to neighbours |
| Faulty sensor (disconnected/stuck) | Anomalously **low or zero** response relative to neighbours |

These require opposite responses from the identification model: the first is the most informative signal and should be attended to strongly; the second should be down-weighted or ignored. Disambiguating them requires understanding the spatial correlation structure of the sensor array under forced excitation.

---

## Central Research Gap

No existing method trains a single model end-to-end that simultaneously:
1. Identifies which structural components are damaged (and how severely)
2. Learns which sensors are trustworthy — **without any external fault flag**

The spatial layout of sensors over a structure creates **predictable response correlations** under forced excitation. A model that internalises these correlations can distinguish "this sensor reads unusually because it is near damage" from "this sensor reads unusually because it is faulty" — purely from the spatial pattern of the measurement, with no preprocessing step.

---

## Central Thesis

> The spatial layout of sensors over a structure creates predictable response correlations under forced excitation. A model that learns these correlations can internally distinguish structural response anomalies from sensor reliability anomalies — without any external fault flag. Robust damage identification then follows from attending appropriately to trustworthy sensors.

This is a stronger claim than "fault-tolerant identification": the model learns a notion of structural physics (spatially correlated forced response) and uses it as a continuous prior on sensor reliability, all from the training data.

---

## Datasets

Three datasets are used, representing the full simulation → laboratory → field progression:

| Dataset | Nature | Sensors (S) | Joints (L) | Damage | Role |
|---------|--------|-------------|------------|--------|------|
| 7-story frame | Simulation | 65 (full) / 9 (sparse) | 70 | Continuous severity, K=1 and K=2 | Design exploration, ablations |
| Qatar SHM Benchmark | Physical lab | 30 | 30 | Binary presence, K=1 (train) / K=2 (test) | Real-world validation |
| LUMO | Field (outdoor ambient) | 18 | 3 | Binary presence, K≤1 only | Field validation, ambient robustness |

The sim → lab → field progression is itself a contribution: design and ablate on clean simulation, validate transfer to physical lab data, then stress-test under real ambient field conditions with non-stationary excitation. LUMO (Leibniz Uni Hannover lattice tower, wind-excited, 18 accel sensors, 3 reversible damage positions) closes the loop. Key finding: the architecture ranking inverts going from lab to field — DR > C on LUMO (F1=0.972 vs 0.946), reversing the Qatar ranking, because the C-head's ∅-class boundary blurs under ambient distribution shift while DR's per-location thresholds remain robust (see Domain Insight 23).

The 7-story **sparse** config (S=9) is specifically calibrated to the 9 sensors used in the physical benchmark `.mat` file, bridging simulation and physical experiment within the same dataset.

---

## Architecture Analysis: Is the C-Head a Good Fit?

### What the C-head already does well

The C-head (DETR-style slot prediction, `MidnC`) is the primary architecture under investigation. Its slot mechanism is genuinely the right paradigm for the damage detection side: variable-cardinality set prediction — detecting K≥0 damaged joints from a single measurement — via learned slot queries and Hungarian-matching training. Unlike per-location regression (DR-head) or single-location models (v1), the C-head handles multi-damage detection without fixed output size or post-hoc thresholding.

**The C-head already provides implicit fault tolerance.** The cross-attention of slot queries over sensor memory naturally down-weights zeroed sensors — if sensor k is disconnected, its memory vector goes near zero and slots attend to it less. Training with masking augmentation reinforces this. For hard faults (disconnect, zero), the C-head is already reasonably robust without modification.

### The importance mechanism: a shared design philosophy

All three model heads (v1, DR, C) implement a form of **learned per-sensor weighting**:

- **v1** (`Midn`): a 1×1 Conv1d produces per-sensor importance logits, softmaxed over the sensor dimension. The damage prediction is a weighted average over sensors: `output = (prediction × importance).sum(S)`.
- **DR** (`MidnDR`): same structure but per-location — `(B, L, S)` importance weights, one distribution over sensors per output location.
- **C-head** (`MidnC`): the slot decoder cross-attention IS the importance mechanism, but expressed more richly — each of K_max slots learns its own attention distribution over all S sensors.

This is a coherent design philosophy: all heads learn *which sensors carry the most evidence* and aggregate accordingly.

### The critical limitation

The importance weights are trained to answer: *"which sensors carry the most evidence for predicting damage?"* This is **not the same** as *"which sensors are trustworthy?"* The two can diverge:

| Sensor state | Importance (learned) | Correct response |
|---|---|---|
| Near damage → high-magnitude response | **High** importance | Attend strongly — correct |
| Stuck at high value (soft fault) | **High** importance (features look like damage) | Down-weight — wrong |
| Zeroed (hard fault) | Low importance (zero features) | Ignore — accidentally correct |
| In undamaged region → near-zero reading | Low importance | Keep as "no damage here" evidence |

The mechanism works for hard faults by accident. For soft faults it can be actively misleading.

A second limitation: the importance weight for each sensor is computed **independently** — from that sensor's features alone, with no knowledge of what its physical neighbours are doing. The 1×1 Conv in v1, the Conv1d importance branch in DR, and the key/query projection in the C-head cross-attention all lack cross-sensor context. The model cannot reason "sensor k is anomalous *relative to its neighbours*" — which is the spatial consistency signal needed to distinguish fault from damage.

### Early vs. late space for fault detection

The question of *where* to introduce sensor-to-sensor reasoning: in raw signal space (before the backbone) or in learned feature space (after).

**Raw space** has interpretable fault signatures — zero variance, constant mean, anomalous coherence with neighbours are all directly readable from time series without learned transformation. Classical SHM methods operate here. However, in raw space damage also changes sensor correlations: a sensor near a damaged joint shows anomalous coherence relative to its neighbours, identical to a fault signature. The damage-fault ambiguity is irresolvable in raw space without already knowing the damage state — a circular dependency.

**Feature space** partially resolves this. After the backbone has compressed temporal signals into structural-response representations, a faulty sensor's feature vector is spatially inconsistent in terms of *structural response pattern*, not just raw magnitude. A sensor in an undamaged region that is also healthy produces a coherent feature; a stuck-high faulty sensor produces a feature inconsistent with what that structural location should read given the global response. Backbone features also make the computation tractable: self-attention over S=30 feature vectors of dimension 768 is cheap; operating on raw time series (S=30 × T=512) is not.

**Conclusion**: feature space is the right primary target. The backbone has already encoded structural physics into compact representations, giving a richer discrimination basis than raw statistics. The existing importance mechanisms confirm that late-space sensor weighting already works for fault tolerance in the easy (hard fault) case. The missing piece is cross-sensor context — a stage where sensors reason about each other's features before importance weights are finalised.

### Field-regime finding: C-head's ∅-class is a liability under ambient excitation

The LUMO field benchmark (ambient wind excitation, K≤1) reveals a regime where the C-head's slot machinery *hurts* rather than helps. The ranking inverts from lab conditions: DR F1=0.972 vs C F1=0.946 on LUMO, while on Qatar they were tied at ~0.99. Two reinforcing mechanisms:

1. **∅-class boundary blur.** The DETR ∅-class decision ("is this slot active or empty?") is learned from training data. Under stationary forced excitation, the healthy embedding distribution is compact and the ∅ boundary is sharp. Under ambient excitation, wind variability spreads the healthy distribution — some low-energy damaged windows fall on the "healthy" side, dropping C's recall to 0.929. DR has no such global gate — each location is thresholded independently, keeping recall at 0.972.

2. **Unnecessary set-prediction overhead.** LUMO has K≤1 everywhere — no multi-damage scenarios. The slot mechanism, Hungarian matching, and ∅-class all exist to solve variable-cardinality set prediction (K>1). When K≤1, DR's simpler "each location votes independently" is a strictly better-matched inductive bias.

**Implication for the research narrative**: the C-head's advantage is *conditional on the regime*. It dominates when K>1 and excitation is stationary (7-story double-damage: C F1=0.942 vs DR 0.849). It loses when K≤1 and excitation is non-stationary (LUMO: DR > C). A future unified architecture should either adapt its ∅ threshold to input distribution statistics or learn a regime-dependent gating strategy.

---

## Sensor-Fault-Aware SDI: Completed Investigation

### Approaches tried

Two approaches were investigated for injecting spatial sensor knowledge into the C-head:

1. **Sensor spatial reasoning layer** (`SensorSpatialLayer` / C+SF+FH) — all-to-all self-attention over sensor features before the slot decoder. **Failed**: contaminated clean sensor representations by distributing fault artifacts into neighbours' features (see "Why SensorSpatialLayer failed" below and Domain Insight 21).

2. **Structural affinity bias** (`R_bias` / C+FH+SB) — learnable (L+1)×S matrix injected as additive logit bias in slot cross-attention at decoder layers ≥1. **Succeeded**: guides attention routing without modifying sensor embeddings. Achieves best damage F1 across all fault types while improving clean F1 (see "Why R_bias succeeded" below and Domain Insight 22).

**Design principle established**: structural spatial priors belong in attention *routing* (logit space), not in feature mixing. Only logit-space injection preserves per-sensor embedding integrity.

The distinction from V1's implicit fault tolerance remains key: V1 learns to be robust to *absent* sensors (hard faults); the spatial bias approach helps discount *present-but-wrong* sensors (soft faults) — the harder and more practically important fault class.

### Sensor fault augmentation

The existing augmentation (random channel masking, structured row/col masking) covers hard faults only (zeroed sensors). The SHM literature identifies a standard taxonomy of accelerometer faults relevant to forced-vibration testing; drift is excluded as it operates over hours/days and is negligible in 2s measurement windows.

| Fault type | Physical cause | Augmentation (current `lib/faults.py`) |
|---|---|---|
| **hard** (complete failure) | Cable disconnect, dead DAQ channel | Zero the sensor |
| **gain** | Mismatched sensitivity, cable resistance | Multiply by U(0.5, 0.8) or U(1.2, 1.7) |
| **bias** (offset) | Poor sensor zeroing, coupling issue | Add ±U(0.2, 1.0) × RMS |
| **gain_bias** (combined) | Loose/corroded connector | Scale U(0.5, 0.9) + offset ±U(0.1, 0.5) × RMS |
| **noise** | EMI / cable degradation | Additive noise σ = U(0.5, 2.0) × RMS |
| **stuck** | Frozen DAQ channel | Replace with sensor mean + noise (0.01 × RMS) |
| **partial** (attenuation) | Loose connector, signal reduced | Multiply by U(0.3, 0.7) |

All are generatable synthetically with no new recordings. Each produces a `y_fault` label for the affected sensors — free supervision from augmentation. Gain+bias as a combined mode is physically motivated: loose or corroded connectors degrade amplitude and introduce DC offset simultaneously.

### Fault head design

A **per-sensor binary fault classifier** (not DETR-style slots) is appropriate here: sensors have known identities (indices 0..29), so fault detection is "classify each of 30 sensors" not "find unknown objects." A simple linear head on the spatially-aware features → P(faulty) per sensor, trained with weighted BCE against augmentation labels.

The fault head is an **open-loop auxiliary output** — it provides a training signal for the spatial reasoning layer and a deployable diagnostic indicator for engineers, but does not directly modulate the damage cross-attention. This keeps the damage head robust even when the fault head misfires on unseen fault types.

### Attention map interpretability

The slot decoder's cross-attention weights over sensors, plotted on the physical sensor grid, provide direct interpretability. For a single-damage detection (the primary evaluation target at this stage):

- **Healthy sensors**: active damage slot attends to the sensor neighbourhood around the detected joint — a spatially concentrated attention map consistent with structural physics
- **Faulty sensor within that neighbourhood**: visible *attention valley* — a dip in a region that should show elevated attention

This gives engineers a two-panel output: (1) which joint is damaged, (2) which sensors were trusted or distrusted in making that prediction. This is a strong interpretability contribution for the SHM domain.

For clean visualisation, the final cross-attention layer of the slot decoder should use a small number of heads (or single-head) so attention maps are directly readable without averaging artefacts.

### Evaluation scope

Fault-robustness evaluation is **stratified by K=1 and K=2** subsets — both are reported in every CSV (see `subset` column in `saved_results/<dataset>-fault/eval_fault_*.csv`). 7-story and Qatar provide real double-damage test sets (split-½ on Qatar, native K=2 partition on 7-story); LUMO is K≤1 only.

The sweep varies:
- Fault type (7 types: hard, gain, bias, gain_bias, noise, stuck, partial)
- Fault fraction (`{0, 0.2, 0.5, 0.8}` sensors affected, 3 random-sensor-selection repeats per cell)
- Injection paradigm (per-sample on 7-story; per-recording on Qatar/LUMO to match physical deployment)

Not currently varied: *fault location relative to damage* (near-damage vs far-damage sensors). This is a natural next ablation — near-damage faults should be the hardest case because the faulted sensor would otherwise carry the strongest evidence.

---

## Experimental Findings: Sensor Fault Robustness Phase

Fault robustness is evaluated across all three datasets (7-story, LUMO, Qatar) using a consistent protocol: training with `--p-hard 0.3 --p-soft 0.3 --p-struct-mask 0.3` (200 epochs), evaluation with fault-ratio sweep `{0.0, 0.2, 0.5, 0.8}`, 7 fault types, 3 repeats. Full per-fault-type tables are in `CLAUDE.md` (7-story and LUMO complete; Qatar pending retraining). *(An earlier 6-point sweep `{0.0, 0.1, 0.33, 0.5, 0.67, 0.8}` was used for the initial LUMO fault run; results in `saved_results/lumo-fault/` reflect that older protocol.)*

### [Historical] Prior Qatar results — checkpoints deleted, not reproducible

> **Status: Not reproducible from current checkpoints.** The numbers below come from a Qatar-only experimental round using nf levels `{1, 3, 5, 10, 15}` (incompatible with the current `{0, 0.2·S, 0.5·S, 0.8·S}` protocol) and include the abandoned C+SF+FH variant (Insight #21). Underlying checkpoints are deleted; no current run produces these values. Kept as a record of the design conclusions that motivated the current C+FH+SB architecture. Replace once `states/qatar-fault/` is retrained and `eval_fault_*.csv` lands.
>
> | Fault Type  | v1    | DR    | C     | C+FH  | C+SF+FH | C+FH+SB |
> |-------------|-------|-------|-------|-------|---------|---------|
> | hard        | 0.974 | 0.989 | 0.988 | 0.987 | 0.960   | **0.988** |
> | gain        | 0.789 | 0.885 | 0.920 | 0.990 | 0.797   | **0.993** |
> | bias        | 0.278 | 0.806 | 0.737 | 0.861 | 0.902   | **0.981** |
> | gain_bias   | 0.352 | 0.956 | 0.774 | 0.929 | 0.849   | **0.990** |
> | noise       | 0.255 | 0.050 | 0.360 | 0.928 | 0.498   | **0.936** |
> | stuck       | 0.974 | 0.989 | 0.987 | 0.985 | 0.957   | **0.988** |
> | partial     | 0.974 | 0.985 | 0.988 | 0.983 | 0.850   | **0.991** |
> | **clean**   | 0.975 | 0.991 | 0.992 | 0.992 | 0.984   | **0.994** |
>
> *v1, DR, C baselines were trained without fault augmentation. C+SF+FH used old extreme-magnitude fault aug and is included as an ablation. C+FH and C+FH+SB used moderate-magnitude aug matching current `lib/faults.py`.*

### Why SensorSpatialLayer (C+SF+FH) failed

Adding all-to-all cross-sensor self-attention *before* the slot decoder — the intuitive
approach — consistently degraded damage F1 relative to fault augmentation alone (C+FH), with
the largest drops on soft fault types (gain −0.193, noise −0.430, partial −0.133) and only
minor effects on hard/stuck faults. The mechanism is contamination: when the faulty sensor
attends to its neighbours and vice versa, the faulty sensor's anomalous features are
distributed into its neighbours' representations via attention-weighted sums. Clean sensors
absorb fault artifacts; the spatial coherence contrast — the signal that makes a faulty
sensor identifiable — is destroyed before the slot decoder sees it. Hard and stuck faults
are immune because a zeroed sensor contributes nothing to attention sums. The lesson is that
spatial reasoning that modifies sensor feature embeddings *before* the damage head is
counterproductive for fault isolation. See Domain Insight 21.

### Why R_bias (C+FH+SB) succeeded

The structural affinity bias injects the spatial prior into attention *routing* (cross-attention
logits) rather than into sensor feature representations. An (L+1)×S parameter matrix R,
physically initialised from the 4-connected grid adjacency (R[l,i]=1 if sensor i is adjacent
to joint l), is applied dynamically at decoder layers ≥1: the bias for slot k is
`softmax(loc_logits_k) @ R`, a per-sample, per-slot weighting that directs each slot to
attend toward sensors near its current location estimate. Sensor embeddings are never
modified — clean sensors stay clean — so no contamination path exists. Layer 0 runs
unbiased because the slot queries have not yet seen data and cannot provide a meaningful
location estimate. R is a learnable parameter, refining the binary adjacency prior through
training. C+FH+SB achieves the best damage F1 on all seven fault types while also improving
clean F1 (0.994 vs 0.992 for baseline C), confirming zero clean-performance cost.
See Domain Insight 22.

**Design principle**: structural spatial priors belong in attention *routing* (logit space),
not in feature mixing. The two approaches are not equivalent; only logit-space injection
preserves the per-sensor embedding integrity that both the damage head and fault
discrimination require.

---

## Open Research Questions

1. ~~Does the sensor spatial reasoning layer alone improve damage F1 over augmentation-only?~~ **Answered**: spatial layer + fault aug (C+SF+FH) *hurts* relative to fault aug alone (C+FH) — feature contamination outweighs any coherence benefit.
2. ~~Does adding the per-sensor fault head improve F1 further?~~ **Answered**: yes — C+FH substantially outperforms C across all soft fault types; C+FH+SB improves further via the structural routing prior.
3. Which soft fault types most commonly occur in real forced-vibration field deployments — and does the model trained on synthetic soft faults generalise to them?
4. Does the attention valley appear cleanly in single-damage cases with a nearby faulty sensor, or does the spatial reasoning need to be stronger to produce interpretable maps?
5. What is the right physical representation of sensor layout — 2D grid coordinates (attention + positional embedding) or explicit adjacency (GNN)? Do they differ in accuracy or interpretability?
6. What fraction of faulty sensors causes graceful degradation vs. catastrophic failure of the damage head?
7. Can the C-head's ∅-class boundary be made adaptive to input distribution statistics (e.g. ambient energy level), so that the slot mechanism retains its K>1 advantage without losing robustness under non-stationary field excitation? Or is a regime-switching strategy (C for K>1, DR for K≤1) more practical?
8. Does temporal-extrapolation splitting on LUMO (train on early scenarios, test on later ones — different season, different bracing pattern) further degrade all heads, and does the ranking remain DR > C > v1?
