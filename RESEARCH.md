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

Two datasets are used, representing the simulation → laboratory progression:

| Dataset | Nature | Sensors (S) | Joints (L) | Damage | Role |
|---------|--------|-------------|------------|--------|------|
| 7-story frame | Simulation | 65 (full) / 9 (sparse) | 70 | Continuous severity, K=1 and K=2 | Design exploration, ablations |
| Qatar SHM Benchmark | Physical lab | 30 | 30 | Binary presence, K=1 (train) / K=2 (test) | Real-world validation |

The sim → lab generalization is itself a contribution: design and ablate on clean simulation, validate transfer to physical data. A future real field-structure dataset would close the full sim → lab → field loop.

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

---

## Open Research Questions

1. Does adding cross-sensor spatial reasoning (sensor self-attention in feature space) meaningfully improve disambiguation of fault vs. damage, or does augmentation-based implicit robustness suffice?
2. What is the right inductive bias for sensor spatial correlations — full self-attention (data-driven), or structured attention over the known physical grid layout?
3. Does explicit fault detection (supervised, from augmentation labels) improve damage localisation compared to purely implicit robustness?
4. Can the same architecture handle both hard faults (zero/disconnect) and soft faults (drift, gain error, stuck value) — given that hard faults are already handled implicitly?
5. How does performance degrade as the fraction of faulty sensors increases — is there a graceful degradation curve, and where does it break?
6. What is the right evaluation protocol — random fault patterns, structured row/column faults, or held-out fault types not seen during training?
