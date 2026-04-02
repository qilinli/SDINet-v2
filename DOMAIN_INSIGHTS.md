# Domain Insights: ML Methodology ↔ Structural Damage Identification

Recorded insights connecting deep learning design choices to structural engineering
rationale. Intended to assist paper writing for domain journals (e.g. Engineering
Structures) where reviewers are structural engineers, not ML researchers.

Each entry: domain rationale → ML implication → quote-ready phrasing.

---

## 1. Damage events are a set, not 70 independent binary decisions

**Domain rationale**
A structure under monitoring experiences K simultaneous failure events — each event
is characterised by a location and a severity. These K events are unordered and
physically distinct. The physics produces K failure modes, not 70 independent
location states.

**ML implication**
Treating the 70 structural locations as independent binary variables (per-location
BCE) imposes a decomposition the physics does not require and cannot naturally
enforce sparsity or count. Set prediction (DETR-style) directly models the problem
structure: K_max slots, each responsible for detecting one damage instance or
reporting none.

**Where in paper**
Methodology (architecture motivation for DETR approach), Discussion (limitations
of per-location BCE)

**Quote-ready**
> Structural damage is inherently a set detection problem: a structure experiencing
> K simultaneous failure events produces K unordered (location, severity) pairs.
> Treating the 70 monitoring locations as 70 independent binary classifiers imposes
> a per-location decomposition that the underlying physics does not require.
> Set-prediction architectures [DETR] model this structure directly, predicting
> each damage instance as a complete (location, severity) tuple without requiring
> a post-hoc count decision.

---

## 2. Counting and localisation are governed by different physical scales

**Domain rationale**
The total number of damaged locations affects global structural dynamics: overall
stiffness reduction, energy redistribution across the structure, and the number of
shifted natural frequency modes. These are global signatures readable from the
aggregate sensor response. By contrast, identifying *which* specific locations are
damaged requires detecting localised anomalies — which sensors respond differently
relative to their neighbours — a comparative, local task.

**ML implication**
A counting head (K prediction) should operate on globally pooled features
(mean over sensor dimension), while the localisation head (presence/severity per
location) should use importance-weighted per-sensor aggregation. Conflating the
two tasks in a single sigmoid threshold forces one head to solve both, which it
cannot do optimally.

**Where in paper**
Methodology (architecture rationale for separate global K head and local
presence branch), Discussion

**Quote-ready**
> The number of simultaneously damaged locations is encoded in the global
> structural response — shifts in natural frequencies, changes in total energy
> dissipation, and modifications to global mode shapes are aggregate signatures
> of damage count. Identifying *which* locations are damaged, however, requires
> comparing localised sensor responses. This physical distinction motivates a
> two-stage architecture: a globally pooled counting module and a per-sensor,
> importance-weighted localisation module.

---

## 3. Maximum credible simultaneous damage count is bounded by structural physics

**Domain rationale**
A monitored civil structure (bridge, building frame) will not experience 20
simultaneous structural failures during normal operational monitoring. If it did,
the structure would have already collapsed or would be visibly unserviceable.
Maintenance monitoring targets early-stage damage, where K is small. Setting
K_max = 4 reflects engineering judgement about credible damage scenarios, not
an arbitrary model hyperparameter.

**ML implication**
K_max in a DETR-style architecture (number of detection slots) is a physically
grounded prior rather than an architectural limitation. It sets a capacity ceiling
that the engineering domain validates. Unlike classification bins in a K-head
(where K ≥ max_k collapses into one catch-all class), exceeding K_max in DETR
causes graceful degradation: K_max damages are reported and one is missed, rather
than a catastrophic misclassification.

**Where in paper**
Methodology (justification of K_max choice), Discussion

**Quote-ready**
> The maximum number of simultaneous damage events credible in a maintained civil
> structure is physically bounded: a structure sustaining more than four or five
> concurrent structural failures would exhibit gross deformations visible to routine
> inspection. Setting K_max = 4 is therefore a domain-informed prior rather than an
> architectural constraint, consistent with the early-warning monitoring context
> in which this system operates.

---

## 4. Unknown damage count at deployment — the training/inference mismatch

**Domain rationale**
In real structural health monitoring, an alarm is triggered by anomalous sensor
readings *before* the damage count is known. The inspection goal is precisely to
determine K and the damage locations. A model requiring K as input cannot be used
in this setting. Similarly, a detection threshold calibrated on training data with
K ∈ {1, 2} will be miscalibrated when K = 0 (healthy) or K ≥ 3 occurs in service.

**ML implication**
Fixed absolute thresholds (sigmoid > θ) are calibrated to a specific K
distribution and degrade outside it. Ratio thresholding (sigmoid > α · max sigmoid)
adapts per sample, normalising by the model's own confidence scale. DETR-style
set prediction eliminates the threshold entirely — the null class handles K = 0
natively and each slot's decision is independent of K.

**Where in paper**
Discussion (practical deployment), Limitations (threshold sensitivity),
Methodology (motivation for ratio thresholding and DETR)

**Quote-ready**
> In operational structural monitoring, sensor anomalies trigger inspection before
> the number of damage events is known — determining K is part of the diagnostic
> objective, not a precondition for it. A detection model that requires a
> pre-specified K, or whose threshold was calibrated to a training distribution
> over K, cannot generalise to unseen damage scenarios. This motivates detection
> formulations that are inherently agnostic to K at inference time.

---

## 5. Multiple-instance learning over sensors is physically motivated

**Domain rationale**
In a sensor network, not all sensors are equally informative about all structural
locations. Sensors physically near a damaged location respond more strongly to
local stiffness changes. Sensors far from the damage carry little location-specific
information. Furthermore, sensor hardware failures (dead sensors, communication
dropouts) are routine in long-term monitoring and must not invalidate predictions.

**ML implication**
Importance-weighted pooling over the sensor dimension (MIL) is physically
motivated: the learned importance weights approximate the sensor-to-location
relevance map. Sensor dropout during training (randomly masking sensor indices)
simulates real-world hardware failures and improves robustness. The shared
importance branch across presence and severity heads reflects the physical fact
that the sensors revealing *whether* a location is damaged are the same ones
revealing *how much* it is damaged.

**Where in paper**
Methodology (MIL architecture rationale), Experiments (sensor dropout ablation)

**Quote-ready**
> The sensitivity of individual sensors to damage at a given structural location
> depends on their spatial proximity and the structural load path connecting them.
> Multiple-instance learning with learned importance weights formalises this
> physical relationship: the importance weights approximate the latent sensor
> relevance map without requiring explicit sensor-to-location assignments.
> Sensor dropout during training further reflects operational reality, where
> communication failures and hardware faults routinely render subsets of sensors
> unavailable.

---

## 6. Ratio thresholding adapts to structural response magnitude variation

**Domain rationale**
Structural dynamic response amplitude depends on loading conditions (traffic
intensity, wind speed, excitation level). A sensor network recording under light
traffic produces smaller response magnitudes than the same structure under heavy
loading. A fixed detection threshold (sigmoid > 0.70) calibrated on one loading
regime may be systematically wrong under another.

**ML implication**
Ratio thresholding — reporting locations where sigmoid(p_l) > α · max_l sigmoid(p_l)
and max_l sigmoid(p_l) > β — normalises by the model's own confidence scale on
each sample. The ratio α is invariant to uniform scaling of logit magnitudes,
making the decision boundary loading-condition invariant. The gate β handles the
K = 0 case (healthy structure) when the model's maximum confidence is low.

**Where in paper**
Methodology (inference procedure), Discussion (generalisation across loading
conditions)

**Quote-ready**
> Structural dynamic response amplitude is a function of operational loading
> conditions and cannot be assumed constant across monitoring sessions. A fixed
> detection threshold calibrated on training data will be miscalibrated under
> loading regimes that produce systematically higher or lower response magnitudes.
> Ratio thresholding normalises the detection decision by the model's own maximum
> confidence on each sample, producing a decision boundary that is invariant to
> loading-induced amplitude scaling.

---

## 7. Sparse damage in 70 locations — BCE class imbalance is physically determined

**Domain rationale**
At any monitoring instant, only K of the L = 70 structural locations are damaged.
Structural damage is spatially sparse by physical necessity — widespread simultaneous
damage corresponds to structural collapse, not the early-warning monitoring regime.
For K_mean = 1.5 and L = 70, the positive class (damaged) represents ~2% of all
location-instances.

**ML implication**
Without correction, binary cross-entropy loss is dominated by the 98% negative
class (undamaged locations) and the model learns to predict no damage everywhere.
The positive class weight pos_weight = (L − K_mean) / K_mean is derived directly
from the physical sparsity ratio, not from hyperparameter search. For combined
single and double damage training (K_mean = 1.5): pos_weight ≈ 45.7.

**Where in paper**
Methodology (loss design), implementation details

**Quote-ready**
> Structural damage is spatially sparse: in a 70-location monitoring network
> subject to one or two simultaneous damage events, fewer than 3% of
> location-monitoring-instances carry a positive damage label. This physical
> sparsity directly determines the binary cross-entropy class weight:
> pos_weight = (L − K_mean) / K_mean, providing a principled, domain-derived
> alternative to empirical hyperparameter search.

---

## 8. Healthy structure (K = 0) must be a trainable scenario

**Domain rationale**
A structural monitoring system spends most of its operational life observing a
healthy structure. False alarms (predicting damage in a healthy structure) have
significant operational cost: unnecessary shutdowns, inspection mobilisation, and
erosion of engineer trust in the monitoring system. A model trained exclusively on
damaged structures has no learned representation of "healthy" and will activate
spuriously on healthy data.

**ML implication**
Training only on K ∈ {1, 2} means sigmoid presence outputs have never been
trained toward zero simultaneously across all locations. Including K = 0 samples
provides the gradient signal needed to suppress all activations for healthy
inputs. Ratio thresholding's β gate partially compensates (requiring minimum
absolute confidence before any location is reported) but cannot substitute for
direct training signal on healthy data.

**Where in paper**
Methodology (training data), Discussion (false alarm rate), Limitations

**Quote-ready**
> Operational deployment requires that the model correctly identifies the absence
> of damage — the condition that characterises the majority of monitoring time.
> Training exclusively on damaged scenarios provides no gradient signal for
> suppressing all location activations simultaneously, resulting in systematic
> false positives on healthy structures. Incorporating healthy (K = 0) training
> samples is therefore not merely a data augmentation choice but a prerequisite
> for operationally credible monitoring.

---

## 9. Structurally connected locations have correlated damage — motivation for GNN

**Domain rationale**
Structural damage propagates along load paths. When a structural element is damaged,
the loads it previously carried are redistributed to adjacent connected elements,
increasing their stress and damage probability. Beam-column joints, adjacent spans,
and elements sharing a node are structurally coupled: damage at one location is
physically correlated with increased damage risk at neighbours.

**ML implication**
Per-location predictions from the B head are made independently — the model has
no mechanism to exploit spatial correlations between connected locations. A graph
neural network over the structural topology (70 nodes, edges from structural
connectivity) allows predicted damage maps to be refined using physical adjacency:
a high-confidence detection at one node can increase confidence at structurally
connected nodes, and isolated activations inconsistent with the load path can be
suppressed.

**Where in paper**
Methodology (Approach D, GNN refinement), Discussion (physical interpretability)

**Quote-ready**
> Structural load redistribution implies that damage at one location increases
> stress — and therefore damage likelihood — at structurally connected neighbours.
> Per-location predictions that ignore this spatial coupling may produce damage
> maps inconsistent with the structural load path. A graph neural network defined
> over the structural connectivity graph allows predicted damage patterns to be
> refined against the physical topology, suppressing physically implausible
> isolated activations and reinforcing patterns consistent with load redistribution.

---

## 10. v1 scalar head fails on real data — physical interpretability of the failure mode

**Domain rationale**
Real accelerometer recordings differ from simulated training data in noise
characteristics, physical units, and excitation spectral content. The v1 global
damage scalar (tanh-activated) must simultaneously encode damage presence and
severity in a single value. Under distribution shift, this scalar can collapse
below the detection threshold even when damage is present, producing a complete
detection failure rather than a graceful degradation.

**ML implication**
The v1 model achieved top_k_recall = 0.00 and F1 = 0.00 on the real benchmark,
despite achieving top_k_recall = 0.98 on synthetic single-damage test data.
The per-location presence logits of the B head are relative scores (each location
scored against others via importance weighting) rather than absolute magnitudes,
making them more robust to global amplitude shifts from domain change.

**Where in paper**
Experiments (real benchmark results), Discussion (robustness to domain shift)

**Quote-ready**
> The v1 global damage scalar must encode both damage presence and severity in a
> single tanh-activated value calibrated on synthetic training data. Under
> real-world recording conditions — which differ in noise floor, excitation
> spectrum, and signal amplitude — this scalar collapses below the detection
> threshold, producing complete detection failure despite strong performance on
> held-out synthetic data. The per-location relative scoring in the B head is
> more robust to this domain shift, as detection decisions depend on the relative
> ordering of location scores rather than their absolute magnitudes.

---

## 11. Transformer cross-attention subsumes the MIL sensor aggregation mechanism

**Domain rationale**
In a sensor network, each structural location is best characterised by a specific
subset of sensors — those physically proximate to it or connected via load paths.
The MIL importance branch (Approach B) learns a scalar weight per (sensor, location)
pair to approximate this physical relevance. Transformer cross-attention learns the
same relationship but with greater expressivity: each damage query attends to sensors
with query-specific, content-dependent patterns that can vary across structural states.

**ML implication**
Cross-attention in the DETR-style decoder replaces the explicit importance branch.
The attention weights over sensor positions play the same role as MIL importance
weights, but are learned jointly with the detection task and updated iteratively
across decoder layers. Sensor dropout at inference maps directly to attention
masking (dropped sensors receive −∞ before softmax), which is more principled than
renormalising importance weights.

**Where in paper**
Methodology (DETR decoder design), Discussion (relationship between MIL and
attention-based aggregation)

**Quote-ready**
> The importance-weighted sensor aggregation introduced in the MIL framework is a
> special case of transformer cross-attention: both mechanisms learn to weight
> sensor contributions to each structural location's prediction according to their
> physical relevance. The DETR-style decoder replaces the explicit importance branch
> with multi-head cross-attention, recovering the MIL inductive bias while allowing
> richer, query-specific sensor relevance patterns that adapt to the structural state.
> Sensor unavailability is handled through attention masking, providing an exact
> treatment of missing sensors rather than the approximate renormalisation required
> by fixed importance weights.

---

## 12. Two-stage prediction mirrors the engineering diagnostic process

**Domain rationale**
Structural engineers diagnose damage in two conceptually distinct steps: first
localise the anomaly (which part of the structure is behaving differently?) and
then quantify its severity (how much has the stiffness or capacity reduced?).
Severity estimation is inherently conditioned on localisation — measuring how
much a member is damaged requires knowing which member to examine.

**ML implication**
A two-stage slot decoder (stage 1: attend over sensors to identify damaged
location; stage 2: re-attend conditioned on predicted location to estimate
severity) mirrors this diagnostic sequence. Severity head inputs that include
the predicted location embedding produce more accurate estimates than a single
head predicting both simultaneously, because the second attention can focus on
the sensors most informative for that specific location's severity.

**Where in paper**
Methodology (decoder architecture motivation), Discussion (interpretability
for structural engineers)

**Quote-ready**
> The two-stage slot decoder mirrors the sequential diagnostic process familiar
> to structural engineers: localisation precedes severity quantification, and
> the latter benefits from knowing the former. By conditioning the severity
> attention on the predicted location, the decoder directs its evidence
> gathering to the sensors most informative for that specific structural member,
> rather than averaging over the full sensor network.

---

## 13. Discrete enumerated locations simplify set detection dramatically vs image detection

**Domain rationale**
In image object detection, objects can appear anywhere in continuous 2D space —
the output space is infinite and bounding box regression requires continuous
coordinate prediction. In SHM, there are exactly L=70 predefined structural
monitoring locations. Every possible damage location is known in advance. The
problem is not searching for where damage might be, but deciding which of the
known candidates are active.

**ML implication**
This discrete structure makes the SHM analogue of DETR far simpler than image
DETR: no bounding box regression, no anchor generation, no IoU-based matching,
no multi-scale feature pyramids. Slot attention operates over 70 location feature
vectors (not continuous spatial queries), and Hungarian matching uses pure
classification cost. Approach B (per-location sigmoid) is the SHM analogue of
dense anchor-based detection (YOLO/SSD). DETR-lite is the SHM analogue of
set-prediction detection, but simplified by the finite discrete location space.

**Where in paper**
Methodology (architecture motivation for DETR-lite), Discussion (comparison to
image detection literature)

**Quote-ready**
> Unlike image object detection, where objects may appear at arbitrary positions
> in a continuous spatial domain, structural damage is constrained to a finite
> set of L predefined monitoring locations. This discrete structure fundamentally
> simplifies the detection problem: slot-based set prediction requires no bounding
> box regression, no anchor design, and no IoU-based matching. Each slot identifies
> its target location by content similarity to a precomputed library of
> location feature vectors, reducing the combinatorial search to an assignment
> problem over L=70 candidates.

---

## 14. DETR's no-object class is a principled "structural health" decision, but creates a training distribution trap

**Domain rationale**
In structural monitoring, reporting "no damage" is as consequential a decision as reporting damage — false alarms carry operational cost (inspection mobilisation, service interruption) while missed detections carry safety cost. A detection model should be able to make an explicit, well-calibrated "nothing detected" decision, not merely output a low score that fails to exceed an arbitrary threshold.

**ML implication**
DETR's ∅ slot class is the correct architectural instantiation of this requirement: each slot either commits to a specific location or explicitly reports absence. However, training exclusively on K=1 data (one damage per window) forces 4 of 5 slots to map to ∅ throughout training, instilling a very strong no-object prior. Under K=2 test inputs that are out-of-distribution (signal is a superposition the model has not seen), this ∅ prior dominates: more slots collapse to no-object rather than firing, causing mean_k_pred to drop *below* 1.0 — paradoxically worse than V1 (softmax, always predicts exactly 1) and DR (threshold, predicts ~1). The no-object weight `no_obj_weight` and the inference threshold on `is_obj = 1 − P(∅)` jointly control the false-alarm / missed-detection tradeoff and must be re-calibrated whenever the training K-distribution changes.

**Where in paper**
Discussion (multi-damage generalisation), Limitations (training distribution sensitivity of slot activation rate), Methodology (no_obj_weight design choice)

**Quote-ready**
> The DETR null class provides a principled "no damage" decision that avoids
> reliance on an arbitrary detection threshold. However, training on single-damage
> scenarios exclusively instils a strong no-object prior in the unused slots: when
> evaluated on double-damage inputs — a superposition not present in training —
> the model's slots suppress more aggressively than under single-damage, producing
> a mean predicted damage count below 1.0. This counter-intuitive behaviour
> highlights a training-distribution trap specific to set-prediction architectures:
> the no-object class, designed to handle absence, can override evidence of presence
> when the input falls outside the training support.

---

## 15. Linear structural systems under white noise allow synthetic multi-damage augmentation by signal superposition

**Domain rationale**
The Qatar benchmark uses broadband white noise excitation. For a linear structural system, the principle of superposition applies: the sensor response to two simultaneous structural changes is approximately equal to the sum of the individual responses to each change independently. This is the same physical linearity that underlies modal analysis and frequency response functions — the tools structural engineers use to characterise structural behaviour.

**ML implication**
Synthetic K=2 training samples can be generated by adding two single-damage sensor windows and forming the union of their damage labels. No new laboratory experiments are required. This augmentation is physically grounded rather than heuristic: it faithfully models the expected sensor response under simultaneous damage up to second-order interaction effects. The 5 real double-damage recordings available in the Qatar benchmark serve as a validation check on this approximation — if the model trained on synthetic superpositions generalises to the real double-damage set, the linearity assumption is confirmed empirically. The approach would not be valid for nonlinear excitation regimes (e.g. earthquake records, where large deformations introduce geometric nonlinearity).

**Where in paper**
Methodology (data augmentation for multi-damage generalisation), Discussion (validity of linearity assumption, comparison to real double-damage test results)

**Quote-ready**
> The linear structural hypothesis — that the sensor response to K simultaneous
> damage events is the superposition of K individual responses — is the foundation
> of classical modal-based SHM and holds for small-amplitude vibration under
> broadband excitation. This physical principle enables synthetic multi-damage
> training data to be constructed by adding pairs of single-damage sensor windows
> and forming the union of their damage labels, requiring no additional laboratory
> experiments. The available real double-damage recordings provide an empirical
> validation of the linearity approximation under the actual test conditions.

---

## 17. Synthetic double-damage augmentation via signal superposition breaks the DETR ∅ trap with minimal single-damage degradation

**Domain rationale**
The Qatar SHM benchmark provides only 5 real double-damage recordings — far too few for supervised multi-damage training. However, the structure is a linear system excited by broadband white noise, so sensor responses superpose approximately: `x_1+2 ≈ x_1 + x_2`. This means synthetic K=2 training windows can be constructed by adding pairs of single-damage windows from different recordings, with `y = max(y_1, y_2)` as the label union. No new experiments are required.

**ML implication**
Training the DETR-style C-head with 50% synthetic K=2 augmentation (`--p-mix 0.5`) and 200 epochs breaks the ∅-prior trap identified in Insight 14. Empirical results on Qatar:

| Setting | Double F1 | Double mean_k_pred | Single F1 |
|---------|-----------|-------------------|-----------|
| Baseline (K=1 only) | 0.317 | 0.805 | 0.990 |
| p_mix=0.3, 50 ep | 0.342 | 1.462 | 0.961 |
| p_mix=0.5, 100 ep | 0.410 | 1.728 | 0.977 |
| p_mix=0.5, 200 ep | **0.508** | **1.871** | **0.984** |

Double-damage F1 doubled from 0.317 to 0.508 while single-damage F1 barely changed (0.990 → 0.984). The residual gap (mean_k_pred=1.87 vs 2.0, precision=0.53) is a localization problem on zero-shot inference: the model correctly predicts K=2 but sometimes misidentifies the second location, having never seen real double-damage signal during training.

Three implementation details matter for quality: (1) partner windows are drawn from a different recording to guarantee K=2 labels; (2) healthy windows are excluded from the partner pool; (3) the sum is divided by √2 to preserve RMS amplitude so val/test windows look in-distribution.

V1 (softmax) is architecturally incompatible with K=2 training — adding p_mix hurts its double-damage performance because the softmax constraint means any K=2 gradient is antagonistic to the K=1 objective. DR shows modest improvement (mean_k_pred: 0.97 → 1.16 at 50 epochs) but is limited by ratio calibration trained on K=1 val data.

**Where in paper**
Methods (synthetic augmentation strategy), Results (multi-damage generalisation experiment), Discussion (architecture asymmetry under K=2 augmentation)

**Quote-ready**
> Under white-noise excitation on a linear structural system, sensor responses
> superpose approximately: the combined response to two damage conditions is well
> approximated by the sum of the individual responses. This physical property
> permits on-the-fly generation of synthetic K=2 training windows by adding pairs
> of single-damage recordings and forming the union of their damage labels —
> requiring no additional laboratory measurements. Training the DETR-style detector
> with 50% synthetic double-damage augmentation (200 epochs) doubled the
> double-damage F1 score from 0.32 to 0.51, while preserving single-damage
> accuracy (F1: 0.99 → 0.98). The remaining localisation error reflects the
> inherent difficulty of zero-shot multi-damage inference: the model correctly
> predicts two damaged locations but occasionally misidentifies the second joint.

**Follow-up (see Insight 18)**
A leave-one-out experiment over the 5 real double-damage recordings confirmed that the residual gap is *spatially specific*, not data-quantity limited. When the model trains on the first 50% of all 5 joint pairs and tests on the second 50%, double-damage F1 reaches **0.9996** — demonstrating the network learns each joint pair's sensor co-activation pattern rather than a general K=2 detector.

---

## 18. Multi-damage generalisation in DETR is spatially specific: the model must see each joint pair, not just K=2 examples

**Domain rationale**
A real SHM deployment may encounter damage at any of the C(L,2) = 435 possible joint pairs on the Qatar structure. A monitoring system that can only detect the specific pairs seen during training is not practically useful — it fails silently on novel combinations. Understanding whether poor multi-damage performance is due to (A) never seeing the specific joint pair or (B) insufficient K=2 training data in general determines what additional experiments are needed.

**ML implication**
Three controlled experiments on the Qatar C-head isolate the root cause:

| Setting | Double-damage F1 | mean_k_pred | Test condition |
|---------|-----------------|-------------|----------------|
| Synthetic augmentation only (p_mix=0.5, 200 ep) | 0.508 | 1.871 | all 5 pairs (unseen) |
| LOO — 4 real pairs in training, 1 held out (300 ep) | 0.439 (mean) | 2.35 | 1 unseen pair per fold |
| All-5 time split — first½ train, second½ test (100 ep) | **0.9996** | 2.002 | same pairs, later in time |

The split result (F1=0.9996) is decisive: when the model sees the first half of each recording during training, it achieves near-perfect detection on the second half. Combined with the LOO variance (fold 3: F1=0.055 for never-seen j21+j25; fold 4: F1=0.740 for adjacent joints partially seen), this confirms root cause **A — spatial specificity**.

The model is not learning a general "two damaged joints are present" representation. It is learning to associate specific sensor co-activation patterns (the signature of joints 3+26 vibrating simultaneously) with K=2 slot activation. When the exact pair is absent from training, those co-activation patterns never appear in the loss gradient, and the model cannot fire two slots for novel combinations even though it has learned to fire two slots for known ones.

Synthetic signal superposition (`x_i + x_j`, physically valid for linear systems) partially bridges the gap — it exposes the model to many virtual joint pair combinations. But the real double-damage signal includes nonlinear structural coupling effects that the linear superposition approximation misses, causing the model's second-slot confidence to be lower on real novel pairs than on synthetic ones.

**Follow-up: targeted synthesis does not close the gap (see Insight 19)**
A targeted variant of synthetic mixing was tested: instead of drawing random partners, the mix pool for each joint is restricted to windows of its known test-pair partner (e.g. joint 3 → partner always from joint 26). This ensures every K=2 training window corresponds to one of the 5 actual test combinations. Despite this, double-damage F1 was **0.490** — essentially unchanged from random synthesis (0.508) and far below split-double (0.9996). The model even predicted *fewer* K=2 confidently (mean_k_pred 1.47 vs 1.87). This confirms that pair coverage is not the bottleneck: the physical approximation error in the synthetic signal is.

**Practical implication for SHM deployment**: to achieve robust K=2 detection across all possible damage combinations, the training set must include real multi-damage recordings covering a representative sample of the joint pair space. Synthetic augmentation can improve generalisation at the margins but cannot substitute for real measurements of the target combinations. For the Qatar structure with 435 possible pairs and 5 real recordings, ~430 pairs are never seen; systematic multi-damage experiments or transfer learning across structures would be required for fully general multi-damage detection.

**Where in paper**
Results (ablation: synthetic vs LOO vs split-double), Discussion (practical limitations of synthetic augmentation, requirement for real multi-damage data in SHM), Future work (multi-damage dataset collection strategy)

**Quote-ready**
> A leave-one-out evaluation across the five available double-damage recordings
> reveals high variance in detection F1 (0.055–0.740), with the lowest scores
> on joint pairs most dissimilar to the training set. When all five pairs are
> included in training via a temporal split, F1 reaches 0.9996, confirming that
> the DETR detector has sufficient architectural capacity for multi-damage
> detection but requires exposure to the specific sensor co-activation patterns
> of each damage combination during training. Synthetic signal superposition
> improves generalisation across unseen pairs but cannot fully substitute for
> real multi-damage recordings, which contain structural coupling effects absent
> from the linear superposition approximation.

---

## 19. Targeted synthetic augmentation does not outperform random mixing: physical signal fidelity, not pair coverage, is the bottleneck

**Domain rationale**
After establishing that multi-damage generalisation is spatially specific (Insight 18), a natural engineering response is: if the model needs to see each joint pair, can we construct targeted synthetic signals for the exact test combinations? Dataset A contains single-damage recordings for all 30 joints, so synthetic K=2 windows for any of the 5 real test pairs can be constructed by mixing the appropriate pair. If targeted synthesis reaches near-real performance, it would eliminate the need for additional physical experiments.

**ML implication**
Targeted synthesis restricts the synthetic mix pool so each K=1 window from joint i is always paired with a window from its known test-pair partner j (e.g. joint 3 always mixed with joint 26). Every K=2 training window therefore matches one of the 5 actual test combinations.

| Setting | Double F1 | mean_k_pred | Notes |
|---------|-----------|-------------|-------|
| Random p_mix=0.5 (200 ep) | 0.508 | 1.871 | random partner from 30 joints |
| **Targeted p_mix=0.5 (200 ep)** | **0.490** | **1.467** | partner restricted to test-pair joint |
| Split-double (real, 100 ep) | 0.9996 | 2.002 | actual physical recordings |

Targeted synthesis was *worse* than random mixing: F1 lower (0.490 vs 0.508) and mean_k_pred lower (1.47 vs 1.87). The model trained with targeted synthesis is less confident about predicting K=2 on the real test recordings than the randomly augmented model.

Two mechanisms explain this:
1. **Diversity collapse**: random synthesis exposes the model to all ~435 pair combinations, giving it many diverse K=2 "shapes" and forcing it to learn a general two-slot activation pattern. Targeted synthesis collapses this to 5 patterns, making the K=2 representation more brittle.
2. **Signal fidelity is the real bottleneck**: the real double-damage response includes nonlinear structural coupling effects that `x_i + x_j` does not capture. These are present regardless of which pairs are targeted. The model trained on synthetic signals forms inaccurate expectations about what K=2 looks like physically, and targeted training amplifies this by making the synthetic pattern more specific and more different from reality.

**Practical implication**: for linear structural systems under broadband excitation, synthetic superposition (`x_i + x_j`) provides a useful but lossy proxy for real multi-damage data. The approximation error — not the coverage of specific pairs — is the fundamental limit of the synthetic augmentation strategy. Better options: (1) collect real multi-damage recordings for the target combinations, (2) use higher-fidelity simulation (FEM) to generate virtual multi-damage responses, or (3) accept the F1≈0.5 ceiling as the operational limit when only single-damage reference data is available.

**Where in paper**
Results (targeted synthesis ablation), Discussion (why physical signal fidelity matters, limits of linear superposition assumption)

**Quote-ready**
> Restricting synthetic mixing to the exact joint combinations present in the
> double-damage test set did not improve detection F1 (0.49 vs 0.51 for random
> mixing), and actually reduced mean predicted damage count (1.47 vs 1.87).
> This demonstrates that pair coverage is not the bottleneck for synthetic
> augmentation: the physical approximation error inherent in linear signal
> superposition limits performance regardless of which combinations are targeted.
> Real multi-damage recordings, which capture the true nonlinear structural
> response, remain necessary for reliable K≥2 detection.

---

## 16. DETR is the only architecture among V1, DR, C with the correct inductive bias for multi-damage generalisation

**Domain rationale**
A civil structure can sustain one, two, or more simultaneous damage events depending on loading history and maintenance state. A monitoring system calibrated on single-damage training data must degrade gracefully when K=2 or K=3 occurs in service. The three architectures degrade differently, and only one has the structural capacity to predict K>1 without post-hoc modifications.

**ML implication**
V1 uses a softmax over L locations: this forces exactly one location to dominate, making K=2 detection structurally impossible — the model can at best split probability mass between two locations rather than assert both are damaged. DR uses independent per-location scores with a ratio threshold, allowing K>1 predictions if multiple locations exceed the threshold simultaneously; threshold recalibration on multi-damage data enables K=2 without retraining. C (DETR-style) predicts an unordered set of (location, severity) tuples with no constraint on count — it is the only architecture whose training objective and output space are specifically designed to handle variable K. The barrier is the ∅ prior from K=1 training (Insight 14), not an architectural limitation. With weakened ∅ prior — achieved by including synthetic or real K=2 training examples, or by lowering the `no_obj_weight` — C is expected to outperform V1 and DR on multi-damage recall because its slots can independently fire for each damage without competing through a shared softmax or a global threshold.

**Where in paper**
Discussion (architecture comparison for multi-damage scenarios), Motivation for Approach C, Future work (multi-damage training)

**Quote-ready**
> Of the three architectures evaluated, only the DETR-style slot predictor
> has an output space and training objective designed for variable-cardinality
> damage sets. The v1 softmax head structurally limits predictions to a single
> dominant location; the DR direct-regression head requires threshold recalibration
> to produce K>1 detections. The slot predictor can, in principle, activate
> independent slots for each damage event without architectural modification.
> The barrier to multi-damage generalisation is not structural but statistical:
> training exclusively on K=1 data instils a strong no-object prior that must
> be weakened — through multi-damage training examples or no-object weight
> adjustment — before the architecture's full capacity is realised.
