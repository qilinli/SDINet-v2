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
Discussion (physical interpretability, motivation for structural spatial priors)
*Note: "Approach D" (GNN post-refinement) was a speculative direction never implemented. The structural prior is instead realised via the R_bias attention routing mechanism (see Insight 22).*

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

> **[Historical — status: not reproducible from current checkpoints.]** The F1=0.00 / top_k_recall=0.00 figures below predate (a) the 7-story preprocessing fix (truncation-restore after FIR ringing on zero-padded simulation tails, see Insight #26), and (b) the `dmg_gate` calibration fix rolled out for v1 on small-L datasets (Qatar, LUMO). Both changes are expected to materially improve v1 on the real benchmark. The *mechanism* argued here — a single tanh-scalar encoding both presence and severity is brittle to domain shift — is still a valid design argument, but the headline numbers are not current evidence for it.

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

> **[Historical — status: not reproducible from current checkpoints.]** The `--p-mix` CLI flag has been removed and the `states/qatar-pmix/` checkpoint cohort is deleted. The table below is preserved as an evidentiary trace of the mechanism (synthetic K=2 superposition breaks the ∅ trap) but cannot be re-run against the current training script. Current Qatar double-damage results are in `states/qatar-dd-split/` and are governed by Insight #18's spatial-specificity conclusion, which superseded the p_mix programme.

**Domain rationale**
The Qatar SHM benchmark provides only 5 real double-damage recordings — far too few for supervised multi-damage training. However, the structure is a linear system excited by broadband white noise, so sensor responses superpose approximately: `x_1+2 ≈ x_1 + x_2`. This means synthetic K=2 training windows can be constructed by adding pairs of single-damage windows from different recordings, with `y = max(y_1, y_2)` as the label union. No new experiments are required.

**ML implication**
Training the DETR-style C-head with 50% synthetic K=2 augmentation (formerly `--p-mix 0.5`, flag since removed) and 200 epochs breaks the ∅-prior trap identified in Insight 14. Empirical results on Qatar (historical — `states/qatar-pmix/` deleted, superseded by LOO/split-double experiments):

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

## 20. Spatial coherence of forced-vibration response distinguishes structural damage from sensor faults

**Domain rationale**
Under forced excitation on a linear structural system, the vibration response at every sensor is structurally constrained — energy propagates from any excitation point through the structural mode shapes to the entire sensor array in a predictable, physically coupled pattern. This creates a key observable:

- **Structural damage** at a joint produces a *spatially coherent* anomaly: multiple sensors shift in amplitude and phase in a pattern consistent with the modified mode shapes. The damaged member's stiffness change propagates outward — sensors physically near the damaged joint show correlated elevated responses, and even distant sensors are shifted in a patterned way.
- **Sensor fault** (cable disconnect, gain drift, stuck value, saturation) produces a *spatially incoherent* anomaly: one sensor reads abnormally while its physical neighbours read normally, which is inconsistent with how forced vibration actually propagates through the structure.

This coherence/incoherence signal is the key discriminating feature between damage and fault — accessible *without any external oracle*, purely from the spatial pattern of the simultaneous sensor readings.

**Why raw signal space is insufficient**
In raw time-series space, damage also distorts inter-sensor correlations near the affected joint — the spatial ambiguity is irresolvable there. Feature space (post-backbone) is the right target: after the backbone has compressed temporal vibration into structural-response representations, a faulty sensor produces a feature vector that is spatially inconsistent in terms of *structural response pattern*, not just raw amplitude. The spatial coherence check becomes tractable and discrimination-effective in feature space.

**Why channel-masking-only augmentation is insufficient**
Training with sensor drop-out (zeroing channels) teaches the model to be robust to *absent* sensors. But the critical practical challenge is *present-but-wrong* sensors — gain drift, saturation, stuck values — where the sensor actively provides a misleading signal. These require the model to have internalised the expected spatial correlation structure, not merely learned to predict from a subset of sensors. The V1 model already achieves the former; the spatial coherence approach targets the latter.

**ML implication**
A sensor spatial reasoning layer — placed between the per-sensor backbone features and the damage detection slots — allows each sensor to observe its physical neighbours in feature space. A sensor whose feature is spatially inconsistent with its neighbours (faulty) becomes identifiable and can be down-weighted before the damage head runs. This requires:
1. Known sensor layout as a structural prior (sensor grid coordinates)
2. A spatial reasoning module (positional-aware self-attention or GNN over the sensor adjacency graph)
3. Feature-space consistency checking — not raw-signal coherence

**Interpretability implication**
The slot decoder's cross-attention weights over sensors, plotted on the physical sensor grid, visualise which sensors the model trusted for each damage prediction. A correctly detected damage shows concentrated attention on the sensor neighbourhood around the damaged joint. A faulty sensor within that neighbourhood appears as an *attention valley* — a visible dip in a region that should otherwise show elevated attention. This is directly interpretable by structural engineers: "the model detected damage at joint X by attending to sensors A, B, C, and flagged sensor D as unreliable."

**Where in paper**
Methodology (motivation for sensor spatial reasoning layer), Discussion (fault tolerance mechanism, interpretability of attention maps), Contribution statement

**Quote-ready**
> Under forced excitation on a linear structural system, structural damage produces spatially coherent sensor response anomalies — multiple sensors shift in a pattern consistent with the modified structural mode shapes. A sensor fault, by contrast, produces a spatially incoherent anomaly: one sensor reads abnormally while its physical neighbours read normally, in violation of structural wave propagation. A model that internalises the expected spatial correlation structure of the sensor array — using the known sensor layout as a structural prior — can distinguish these two physically distinct events from the measurement pattern alone, without any external fault indicator. This spatial coherence check is the mechanistic basis of fault-aware structural damage identification.

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

---

## 21. Sensor self-attention before the slot decoder propagates fault artifacts into clean sensor representations

> **[Historical — status: not reproducible from current checkpoints.]** The numbers in the Δ table below are from a prior Qatar experiment using the old nf levels `{1, 3, 5, 10, 15}` (incompatible with the current `{0, 0.2·S, 0.5·S, 0.8·S}` protocol) and the `SensorSpatialLayer` code path has since been removed — checkpoints deleted. The *mechanism* argued here (all-to-all self-attention over sensor features contaminates fault-isolation cues, motivating the logit-space structural-bias alternative, Insight #22) is design-load-bearing and still current; the per-fault-type numbers should be treated as historical evidence rather than reproducible benchmarks.

**Domain rationale**
Spatial coherence under forced vibration means a faulty sensor is an outlier *relative to its physical neighbours* — the discriminating signal lives in that local contrast. An all-to-all cross-sensor self-attention layer placed before the slot decoder blurs this contrast in the wrong direction: the faulty sensor's anomalous feature representation is distributed into its neighbours' representations via attention-weighted sums, while its own signature is diluted by the normal features of those neighbours. The network mixes the signals of all sensors together before the damage head sees them, which erases the very spatial incoherence pattern it is designed to exploit.

**ML implication**
Training with sensor fault augmentation and a `SensorSpatialLayer` (multi-head self-attention over all S sensor features) before the `MidnC` slot decoder consistently degrades damage F1 relative to fault augmentation alone. The degradation is asymmetric across fault types.

| Fault | C+FH (aug only) | C+SF+FH (aug + spatial layer) | Δ |
|-------|-----------------|-------------------------------|---|
| gain (mean n=1…15) | 0.990 | 0.797 | −0.193 |
| noise | 0.928 | 0.498 | −0.430 |
| partial | 0.983 | 0.850 | −0.133 |
| bias | 0.861 | 0.902 | +0.041 *(exception — see below)* |
| hard | 0.987 | 0.960 | −0.027 |
| stuck | 0.985 | 0.957 | −0.028 |
| **clean** | 0.992 | 0.984 | −0.008 |

Hard and stuck faults (zeroed sensor) are nearly immune: a zeroed sensor contributes nothing to attention sums, so its feature vector cannot contaminate neighbours. Soft faults (gain, noise, partial, gain_bias) — present-but-wrong sensors — spread their anomalous features into neighbours and degrade the most. Bias faults are a partial exception: bias shifts are spatially coherent (a stuck-offset pattern), so mixing may marginally help the model classify them. Clean F1 drops too because self-attention blurs clean representations even with no faults present. The lesson: placing all-to-all feature mixing before a detector that needs clean, per-sensor representations is counterproductive for fault isolation; the spatial reasoning layer must not modify sensor embeddings before the damage head runs.

**Where in paper**
Discussion (ablation of spatial reasoning design choices; mechanistic explanation of the negative result; motivates the logit-bias alternative, Insight 22)

**Quote-ready**
> Training with sensor fault augmentation and a spatial self-attention layer before the
> slot decoder consistently degraded damage F1 relative to fault augmentation alone, with
> the largest drops on soft fault types (gain: −0.19, noise: −0.43, partial: −0.13) and
> negligible effect on hard and stuck faults whose contribution to attention sums is near
> zero. This asymmetry identifies the mechanism: all-to-all self-attention distributes the
> faulty sensor's anomalous feature representation into its physical neighbours, degrading
> the spatial coherence signal that motivates the spatial layer in the first place. Spatial
> reasoning that modifies sensor feature embeddings before the damage head is
> counterproductive for fault isolation; the structural prior must act on attention routing,
> not on sensor representations.

---

## 22. Dynamic location-conditional structural affinity bias guides slot attention routing in logit space without modifying sensor features

**Domain rationale**
Sensors physically adjacent to structural joint l carry the strongest diagnostic signal for damage at l: they are closer to the modified stiffness, and structural coupling distributes forced-vibration energy through local load paths preferentially along 4-connected elements of the sensor grid. This adjacency relationship is determined by the installation plan and is fixed for the structure's lifetime — it is a hard physical prior. Rather than teaching the model what sensors "know about each other" (feature mixing, which contaminates), we can directly bias *where each slot attends* based on its current location estimate, injecting the prior into attention routing instead.

**ML implication**
A learnable (L+1)×S matrix **R**, physically initialised from 4-connected grid adjacency (R[l,i]=1 if sensor i is adjacent to joint l, else 0; last row for no-object = 0), is incorporated as an additive bias to slot cross-attention logits at decoder layers ≥1:

```
bias[b, slot, sensor] = softmax(loc_logits[b, slot]) @ R   →   shape (B, K, S)
```

This is dynamic — it changes per sample and per decoder layer as each slot refines its location prediction — and it does not touch sensor feature embeddings. Each slot predicted at location l preferentially attends to sensors adjacent to l; a faulty sensor far from the predicted location gets lower attention weight. Layer 0 always runs without bias: slot queries have not yet seen sensor data, so `loc_head(queries)` is sample-identical and provides no meaningful location estimate to weight R's rows. R is a learnable `nn.Parameter`, allowing training to refine the binary adjacency prior and discover structural coupling patterns (e.g. longer-range coupling along load-path columns or rows) that simple 4-connectivity does not capture.

Because R acts only on attention logits, sensor embeddings remain unmodified — no contamination path exists.

*Numbers from prior Qatar experiment (old nf levels {1,3,5,10,15}, checkpoints deleted). Analysis and design conclusions remain valid; updated numbers pending consistent cross-dataset re-eval. See `CLAUDE.md` for current 7-story and LUMO fault results.*

| Fault | C (no aug) | C+FH | C+FH+SB |
|-------|-----------|------|---------|
| gain (mean) | 0.920 | 0.990 | **0.993** |
| bias | 0.737 | 0.861 | **0.981** |
| gain_bias | 0.774 | 0.929 | **0.990** |
| noise | 0.360 | 0.928 | **0.936** |
| partial | 0.988 | 0.983 | **0.991** |
| hard | 0.988 | 0.987 | **0.988** |
| **clean** | 0.992 | 0.992 | **0.994** |

C+FH+SB achieves the best damage F1 across all seven fault types and all fault counts, while simultaneously improving clean-condition F1 — confirming that the structural routing prior is complementary to, not in tension with, baseline accuracy. The design principle established: **structural spatial priors belong in attention routing (logit space), not in feature mixing.**

**Where in paper**
Methodology (design of structural affinity bias; motivation for logit-space vs. feature-space injection), Results (fault robustness comparison table), Discussion (mechanistic contrast with SensorSpatialLayer — Insight 21; physical grounding of R initialisation from grid adjacency)

**Quote-ready**
> We propose a dynamic location-conditional structural affinity bias: a learnable (L+1)×S
> matrix R, physically initialised from the 4-connected adjacency of the sensor grid,
> incorporated as an additive logit bias to slot cross-attention at decoder layers beyond
> the first. At each layer, the bias for slot k is computed as softmax(loc_logits_k) @ R —
> a per-sample, per-slot weighting that directs each slot to attend to sensors near its
> current location estimate. Because R acts on attention logits rather than on sensor
> feature embeddings, per-sensor representations remain unmodified and no fault
> contamination path exists. The model trained with this bias (C+FH+SB) achieves the best
> damage F1 across all seven synthetic fault types and all fault counts evaluated, while
> simultaneously improving clean-condition F1 (0.994 vs 0.992 for the unaugmented C
> baseline), confirming that structural routing priors are complementary to baseline
> accuracy rather than in tension with it.

---

## 23. DETR slot machinery is costly under ambient excitation when K≤1 — per-location regression is more robust to distribution shift

**Domain rationale**
In long-term field SHM the structure is excited by ambient forces (wind, traffic) rather than controlled shaker input. The excitation is non-stationary: amplitude varies by orders of magnitude across recordings depending on weather. When the structure is healthy, the monitoring system must confidently output "no damage" despite large input variability. When K≤1 (single or no damage), the diagnostic question reduces to "is anything wrong at all, and if so where?" — there is no set-assignment problem.

**ML implication**
The C-head's DETR ∅-class slot must learn a decision boundary between "healthy embedding region" and "damage embedding region". Under stationary forced excitation (Qatar, 7-story) this boundary is sharp — the embedding distribution for healthy windows is tight. Under ambient excitation (LUMO) the healthy embedding distribution spreads out due to wind variability, blurring the ∅ boundary. The C-head's recall drops (0.929 on LUMO vs 0.992 on Qatar) because some low-energy damaged windows fall on the wrong side of the now-fuzzy ∅ decision.

DR-head has no global gating decision — each location regresses severity independently and is thresholded individually. There is no ∅-class boundary to blur. This makes DR inherently more robust to input distribution shift when K≤1: on LUMO DR achieves F1=0.972 (−0.019 from Qatar) while C achieves F1=0.946 (−0.046 from Qatar).

The ranking inversion (C > DR on stationary lab data, DR > C on ambient field data) reveals that C's slot mechanism is an *advantage* only when the set-prediction problem is hard (K>1, variable cardinality). When K≤1 everywhere, the slot machinery is overhead — it forces the model to solve a harder gating problem that DR avoids entirely. This is visible in the cross-benchmark comparison:

| Model | 7-story (sim) | Qatar (lab) | LUMO (field) |
|-------|--------------|-------------|-------------|
| C     | 0.994–0.999  | 0.992       | 0.946       |
| DR    | 0.924–0.961  | 0.991       | 0.972       |

**Where in paper**
Discussion (architecture comparison across data regimes), Results (sim → lab → field degradation analysis)

**Quote-ready**
> The DETR-style slot mechanism (C-head) outperforms per-location regression (DR-head)
> under stationary forced excitation where multi-damage set prediction is required, but
> the relationship inverts under ambient field conditions with K≤1. On the LUMO field
> benchmark, DR achieves F1=0.972 vs C's 0.946 — a reversal of the Qatar lab ranking
> (C=0.992, DR=0.991). The mechanism is the ∅-class decision boundary: under stationary
> excitation, the healthy embedding distribution is compact and the ∅ boundary is sharp;
> under ambient wind variability, the distribution spreads and the boundary blurs,
> reducing C's recall to 0.929. DR avoids this failure mode entirely because each
> location is thresholded independently with no global gating decision. This suggests
> that slot-based architectures are best suited to multi-damage regimes (K>1) where
> set prediction is genuinely required, while per-location regression is preferable
> for single-damage field monitoring under non-stationary excitation.

---

## 24. Global RMS normalization is physically grounded transmissibility-like preprocessing — fundamentally different from time-series forecasting normalization

**Domain rationale**
In structural dynamics, dividing each sensor's RMS by a shared reference RMS approximates the transmissibility function — the energy transfer ratio between two points in the structure. For a linear structure under broadband excitation, this ratio encodes the structural transfer characteristics: stiffness distribution, damping, and connectivity. Damage (stiffness loss, cracking, connection degradation) alters local stiffness and damping, changing how vibration energy distributes spatially. The normalised feature vector therefore captures the structural state while being invariant to excitation amplitude.

We use the *global-RMS* variant (denominator = RMS across all sensors and time steps jointly, `ref_channel=None` in `lib/preprocessing.normalize_rms`) rather than a single fixed reference sensor. The rationale is below under "Global vs single-sensor reference." See also Insight #30 for the empirically-confirmed fault-robustness advantage of mean-RMS over robust (median) alternatives.

This is fundamentally different from normalization techniques in the time-series forecasting literature (RevIN, DAIN, Dish-TS, SAN, FAN), which address temporal distribution shift by normalizing each channel independently using its own mean/variance statistics. Those methods are designed for a different problem: non-stationarity over long horizons where train and test distributions diverge temporally.

**Key distinctions from forecasting normalization**

1. *Excitation invariance.* If the input force doubles, all sensor RMS values roughly double, and the inter-sensor ratios stay constant. RevIN and its descendants would also remove amplitude scaling, but in a less physically grounded way — they normalize each sensor's temporal waveform independently, stripping out the cross-sensor amplitude relationships that encode damage location and severity.

2. *Cross-sensor structure preservation.* The damage signature lives in the relative responses between sensors. Reference-sensor RMS normalization preserves this spatial pattern explicitly. Methods like RevIN, DAIN, and SAN normalize each channel independently using its own statistics, destroying the inter-sensor amplitude ratios.

3. *Stationarity is not the primary concern.* Forecasting methods are designed to handle temporal non-stationarity causing train-test distribution mismatch over time. In SHM under repeated or ambient excitation, the signal within a measurement window is approximately stationary. The "distribution shift" of interest is between undamaged and damaged states — exactly the signal to preserve, not normalize away.

**Empirical evidence**
Qatar (with reference-sensor RMS normalization) shows near-zero SDI F1 degradation under preserving sensor faults (gain, bias, partial) even at 80% sensor fault ratio (F1=0.992 vs 0.995 clean). The same model architecture on 7-story and LUMO (without normalization) degrades measurably under identical fault injection. An extreme-gain stress test on Qatar (scale=0.001 to 100×) confirms the model is truly amplitude-invariant within realistic ranges: gain scales 0.001–2.0× cause <1.5% F1 drop, while only extreme amplification (10×+) overwhelms BatchNorm statistics and causes collapse.

**ML implication**
Global RMS normalization should be applied as a standard preprocessing step across all SHM datasets. It forces the model to learn frequency/phase/mode-shape features rather than absolute amplitudes, providing inherent robustness to gain-type sensor faults without any model architecture change. This is complementary to (not redundant with) fault-augmented training: normalization handles amplitude-preserving faults at the input level, while fault augmentation trains the model to handle destructive faults (hard, stuck, noise) that normalization cannot address.

We use global (all-sensor) RMS rather than single-sensor reference for fault robustness: a single reference sensor is a single point of failure — if it is faulted (zeroed, stuck, biased), dividing by its corrupted RMS contaminates every sensor in the window. Global RMS degrades gracefully under sensor faults because any one faulted sensor contributes only ~1/S to the denominator. While single-sensor referencing gives a purer transmissibility estimate, the fault robustness benefit of global RMS outweighs the marginal physics accuracy loss for our experimental setting.

**Where in paper**
Methodology (data preprocessing, justification for normalization choice), Discussion (comparison with forecasting normalization literature, fault robustness analysis)

**Quote-ready**
> Per-window global RMS normalization divides all sensor channels by the root-mean-square
> amplitude computed across all sensors jointly. This preserves cross-sensor amplitude
> ratios — the spatial energy distribution pattern that encodes structural damage location
> and severity — while providing excitation-amplitude invariance. Unlike single-sensor
> referencing (which approximates transmissibility more closely but creates a single point
> of failure under sensor faults), global RMS degrades gracefully: faulting k of S sensors
> perturbs the denominator by only ~k/S. Unlike instance normalization methods from the
> time-series forecasting literature (RevIN, DAIN, Dish-TS) which normalize each channel
> independently, global RMS normalization preserves the inter-sensor amplitude structure
> that is the primary damage signature.

---

## 25. Future direction: fault-aware learnable normalization combining structural physics with adaptive gating

**Domain rationale**
Reference-sensor RMS normalization is effective but static — it applies a fixed, physically motivated transformation. Under long-term field monitoring, environmental and operational variability (temperature affecting stiffness, varying traffic loads, seasonal wind patterns) creates distribution shift that a fixed normalization cannot fully absorb. The challenge maps onto the "intra-space shift vs inter-space shift" distinction from Dish-TS: environmental variation is nuisance shift to remove, while damage is signal shift to preserve.

**Research directions**

1. *Learnable gating adapted from DAIN.* DAIN's learnable normalization layer could be adapted to operate across the sensor array rather than per-channel. Instead of normalizing each sensor independently, a cross-sensor gating mechanism could learn to highlight damage-sensitive sensors and suppress uninformative ones — complementing the fixed RMS-referencing rather than replacing it.

2. *Frequency-domain normalization.* Inspired by FAN's frequency-aware approach: a normalization scheme operating on the power spectral density of sensor signals could separate structural response characteristics (natural frequencies, mode shapes) from excitation characteristics (broadband energy level). PSD ratios between sensors are a richer version of the scalar RMS ratio, capturing frequency-dependent transmissibility rather than broadband energy transfer.

3. *Transmissibility-based features.* The natural progression of RMS-referencing within the structural dynamics literature — power spectral density ratios, coherence functions, and mode shape curvature features. These all share the core principle (ratio-based features that cancel excitation effects) but capture richer frequency-dependent and spatial information.

**Where in paper**
Future work, Discussion (connecting SHM preprocessing to broader normalization literature)

---

## 26. Anti-alias FIR decimation is mandatory before downsampling vibration data — stride decimation aliases high-frequency content into the structural band

**Domain rationale**
Structural damage signatures live in the low-frequency modal response band: mode shapes, natural frequency shifts, and damping changes typically occur below 50–100 Hz for civil structures. Raw accelerometer sampling rates (1000–1652 Hz) far exceed what is needed and waste model capacity on high-frequency content (electrical noise, sensor resonance, aliased machinery harmonics) that carries no damage information.

Downsampling reduces the time dimension and focuses the model on the relevant frequency band, but *how* the downsampling is performed matters critically. Two approaches:

1. **Stride decimation** (`x[::k]`): simply takes every k-th sample. Any frequency content above the new Nyquist frequency (Fs/2k) is *not removed* — it folds back (aliases) into the lower frequency band, appearing as phantom spectral content that the model cannot distinguish from real structural response. This is the classic aliasing problem from signal processing.

2. **FIR anti-alias decimation** (`scipy.signal.decimate(..., ftype='fir', zero_phase=True)`): first applies a low-pass FIR filter with cutoff at the new Nyquist frequency to remove all content above Fs/2k, then downsamples. The filter is applied zero-phase (forward + backward pass) to avoid group delay distortion that would misalign channels temporally.

**Why this matters for SHM specifically**

- *Aliased content corrupts damage features.* A vibration sensor at 1652 Hz sampling rate contains energy up to 826 Hz. If stride-decimated 4× to 413 Hz, content in the 207–826 Hz band folds back into 0–207 Hz — directly overlapping with structural modes. The model sees aliased artefacts as if they were real structural response, making damage features less discriminative.

- *Aliasing is damage-state-dependent.* Different damage states change the broadband frequency response, so the aliased content differs between healthy and damaged conditions. This creates a confound: the model may learn to discriminate damage states partly through aliased artefacts rather than true modal features, resulting in a classifier that appears to work on in-distribution data but fails under distribution shift (different excitation, environmental conditions, sensor placement).

- *Consistent preprocessing across datasets.* For fair comparison across the sim→lab→field trio (7-story 1000 Hz, Qatar 1024 Hz, LUMO 1652 Hz), all must use the same decimation strategy. Inconsistent preprocessing (one dataset with anti-alias filtering, another without) means performance differences could reflect preprocessing artefacts rather than genuine model or dataset characteristics.

**Implementation**

Qatar and LUMO use `scipy.signal.decimate(x, factor, ftype='fir', zero_phase=True)` prior to windowing. 7-story is a special case: its raw window is 1000 samples at 1000 Hz, but the final 500 samples are zero-padded (the simulated impact response fully decays by t=0.5 s), so the correct operation is to *truncate* to the first 500 non-padded samples — no decimation is needed because no information is discarded and Nyquist stays at 500 Hz.

| Dataset | Raw Fs (Hz) | Preprocessing | Effective Fs (Hz) | Nyquist (Hz) | T (samples) |
|---------|-------------|---------------|--------------------|--------------|-------------|
| 7-story | 1000 | truncate first 500 samples (tail is zero-padded) | 1000 | 500 | 500 |
| Qatar | 1024 | FIR decimate 4× | 256 | 128 | 512 |
| LUMO | 1652 | FIR decimate 4× | 413 | 207 | 512 |

All effective Nyquist frequencies are well above the structural mode band (<50 Hz for these structures), ensuring no loss of damage-relevant information.

**Previous state and what changed**

- *Qatar*: already used `scipy.signal.decimate` with FIR filter — no change.
- *LUMO*: previously used stride decimation (`chunk[::4]`) — every 4th sample with no filtering. Content in the 207–826 Hz band was aliased into 0–207 Hz. Now: FIR decimation applied to the full recording before windowing, eliminating aliasing.
- *7-story*: a short-lived intermediate state applied FIR 2× decimation over the full 1000 samples (500→250 Hz Nyquist). This was reverted in the `fix 7-story preprocessing` commit on discovering that the second half of each raw window is zero-padded by the simulation — decimating *padded* data introduces FIR ringing and transient filter startup/endup artefacts at window boundaries, contaminating the useful signal in the first 500 samples. The current pipeline truncates instead, which is information-preserving: raw data is bandlimited to 500 Hz by the simulator, so operating at 1000 Hz sampling over 500 real samples is the correct representation.

**ML implication**
Anti-alias filtering is a form of input denoising that removes high-frequency content the model cannot usefully learn from. Without it, the model must either (a) learn to ignore aliased content, wasting capacity, or (b) overfit to aliased artefacts that do not generalise. The FIR filter is a hard prior encoding the known physics: structural damage information lives below the Nyquist of the decimated signal. This is analogous to how convolutional stride layers in vision models implicitly rely on preceding convolutions to anti-alias before spatial downsampling (Zhang 2019, "Making Convolutional Networks Shift-Invariant Again").

**Where in paper**
Methodology (data preprocessing — describe the decimation pipeline and justify anti-alias filtering). If the paper includes an ablation study, a comparison of stride vs FIR decimation on LUMO (where the aliasing was worst due to 4× decimation from 1652 Hz) would directly demonstrate the impact.

**Quote-ready**
> All accelerometer signals are decimated to the target sampling rate using a zero-phase
> FIR anti-alias filter (scipy.signal.decimate) prior to windowing. This removes spectral
> content above the new Nyquist frequency that would otherwise alias into the structural
> response band (typically <50 Hz for civil structures), where it is indistinguishable
> from genuine modal features. Stride decimation without filtering was previously used
> for the LUMO dataset (4× decimation from 1652 Hz), which allowed content in the
> 207–826 Hz band to fold back into the 0–207 Hz analysis band. Consistent FIR
> decimation across all datasets ensures that performance comparisons reflect genuine
> model and dataset characteristics rather than preprocessing artefacts.

---

## 27. Stacking "obvious-looking" fault-robustness priors onto C+fh+sb regresses — each intervention alone is already neutral-to-negative, confirmed at both 50 and 200 epochs

**Domain rationale**
The C+fh+sb architecture already composes three ideas: DETR slot prediction, fault-head auxiliary BCE, and structural affinity bias R in cross-attention logits. The natural next question is: can we strengthen each piece to close the remaining gap vs v1 at K=1-under-fault? Three interventions looked individually sensible — (A) use the fault head's own `p_fault` as a cross-attention logit penalty `−λ·p_fault` (detached); (B) freeze R to its physics initialisation and apply it at decoder layer 0 as well; (C) DETR-style aux loss applied at every decoder layer output. A 5-run 50-epoch ablation on 7-story-fault (R0 base + R1-A, R2-B, R3-C, R4-ABC) returned unanimous negative results:

| Run    | K=1 clean | K=1 nf=32 | K=1 nf=52 | K=2 clean | K=2 nf=52 | fault_F1@32 |
|--------|-----------|-----------|-----------|-----------|-----------|-------------|
| base50 | 0.830     | 0.804     | 0.696     | 0.753     | 0.636     | 0.975       |
| A50    | 0.838     | 0.743     | 0.575     | 0.738     | 0.561     | 0.975       |
| B50    | 0.809     | 0.767     | 0.628     | 0.710     | 0.571     | 0.975       |
| C50    | 0.840     | 0.799     | 0.676     | 0.740     | 0.611     | 0.974       |
| ABC50  | 0.763     | 0.695     | 0.546     | 0.705     | 0.528     | 0.974       |

None of A/B/C individually beats R0 at nf=32 or nf=52. Their combination (ABC) does not cancel to neutral — it stacks negatively, regressing K=1-nf=52 from 0.696 → 0.546 (−0.150).

**200-epoch confirmation.** The same 5-run protocol at 200 epochs (`states/7story-fault-ablate200/`, `saved_results/7story-fault-ablate200/ablation_summary.csv`) produces the same ranking: base200 wins all K=1 conditions; ABC200 is worst; A200 is the single worst ablation at nf=52; C200 (aux-loss) is closest to base but still below. K=1 F1 across fault types (mean):

| Run     | K=1 clean | K=1 nf=13 | K=1 nf=32 | K=1 nf=52 | K=2 nf=52 | fault_F1@32 |
|---------|-----------|-----------|-----------|-----------|-----------|-------------|
| base200 | 0.921     | 0.917     | 0.907     | 0.835     | 0.784     | 0.978       |
| A200    | 0.901     | 0.885     | 0.858     | 0.752     | 0.723     | 0.978       |
| B200    | 0.906     | 0.904     | 0.895     | 0.828     | 0.783     | 0.978       |
| C200    | 0.919     | 0.914     | 0.901     | 0.816     | 0.780     | 0.977       |
| ABC200  | 0.893     | 0.873     | 0.838     | 0.717     | 0.724     | 0.977       |

The gap magnitudes shrink (more epochs mask some of the regression), but the ordering is identical and the combined variant (ABC) remains the worst, confirming this is not an under-training artifact. Fault-head F1 stays ≥0.977 in all runs — consistent with the 50-epoch finding that the detach protects fault-head supervision.

**ML implication**
Each intervention's failure mechanism is a different failure of the same assumption — that "the model can only benefit from more signal about sensors":

- **(A) fault-gate**: at λ=3 a confidently-faulted sensor gets its attention logit driven down by ~3, reducing its post-softmax weight to ~5% of baseline. This is exactly what a hand-engineered fault-robustness scheme would do. But under training-time fault augmentation (p_hard/p_soft/p_struct_mask=0.3) the decoder already learns to down-weight faulted sensors implicitly — the hard gate double-penalises them and also suppresses clean sensors the slot should attend to (because p_fault is detached and cannot be recalibrated by decoder gradients). Net effect: lower K=1 F1 across all fault ratios.
- **(B) freeze R + layer-0**: treating R as a hard physical prior (freeze) and applying it from layer 0 matches the "honest physics-informed" framing of Insight 22. But R as implemented is 4-connected grid adjacency — an approximation of coupling, not ground truth. The learnable variant (Insight 22) converges to a bias pattern different from binary adjacency. Freezing locks in the approximation; applying at layer 0 propagates its errors through every subsequent layer. The layer-0 bias computes `softmax(loc_head(slot_queries)) @ R`, but slot queries at layer 0 have not yet seen sensor data — the location prior is sample-identical and biased toward the class-frequency distribution of damage locations in the training set, which is not a useful routing prior.
- **(C) aux loss**: shared heads applied at every decoder layer's output. DETR's original aux loss works because DETR has 6 decoder layers and the aux signal provides a learning curriculum from coarse (early layers) to fine (late layers) predictions. At decoder depth = 2 (our MidnC configuration), there is essentially one intermediate layer and one final layer — the aux loss is applied to a slot representation that has received only half the refinement, effectively asking shared heads to commit to predictions before they are ready. At 50 epochs and this depth the effect is ~neutral (−0.001 mean K=1 F1), but adds training-time cost with no observable benefit.

The combined regression (ABC50 K=1 F1 across nf=0,13,32,52 averaged = 0.687 vs base50 0.789; Δ=−0.102) is an amplification, not a sum: (A) distorts attention, (B) distorts the routing prior, (C) over-commits intermediate representations to those distorted signals. All three share a common pattern: **they replace learning signal with hand-engineered regularisation.** When training-time augmentation already provides the supervision the regularisation was meant to substitute for, the regularisation becomes a constraint rather than a scaffold.

Fault-head supervision itself is unaffected — all 5 runs keep fault_F1@nf=32 ≥ 0.974. The detach of `p_fault` into the A-gate successfully isolates fault-head gradients from decoder supervision, but preserving the fault head does not rescue the downstream decoder.

**Where in paper**
Ablation / negative results section. The table above belongs in the supplementary. The framing in the main text is: "We evaluated three extensions that compose the existing three innovations more tightly (fault-gate cross-attention, frozen physics-initialised R with layer-0 application, deep decoder supervision). All three regressed on K=1 fault-robustness at 50 epochs. We interpret this as evidence that under strong fault augmentation (p=0.3 each of hard/soft/struct-mask), the decoder implicitly learns the sensor-down-weighting behaviour that explicit regularisation was meant to provide, and that hand-engineered priors compete with rather than complement the learned regularisation."

**Quote-ready**
> To close the remaining gap between the slot-based C+fh+sb model and the v1 head at K=1
> under sensor faults, we evaluated three extensions, each reinforcing one of the model's
> three existing priors: (A) a fault-probability penalty on sensor cross-attention logits;
> (B) a frozen, physics-initialised structural affinity bias applied from the first decoder
> layer; (C) DETR-style deep decoder supervision via per-layer Hungarian-matched auxiliary
> losses. In a 5-run, 50-epoch ablation on 7-story-fault under matched augmentation
> (p_hard=p_soft=p_struct_mask=0.3), all three extensions regressed single-damage F1 at
> non-zero fault ratios, and their composition regressed K=1 F1 at 52 faulted sensors
> from 0.696 to 0.546. Fault-head F1 at 32 faulted sensors was preserved (≥0.974 in all
> runs), confirming that the gradient detach in (A) did not contaminate fault-head
> supervision. We interpret the negative result as evidence that, under strong fault
> augmentation, the decoder implicitly learns the sensor-down-weighting and spatial-
> routing behaviour the explicit regularisation was meant to scaffold — and that
> hand-engineered priors compete with the learned regularisation rather than complementing
> it.

---

## 28. Fault-head and structural-bias are synergistic, not additive — physics priors in different spaces (feature state × attention routing) compose; in the same space they compete

**Domain rationale**
The C-head architecture exposes two independent physics priors: a per-sensor fault-detection head (fh) supervised by per-sensor BCE against injected fault ground truth, and a structural-affinity bias (sb = R_bias) that biases slot cross-attention logits toward sensors near each slot's currently-predicted location. Each prior is individually well-motivated and each could plausibly help fault-robust damage localisation. A clean isolation — training C, C+fh, C+sb, and C+fh+sb on 7-story-fault with identical fault augmentation (p_hard=p_soft=p_struct_mask=0.3) at 200 epochs — shows a striking interaction effect:

| Model    | K=1 clean | K=1 nf=32 | K=1 nf=52 | K=2 clean | K=2 nf=52 | Δ vs C (K=1 nf=52) |
|----------|-----------|-----------|-----------|-----------|-----------|--------------------|
| C        | 0.927     | 0.911     | 0.849     | 0.864     | 0.806     | —                  |
| C+fh     | 0.920     | 0.909     | 0.855     | 0.859     | 0.801     | +0.006             |
| C+sb     | 0.916     | 0.898     | 0.828     | 0.854     | 0.782     | −0.021             |
| C+fh+sb  | **0.939** | **0.926** | **0.872** | **0.873** | **0.817** | **+0.023**         |

Neither component alone is helpful — fh is essentially neutral (+0.006), sb is slightly harmful (−0.021). Their combination gains +0.023 at the hardest operating point — an effect roughly three times the sum of the individual effects, and positive where neither alone is. This is not an additive composition; it is a synergistic interaction.

**ML implication**
The two priors are expressed in fundamentally different spaces of the decoder, and each one supplies what the other is missing:

- **sb** is a geometric prior in **attention-logit space**: it answers *where should slot k look?* by biasing cross-attention toward sensors spatially adjacent to the predicted location. But sb is state-blind — a faulted sensor adjacent to the true damage is routed to identically to a healthy one.
- **fh** is a state prior in **feature space**: its BCE auxiliary loss pushes the encoder to produce sensor embeddings that separably encode fault state. But on its own, the decoder has no structured mechanism to exploit that separability for localisation — the extra fault-aware direction in embedding space does not translate into better slot routing.

When composed, sb routes the slot into the correct spatial neighbourhood and fh ensures the sensors *in* that neighbourhood carry discriminable health information. Soft cross-attention then implicitly computes `attention_weight ∝ spatial_proximity × health` — not a hard gate on either, but a multiplicative weighting in which both factors must be non-zero for the sensor to contribute meaningfully. Neither factor alone suffices: sb without fh routes to the right place but accepts broken signal; fh without sb produces the right features but has no routing structure to amplify healthy ones over faulted ones in the local neighbourhood.

**The failure modes are concrete and complementary:**

- **C+sb alone fails at high fault rates** (−0.021 K=1 nf=52). The spatial prior commits with high confidence to faulted sensors adjacent to the true damage; the Hungarian matcher pays the full mistake cost. On clean data the commitment is harmless; as fault rate grows the cost grows with it.
- **C+fh alone is flat-to-slightly-negative on clean data** (−0.007 K=1 clean). The extra BCE loss reallocates encoder capacity toward fault detection, but the decoder cannot structurally exploit the resulting fault-aware features without a spatial routing mechanism to focus attention locally. The auxiliary task is "free" signal with no downstream architectural hook.

**Composition principle established.** This result generalises beyond the specific fh/sb pair: **for physics-informed priors in learned models, where the prior is injected (which space, which layer, additive vs multiplicative) matters as much as what the prior encodes.** Priors injected in orthogonal spaces (logit routing × feature state × input weighting) can be complementary because each supplies a distinct ingredient the model composes through its existing computation. Priors injected in the same space compete for the same routing capacity.

This frames and explains three earlier observations in this codebase:

- **Insight 21** (SensorSpatialLayer abandoned): putting a geometric prior in *feature* space contaminated the very features fh would later need to keep clean. Feature space is where sensor state lives; overwriting it with geometric mixing destroys the signal fh extracts.
- **Insight 22** (sb as logit-space routing prior): the correct space for a geometric prior is attention logits, not features — leaving the feature space untouched so the state prior (fh) has an unobstructed signal to learn.
- **Insight 27** (50-epoch stacking ablation was negative): the fault-gate variant (−λ·p_fault on cross-attention logits) put fh's signal back into logit space where sb already lives — competing for the same routing capacity, exactly the over-composition failure the synergy principle warns against.

The synergy of fh+sb is thus not an accidental empirical fact but the predictable outcome of a deeper design rule: **compose priors across spaces, not within them.**

**Where in paper**
Methodology (justification for keeping fh in feature-supervision space and sb in logit-routing space — explicitly frame as orthogonal injection). Ablation / Results (the 4-row table above). Discussion (the orthogonal-space composition principle as a design heuristic that generalises to other SHM tasks: e.g. temperature-compensation priors in feature space + sensor-layout priors in logit space; input-amplitude priors via preprocessing × modal priors via output-head weighting).

**Quote-ready**
> The C+fh+sb configuration achieves the best K=1 and K=2 F1 across all fault ratios on
> 7-story-fault, improving on the plain-C baseline by +0.012 at K=1 clean and +0.023 at
> K=1 with 52 of 65 sensors faulted. Individually, neither extension is beneficial: C+fh
> is approximately neutral (−0.007 K=1 clean) and C+sb slightly regresses under heavy
> faults (−0.021 K=1 nf=52). The synergy reflects the orthogonality of the two priors'
> injection sites in the decoder: the fault head operates in feature space, supervising
> the encoder to produce sensor embeddings that separably encode fault state; the
> structural-affinity bias operates in attention-logit space, routing each damage slot
> toward sensors spatially adjacent to its predicted location. When composed, the
> slot attends to a spatially-correct neighbourhood in which faulted sensors are
> simultaneously discriminable via their embeddings, so soft cross-attention implicitly
> computes a multiplicative weighting of spatial proximity and sensor health. Neither
> factor alone suffices for fault-robust localisation; their composition succeeds
> because they occupy orthogonal spaces of the decoder and do not compete for the
> same routing capacity. This identifies a reusable design principle for physics-
> informed architectures: compose priors across spaces, not within them.

---

## 29. HP sweeps at short epoch budgets select for convergence speed, not terminal optimum — re-rank at full budget before declaring a winner

**Domain rationale**
When validating c-fh-sb HP choices on 7-story-fault, a 10-run, 50-epoch sweep over num-slots, no-obj-weight, augmentation strength, and neck dropout showed a clear winner: no-obj-weight = 0.2 ("n20") beat the default 0.1 baseline by +0.122 K=1 clean F1 (0.889 vs 0.767). Every K=1 and K=2 fault condition ranked n20 first, often by ≥0.10. Retraining n20 at the standard 200-epoch budget **reverses the result**: n20 underperforms the default on every SDI metric (K=1 clean −0.005, K=2 nf=52 −0.027), with the only positive delta being a negligible +0.002 in fault-head F1.

**ML implication**
A stronger ∅-class weight (n20) biases the DETR slot head toward fewer activations from epoch 0, accelerating convergence into the K=1-dominant regime where most samples have one damage and most slots should emit ∅. At 50 epochs the default (n=0.1) has not yet learned to suppress spurious slots, so n20's head start dominates. Given full training time, the default catches up and slightly surpasses n20 because weaker ∅ pressure leaves more representational capacity in the active slots — which matters more at K=2, where suppressing second-slot activations is costly. The 50-epoch ranking tracks *learning rate along the loss surface*, not *position of the final minimum*.

**Methodological lesson**
Short-budget sweeps are legitimate for pruning — they cheaply reject configurations that cannot even reach convergence. But they must not be used to *select among* candidates near the frontier. A winner at a shortened budget is a hypothesis; terminal ranking requires training each candidate (or at minimum the top 2-3) at the full budget. This is a general pattern for HP selection on non-convex losses — the HP that reaches any given loss value fastest is not the same as the HP that reaches the *lowest* loss value, because regularisation-like HPs that accelerate early-phase optimisation often over-constrain the final-phase fit.

**Where in paper**
Supplementary / ablation appendix. Useful if reviewers ask "did you sweep HPs?" — the short answer is "yes, the defaults were confirmed at full training budget to be near-optimal for this head."

**Quote-ready**
> A ten-configuration, 50-epoch hyperparameter sweep over the DETR no-object weight,
> slot count, augmentation intensity, and neck dropout identified a clear 50-epoch
> winner — no-object weight doubled from 0.1 to 0.2 — that dominated every
> single- and double-damage fault condition. Re-training this configuration at the
> 200-epoch reference budget reversed the ranking: the default recovered and
> slightly surpassed the sweep-winner on every SDI metric. The 50-epoch advantage
> reflects faster convergence toward the K=1-dominant regime, not a lower terminal
> loss. Short-budget sweeps can prune pathological settings, but terminal selection
> requires full-budget confirmation; the published defaults are therefore retained.

**Confirmation at n=2 (Round 2, April 2026)**
A second-round, 100-epoch sweep picked a different winner on a different HP axis: `num_slots=3` ("s3") beat the 5-slot default by +0.037 K=1 clean F1 (0.906 vs 0.869) at 100 epochs. Retraining s3 at 200 epochs reproduces the same reversal pattern as n20, *more strongly*: K=1 clean Δ=−0.013, K=1 nf=52 Δ=**−0.029** (17× the n20 gap at nf=52), K=2 nf=52 Δ=−0.032. Mechanism is analogous but operates through architectural capacity rather than loss weighting — fewer slots converge faster to the K≤2-dominant regime but hit a lower ceiling once the 5-slot default catches up on suppression. s3 loses harder than n20 because reducing `num_slots` removes decoder parameters (a hard capacity cap), whereas stronger ∅-weighting only biases an existing loss term. Two independent short-budget winners — one loss-weighting, one architectural — now both confirmed null at full budget, sufficient to treat the "confirm at ≥200 epochs" step as an operating protocol on this task.

## 30. Fault-contrast from mean-RMS normalization is load-bearing only when the backbone erases amplitude internally — amplitude-preserving tokenisers make the preprocessing redundant

**Domain rationale**
Under fault-aware training with gain / bias / partial faults, the trained model learns to exploit *per-sensor amplitude contrast* as a damage-vs-fault detection cue: a gain- or bias-faulted sensor has anomalously high RMS, and the model uses this to flag the sensor and re-weight attention. Where this cue enters the model depends on the backbone's treatment of amplitude:

- **Amplitude-erasing backbones** (DenseNet with BatchNorm in early convs) actively discard per-channel scale inside their feature extractor. The only place they can access amplitude-contrast information is the pre-processing — specifically the global-mean RMS denominator, which couples all sensors' scales together. A biased sensor inflates the denominator, which suppresses clean channels, producing a high-contrast input pattern the downstream attention learns to use.
- **Amplitude-preserving tokenisers** (iTransformer's per-sensor `Linear(T, D)`) carry absolute per-sensor amplitude directly into the token magnitude. The same fault-contrast cue is available intrinsically; external pre-normalization becomes redundant.

**Empirical evidence** — two ablations, same augmentation (`--p-hard 0.3 --p-soft 0.3 --p-struct-mask 0.3`, 200 epochs, C+fh+sb head):

1. **DenseNet backbone**, 4-way norm ablation (`states/7story-fault-norm-{median,none,median-mil}` + baseline mean). **Mean strictly dominates every alternative at every cell**. K=1 mean F1 by nf={0,13,32,52}: mean 0.939 / 0.935 / 0.926 / 0.872; median 0.915 / 0.911 / 0.903 / 0.837; no-norm 0.910 / 0.907 / 0.895 / 0.826. K=2 mirrors. At K=1 nf=52 median loses to mean on all 7 fault types, largest gaps on the bias family (bias −0.049, gain_bias −0.043, gain −0.023). Robust-statistic intuition fails here because the model has learned to use the non-robust coupling.

2. **iTransformer backbone**, 2-way ablation (`states/7story-fault-itransformer` mean vs `states/7story-fault-it-no-norm` none). **The gap collapses to within ±0.005 at every cell**. K=1: 0.944/0.939/0.929/0.860 (mean) vs 0.940/0.938/0.929/0.862 (no-norm), Δ ∈ {−0.004, −0.001, 0.000, +0.002}. K=2: 0.878/0.876/0.867/0.812 (mean) vs 0.878/0.875/0.867/0.817 (no-norm), Δ ∈ {0.000, −0.001, +0.001, +0.005}. The strongest pro-no-norm cell is the K=2 extreme-fault corner (nf=52, +0.005), consistent with the Linear tokeniser dominating the fault signal under severe amplitude distortion.

**ML implication**
A pre-processing step's role in the end-to-end pipeline is a function of *what the downstream architecture already provides*. Mean-RMS pre-normalization is not universally "the right preprocessing for SHM" — it is load-bearing under backbones whose first layer erases absolute amplitude (BN/GN-based convs) and redundant under tokenisers that preserve it (per-sensor Linear / MLP in iTransformer-family models). With an amplitude-preserving tokeniser the choice of aggregator matters only for excitation-amplitude invariance and numerical conditioning, not for fault-contrast — so a switch to `none` or `median` is approximately neutral rather than strongly harmful.

This generalises: whenever a pre-processing step and a model component both expose the same signal, only one of them carries it end-to-end. Removing or changing the pre-processing is neutral *for that specific signal*, and the net effect becomes whatever other jobs the pre-processing was doing (excitation invariance, SNR, conditioning). Pre-processing ablations should therefore be run per backbone, not interpreted as architecture-independent.

**Methodological lesson**
The correct phrasing of the original DenseNet finding is: "under a backbone that erases amplitude, the pre-normalization statistic matters and the non-robust choice is strictly best." The iTransformer counter-ablation narrows the scope precisely: "under a backbone that preserves amplitude, the pre-normalization statistic is approximately neutral." A reviewer-style challenge of the form "have you tried robust normalization?" therefore has two valid answers depending on backbone, not one.

**Where in paper**
Appendix / ablation — present as two sub-tables (DenseNet and iTransformer) supporting the joint conclusion: under fault-aware training, the model exploits amplitude contrast; *where* it picks this up — external pre-norm vs internal tokeniser — depends on the backbone's amplitude handling.

**Quote-ready**
> Fault-aware training teaches the model to exploit per-sensor amplitude
> contrast as a detection cue. Where this cue enters the model depends on
> the backbone: an amplitude-erasing DenseNet backbone discards absolute
> scale internally and can only access the cue through the external global-
> mean RMS denominator, which couples all sensors' scales. An iTransformer
> with a per-sensor Linear tokeniser preserves absolute amplitude into the
> token magnitude directly and does not need the external coupling. A four-
> way normalisation ablation on DenseNet C+fh+sb (mean, median, none,
> median+MIL) shows mean strictly dominating every alternative across fault
> ratios — a ≈3–5 F1-point gap at K=1, n_faulted=52 on the bias-family
> faults that most strongly perturb the mean. The same ablation on
> iTransformer C+fh+sb (mean vs none) shows the gap vanishing: Δ ∈ [−0.005,
> +0.005] across the entire fault-ratio × K sweep, with no-norm actually
> winning at the K=2 extreme-fault corner. The lesson generalises: a pre-
> processing step is only load-bearing end-to-end if it carries information
> the downstream model cannot otherwise recover from its inputs.

## 31. The ASCE 4-story impact-hammer benchmark has a fundamental 4-fold damage-identifiability ceiling from symmetric mass × symmetric sensor layout × symmetric excitation

**Domain rationale**
The ASCE benchmark (Case 1, 12-DOF rigid-floor, symmetric mass) instruments each floor with 4 accelerometers at the mid-edges of a 2-bay × 2-bay plan (nodes 2, 4, 6, 8 — two x-axis sensors, two y-axis sensors per floor), and excites the roof with a diagonal half-sine impulse of equal x and y components applied at the roof centre. Under rigid-floor + symmetric mass, each story contributes 8 diagonal braces arranged around the floor perimeter, split 4-axis-x × 4-axis-y. The floor-level observable from the sensors is rigid-body translation (x, y) plus rotation — but with the dominant excitation mode being translation of a symmetric mass, the rotational component is weakly excited. The four corner braces on a given (story, axis) all enter the story-stiffness matrix identically, so any single-sample damage at any of those four corners produces an essentially identical free-decay response at the four mid-edge sensors.

**Empirical evidence** (B plain-regression head, K=1 scenarios on ASCE test set):
- For every true damaged brace `l`, the three braces that co-activate at rate 0.25 each are exactly the three other corners in the same (story, axis) 4-group: brace 0 → {1, 4, 5}; brace 2 → {3, 6, 7}; brace 8 → {9, 12, 13}; … (two 4-groups per story × 4 stories = 8 equivalence classes, each of size 4).
- Per-sample `max(pred)` at true-positive locations is *lower* than `max(pred)` at true-negative locations (0.147 vs 0.176) — the model ranks a peer brace above the true one on average, because the label is arbitrary within the 4-group.
- v1 softmax entropy on K=1 ≈ 1.56, very close to log(4) = 1.386 — the mass is split roughly evenly across exactly 4 locations. Raising the `ratio_alpha` threshold from 0.1 to 0.5 does not change `mean_k_pred` (≈ 4.0) or recall (1.0) — the four softmax values inside a 4-group are almost equal, so no fractional threshold can break them apart.

**Refinement — the equivalence is strict 2-fold, approximate 4-fold.**
Pairwise L2 distance between mean K=1 free-decay responses (30 samples/brace) within story 1:

```
       b0     b1     b2     b3     b4     b5     b6     b7
b0   0.000  0.396  6.214  6.427  2.242  2.178  6.951  6.412
b1   0.396  0.000  6.204  6.420  2.212  2.082  6.951  6.406
b4   2.242  2.212  6.267  6.472  0.000  0.677  7.200  6.657
b5   2.178  2.082  5.947  6.163  0.677  0.000  6.905  6.340
```
Within-class per-sample spread ≈ 2.0–3.2; ‖K=0 mean − K=1(b0) mean‖ = 4.44.

Two nested equivalence scales coexist:
- **Strict 2-fold (pair equivalence):** d(b0,b1)=0.40, d(b2,b3)=0.40, d(b4,b5)=0.68, d(b6,b7)=0.86 — all far below within-class noise. Each pair is one class from the sensor data. Four pairs/story × 4 stories = 16 strictly-resolvable classes out of 32 brace labels.
- **Approximate 4-fold (group equivalence):** d(b0,b4)=2.24 ≈ within-class spread 2.8 — borderline resolvable. {0,1}∪{4,5} and {2,3}∪{6,7} cluster at the edge of noise; statistically distinguishable in principle but swamped for any single sample.
- **Cross-4-group distances** (b0↔b6, b2↔b4, etc.) ≈ 6–7 — comfortably separable.

**Identifiability ceiling (closed form)**
For any head whose output at the 4-group is approximately uniform, K=1 scoring gives:
- recall ≈ K / |group| = 1/4 ≈ 0.25 if only one brace is called; or 4/4 = 1.0 if all four are called, at precision 1/4 = 0.25
- max achievable single-damage F1 ≈ 0.4 from either corner of this trade-off (weaker 4-fold reading) or ≈ 0.67 (P=1/2, R=1 under strict pair ceiling)
- observed ASCE K=1 F1 across B/v1/C/C+fh/C+fh+sb is 0.38–0.42 — models saturate at the weaker 4-fold ceiling, not the tighter 2-fold ceiling. Training curves confirm the plateau is geometric, not optimisation: val top-K recall hits 0.308 at epoch 20 of 200 for C+fh+sb and stays flat to epoch 200; B's best checkpoint is epoch 10 of 200.

This is not a training, calibration, or architecture defect — it is a modal observability limit imposed by the dataset geometry. There is some headroom between the achieved 4-fold ceiling and the tighter 2-fold ceiling (F1 0.4 → 0.67) that richer heads or pair-aware losses could in principle claim, but the 2-fold floor is hard: the pair-equivalent braces are indistinguishable from the 16-channel free-decay data.

**ML implication**
- Damage identification benchmarks that pair symmetric structure × symmetric sensors × symmetric excitation will under-measure the capability of any model, because the labels they use to score predictions are not recoverable from the inputs they supply. The comparison between heads on such a benchmark measures "which head handles the identifiability ceiling most gracefully", not "which head localises best".
- A sharper ASCE protocol for SDI comparisons would either (a) break symmetry — asymmetric excitation (single-corner hammer) or asymmetric sensor placement, (b) relabel to the 4-group level (reducing the 32-brace task to an 8-class story-axis problem where the ceiling lifts), or (c) accept the ceiling and treat top-K recall with K=true_damaged × 4 as the principal metric. Concrete recipe for option (a) in `data/asce_hammer/README.md` "Regenerating a symmetry-broken variant" section — apply the impulse at a roof corner (node 7) instead of the roof centre, generating a DOF-12 torque component that distinguishes the four corners per (story, axis).
- For the current fault-robust experiment, ASCE's role shifts from "another lab-style SDI benchmark" to "a symmetric-observability stress test": how does each head's fault-robustness generalise when the clean-data ceiling is geometry-limited rather than architecture-limited? The answer is meaningful even with a low absolute F1 — a head that loses little under faults on ASCE is robust in a structurally different regime than Qatar (high-L lab) or LUMO (low-L field).

**Where in paper**
Appendix / dataset-characterisation subsection for ASCE. Critical to include, since otherwise a reader would read ASCE F1 ≈ 0.40 as a model failure rather than a dataset ceiling. The 4-way softmax entropy check + the 4-group co-activation table from the B model are the two readable pieces of evidence.

**Quote-ready**
> The ASCE 4-story benchmark's symmetric mass distribution, 4-sensor-per-floor
> mid-edge layout, and diagonal roof-centre impulse jointly produce an
> eight-class story-axis equivalence partition of its 32 brace labels.
> Under Case 1's rigid-floor + symmetric-mass assumption, the four corner
> braces on a given (story, axis) contribute identically to the story
> stiffness matrix, and the mid-edge sensors resolve only floor translation —
> not the rotational component that would distinguish corners. A plain
> per-location regression head trained on this dataset co-activates each
> true-damaged brace with its three same-group peers at a uniform ~0.25
> rate, and a softmax head at K=1 places its entire probability mass on
> those four locations with entropy ≈ log 4. Single-damage F1 is therefore
> geometry-capped near 0.4 regardless of architecture — a dataset-level
> modal-observability ceiling, not a model-capacity limit. We report this
> ceiling as a first-class property of the benchmark and treat within-ceiling
> relative performance as the meaningful comparison axis.

---

## 32. DETR's `no_obj_weight=0.1` default is miscalibrated for SHM — the ∅ class is the majority outcome, not a rare one

**Domain rationale**
DETR's set-loss default down-weights the ∅ (no-object) class by 0.1 because in image detection there are typically ~100 slot queries and ~5 actual objects per image — ∅ slots outnumber foreground ~20:1, so without down-weighting the ∅ gradient would drown out the foreground signal. In structural damage identification the query budget is small (K_max=5 slots) and the foreground count is low (K_true ∈ {0, 1, 2}), so ∅ is still the *majority* slot outcome but only by a factor of 2–3×. Furthermore, K=0 samples (structure is healthy — the *deployment* default condition) contribute an all-∅ assignment whose total gradient under `no_obj_weight=0.1` is 5 × 0.1 = 0.5 per sample vs 1.4 for a K=1 sample and 2.3 for a K=2 sample — the model sees K=0 samples as 3× *weaker* training signal than damaged samples.

On 7story-fault-k0 with 200 undamaged samples in a 40k-sample training set (~0.5%), this under-supervision makes the ∅ class practically invisible. The trained slot head fires spuriously on ~50% of held-out healthy monitoring windows (`sample_far = 0.500` at clean). Raising `no_obj_weight` to 0.5 roughly rebalances the effective gradient contribution, and K=0 sample_far drops to 0.022 grand (0.000 clean, 0.062 at 80% sensor faults). The damaged-task F1 *also* improves (grand +0.044) because sharper ∅ confidence reduces spurious extra-slot firings on K=1 and K=2 samples — precision on damaged samples goes from 0.85 to 0.93 with no recall regression.

**ML implication**
The DETR set-loss hyperparameter set was calibrated to an image-detection regime (K_max ≫ K_true, large datasets with balanced K distributions). In specialised domains with different query/target ratios and *heavily skewed* K distributions (including common K=0 cases that the original DETR never encountered for image classification), the `no_obj_weight` default must be re-tuned. For SHM with K_max=5 and K ∈ {0, 1, 2}, `no_obj_weight=0.5` — five times DETR's default — is the correct scale. Under this adjustment, a slot-based head matches the single-argmax (softmax) head's precision on K=1 (0.93 vs 0.94), resolves the K=0 FAR problem that slot-based models were previously presumed unable to handle without architectural additions, and retains the K=2 advantage (+0.24 F1 grand vs softmax). No architectural change required.

**Where in paper**
Loss-design section / ablation study. This is the biggest single-hyperparameter lever across our tuning experiments and deserves its own subsection.

**Quote-ready**
> The default `no_obj_weight=0.1` in DETR-style set criteria is calibrated for
> image object detection, where ∅ slots outnumber foreground slots ~20:1. In
> structural damage identification with K_max=5 slots and K ∈ {0, 1, 2}, ∅
> remains the majority outcome but only by a factor of 2–3×, and K=0 healthy
> samples — the deployment-default condition — under-weight the ∅ gradient
> by 3× relative to damaged samples. We raise `no_obj_weight` to 0.5 and find
> that it simultaneously reduces the healthy-sample false-alarm rate from
> 50% to 2% and improves damaged-task F1 by 0.044, with no recall regression.
> The slot-based head then matches the single-argmax baseline's K=1 precision
> while retaining a +0.24 F1 advantage on K=2 multi-damage detection.

---

## 33. Auxiliary per-sensor heads competing with main-task supervision through a shared encoder hurts main-task performance — place the aux head before the encoder

**Domain rationale**
The sensor fault head (per-sensor binary "is this sensor broken?") and the damage slot head (cross-sensor pattern "which structural location is damaged?") operate at different physical scales. Fault detection is a per-sensor question — it benefits from isolated per-sensor features. Damage localisation is a cross-sensor pattern-matching question — it benefits from features that aggregate across sensors. Placing both heads downstream of a shared encoder (the iT self-attention stack) forces the encoder to serve both objectives simultaneously: gradients from the fault BCE loss push the encoder toward features that distinguish faulted from clean sensors, competing with damage-localisation gradients that push it toward features that express cross-sensor damage patterns. This competition uses encoder capacity that would otherwise be dedicated to the main task.

Moving the fault head *before* the encoder (reading raw per-sensor tokenizer output, with gradients flowing back only through the tokenizer) eliminates the competition. The encoder now has a single objective. On 7story-fault-k0 this is worth +0.04 damaged-task F1 grand and +0.03 K=1 F1 grand. The price is reduced fault-detection sensitivity (fault_f1 0.92 → 0.74) because pre-encoder features cannot exploit cross-sensor consistency cues that the post-encoder representation provides. For SHM deployment this is the right trade: fault detection is auxiliary (its role is to *not hurt* damage detection, not to succeed on its own), and a 0.74 fault F1 is still actionable for sensor-health reporting.

**ML implication**
Multi-head architectures with differing-scale objectives should not share feature pathways further than necessary. When one head operates at a fine granularity (per-element / per-sensor / per-pixel) and another at a coarse granularity (whole-sample / cross-element / set-level), the coarse head is best served by a dedicated deep path while the fine-grained head should tap the earliest stable representation. The standard pattern — "stack many heads on the encoder output and let them all compete for capacity" — is an assumption that fails when the heads have meaningfully different feature-scale requirements.

**Where in paper**
Architecture-ablation subsection. The pre-encoder-fault-head change is the second-largest tuning lever after `no_obj_weight=0.5` and together they define the proposed configuration.

**Quote-ready**
> Placing the per-sensor fault-detection head downstream of the cross-sensor
> encoder forces the encoder to allocate capacity to both objectives
> simultaneously, producing a gradient-path competition between per-sensor
> fault supervision and cross-sensor damage supervision. Moving the fault
> head *before* the encoder — reading raw per-sensor tokenizer output —
> releases the encoder to serve damage localisation exclusively and
> improves damaged-task F1 by 0.04 grand. Fault-detection sensitivity falls
> from 0.92 to 0.74 because the pre-encoder representation cannot exploit
> cross-sensor consistency cues, but this is the correct trade-off in a
> structural monitoring deployment where damage identification is the
> primary objective.

---

## 34. The "softmax-is-architecturally-best-at-K=1" intuition is a half-truth — slot heads with properly supervised ∅ class match softmax precision without inheriting softmax's K=1 cap

**Domain rationale**
Before this work, the standard framing of set-prediction vs softmax heads on SHM was: softmax heads (v1) win single-damage identification (K=1) because their normalised output naturally produces a single argmax prediction with bounded false-positive rate; set-prediction heads (C) win multi-damage (K≥2) because they can emit multiple slot predictions; but C cannot match softmax's K=1 precision because its slots may fire spuriously. This framed softmax as structurally optimal for K=1, making iT+C a "multi-damage specialist" rather than a universal winner.

The finding: this framing held *only* because slot heads had been trained with under-supervised ∅ class (`no_obj_weight=0.1`). Under that regime, slot predictions at K=1 had ~0.85 precision vs softmax's ~0.94 — a 0.09 gap that looked like an architectural ceiling. Once ∅ is properly supervised (nw=0.5), slot head precision reaches 0.93 — essentially tied with softmax (0.94) — and K=1 F1 grand hits 0.967 vs softmax's 0.962 (within noise). The slot head now *absorbs* softmax's K=1 strength without inheriting softmax's K=2 architectural cap (K=2 F1 0.886 vs softmax's 0.646). The "softmax is structurally better at K=1" claim was really "softmax trains its implicit ∅ behaviour automatically (via normalisation), slot heads need explicit ∅ supervision to match it."

**ML implication**
Apparent "architectural" advantages of one head over another should be tested under matched loss-calibration conditions before being attributed to architecture. A proper ablation of set-prediction vs single-argmax heads must include a sweep of the ∅-class weight — otherwise the comparison is between "a slot head starved for ∅ gradient" and "a softmax head whose ∅ behaviour is built into its output normalisation." The slot head is not inherently worse at K=1; it was under-supervised at K=1 under the default loss configuration.

**Where in paper**
Discussion / architectural comparison. This finding reframes the headline — the proposal is no longer "accept the K=1 cost to gain K=2" but "the slot head wins uniformly, we just had to tune the ∅ weight."

**Quote-ready**
> Under the standard DETR loss calibration (`no_obj_weight=0.1`), slot-based
> heads underperform single-argmax heads on single-damage identification by
> approximately 0.08 F1. This gap has previously been attributed to an
> architectural disadvantage of set prediction at K=1. We show it is instead
> a loss-calibration artefact: the slot head receives insufficient ∅-class
> gradient to sharpen its foreground-vs-background decision boundary. Under
> task-appropriate `no_obj_weight=0.5`, the slot head's K=1 precision
> matches the softmax head's (0.93 vs 0.94) and K=1 F1 is statistically
> indistinguishable (0.967 vs 0.962). The slot head retains its structural
> advantage on multi-damage (+0.24 K=2 F1 grand) with no K=1 cost.

---

## 35. Drop-in tokenizer replacements that aggressively compress the time dimension require more training capacity than the rest of the pipeline was tuned for

**Domain rationale**
The legacy iT+C encoder tokenizes each sensor's T=500 time series with a single `Linear(T, D)` layer — a learned linear combination that preserves any feature expressible as a weighted sum of time samples. A multi-scale dilated-conv tokenizer (stride=4, AAP(1), 4 dilation branches) was tested as a "richer" alternative: more principled multi-scale receptive fields, fewer parameters in the tokenizer proper. But it compresses T=500 → 125 → 1 before entering the iT encoder, leaving the encoder with only 4 scalar features per sensor per branch (i.e. 4 scalars per sensor total, up to a learned projection). On 7story-fault-k0, this tokenizer combined with the downstream 4-layer iT encoder and 2-layer decoder fails to converge at the 200-epoch budget used for linear-tokenizer runs: both msc+post-fh and msc+pre-fh at decoder-depth 2 plateau near val_top_k_recall=0.55–0.76 while linear variants reach 0.95+. Extending the decoder to 4 layers (msc+pre-fh, dec=4) partially recovers — val 0.85, test K=1 F1 0.85, K=2 F1 0.80 — but still strictly below the linear-tokenizer baseline's 0.95 / 0.89.

This is not an indictment of multi-scale convolution as a tokenizer — it is a training-budget mismatch. The msc tokenizer with AAP-to-1 discards the T' sequence that would give the downstream iT something to attend over; under that constraint the model has to learn damage patterns from 4 per-sensor scalars and needs much more optimisation to do so.

**ML implication**
Drop-in tokenizer replacements in a pipeline tuned for one tokenizer (hyperparameters, epoch budget, learning rate, decoder depth) are not actually drop-in. A tokenizer that changes the *shape* of what enters the encoder (scalar-per-sensor vs vector-per-sensor-time, full resolution vs compressed) changes the optimisation landscape enough to require its own tuning pass. Our msc variant should either (a) preserve the time sequence (stride=1, no AAP collapse, feed `(B, S, T', D)` into the encoder so the encoder has something to attend over in time), or (b) inherit a different training budget (more epochs, deeper decoder, LR warm-up) — both are non-trivial changes. We defer the redesign to future work and use the linear tokenizer as the proposal's backbone.

**Where in paper**
Ablation-study subsection on tokenizer choice. Report as a negative result — "we tried multi-scale convolution, it did not pay off under the shared training budget, here's why we believe this is a training-budget mismatch rather than an architectural inferiority."

**Quote-ready**
> Replacing the Linear(T, D) tokenizer with a multi-scale dilated convolution
> (stride=4, global pooling, 4 dilation branches) failed to converge under
> the 200-epoch training budget used for the linear-tokenizer baselines.
> The compressed representation (4 scalars per sensor after pooling) leaves
> the downstream iT encoder with no time-resolution information to attend
> over, and recovering damage patterns from this compressed representation
> requires either deeper decoder capacity (partially verified: decoder
> depth 4 recovers val top-K recall from 0.55 to 0.85 but still trails the
> linear baseline's 0.95) or a substantially longer training budget. We
> treat this as a training-budget mismatch, not an architectural inferiority,
> and retain the linear tokenizer as the proposal backbone. A redesign that
> preserves the time sequence rather than pooling to a scalar is deferred
> to future work.
