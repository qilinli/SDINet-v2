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
