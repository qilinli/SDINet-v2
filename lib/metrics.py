from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import average_precision_score

# ---------------------------------------------------------------------------
# Normalized label space: y_norm = raw_damage / 0.15 - 1  ∈ [-1, 1]
# Undamaged → -1.0,  max-damaged → +1.0
# Distributed map comparison uses [0, 1] space: (y_norm + 1) / 2
# ---------------------------------------------------------------------------
PRESENCE_NORM_THRESH: float = -1.0 + 1e-4


# ---------------------------------------------------------------------------
# Distributed damage maps — both expressed in [0, 1] for fair comparison
# ---------------------------------------------------------------------------

def distributed_map_dr(pred: Tensor) -> Tensor:
    """
    Distributed damage map for the DR head — the prediction IS the map.

    DR directly outputs severity ∈ [0, 1] per location, so the damage map
    is the raw output with no further transformation.

    Args:
        pred: (B, L)  sigmoid-activated severity prediction ∈ [0, 1]
    Returns:
        (B, L) — identical to input
    """
    return pred




def distributed_map_v1(dmg_pred: Tensor, loc_pred: Tensor) -> Tensor:
    """
    Distributed damage map for the v1 head, remapped to [0, 1] for fair comparison.

    Formula: ``(dmg_tanh + 1) / 2 * softmax(loc)``

    Note: v1's internal ``val_one_epoch`` uses the equivalent ``(…).mul(2).sub(1)``
    form in [-1, 1] space.  This version is in [0, 1] for a common comparison
    baseline against ``distributed_map_c`` and ``distributed_map_dr``.

    Args:
        dmg_pred: (B, 1) or (B, 1, N)  Tanh-activated global damage scalar
        loc_pred: (B, L) or (B, L, N)  raw location logits
    Returns:
        Tensor of same spatial shape as loc_pred, values in [0, 1]
    """
    softmax_dim = -2 if loc_pred.dim() == 3 else -1
    return (dmg_pred + 1.0) / 2.0 * loc_pred.softmax(dim=softmax_dim)


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def map_mse(pred_map: Tensor, target_norm: Tensor) -> Tensor:
    """
    MSE between the predicted damage map and the normalized ground truth.

    Both are aligned in [0, 1] space:
    - ``pred_map``     — output of :func:`distributed_map_v1` or :func:`distributed_map_dr`
    - ``target_norm``  — raw normalized labels ``y_norm ∈ [-1, 1]`` (converted to [0, 1] internally)

    When ``pred_map`` is (B, L, N) (subset evaluation), ``target_norm`` (B, L)
    is broadcast along the N dimension.

    Returns:
        (N,) tensor of per-subset batch-summed MSEs for accumulation, or a
        scalar when ``pred_map`` is (B, L).
    """
    target_01 = (target_norm + 1.0) / 2.0
    if pred_map.dim() == 3:
        target_01 = target_01.unsqueeze(-1).expand_as(pred_map)
    return F.mse_loss(pred_map, target_01, reduction="none").mean(1).sum(0)


@torch.no_grad()
def presence_f1_stats(
    presence_logits: Tensor,
    y_norm: Tensor,
    threshold: float = 0.5,
    presence_norm_thresh: float = PRESENCE_NORM_THRESH,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Accumulate TP, FP, FN counts for damage-presence detection.

    Call this on each batch and sum the returned counts; compute the final F1
    at epoch end with :func:`f1_from_counts`.

    Args:
        presence_logits:     (B, L) raw logits from the B-head
        y_norm:              (B, L) normalized labels ∈ [-1, 1]
        threshold:           Probability threshold for binarising predictions
        presence_norm_thresh: Labels above this value count as "damaged"
    Returns:
        (tp, fp, fn) as scalar long tensors
    """
    y_presence   = y_norm > presence_norm_thresh
    pred_presence = torch.sigmoid(presence_logits) > threshold
    tp = (pred_presence &  y_presence).sum().long()
    fp = (pred_presence & ~y_presence).sum().long()
    fn = (~pred_presence &  y_presence).sum().long()
    return tp, fp, fn


def f1_from_counts(
    tp: int | Tensor,
    fp: int | Tensor,
    fn: int | Tensor,
    eps: float = 1e-8,
) -> tuple[float, float, float]:
    """
    Compute F1, precision, and recall from accumulated TP/FP/FN counts.

    Returns:
        (f1, precision, recall) as plain floats
    """
    tp, fp, fn = float(tp), float(fp), float(fn)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2.0 * precision * recall / (precision + recall + eps)
    return f1, precision, recall


@torch.no_grad()
def fault_detection_metrics(
    fault_prob: Tensor,
    y_fault: Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Per-sensor binary fault detection metrics.

    Args:
        fault_prob: (B, S) sigmoid-activated fault probabilities ∈ [0, 1]
        y_fault:    (B, S) binary ground truth ∈ {0, 1}
        threshold:  decision threshold (default 0.5)

    Returns:
        dict with keys: fault_f1, fault_precision, fault_recall
    """
    pred = fault_prob >= threshold
    gt   = y_fault > 0.5
    tp = int((pred &  gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    f1, prec, rec = f1_from_counts(tp, fp, fn)
    return {"fault_f1": f1, "fault_precision": prec, "fault_recall": rec}


@torch.no_grad()
def top_k_recall(
    presence_logits: Tensor,
    y_norm: Tensor,
    k: int = 2,
    presence_norm_thresh: float = PRESENCE_NORM_THRESH,
) -> float:
    """
    Fraction of truly damaged locations appearing in the top-k predictions.

    Threshold-free: useful when the optimal detection threshold is unknown or
    when K (number of damages) is known a-priori.

    Args:
        presence_logits:     (B, L) raw logits or scores (higher → more likely damaged)
        y_norm:              (B, L) normalized labels
        k:                   Number of top-scoring locations to consider per sample
        presence_norm_thresh: Threshold defining "truly damaged"
    Returns:
        Recall scalar ∈ [0, 1]
    """
    y_presence = y_norm > presence_norm_thresh          # (B, L) bool
    probs      = torch.sigmoid(presence_logits)          # (B, L)
    topk_mask  = torch.zeros_like(probs, dtype=torch.bool)
    topk_idx   = probs.topk(k, dim=-1).indices           # (B, k)
    topk_mask.scatter_(-1, topk_idx, True)
    tp        = (topk_mask & y_presence).sum().float()
    total_pos = y_presence.sum().float().clamp(min=1.0)
    return (tp / total_pos).item()


@torch.no_grad()
def average_precision(
    presence_logits: Tensor,
    y_norm: Tensor,
    presence_norm_thresh: float = PRESENCE_NORM_THRESH,
) -> float:
    """
    Area under the Precision-Recall curve for damage presence detection.

    Threshold-free summary of detection quality: integrates P-R tradeoff
    across all possible decision thresholds.  More informative than F1 at a
    single threshold, especially under class imbalance (K << L).

    Args:
        presence_logits:     (B, L) raw logits
        y_norm:              (B, L) normalized labels ∈ [-1, 1]
        presence_norm_thresh: Labels above this value count as "damaged"
    Returns:
        AP ∈ [0, 1]
    """
    probs  = torch.sigmoid(presence_logits).cpu().numpy().ravel()
    labels = (y_norm > presence_norm_thresh).cpu().numpy().ravel().astype(int)
    return float(average_precision_score(labels, probs))


@torch.no_grad()
def severity_mae_at_detected(
    presence_logits: Tensor,
    severity: Tensor,
    y_norm: Tensor,
    k: int | None = None,
    presence_norm_thresh: float = PRESENCE_NORM_THRESH,
) -> float:
    """
    Mean Absolute Error of severity at *correctly detected* locations only.

    For each sample, top-K locations are selected by presence score.  Among
    those that are truly damaged (true positives), the MAE between predicted
    and true severity is computed.  Locations that were missed (false
    negatives) or incorrectly flagged (false positives) are excluded.

    This decouples severity accuracy from localization accuracy: a model that
    finds the right location should not be penalised for what it predicts at
    wrong locations.

    Args:
        presence_logits:     (B, L) raw logits — used for ranking locations
        severity:            (B, L) sigmoid-activated severity ∈ [0, 1]
        y_norm:              (B, L) normalized labels ∈ [-1, 1]
        k:                   Locations to consider per sample.  If None,
                             uses the true K for each sample individually
                             (oracle K — best-case localization scenario).
        presence_norm_thresh: Threshold defining "truly damaged"
    Returns:
        MAE scalar.  Returns 0.0 if no true positives exist in the batch.
    """
    probs     = torch.sigmoid(presence_logits)           # (B, L)
    y_pres    = y_norm > presence_norm_thresh             # (B, L) bool
    y_sev     = (y_norm + 1.0) / 2.0                     # (B, L) in [0, 1]

    total_mae = 0.0
    total_tp  = 0

    for b in range(y_norm.size(0)):
        true_locs = y_pres[b].nonzero(as_tuple=True)[0]
        K = int(true_locs.numel()) if k is None else k
        if K == 0:
            continue
        topk_idx = probs[b].topk(K).indices              # (K,)
        for idx in topk_idx:
            if y_pres[b, idx]:                            # true positive
                total_mae += abs(severity[b, idx].item() - y_sev[b, idx].item())
                total_tp  += 1

    return total_mae / max(total_tp, 1)


@torch.no_grad()
def evaluate_all_v1(
    dmg_pred: Tensor,
    loc_pred: Tensor,
    y_norm: Tensor,
    k: int | None = None,
    presence_norm_thresh: float = PRESENCE_NORM_THRESH,
    temperature: float = 1.0,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
    dmg_gate: float | None = None,
) -> dict[str, float]:
    """
    Compute the evaluation suite for the v1 head using the same metric keys
    as :func:`evaluate_all_dr` and :func:`evaluate_all_c`.

    v1 predicts a single global damage scalar (``dmg_pred``, tanh-activated)
    and raw location logits (``loc_pred``).  The adaptation to the shared
    metric interface is:

    - ``map_mse``:      :func:`distributed_map_v1` in [0, 1] space.
    - Localization:     ``softmax(loc_pred)`` used as per-location presence
                        scores for ``top_k_recall`` and ``ap`` — natural for a
                        single-damage softmax head.
    - ``severity_mae``: scalar ``(dmg_pred + 1) / 2`` broadcast to all L
                        locations (v1 has no per-location severity).
    - ``f1`` / ``precision`` / ``recall``: ``sigmoid(loc_pred) > 0.5`` for
                        API consistency; prefer ``top_k_recall`` and ``ap``
                        for v1 comparisons.

    Args:
        dmg_pred: (B, 1) tanh-activated global damage scalar
        loc_pred: (B, L) raw location logits
        y_norm:   (B, L) normalized ground-truth labels ∈ [-1, 1]
        k:        Fixed K for top-K metrics. None = use true K per sample.
        presence_norm_thresh: Threshold separating undamaged from damaged labels.
        dmg_gate: If set, predict K=0 for windows where ``dmg_pred < dmg_gate``.
                  Critical for small L where softmax max ≥ 1/L always, making
                  the beta gate ineffective.
    """
    loc_scaled = loc_pred / temperature
    dist_map  = distributed_map_v1(dmg_pred, loc_scaled)  # (B, L) in [0, 1]
    target_01 = (y_norm + 1.0) / 2.0
    mse = F.mse_loss(dist_map, target_01).item()

    # Softmax probabilities as presence scores (natural for v1 single-damage head)
    loc_probs   = loc_scaled.softmax(dim=-1)               # (B, L)
    y_pres_bool = y_norm > presence_norm_thresh
    k_true_per  = y_pres_bool.sum(dim=-1).float()

    # Top-K recall using softmax ranking
    if k is None:
        hits = total_pos = 0
        for b in range(y_norm.size(0)):
            K = int(k_true_per[b].item())
            if K == 0:
                continue
            topk_idx  = loc_probs[b].topk(K).indices
            topk_mask = torch.zeros(y_norm.size(1), dtype=torch.bool, device=y_norm.device)
            topk_mask[topk_idx] = True
            hits      += int((topk_mask & y_pres_bool[b]).sum())
            total_pos += K
        tkr = hits / max(total_pos, 1)
    else:
        topk_mask = torch.zeros_like(loc_probs, dtype=torch.bool)
        topk_mask.scatter_(-1, loc_probs.topk(k, dim=-1).indices, True)
        tkr = (
            (topk_mask & y_pres_bool).sum().float()
            / y_pres_bool.sum().float().clamp(min=1.0)
        ).item()

    # AP using softmax probabilities
    probs_np  = loc_probs.cpu().numpy().ravel()
    labels_np = y_pres_bool.cpu().numpy().ravel().astype(int)
    ap = float(average_precision_score(labels_np, probs_np))

    # Severity MAE: broadcast scalar damage to all locations
    sev_broadcast = ((dmg_pred + 1.0) / 2.0).expand_as(loc_scaled)   # (B, L)
    sev_mae = severity_mae_at_detected(
        loc_scaled, sev_broadcast, y_norm,
        k=k, presence_norm_thresh=presence_norm_thresh,
    )

    # F1: ratio threshold on softmax probs (if calibrated), else sigmoid > 0.5 fallback
    if ratio_alpha is not None:
        max_prob  = loc_probs.max(dim=-1, keepdim=True).values
        pred_pres = (loc_probs > ratio_alpha * max_prob) & (max_prob > ratio_beta)
    else:
        pred_pres = torch.sigmoid(loc_scaled) > 0.5
    # Damage-scalar gate: predict K=0 for windows where dmg_pred < dmg_gate.
    # Essential for small L where softmax(max) >= 1/L always, making beta
    # ineffective — the damage scalar is the only way to express "no damage".
    if dmg_gate is not None:
        pred_pres = pred_pres & (dmg_pred > dmg_gate)
    y_pres_bool_f1 = y_norm > presence_norm_thresh
    tp = (pred_pres &  y_pres_bool_f1).sum().long()
    fp = (pred_pres & ~y_pres_bool_f1).sum().long()
    fn = (~pred_pres & y_pres_bool_f1).sum().long()
    f1, prec, rec = f1_from_counts(tp, fp, fn)

    mean_k_pred = pred_pres.sum(dim=-1).float().mean().item()
    mean_k_true = k_true_per.mean().item()

    return {
        "map_mse":      mse,
        "top_k_recall": tkr,
        "severity_mae": sev_mae,
        "ap":           ap,
        "f1":           f1,
        "precision":    prec,
        "recall":       rec,
        "mean_k_pred":  mean_k_pred,
        "mean_k_true":  mean_k_true,
    }


# ---------------------------------------------------------------------------
# Approach-DR evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_all_dr(
    pred: Tensor,
    y_norm: Tensor,
    k: int | None = None,
    presence_norm_thresh: float = PRESENCE_NORM_THRESH,
    threshold: float = 0.5,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
) -> dict[str, float]:
    """
    Compute the full evaluation suite for the DR head on a dataset split.

    DR outputs a single severity value ∈ [0, 1] per location, used directly
    as both the damage map (no transformation) and the presence score.

    Args:
        pred:      (N, L) sigmoid-activated severity ∈ [0, 1]
        y_norm:    (N, L) normalized ground-truth labels ∈ [-1, 1]
        k:         Fixed K for top-K metrics. None = use true K per sample.
        threshold: Presence threshold on pred (default 0.5).
        ratio_alpha: Ratio rule: predict l if pred[l] > α * max pred.
        ratio_beta:  Absolute gate: max pred must exceed β (default 0.0).

    Returns the same dict as :func:`evaluate_all_v1` for cross-model comparison.
    """
    dist_map  = distributed_map_dr(pred)                        # (N, L) = pred
    mse       = map_mse(dist_map, y_norm).item() / y_norm.size(0)

    y_pres_bool = y_norm > presence_norm_thresh
    k_true_per  = y_pres_bool.sum(dim=-1).float()

    # Top-K recall: rank by pred value
    if k is None:
        hits = total_pos = 0
        for b in range(y_norm.size(0)):
            K = int(k_true_per[b].item())
            if K == 0:
                continue
            topk_idx  = pred[b].topk(K).indices
            topk_mask = torch.zeros(y_norm.size(1), dtype=torch.bool, device=y_norm.device)
            topk_mask[topk_idx] = True
            hits      += int((topk_mask & y_pres_bool[b]).sum())
            total_pos += K
        tkr = hits / max(total_pos, 1)
    else:
        topk_mask = torch.zeros_like(pred, dtype=torch.bool)
        topk_mask.scatter_(-1, pred.topk(k, dim=-1).indices, True)
        tkr = (
            (topk_mask & y_pres_bool).sum().float()
            / y_pres_bool.sum().float().clamp(min=1.0)
        ).item()

    # AP: pred values serve directly as probability scores
    probs_np  = pred.cpu().numpy().ravel()
    labels_np = y_pres_bool.cpu().numpy().ravel().astype(int)
    ap = float(average_precision_score(labels_np, probs_np))

    # Severity MAE at correctly detected (true-positive) locations
    y_sev     = (y_norm + 1.0) / 2.0
    total_mae = 0.0
    total_tp  = 0
    for b in range(y_norm.size(0)):
        K = int(k_true_per[b].item()) if k is None else k
        if K == 0:
            continue
        topk_idx = pred[b].topk(K).indices
        for idx in topk_idx:
            if y_pres_bool[b, idx]:
                total_mae += abs(pred[b, idx].item() - y_sev[b, idx].item())
                total_tp  += 1
    sev_mae = total_mae / max(total_tp, 1)

    # Binary presence: absolute or ratio thresholding on pred
    if ratio_alpha is not None:
        max_pred  = pred.max(dim=-1, keepdim=True).values
        pred_pres = (pred > ratio_alpha * max_pred) & (max_pred > ratio_beta)
    else:
        pred_pres = pred > threshold

    tp = (pred_pres &  y_pres_bool).sum().long()
    fp = (pred_pres & ~y_pres_bool).sum().long()
    fn = (~pred_pres & y_pres_bool).sum().long()
    f1, prec, rec = f1_from_counts(tp, fp, fn)

    mean_k_pred = pred_pres.sum(dim=-1).float().mean().item()
    mean_k_true = k_true_per.mean().item()

    return {
        "map_mse":      mse,
        "top_k_recall": tkr,
        "severity_mae": sev_mae,
        "ap":           ap,
        "f1":           f1,
        "precision":    prec,
        "recall":       rec,
        "mean_k_pred":  mean_k_pred,
        "mean_k_true":  mean_k_true,
    }


# ---------------------------------------------------------------------------
# Approach-C slot decoding
# ---------------------------------------------------------------------------

def _c_slot_decode(loc_logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Pure DETR slot decoding for the C-head.

    A slot is active when the no-object class does NOT win the argmax
    competition — no threshold, no free parameter.

        active[b, k] = (argmax over L+1 classes) ≠ L   (the ∅ index)

    ``is_obj`` (= 1 − P(∅)) is returned only for ranking slots by confidence
    in top-K recall; it is NOT used for the active/inactive decision.

    Args:
        loc_logits: (B, K, L+1)  raw location logits (last dim = no-object ∅)
    Returns:
        active   : (B, K) bool  — slot fires (argmax ≠ ∅)
        pred_loc : (B, K) int   — argmax location index (valid when active)
        is_obj   : (B, K) float — P(not ∅) for ranking only
    """
    no_obj_idx = loc_logits.size(-1) - 1
    argmax     = loc_logits.argmax(-1)                 # (B, K)
    active     = argmax != no_obj_idx                  # (B, K) bool
    pred_loc   = loc_logits[..., :-1].argmax(-1)       # (B, K) best location
    is_obj     = 1.0 - loc_logits.softmax(-1)[..., -1] # (B, K) for ranking
    return active, pred_loc, is_obj


def _c_slot_decode_ensemble(
    loc_logits: Tensor,
    severity: Tensor | None = None,
    use_severity: bool = True,
) -> Tensor:
    """
    Ensemble slot decoding for the C-head — all K slots fold into one per-location
    score map.  Recovers v1-style MIL redundancy at inference without retraining:
    instead of committing to a single argmax per slot, each slot casts a soft
    vote weighted by its is-object confidence (and optionally severity).

        M[b, l] = Σ_k is_obj[b, k] · (sev[b, k]) · softmax(loc_logits[b, k])[:L][l]

    Args:
        loc_logits:   (B, K, L+1) raw location logits (last dim = ∅).
        severity:     (B, K) sigmoid severity ∈ [0, 1]; required when use_severity.
        use_severity: Include severity factor in the slot weight (default True).

    Returns:
        M: (B, L) non-negative ensemble damage score per location.  Can exceed 1
        when multiple slots vote for the same location; detection uses ratio
        thresholding which is scale-invariant.
    """
    probs  = loc_logits.softmax(-1)                       # (B, K, L+1)
    loc_p  = probs[..., :-1]                              # (B, K, L)
    is_obj = 1.0 - probs[..., -1]                         # (B, K)
    w      = is_obj
    if use_severity and severity is not None:
        w = w * severity
    return (loc_p * w.unsqueeze(-1)).sum(dim=1)           # (B, L)


@torch.no_grad()
def evaluate_all_c(
    loc_logits: Tensor,
    severity: Tensor,
    y_norm: Tensor,
    k: int | None = None,
    presence_norm_thresh: float = PRESENCE_NORM_THRESH,
    ensemble: bool = False,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
    use_severity: bool = True,
    **_,
) -> dict[str, float]:
    """
    Full evaluation for the C-head.

    Two decoding modes:

    * **Default (ensemble=False)** — pure DETR slot decoding.  A slot fires when
      the no-object class does not win the argmax.  ``map_mse`` and ``ap`` are
      returned as NaN because this mode does not produce a soft per-location map.

    * **ensemble=True** — the K slots' softmax distributions are weighted by
      ``is_obj`` (and optionally ``severity``) and summed into a single per-location
      score map (see :func:`_c_slot_decode_ensemble`).  Detection then uses DR-style
      ratio thresholding: ``pred[l] > α·max(pred)`` AND ``max(pred) > β``.  All
      metrics (including ``map_mse`` and ``ap``) become well-defined in this mode.

    Args:
        loc_logits:    (N, K_max, L+1) raw location logits (last = ∅).
        severity:      (N, K_max)      sigmoid severity ∈ [0, 1].
        y_norm:        (N, L)          normalized ground truth ∈ [-1, 1].
        k:             Fixed K for top-K recall.  None = use true K per sample.
        ensemble:      Use ensemble soft-map decoding instead of argmax slots.
        ratio_alpha:   α for the ratio rule when ensemble=True.  None falls back
                       to absolute threshold 0.5 on the ensemble map.
        ratio_beta:    β for the ratio rule when ensemble=True.
        use_severity:  Multiply the ensemble weight by slot severity (ensemble only).
    """
    if ensemble:
        M = _c_slot_decode_ensemble(loc_logits, severity, use_severity=use_severity)
        return evaluate_all_dr(
            M, y_norm,
            k=k,
            presence_norm_thresh=presence_norm_thresh,
            ratio_alpha=ratio_alpha,
            ratio_beta=ratio_beta,
        )

    NaN = float("nan")

    active, pred_loc, is_obj = _c_slot_decode(loc_logits)   # (N, K) each

    y_pres_bool = y_norm > presence_norm_thresh          # (N, L) bool
    k_true_per  = y_pres_bool.sum(-1).float()            # (N,)
    y_sev       = (y_norm + 1.0) / 2.0                  # (N, L) in [0, 1]

    # Top-K recall: rank slots by is_obj, take top-K, check predicted locations
    hits = total_pos = 0
    for b in range(y_norm.size(0)):
        K = int(k_true_per[b].item()) if k is None else k
        if K == 0:
            continue
        K_slots = min(K, is_obj.size(-1))
        top_slots = is_obj[b].topk(K_slots).indices          # top-K slots by confidence
        pred_set  = set(pred_loc[b, top_slots].tolist())
        true_set  = set(y_pres_bool[b].nonzero(as_tuple=False)[:, 0].tolist())
        hits      += len(pred_set & true_set)
        total_pos += int(k_true_per[b].item()) if k is None else k
    tkr = hits / max(total_pos, 1)

    # F1 / precision / recall from active slots (set-based, deduplication included)
    tp = fp = fn = 0
    sev_mae_total = 0.0
    sev_mae_count = 0
    for b in range(y_norm.size(0)):
        pred_set = set(pred_loc[b, active[b]].tolist())
        true_set = set(y_pres_bool[b].nonzero(as_tuple=False)[:, 0].tolist())
        tp_locs  = pred_set & true_set
        tp += len(tp_locs)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
        # Severity MAE: for each TP location, use the highest-confidence active slot
        for loc in tp_locs:
            best_k = max(
                (k_idx for k_idx in range(active.size(-1))
                 if active[b, k_idx] and pred_loc[b, k_idx].item() == loc),
                key=lambda ki: is_obj[b, ki].item(),
                default=None,
            )
            if best_k is not None:
                sev_mae_total += abs(severity[b, best_k].item() - y_sev[b, loc].item())
                sev_mae_count += 1

    f1, prec, rec = f1_from_counts(tp, fp, fn)
    sev_mae    = sev_mae_total / max(sev_mae_count, 1)
    mean_k_pred = active.sum(-1).float().mean().item()
    mean_k_true = k_true_per.mean().item()

    return {
        "map_mse":      NaN,
        "top_k_recall": tkr,
        "severity_mae": sev_mae,
        "ap":           NaN,
        "f1":           f1,
        "precision":    prec,
        "recall":       rec,
        "mean_k_pred":  mean_k_pred,
        "mean_k_true":  mean_k_true,
    }
