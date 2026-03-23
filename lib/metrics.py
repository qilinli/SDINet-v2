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

def distributed_map_b(presence_logits: Tensor, severity: Tensor) -> Tensor:
    """
    Distributed damage map for the B-head, in [0, 1] space.

    Formula: ``sigmoid(presence_logits) * severity``

    Properties:
    - Undamaged (presence prob → 0): map → 0
    - Damaged (presence prob → 1, severity → s): map → s
    - Target space: ``(y_norm + 1) / 2 = raw_damage / 0.15 ∈ [0, 1]``

    Args:
        presence_logits: (B, L) or (B, L, N)  raw presence logits
        severity:        (B, L) or (B, L, N)  sigmoid-activated ∈ [0, 1]
    Returns:
        Tensor of same shape, values in [0, 1]
    """
    return torch.sigmoid(presence_logits) * severity


def distributed_map_v1(dmg_pred: Tensor, loc_pred: Tensor) -> Tensor:
    """
    Distributed damage map for the v1 head, remapped to [0, 1] for fair comparison.

    Formula: ``(dmg_tanh + 1) / 2 * softmax(loc)``

    Note: v1's internal ``val_one_epoch`` uses the equivalent ``(…).mul(2).sub(1)``
    form in [-1, 1] space.  This version is in [0, 1] for a common comparison
    baseline against ``distributed_map_b``.

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
    - ``pred_map``     — output of :func:`distributed_map_b` or :func:`distributed_map_v1`
    - ``target_norm``  — ``(y_norm + 1) / 2``, where y_norm is the model's raw label

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
) -> dict[str, float]:
    """
    Compute the evaluation suite for the v1 head using the same metric keys
    as :func:`evaluate_all`.

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
    """
    dist_map  = distributed_map_v1(dmg_pred, loc_pred)   # (B, L) in [0, 1]
    target_01 = (y_norm + 1.0) / 2.0
    mse = F.mse_loss(dist_map, target_01).item()

    # Softmax probabilities as presence scores (natural for v1 single-damage head)
    loc_probs   = loc_pred.softmax(dim=-1)                # (B, L)
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
    sev_broadcast = ((dmg_pred + 1.0) / 2.0).expand_as(loc_pred)   # (B, L)
    sev_mae = severity_mae_at_detected(
        loc_pred, sev_broadcast, y_norm,
        k=k, presence_norm_thresh=presence_norm_thresh,
    )

    # F1 using sigmoid(loc_pred) for API consistency with evaluate_all
    tp, fp, fn    = presence_f1_stats(loc_pred, y_norm,
                                      presence_norm_thresh=presence_norm_thresh)
    f1, prec, rec = f1_from_counts(tp, fp, fn)

    pred_pres   = torch.sigmoid(loc_pred) > 0.5
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


@torch.no_grad()
def evaluate_all(
    presence_logits: Tensor,
    severity: Tensor,
    y_norm: Tensor,
    k: int | None = None,
    presence_norm_thresh: float = PRESENCE_NORM_THRESH,
) -> dict[str, float]:
    """
    Compute the full evaluation suite for the B-head on a dataset split.

    Designed for test-set evaluation after training.  Concatenate predictions
    from all batches before calling::

        pres_all = torch.cat([...])   # (N_samples, L)
        sev_all  = torch.cat([...])   # (N_samples, L)
        y_all    = torch.cat([...])   # (N_samples, L)
        results  = evaluate_all(pres_all, sev_all, y_all)

    Args:
        presence_logits: (N, L) raw presence logits
        severity:        (N, L) sigmoid-activated severity ∈ [0, 1]
        y_norm:          (N, L) normalized ground-truth labels ∈ [-1, 1]
        k:               Fixed K for top-K metrics.  None = use true K per sample.
        presence_norm_thresh: Threshold separating undamaged from damaged labels.

    Returns:
        Dict with keys:

        ``map_mse``
            Distributed-map MSE in [0, 1] space.  Primary comparison metric,
            valid for both v1 and B heads.

        ``top_k_recall``
            Fraction of true damaged locations in the top-K predictions.
            K is the true number of damages per sample when ``k=None``.

        ``severity_mae``
            MAE at correctly detected locations only (true positives).
            Decouples severity accuracy from localization accuracy.

        ``ap``
            Average Precision (area under PR curve).  Threshold-free detection
            quality summary; robust under heavy class imbalance.

        ``f1``, ``precision``, ``recall``
            Presence detection at the default threshold (0.5).  Reported for
            reference; prefer ``ap`` for threshold-independent comparison.

        ``mean_k_pred``, ``mean_k_true``
            Mean number of predicted and true damaged locations per sample
            (at threshold 0.5).  Useful for diagnosing over/under-detection.
    """
    dist_map  = distributed_map_b(presence_logits, severity)   # (N, L)
    mse       = map_mse(dist_map, y_norm).item() / y_norm.size(0)

    # True K per sample (used when k=None)
    y_pres_bool = y_norm > presence_norm_thresh                 # (N, L) bool
    k_true_per  = y_pres_bool.sum(dim=-1).float()               # (N,)

    # Top-K recall: use per-sample true K if k is None
    if k is None:
        probs = torch.sigmoid(presence_logits)
        hits = total_pos = 0
        for b in range(y_norm.size(0)):
            K = int(k_true_per[b].item())
            if K == 0:
                continue
            topk_idx  = probs[b].topk(K).indices
            topk_mask = torch.zeros(y_norm.size(1), dtype=torch.bool, device=y_norm.device)
            topk_mask[topk_idx] = True
            hits      += int((topk_mask & y_pres_bool[b]).sum())
            total_pos += K
        tkr = hits / max(total_pos, 1)
    else:
        tkr = top_k_recall(presence_logits, y_norm, k=k,
                           presence_norm_thresh=presence_norm_thresh)

    sev_mae = severity_mae_at_detected(
        presence_logits, severity, y_norm,
        k=k, presence_norm_thresh=presence_norm_thresh,
    )
    ap = average_precision(presence_logits, y_norm,
                           presence_norm_thresh=presence_norm_thresh)

    tp, fp, fn = presence_f1_stats(presence_logits, y_norm,
                                   presence_norm_thresh=presence_norm_thresh)
    f1, prec, rec = f1_from_counts(tp, fp, fn)

    pred_pres = torch.sigmoid(presence_logits) > 0.5
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
