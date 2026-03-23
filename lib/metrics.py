from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

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
