from __future__ import annotations

import sys
from typing import Callable

import torch
from torch import Tensor, nn


def _importance_dropout_impl(x: Tensor, p: float, inplace: bool = False) -> Tensor:
    if not inplace:
        x = x.clone()
    mask = torch.rand_like(x) < p
    invalid = mask.sum(-1) == mask.size(-1)
    while invalid.any():
        mask[invalid] = torch.rand_like(x[invalid]) < p
        invalid = mask.sum(-1) == mask.size(-1)
    x[mask] = -float("inf")
    return x


# torch.compile is not supported on Windows; fall back to plain Python silently.
if sys.platform == "win32":
    importance_dropout = _importance_dropout_impl
else:
    importance_dropout = torch.compile(_importance_dropout_impl)


class Midn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        importance_dropout: float = 0.0,
        temperature: float = 1.0,
        val_temperature: float | None = None,
        pred_activation: "Callable[[], nn.Module]" = nn.Tanh,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = nn.Conv1d(in_channels, 2 * out_channels, 1)
        self.pred_activation = pred_activation()
        self.temperature = temperature
        self.val_temperature = val_temperature or temperature
        self.importance_dropout = importance_dropout

    def forward(
        self, x: torch.Tensor, reduce: bool = True
    ) -> "Tensor | tuple[Tensor, Tensor]":
        x = self.layer(x)
        prediction = x[:, : self.out_channels, :]
        importance = (
            x[:, self.out_channels :, :] * (self.temperature if self.training else self.val_temperature)
        )
        if self.training:
            importance = importance_dropout(importance, self.importance_dropout)

        importance = importance.softmax(-1)
        dmg_importance, loc_importance = importance[:, :1], importance[:, 1:]
        dmg_prediction, loc_prediction = self.pred_activation(prediction[:, :1]), prediction[:, 1:]

        if reduce:
            return (dmg_prediction * dmg_importance).sum(2), (loc_prediction * loc_importance).sum(2)
        else:
            return dmg_prediction, dmg_importance, loc_prediction, loc_importance


class MidnB(nn.Module):
    """
    Multi-damage MIL head with shared per-location importance (Approach B).

    Three independent Conv1d branches:
    - ``importance_branch``: which sensors are informative for each location
    - ``presence_branch``:   is there damage here? (logits → BCE loss)
    - ``severity_branch``:   how much damage?     (sigmoid → [0, 1] → MSE loss)

    Sharing importance across presence and severity is physically motivated:
    the sensors that reveal *whether* location k is damaged are the same ones
    that reveal *how much* it is damaged.

    Distributed damage map = ``sigmoid(presence_logits) * severity ∈ [0, 1]``
    Compare against ``(y_norm + 1) / 2`` in [0, 1] space.

    Args:
        in_channels:         Feature dimension from the neck (embed_dim).
        num_locations:       Number of structural locations L (default 70).
        importance_dropout:  Fraction of sensor importance logits to mask to
                             -inf before softmax during training (same as v1).
        temperature:         Scale applied to importance logits during training.
        val_temperature:     Scale during evaluation (defaults to ``temperature``).
    """

    def __init__(
        self,
        in_channels: int,
        num_locations: int = 70,
        importance_dropout: float = 0.0,
        temperature: float = 1.0,
        val_temperature: float | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_locations = num_locations
        self.temperature = temperature
        self.val_temperature = val_temperature or temperature
        self.importance_dropout_p = importance_dropout

        self.importance_branch = nn.Conv1d(in_channels, num_locations, 1)
        self.presence_branch   = nn.Conv1d(in_channels, num_locations, 1)
        self.severity_branch   = nn.Conv1d(in_channels, num_locations, 1)

    def forward(
        self, x: Tensor, reduce: bool = True
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x:      (B, in_channels, S) per-sensor neck features
            reduce: Whether to aggregate over the sensor dimension S.

        Returns (reduce=True):
            presence_logits: (B, L) — raw logits for BCE
            severity:        (B, L) — sigmoid output ∈ [0, 1]

        Returns (reduce=False):
            presence_raw:  (B, L, S) — per-sensor presence logits (pre-aggregation)
            severity_raw:  (B, L, S) — per-sensor severity (post-sigmoid, pre-aggregation)
            importance:    (B, L, S) — importance weights (softmax over S, sums to 1)

        The ``reduce=False`` interface is used by ``val_one_epoch_b`` to evaluate
        sensor-failure robustness: subset-indexing the S dimension then
        renormalising importance mirrors the pattern in v1's ``val_one_epoch``.
        """
        temp = self.temperature if self.training else self.val_temperature
        imp_logits = self.importance_branch(x) * temp          # (B, L, S)
        if self.training and self.importance_dropout_p > 0.0:
            imp_logits = importance_dropout(imp_logits, self.importance_dropout_p)
        imp = imp_logits.softmax(-1)                           # (B, L, S)

        presence_raw = self.presence_branch(x)                 # (B, L, S) logits
        severity_raw = torch.sigmoid(self.severity_branch(x))  # (B, L, S) ∈ [0, 1]

        if reduce:
            presence_logits = (presence_raw * imp).sum(-1)     # (B, L)
            severity        = (severity_raw * imp).sum(-1)     # (B, L)
            return presence_logits, severity
        else:
            return presence_raw, severity_raw, imp
