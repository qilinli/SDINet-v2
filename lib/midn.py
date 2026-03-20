from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn


@torch.compile
def importance_dropout(x: Tensor, p: float, inplace: bool = False) -> Tensor:
    if not inplace:
        x = x.clone()
    mask = torch.rand_like(x) < p
    invalid = mask.sum(-1) == mask.size(-1)
    while invalid.any():
        mask[invalid] = torch.rand_like(x[invalid]) < p
        invalid = mask.sum(-1) == mask.size(-1)
    x[mask] = -float("inf")
    return x


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
