from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn

# ---------------------------------------------------------------------------
# SDIDenseNet default hyperparameters (also used as constructor defaults)
# ---------------------------------------------------------------------------
SDI_DEFAULT_IN_CHANNELS: int = 1
SDI_DEFAULT_BASE_CHANNELS: int = 24
SDI_DEFAULT_GROWTH_RATE: int = 12
SDI_DEFAULT_STRUCTURE: tuple[int, int, int] = (6, 6, 6)
# Initial stem: conv along time axis only (kernel height = patch_size, width = 1)
SDI_DEFAULT_PATCH_SIZE: int = 20
SDI_DEFAULT_PATCH_STRIDE: int = 5
SDI_DEFAULT_COMPRESSION: float = 0.0
SDI_DEFAULT_KERNEL_SIZE: tuple[int, int] = (5, 1)
SDI_DEFAULT_BN_MOMENTUM: float = 0.01
SDI_DEFAULT_BN_EPS: float = 1.1e-5
SDI_DEFAULT_BN_PARAMS: dict = {
    "momentum": SDI_DEFAULT_BN_MOMENTUM,
    "eps": SDI_DEFAULT_BN_EPS,
}
# Between dense stages: halve time resolution, keep sensor axis unchanged
SDI_DOWNPOOL_KERNEL: tuple[int, int] = (2, 1)
# DenseBlock: bottleneck expansion factor for latent conv width
SDI_LATENT_GROWTH_MULTIPLIER: int = 2
# 1×1 conv inside DenseBlock (channel mixing)
SDI_DENSE_POINTWISE_KERNEL: int = 1


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        growth_rate: int,
        depth: int,
        kernel_size: tuple[int, int],
        activation: Callable[[], nn.Module],
        bn_params: dict,
    ) -> None:
        super(DenseBlock, self).__init__()
        layers = []
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        for i in range(depth):
            layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate, **bn_params),
                    activation(),
                    nn.Conv2d(
                        in_channels + i * growth_rate,
                        latent_channels,
                        kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(latent_channels, **bn_params),
                    activation(),
                    nn.Conv2d(
                        latent_channels, growth_rate, SDI_DENSE_POINTWISE_KERNEL
                    ),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [x]
        for layer in self.layers:
            xs.append(layer(torch.cat(xs, dim=1)))
        return torch.cat(xs, dim=1)


class SDIDenseNet(nn.Sequential):
    def __init__(
        self,
        in_channels: int = SDI_DEFAULT_IN_CHANNELS,
        base_channels: int = SDI_DEFAULT_BASE_CHANNELS,
        growth_rate: int = SDI_DEFAULT_GROWTH_RATE,
        structure: Sequence[int] = SDI_DEFAULT_STRUCTURE,
        patch_size: int = SDI_DEFAULT_PATCH_SIZE,
        patch_stride: int = SDI_DEFAULT_PATCH_STRIDE,
        compression: float = SDI_DEFAULT_COMPRESSION,
        activation: Callable[[], nn.Module] = nn.ReLU,
        kernel_size: tuple[int, int] = SDI_DEFAULT_KERNEL_SIZE,
        bn_params: dict | None = None,
    ) -> None:
        if bn_params is None:
            bn_params = dict(SDI_DEFAULT_BN_PARAMS)
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                base_channels,
                (patch_size, 1),
                (patch_stride, 1),
                (patch_size // 2, 0),
                bias=False,
            )
        ]
        channels = base_channels
        latent_channels = SDI_LATENT_GROWTH_MULTIPLIER * growth_rate
        for i, depth in enumerate(structure):
            if i != 0:
                layers.append(nn.BatchNorm2d(channels, **bn_params))
                layers.append(nn.AvgPool2d(SDI_DOWNPOOL_KERNEL))
                layers.append(activation())
                new_channels = round(channels * (1.0 - compression))
                layers.append(nn.Conv2d(channels, new_channels, 1, bias=False))
                channels = new_channels
            layers.append(
                DenseBlock(
                    channels,
                    latent_channels,
                    growth_rate,
                    depth,
                    kernel_size,
                    activation,
                    bn_params,
                )
            )
            channels += growth_rate * depth
        super(SDIDenseNet, self).__init__(*layers)
