from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lib.densenet import SDIDenseNet
from lib.midn import Midn


@dataclass(frozen=True)
class ModelConfig:
    # input shape assumptions (after preprocessing)
    # Must match `sensor_dim` in `lib.data_safetensors.input_preprocess`: raw acc is
    # 3-axis per sensor, but by default only the first (x) axis is kept → 1 input channel.
    in_channels: int = 1
    time_len: int = 500
    n_sensors: int = 65

    # backbone
    structure: tuple[int, int, int] = (6, 6, 6)

    # neck / head dims
    embed_dim: int = 768
    out_channels: int = 71  # 1 dmg + 70 loc

    # head behavior
    importance_dropout: float = 0.5
    temperature: float = 1e-2
    val_temperature: float = 1e-2

    # regularization
    neck_dropout: float = 0.0


def _infer_neck_in_channels(feature_extractor: nn.Module, cfg: ModelConfig) -> int:
    feature_extractor.eval()
    with torch.inference_mode():
        dummy = torch.zeros(
            (1, cfg.in_channels, cfg.time_len, cfg.n_sensors), dtype=torch.float32
        )
        feats = feature_extractor(dummy)  # (B, C, T', S)
        feats = torch.flatten(feats, 1, 2)  # (B, C*T', S)
        return int(feats.size(1))


def build_model(cfg: ModelConfig = ModelConfig()) -> nn.Sequential:
    feature_extractor = SDIDenseNet(cfg.in_channels, structure=cfg.structure, bn_params={})
    neck_in = _infer_neck_in_channels(feature_extractor, cfg)

    neck = nn.Sequential(
        nn.Flatten(1, 2),
        nn.Conv1d(neck_in, cfg.embed_dim, 1),
        nn.ReLU(True),
        nn.Dropout(cfg.neck_dropout),
        nn.Conv1d(cfg.embed_dim, cfg.embed_dim, 1),
        nn.ReLU(True),
    )
    head = Midn(
        cfg.embed_dim,
        cfg.out_channels,
        cfg.importance_dropout,
        temperature=cfg.temperature,
        val_temperature=cfg.val_temperature,
    )
    return nn.Sequential(feature_extractor, neck, head)

