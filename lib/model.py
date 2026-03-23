from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lib.densenet import SDIDenseNet
from lib.midn import Midn, MidnB


# ---------------------------------------------------------------------------
# v1 config (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# v2 / Approach-B config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfigB:
    """
    Configuration for the multi-damage B-head model.

    Backbone and neck are identical to v1 (``ModelConfig``).  The head is
    replaced by :class:`~lib.midn.MidnB` which outputs independent presence
    logits and severity values for each of the ``num_locations`` structural
    locations instead of the v1 single-damage (scalar dmg + softmax loc) pair.

    Loss hyperparameters are stored here so that ``build_model`` can create the
    matching :class:`~lib.losses.PresenceSeverityLoss` in one place.

    Args:
        num_locations:       Structural locations L (default 70, matching v1).
        presence_pos_weight: BCE positive-class weight.  Rule of thumb:
                             ``(L - K_mean) / K_mean``.  Default 69 ≈ single damage.
        severity_weight:     Relative scale of severity MSE vs presence BCE.
    """

    # --- shared with ModelConfig ---
    in_channels: int = 1
    time_len: int = 500
    n_sensors: int = 65
    structure: tuple[int, int, int] = (6, 6, 6)
    embed_dim: int = 768
    neck_dropout: float = 0.0

    # --- B-head specific ---
    num_locations: int = 70
    importance_dropout: float = 0.5
    temperature: float = 1e-2
    val_temperature: float = 1e-2

    # --- loss hyperparameters (used by build_criterion_b) ---
    presence_pos_weight: float = 69.0
    severity_weight: float = 1.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _infer_neck_in_channels(
    feature_extractor: nn.Module,
    cfg: ModelConfig | ModelConfigB,
) -> int:
    feature_extractor.eval()
    with torch.inference_mode():
        dummy = torch.zeros(
            (1, cfg.in_channels, cfg.time_len, cfg.n_sensors), dtype=torch.float32
        )
        feats = feature_extractor(dummy)        # (B, C, T', S)
        feats = torch.flatten(feats, 1, 2)      # (B, C*T', S)
        return int(feats.size(1))


def _build_neck(neck_in: int, cfg: ModelConfig | ModelConfigB) -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(1, 2),
        nn.Conv1d(neck_in, cfg.embed_dim, 1),
        nn.ReLU(True),
        nn.Dropout(cfg.neck_dropout),
        nn.Conv1d(cfg.embed_dim, cfg.embed_dim, 1),
        nn.ReLU(True),
    )


# ---------------------------------------------------------------------------
# Model factory — dispatches on config type
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig | ModelConfigB = ModelConfig()) -> nn.Sequential:
    """
    Build the full SDINet model: backbone → neck → head.

    Pass a :class:`ModelConfig` to get the original v1 head (``Midn``).
    Pass a :class:`ModelConfigB` to get the multi-damage B-head (``MidnB``).

    The returned ``nn.Sequential`` is always structured as::

        model[0]  —  SDIDenseNet  (feature extractor)
        model[1]  —  neck Conv1d block
        model[2]  —  Midn or MidnB (prediction head)

    This layout is relied upon by the training/eval loops which slice
    ``model[:2]`` for the feature forward pass and ``model[2]`` for the head.
    """
    feature_extractor = SDIDenseNet(cfg.in_channels, structure=cfg.structure, bn_params={})
    neck_in = _infer_neck_in_channels(feature_extractor, cfg)
    neck = _build_neck(neck_in, cfg)

    if isinstance(cfg, ModelConfigB):
        head = MidnB(
            cfg.embed_dim,
            cfg.num_locations,
            cfg.importance_dropout,
            temperature=cfg.temperature,
            val_temperature=cfg.val_temperature,
        )
    else:
        head = Midn(
            cfg.embed_dim,
            cfg.out_channels,
            cfg.importance_dropout,
            temperature=cfg.temperature,
            val_temperature=cfg.val_temperature,
        )

    return nn.Sequential(feature_extractor, neck, head)


def build_criterion_b(cfg: ModelConfigB):
    """Convenience: build the matching PresenceSeverityLoss from a ModelConfigB."""
    from lib.losses import PresenceSeverityLoss
    return PresenceSeverityLoss(
        pos_weight=cfg.presence_pos_weight,
        severity_weight=cfg.severity_weight,
    )

