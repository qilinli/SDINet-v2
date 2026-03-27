from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lib.densenet import SDIDenseNet
from lib.midn import Midn, MidnC, MidnDR


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
# Direct Regression config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfigDR:
    """Configuration for the direct regression head (Approach DR)."""

    in_channels: int = 1
    time_len: int = 500
    n_sensors: int = 65
    structure: tuple[int, int, int] = (6, 6, 6)
    embed_dim: int = 768
    neck_dropout: float = 0.0

    num_locations: int = 70
    importance_dropout: float = 0.5
    temperature: float = 1e-2
    val_temperature: float = 1e-2


# ---------------------------------------------------------------------------
# Approach-C (DETR-style) config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfigC:
    """
    Configuration for the DETR-style slot-prediction head (Approach C).

    Backbone and neck are identical to v1 and B.  The head is replaced by
    :class:`~lib.midn.MidnC` which uses K_max learnable slot queries that
    cross-attend over per-sensor features to jointly predict location and
    severity for each potential damage.

    Loss hyperparameters are stored here so that ``build_criterion_c`` can
    create the matching :class:`~lib.losses.SetCriterion` in one call.

    Args:
        num_slots:          Slot count K_max (default: num_locations + 1).
        num_decoder_layers: Transformer decoder depth (default 2).
        nhead:              Attention heads — embed_dim must be divisible (default 8).
        num_locations:      Structural locations L (default 70).
        no_obj_weight:      ∅-class down-weighting in location CE (default 0.1).
        loc_weight:         Scale of location CE in total loss (default 1.0).
        sev_weight:         Scale of severity MSE in total loss and matching cost
                            (default 1.0).
    """

    # --- shared with ModelConfig ---
    in_channels: int = 1
    time_len: int = 500
    n_sensors: int = 65
    structure: tuple[int, int, int] = (6, 6, 6)
    embed_dim: int = 768
    neck_dropout: float = 0.0

    # --- C-head specific ---
    num_locations: int = 70
    num_slots: int | None = 5      # None → falls back to num_locations + 1 in __post_init__
    num_decoder_layers: int = 2
    nhead: int = 8

    def __post_init__(self) -> None:
        if self.num_slots is None:
            object.__setattr__(self, "num_slots", self.num_locations + 1)

    # --- loss hyperparameters (used by build_criterion_c) ---
    no_obj_weight: float = 0.1
    loc_weight: float = 1.0
    sev_weight: float = 1.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _infer_neck_in_channels(
    feature_extractor: nn.Module,
    cfg: ModelConfig | ModelConfigC | ModelConfigDR,
) -> int:
    feature_extractor.eval()
    with torch.inference_mode():
        dummy = torch.zeros(
            (1, cfg.in_channels, cfg.time_len, cfg.n_sensors), dtype=torch.float32
        )
        feats = feature_extractor(dummy)        # (B, C, T', S)
        feats = torch.flatten(feats, 1, 2)      # (B, C*T', S)
        return int(feats.size(1))


def _build_neck(neck_in: int, cfg: ModelConfig | ModelConfigC | ModelConfigDR) -> nn.Sequential:
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

def build_model(cfg: ModelConfig | ModelConfigC | ModelConfigDR = ModelConfig()) -> nn.Sequential:
    """
    Build the full SDINet model: backbone → neck → head.

    Pass a :class:`ModelConfig` to get the original v1 head (``Midn``).

    The returned ``nn.Sequential`` is always structured as::

        model[0]  —  SDIDenseNet  (feature extractor)
        model[1]  —  neck Conv1d block
        model[2]  —  prediction head

    This layout is relied upon by the training/eval loops which slice
    ``model[:2]`` for the feature forward pass and ``model[2]`` for the head.
    """
    feature_extractor = SDIDenseNet(cfg.in_channels, structure=cfg.structure, bn_params={})
    neck_in = _infer_neck_in_channels(feature_extractor, cfg)
    neck = _build_neck(neck_in, cfg)

    if isinstance(cfg, ModelConfigDR):
        head = MidnDR(
            cfg.embed_dim,
            cfg.num_locations,
            cfg.importance_dropout,
            temperature=cfg.temperature,
            val_temperature=cfg.val_temperature,
        )
    elif isinstance(cfg, ModelConfigC):
        head = MidnC(
            cfg.embed_dim,
            cfg.num_slots,
            cfg.num_locations,
            cfg.num_decoder_layers,
            cfg.nhead,
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


def build_criterion_c(cfg: ModelConfigC):
    """Convenience: build the matching SetCriterion from a ModelConfigC."""
    from lib.losses import SetCriterion
    return SetCriterion(
        num_locations=cfg.num_locations,
        no_obj_weight=cfg.no_obj_weight,
        loc_weight=cfg.loc_weight,
        sev_weight=cfg.sev_weight,
    )


# ---------------------------------------------------------------------------
# Checkpoint loaders
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path,
    *,
    device=None,
    model_cfg=None,
) -> nn.Sequential:
    """Load a v1 (Midn) checkpoint. Infers default ModelConfig if not provided."""
    if model_cfg is None:
        model_cfg = ModelConfig()
    model = build_model(model_cfg)
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if device is not None:
        model = model.to(device)
    return model


def load_model_dr_from_checkpoint(
    checkpoint_path,
    *,
    device=None,
    model_cfg=None,
) -> nn.Sequential:
    """Load an Approach-DR (MidnDR) checkpoint. Infers default ModelConfigDR if not provided."""
    if model_cfg is None:
        model_cfg = ModelConfigDR()
    model = build_model(model_cfg)
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if device is not None:
        model = model.to(device)
    return model


def load_model_c_from_checkpoint(
    checkpoint_path,
    *,
    device=None,
    model_cfg=None,
) -> nn.Sequential:
    """Load an Approach-C (MidnC) checkpoint. Infers architecture from weights if model_cfg not provided."""
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if model_cfg is None:
        num_slots     = state["2.queries"].shape[0]
        num_locations = state["2.loc_head.weight"].shape[0] - 1  # L+1 outputs
        model_cfg = ModelConfigC(num_slots=num_slots, num_locations=num_locations)
    model = build_model(model_cfg)
    model.load_state_dict(state)
    if device is not None:
        model = model.to(device)
    return model
