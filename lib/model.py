from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lib.densenet import SDIDenseNet
from lib.midn import Midn, MidnC, MidnDR, PlainDR


# ---------------------------------------------------------------------------
# v1 config (unchanged)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    # input shape assumptions (after preprocessing)
    # Must match `sensor_dim` in `lib.data_7story.input_preprocess`: raw acc is
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
    plain: bool = False  # True = mean-pool baseline (PlainDR), no learned sensor attention


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

    # --- sensor spatial reasoning (default off — backward compatible) ---
    use_spatial_layer:  bool  = False
    num_spatial_layers: int   = 1
    spatial_nhead:      int   = 8

    # --- fault detection head (default off — backward compatible) ---
    use_fault_head:    bool  = False
    fault_loss_weight: float = 1.0
    fault_pos_weight:  float = 5.0   # BCE pos_weight ≈ (S − mean_faults) / mean_faults

    # --- structural location-sensor affinity bias (default off — backward compatible) ---
    # Qatar only: learnable (L+1, S) bias matrix initialised from 6×5 grid 4-connectivity.
    use_structural_bias: bool = False


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

def build_model(
    cfg: ModelConfig | ModelConfigC | ModelConfigDR = ModelConfig(),
    structural_affinity: "torch.Tensor | None" = None,
) -> nn.Sequential:
    """
    Build the full SDINet model: backbone → neck → head.

    Pass a :class:`ModelConfig` to get the original v1 head (``Midn``).

    The returned ``nn.Sequential`` is always structured as::

        model[0]  —  SDIDenseNet  (feature extractor)
        model[1]  —  neck Conv1d block
        model[2]  —  prediction head

    This layout is relied upon by the training/eval loops which slice
    ``model[:2]`` for the feature forward pass and ``model[2]`` for the head.

    structural_affinity: (L+1, S) binary tensor passed to MidnC when
        cfg.use_structural_bias=True.  Must be provided by the caller
        (train.py selects the correct affinity based on dataset name).
    """
    feature_extractor = SDIDenseNet(cfg.in_channels, structure=cfg.structure, bn_params={})
    neck_in = _infer_neck_in_channels(feature_extractor, cfg)
    neck = _build_neck(neck_in, cfg)

    if isinstance(cfg, ModelConfigDR):
        if cfg.plain:
            head = PlainDR(cfg.embed_dim, cfg.num_locations)
        else:
            head = MidnDR(
                cfg.embed_dim,
                cfg.num_locations,
                cfg.importance_dropout,
                temperature=cfg.temperature,
                val_temperature=cfg.val_temperature,
            )
    elif isinstance(cfg, ModelConfigC):
        if cfg.use_structural_bias and structural_affinity is None:
            raise ValueError(
                "cfg.use_structural_bias=True but no structural_affinity was passed "
                "to build_model().  Pass the dataset-specific affinity tensor."
            )
        head = MidnC(
            cfg.embed_dim,
            cfg.num_slots,
            cfg.num_locations,
            cfg.num_decoder_layers,
            cfg.nhead,
            n_sensors=cfg.n_sensors,
            use_spatial_layer=cfg.use_spatial_layer,
            num_spatial_layers=cfg.num_spatial_layers,
            spatial_nhead=cfg.spatial_nhead,
            use_fault_head=cfg.use_fault_head,
            structural_affinity=structural_affinity,
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


def build_criterion_fault(cfg: ModelConfigC):
    """Build FaultBCELoss if fault head is enabled, else return None."""
    if not cfg.use_fault_head:
        return None
    from lib.losses import FaultBCELoss
    return FaultBCELoss(pos_weight=cfg.fault_pos_weight)


# ---------------------------------------------------------------------------
# Checkpoint loaders
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path,
    *,
    device=None,
    model_cfg=None,
) -> nn.Sequential:
    """Load a v1 (Midn) checkpoint. Infers architecture from weights if model_cfg not provided."""
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if model_cfg is None:
        # Midn uses Conv1d(in, 2*out_channels, 1), so weight shape is [2*out_channels, embed_dim, 1]
        out_channels = state["2.layer.weight"].shape[0] // 2
        embed_dim    = state["2.layer.weight"].shape[1]
        model_cfg = ModelConfig(out_channels=out_channels, embed_dim=embed_dim)
    model = build_model(model_cfg)
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
    """Load an Approach-DR (MidnDR) or plain baseline (PlainDR) checkpoint. Infers architecture from weights."""
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if model_cfg is None:
        if "2.importance_branch.weight" in state:
            num_locations = state["2.importance_branch.weight"].shape[0]
            embed_dim     = state["2.importance_branch.weight"].shape[1]
            plain = False
        elif "2.linear.weight" in state:
            num_locations = state["2.linear.weight"].shape[0]
            embed_dim     = state["2.linear.weight"].shape[1]
            plain = True
        else:
            raise ValueError("Cannot infer DR/B architecture from checkpoint keys")
        model_cfg = ModelConfigDR(num_locations=num_locations, embed_dim=embed_dim, plain=plain)
    model = build_model(model_cfg)
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
        use_spatial   = "2.spatial.layers.layers.0.self_attn.in_proj_weight" in state
        use_fault     = "2.fault_head.weight" in state
        use_struct    = "2.R_bias" in state
        # n_sensors only needed for SensorPositionalEncoding; use default (65) otherwise —
        # _infer_neck_in_channels returns C×T' which does not depend on S.
        n_sensors     = state["2.pos_enc.pos_emb"].shape[0] if use_spatial else ModelConfigC().n_sensors
        num_spatial_layers = (
            sum(1 for k in state if k.startswith("2.spatial.layers.layers.") and k.endswith(".norm1.weight"))
            if use_spatial else 1
        )
        model_cfg = ModelConfigC(
            num_slots=num_slots,
            num_locations=num_locations,
            n_sensors=n_sensors,
            use_spatial_layer=use_spatial,
            num_spatial_layers=num_spatial_layers,
            use_fault_head=use_fault,
            use_structural_bias=use_struct,
        )
    else:
        use_struct = model_cfg.use_structural_bias
    structural_affinity = state["2.R_bias"] if use_struct else None
    model = build_model(model_cfg, structural_affinity=structural_affinity)
    model.load_state_dict(state)
    if device is not None:
        model = model.to(device)
    return model
