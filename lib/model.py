from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lib.densenet import SDIDenseNet
from lib.itransformer import POS_EMB_TYPES, iTransformerEncoder
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

    # --- structural location-sensor affinity bias (default off — backward compatible) ---
    # Qatar only: learnable (L+1, S) bias matrix initialised from 6×5 grid 4-connectivity.
    use_structural_bias: bool = False

    fault_gate_lambda: float = 0.0
    freeze_r_bias:     bool  = False
    aux_loss:          bool  = False

    # --- Option 4: magnitude-invariant memory (default off — backward compatible) ---
    use_l2_norm_memory: bool = False
    # --- Option 5: MIL-style slot readout via last-layer cross-attention weights ---
    use_mil_readout:    bool = False
    # --- Final decoder-output LayerNorm (canonical pre-norm transformer hygiene).
    # Default True for new trainings; auto-detected as False when loading old
    # checkpoints lacking `2.decoder.norm.*` keys.
    use_final_decoder_norm: bool = True

    # --- Encoder backbone for C-head.
    # "densenet" (default): SDIDenseNet + MIL neck (original v1/B/C backbone).
    # "itransformer":       per-sensor Linear(T, D) tokenisation + sensor-axis
    #                       self-attention (see lib/itransformer.py).
    encoder_type:      str = "densenet"
    encoder_num_layers: int = 2
    encoder_nhead:      int = 8
    encoder_dropout:    float = 0.0
    encoder_pos_emb:   str = "learned"   # iTransformer: "learned" | "none" | "rope"

    # --- Phase B1: iTransformer tokenizer choice.
    # "linear" (default):         Linear(T, D) — legacy behaviour.
    # "multiscale_conv":          parallel dilated Conv1d branches, preserves
    #                             sub-window temporal structure.
    tokenizer_type:    str = "linear"

    # --- Phase B2: fault head placement.
    # "decoder" (default):        fault head is a submodule of MidnC, reads
    #                             post-encoder features (legacy behaviour).
    # "encoder":                  fault head is a submodule of
    #                             iTransformerEncoder, reads raw per-sensor
    #                             tokens before any cross-sensor attention
    #                             mixing (more fault-robust at high nf).
    fault_head_location: str = "decoder"


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
    use_itransformer = (
        isinstance(cfg, ModelConfigC) and cfg.encoder_type == "itransformer"
    )
    if use_itransformer:
        feature_extractor = iTransformerEncoder(
            n_sensors=cfg.n_sensors,
            time_len=cfg.time_len,
            embed_dim=cfg.embed_dim,
            num_layers=cfg.encoder_num_layers,
            nhead=cfg.encoder_nhead,
            dropout=cfg.encoder_dropout,
            pos_emb_type=cfg.encoder_pos_emb,
            tokenizer_type=cfg.tokenizer_type,
            fault_head_location=cfg.fault_head_location if cfg.use_fault_head else "decoder",
        )
        # nn.Identity keeps the model[:2] / model[2] slicing contract
        # (training/eval loops split the sequential at index 2).
        # Identity.forward returns its input verbatim, so the
        # (sensor_features, fault_prob) tuple passes through to MidnC.
        neck = nn.Identity()
    else:
        # Phase-B (tokenizer/fault-head-location) attributes only exist on
        # ModelConfigC. For v1 (ModelConfig) and B/DR (ModelConfigDR) the
        # check is vacuous — they always use the DenseNet path with default
        # tokenizer and don't own a fault head.
        if isinstance(cfg, ModelConfigC) and (
            cfg.tokenizer_type != "linear" or cfg.fault_head_location != "decoder"
        ):
            raise ValueError(
                "tokenizer_type != 'linear' and fault_head_location != 'decoder' "
                "are only valid with encoder_type='itransformer'."
            )
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
            fault_gate_lambda=cfg.fault_gate_lambda,
            freeze_r_bias=cfg.freeze_r_bias,
            aux_loss=cfg.aux_loss,
            use_l2_norm_memory=cfg.use_l2_norm_memory,
            use_mil_readout=cfg.use_mil_readout,
            use_final_decoder_norm=cfg.use_final_decoder_norm,
            fault_head_location=cfg.fault_head_location,
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
    return FaultBCELoss()


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
        use_struct    = "2.R_bias" in state
        use_l2        = "2._l2_norm_memory_flag" in state
        use_mil       = "2.sensor_loc_head.weight" in state
        use_final_ln  = "2.decoder.norm.weight" in state

        # Fault head auto-detect: key path depends on Phase-B2 location.
        fault_in_decoder = "2.fault_head.weight" in state
        fault_in_encoder = "0.fault_head.weight" in state
        use_fault        = fault_in_decoder or fault_in_encoder
        fault_head_location = "encoder" if fault_in_encoder else "decoder"

        # Encoder auto-detect: iTransformer has a tokenizer at "0.tokenize.*";
        # DenseNet backbone has "0.features.*".
        is_itransformer = (
            "0.tokenize.weight" in state
            or "0.tokenize.proj.weight" in state
        )
        encoder_type    = "itransformer" if is_itransformer else "densenet"

        if is_itransformer:
            # Tokenizer auto-detect (Phase B1):
            #   - linear:          "0.tokenize.weight" shape (D, T)
            #   - multiscale_conv: "0.tokenize.proj.weight" + "0.tokenize.branches.*"
            if "0.tokenize.weight" in state:
                tokenizer_type = "linear"
                time_len  = state["0.tokenize.weight"].shape[1]   # (D, T)
                embed_dim = state["0.tokenize.weight"].shape[0]
            else:
                tokenizer_type = "multiscale_conv"
                embed_dim = state["0.tokenize.proj.weight"].shape[0]
                # MultiScaleConvTokenizer doesn't encode T in its params; fall back to
                # ModelConfigC default (matches the fixed preprocessing contract).
                time_len  = ModelConfigC().time_len

            # Positional-embedding variant: learned keeps "0.pos_emb";
            # rope uses custom encoder whose layers hold in_proj.weight (dotted);
            # none has neither.
            has_learned_pe = "0.pos_emb" in state
            has_rope = "0.layers.layers.0.self_attn.in_proj.weight" in state
            if has_learned_pe:
                encoder_pos_emb = "learned"
            elif has_rope:
                encoder_pos_emb = "rope"
            else:
                encoder_pos_emb = "none"

            if has_learned_pe:
                n_sensors = state["0.pos_emb"].shape[0]
            elif "0._n_sensors_marker" in state:
                n_sensors = int(state["0._n_sensors_marker"].item())
            else:
                n_sensors = ModelConfigC().n_sensors

            encoder_num_layers = sum(
                1 for k in state
                if k.startswith("0.layers.layers.") and k.endswith(".norm1.weight")
            )
        else:
            # DenseNet path: n_sensors only matters if spatial layer is on;
            # time_len/embed_dim come from defaults.
            n_sensors = (
                state["2.pos_enc.pos_emb"].shape[0] if use_spatial
                else ModelConfigC().n_sensors
            )
            time_len  = ModelConfigC().time_len
            embed_dim = ModelConfigC().embed_dim
            encoder_num_layers = ModelConfigC().encoder_num_layers
            tokenizer_type = "linear"  # not used for densenet; default for dataclass

        num_spatial_layers = (
            sum(1 for k in state if k.startswith("2.spatial.layers.layers.") and k.endswith(".norm1.weight"))
            if use_spatial else 1
        )
        # Decoder depth auto-detect (MidnC layers live under "2.decoder.layers.N.*").
        num_decoder_layers = sum(
            1 for k in state
            if k.startswith("2.decoder.layers.") and k.endswith(".norm1.weight")
        ) or ModelConfigC().num_decoder_layers
        model_cfg = ModelConfigC(
            num_slots=num_slots,
            num_locations=num_locations,
            n_sensors=n_sensors,
            time_len=time_len,
            embed_dim=embed_dim,
            num_decoder_layers=num_decoder_layers,
            use_spatial_layer=use_spatial,
            num_spatial_layers=num_spatial_layers,
            use_fault_head=use_fault,
            use_structural_bias=use_struct,
            use_l2_norm_memory=use_l2,
            use_mil_readout=use_mil,
            use_final_decoder_norm=use_final_ln,
            encoder_type=encoder_type,
            encoder_num_layers=encoder_num_layers,
            encoder_pos_emb=encoder_pos_emb if is_itransformer else "learned",
            tokenizer_type=tokenizer_type,
            fault_head_location=fault_head_location,
        )
    else:
        use_struct = model_cfg.use_structural_bias
    structural_affinity = state["2.R_bias"] if use_struct else None
    model = build_model(model_cfg, structural_affinity=structural_affinity)
    # Backwards-compat: pre-pos_emb-variant iT checkpoints lack the _n_sensors_marker
    # buffer. Inject it from the already-inferred n_sensors so strict load succeeds.
    if "0.tokenize.weight" in state and "0._n_sensors_marker" not in state:
        state["0._n_sensors_marker"] = torch.tensor(model_cfg.n_sensors, dtype=torch.long)
    model.load_state_dict(state)
    if device is not None:
        model = model.to(device)
    return model
