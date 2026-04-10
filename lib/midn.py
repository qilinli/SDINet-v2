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


class MidnDR(nn.Module):
    """
    Direct Regression head (Approach DR).

    Predicts a damage severity value ∈ [0, 1] for each of the L structural
    locations in a single forward pass, supervised directly by MSE against the
    normalised ground truth (y_norm + 1) / 2.

    Architecture mirrors the MIL aggregation in Midn but with a single
    prediction branch — no presence/severity decomposition.  This is the
    conceptual baseline against which the Approach-C slot predictor is
    compared.

    The core limitation this exposes: the MSE gradient pushes predictions at
    all L locations toward their target simultaneously, including the K_max-K
    undamaged ones whose target is 0.  Any feature noise produces small
    non-zero outputs everywhere, requiring a post-hoc threshold to identify
    damaged locations.

    Args:
        in_channels:         Feature dimension from the neck (embed_dim).
        num_locations:       Number of structural locations L (default 70).
        importance_dropout:  Fraction of importance logits masked during training.
        temperature:         Scale on importance logits during training.
        val_temperature:     Scale during evaluation.
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
        self.temperature = temperature
        self.val_temperature = val_temperature or temperature
        self.importance_dropout_p = importance_dropout

        self.importance_branch = nn.Conv1d(in_channels, num_locations, 1)
        self.pred_branch       = nn.Conv1d(in_channels, num_locations, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, S) per-sensor neck features.

        Returns:
            pred: (B, L) damage severity ∈ [0, 1]
        """
        temp = self.temperature if self.training else self.val_temperature
        imp_logits = self.importance_branch(x) * temp          # (B, L, S)
        if self.training and self.importance_dropout_p > 0.0:
            imp_logits = importance_dropout(imp_logits, self.importance_dropout_p)
        imp = imp_logits.softmax(-1)                           # (B, L, S)

        pred_raw = torch.sigmoid(self.pred_branch(x))         # (B, L, S) ∈ [0, 1]
        return (pred_raw * imp).sum(-1)                        # (B, L)


class PlainDR(nn.Module):
    """
    Plain regression baseline (Approach B).

    Simplest possible head: mean-pool over sensors, linear projection to L
    locations, sigmoid activation.  No learned sensor attention — isolates
    the contribution of the MIL importance-weighted aggregation in MidnDR.

    Output shape and value range are identical to MidnDR ((B, L) ∈ [0, 1]),
    so the entire DR training, calibration, and evaluation pipeline is reused.
    """

    def __init__(self, in_channels: int, num_locations: int = 70) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, num_locations)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, S) per-sensor neck features.
        Returns:
            pred: (B, L) damage severity ∈ [0, 1]
        """
        return torch.sigmoid(self.linear(x.mean(dim=-1)))


class SensorPositionalEncoding(nn.Module):
    """
    Learned per-sensor positional embedding added to sensor feature memory.

    Each of the S sensor positions gets a learnable embed_dim vector.
    Initialised small (std=0.02) so it perturbs rather than dominates early in training.

    Args:
        n_sensors:  Number of sensors S (dataset-specific: 30 for Qatar, 65 for 7-story).
        embed_dim:  Feature dimension (must match the neck output).
    """

    def __init__(self, n_sensors: int, embed_dim: int) -> None:
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(n_sensors, embed_dim) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, S, embed_dim) sensor features from neck.
        Returns:
            (B, S, embed_dim) with positional embedding added.
        """
        return x + self.pos_emb  # broadcast over batch


class SensorSpatialLayer(nn.Module):
    """
    Cross-sensor self-attention layer for spatial coherence reasoning.

    Placed between the neck output and the slot decoder. Each sensor attends
    to all other sensors, updating its representation based on what its
    neighbours (in feature space) are reading. Spatially incoherent sensors
    (faults) develop distinctive features after this layer, allowing the
    damage slots to naturally down-weight them via cross-attention.

    Args:
        embed_dim:   Feature dimension (must match neck output).
        nhead:       Number of attention heads (embed_dim must be divisible).
        num_layers:  TransformerEncoder depth (1–2 typically sufficient).
    """

    def __init__(self, embed_dim: int, nhead: int, num_layers: int) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,   # pre-norm for stability
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, S, embed_dim) sensor features (positional embeddings already added).
        Returns:
            (B, S, embed_dim) spatially-aware sensor features.
        """
        return self.layers(x)


class MidnC(nn.Module):
    """
    DETR-style damage slot predictor (Approach C).

    K_max learnable slot queries cross-attend over per-sensor neck features via
    a standard TransformerDecoder.  Each slot independently predicts:

    - **location logits** over L+1 classes (L structural locations + ∅ no-object)
    - **severity** ∈ [0, 1] (sigmoid-activated)

    Unlike Approach B's 70 parallel per-location predictions, Approach C treats
    damage detection as a *set prediction* problem: the model must decide both
    *whether* and *where* each slot's damage is, jointly, through attention.

    At inference:
        ``is_obj[k] = 1 − softmax(loc_logits[k])[∅]``  — slot activity score
        ``loc_probs[k] = softmax(loc_logits[k])[:L]``  — location distribution
        ``damage_map[l] = Σ_k  is_obj[k] * severity[k] * loc_probs[k, l]``

    This map is in [0, 1] and directly comparable to distributed_map_v1 and
    distributed_map_dr via the shared ``map_mse`` metric.

    Args:
        embed_dim:           Feature dimension from the neck (default 768).
        num_slots:           Maximum simultaneous damages K_max (default 4).
        num_locations:       Structural locations L (default 70).
        num_decoder_layers:  Transformer decoder depth (default 2).
        nhead:               Attention heads (default 8; embed_dim must be divisible).
        n_sensors:           Number of sensors S — required when use_spatial_layer
                             or use_fault_head is True (default 0).
        use_spatial_layer:   Enable SensorPositionalEncoding + SensorSpatialLayer
                             before the slot decoder (default False).
        num_spatial_layers:  Depth of the SensorSpatialLayer encoder (default 1).
        spatial_nhead:       Attention heads in SensorSpatialLayer (default 8).
        use_fault_head:      Enable per-sensor binary fault classifier (default False).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_slots: int = 4,
        num_locations: int = 70,
        num_decoder_layers: int = 2,
        nhead: int = 8,
        n_sensors: int = 0,
        use_spatial_layer: bool = False,
        num_spatial_layers: int = 1,
        spatial_nhead: int = 8,
        use_fault_head: bool = False,
        structural_affinity: "Tensor | None" = None,
    ) -> None:
        super().__init__()
        self.num_slots         = num_slots
        self.num_locations     = num_locations
        self.no_obj_idx        = num_locations
        self.use_spatial_layer = use_spatial_layer
        self.use_fault_head    = use_fault_head

        # Learnable slot queries — one per potential damage
        self.queries = nn.Parameter(torch.randn(num_slots, embed_dim))

        # Transformer decoder: slots attend over sensor-feature memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,          # pre-norm for training stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Per-slot prediction heads
        self.loc_head = nn.Linear(embed_dim, num_locations + 1)   # L + ∅
        self.sev_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

        # Optional: sensor spatial reasoning (positional encoding + self-attention)
        if use_spatial_layer:
            self.pos_enc = SensorPositionalEncoding(n_sensors, embed_dim)
            self.spatial = SensorSpatialLayer(embed_dim, spatial_nhead, num_spatial_layers)

        # Optional: per-sensor binary fault classifier
        if use_fault_head:
            self.fault_head = nn.Linear(embed_dim, 1)

        # Optional: structural location-sensor affinity bias in slot cross-attention
        # R_bias[l, i]: affinity of sensor i for structural location l.
        # At decoder layer k≥1, bias = softmax(loc_logits) @ R_bias is added to
        # cross-attention logits, directing each slot toward structurally relevant sensors.
        self.use_structural_bias = structural_affinity is not None
        if self.use_structural_bias:
            self.R_bias = nn.Parameter(structural_affinity.clone())  # (L+1, S), learnable
            self._nhead = nhead

    def forward(self, x: Tensor) -> tuple:
        """
        Args:
            x: (B, embed_dim, S) per-sensor features from the neck.

        Returns:
            loc_logits:  (B, K_max, L+1)  raw location logits (∅ = index L)
            severity:    (B, K_max)       sigmoid-activated severity ∈ [0, 1]
            fault_prob:  (B, S)           per-sensor fault probability ∈ [0, 1],
                                          or None when use_fault_head=False
        """
        B = x.size(0)

        # Sensor features as transformer memory: (B, S, embed_dim)
        memory = x.permute(0, 2, 1)

        # Optional: add positional embeddings and cross-sensor self-attention
        if self.use_spatial_layer:
            memory = self.pos_enc(memory)
            memory = self.spatial(memory)

        # Slot queries → batch: (B, K_max, embed_dim)
        tgt = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attend: slots gather information from sensor memory.
        # With structural bias: layer 0 is unbiased (slot queries haven't seen data yet);
        # layers ≥1 add a dynamic location-conditioned bias to cross-attention logits so
        # each slot attends more to sensors near its current predicted location.
        if self.use_structural_bias:
            S_mem = memory.size(1)
            bias_compatible = (S_mem == self.R_bias.size(-1))
            for i, layer in enumerate(self.decoder.layers):
                if i == 0 or not bias_compatible:
                    tgt = layer(tgt, memory)
                else:
                    loc_probs = self.loc_head(tgt).softmax(-1)          # (B, K, L+1)
                    bias = loc_probs @ self.R_bias                       # (B, K, S)
                    bias = bias.unsqueeze(1).expand(-1, self._nhead, -1, -1)
                    bias = bias.reshape(B * self._nhead, -1, S_mem)     # (B*H, K, S)
                    tgt = layer(tgt, memory, memory_mask=bias)
            slots = tgt
        else:
            slots = self.decoder(tgt, memory)                           # (B, K_max, embed_dim)

        loc_logits = self.loc_head(slots)             # (B, K_max, L+1)
        severity   = self.sev_head(slots).squeeze(-1) # (B, K_max)

        fault_prob = None
        if self.use_fault_head:
            fault_prob = torch.sigmoid(self.fault_head(memory)).squeeze(-1)  # (B, S)

        return loc_logits, severity, fault_prob
