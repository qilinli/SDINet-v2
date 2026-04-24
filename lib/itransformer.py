"""
lib/itransformer.py — iTransformer-style encoder for Approach C.

Drop-in replacement for the DenseNet backbone + MIL neck that MidnC currently
sees.  Unlike the DenseNet pipeline — which treats the input as an image
``(B, 1, T, S)`` and convolves jointly over time and sensor dimensions — this
encoder projects each sensor's full time series into a single token and then
mixes tokens via self-attention over the sensor dimension.

    (B, 1, T, S)                 raw window
        ↓  permute to (B, S, T)
        ↓  Linear(T, embed_dim)  — per-sensor tokenisation (weights shared across sensors)
        ↓  LayerNorm(embed_dim)
        ↓  + positional embedding (learned / none / RoPE)
        ↓  TransformerEncoder(depth=2)  — self-attention across S
        ↓  transpose
    (B, embed_dim, S)            decoder-compatible memory

Shape contract matches the output of ``SDIDenseNet → _build_neck`` so that
``MidnC`` needs no changes.

Positional-embedding variants (`pos_emb_type`):
  - "learned"  (default): additive learned `nn.Parameter(S, D)`.
  - "none"             : no sensor position encoding — attention is permutation
                         invariant.  Useful for sensor-count polymorphism and
                         testing whether location information matters at all.
  - "rope"             : Rotary positional embedding applied to Q/K inside each
                         self-attention layer.  Requires the custom
                         :class:`RoPEEncoderLayer` below because ``nn.MultiheadAttention``
                         does not expose Q/K between projection and scaled-dot-product.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


POS_EMB_TYPES: tuple[str, ...] = ("learned", "none", "rope")
TOKENIZER_TYPES: tuple[str, ...] = ("linear", "multiscale_conv")
FAULT_HEAD_LOCATIONS: tuple[str, ...] = ("decoder", "encoder")


# ---------------------------------------------------------------------------
# Multi-scale dilated-conv tokenizer (Phase B: preserves sub-window temporal
# structure that Linear(T, D) collapses into one shot)
# ---------------------------------------------------------------------------

class MultiScaleConvTokenizer(nn.Module):
    """Per-sensor tokenizer with parallel dilated-conv branches.

    Each of the S sensors is tokenised independently (weights shared across
    sensors). Branches at different dilations capture short → long timescales
    within the window:
        dilation=1, k=7  →  effective RF 7  samples  (~7 ms at 1 kHz)
        dilation=2, k=7  →  effective RF 13
        dilation=4, k=7  →  effective RF 25
        dilation=8, k=7  →  effective RF 49

    Default ``stride=4`` reduces per-branch feature-map length from T to ~T/4,
    cutting peak activation memory 4× vs stride=1 (important for TITAN V training).
    Dilation still controls receptive field independently.

    Parameter count at defaults: ~596k (6k in the 4 conv branches, ~590k in the
    final Linear(4*branch_dim, embed_dim) projection). Comparable to the
    Linear(T=500, D=768) baseline (~384k) but with learned temporal locality.
    """

    def __init__(
        self,
        time_len: int,
        embed_dim: int,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 7,
        stride: int = 4,
        branch_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.time_len = time_len
        self.embed_dim = embed_dim
        branch_dim = branch_dim if branch_dim is not None else max(embed_dim // len(dilations), 64)
        # Stride reduces the intermediate feature-map length from T to ~T/stride,
        # cutting peak activation memory (and backward-pass saved activations) by
        # the stride factor. Dilation controls receptive field independently.
        self.branches = nn.ModuleList([
            nn.Conv1d(
                1, branch_dim, kernel_size=kernel_size,
                dilation=d, stride=stride,
                padding=((kernel_size - 1) * d) // 2,
            )
            for d in dilations
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(branch_dim * len(dilations), embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, S, T) raw per-sensor time series.
        Returns:
            (B, S, embed_dim) sensor tokens.
        """
        B, S, T = x.shape
        x = x.reshape(B * S, 1, T)
        feats = [F.relu(b(x)) for b in self.branches]        # each (B*S, branch_dim, T)
        feats = [self.pool(f).squeeze(-1) for f in feats]    # each (B*S, branch_dim)
        feats = torch.cat(feats, dim=-1)                      # (B*S, total_dim)
        tokens = self.proj(feats).view(B, S, -1)              # (B, S, embed_dim)
        return tokens


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin tables for Rotary Position Embedding.

    Returns (cos, sin) each of shape (1, 1, seq_len, head_dim), pre-shaped for
    (B, H, S, D_h) attention tensors.  Uses the "half-rotate" convention used
    by e.g. Llama / RoFormer — pairs (d_i, d_{i+D/2}) not (d_{2i}, d_{2i+1}).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum("s,h->sh", positions, inv_freq)      # (S, D/2)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)       # (S, D)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)       # (S, D)
    return cos[None, None], sin[None, None]                   # (1, 1, S, D)


def _rotate_half(x: Tensor) -> Tensor:
    """Half-rotate: (x1, x2) -> (-x2, x1) along the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to a tensor of shape (..., S, D).  cos, sin broadcast over it."""
    return (q * cos) + (_rotate_half(q) * sin)


class RoPEMultiheadAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding on Q and K.

    Packaged to match the minimal surface nn.TransformerEncoderLayer needs:
    takes (B, S, D), returns (B, S, D).  cos/sin are passed in per-forward so
    a single RoPE cache can be shared across layers.
    """

    def __init__(self, embed_dim: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by nhead={nhead}")
        self.embed_dim = embed_dim
        self.nhead     = nhead
        self.head_dim  = embed_dim // nhead
        self.dropout   = dropout

        self.in_proj  = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, S, D = x.shape
        qkv = self.in_proj(x)                                  # (B, S, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)  # (B, H, S, D_h)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
        )                                                       # (B, H, S, D_h)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class RoPEEncoderLayer(nn.Module):
    """Pre-norm TransformerEncoderLayer with RoPE self-attention.

    Mirrors ``nn.TransformerEncoderLayer(norm_first=True)`` so the overall
    iTransformer stack has identical residual / FFN / normalization structure
    to the learned-PE baseline — only the attention positional handling differs.
    """

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(embed_dim, nhead, dropout)
        self.linear1   = nn.Linear(embed_dim, dim_feedforward)
        self.linear2   = nn.Linear(dim_feedforward, embed_dim)
        self.norm1     = nn.LayerNorm(embed_dim)
        self.norm2     = nn.LayerNorm(embed_dim)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.dropout1(self.self_attn(self.norm1(x), cos, sin))
        x = x + self.dropout2(self.linear2(self.dropout_ff(F.relu(self.linear1(self.norm2(x))))))
        return x


class RoPEEncoder(nn.Module):
    """Stack of :class:`RoPEEncoderLayer` plus a final LayerNorm (pre-norm hygiene)."""

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(embed_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------


class iTransformerEncoder(nn.Module):
    """Sensor-token iTransformer encoder.

    One token per sensor (shape ``embed_dim``), self-attention across S tokens
    to mix cross-sensor information before the slot decoder.

    Args:
        n_sensors:   Number of sensors S.
        time_len:    Time window length T (post-preprocessing).
        embed_dim:   Token dimension (must match ``MidnC.embed_dim`` and be
                     divisible by ``nhead``).
        num_layers:  TransformerEncoder depth over the sensor dimension.
        nhead:       Attention heads.
        dim_feedforward: FFN inner dim (default 4*embed_dim, matches DETR).
        dropout:     Dropout in the transformer layers.
        pos_emb_type: one of :data:`POS_EMB_TYPES`.
    """

    def __init__(
        self,
        n_sensors: int,
        time_len: int,
        embed_dim: int = 768,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        pos_emb_type: str = "learned",
        tokenizer_type: str = "linear",
        fault_head_location: str = "decoder",
    ) -> None:
        super().__init__()
        if pos_emb_type not in POS_EMB_TYPES:
            raise ValueError(
                f"pos_emb_type must be one of {POS_EMB_TYPES}, got {pos_emb_type!r}"
            )
        if tokenizer_type not in TOKENIZER_TYPES:
            raise ValueError(
                f"tokenizer_type must be one of {TOKENIZER_TYPES}, got {tokenizer_type!r}"
            )
        if fault_head_location not in FAULT_HEAD_LOCATIONS:
            raise ValueError(
                f"fault_head_location must be one of {FAULT_HEAD_LOCATIONS}, "
                f"got {fault_head_location!r}"
            )
        self.n_sensors = n_sensors
        self.time_len  = time_len
        self.embed_dim = embed_dim
        self.pos_emb_type = pos_emb_type
        self.tokenizer_type = tokenizer_type
        self.fault_head_location = fault_head_location

        # Per-sensor tokenisation: shared weights across sensors.
        # "linear"          → Linear(T, D) — single shot, no temporal structure.
        # "multiscale_conv" → parallel dilated Conv1d branches + AAP + Linear.
        if tokenizer_type == "linear":
            self.tokenize = nn.Linear(time_len, embed_dim)
        else:
            self.tokenize = MultiScaleConvTokenizer(time_len, embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)

        # Optional pre-encoder fault head (Phase B2): reads raw per-sensor tokens
        # before any cross-sensor attention mixing, preserving per-sensor fault
        # identity — especially important at high fault ratios where post-encoder
        # memory carries contamination from faulted neighbours.
        if fault_head_location == "encoder":
            self.fault_head = nn.Linear(embed_dim, 1)

        # Persistent marker so checkpoints encode n_sensors even when
        # pos_emb_type != "learned" (no S-shaped parameter survives otherwise).
        self.register_buffer(
            "_n_sensors_marker", torch.tensor(n_sensors, dtype=torch.long), persistent=True
        )

        dim_ff = dim_feedforward or embed_dim * 4

        if pos_emb_type == "rope":
            head_dim = embed_dim // nhead
            if head_dim % 2 != 0:
                raise ValueError(
                    f"RoPE requires even head_dim; got embed_dim={embed_dim}, nhead={nhead} "
                    f"→ head_dim={head_dim}"
                )
            # One RoPE cache shared across all layers (buffers move with .to(device)).
            cos, sin = _build_rope_cache(n_sensors, head_dim)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self.layers = RoPEEncoder(embed_dim, nhead, num_layers, dim_ff, dropout)
        else:
            if pos_emb_type == "learned":
                # Learnable per-sensor positional embedding — small init so it
                # perturbs rather than dominates early.
                self.pos_emb = nn.Parameter(torch.randn(n_sensors, embed_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.layers = nn.TransformerEncoder(
                encoder_layer, num_layers, norm=nn.LayerNorm(embed_dim)
            )

    def forward(self, x: Tensor) -> "tuple[Tensor, Tensor | None]":
        """
        Args:
            x: (B, 1, T, S) input window (post-preprocessing).

        Returns:
            (sensor_features, fault_prob):
              sensor_features: (B, embed_dim, S) per-sensor tokens for the decoder.
              fault_prob:      (B, S) per-sensor fault probability ∈ [0, 1] when
                               ``fault_head_location="encoder"``, else ``None``.
                               Downstream MidnC uses the encoder-provided value if
                               present and skips its own fault head.
        """
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"iTransformerEncoder expects (B, 1, T, S), got {tuple(x.shape)}")
        if x.size(2) != self.time_len:
            raise ValueError(
                f"iTransformerEncoder initialised with time_len={self.time_len}, "
                f"got T={x.size(2)}"
            )
        if x.size(3) != self.n_sensors:
            raise ValueError(
                f"iTransformerEncoder initialised with n_sensors={self.n_sensors}, "
                f"got S={x.size(3)}"
            )

        x = x.squeeze(1).transpose(1, 2)       # (B, S, T)
        tokens = self.tokenize(x)              # (B, S, embed_dim)
        tokens = self.token_norm(tokens)

        # Pre-encoder fault head (Phase B2): reads per-sensor tokens before
        # cross-sensor attention mixing.
        fault_prob: "Tensor | None" = None
        if self.fault_head_location == "encoder":
            fault_prob = torch.sigmoid(self.fault_head(tokens)).squeeze(-1)   # (B, S)

        if self.pos_emb_type == "learned":
            tokens = tokens + self.pos_emb     # broadcast over batch
        if self.pos_emb_type == "rope":
            tokens = self.layers(tokens, self.rope_cos, self.rope_sin)
        else:
            tokens = self.layers(tokens)       # (B, S, embed_dim)
        sensor_features = tokens.transpose(1, 2).contiguous()   # (B, embed_dim, S)
        return sensor_features, fault_prob
