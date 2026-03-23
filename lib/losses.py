from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Label-space constants — must match data_safetensors.target_preprocess:
#   y_norm = raw_damage / 0.15 - 1
#   Undamaged → -1.0,  max-damaged → +1.0
# ---------------------------------------------------------------------------
UNDAMAGED_NORM: float = -1.0
PRESENCE_THRESHOLD: float = UNDAMAGED_NORM + 1e-4   # any non-zero damage


class PresenceSeverityLoss(nn.Module):
    """
    Combined loss for the B-head (presence + severity decomposition).

    Two components:
    1. Weighted BCE on presence logits — which locations have damage?
       ``pos_weight`` compensates heavy class imbalance (K << L locations damaged).
    2. Severity MSE — how much damage, *only at truly damaged locations*?
       Restricting to damaged locations avoids trivially-easy zero targets
       swamping the gradient and forcing the severity branch to learn.

    Label conventions (normalized space, ``y_norm = raw / 0.15 - 1``):
        - Presence target: 1 where ``y_norm > presence_threshold`` else 0
        - Severity target: ``(y_norm + 1) / 2 ∈ [0, 1]``

    Args:
        pos_weight: BCE positive-class weight.
            Rule of thumb: ``(L - K) / K`` where K = mean damages, L = locations.
            Default 69 ≈ (70 - 1) / 1 for single-damage frames.
        severity_weight: Relative scale of severity MSE versus presence BCE.
        presence_threshold: Normalized label threshold separating undamaged (-1)
            from damaged. Default ``-1 + 1e-4`` catches floating-point artefacts.
    """

    def __init__(
        self,
        pos_weight: float = 69.0,
        severity_weight: float = 1.0,
        presence_threshold: float = PRESENCE_THRESHOLD,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.severity_weight = severity_weight
        self.presence_threshold = presence_threshold

    def forward(
        self,
        presence_logits: Tensor,  # (B, L) raw logits
        severity: Tensor,         # (B, L) sigmoid output ∈ [0, 1]
        y_norm: Tensor,           # (B, L) normalized labels ∈ [-1, 1]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            total_loss: combined loss for backprop
            bce_loss:   presence BCE (detached scalar, for logging)
            sev_loss:   severity MSE (detached scalar, for logging)
        """
        y_presence = (y_norm > self.presence_threshold).float()  # (B, L) binary
        y_severity = (y_norm + 1.0) / 2.0                        # (B, L) ∈ [0, 1]

        bce_loss = F.binary_cross_entropy_with_logits(
            presence_logits,
            y_presence,
            pos_weight=self.pos_weight.to(presence_logits.device),
        )

        damage_mask = y_presence.bool()
        if damage_mask.any():
            sev_loss = F.mse_loss(severity[damage_mask], y_severity[damage_mask])
        else:
            # No damaged locations in this batch (e.g. undamaged-only splits).
            # Multiply by 0 to preserve the computation graph while contributing
            # no gradient — avoids a detached zero that would upset autocast.
            sev_loss = (severity * 0.0).sum()

        total = bce_loss + self.severity_weight * sev_loss
        return total, bce_loss, sev_loss
