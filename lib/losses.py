from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Label-space constants — must match data_safetensors.target_preprocess:
#   y_norm = raw_damage / 0.15 - 1
#   Undamaged → -1.0,  max-damaged → +1.0
# ---------------------------------------------------------------------------
UNDAMAGED_NORM: float = -1.0
PRESENCE_THRESHOLD: float = UNDAMAGED_NORM + 1e-4   # any non-zero damage


# ---------------------------------------------------------------------------
# Approach-C: DETR-style set criterion with Hungarian matching
# ---------------------------------------------------------------------------

class SetCriterion(nn.Module):
    """
    DETR-style set prediction loss for the C-head (MidnC).

    Each sample has K_max predicted slots, each slot outputting:
    - location logits over L+1 classes (L real locations + ∅ no-object)
    - severity ∈ [0, 1]

    Training finds the optimal bipartite assignment between predicted slots and
    ground-truth damages (Hungarian algorithm), then applies:
    1. Location cross-entropy over all K_max slots (unmatched → ∅ class).
    2. Severity MSE restricted to matched (non-∅) slots.

    Because locations are a finite discrete set (no continuous coordinate
    regression), the matching cost is simply:
        cost[k, g] = −log P(slot k → gt_loc[g]) + sev_weight * |sev_k − gt_sev_g|

    Args:
        num_locations:  Number of structural locations L (default 70).
        no_obj_weight:  Weight for the ∅ (no-object) class in the CE loss.
                        Should be < 1 since most slots are ∅ (K_max >> K_mean).
                        Default 0.1 follows the DETR paper.
        loc_weight:     Scale of the location CE term in the total loss.
        sev_weight:     Scale of severity MSE in the total loss, and relative
                        weight of severity cost in the Hungarian matching.
    """

    def __init__(
        self,
        num_locations: int = 70,
        no_obj_weight: float = 0.1,
        loc_weight: float = 1.0,
        sev_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_locations = num_locations
        self.no_obj_idx = num_locations          # ∅ class is the last index
        self.loc_weight = loc_weight
        self.sev_weight = sev_weight

        # Per-class weight vector for location CE: 1.0 for real locations, down-weight ∅
        ce_weight = torch.ones(num_locations + 1)
        ce_weight[-1] = no_obj_weight
        self.register_buffer("ce_weight", ce_weight)

    @torch.no_grad()
    def _match(
        self,
        loc_logits_b: Tensor,   # (K_max, L+1)
        severity_b: Tensor,     # (K_max,)
        gt_locs: Tensor,        # (K_gt,) long
        gt_sevs: Tensor,        # (K_gt,) float ∈ [0, 1]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Hungarian matching for a single sample.

        Returns:
            loc_targets  (K_max,) long  — matched GT location or no_obj_idx
            sev_targets  (K_max,) float — matched GT severity (0 for ∅ slots)
            sev_mask     (K_max,) bool  — True for matched (non-∅) slots
        """
        K_max = loc_logits_b.size(0)
        K_gt  = gt_locs.size(0)
        dev   = loc_logits_b.device

        loc_targets = torch.full((K_max,), self.no_obj_idx, dtype=torch.long, device=dev)
        sev_targets = torch.zeros(K_max, device=dev)
        sev_mask    = torch.zeros(K_max, dtype=torch.bool, device=dev)

        if K_gt == 0:
            return loc_targets, sev_targets, sev_mask

        # Location cost: -log P(slot k predicts gt location g)
        log_probs = F.log_softmax(loc_logits_b, dim=-1)          # (K_max, L+1)
        loc_cost  = -log_probs[:, gt_locs]                        # (K_max, K_gt)

        # Severity cost: L1 distance between predicted and GT severity
        sev_cost = (severity_b.unsqueeze(1) - gt_sevs.unsqueeze(0)).abs()  # (K_max, K_gt)

        cost = (loc_cost + self.sev_weight * sev_cost).cpu().numpy()

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            loc_targets[r] = gt_locs[c]
            sev_targets[r] = gt_sevs[c]
            sev_mask[r]    = True

        return loc_targets, sev_targets, sev_mask

    def forward(
        self,
        loc_logits: Tensor,   # (B, K_max, L+1)
        severity: Tensor,     # (B, K_max)
        y_norm: Tensor,       # (B, L) normalized labels ∈ [-1, 1]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            total_loss: combined loss for backprop
            loc_loss:   location CE (detached scalar, for logging)
            sev_loss:   severity MSE (detached scalar, for logging)
        """
        B   = y_norm.size(0)
        dev = loc_logits.device

        y_pres = y_norm > PRESENCE_THRESHOLD          # (B, L) bool
        y_sev  = (y_norm + 1.0) / 2.0                 # (B, L) ∈ [0, 1]

        all_loc_tgt: list[Tensor] = []
        all_sev_tgt: list[Tensor] = []
        all_sev_msk: list[Tensor] = []

        for b in range(B):
            gt_locs = y_pres[b].nonzero(as_tuple=True)[0]   # (K_gt,) long
            gt_sevs = y_sev[b, gt_locs]                      # (K_gt,) float
            loc_t, sev_t, sev_m = self._match(
                loc_logits[b].detach(), severity[b].detach(), gt_locs, gt_sevs
            )
            all_loc_tgt.append(loc_t)
            all_sev_tgt.append(sev_t)
            all_sev_msk.append(sev_m)

        loc_targets = torch.stack(all_loc_tgt)   # (B, K_max) long
        sev_targets = torch.stack(all_sev_tgt)   # (B, K_max)
        sev_masks   = torch.stack(all_sev_msk)   # (B, K_max) bool

        # Location CE: all K_max slots, with class-reweighting for ∅
        loc_loss = F.cross_entropy(
            loc_logits.reshape(-1, self.num_locations + 1),
            loc_targets.reshape(-1),
            weight=self.ce_weight.to(dev),
        )

        # Severity MSE: matched slots only
        if sev_masks.any():
            sev_loss = F.mse_loss(severity[sev_masks], sev_targets[sev_masks])
        else:
            sev_loss = (severity * 0.0).sum()

        total = self.loc_weight * loc_loss + self.sev_weight * sev_loss
        return total, loc_loss, sev_loss
