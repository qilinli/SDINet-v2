from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import trange

from lib.metrics import (
    PRESENCE_NORM_THRESH,
    distributed_map_b,
    distributed_map_v1,
    f1_from_counts,
    map_mse,
    presence_f1_stats,
)

# Training / evaluation defaults
DEFAULT_DROP_NUM = 10
DEFAULT_DROP_DEN = 65
DEFAULT_DROP_RATE = DEFAULT_DROP_NUM / DEFAULT_DROP_DEN

SUBSET_RNG_SEED = 42
DEFAULT_VAL_SUBSET_COUNT = 51
DEFAULT_VAL_NUM_SENSORS = 65
DEFAULT_VAL_SUBSET_SIZE = 10
EPS = 1e-12
MIXED_PRECISION_MODE = "no"

# Optimizer / scheduler defaults
DEFAULT_BASE_LR = 5e-4
DEFAULT_WEIGHT_DECAY = 1.0e-2
DEFAULT_ADAMW_BETAS = (0.9, 0.999)


@torch.no_grad()
def randomise_bag_size(
    x: torch.Tensor, drop_rate: float = DEFAULT_DROP_RATE
) -> torch.Tensor:
    """Randomly keep a non-empty subset of sensors from the last axis."""
    if x.size(-1) == 0:
        raise ValueError("Input has no sensor dimension to sample from")
    if drop_rate <= 0.0:
        return x
    if drop_rate >= 1.0:
        # Keep exactly one sensor to avoid empty-tensor forward passes.
        keep_idx = torch.randint(x.size(-1), (1,), device=x.device)
        return x.index_select(-1, keep_idx)

    while not (mask := torch.rand(x.size(-1), device=x.device) < drop_rate).any():
        pass
    return x[..., mask]


def train_one_epoch(
    model,
    opt,
    sched,
    train_dl: DataLoader,
    accel: Accelerator,
    ema=None,
) -> float:
    model.train()
    accum = 0.0
    for x, y in train_dl:
        y_dmg, y_loc = y.max(-1, keepdim=True)
        y_loc = y_loc[:, 0]
        opt.zero_grad()
        with accel.autocast():
            x = randomise_bag_size(x)
            y_hat_dmg, y_hat_loc = model(x)
            dmg_loss = F.mse_loss(y_hat_dmg, y_dmg)
            loc_loss = F.cross_entropy(y_hat_loc, y_loc)
            loss = dmg_loss + loc_loss
            if ema is not None:
                ema.update_parameters(model)
        accel.backward(loss)
        opt.step()
        sched.step()
        accum += loss.item()
    return accum / len(train_dl)


@lru_cache
def gen_sensor_subsets(
    num_subsets: int, subset_size: int, total_sensors: int
) -> torch.Tensor:
    '''
    Generate deterministic, unique random sensor subsets for validation.

    Each generated subset is formed by taking the first `subset_size` indices
    from a random permutation of `total_sensors`. The selected prefixes are
    enforced to be unique across all `num_subsets`.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(subset_size, num_subsets)` containing sensor indices.
    '''
    if subset_size > total_sensors:
        raise ValueError(
            f"subset_size ({subset_size}) must be <= total_sensors ({total_sensors})"
        )
    # Fixed seed keeps validation deterministic across runs.
    rng = torch.Generator().manual_seed(SUBSET_RNG_SEED)
    out = torch.empty((num_subsets, total_sensors), dtype=torch.long)
    for i in range(num_subsets):
        # Build a random sensor ordering, then keep the first `subset_size` sensors.
        torch.randperm(total_sensors, out=out[i], generator=rng)
        # Ensure each selected subset prefix is unique among previously built subsets.
        while (out[i : i + 1, :subset_size] == out[:i, :subset_size]).all(1).any():
            torch.randperm(total_sensors, out=out[i], generator=rng)
    # Return shape: (subset_size, num_subsets) to match downstream indexing layout.
    return out[:, :subset_size].T


@torch.inference_mode()
def val_one_epoch(
    model, val_dl: DataLoader, subset_size: int = DEFAULT_VAL_SUBSET_SIZE
) -> tuple[float, float, float]:
    '''
    Validate one epoch under sampled sensor-failure subsets.

    The routine evaluates loss, distributed-map MSE, and location accuracy on a
    reduced set of `DEFAULT_VAL_SUBSET_COUNT` randomly generated sensor subsets.
    This keeps validation lightweight while preserving a stable convergence
    signal across epochs.
    '''
    model.eval()
    state = deepcopy(model.state_dict())
    sensor_subsets = gen_sensor_subsets(
        DEFAULT_VAL_SUBSET_COUNT,
        subset_size=subset_size,
        total_sensors=DEFAULT_VAL_NUM_SENSORS,
    )

    total_losses = torch.zeros((sensor_subsets.size(0),))
    total_mse    = torch.zeros((sensor_subsets.size(0),))
    tkr_hits = tkr_total = 0

    for x, y in val_dl:
        y_dmg, y_loc = y.max(-1, keepdim=True)
        y_loc = y_loc[:, 0]

        y_hat_dmg, i_dmg, y_hat_loc, i_loc = model[2](model[:2](x.float()), False)

        # ---- full-sensor predictions for top_k_recall ----------------------
        # i_dmg / i_loc are softmax-normalised over all S sensors already
        dmg_pred_full = (y_hat_dmg * i_dmg).sum(-1)           # (B, 1)
        loc_pred_full = (y_hat_loc * i_loc).sum(-1)            # (B, L)
        loc_probs     = loc_pred_full.softmax(-1)              # (B, L)
        y_pres        = y > PRESENCE_NORM_THRESH               # (B, L) bool
        k_true        = y_pres.sum(-1)                         # (B,)
        for b in range(y.size(0)):
            K = int(k_true[b].item())
            if K == 0:
                continue
            topk_mask = torch.zeros(y.size(-1), dtype=torch.bool, device=y.device)
            topk_mask[loc_probs[b].topk(K).indices] = True
            tkr_hits  += int((topk_mask & y_pres[b]).sum())
            tkr_total += K

        # ---- subset-based loss and map_mse ---------------------------------
        y_hat_dmg, y_hat_loc = y_hat_dmg[..., sensor_subsets], y_hat_loc[..., sensor_subsets]
        i_dmg, i_loc = i_dmg[..., sensor_subsets], i_loc[..., sensor_subsets]
        i_dmg = i_dmg / (i_dmg.sum(-1, keepdim=True) + EPS)
        i_loc = i_loc / (i_loc.sum(-1, keepdim=True) + EPS)

        dmg_preds = torch.einsum("becs,becs->bec", y_hat_dmg, i_dmg)
        loc_preds = torch.einsum("becs,becs->bec", y_hat_loc, i_loc)

        l_dmg = (
            F.mse_loss(
                dmg_preds,
                y_dmg[..., None].expand(-1, -1, dmg_preds.size(-1)),
                reduction="none",
            )
            .mean(1).sum(0).cpu()
        )
        l_loc = (
            F.cross_entropy(
                loc_preds,
                y_loc[..., None].expand(-1, loc_preds.size(-1)),
                reduction="none",
            )
            .sum(0).cpu()
        )
        total_losses += l_dmg + l_loc

        # map_mse in [0, 1] space — consistent with metrics.py
        dist_map = distributed_map_v1(dmg_preds, loc_preds)    # (B, L, N)
        total_mse += map_mse(dist_map, y).cpu()                # (N,)

    model.load_state_dict(state)
    denom = len(val_dl.dataset)
    return (
        torch.median(total_losses).item() / denom,
        torch.median(total_mse).item() / denom,
        tkr_hits / max(tkr_total, 1),
    )


def do_training(model, opt, sched, train_dl, val_dl, epochs: int, ema=None):
    accel = Accelerator(mixed_precision=MIXED_PRECISION_MODE)
    model, opt, sched, train_dl, val_dl = accel.prepare(model, opt, sched, train_dl, val_dl)

    epoch_bar = trange(epochs)
    val_losses: list[float] = []
    train_losses: list[float] = []
    train_loss = float("inf")
    val_loss = float("inf")
    val_mse = float("inf")
    val_acc = 0.0
    val_accs: list[float] = []
    val_mses: list[float] = []

    for _epoch in epoch_bar:
        train_loss = train_one_epoch(model, opt, sched, train_dl, accel, ema)
        train_losses.append(train_loss)
        epoch_bar.set_description(
            f"train_loss {train_loss:.3e} | val_loss {val_loss:.3e} | val_map_mse {val_mse:.3e} | val_top_k_recall {val_acc:.3f}"
        )

        val_loss, val_mse, val_acc = val_one_epoch(
            model, val_dl, DEFAULT_VAL_SUBSET_SIZE
        )
        val_mses.append(val_mse)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        epoch_bar.set_description(
            f"train_loss {train_loss:.3e} | val_loss {val_loss:.3e} | val_map_mse {val_mse:.3e} | val_top_k_recall {val_acc:.3f}"
        )

    # Return the (possibly accelerator-prepared) model so callers can save
    # the exact trained weights.
    return train_losses, val_losses, val_dl, val_accs, val_mses, model


def get_opt_and_sched(model, train_dl: DataLoader, epochs: int):
    base_lr = DEFAULT_BASE_LR
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        betas=DEFAULT_ADAMW_BETAS,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, base_lr, epochs=epochs, steps_per_epoch=len(train_dl)
    )
    return opt, sched


# ===========================================================================
# Approach-B (multi-damage) training and evaluation
# ===========================================================================

def train_one_epoch_b(
    model,
    opt,
    sched,
    train_dl: DataLoader,
    accel: Accelerator,
    criterion,
    ema=None,
) -> tuple[float, float, float]:
    """
    One training epoch for the B-head model.

    Args:
        criterion: :class:`~lib.losses.PresenceSeverityLoss` instance.

    Returns:
        (avg_total_loss, avg_bce_loss, avg_sev_loss) over the epoch.
        The BCE and severity components are logged separately for diagnosing
        which part of the loss dominates (useful for tuning ``pos_weight``
        and ``severity_weight``).
    """
    model.train()
    accum_total = accum_bce = accum_sev = 0.0

    for x, y in train_dl:
        opt.zero_grad()
        with accel.autocast():
            x = randomise_bag_size(x)
            presence_logits, severity = model(x)              # (B, L), (B, L)
            total, bce, sev = criterion(presence_logits, severity, y)
            if ema is not None:
                ema.update_parameters(model)
        accel.backward(total)
        opt.step()
        sched.step()
        accum_total += total.item()
        accum_bce   += bce.item()
        accum_sev   += sev.item()

    n = len(train_dl)
    return accum_total / n, accum_bce / n, accum_sev / n


@torch.inference_mode()
def val_one_epoch_b(
    model,
    val_dl: DataLoader,
    criterion,
    subset_size: int = DEFAULT_VAL_SUBSET_SIZE,
) -> tuple[float, float, float]:
    """
    Validation epoch for the B-head model with sensor-failure robustness eval.

    Evaluation strategy (mirrors v1's ``val_one_epoch``):
    - Sensor subsets are generated deterministically (same seed as v1).
    - For each batch, the per-sensor outputs (``reduce=False``) are indexed
      into N=51 sensor subsets of size C=10.  Importance weights are
      renormalised within each subset (over the C dim), then aggregated.
    - Per-subset map MSE is accumulated and the median is taken — reflecting
      typical performance under sensor failure rather than best or worst case.
    - Presence F1 is computed on the *full-sensor* (all 65 sensors) prediction
      since F1 is a detection metric independent of sensor-failure robustness.

    Returns:
        (val_loss, val_map_mse, val_f1)
        - val_loss:    Median subset BCE loss / dataset size  (convergence signal)
        - val_map_mse: Median subset distributed-map MSE / dataset size
                       (main comparison metric — comparable to v1's val_mse)
        - val_f1:      Full-sensor presence F1
    """
    model.eval()
    state = deepcopy(model.state_dict())

    sensor_subsets = gen_sensor_subsets(
        DEFAULT_VAL_SUBSET_COUNT,
        subset_size=subset_size,
        total_sensors=DEFAULT_VAL_NUM_SENSORS,
    )  # (C=10, N=51)
    N = sensor_subsets.size(1)

    # Move to model device once before the loop.
    device = next(model.parameters()).device
    sensor_subsets = sensor_subsets.to(device)
    pos_weight     = criterion.pos_weight.to(device)

    total_losses = torch.zeros(N)   # (N,) accumulated subset loss
    total_mse    = torch.zeros(N)   # (N,) accumulated subset map MSE
    tp_acc = fp_acc = fn_acc = 0    # full-sensor F1 accumulators

    for x, y in val_dl:
        y = y.to(device)
        features = model[:2](x.float().to(device))    # (B, embed_dim, S=65)

        # Per-sensor outputs — no sensor reduction yet
        presence_raw, severity_raw, importance = model[2](features, reduce=False)
        # presence_raw:  (B, L, S=65) — logits
        # severity_raw:  (B, L, S=65) — sigmoid ∈ [0, 1]
        # importance:    (B, L, S=65) — softmax over S

        # ---- Full-sensor aggregation (used only for F1) -------------------
        pres_full = (presence_raw * importance).sum(-1)   # (B, L)
        tp, fp, fn = presence_f1_stats(pres_full, y)
        tp_acc += tp.item(); fp_acc += fp.item(); fn_acc += fn.item()

        # ---- Subset-based evaluation --------------------------------------
        # Index sensor dim: (B, L, S)[..., (C, N)] → (B, L, C, N)
        pres_sub = presence_raw[..., sensor_subsets]      # (B, L, C, N)
        sev_sub  = severity_raw[..., sensor_subsets]      # (B, L, C, N)
        imp_sub  = importance[..., sensor_subsets]         # (B, L, C, N)

        # Renormalise importance within each subset (over C, dim=-2).
        # This mirrors v1's renormalisation in val_one_epoch and ensures
        # importance weights sum to 1 for every subset independently.
        imp_sub = imp_sub / (imp_sub.sum(dim=-2, keepdim=True) + EPS)

        # Weighted sum over C sensors → (B, L, N) per-subset predictions
        pres_pred_n = (pres_sub * imp_sub).sum(dim=-2)    # (B, L, N)
        sev_pred_n  = (sev_sub  * imp_sub).sum(dim=-2)    # (B, L, N)

        # Distributed map MSE (main comparison metric)
        dist_map_n = distributed_map_b(pres_pred_n, sev_pred_n)         # (B, L, N)
        total_mse += map_mse(dist_map_n, y).cpu()                       # (N,)

        # BCE per subset (loss convergence signal)
        y_pres_n = (
            y.unsqueeze(-1).expand(-1, -1, N) > criterion.presence_threshold
        ).float()
        bce_n = F.binary_cross_entropy_with_logits(
            pres_pred_n,
            y_pres_n,
            pos_weight=pos_weight,
            reduction="none",
        ).mean(1).sum(0).cpu()                                           # (N,)
        total_losses += bce_n

    model.load_state_dict(state)
    denom  = len(val_dl.dataset)
    f1, _, _ = f1_from_counts(tp_acc, fp_acc, fn_acc)
    return (
        torch.median(total_losses).item() / denom,
        torch.median(total_mse).item() / denom,
        f1,
    )


def do_training_b(
    model,
    opt,
    sched,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    criterion,
    ema=None,
):
    """
    Full training loop for the B-head model.

    Return structure is parallel to :func:`do_training` so that the same
    plotting and checkpointing code in ``main.py`` / ``main_b.py`` works
    for both heads:

    Returns:
        (train_losses, val_losses, val_dl, val_f1s, val_mses, model)

        - train_losses: per-epoch average total training loss
        - val_losses:   per-epoch median subset val loss
        - val_dl:       the validation DataLoader (pass-through, matches v1 API)
        - val_f1s:      per-epoch full-sensor presence F1   (analogous to v1 val_accs)
        - val_mses:     per-epoch median subset map MSE     (analogous to v1 val_mses)
        - model:        the accelerator-prepared model

    Additional loss components (bce / severity breakdown) are printed in the
    progress bar but not returned; inspect ``train_bce_losses`` and
    ``train_sev_losses`` via tqdm output or add them to the return if needed.
    """
    accel = Accelerator(mixed_precision=MIXED_PRECISION_MODE)
    model, opt, sched, train_dl, val_dl = accel.prepare(
        model, opt, sched, train_dl, val_dl
    )

    train_losses: list[float] = []
    val_losses:   list[float] = []
    val_f1s:      list[float] = []
    val_mses:     list[float] = []

    val_loss = val_mse = float("inf")
    val_f1 = train_loss = 0.0

    epoch_bar = trange(epochs)
    for _epoch in epoch_bar:
        train_loss, bce, sev = train_one_epoch_b(
            model, opt, sched, train_dl, accel, criterion, ema
        )
        train_losses.append(train_loss)
        epoch_bar.set_description(
            f"train_loss {train_loss:.3e} (bce={bce:.2e} sev={sev:.2e})"
            f" | val_loss {val_loss:.3e} | val_map_mse {val_mse:.3e} | val_f1 {val_f1:.3f}"
        )

        val_loss, val_mse, val_f1 = val_one_epoch_b(
            model, val_dl, criterion, DEFAULT_VAL_SUBSET_SIZE
        )
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        val_f1s.append(val_f1)

        epoch_bar.set_description(
            f"train_loss {train_loss:.3e} (bce={bce:.2e} sev={sev:.2e})"
            f" | val_loss {val_loss:.3e} | val_map_mse {val_mse:.3e} | val_f1 {val_f1:.3f}"
        )

    return train_losses, val_losses, val_dl, val_f1s, val_mses, model

