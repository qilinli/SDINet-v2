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
    _c_slot_decode,
    distributed_map_dr,
    distributed_map_v1,
    f1_from_counts,
    map_mse,
    presence_f1_stats,
)

# Training / evaluation defaults
DEFAULT_DROP_RATE = 0.0

SUBSET_RNG_SEED = 42
DEFAULT_VAL_SUBSET_COUNT = 51
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
    drop_rate: float = DEFAULT_DROP_RATE,
) -> float:
    model.train()
    accum = 0.0
    for x, y in train_dl:
        y_dmg, y_loc = y.max(-1, keepdim=True)
        y_loc = y_loc[:, 0]
        damaged = y_dmg[:, 0] > PRESENCE_NORM_THRESH          # (B,) bool — healthy mask
        opt.zero_grad()
        with accel.autocast():
            x = randomise_bag_size(x, drop_rate)
            y_hat_dmg, y_hat_loc = model(x)
            dmg_loss = F.mse_loss(y_hat_dmg, y_dmg)
            # Only apply localization CE on samples that actually have damage;
            # for healthy samples argmax(y_norm) = 0 by tie-breaking convention,
            # which would train the model to wrongly localize damage at location 0.
            if damaged.any():
                loc_loss = F.cross_entropy(y_hat_loc[damaged], y_loc[damaged])
            else:
                loc_loss = y_hat_loc.sum() * 0.0               # no-op, keeps graph
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
    model, val_dl: DataLoader, subset_size: int | None = None
) -> tuple[float, float, float]:
    '''
    Validate one epoch on the full sensor set (default) or under sampled
    sensor-failure subsets.

    Pass ``subset_size=DEFAULT_VAL_SUBSET_SIZE`` to replicate the fault-tolerance
    evaluation protocol from the original v1 paper (51 subsets of 10 sensors).
    By default (``subset_size=None``) all sensors are used, which is faster and
    measures clean accuracy without injected sensor failures.
    '''
    model.eval()
    state = deepcopy(model.state_dict())

    _x_peek, _ = next(iter(val_dl))
    n_sensors = _x_peek.size(-1)
    if subset_size is None or n_sensors <= subset_size:
        # full sensor set — no fault injection
        sensor_subsets = torch.arange(n_sensors).unsqueeze(1)   # (n_sensors, 1)
    else:
        sensor_subsets = gen_sensor_subsets(
            DEFAULT_VAL_SUBSET_COUNT, subset_size=subset_size, total_sensors=n_sensors,
        )

    total_losses = torch.zeros((sensor_subsets.size(0),))
    total_mse    = torch.zeros((sensor_subsets.size(0),))
    tkr_hits = tkr_total = 0

    for x, y in val_dl:
        y_dmg, y_loc = y.max(-1, keepdim=True)
        y_loc = y_loc[:, 0]
        damaged = y_dmg[:, 0] > PRESENCE_NORM_THRESH          # (B,) — exclude healthy

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
        # Localization CE is only meaningful for samples with actual damage.
        if damaged.any():
            l_loc = (
                F.cross_entropy(
                    loc_preds[damaged],
                    y_loc[damaged, None].expand(-1, loc_preds.size(-1)),
                    reduction="none",
                )
                .sum(0).cpu()
            )
        else:
            l_loc = torch.zeros_like(l_dmg)
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


def do_training(model, opt, sched, train_dl, val_dl, epochs: int, ema=None,
                drop_rate: float = DEFAULT_DROP_RATE):
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
        train_loss = train_one_epoch(model, opt, sched, train_dl, accel, ema, drop_rate)
        train_losses.append(train_loss)
        epoch_bar.set_description(
            f"train_loss {train_loss:.3e} | val_loss {val_loss:.3e} | val_map_mse {val_mse:.3e} | val_top_k_recall {val_acc:.3f}"
        )

        val_loss, val_mse, val_acc = val_one_epoch(model, val_dl)
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
# Approach-C (DETR-style set prediction) training and evaluation
# ===========================================================================

def train_one_epoch_c(
    model,
    opt,
    sched,
    train_dl: DataLoader,
    accel: Accelerator,
    criterion,
    ema=None,
    drop_rate: float = DEFAULT_DROP_RATE,
) -> tuple[float, float, float]:
    """
    One training epoch for the C-head model.

    Args:
        criterion: :class:`~lib.losses.SetCriterion` instance.

    Returns:
        (avg_total_loss, avg_loc_loss, avg_sev_loss) over the epoch.
    """
    model.train()
    accum_total = accum_loc = accum_sev = 0.0

    for x, y in train_dl:
        opt.zero_grad()
        with accel.autocast():
            x = randomise_bag_size(x, drop_rate)
            loc_logits, severity = model(x)              # (B, K, L+1), (B, K)
            total, loc, sev = criterion(loc_logits, severity, y)
            if ema is not None:
                ema.update_parameters(model)
        accel.backward(total)
        opt.step()
        sched.step()
        accum_total += total.item()
        accum_loc   += loc.item()
        accum_sev   += sev.item()

    n = len(train_dl)
    return accum_total / n, accum_loc / n, accum_sev / n


@torch.inference_mode()
def val_one_epoch_c(
    model,
    val_dl: DataLoader,
    criterion,
) -> tuple[float, float, float]:
    """
    Validation epoch for the C-head model (pure DETR slot decoding).

    Top-K recall ranks slots by is-object score and checks if the top-K
    predicted locations cover the true damaged locations.  No threshold
    needed — provides a meaningful signal even before calibration.

    Returns:
        (val_loss, val_map_mse=NaN, val_top_k_recall)
        val_map_mse is NaN because map_mse requires a soft aggregated map
        which is not produced by pure DETR decoding.
    """
    model.eval()
    device = next(model.parameters()).device

    accum_loss = 0.0
    tkr_hits = tkr_total = 0

    for x, y in val_dl:
        x, y = x.float().to(device), y.to(device)
        B = x.size(0)
        loc_logits, severity = model(x)

        total, _, _ = criterion(loc_logits, severity, y)
        accum_loss += total.item() * B

        # Top-k recall: rank slots by is_obj, take top-K per sample
        is_obj, pred_loc = _c_slot_decode(loc_logits)   # (B, K) each
        y_pres = y > PRESENCE_NORM_THRESH                # (B, L) bool
        k_true = y_pres.sum(-1)                          # (B,)
        for b in range(B):
            K = int(k_true[b].item())
            if K == 0:
                continue
            K_slots   = min(K, is_obj.size(-1))
            top_slots = is_obj[b].topk(K_slots).indices
            pred_set  = set(pred_loc[b, top_slots].tolist())
            true_set  = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
            tkr_hits  += len(pred_set & true_set)
            tkr_total += K

    n_samples = len(val_dl.dataset)
    return accum_loss / n_samples, float("nan"), tkr_hits / max(tkr_total, 1)


def do_training_c(
    model,
    opt,
    sched,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    criterion,
    ema=None,
    drop_rate: float = DEFAULT_DROP_RATE,
):
    """
    Full training loop for the C-head model.

    Return structure mirrors :func:`do_training`:

    Returns:
        (train_losses, val_losses, val_dl, val_f1s, val_mses, model)
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
    val_tkr = train_loss = 0.0

    epoch_bar = trange(epochs)
    for _epoch in epoch_bar:
        train_loss, loc, sev = train_one_epoch_c(
            model, opt, sched, train_dl, accel, criterion, ema, drop_rate
        )
        train_losses.append(train_loss)
        epoch_bar.set_description(
            f"train_loss {train_loss:.3e} (loc={loc:.2e} sev={sev:.2e})"
            f" | val_loss {val_loss:.3e} | val_top_k_recall {val_tkr:.3f}"
        )

        val_loss, val_mse, val_tkr = val_one_epoch_c(model, val_dl, criterion)
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        val_f1s.append(val_tkr)

        epoch_bar.set_description(
            f"train_loss {train_loss:.3e} (loc={loc:.2e} sev={sev:.2e})"
            f" | val_loss {val_loss:.3e} | val_top_k_recall {val_tkr:.3f}"
        )

    return train_losses, val_losses, val_dl, val_f1s, val_mses, model


# ===========================================================================
# Approach-DR (direct regression) training and evaluation
# ===========================================================================

def train_one_epoch_dr(
    model,
    opt,
    sched,
    train_dl: DataLoader,
    accel: Accelerator,
    ema=None,
    drop_rate: float = DEFAULT_DROP_RATE,
    pos_weight: float | None = None,
) -> float:
    """One training epoch for the DR head.

    Loss:
        pos_weight=None  →  MSE(pred, (y_norm + 1) / 2)   — continuous severity targets
        pos_weight=float →  weighted BCE(pred, y_presence) — binary labels (e.g. Qatar)

    BCE with pos_weight is more principled when labels are binary {-1, +1}:
    it uses the correct gradient geometry for classification and handles the
    class imbalance (K damaged / L total locations).  MSE on binary targets
    converges, but with weaker gradient signal and no imbalance compensation.
    """
    model.train()
    accum = 0.0

    for x, y in train_dl:
        opt.zero_grad()
        with accel.autocast():
            x    = randomise_bag_size(x, drop_rate)
            pred = model(x)                                    # (B, L) ∈ [0, 1]
            if pos_weight is not None:
                target = (y > PRESENCE_NORM_THRESH).float()
                # F.binary_cross_entropy doesn't support pos_weight directly;
                # equivalent weighting: w=pos_weight for positives, w=1 for negatives
                w = target * (pos_weight - 1.0) + 1.0
                # clamp: MIL aggregation (sigmoid * softmax).sum() is theoretically
                # in [0,1] but floating-point rounding can produce values like 1.0000001
                loss = F.binary_cross_entropy(pred.clamp(0.0, 1.0), target, weight=w)
            else:
                loss = F.mse_loss(pred, (y + 1.0) / 2.0)
            if ema is not None:
                ema.update_parameters(model)
        accel.backward(loss)
        opt.step()
        sched.step()
        accum += loss.item()

    return accum / len(train_dl)


@torch.inference_mode()
def val_one_epoch_dr(
    model,
    val_dl: DataLoader,
    pos_weight: float | None = None,
) -> tuple[float, float, float]:
    """
    Validation epoch for the DR head (full-sensor, no sensor subsets).

    Returns:
        (val_loss, val_map_mse, val_top_k_recall)
    """
    model.eval()
    device = next(model.parameters()).device

    accum_loss = 0.0
    accum_mse  = 0.0
    tkr_hits = tkr_total = 0

    for x, y in val_dl:
        x, y = x.float().to(device), y.to(device)
        B = x.size(0)
        pred = model(x)                                        # (B, L) ∈ [0, 1]

        if pos_weight is not None:
            target = (y > PRESENCE_NORM_THRESH).float()
            w = target * (pos_weight - 1.0) + 1.0
            # reduction='mean' gives per-element mean; multiply by B to get per-sample sum
            accum_loss += F.binary_cross_entropy(pred.clamp(0.0, 1.0), target, weight=w).item() * B
        else:
            accum_loss += F.mse_loss(pred, (y + 1.0) / 2.0).item() * B
        accum_mse += map_mse(distributed_map_dr(pred), y).item()

        # Top-k recall: rank locations by pred score, check top-K against ground truth.
        y_pres = y > PRESENCE_NORM_THRESH                          # (B, L) bool
        k_true = y_pres.sum(-1)                                    # (B,)
        for b in range(B):
            K = int(k_true[b].item())
            if K == 0:
                continue
            topk_mask = torch.zeros(y.size(-1), dtype=torch.bool, device=y.device)
            topk_mask[pred[b].topk(K).indices] = True
            tkr_hits  += int((topk_mask & y_pres[b]).sum())
            tkr_total += K

    n_samples = len(val_dl.dataset)
    # Both accum_loss and accum_mse are sums over samples → divide by n_samples for
    # per-sample means on the same scale as train_loss and v1/B val metrics.
    return accum_loss / n_samples, accum_mse / n_samples, tkr_hits / max(tkr_total, 1)


def do_training_dr(
    model,
    opt,
    sched,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    ema=None,
    drop_rate: float = DEFAULT_DROP_RATE,
    pos_weight: float | None = None,
):
    """
    Full training loop for the DR head.

    Returns:
        (train_losses, val_losses, val_dl, val_f1s, val_mses, model)
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
    val_tkr = train_loss = 0.0

    epoch_bar = trange(epochs)
    for _epoch in epoch_bar:
        train_loss = train_one_epoch_dr(model, opt, sched, train_dl, accel, ema, drop_rate, pos_weight)
        train_losses.append(train_loss)
        epoch_bar.set_description(
            f"train_loss {train_loss:.3e}"
            f" | val_loss {val_loss:.3e} | val_map_mse {val_mse:.3e} | val_top_k_recall {val_tkr:.3f}"
        )

        val_loss, val_mse, val_tkr = val_one_epoch_dr(model, val_dl, pos_weight)
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        val_f1s.append(val_tkr)

        epoch_bar.set_description(
            f"train_loss {train_loss:.3e}"
            f" | val_loss {val_loss:.3e} | val_map_mse {val_mse:.3e} | val_top_k_recall {val_tkr:.3f}"
        )

    return train_losses, val_losses, val_dl, val_f1s, val_mses, model

