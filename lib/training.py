from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import trange

# Training / evaluation defaults
DEFAULT_DROP_NUM = 10
DEFAULT_DROP_DEN = 65
DEFAULT_DROP_RATE = DEFAULT_DROP_NUM / DEFAULT_DROP_DEN

SUBSET_RNG_SEED = 42
DEFAULT_VAL_SUBSET_COUNT = 51
DEFAULT_VAL_NUM_SENSORS = 65
DEFAULT_VAL_SUBSET_SIZE = 10
EPS = 1e-12
PERCENT_SCALE = 100.0
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
    total_mse = torch.zeros((sensor_subsets.size(0),))
    loc_corr = torch.zeros((sensor_subsets.size(0),))

    for x, y in val_dl:
        y_dmg, y_loc = y.max(-1, keepdim=True)
        y_loc = y_loc[:, 0]

        y_hat_dmg, i_dmg, y_hat_loc, i_loc = model[2](model[:2](x.float()), False)
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
            .mean(1)
            .sum(0)
            .cpu()
        )
        l_loc = (
            F.cross_entropy(
                loc_preds,
                y_loc[..., None].expand(-1, loc_preds.size(-1)),
                reduction="none",
            )
            .sum(0)
            .cpu()
        )
        total_losses += l_dmg + l_loc

        distributed_pred = (dmg_preds.add(1).div(2) * loc_preds.softmax(1)).mul(2).sub(1)
        total_mse += (
            F.mse_loss(
                distributed_pred,
                y[..., None].expand(-1, -1, dmg_preds.size(-1)),
                reduction="none",
            )
            .mean(1)
            .sum(0)
            .cpu()
        )

        loc_corr += (loc_preds.argmax(1) == y_loc[:, None]).sum(0).cpu()

    model.load_state_dict(state)
    denom = len(val_dl.dataset)
    return (
        torch.median(total_losses).item() / denom,
        torch.median(total_mse).item() / denom,
        torch.median(loc_corr).item() / denom,
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
            f"Train Loss: {train_loss:10.04e}, Val Loss: {val_loss:10.04e} | Val MSE: {val_mse:10.04e} | Val Acc: {val_acc:5.02f}%"
        )

        val_loss, val_mse, val_acc = val_one_epoch(
            model, val_dl, DEFAULT_VAL_SUBSET_SIZE
        )
        val_acc *= PERCENT_SCALE
        val_mses.append(val_mse)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        epoch_bar.set_description(
            f"Train Loss: {train_loss:10.04e}, Val Loss: {val_loss:10.04e} | Val MSE: {val_mse:10.04e} | Val Acc: {val_acc:5.02f}%"
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

