"""
train.py — unified training entry point for SDINet-v2.

Replaces main.py, main_c.py, and main_dr.py.

Usage::

    python train.py --model v1  --dataset 7story [--epochs N]
    python train.py --model c   --dataset qatar  [--epochs N] [--sev-weight 0.0]
    python train.py --model dr  --dataset tower  [--epochs N]
    python train.py --model v1  --dataset 7story --epochs 1   # smoke-test

Dataset-specific flags (window-size, overlap, downsample, tower-excitation, snr)
are only meaningful for their respective datasets; other datasets ignore them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

# Must be set before any torch import to take effect
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import torch

from lib.datasets import get_dataset
from lib.model import ModelConfig, ModelConfigC, ModelConfigDR, build_model, build_criterion_c
from lib.training import do_training, do_training_c, do_training_dr, get_opt_and_sched
from lib.visualization import plot_training_results


# ---------------------------------------------------------------------------
# Dispatch tables (keyed by model name)
# ---------------------------------------------------------------------------

def _calibrate_fn(model_name: str):
    from lib.calibration import calibrate_v1_ratio, calibrate_c_obj_threshold, calibrate_dr_ratio
    return {"v1": calibrate_v1_ratio, "c": calibrate_c_obj_threshold, "dr": calibrate_dr_ratio}[model_name]


def _eval_loader_fn(model_name: str):
    from lib.calibration import (
        eval_on_loader_v1_calibrated,
        eval_on_loader_c_calibrated,
        eval_on_loader_dr_calibrated,
    )
    return {
        "v1": eval_on_loader_v1_calibrated,
        "c":  eval_on_loader_c_calibrated,
        "dr": eval_on_loader_dr_calibrated,
    }[model_name]


def _real_test_fn(model_name: str):
    from lib.calibration import (
        do_real_test_v1_calibrated,
        do_real_test_c_calibrated,
        do_real_test_dr_calibrated,
    )
    return {
        "v1": do_real_test_v1_calibrated,
        "c":  do_real_test_c_calibrated,
        "dr": do_real_test_dr_calibrated,
    }[model_name]


# ---------------------------------------------------------------------------
# Dataloader kwargs builder
# ---------------------------------------------------------------------------

def _build_dl_kwargs(args, dataset) -> dict:
    """
    Map CLI flags → the keyword dict accepted by DatasetConfig.get_dataloaders().

    Each dataset's wrapper function uses **_ to ignore irrelevant keys, so it
    is safe to always include all keys here.
    """
    root = args.root if args.root is not None else dataset.default_root
    base = dict(
        root=root,
        num_workers=args.num_workers,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )
    if dataset.name == "7story":
        base["snr"] = args.snr
    elif dataset.name == "tower":
        base["tower_excitation"] = tuple(args.tower_excitation)
    elif dataset.name == "qatar":
        base["window_size"] = args.window_size
        base["overlap"]     = args.overlap
        base["downsample"]  = args.downsample
    return base


# ---------------------------------------------------------------------------
# Model config builder
# ---------------------------------------------------------------------------

def _build_model_config(args, dataset):
    """
    Build ModelConfig / ModelConfigC / ModelConfigDR from CLI args + DatasetConfig.

    Dataset-mandated overrides (e.g. Qatar+C → sev_weight=0.0) are fetched from
    the registry; any explicit CLI flag takes priority.
    """
    time_len = dataset.time_len
    if time_len is None:
        time_len = args.window_size // args.downsample

    shared = dict(
        n_sensors=dataset.n_sensors,
        time_len=time_len,
        structure=tuple(args.structure),
        embed_dim=args.embed_dim,
        neck_dropout=args.neck_dropout,
    )

    if args.model == "v1":
        return ModelConfig(
            **shared,
            out_channels=dataset.n_locations + 1,
            importance_dropout=args.importance_dropout,
            temperature=args.temperature,
            val_temperature=args.val_temperature if args.val_temperature is not None else args.temperature,
        )

    elif args.model == "c":
        registry_overrides = dataset.model_config_overrides("c")
        # CLI flags win over registry defaults
        sev_weight = args.sev_weight if args.sev_weight is not None else registry_overrides.get("sev_weight", 1.0)
        return ModelConfigC(
            **shared,
            num_locations=dataset.n_locations,
            num_slots=args.num_slots,
            num_decoder_layers=args.num_decoder_layers,
            nhead=args.nhead,
            no_obj_weight=args.no_obj_weight,
            loc_weight=args.loc_weight,
            sev_weight=sev_weight,
        )

    elif args.model == "dr":
        return ModelConfigDR(
            **shared,
            num_locations=dataset.n_locations,
            importance_dropout=args.importance_dropout,
            temperature=args.temperature,
            val_temperature=args.val_temperature if args.val_temperature is not None else args.temperature,
        )

    raise ValueError(f"Unknown model {args.model!r}")


# ---------------------------------------------------------------------------
# Training dispatcher
# ---------------------------------------------------------------------------

def _run_training(args, model, ema, opt, sched, train_dl, val_dl, criterion=None):
    kwargs = dict(drop_rate=args.drop_rate)
    if args.model == "v1":
        return do_training(model, opt, sched, train_dl, val_dl, args.epochs, ema=ema, **kwargs)
    elif args.model == "c":
        return do_training_c(model, opt, sched, train_dl, val_dl, args.epochs, criterion, ema=ema, **kwargs)
    elif args.model == "dr":
        return do_training_dr(model, opt, sched, train_dl, val_dl, args.epochs, ema=ema, **kwargs,
                              pos_weight=_dr_pos_weight(args))
    raise ValueError(f"Unknown model {args.model!r}")


def _dr_pos_weight(args) -> float | None:
    """Resolve pos_weight for the DR head.

    Registry default for Qatar: L-1 (passed via --bce-pos-weight).
    Set --bce-pos-weight 0 or negative to fall back to MSE.
    None means use MSE (non-Qatar datasets).
    """
    if args.bce_pos_weight is None:
        return None
    if args.bce_pos_weight <= 0:
        return None
    return args.bce_pos_weight


# ---------------------------------------------------------------------------
# Post-training: calibration + eval
# ---------------------------------------------------------------------------

def _run_post_training(args, dataset, trained_model, dl_kwargs, ckpt_path, save_dir):
    from lib.calibration import save_calibration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calibrate  = _calibrate_fn(args.model)
    eval_loader = _eval_loader_fn(args.model)
    real_test   = _real_test_fn(args.model)

    # Calibration: use val loaders from the registry helper
    cal_loaders = dataset.get_calibration_val_loaders(**dl_kwargs)
    cal = calibrate(trained_model, cal_loaders, device)
    if ckpt_path is not None:
        save_calibration(cal, ckpt_path)

    # Test evaluation
    for label, test_dl in dataset.get_test_loaders(**dl_kwargs):
        print(f"\n[test / {label}]")
        for k, v in eval_loader(trained_model, test_dl, device, **cal).items():
            print(f"  {k}: {v:.4f}")

    # Optional extra test (Qatar double-damage)
    extra_dl = dataset.get_extra_test_loader(**dl_kwargs)
    if extra_dl is not None:
        print(f"\n[test / {dataset.name} double]")
        for k, v in eval_loader(trained_model, extra_dl, device, **cal).items():
            print(f"  {k}: {v:.4f}")

    # Real benchmark (7-story only)
    if dataset.supports_real_benchmark and args.run_real_test:
        print("\n[real benchmark]")
        try:
            real_test(trained_model, device=device, print_result=True, **cal)
        except (FileNotFoundError, OSError) as e:
            print(f"  Skipping (missing benchmark .mat): {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    dataset = get_dataset(args.dataset)
    dl_kwargs = _build_dl_kwargs(args, dataset)

    train_dl, val_dl, _ = dataset.get_dataloaders(**dl_kwargs)

    model_cfg = _build_model_config(args, dataset)
    model = build_model(model_cfg)
    criterion = build_criterion_c(model_cfg) if args.model == "c" else None

    ema = torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
    )

    opt, sched = get_opt_and_sched(model, train_dl, args.epochs)

    train_losses, val_losses, _, val_accs, val_mses, trained_model = _run_training(
        args, model, ema, opt, sched, train_dl, val_dl, criterion
    )

    # Save checkpoint
    states_dir = Path(__file__).resolve().parent / "states" / args.dataset
    states_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = None
    if args.save_checkpoint:
        ckpt_path = states_dir / f"{args.model}-combined-{uuid4()}.pt"
        torch.save(trained_model.state_dict(), ckpt_path)
        print(f"[checkpoint] Saved: {ckpt_path}")

    # Plots
    val_acc_label = "Val F1" if args.model in ("c", "dr") else "Val top-k recall"
    save_dir = f"saved_results/{args.dataset}/{args.model}"
    plot_training_results(
        train_losses, val_losses, val_accs, val_mses,
        save_dir=save_dir,
        show=False,
        val_acc_label=val_acc_label,
    )

    _run_post_training(args, dataset, trained_model, dl_kwargs, ckpt_path, save_dir=save_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from lib.datasets import DATASETS
    from lib.data_tower import TOWER_DEFAULT_ROOT
    from lib.data_qatar import QATAR_DEFAULT_ROOT

    parser = argparse.ArgumentParser(
        description="Train SDINet v1 / C / DR on any registered dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required ---
    parser.add_argument("--model",   required=True, choices=["v1", "c", "dr"],
                        help="Model head to train")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS),
                        help="Dataset to train on")

    # --- shared training ---
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--eval-batch-size", type=int,   default=32)
    parser.add_argument("--drop-rate",       type=float, default=0.0)
    parser.add_argument("--num-workers",     type=int,   default=0)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--no-checkpoint",   dest="save_checkpoint", action="store_false",
                        help="Skip saving a UUID checkpoint")
    parser.add_argument("--no-real-test",    dest="run_real_test",   action="store_false",
                        help="Skip real .mat benchmark (7-story only)")
    parser.set_defaults(save_checkpoint=True, run_real_test=True)

    # --- dataset root override ---
    parser.add_argument("--root", default=None,
                        help="Override dataset root path (default: per-dataset default in registry)")

    # --- backbone / neck ---
    parser.add_argument("--structure",    type=int, nargs=3, default=[6, 6, 6],
                        metavar=("B1", "B2", "B3"),
                        help="DenseNet block depths")
    parser.add_argument("--embed-dim",    type=int,   default=768)
    parser.add_argument("--neck-dropout", type=float, default=0.0)

    # --- v1 / DR head knobs ---
    parser.add_argument("--importance-dropout", type=float, default=0.5)
    parser.add_argument("--temperature",        type=float, default=1e-2)
    parser.add_argument("--val-temperature",    type=float, default=None,
                        help="Eval temperature (defaults to --temperature)")

    # --- C-head knobs ---
    parser.add_argument("--num-slots",          type=int,   default=5,
                        help="Slot count K_max (default: 5)")
    parser.add_argument("--num-decoder-layers", type=int,   default=2)
    parser.add_argument("--nhead",              type=int,   default=8)
    parser.add_argument("--no-obj-weight",      type=float, default=0.1)
    parser.add_argument("--loc-weight",         type=float, default=1.0)
    parser.add_argument("--sev-weight",         type=float, default=None,
                        help="Severity loss weight (default: 1.0, or 0.0 for Qatar via registry)")

    # --- DR-head knobs ---
    parser.add_argument("--bce-pos-weight", type=float, default=None,
                        help="BCE pos_weight for binary datasets (default: L-1 from registry; 0=use MSE)")

    # --- dataset-specific ---
    parser.add_argument("--snr",              type=float, default=-1.0,
                        help="SNR for noise injection (7-story only; -1=off)")
    parser.add_argument("--tower-excitation", nargs="+", default=["EQ", "WN", "sine"],
                        help="Excitation types to include (tower only)")
    parser.add_argument("--window-size",      type=int,   default=2048,
                        help="Window size in samples (qatar only; 2048=2s at 1024 Hz)")
    parser.add_argument("--overlap",          type=float, default=0.5,
                        help="Window overlap fraction (qatar only)")
    parser.add_argument("--downsample",       type=int,   default=4,
                        help="Decimation factor (qatar only; 4=256 Hz)")

    args = parser.parse_args()

    # Apply registry training overrides as argparse defaults (CLI wins)
    dataset = get_dataset(args.dataset)
    reg_train = dataset.training_overrides(args.model)
    if args.model == "dr" and args.bce_pos_weight is None and "pos_weight" in reg_train:
        args.bce_pos_weight = reg_train["pos_weight"]

    main(args)
