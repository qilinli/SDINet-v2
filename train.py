"""
train.py — unified training entry point for SDINet-v2.

Replaces main.py, main_c.py, and main_dr.py.

Usage::

    python train.py --model v1  --dataset 7story [--epochs N]
    python train.py --model c   --dataset qatar  [--epochs N] [--sev-weight 0.0]
    python train.py --model dr  --dataset tower  [--epochs N]
    python train.py --model v1  --dataset 7story --epochs 1   # smoke-test

Dataset-specific flags (window-size, overlap, downsample, tower-excitation)
are only meaningful for their respective datasets; other datasets ignore them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4


import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import torch

from lib.datasets import get_dataset
from lib.itransformer import POS_EMB_TYPES
from lib.model import ModelConfig, ModelConfigC, ModelConfigDR, build_model, build_criterion_c, build_criterion_fault
from lib.training import do_training, do_training_c, do_training_dr, get_opt_and_sched, get_opt_and_sched_c
from lib.visualization import plot_training_results


# ---------------------------------------------------------------------------
# Dispatch tables (keyed by model name)
# ---------------------------------------------------------------------------

def _calibrate_fn(model_name: str):
    from lib.calibration import calibrate_v1_ratio, calibrate_c_obj_threshold, calibrate_dr_ratio
    return {"v1": calibrate_v1_ratio, "c": calibrate_c_obj_threshold, "dr": calibrate_dr_ratio, "b": calibrate_dr_ratio}[model_name]


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
        "b":  eval_on_loader_dr_calibrated,
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
    # fault augmentation — shared across all datasets (loaders ignore irrelevant ones via **_)
    base["p_hard"]        = args.p_hard
    base["p_soft"]        = args.p_soft
    base["p_struct_mask"] = args.p_struct_mask

    if dataset.name in ("7story", "7story-sparse"):
        if args.subsets:
            base["subsets"] = args.subsets
        base["norm_method"] = args.norm_method
    elif dataset.name in ("asce", "asce-columns"):
        base["norm_method"] = args.norm_method
    elif dataset.name == "tower":
        base["tower_excitation"] = tuple(args.tower_excitation)
    elif dataset.name == "lumo":
        base["window_size"] = args.window_size
        base["overlap"]     = args.overlap
        base["downsample"]  = args.downsample
    elif dataset.name == "qatar":
        base["window_size"]     = args.window_size
        base["overlap"]         = args.overlap
        base["downsample"]      = args.downsample
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
        if args.use_structural_bias and args.num_decoder_layers < 2:
            print("[warning] --use-structural-bias has no effect with --num-decoder-layers < 2 "
                  "(layer 0 is always unbiased; need at least one biased layer).")
        registry_overrides = dataset.model_config_overrides("c")
        # CLI flags win over registry defaults
        sev_weight = args.sev_weight if args.sev_weight is not None else registry_overrides.get("sev_weight", 1.0)
        num_slots = args.num_slots if args.num_slots is not None else registry_overrides.get("num_slots", 5)
        return ModelConfigC(
            **shared,
            num_locations=dataset.n_locations,
            num_slots=num_slots,
            num_decoder_layers=args.num_decoder_layers,
            nhead=args.nhead,
            no_obj_weight=args.no_obj_weight,
            loc_weight=args.loc_weight,
            sev_weight=sev_weight,
            use_spatial_layer=args.use_spatial_layer,
            num_spatial_layers=args.num_spatial_layers,
            spatial_nhead=args.spatial_nhead,
            use_fault_head=args.use_fault_head,
            fault_loss_weight=args.fault_loss_weight,
            use_structural_bias=args.use_structural_bias,
            fault_gate_lambda=args.fault_gate_lambda,
            freeze_r_bias=args.freeze_r_bias,
            aux_loss=args.aux_loss,
            use_l2_norm_memory=args.use_l2_norm_memory,
            use_mil_readout=args.use_mil_readout,
            encoder_type=args.encoder_type,
            encoder_num_layers=args.encoder_num_layers,
            encoder_nhead=args.encoder_nhead,
            encoder_dropout=args.encoder_dropout,
            encoder_pos_emb=args.encoder_pos_emb,
            tokenizer_type=args.tokenizer_type,
            fault_head_location=args.fault_head_location,
        )

    elif args.model == "dr":
        return ModelConfigDR(
            **shared,
            num_locations=dataset.n_locations,
            importance_dropout=args.importance_dropout,
            temperature=args.temperature,
            val_temperature=args.val_temperature if args.val_temperature is not None else args.temperature,
        )

    elif args.model == "b":
        return ModelConfigDR(
            **shared,
            num_locations=dataset.n_locations,
            plain=True,
        )

    raise ValueError(f"Unknown model {args.model!r}")


# ---------------------------------------------------------------------------
# Training dispatcher
# ---------------------------------------------------------------------------

def _run_training(args, model, opt, sched, train_dl, val_dl, criterion=None, fault_criterion=None):
    kwargs = dict(drop_rate=args.drop_rate, val_every=args.val_every)
    if args.model == "v1":
        return do_training(model, opt, sched, train_dl, val_dl, args.epochs, **kwargs)
    elif args.model == "c":
        # iTransformer self-attention logits overflow fp16 after enough training
        # drift — train it in fp32.  Seen on 7story-fault-itransformer/c-fh-sb-it
        # (clean descent to ep 88, catastrophic loss cliff at ep 89, stuck).
        mp = "no" if args.encoder_type == "itransformer" else None
        return do_training_c(
            model, opt, sched, train_dl, val_dl, args.epochs, criterion,
            fault_criterion=fault_criterion, fault_loss_weight=args.fault_loss_weight,
            aux_loss_weight=args.aux_loss_weight,
            mixed_precision=mp,
            **kwargs,
        )
    elif args.model in ("dr", "b"):
        return do_training_dr(model, opt, sched, train_dl, val_dl, args.epochs, **kwargs,
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

    # Calibration: use val loaders from the registry helper
    cal_loaders = dataset.get_calibration_val_loaders(**dl_kwargs)
    cal = calibrate(trained_model, cal_loaders, device)
    if args.model == "c" and getattr(args, "c_calibrate_ensemble", False):
        from lib.calibration import calibrate_c_ensemble
        cal.update(calibrate_c_ensemble(
            trained_model, cal_loaders, device,
            use_severity=not args.c_ensemble_no_severity,
        ))
    if ckpt_path is not None and cal:
        save_calibration(cal, ckpt_path)

    # Test evaluation
    for label, test_dl in dataset.get_test_loaders(**dl_kwargs):
        print(f"\n[test / {label}]")
        for k, v in eval_loader(trained_model, test_dl, device, **cal).items():
            print(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    dataset = get_dataset(args.dataset)
    dl_kwargs = _build_dl_kwargs(args, dataset)

    train_dl, val_dl, _ = dataset.get_dataloaders(**dl_kwargs)

    model_cfg = _build_model_config(args, dataset)
    structural_affinity = (
        dataset.build_structural_affinity()
        if getattr(args, "use_structural_bias", False)
        else None
    )
    model = build_model(model_cfg, structural_affinity=structural_affinity)
    criterion = build_criterion_c(model_cfg) if args.model == "c" else None
    fault_criterion = build_criterion_fault(model_cfg) if args.model == "c" else None

    if args.model == "c":
        opt, sched = get_opt_and_sched_c(model, train_dl, args.epochs)
    else:
        opt, sched = get_opt_and_sched(model, train_dl, args.epochs)

    train_losses, val_losses, _, val_accs, val_mses, trained_model, best_state = _run_training(
        args, model, opt, sched, train_dl, val_dl, criterion, fault_criterion=fault_criterion
    )

    # Save checkpoint
    if args.label:
        model_label = args.label
    elif args.model == "c":
        # Head-to-toe auto-label: walks the data path in order.
        #   c-<norm>[-msc]-<enc><depth>[-posemb][-fault][-sb][-sKdD][-mil]
        # See CLAUDE.md for the full convention.
        norm_tag = {"mean": "nm", "none": "nn"}[args.norm_method]
        enc_tag  = "dn" if args.encoder_type == "densenet" \
                   else f"it{args.encoder_num_layers}"
        parts = ["c", norm_tag]
        if args.tokenizer_type == "multiscale_conv":
            parts.append("msc")
        parts.append(enc_tag)
        if args.encoder_type == "itransformer" and args.encoder_pos_emb != "learned":
            parts.append({"none": "pen", "rope": "per"}[args.encoder_pos_emb])
        if args.use_fault_head:
            parts.append("pfh" if args.fault_head_location == "encoder" else "fh")
        if args.use_structural_bias:
            parts.append("sb")
        # no_obj_weight marker: omit when default 0.1. Encode as nw<digit> where
        # digit = value×10 (nw5 = 0.5, nw3 = 0.3, nw2 = 0.2, etc).
        if abs(args.no_obj_weight - 0.1) > 1e-6:
            parts.append(f"nw{int(round(args.no_obj_weight * 10))}")
        # Slot count and decoder depth each appended independently when non-default.
        k_eff = args.num_slots if args.num_slots is not None else 5
        if k_eff != 5:
            parts.append(f"s{k_eff}")
        if args.num_decoder_layers != 2:
            parts.append(f"d{args.num_decoder_layers}")
        # embed_dim marker: omit at default 768. Encode as e<N>.
        if args.embed_dim != 768:
            parts.append(f"e{args.embed_dim}")
        if args.use_mil_readout:
            parts.append("mil")
        model_label = "-".join(parts)
    else:
        model_label = args.model
    run_name   = f"{args.dataset}-{args.tag}" if args.tag else args.dataset
    states_dir = Path(__file__).resolve().parent / "states" / run_name
    states_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = None
    if args.save_checkpoint:
        run_uuid = uuid4()
        if best_state is not None:
            best_epoch, best_weights = best_state
            ckpt_path = states_dir / f"{model_label}-{run_uuid}-best{best_epoch}.pt"
            trained_model.load_state_dict(best_weights)
            torch.save(best_weights, ckpt_path)
        else:
            ckpt_path = states_dir / f"{model_label}-{run_uuid}.pt"
            torch.save(
                {k: v.detach().cpu() for k, v in trained_model.state_dict().items()},
                ckpt_path,
            )
        print(f"[checkpoint] Saved: {ckpt_path}")

    # Plots
    val_acc_label = "Val F1" if args.model in ("c", "dr") else "Val top-k recall"
    save_dir = f"saved_results/{run_name}/{model_label}"
    plot_training_results(
        train_losses, val_losses, val_accs, val_mses,
        save_dir=save_dir,
        show=False,
        val_acc_label=val_acc_label,
    )

    _run_post_training(args, dataset, trained_model, dl_kwargs, ckpt_path, save_dir=save_dir)

    # Optional fault evaluation
    if getattr(args, "eval_fault", False) and ckpt_path is not None:
        _run_eval_fault(args, ckpt_path, model_label, run_name)


def _run_eval_fault(args, ckpt_path, model_label, run_name):
    """Run evaluate_fault.py as a subprocess after training."""
    import subprocess

    # Map model type to the right CLI flag for evaluate_fault.py
    model_flag = {"v1": "--v1", "c": "--c", "dr": "--dr", "b": "--b"}[args.model]
    label_flag = {"v1": "--v1-label", "c": "--c-label", "dr": "--dr-label", "b": "--b-label"}[args.model]

    # Display label: e.g. "C+fh+sb", "v1", "B"
    _display = {"v1": "v1", "c": "C", "dr": "DR", "b": "B"}
    display = _display[args.model]
    if args.model == "c":
        if args.use_fault_head:
            display += "+fh"
        if args.use_structural_bias:
            display += "+sb"
        if args.encoder_type == "itransformer":
            display += "+it"

    out_stem = f"saved_results/{run_name}/eval_fault_{model_label}"

    cmd = [
        sys.executable, "evaluate_fault.py",
        "--dataset", args.dataset,
        model_flag, str(ckpt_path),
        label_flag, display,
        "--out", out_stem,
    ]
    if args.root is not None:
        cmd += ["--root", args.root]
    if args.dataset in ("7story", "7story-sparse", "asce", "asce-columns"):
        cmd += ["--norm-method", args.norm_method]

    print(f"\n{'=' * 60}")
    print(f"  Running fault evaluation: {display}")
    print(f"{'=' * 60}\n")
    subprocess.call(cmd)


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
    parser.add_argument("--model",   required=True, choices=["v1", "c", "dr", "b"],
                        help="Model head to train")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS),
                        help="Dataset to train on")

    # --- shared training ---
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--val-every",       type=int,   default=10,
                        help="Run validation every N epochs (default 10; always runs on last epoch)")
    parser.add_argument("--batch-size",      type=int,   default=128)
    parser.add_argument("--eval-batch-size", type=int,   default=128)
    parser.add_argument("--drop-rate",       type=float, default=0.0)
    parser.add_argument("--num-workers",     type=int,   default=8)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--no-checkpoint",   dest="save_checkpoint", action="store_false",
                        help="Skip saving a UUID checkpoint")
    parser.set_defaults(save_checkpoint=True)

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
    parser.add_argument("--num-slots",          type=int,   default=None,
                        help="Slot count K_max (default: 5, or registry override)")
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
    parser.add_argument("--subsets",          nargs="+", default=None,
                        choices=["single", "double", "undamaged"],
                        help="Training subsets for 7story datasets "
                             "(default: single double undamaged — undamaged is pulled from both "
                             "unc=0 and unc=1, ~200 files total).")
    parser.add_argument("--tower-excitation", nargs="+", default=["EQ", "WN", "sine"],
                        help="Excitation types to include (tower only)")
    parser.add_argument("--window-size",      type=int,   default=2048,
                        help="Window size in samples (qatar only; 2048=2s at 1024 Hz)")
    parser.add_argument("--overlap",          type=float, default=0.5,
                        help="Window overlap fraction (qatar only)")
    parser.add_argument("--downsample",       type=int,   default=4,
                        help="Decimation factor (qatar only; 4=256 Hz)")
    # --- sensor spatial reasoning (C-head only) ---
    parser.add_argument("--use-spatial-layer",   action="store_true", default=False,
                        help="Enable SensorPositionalEncoding + SensorSpatialLayer before slot decoder.")
    parser.add_argument("--num-spatial-layers",  type=int,   default=1,
                        help="Depth of the SensorSpatialLayer TransformerEncoder (default 1).")
    parser.add_argument("--spatial-nhead",       type=int,   default=8,
                        help="Attention heads in SensorSpatialLayer (default 8).")
    # --- fault detection head (C-head only) ---
    parser.add_argument("--use-fault-head",      action="store_true", default=False,
                        help="Enable per-sensor binary fault classifier head.")
    parser.add_argument("--use-structural-bias", action="store_true", default=False,
                        help="C-head: learnable location-sensor affinity bias in slot "
                             "cross-attention, initialised from Qatar 6×5 grid 4-connectivity. "
                             "Qatar only; requires --num-decoder-layers >= 2.")
    parser.add_argument("--fault-loss-weight",   type=float, default=1.0,
                        help="Scale factor for fault BCE loss (default 1.0).")
    parser.add_argument("--fault-gate-lambda",   type=float, default=0.0,
                        help="Subtract λ·p_fault (detached) from cross-attention logits "
                             "so slots avoid faulted sensors (default 0.0 = off).")
    parser.add_argument("--freeze-r-bias",       action="store_true", default=False,
                        help="Freeze R_bias at physics init AND apply it at decoder layer 0 "
                             "(default off = learnable bias applied only at layers ≥1).")
    parser.add_argument("--aux-loss",            action="store_true", default=False,
                        help="Deep supervision: apply SetCriterion at every decoder layer's output.")
    parser.add_argument("--aux-loss-weight",     type=float, default=1.0,
                        help="Scale applied to each intermediate-layer loss (default 1.0).")
    # --- magnitude-invariant memory (Option 4) ---
    parser.add_argument("--use-l2-norm-memory",  action="store_true", default=False,
                        help="C-head: L2-normalize sensor features before decoder cross-attention. "
                             "Makes attention scores magnitude-invariant so fault-induced feature "
                             "scaling no longer biases routing.")
    # --- MIL-style slot readout (Option 5) ---
    parser.add_argument("--use-mil-readout",     action="store_true", default=False,
                        help="C-head: replace slot-state location head with attention-weighted "
                             "per-sensor voting using last-layer cross-attention weights. "
                             "Brings v1-style MIL redundancy into the slot architecture.")
    # --- Encoder backbone for C-head ---
    parser.add_argument("--encoder-type", choices=["densenet", "itransformer"], default="densenet",
                        help="C-head encoder. 'densenet' (default) = SDIDenseNet + MIL neck (original "
                             "v1/B/C backbone). 'itransformer' = per-sensor Linear(T, D) tokenisation + "
                             "sensor-axis self-attention. See lib/itransformer.py.")
    parser.add_argument("--encoder-num-layers", type=int, default=2,
                        help="iTransformer: TransformerEncoder depth over the sensor dimension.")
    parser.add_argument("--encoder-nhead",      type=int, default=8,
                        help="iTransformer: attention heads.")
    parser.add_argument("--encoder-dropout",    type=float, default=0.0,
                        help="iTransformer: dropout in encoder layers.")
    parser.add_argument("--encoder-pos-emb",    choices=POS_EMB_TYPES, default="learned",
                        help="iTransformer positional embedding: 'learned' (default, additive "
                             "learned nn.Parameter(S,D)), 'none' (permutation-invariant attention), "
                             "'rope' (rotary PE on Q/K in attention).")
    parser.add_argument("--tokenizer-type",     choices=["linear", "multiscale_conv"], default="linear",
                        help="iTransformer tokenizer. 'linear' (default): Linear(T, D) one-shot "
                             "projection — legacy behaviour. 'multiscale_conv': parallel dilated "
                             "Conv1d branches that preserve sub-window temporal structure.")
    parser.add_argument("--fault-head-location", choices=["decoder", "encoder"], default="decoder",
                        help="Where to place the per-sensor fault head. 'decoder' (default): inside "
                             "MidnC, reads post-encoder features — legacy behaviour. 'encoder': inside "
                             "iTransformerEncoder, reads raw per-sensor tokens before cross-sensor "
                             "attention mixing — more fault-robust at high nf. Only valid with "
                             "--encoder-type=itransformer + --use-fault-head.")
    # --- C-head ensemble decoding (Option 1) ---
    parser.add_argument("--c-calibrate-ensemble", action="store_true", default=False,
                        help="C-head: sweep ensemble ratio-threshold (α, β) on val and "
                             "save to calibration sidecar for eval-time soft-map decoding.")
    parser.add_argument("--c-ensemble-no-severity", action="store_true", default=False,
                        help="C-head ensemble: drop the severity factor from the slot weight "
                             "(weight = is_obj only, default: is_obj × severity).")
    # --- fault augmentation (all datasets) ---
    parser.add_argument("--norm-method", choices=["mean", "none"], default="none",
                        help="RMS normalization aggregation (7story only). Choices: 'mean' (global RMS), "
                             "'none' (no normalization, default). Pass '--norm-method mean' explicitly for "
                             "c-variants — the fault-aware head exploits the mean-RMS amplitude cue as a "
                             "fault-contrast signal (Insight #30). v1/b match original-paper preprocessing "
                             "under the default. The 'median' option was removed — it strictly regressed "
                             "at every cell of the norm-ablation table.")
    parser.add_argument("--p-hard",              type=float, default=0.0,
                        help="Per-sensor probability of hard fault (zero-out) during training.")
    parser.add_argument("--p-soft",              type=float, default=0.0,
                        help="Per-sensor probability of soft fault during training.")
    parser.add_argument("--p-struct-mask",       type=float, default=0.0,
                        help="Per-window probability of structured group masking (default 0.0 = off).")
    parser.add_argument("--eval-fault",       action="store_true", default=False,
                        help="Run evaluate_fault.py automatically after training")
    parser.add_argument("--tag",              default="",
                        help="Suffix appended to output dirs, e.g. 'fault' → states/qatar-fault/")
    parser.add_argument("--label",            default="",
                        help="Override model label for checkpoint/plot naming, e.g. 'c-fh-nobj02'")

    args = parser.parse_args()

    # Apply registry training overrides as argparse defaults (CLI wins)
    dataset = get_dataset(args.dataset)
    reg_train = dataset.training_overrides(args.model)
    if args.model == "dr" and args.bce_pos_weight is None and "pos_weight" in reg_train:
        args.bce_pos_weight = reg_train["pos_weight"]

    main(args)
