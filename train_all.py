"""
train_all.py — run v1, C, and/or DR training sequentially via train.py.

Usage::

    python train_all.py --dataset 7story               # v1 + dr, 200 epochs each
    python train_all.py --dataset qatar --models v1 c dr --epochs 50
    python train_all.py --dataset tower --models dr --epochs 100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os

from lib.datasets import DATASETS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train v1, C, DR sequentially on a given dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASETS),
                        help="Dataset to train on")
    parser.add_argument("--models", nargs="+", default=["v1", "dr"],
                        choices=["v1", "c", "dr", "b"],
                        help="Which models to train")
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch-size",      type=int,   default=128)
    parser.add_argument("--eval-batch-size", type=int,   default=32)
    parser.add_argument("--drop-rate",       type=float, default=0.0)
    parser.add_argument("--num-workers",     type=int,   default=0)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--root",            default=None,
                        help="Override dataset root path")

    # backbone / neck
    parser.add_argument("--structure",    type=int, nargs=3, default=[6, 6, 6],
                        metavar=("B1", "B2", "B3"))
    parser.add_argument("--embed-dim",    type=int,   default=768)
    parser.add_argument("--neck-dropout", type=float, default=0.0)

    # dataset-specific
    parser.add_argument("--subsets",          nargs="+", default=None,
                        choices=["healthy", "single", "double"],
                        help="Training subsets for 7story datasets (default: single double)")
    parser.add_argument("--tower-excitation", nargs="+", default=["EQ", "WN", "sine"])
    parser.add_argument("--window-size",      type=int,   default=2048)
    parser.add_argument("--overlap",          type=float, default=0.5)
    parser.add_argument("--downsample",       type=int,   default=4)

    # model-specific (passed through to train.py which ignores irrelevant ones)
    parser.add_argument("--tag",                 default="",
                        help="Output dir suffix, e.g. 'pmix' → states/qatar-pmix/")
    parser.add_argument("--held-out-double",    type=int,   default=None,
                        choices=[0, 1, 2, 3, 4],
                        help="Hold out this double-damage recording index (0-4) for test; "
                             "add the other 4 to training (Qatar only).")
    parser.add_argument("--split-double",       action="store_true", default=False,
                        help="Use first½ of all 5 double-damage recordings for training, "
                             "second½ for test (Qatar only).")
    parser.add_argument("--bce-pos-weight",     type=float, default=None)
    parser.add_argument("--num-slots",          type=int,   default=5)
    parser.add_argument("--num-decoder-layers", type=int,   default=2)
    parser.add_argument("--nhead",              type=int,   default=8)
    parser.add_argument("--no-obj-weight",      type=float, default=0.1)
    parser.add_argument("--loc-weight",         type=float, default=1.0)
    parser.add_argument("--sev-weight",         type=float, default=None)
    # --- sensor spatial reasoning ---
    parser.add_argument("--use-spatial-layer",   action="store_true", default=False)
    parser.add_argument("--num-spatial-layers",  type=int,   default=1)
    parser.add_argument("--spatial-nhead",       type=int,   default=8)
    # --- fault detection head ---
    parser.add_argument("--use-fault-head",      action="store_true", default=False)
    parser.add_argument("--fault-loss-weight",   type=float, default=1.0)
    parser.add_argument("--fault-pos-weight",    type=float, default=5.0)
    parser.add_argument("--use-structural-bias", action="store_true", default=False)
    # --- fault augmentation ---
    parser.add_argument("--p-hard",              type=float, default=0.0)
    parser.add_argument("--p-soft",              type=float, default=0.0)
    parser.add_argument("--p-struct-mask",       type=float, default=0.0)

    args = parser.parse_args()

    # Build the shared flags to forward to train.py
    shared = [
        "--dataset",        args.dataset,
        "--epochs",         str(args.epochs),
        "--batch-size",     str(args.batch_size),
        "--eval-batch-size",str(args.eval_batch_size),
        "--drop-rate",      str(args.drop_rate),
        "--num-workers",    str(args.num_workers),
        "--seed",           str(args.seed),
        "--structure",      *[str(b) for b in args.structure],
        "--embed-dim",      str(args.embed_dim),
        "--neck-dropout",   str(args.neck_dropout),
    ]
    if args.subsets:
        shared += ["--subsets"] + args.subsets
    shared += [
        "--tower-excitation", *args.tower_excitation,
        "--window-size",    str(args.window_size),
        "--overlap",        str(args.overlap),
        "--downsample",     str(args.downsample),
        "--num-decoder-layers", str(args.num_decoder_layers),
        "--nhead",          str(args.nhead),
        "--no-obj-weight",  str(args.no_obj_weight),
        "--loc-weight",     str(args.loc_weight),
    ]
    if args.tag:
        shared += ["--tag", args.tag]
    if args.held_out_double is not None:
        shared += ["--held-out-double", str(args.held_out_double)]
    if args.split_double:
        shared += ["--split-double"]
    if args.root is not None:
        shared += ["--root", args.root]
    if args.bce_pos_weight is not None:
        shared += ["--bce-pos-weight", str(args.bce_pos_weight)]
    if args.num_slots is not None:
        shared += ["--num-slots", str(args.num_slots)]
    if args.sev_weight is not None:
        shared += ["--sev-weight", str(args.sev_weight)]
    if args.use_spatial_layer:
        shared += ["--use-spatial-layer"]
    shared += ["--num-spatial-layers", str(args.num_spatial_layers),
               "--spatial-nhead",      str(args.spatial_nhead)]
    if args.use_fault_head:
        shared += ["--use-fault-head"]
    shared += ["--fault-loss-weight", str(args.fault_loss_weight),
               "--fault-pos-weight",  str(args.fault_pos_weight)]
    if args.use_structural_bias:
        shared += ["--use-structural-bias"]
    if args.p_hard > 0.0:
        shared += ["--p-hard", str(args.p_hard)]
    if args.p_soft > 0.0:
        shared += ["--p-soft", str(args.p_soft)]
    if args.p_struct_mask > 0.0:
        shared += ["--p-struct-mask", str(args.p_struct_mask)]

    for model in args.models:
        sep = "=" * 60
        print(f"\n{sep}\n  Training {model.upper()}  ({args.epochs} epochs) on {args.dataset}\n{sep}\n")
        cmd = [sys.executable, "train.py", "--model", model] + shared
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[train_all] train.py exited with code {ret} for model={model}", file=sys.stderr)
            sys.exit(ret)


if __name__ == "__main__":
    main()
