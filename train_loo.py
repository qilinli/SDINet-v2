"""
train_loo.py — leave-one-out training over the 5 Qatar double-damage recordings.

Trains the C-head 5 times, each time holding out one double-damage recording for
test and adding the other 4 to the training set.  Results are printed in a summary
table at the end.

Usage::

    CUDA_VISIBLE_DEVICES=2 python train_loo.py --epochs 200 --p-mix 0.5 --tag dd
    CUDA_VISIBLE_DEVICES=2 python train_loo.py --epochs 200 --p-mix 0.5 --tag dd --held-out 0 1 2

Double-damage recording index → file mapping:
    0: j03_j26.npz  (joints 3 & 26)
    1: j07_j14.npz  (joints 7 & 14)
    2: j13_j23.npz  (joints 13 & 23)
    3: j21_j25.npz  (joints 21 & 25)
    4: j23_j24.npz  (joints 23 & 24)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

DOUBLE_FILES = [
    "j03_j26 (joints 3 & 26)",
    "j07_j14 (joints 7 & 14)",
    "j13_j23 (joints 13 & 23)",
    "j21_j25 (joints 21 & 25)",
    "j23_j24 (joints 23 & 24)",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-out training over 5 Qatar double-damage recordings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--held-out", nargs="+", type=int, default=list(range(5)),
                        choices=[0, 1, 2, 3, 4],
                        metavar="IDX",
                        help="Which held-out indices to run (default: all 5)")
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch-size",      type=int,   default=128)
    parser.add_argument("--eval-batch-size", type=int,   default=32)
    parser.add_argument("--drop-rate",       type=float, default=0.0)
    parser.add_argument("--num-workers",     type=int,   default=0)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--root",            default=None)

    # backbone / neck
    parser.add_argument("--structure",    type=int, nargs=3, default=[6, 6, 6],
                        metavar=("B1", "B2", "B3"))
    parser.add_argument("--embed-dim",    type=int,   default=768)
    parser.add_argument("--neck-dropout", type=float, default=0.0)

    # Qatar windowing
    parser.add_argument("--window-size", type=int,   default=2048)
    parser.add_argument("--overlap",     type=float, default=0.5)
    parser.add_argument("--downsample",  type=int,   default=4)

    # C-head knobs
    parser.add_argument("--p-mix",              type=float, default=0.5)
    parser.add_argument("--num-slots",          type=int,   default=5)
    parser.add_argument("--num-decoder-layers", type=int,   default=2)
    parser.add_argument("--nhead",              type=int,   default=8)
    parser.add_argument("--no-obj-weight",      type=float, default=0.1)
    parser.add_argument("--loc-weight",         type=float, default=1.0)
    parser.add_argument("--sev-weight",         type=float, default=None)

    parser.add_argument("--tag", default="dd",
                        help="Output dir suffix → states/qatar-<tag>/")

    args = parser.parse_args()

    shared = [
        "--model",          "c",
        "--dataset",        "qatar",
        "--epochs",         str(args.epochs),
        "--batch-size",     str(args.batch_size),
        "--eval-batch-size",str(args.eval_batch_size),
        "--drop-rate",      str(args.drop_rate),
        "--num-workers",    str(args.num_workers),
        "--seed",           str(args.seed),
        "--structure",      *[str(b) for b in args.structure],
        "--embed-dim",      str(args.embed_dim),
        "--neck-dropout",   str(args.neck_dropout),
        "--window-size",    str(args.window_size),
        "--overlap",        str(args.overlap),
        "--downsample",     str(args.downsample),
        "--p-mix",          str(args.p_mix),
        "--num-slots",      str(args.num_slots),
        "--num-decoder-layers", str(args.num_decoder_layers),
        "--nhead",          str(args.nhead),
        "--no-obj-weight",  str(args.no_obj_weight),
        "--loc-weight",     str(args.loc_weight),
        "--tag",            args.tag,
    ]
    if args.root is not None:
        shared += ["--root", args.root]
    if args.sev_weight is not None:
        shared += ["--sev-weight", str(args.sev_weight)]

    exit_codes: dict[int, int] = {}

    for idx in args.held_out:
        sep = "=" * 60
        print(f"\n{sep}\n  LOO fold {idx}/4 — held out: {DOUBLE_FILES[idx]}\n{sep}\n")
        cmd = [sys.executable, "train.py"] + shared + ["--held-out-double", str(idx)]
        ret = subprocess.call(cmd)
        exit_codes[idx] = ret
        if ret != 0:
            print(f"[train_loo] train.py exited with code {ret} for held_out={idx}",
                  file=sys.stderr)

    # Summary
    print("\n" + "=" * 60)
    print("  LOO summary")
    print("=" * 60)
    for idx in args.held_out:
        status = "OK" if exit_codes[idx] == 0 else f"FAILED (code {exit_codes[idx]})"
        print(f"  held_out={idx}  {DOUBLE_FILES[idx]:30s}  {status}")

    if any(c != 0 for c in exit_codes.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
