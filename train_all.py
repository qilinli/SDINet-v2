"""
train_all.py — run both v1 and Approach-B training sequentially.

Both models train on the single+double combined dataset.

Usage::

    python train_all.py              # 200 epochs each
    python train_all.py --epochs 50  # quick smoke-test
    python train_all.py --models b   # only Approach-B
"""

from __future__ import annotations

import argparse

from main   import RunConfig,  main as train_v1
from main_b import RunConfigB, main as train_b


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train v1 and Approach-B on single+double combined data"
    )
    parser.add_argument("--models", nargs="+", default=["v1", "b"],
                        choices=["v1", "b"],
                        help="Which models to train (default: both)")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    if "v1" in args.models:
        print(f"\n{'='*60}\n  Training v1  ({args.epochs} epochs)\n{'='*60}\n")
        train_v1(RunConfig(epochs=args.epochs))

    if "b" in args.models:
        print(f"\n{'='*60}\n  Training B   ({args.epochs} epochs)\n{'='*60}\n")
        train_b(RunConfigB(epochs=args.epochs))


if __name__ == "__main__":
    main()
