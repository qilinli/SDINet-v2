"""
train_all.py — train all four model × subset combinations sequentially.

Combinations:
  1. v1   + single
  2. v1   + double
  3. B    + single
  4. B    + double

Each run saves a checkpoint under ``states/`` and optionally runs the real
benchmark at the end.  After all runs complete, ``evaluate.py`` can be used
to compare any pair of checkpoints.

Usage::

    python train_all.py                        # all four, 200 epochs each
    python train_all.py --epochs 50            # quick smoke-test
    python train_all.py --subsets single       # only single-damage rows
    python train_all.py --models b             # only Approach-B rows
"""

from __future__ import annotations

import argparse

from main   import RunConfig,  main as train_v1
from main_b import RunConfigB, main as train_b

# pos_weight rule of thumb: (L - K) / K  where L=70
_POS_WEIGHT = {"single": 69.0, "double": 34.0}


def run_all(
    subsets: list[str],
    models: list[str],
    epochs: int,
) -> None:
    for subset in subsets:
        if "v1" in models:
            print(f"\n{'='*60}")
            print(f"  Training v1 on [{subset}]  ({epochs} epochs)")
            print(f"{'='*60}\n")
            train_v1(RunConfig(subset_name=subset, epochs=epochs))

        if "b" in models:
            print(f"\n{'='*60}")
            print(f"  Training B  on [{subset}]  ({epochs} epochs)")
            print(f"{'='*60}\n")
            train_b(RunConfigB(
                subset_name=subset,
                epochs=epochs,
                presence_pos_weight=_POS_WEIGHT[subset],
            ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all v1 / B × single / double combinations")
    parser.add_argument("--subsets", nargs="+", default=["single", "double"],
                        choices=["single", "double"],
                        help="Which subsets to train on (default: both)")
    parser.add_argument("--models", nargs="+", default=["v1", "b"],
                        choices=["v1", "b"],
                        help="Which model heads to train (default: both)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Epochs per run (default: 200)")
    args = parser.parse_args()

    run_all(args.subsets, args.models, args.epochs)


if __name__ == "__main__":
    main()
