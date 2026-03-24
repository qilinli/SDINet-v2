"""
evaluate.py — full model comparison table.

Evaluates a v1 checkpoint and an Approach-B checkpoint on:
  • single-damage test set
  • double-damage test set
  • real-world MATLAB benchmark

and prints a side-by-side metric table.

Usage::

    python evaluate.py \\
        --v1 states/single-damage-sparse-<uuid>.pt \\
        --b  states/multi-damage-B-<uuid>.pt

Both checkpoints are required.  Optional flags::

    --data-root   path to safetensors root  (default: data/safetensors/unc=0)
    --snr         SNR for noise injection   (default: -1.0 = no noise)
    --seed        dataloader split seed     (default: 42)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lib.calibration import (
    do_real_test_b_calibrated,
    eval_on_loader_b_calibrated,
    load_calibration,
)
from lib.data_safetensors import get_dataloaders
from lib.testing import (
    do_real_test,
    eval_on_loader_v1,
    load_model_from_checkpoint,
    load_model_b_from_checkpoint,
)

METRICS = [
    "map_mse",
    "top_k_recall",
    "ap",
    "f1",
    "precision",
    "recall",
    "severity_mae",
    "mean_k_pred",
    "mean_k_true",
]


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def run_evaluation(
    v1_ckpt: str | Path,
    b_ckpt: str | Path,
    *,
    data_root: str = "data/safetensors/unc=0",
    snr: float = -1.0,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Run the full evaluation suite and return a dict of metric dicts.

    Keys: ``v1/single``, ``v1/double``, ``v1/real``, ``B/single``, ``B/double``, ``B/real``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_v1 = load_model_from_checkpoint(v1_ckpt, device=device)
    model_b  = load_model_b_from_checkpoint(b_ckpt,  device=device)

    cal = load_calibration(b_ckpt)
    b_label = f"B (T={cal['temperature']:.2f}, θ={cal['threshold']:.2f})"
    print(f"[calibration] B checkpoint: {b_label}")

    *_, test_single = get_dataloaders("single", snr, root=data_root, num_workers=0, seed=seed)
    *_, test_double = get_dataloaders("double", snr, root=data_root, num_workers=0, seed=seed)

    return {
        "v1/single":  eval_on_loader_v1(model_v1, test_single, device),
        "v1/double":  eval_on_loader_v1(model_v1, test_double, device),
        "v1/real":    do_real_test(model_v1, device=device, print_result=False),
        "B/single":   eval_on_loader_b_calibrated(model_b, test_single, device, **cal),
        "B/double":   eval_on_loader_b_calibrated(model_b, test_double, device, **cal),
        "B/real":     do_real_test_b_calibrated(model_b, device=device, print_result=False, **cal),
    }


def print_table(results: dict[str, dict[str, float]]) -> None:
    cols = list(results.keys())
    col_w = 12
    print(f"{'metric':<16}" + "".join(f"{c:>{col_w}}" for c in cols))
    print("-" * (16 + col_w * len(cols)))
    for m in METRICS:
        print(f"{m:<16}" + "".join(f"{results[c][m]:>{col_w}.4f}" for c in cols))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate v1 and Approach-B checkpoints on single/double/real test sets."
    )
    parser.add_argument("--v1", required=True, metavar="CKPT",
                        help="v1 (Midn) checkpoint path")
    parser.add_argument("--b",  required=True, metavar="CKPT",
                        help="Approach-B (MidnB) checkpoint path")
    parser.add_argument("--data-root", default="data/safetensors/unc=0",
                        help="Safetensors data root (default: data/safetensors/unc=0)")
    parser.add_argument("--snr", type=float, default=-1.0,
                        help="SNR for noise injection (default: -1.0 = no noise)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Dataloader split seed (default: 42)")
    args = parser.parse_args()

    results = run_evaluation(
        args.v1, args.b,
        data_root=args.data_root,
        snr=args.snr,
        seed=args.seed,
    )
    print_table(results)


if __name__ == "__main__":
    main()
