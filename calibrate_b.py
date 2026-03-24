"""
calibrate_b.py — post-hoc calibration CLI for an existing B checkpoint.

Calibration runs automatically at the end of main_b.py and saves a JSON
sidecar next to the checkpoint.  Use this script to (re-)calibrate a
checkpoint manually, or to inspect the calibrated results.

Usage::

    python calibrate_b.py --b states/b-combined-<uuid>.pt
"""

from __future__ import annotations

import argparse

import torch

from lib.calibration import (
    calibrate_b,
    do_real_test_b_calibrated,
    eval_on_loader_b_calibrated,
    save_calibration,
)
from lib.data_safetensors import get_dataloaders
from lib.testing import load_model_b_from_checkpoint

DATA_ROOT = "data/safetensors/unc=0"
SNR       = -1.0
SEED      = 42

METRICS = [
    "map_mse", "top_k_recall", "ap", "f1", "precision",
    "recall", "severity_mae", "mean_k_pred", "mean_k_true",
]


def print_results(label: str, results: dict) -> None:
    print(f"\n[{label}]")
    for k in METRICS:
        print(f"  {k}: {results[k]:.4f}")


def main(b_ckpt: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model_b_from_checkpoint(b_ckpt, device=device)

    _, val_single, _ = get_dataloaders("single", SNR, root=DATA_ROOT,
                                       num_workers=0, eval_batch_size=256, seed=SEED)
    _, val_double, _ = get_dataloaders("double", SNR, root=DATA_ROOT,
                                       num_workers=0, eval_batch_size=256, seed=SEED)
    _, _, test_single = get_dataloaders("single", SNR, root=DATA_ROOT,
                                        num_workers=0, eval_batch_size=256, seed=SEED)
    _, _, test_double = get_dataloaders("double", SNR, root=DATA_ROOT,
                                        num_workers=0, eval_batch_size=256, seed=SEED)

    cal = calibrate_b(model, [val_single, val_double], device)
    save_calibration(cal, b_ckpt)

    for label, dl in [("test / single", test_single), ("test / double", test_double)]:
        print_results(label, eval_on_loader_b_calibrated(model, dl, device, **cal))

    try:
        print_results("real benchmark",
                      do_real_test_b_calibrated(model, device=device,
                                                print_result=False, **cal))
    except (FileNotFoundError, OSError) as e:
        print(f"\n[real benchmark] Skipping (missing .mat): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", required=True, metavar="CKPT",
                        help="Approach-B checkpoint path")
    args = parser.parse_args()
    main(args.b)
