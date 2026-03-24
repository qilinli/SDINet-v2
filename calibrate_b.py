"""
calibrate_b.py — compare absolute vs ratio thresholding for Approach-B.

Runs both calibration methods on the val set and evaluates on test + real,
printing a three-column comparison table:

    baseline          T=1, absolute θ=0.5   (uncalibrated)
    abs-threshold     T tuned, θ tuned       (original calibration)
    ratio-threshold   T tuned, α tuned, β tuned

Usage::

    python calibrate_b.py --b states/b-combined-<uuid>.pt
"""

from __future__ import annotations

import argparse

import torch

from lib.calibration import (
    calibrate_b,
    calibrate_b_ratio,
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


def _header(cols: list[str]) -> None:
    w = 14
    print(f"\n{'metric':<16}" + "".join(f"{c:>{w}}" for c in cols))
    print("-" * (16 + w * len(cols)))


def _row(metric: str, results: list[dict]) -> None:
    w = 14
    print(f"{metric:<16}" + "".join(f"{r[metric]:>{w}.4f}" for r in results))


def print_comparison(labels: list[str], results: list[dict]) -> None:
    _header(labels)
    for m in METRICS:
        _row(m, results)


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

    # ---- calibrate both methods ----
    print("=== Calibrating absolute threshold ===")
    cal_abs   = calibrate_b(model, [val_single, val_double], device)

    print("\n=== Calibrating ratio threshold ===")
    cal_ratio = calibrate_b_ratio(model, [val_single, val_double], device)

    # save ratio calibration as the new sidecar (richer method)
    save_calibration(cal_ratio, b_ckpt)

    # ---- evaluate on test sets ----
    baseline_kwargs  = {"temperature": 1.0,                   "threshold": 0.5}
    abs_kwargs       = {"temperature": cal_abs["temperature"], "threshold": cal_abs["threshold"]}
    ratio_kwargs     = {"temperature": cal_ratio["temperature"],
                        "ratio_alpha": cal_ratio["ratio_alpha"],
                        "ratio_beta":  cal_ratio["ratio_beta"]}

    labels = ["baseline", "abs-thresh", "ratio-thresh"]

    for subset, dl in [("single", test_single), ("double", test_double)]:
        results = [
            eval_on_loader_b_calibrated(model, dl, device, **baseline_kwargs),
            eval_on_loader_b_calibrated(model, dl, device, **abs_kwargs),
            eval_on_loader_b_calibrated(model, dl, device, **ratio_kwargs),
        ]
        print(f"\n{'='*60}\n  test / {subset}\n{'='*60}")
        print_comparison(labels, results)

    # ---- real benchmark ----
    print(f"\n{'='*60}\n  real benchmark\n{'='*60}")
    try:
        results = [
            do_real_test_b_calibrated(model, device=device, print_result=False, **baseline_kwargs),
            do_real_test_b_calibrated(model, device=device, print_result=False, **abs_kwargs),
            do_real_test_b_calibrated(model, device=device, print_result=False, **ratio_kwargs),
        ]
        print_comparison(labels, results)
    except (FileNotFoundError, OSError) as e:
        print(f"  Skipping (missing .mat): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", required=True, metavar="CKPT",
                        help="Approach-B checkpoint path")
    args = parser.parse_args()
    main(args.b)
