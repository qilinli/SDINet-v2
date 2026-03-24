"""
calibrate_b.py — post-hoc calibration for Approach-B without retraining.

Two methods, tuned on the combined val set then evaluated on test + real:

  1. Temperature scaling  — divides raw logits by T before sigmoid
                            optimises val map_mse (primary metric)
  2. Threshold tuning     — changes the sigmoid > θ decision boundary
                            optimises val F1 (at the temperature from step 1)

Usage::

    python calibrate_b.py --b states/b-combined-<uuid>.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from lib.data_safetensors import get_dataloaders
from lib.metrics import (
    distributed_map_b,
    evaluate_all,
    f1_from_counts,
    map_mse,
    presence_f1_stats,
)
from lib.testing import do_real_test_b, load_model_b_from_checkpoint

DATA_ROOT = "data/safetensors/unc=0"
SNR       = -1.0
SEED      = 42


# ---------------------------------------------------------------------------
# Collect val predictions for single + double combined
# ---------------------------------------------------------------------------

@torch.inference_mode()
def collect_val_predictions(
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (presence_logits, severity, y_norm) from both single+double val sets."""
    model.eval()
    all_p, all_s, all_y = [], [], []

    for subset in ("single", "double"):
        _, val_dl, _ = get_dataloaders(
            subset, SNR,
            root=DATA_ROOT, num_workers=0, eval_batch_size=256, seed=SEED,
        )
        for x, y in val_dl:
            p, s = model(x.to(device))
            all_p.append(p.cpu())
            all_s.append(s.cpu())
            all_y.append(y.cpu())

    return torch.cat(all_p), torch.cat(all_s), torch.cat(all_y)


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def sweep_temperature(
    logits: torch.Tensor,
    severity: torch.Tensor,
    y_norm: torch.Tensor,
    temps: list[float],
) -> tuple[float, float]:
    """Return (best_T, best_map_mse) by minimising map_mse over temperature grid."""
    best_T, best_mse = 1.0, float("inf")
    n = y_norm.size(0)

    for T in temps:
        scaled = logits / T
        dm     = distributed_map_b(scaled, severity)
        mse    = map_mse(dm, y_norm).item() / n
        if mse < best_mse:
            best_mse, best_T = mse, T

    return best_T, best_mse


def sweep_threshold(
    logits: torch.Tensor,
    y_norm: torch.Tensor,
    thresholds: list[float],
) -> tuple[float, float]:
    """Return (best_threshold, best_F1) by maximising F1 over threshold grid."""
    best_thr, best_f1 = 0.5, -1.0

    for thr in thresholds:
        tp, fp, fn = presence_f1_stats(logits, y_norm, threshold=thr)
        f1, _, _   = f1_from_counts(tp, fp, fn)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr, best_f1


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

METRICS = [
    "map_mse", "top_k_recall", "ap", "f1", "precision",
    "recall", "severity_mae", "mean_k_pred", "mean_k_true",
]
HEADER  = f"{'metric':<16}{'baseline':>12}{'temp-scaled':>14}{'thr-tuned':>12}"
SEP     = "-" * len(HEADER)


def print_comparison(baseline: dict, temp_only: dict, both: dict) -> None:
    print(HEADER)
    print(SEP)
    for m in METRICS:
        print(f"{m:<16}{baseline[m]:>12.4f}{temp_only[m]:>14.4f}{both[m]:>12.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(b_ckpt: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model_b_from_checkpoint(b_ckpt, device=device)

    # ---- collect val predictions ----
    print("Collecting val predictions …")
    val_p, val_s, val_y = collect_val_predictions(model, device)

    # ---- step 1: temperature sweep (optimise map_mse) ----
    temps  = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
    best_T, _ = sweep_temperature(val_p, val_s, val_y, temps)

    # fine-grained search around best_T
    lo, hi = max(0.05, best_T * 0.5), best_T * 2.0
    fine_temps = torch.linspace(lo, hi, 40).tolist()
    best_T, best_temp_mse = sweep_temperature(val_p, val_s, val_y, fine_temps)
    print(f"  Best temperature: T={best_T:.3f}  (val map_mse={best_temp_mse:.6f})")

    # ---- step 2: threshold sweep on temperature-scaled logits ----
    scaled_val_p = val_p / best_T
    thresholds   = torch.linspace(0.05, 0.95, 91).tolist()
    best_thr, best_f1 = sweep_threshold(scaled_val_p, val_y, thresholds)
    print(f"  Best threshold:   θ={best_thr:.3f}  (val F1={best_f1:.4f})\n")

    # ---- collect test predictions ----
    print("Evaluating on test sets …")

    def collect_test(subset: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, test_dl = get_dataloaders(
            subset, SNR,
            root=DATA_ROOT, num_workers=0, eval_batch_size=256, seed=SEED,
        )
        P, S, Y = [], [], []
        with torch.inference_mode():
            for x, y in test_dl:
                p, s = model(x.to(device))
                P.append(p.cpu()); S.append(s.cpu()); Y.append(y.cpu())
        return torch.cat(P), torch.cat(S), torch.cat(Y)

    for subset in ("single", "double"):
        test_p, test_s, test_y = collect_test(subset)
        scaled_test_p = test_p / best_T

        baseline  = evaluate_all(test_p,        test_s, test_y)
        temp_only = evaluate_all(scaled_test_p, test_s, test_y)
        both      = evaluate_all(scaled_test_p, test_s, test_y, threshold=best_thr)

        print(f"\n[test / {subset}]  (T={best_T:.3f}, θ={best_thr:.3f})")
        print_comparison(baseline, temp_only, both)

    # ---- real benchmark ----
    print("\n[real benchmark]")
    try:
        from lib.testing import load_real_test_tensors, DAMAGE_PHYSICAL_SCALE

        test_data, test_target = load_real_test_tensors()
        x_real = test_data[None, None, ...].to(device)

        with torch.inference_mode():
            model.eval()
            real_p, real_s = model(x_real)

        y_norm_real = (
            test_target.squeeze().to(device) / DAMAGE_PHYSICAL_SCALE - 1.0
        ).unsqueeze(0)

        real_scaled = real_p / best_T

        baseline  = evaluate_all(real_p,        real_s, y_norm_real)
        temp_only = evaluate_all(real_scaled,    real_s, y_norm_real)
        both      = evaluate_all(real_scaled,    real_s, y_norm_real, threshold=best_thr)

        print(f"  (T={best_T:.3f}, θ={best_thr:.3f})")
        print_comparison(baseline, temp_only, both)

    except (FileNotFoundError, OSError) as e:
        print(f"  Skipping (missing benchmark .mat): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", required=True, metavar="CKPT",
                        help="Approach-B checkpoint path")
    args = parser.parse_args()
    main(args.b)
