"""
lib/calibration.py — post-hoc calibration for Approach-B.

Two parameter-free methods tuned on the val set:

  temperature (T)   divide raw logits by T before sigmoid
                    optimises val map_mse (affects the continuous damage map)

  threshold (θ)     sigmoid > θ decision boundary for presence detection
                    optimises val F1 (affects counting / precision / recall)

Typical usage (called automatically at end of main_b.py)::

    cal = calibrate_b(model, val_loaders, device)
    # cal = {"temperature": 3.1, "threshold": 0.70}
    save_calibration(cal, ckpt_path)          # writes <ckpt>.json sidecar

Load later::

    cal = load_calibration(ckpt_path)         # reads sidecar, or returns defaults
    results = eval_on_loader_b_calibrated(model, dl, device, **cal)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from lib.metrics import (
    distributed_map_b,
    evaluate_all,
    f1_from_counts,
    map_mse,
    presence_f1_stats,
)

# ---------------------------------------------------------------------------
# Calibration grid defaults
# ---------------------------------------------------------------------------

_TEMP_COARSE  = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
_FINE_STEPS   = 40
_THR_STEPS    = 91   # 0.05 … 0.95


# ---------------------------------------------------------------------------
# Core sweeps
# ---------------------------------------------------------------------------

def _sweep_temperature(
    logits: torch.Tensor,
    severity: torch.Tensor,
    y_norm: torch.Tensor,
    temps: list[float],
) -> tuple[float, float]:
    """Return (best_T, best_map_mse)."""
    best_T, best_mse = 1.0, float("inf")
    n = y_norm.size(0)
    for T in temps:
        dm  = distributed_map_b(logits / T, severity)
        mse = map_mse(dm, y_norm).item() / n
        if mse < best_mse:
            best_mse, best_T = mse, T
    return best_T, best_mse


def _sweep_threshold(
    logits: torch.Tensor,
    y_norm: torch.Tensor,
    thresholds: list[float],
) -> tuple[float, float]:
    """Return (best_threshold, best_F1)."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        tp, fp, fn = presence_f1_stats(logits, y_norm, threshold=thr)
        f1, _, _   = f1_from_counts(tp, fp, fn)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.inference_mode()
def calibrate_b(
    model: torch.nn.Module,
    val_loaders,          # iterable of DataLoaders (e.g. [val_single, val_double])
    device: torch.device | str,
) -> dict[str, float]:
    """
    Tune temperature and threshold on validation data.

    Args:
        model:       Trained MidnB model.
        val_loaders: One or more DataLoaders whose predictions are pooled for calibration.
        device:      Device to run inference on.

    Returns:
        ``{"temperature": T, "threshold": θ}``
    """
    device = torch.device(device)
    model  = model.to(device)
    model.eval()

    # ---- collect val predictions ----
    all_p, all_s, all_y = [], [], []
    for dl in val_loaders:
        for x, y in dl:
            p, s = model(x.to(device))
            all_p.append(p.cpu())
            all_s.append(s.cpu())
            all_y.append(y.cpu())
    logits   = torch.cat(all_p)
    severity = torch.cat(all_s)
    y_norm   = torch.cat(all_y)

    # ---- step 1: coarse → fine temperature sweep (optimise map_mse) ----
    best_T, _ = _sweep_temperature(logits, severity, y_norm, _TEMP_COARSE)
    lo        = max(0.05, best_T * 0.5)
    hi        = best_T * 2.0
    fine      = torch.linspace(lo, hi, _FINE_STEPS).tolist()
    best_T, _ = _sweep_temperature(logits, severity, y_norm, fine)

    # ---- step 2: threshold sweep on temperature-scaled logits (optimise F1) ----
    scaled_logits      = logits / best_T
    thresholds         = torch.linspace(0.05, 0.95, _THR_STEPS).tolist()
    best_thr, best_f1  = _sweep_threshold(scaled_logits, y_norm, thresholds)

    print(f"[calibration] temperature={best_T:.3f}  threshold={best_thr:.3f}  (val F1={best_f1:.4f})")
    return {"temperature": best_T, "threshold": best_thr}


def save_calibration(cal: dict[str, float], ckpt_path: str | Path) -> Path:
    """Write calibration params as a JSON sidecar next to the checkpoint."""
    sidecar = Path(ckpt_path).with_suffix(".json")
    sidecar.write_text(json.dumps(cal, indent=2))
    print(f"[calibration] Saved: {sidecar}")
    return sidecar


def load_calibration(
    ckpt_path: str | Path,
    default_temperature: float = 1.0,
    default_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Load calibration params from the JSON sidecar next to a checkpoint.

    Returns defaults if the sidecar does not exist (uncalibrated model).
    """
    sidecar = Path(ckpt_path).with_suffix(".json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    return {"temperature": default_temperature, "threshold": default_threshold}


# ---------------------------------------------------------------------------
# Calibrated eval helpers (mirror eval_on_loader_b / do_real_test_b)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_on_loader_b_calibrated(
    model: torch.nn.Module,
    dl,
    device: torch.device | str | None = None,
    temperature: float = 1.0,
    threshold: float = 0.5,
) -> dict[str, float]:
    """``evaluate_all`` with temperature scaling + custom threshold."""
    from lib.metrics import evaluate_all

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model  = model.to(device)
    model.eval()

    P, S, Y = [], [], []
    for x, y in dl:
        p, s = model(x.to(device))
        P.append(p.cpu()); S.append(s.cpu()); Y.append(y.cpu())

    logits = torch.cat(P) / temperature
    return evaluate_all(logits, torch.cat(S), torch.cat(Y), threshold=threshold)


@torch.inference_mode()
def do_real_test_b_calibrated(
    model: torch.nn.Module,
    device: torch.device | str | None = None,
    temperature: float = 1.0,
    threshold: float = 0.5,
    print_result: bool = True,
) -> dict[str, float]:
    """Real .mat benchmark eval with temperature scaling + custom threshold."""
    from lib.testing import load_real_test_tensors, DAMAGE_PHYSICAL_SCALE

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model  = model.to(device)
    model.eval()

    test_data, test_target = load_real_test_tensors()
    x = test_data[None, None, ...].to(device)

    presence_logits, severity = model(x)
    y_norm = (test_target.squeeze().to(device) / DAMAGE_PHYSICAL_SCALE - 1.0).unsqueeze(0)

    result = evaluate_all(presence_logits / temperature, severity, y_norm, threshold=threshold)

    if print_result:
        from lib.testing import _print_eval_results
        _print_eval_results(result)

    return result
