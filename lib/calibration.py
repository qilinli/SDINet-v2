"""
lib/calibration.py — post-hoc calibration for v1, C, and DR heads.

Temperature scaling and ratio thresholding tuned on the val set.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from lib.metrics import (
    PRESENCE_NORM_THRESH,
    _c_slot_decode,
    distributed_map_v1,
    evaluate_all_c,
    evaluate_all_dr,
    evaluate_all_v1,
    f1_from_counts,
    map_mse,
    presence_f1_stats,
)

# ---------------------------------------------------------------------------
# Calibration grid defaults
# ---------------------------------------------------------------------------

_TEMP_COARSE  = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
_FINE_STEPS   = 40
_THR_STEPS    = 91         # 0.05 … 0.95  (absolute threshold)
_ALPHA_STEPS  = 81         # 0.10 … 0.90  (ratio α)
_BETA_VALUES  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]  # absolute gate β


# ---------------------------------------------------------------------------
# Core sweeps
# ---------------------------------------------------------------------------

def _sweep_temperature(map_fn, pred1, pred2, y_norm, temps):
    """Return (best_T, best_map_mse). map_fn(pred1/T, pred2) must return a (N,L) damage map."""
    best_T, best_mse = 1.0, float("inf")
    n = y_norm.size(0)
    for T in temps:
        mse = map_mse(map_fn(pred1 / T, pred2), y_norm).item() / n
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


def _sweep_ratio_threshold(probs, y_norm, alphas, betas):
    """
    Return (best_alpha, best_beta, best_F1) by 2D grid search.

    Works on any per-location probability/score in [0,1] — sigmoid(logits) for B/DR.
    Ratio rule: predict l if probs[l] > alpha * max(probs)  AND  max(probs) > beta.
    """
    best_alpha, best_beta, best_f1 = 0.5, 0.0, -1.0
    max_prob = probs.max(dim=-1, keepdim=True).values
    y_pres   = y_norm > PRESENCE_NORM_THRESH

    for alpha in alphas:
        for beta in betas:
            pred = (probs > alpha * max_prob) & (max_prob > beta)
            tp   = (pred &  y_pres).sum().long()
            fp   = (pred & ~y_pres).sum().long()
            fn   = (~pred & y_pres).sum().long()
            f1, _, _ = f1_from_counts(tp, fp, fn)
            if f1 > best_f1:
                best_f1, best_alpha, best_beta = f1, alpha, beta

    return best_alpha, best_beta, best_f1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.inference_mode()
def calibrate_v1_ratio(
    model: torch.nn.Module,
    val_loaders,
    device: torch.device | str,
) -> dict[str, float]:
    """
    Tune temperature + ratio threshold (α, β) for v1 on validation data.

    Temperature T scales ``loc_pred / T`` before softmax, sharpening or
    softening the location distribution (optimises val map_mse).
    Ratio rule: predict location l if
        softmax(loc/T)[l] > alpha * max_l softmax(loc/T)[l]   AND
        max_l softmax(loc/T)[l] > beta

    Returns:
        ``{"temperature": T, "ratio_alpha": α, "ratio_beta": β}``
    """
    device = torch.device(device)
    model  = model.to(device)
    model.eval()

    all_d, all_l, all_y = [], [], []
    for dl in val_loaders:
        for x, y in dl:
            d, l = model(x.to(device))
            all_d.append(d.cpu()); all_l.append(l.cpu()); all_y.append(y.cpu())
    dmg    = torch.cat(all_d)
    loc    = torch.cat(all_l)
    y_norm = torch.cat(all_y)

    # Step 1: coarse → fine temperature sweep (optimise map_mse)
    # _sweep_temperature(map_fn, pred1, pred2, ...) calls map_fn(pred1/T, pred2)
    # so pred1=loc, pred2=dmg, map_fn scales loc
    map_fn = lambda loc_scaled, d: distributed_map_v1(d, loc_scaled)
    best_T, _ = _sweep_temperature(map_fn, loc, dmg, y_norm, _TEMP_COARSE)
    lo, hi    = max(0.05, best_T * 0.5), best_T * 2.0
    best_T, _ = _sweep_temperature(map_fn, loc, dmg, y_norm,
                                   torch.linspace(lo, hi, _FINE_STEPS).tolist())

    # Step 2: 2D sweep over (α, β) on temperature-scaled softmax probs
    probs = torch.softmax(loc / best_T, dim=-1)             # (N, L) ∈ [0, 1]
    alphas = torch.linspace(0.10, 0.90, _ALPHA_STEPS).tolist()
    best_alpha, best_beta, best_f1 = _sweep_ratio_threshold(
        probs, y_norm, alphas, _BETA_VALUES
    )

    print(f"[calibration/v1/ratio] temperature={best_T:.3f}  "
          f"alpha={best_alpha:.3f}  beta={best_beta:.3f}  (val F1={best_f1:.4f})")
    return {"temperature": best_T, "ratio_alpha": best_alpha, "ratio_beta": best_beta}


@torch.inference_mode()
def eval_on_loader_v1_calibrated(
    model: torch.nn.Module,
    dl,
    device: torch.device | str | None = None,
    temperature: float = 1.0,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
) -> dict[str, float]:
    """``evaluate_all_v1`` with temperature scaling + ratio thresholding."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model  = model.to(device)
    model.eval()

    D, L, Y = [], [], []
    for x, y in dl:
        d, l = model(x.to(device))
        D.append(d.cpu()); L.append(l.cpu()); Y.append(y.cpu())

    kwargs = {"ratio_alpha": ratio_alpha, "ratio_beta": ratio_beta} \
             if ratio_alpha is not None else {}
    return evaluate_all_v1(torch.cat(D), torch.cat(L), torch.cat(Y),
                           temperature=temperature, **kwargs)


@torch.inference_mode()
def do_real_test_v1_calibrated(
    model: torch.nn.Module,
    device: torch.device | str | None = None,
    temperature: float = 1.0,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
    print_result: bool = True,
    spec=None,
) -> dict[str, float]:
    """Real .mat benchmark eval for v1 with temperature + ratio thresholding."""
    from lib.data_safetensors import load_real_test_tensors, DAMAGE_PHYSICAL_SCALE, DEFAULT_BENCHMARK

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model  = model.to(device)
    model.eval()

    test_data, test_target = load_real_test_tensors(spec=spec if spec is not None else DEFAULT_BENCHMARK)
    x = test_data[None, None, ...].to(device)

    dmg_pred, loc_pred = model(x)
    y_norm = (test_target.squeeze().to(device) / DAMAGE_PHYSICAL_SCALE - 1.0).unsqueeze(0)

    kwargs = {"ratio_alpha": ratio_alpha, "ratio_beta": ratio_beta} \
             if ratio_alpha is not None else {}
    result = evaluate_all_v1(dmg_pred, loc_pred, y_norm, temperature=temperature, **kwargs)

    if print_result:
        from lib.data_safetensors import _print_eval_results
        _print_eval_results(result)

    return result


# ---------------------------------------------------------------------------
# Approach-C calibration
# ---------------------------------------------------------------------------

@torch.inference_mode()
def calibrate_c_obj_threshold(
    model: torch.nn.Module,
    val_loaders,
    device: torch.device | str,
) -> dict[str, float]:
    """
    Find the is-object threshold that maximises F1 for the C-head.

    In pure DETR decoding a slot is active when its is-object score exceeds
    ``obj_threshold``.  We sweep over [0.01, 0.99] and pick the value with
    the best val F1.

    Returns:
        ``{"obj_threshold": θ}``
    """
    device = torch.device(device)
    model  = model.to(device)
    model.eval()

    all_l, all_s, all_y = [], [], []
    for dl in val_loaders:
        for x, y in dl:
            l, s = model(x.to(device))
            all_l.append(l.cpu()); all_s.append(s.cpu()); all_y.append(y.cpu())
    loc_logits = torch.cat(all_l)
    y_norm     = torch.cat(all_y)

    is_obj, pred_loc = _c_slot_decode(loc_logits)   # (N, K) each
    y_pres = y_norm > PRESENCE_NORM_THRESH           # (N, L) bool

    thresholds = torch.linspace(0.01, 0.99, 99).tolist()
    best_f1, best_theta = -1.0, 0.5
    for theta in thresholds:
        active = is_obj > theta
        tp = fp = fn = 0
        for b in range(y_norm.size(0)):
            pred_set = set(pred_loc[b, active[b]].tolist())
            true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
            tp += len(pred_set & true_set)
            fp += len(pred_set - true_set)
            fn += len(true_set - pred_set)
        _, _, f1 = f1_from_counts(tp, fp, fn)
        if f1 > best_f1:
            best_f1, best_theta = f1, theta

    print(f"[calibration/C] obj_threshold={best_theta:.3f}  (val F1={best_f1:.4f})")
    return {"obj_threshold": best_theta}


# ---------------------------------------------------------------------------
# Approach-DR calibration
# ---------------------------------------------------------------------------

@torch.inference_mode()
def calibrate_dr_ratio(
    model: torch.nn.Module,
    val_loaders,
    device: torch.device | str,
) -> dict[str, float]:
    """
    Tune ratio threshold (α, β) for the DR head on validation data.

    DR outputs sigmoid-activated values ∈ [0, 1] directly, so temperature
    scaling does not apply.  Only the ratio detection threshold is tuned.

    Ratio rule: predict location l if
        pred[l] > alpha * max_l pred[l]   AND   max_l pred[l] > beta

    Returns:
        ``{"ratio_alpha": α, "ratio_beta": β}``
    """
    device = torch.device(device)
    model  = model.to(device)
    model.eval()

    all_p, all_y = [], []
    for dl in val_loaders:
        for x, y in dl:
            p = model(x.to(device))
            all_p.append(p.cpu()); all_y.append(y.cpu())
    pred   = torch.cat(all_p)
    y_norm = torch.cat(all_y)

    alphas = torch.linspace(0.10, 0.90, _ALPHA_STEPS).tolist()
    best_alpha, best_beta, best_f1 = _sweep_ratio_threshold(
        pred, y_norm, alphas, _BETA_VALUES
    )

    print(f"[calibration/DR/ratio] alpha={best_alpha:.3f}  beta={best_beta:.3f}  (val F1={best_f1:.4f})")
    return {"ratio_alpha": best_alpha, "ratio_beta": best_beta}


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
# Calibrated eval helpers for Approach-DR
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_on_loader_dr_calibrated(
    model: torch.nn.Module,
    dl,
    device: torch.device | str | None = None,
    threshold: float = 0.5,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
) -> dict[str, float]:
    """``evaluate_all_dr`` with absolute or ratio thresholding."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model  = model.to(device)
    model.eval()

    P, Y = [], []
    for x, y in dl:
        p = model(x.to(device))
        P.append(p.cpu()); Y.append(y.cpu())

    kwargs = {"ratio_alpha": ratio_alpha, "ratio_beta": ratio_beta} \
             if ratio_alpha is not None else {"threshold": threshold}
    return evaluate_all_dr(torch.cat(P), torch.cat(Y), **kwargs)


@torch.inference_mode()
def do_real_test_dr_calibrated(
    model: torch.nn.Module,
    device: torch.device | str | None = None,
    threshold: float = 0.5,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
    print_result: bool = True,
    spec=None,
) -> dict[str, float]:
    """Real .mat benchmark eval for DR head with absolute or ratio thresholding."""
    from lib.data_safetensors import load_real_test_tensors, DAMAGE_PHYSICAL_SCALE, DEFAULT_BENCHMARK

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model  = model.to(device)
    model.eval()

    test_data, test_target = load_real_test_tensors(spec=spec if spec is not None else DEFAULT_BENCHMARK)
    x = test_data[None, None, ...].to(device)

    pred   = model(x)
    y_norm = (test_target.squeeze().to(device) / DAMAGE_PHYSICAL_SCALE - 1.0).unsqueeze(0)

    kwargs = {"ratio_alpha": ratio_alpha, "ratio_beta": ratio_beta} \
             if ratio_alpha is not None else {"threshold": threshold}
    result = evaluate_all_dr(pred, y_norm, **kwargs)

    if print_result:
        from lib.data_safetensors import _print_eval_results
        _print_eval_results(result)

    return result


# ---------------------------------------------------------------------------
# Calibrated eval helpers for Approach-C
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_on_loader_c_calibrated(
    model: torch.nn.Module,
    dl,
    device: torch.device | str | None = None,
    obj_threshold: float = 0.5,
    **_,
) -> dict[str, float]:
    """``evaluate_all_c`` with calibrated is-object threshold."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model  = model.to(device)
    model.eval()

    L, S, Y = [], [], []
    for x, y in dl:
        l, s = model(x.to(device))
        L.append(l.cpu()); S.append(s.cpu()); Y.append(y.cpu())

    return evaluate_all_c(torch.cat(L), torch.cat(S), torch.cat(Y),
                          obj_threshold=obj_threshold)


@torch.inference_mode()
def do_real_test_c_calibrated(
    model: torch.nn.Module,
    device: torch.device | str | None = None,
    obj_threshold: float = 0.5,
    print_result: bool = True,
    spec=None,
    **_,
) -> dict[str, float]:
    """Real .mat benchmark eval for C-head with calibrated is-object threshold."""
    from lib.data_safetensors import load_real_test_tensors, DAMAGE_PHYSICAL_SCALE, DEFAULT_BENCHMARK

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model  = model.to(device)
    model.eval()

    test_data, test_target = load_real_test_tensors(spec=spec if spec is not None else DEFAULT_BENCHMARK)
    x = test_data[None, None, ...].to(device)

    loc_logits, severity = model(x)
    y_norm = (test_target.squeeze().to(device) / DAMAGE_PHYSICAL_SCALE - 1.0).unsqueeze(0)

    result = evaluate_all_c(loc_logits, severity, y_norm, obj_threshold=obj_threshold)

    if print_result:
        from lib.data_safetensors import _print_eval_results
        _print_eval_results(result)

    return result
