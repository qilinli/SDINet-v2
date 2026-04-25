"""
evaluate_fault.py — Fault robustness evaluation for C, v1, DR, and B heads.

Supports Qatar, LUMO, and 7-story datasets via --dataset flag.

Qatar / LUMO: recording-consistent fault injection — for each recording,
n_faulted sensors are selected once and the same fault is applied to ALL
windows.  Matches physical reality (a disconnected cable affects the whole
test session).

7-story: per-batch fault injection — each simulation sample is independent,
so faults are injected fresh per batch.  Results are reported separately for
single-damage (K=1) and double-damage (K=2) ground-truth subsets.

ASCE: per-batch fault injection (same paradigm as 7-story — each simulated
scenario is independent).  K ∈ {0..5} per scenario so results are reported
as a single row per condition (no subset split).

Fault severity is specified via --fault-ratio (fraction of sensors), which is
converted to absolute counts per dataset (S=65/30/18).  Use --n-faulted for
backward-compatible absolute counts (overrides --fault-ratio).

Usage
-----
python evaluate_fault.py --dataset qatar --c <ckpt> [--v1 <ckpt>] [--dr <ckpt>] \\
  [--fault-ratio 0.0 0.1 0.33 0.5 0.67 0.8] [--n-repeats 3] \\
  [--root data/Qatar/processed] [--out saved_results/qatar/eval_fault]

python evaluate_fault.py --dataset lumo --b <ckpt> [--dr <ckpt>] [--v1 <ckpt>] \\
  [--root data/LUMO] [--out saved_results/lumo/eval_fault]

python evaluate_fault.py --dataset 7story --c <ckpt> [--v1 <ckpt>] [--dr <ckpt>] \\
  [--root data/7-story-frame/safetensors/unc=0] \\
  [--out saved_results/7story-fault/eval_fault]

python evaluate_fault.py --dataset asce --c <ckpt> [--v1 <ckpt>] [--b <ckpt>] \\
  [--root data/asce_hammer] \\
  [--out saved_results/asce-fault/eval_fault]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from lib.calibration import load_calibration
from lib.data_lumo import get_lumo_test_by_recording
from lib.data_qatar import get_qatar_test_by_recording, get_qatar_double_by_recording
from lib.datasets import DATASETS, QATAR_HELD_OUT_DOUBLE, get_dataset
from lib.faults import FAULT_TYPES, _apply_soft_fault, inject_faults_batch
from lib.metrics import (
    PRESENCE_NORM_THRESH,
    _c_slot_decode,
    _c_slot_decode_ensemble,
    f1_from_counts,
)
from lib.model import (
    load_model_c_from_checkpoint,
    load_model_dr_from_checkpoint,
    load_model_from_checkpoint,
)

_DMG_METRICS   = ["f1", "precision", "recall", "top_k_recall"]
_K0_METRICS    = ["sample_far", "mean_k_pred"]
_FAULT_METRICS = ["fault_f1", "fault_precision", "fault_recall"]
_ALL_METRICS   = _DMG_METRICS + _K0_METRICS + _FAULT_METRICS


# ---------------------------------------------------------------------------
# Shared fault injection loop
# ---------------------------------------------------------------------------

def _inject(x: torch.Tensor, sensor_idx: list[int], fault_type: str) -> torch.Tensor:
    """Apply fault in-place on a single window tensor (1, T, S)."""
    if fault_type == "hard":
        x[:, :, sensor_idx] = 0.0
    else:
        _apply_soft_fault(x, sensor_idx, fault_type)
    return x


def _inject_recording(x: torch.Tensor, sensor_idx: list[int], fault_type: str) -> None:
    """Apply fault in-place on a full recording (N, 1, T, S).

    One random parameter draw per call — recording-consistent fault magnitude.
    Per-window RMS is still used for relative scaling.
    """
    if fault_type == "hard":
        x[:, :, :, sensor_idx] = 0.0
        return

    seg = x[:, :, :, sensor_idx]                                    # (N, 1, T, k)
    rms = seg.pow(2).mean(dim=2, keepdim=True).sqrt() + 1e-8       # (N, 1, 1, k)

    if fault_type == "gain":
        scale = torch.rand(1).item()
        scale = scale * 0.30 + 0.50 if torch.rand(1).item() < 0.5 else scale * 0.50 + 1.20
        x[:, :, :, sensor_idx] = seg * scale
    elif fault_type == "bias":
        magnitude = (torch.rand(1).item() * 0.8 + 0.2) * rms
        sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0
        x[:, :, :, sensor_idx] = seg + sign * magnitude
    elif fault_type == "gain_bias":
        scale = torch.rand(1).item() * 0.40 + 0.50
        magnitude = (torch.rand(1).item() * 0.4 + 0.1) * rms
        sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0
        x[:, :, :, sensor_idx] = seg * scale + sign * magnitude
    elif fault_type == "noise":
        noise_scale = (torch.rand(1).item() * 1.5 + 0.5) * rms
        x[:, :, :, sensor_idx] = seg + torch.randn_like(seg) * noise_scale
    elif fault_type == "stuck":
        mean_val = seg.mean(dim=2, keepdim=True)                    # (N, 1, 1, k)
        x[:, :, :, sensor_idx] = mean_val + torch.randn_like(seg) * (0.01 * rms)
    elif fault_type == "partial":
        scale = torch.rand(1).item() * 0.40 + 0.30
        x[:, :, :, sensor_idx] = seg * scale


# ---------------------------------------------------------------------------
# C-head runner
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _run_condition_c(
    model: torch.nn.Module,
    recordings: list[tuple[torch.Tensor, torch.Tensor]],
    fault_type: str,
    n_faulted: int,
    n_repeats: int,
    batch_size: int,
    device: torch.device,
    ensemble_alpha: float | None = None,
    ensemble_beta: float = 0.0,
    ensemble_use_severity: bool = True,
) -> dict[str, float]:
    d_tp = d_fp = d_fn = 0
    tkr_hits = tkr_total = 0
    f_tp = f_fp = f_fn = 0
    has_fault_head = False

    repeats = max(1, n_repeats) if n_faulted > 0 else 1

    for repeat in range(repeats):
        rng = torch.Generator()
        rng.manual_seed(repeat * 997)

        for x, y in recordings:
            N, _, T, S = x.shape
            y_pres = y > PRESENCE_NORM_THRESH  # (N, L)

            if n_faulted == 0:
                x_f   = x
                y_flt = torch.zeros(N, S)
            else:
                sensor_idx = torch.randperm(S, generator=rng)[:n_faulted].tolist()
                x_f   = x.clone()
                y_flt = torch.zeros(N, S)
                y_flt[:, sensor_idx] = 1.0
                _inject_recording(x_f, sensor_idx, fault_type)

            all_loc, all_sev = [], []
            for i in range(0, N, batch_size):
                xb = x_f[i:i + batch_size].to(device)
                yf = y_flt[i:i + batch_size].to(device)
                loc_logits, severity, fault_prob, _ = model(xb)
                all_loc.append(loc_logits.cpu())
                all_sev.append(severity.cpu())

                if fault_prob is not None:
                    has_fault_head = True
                    pred_f = fault_prob >= 0.5
                    gt_f   = yf > 0.5
                    f_tp  += int((pred_f &  gt_f).sum())
                    f_fp  += int((pred_f & ~gt_f).sum())
                    f_fn  += int((~pred_f & gt_f).sum())

            loc_all = torch.cat(all_loc)
            sev_all = torch.cat(all_sev)
            k_true = y_pres.sum(-1)

            if ensemble_alpha is not None:
                M = _c_slot_decode_ensemble(loc_all, sev_all, use_severity=ensemble_use_severity)
                max_M = M.max(dim=-1, keepdim=True).values
                pred_pres = (M > ensemble_alpha * max_M) & (max_M > ensemble_beta)
                for b in range(N):
                    pred_set = set(pred_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                    true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                    d_tp += len(pred_set & true_set)
                    d_fp += len(pred_set - true_set)
                    d_fn += len(true_set - pred_set)
                    K = int(k_true[b].item())
                    if K > 0:
                        ps = set(M[b].topk(K).indices.tolist())
                        tkr_hits  += len(ps & true_set)
                        tkr_total += K
            else:
                active, pred_loc, is_obj = _c_slot_decode(loc_all)
                for b in range(N):
                    pred_set = set(pred_loc[b, active[b]].tolist())
                    true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                    d_tp += len(pred_set & true_set)
                    d_fp += len(pred_set - true_set)
                    d_fn += len(true_set - pred_set)

                    K = int(k_true[b].item())
                    if K > 0:
                        K_slots   = min(K, is_obj.size(-1))
                        top_slots = is_obj[b].topk(K_slots).indices
                        ps = set(pred_loc[b, top_slots].tolist())
                        tkr_hits  += len(ps & true_set)
                        tkr_total += K

    return _make_row(d_tp, d_fp, d_fn, tkr_hits, tkr_total,
                     f_tp, f_fp, f_fn, has_fault_head)


# ---------------------------------------------------------------------------
# v1-head runner
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _run_condition_v1(
    model: torch.nn.Module,
    recordings: list[tuple[torch.Tensor, torch.Tensor]],
    fault_type: str,
    n_faulted: int,
    n_repeats: int,
    batch_size: int,
    device: torch.device,
    temperature: float = 1.0,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
    dmg_gate: float | None = None,
) -> dict[str, float]:
    d_tp = d_fp = d_fn = 0
    tkr_hits = tkr_total = 0

    repeats = max(1, n_repeats) if n_faulted > 0 else 1

    for repeat in range(repeats):
        rng = torch.Generator()
        rng.manual_seed(repeat * 997)

        for x, y in recordings:
            N, _, T, S = x.shape
            y_pres = y > PRESENCE_NORM_THRESH  # (N, L)

            if n_faulted == 0:
                x_f = x
            else:
                sensor_idx = torch.randperm(S, generator=rng)[:n_faulted].tolist()
                x_f = x.clone()
                _inject_recording(x_f, sensor_idx, fault_type)

            all_probs = []
            all_dmg = []
            for i in range(0, N, batch_size):
                xb = x_f[i:i + batch_size].to(device)
                dmg, loc = model(xb)
                probs = torch.softmax(loc / temperature, dim=-1).cpu()  # (B, L)
                all_probs.append(probs)
                all_dmg.append(dmg.cpu())

            probs_all = torch.cat(all_probs)  # (N, L)
            dmg_all = torch.cat(all_dmg).squeeze(-1)  # (N,)

            # ratio threshold → predicted presence per sample
            if ratio_alpha is not None:
                max_p = probs_all.max(dim=-1, keepdim=True).values
                pred_pres = (probs_all > ratio_alpha * max_p) & (max_p > ratio_beta)
            else:
                pred_pres = probs_all > 0.5

            # dmg_gate: predict K=0 for windows where dmg < gate
            if dmg_gate is not None:
                pred_pres = pred_pres & (dmg_all > dmg_gate).unsqueeze(-1)

            k_true = y_pres.sum(-1)
            for b in range(N):
                pred_set = set(pred_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                d_tp += len(pred_set & true_set)
                d_fp += len(pred_set - true_set)
                d_fn += len(true_set - pred_set)

                K = int(k_true[b].item())
                if K > 0:
                    top_locs = probs_all[b].topk(K).indices.tolist()
                    tkr_hits  += len(set(top_locs) & true_set)
                    tkr_total += K

    return _make_row(d_tp, d_fp, d_fn, tkr_hits, tkr_total, 0, 0, 0, False)


# ---------------------------------------------------------------------------
# DR-head runner
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _run_condition_dr(
    model: torch.nn.Module,
    recordings: list[tuple[torch.Tensor, torch.Tensor]],
    fault_type: str,
    n_faulted: int,
    n_repeats: int,
    batch_size: int,
    device: torch.device,
    threshold: float = 0.5,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
) -> dict[str, float]:
    d_tp = d_fp = d_fn = 0
    tkr_hits = tkr_total = 0

    repeats = max(1, n_repeats) if n_faulted > 0 else 1

    for repeat in range(repeats):
        rng = torch.Generator()
        rng.manual_seed(repeat * 997)

        for x, y in recordings:
            N, _, T, S = x.shape
            y_pres = y > PRESENCE_NORM_THRESH  # (N, L)

            if n_faulted == 0:
                x_f = x
            else:
                sensor_idx = torch.randperm(S, generator=rng)[:n_faulted].tolist()
                x_f = x.clone()
                _inject_recording(x_f, sensor_idx, fault_type)

            all_pred = []
            for i in range(0, N, batch_size):
                xb = x_f[i:i + batch_size].to(device)
                pred = model(xb).cpu()  # (B, L)
                all_pred.append(pred)

            pred_all = torch.cat(all_pred)  # (N, L)

            if ratio_alpha is not None:
                max_p = pred_all.max(dim=-1, keepdim=True).values
                pred_pres = (pred_all > ratio_alpha * max_p) & (max_p > ratio_beta)
            else:
                pred_pres = pred_all > threshold

            k_true = y_pres.sum(-1)
            for b in range(N):
                pred_set = set(pred_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                d_tp += len(pred_set & true_set)
                d_fp += len(pred_set - true_set)
                d_fn += len(true_set - pred_set)

                K = int(k_true[b].item())
                if K > 0:
                    top_locs = pred_all[b].topk(K).indices.tolist()
                    tkr_hits  += len(set(top_locs) & true_set)
                    tkr_total += K

    return _make_row(d_tp, d_fp, d_fn, tkr_hits, tkr_total, 0, 0, 0, False)


# ---------------------------------------------------------------------------
# 7-story runners (per-batch fault injection, split by K=1 / K=2)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _run_condition_7story_c(
    model: torch.nn.Module,
    test_dl,
    fault_type: str,
    n_faulted: int,
    n_repeats: int,
    device: torch.device,
    ensemble_alpha: float | None = None,
    ensemble_beta: float = 0.0,
    ensemble_use_severity: bool = True,
) -> dict[str, dict[str, float]]:
    """Returns {"single": row, "double": row, "undamaged": row} keyed by K_true."""
    repeats = max(1, n_repeats) if n_faulted > 0 else 1
    def _new_counter():
        return dict(d_tp=0, d_fp=0, d_fn=0, tkr_hits=0, tkr_total=0,
                    f_tp=0, f_fp=0, f_fn=0, has_fault_head=False,
                    n_samples=0, n_fp_samples=0, total_k_pred=0)
    counters = {"single": _new_counter(), "double": _new_counter(), "undamaged": _new_counter()}

    for repeat in range(repeats):
        rng = torch.Generator()
        rng.manual_seed(repeat * 997)

        for batch in test_dl:
            x, y = batch[0].float().to(device), batch[1].to(device)
            if n_faulted > 0:
                x, y_fault_gt = inject_faults_batch(x, fault_type, n_faulted, rng)
                x = x.to(device)
                y_fault_gt = y_fault_gt.to(device)
            else:
                y_fault_gt = torch.zeros(x.size(0), x.size(-1), device=device)

            loc_logits, severity, fault_prob, _ = model(x)
            y_pres = y > PRESENCE_NORM_THRESH  # (B, L)
            k_true = y_pres.sum(-1)            # (B,)

            if ensemble_alpha is not None:
                M = _c_slot_decode_ensemble(loc_logits, severity, use_severity=ensemble_use_severity)
                max_M = M.max(dim=-1, keepdim=True).values
                pred_pres = (M > ensemble_alpha * max_M) & (max_M > ensemble_beta)
                for b in range(x.size(0)):
                    K = int(k_true[b].item())
                    subset = "undamaged" if K == 0 else ("double" if K >= 2 else "single")
                    c = counters[subset]

                    pred_set = set(pred_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                    true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                    fp_this = len(pred_set - true_set)
                    c["d_tp"] += len(pred_set & true_set)
                    c["d_fp"] += fp_this
                    c["d_fn"] += len(true_set - pred_set)
                    c["n_samples"]    += 1
                    c["total_k_pred"] += len(pred_set)
                    if fp_this > 0:
                        c["n_fp_samples"] += 1

                    if K > 0:
                        ps = set(M[b].topk(K).indices.tolist())
                        c["tkr_hits"]  += len(ps & true_set)
                        c["tkr_total"] += K

                    if fault_prob is not None:
                        c["has_fault_head"] = True
                        pred_f = fault_prob[b] >= 0.5
                        gt_f   = y_fault_gt[b] > 0.5
                        c["f_tp"] += int((pred_f &  gt_f).sum())
                        c["f_fp"] += int((pred_f & ~gt_f).sum())
                        c["f_fn"] += int((~pred_f & gt_f).sum())
            else:
                active, pred_loc, is_obj = _c_slot_decode(loc_logits)
                for b in range(x.size(0)):
                    K = int(k_true[b].item())
                    subset = "undamaged" if K == 0 else ("double" if K >= 2 else "single")
                    c = counters[subset]

                    pred_set = set(pred_loc[b, active[b]].tolist())
                    true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                    fp_this = len(pred_set - true_set)
                    c["d_tp"] += len(pred_set & true_set)
                    c["d_fp"] += fp_this
                    c["d_fn"] += len(true_set - pred_set)
                    c["n_samples"]    += 1
                    c["total_k_pred"] += len(pred_set)
                    if fp_this > 0:
                        c["n_fp_samples"] += 1

                    if K > 0:
                        K_slots   = min(K, is_obj.size(-1))
                        top_slots = is_obj[b].topk(K_slots).indices
                        ps = set(pred_loc[b, top_slots].tolist())
                        c["tkr_hits"]  += len(ps & true_set)
                        c["tkr_total"] += K

                    if fault_prob is not None:
                        c["has_fault_head"] = True
                        pred_f = fault_prob[b] >= 0.5
                        gt_f   = y_fault_gt[b] > 0.5
                        c["f_tp"] += int((pred_f &  gt_f).sum())
                        c["f_fp"] += int((pred_f & ~gt_f).sum())
                        c["f_fn"] += int((~pred_f & gt_f).sum())

    return {
        k: _make_row(c["d_tp"], c["d_fp"], c["d_fn"],
                     c["tkr_hits"], c["tkr_total"],
                     c["f_tp"],  c["f_fp"],  c["f_fn"],
                     c["has_fault_head"],
                     n_samples=c["n_samples"], n_fp_samples=c["n_fp_samples"],
                     total_k_pred=c["total_k_pred"],
                     is_undamaged=(k == "undamaged"))
        for k, c in counters.items() if c["n_samples"] > 0
    }


@torch.inference_mode()
def _run_condition_7story_v1(
    model: torch.nn.Module,
    test_dl,
    fault_type: str,
    n_faulted: int,
    n_repeats: int,
    device: torch.device,
    temperature: float = 1.0,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
    dmg_gate: float | None = None,
) -> dict[str, dict[str, float]]:
    repeats = max(1, n_repeats) if n_faulted > 0 else 1
    def _new_counter():
        return dict(d_tp=0, d_fp=0, d_fn=0, tkr_hits=0, tkr_total=0,
                    n_samples=0, n_fp_samples=0, total_k_pred=0)
    counters = {"single": _new_counter(), "double": _new_counter(), "undamaged": _new_counter()}

    for repeat in range(repeats):
        rng = torch.Generator()
        rng.manual_seed(repeat * 997)

        for batch in test_dl:
            x, y = batch[0].float().to(device), batch[1].to(device)
            if n_faulted > 0:
                x, _ = inject_faults_batch(x, fault_type, n_faulted, rng)
                x = x.to(device)

            dmg, loc = model(x)
            probs = torch.softmax(loc / temperature, dim=-1)  # (B, L)

            if ratio_alpha is not None:
                max_p = probs.max(dim=-1, keepdim=True).values
                pred_pres = (probs > ratio_alpha * max_p) & (max_p > ratio_beta)
            else:
                pred_pres = probs > 0.5

            if dmg_gate is not None:
                pred_pres = pred_pres & (dmg.squeeze(-1) > dmg_gate).unsqueeze(-1)

            y_pres = y > PRESENCE_NORM_THRESH
            k_true = y_pres.sum(-1)

            for b in range(x.size(0)):
                K = int(k_true[b].item())
                subset = "undamaged" if K == 0 else ("double" if K >= 2 else "single")
                c = counters[subset]
                pred_set = set(pred_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                fp_this = len(pred_set - true_set)
                c["d_tp"] += len(pred_set & true_set)
                c["d_fp"] += fp_this
                c["d_fn"] += len(true_set - pred_set)
                c["n_samples"]    += 1
                c["total_k_pred"] += len(pred_set)
                if fp_this > 0:
                    c["n_fp_samples"] += 1
                if K > 0:
                    top_locs = probs[b].topk(K).indices.tolist()
                    c["tkr_hits"]  += len(set(top_locs) & true_set)
                    c["tkr_total"] += K

    return {
        k: _make_row(c["d_tp"], c["d_fp"], c["d_fn"],
                     c["tkr_hits"], c["tkr_total"], 0, 0, 0, False,
                     n_samples=c["n_samples"], n_fp_samples=c["n_fp_samples"],
                     total_k_pred=c["total_k_pred"],
                     is_undamaged=(k == "undamaged"))
        for k, c in counters.items() if c["n_samples"] > 0
    }


@torch.inference_mode()
def _run_condition_7story_dr(
    model: torch.nn.Module,
    test_dl,
    fault_type: str,
    n_faulted: int,
    n_repeats: int,
    device: torch.device,
    threshold: float = 0.5,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
) -> dict[str, dict[str, float]]:
    repeats = max(1, n_repeats) if n_faulted > 0 else 1
    def _new_counter():
        return dict(d_tp=0, d_fp=0, d_fn=0, tkr_hits=0, tkr_total=0,
                    n_samples=0, n_fp_samples=0, total_k_pred=0)
    counters = {"single": _new_counter(), "double": _new_counter(), "undamaged": _new_counter()}

    for repeat in range(repeats):
        rng = torch.Generator()
        rng.manual_seed(repeat * 997)

        for batch in test_dl:
            x, y = batch[0].float().to(device), batch[1].to(device)
            if n_faulted > 0:
                x, _ = inject_faults_batch(x, fault_type, n_faulted, rng)
                x = x.to(device)

            pred = model(x)  # (B, L)

            if ratio_alpha is not None:
                max_p = pred.max(dim=-1, keepdim=True).values
                pred_pres = (pred > ratio_alpha * max_p) & (max_p > ratio_beta)
            else:
                pred_pres = pred > threshold

            y_pres = y > PRESENCE_NORM_THRESH
            k_true = y_pres.sum(-1)

            for b in range(x.size(0)):
                K = int(k_true[b].item())
                subset = "undamaged" if K == 0 else ("double" if K >= 2 else "single")
                c = counters[subset]
                pred_set = set(pred_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                fp_this = len(pred_set - true_set)
                c["d_tp"] += len(pred_set & true_set)
                c["d_fp"] += fp_this
                c["d_fn"] += len(true_set - pred_set)
                c["n_samples"]    += 1
                c["total_k_pred"] += len(pred_set)
                if fp_this > 0:
                    c["n_fp_samples"] += 1
                if K > 0:
                    top_locs = pred[b].topk(K).indices.tolist()
                    c["tkr_hits"]  += len(set(top_locs) & true_set)
                    c["tkr_total"] += K

    return {
        k: _make_row(c["d_tp"], c["d_fp"], c["d_fn"],
                     c["tkr_hits"], c["tkr_total"], 0, 0, 0, False,
                     n_samples=c["n_samples"], n_fp_samples=c["n_fp_samples"],
                     total_k_pred=c["total_k_pred"],
                     is_undamaged=(k == "undamaged"))
        for k, c in counters.items() if c["n_samples"] > 0
    }


# ---------------------------------------------------------------------------
# ASCE runners (per-batch fault injection, single counter — K ∈ {0..5})
# ---------------------------------------------------------------------------

def _asce_subset_for(K: int) -> str:
    """Map K_true to ASCE subset label. K=0 → 'undamaged', K∈{1..5} → 'k1'..'k5'."""
    return "undamaged" if K == 0 else f"k{K}"


def _new_asce_counter() -> dict:
    return dict(d_tp=0, d_fp=0, d_fn=0, tkr_hits=0, tkr_total=0,
                f_tp=0, f_fp=0, f_fn=0, has_fault_head=False,
                n_samples=0, n_fp_samples=0, total_k_pred=0)


_ASCE_SUBSETS = ("undamaged", "k1", "k2", "k3", "k4", "k5")


@torch.inference_mode()
def _run_condition_asce_c(
    model: torch.nn.Module,
    test_dl,
    fault_type: str,
    n_faulted: int,
    n_repeats: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Returns {subset: row} for subset in ('undamaged','k1',…,'k5')."""
    counters = {s: _new_asce_counter() for s in _ASCE_SUBSETS}
    repeats = max(1, n_repeats) if n_faulted > 0 else 1

    for repeat in range(repeats):
        rng = torch.Generator()
        rng.manual_seed(repeat * 997)

        for batch in test_dl:
            x, y = batch[0].float().to(device), batch[1].to(device)
            if n_faulted > 0:
                x, y_fault_gt = inject_faults_batch(x, fault_type, n_faulted, rng)
                x = x.to(device)
                y_fault_gt = y_fault_gt.to(device)
            else:
                y_fault_gt = torch.zeros(x.size(0), x.size(-1), device=device)

            loc_logits, severity, fault_prob, _ = model(x)
            active, pred_loc, is_obj = _c_slot_decode(loc_logits)
            y_pres = y > PRESENCE_NORM_THRESH
            k_true = y_pres.sum(-1)

            for b in range(x.size(0)):
                K = int(k_true[b].item())
                c = counters[_asce_subset_for(K)]

                pred_set = set(pred_loc[b, active[b]].tolist())
                true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                fp_this = len(pred_set - true_set)
                c["d_tp"] += len(pred_set & true_set)
                c["d_fp"] += fp_this
                c["d_fn"] += len(true_set - pred_set)
                c["n_samples"]    += 1
                c["total_k_pred"] += len(pred_set)
                if fp_this > 0:
                    c["n_fp_samples"] += 1

                if K > 0:
                    K_slots   = min(K, is_obj.size(-1))
                    top_slots = is_obj[b].topk(K_slots).indices
                    ps = set(pred_loc[b, top_slots].tolist())
                    c["tkr_hits"]  += len(ps & true_set)
                    c["tkr_total"] += K

                if fault_prob is not None:
                    c["has_fault_head"] = True
                    pred_f = fault_prob[b] >= 0.5
                    gt_f   = y_fault_gt[b] > 0.5
                    c["f_tp"] += int((pred_f &  gt_f).sum())
                    c["f_fp"] += int((pred_f & ~gt_f).sum())
                    c["f_fn"] += int((~pred_f & gt_f).sum())

    return {
        k: _make_row(c["d_tp"], c["d_fp"], c["d_fn"],
                     c["tkr_hits"], c["tkr_total"],
                     c["f_tp"], c["f_fp"], c["f_fn"], c["has_fault_head"],
                     n_samples=c["n_samples"], n_fp_samples=c["n_fp_samples"],
                     total_k_pred=c["total_k_pred"],
                     is_undamaged=(k == "undamaged"))
        for k, c in counters.items() if c["n_samples"] > 0
    }


@torch.inference_mode()
def _run_condition_asce_v1(
    model: torch.nn.Module,
    test_dl,
    fault_type: str,
    n_faulted: int,
    n_repeats: int,
    device: torch.device,
    temperature: float = 1.0,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
    dmg_gate: float | None = None,
) -> dict[str, dict[str, float]]:
    counters = {s: _new_asce_counter() for s in _ASCE_SUBSETS}
    repeats = max(1, n_repeats) if n_faulted > 0 else 1

    for repeat in range(repeats):
        rng = torch.Generator()
        rng.manual_seed(repeat * 997)

        for batch in test_dl:
            x, y = batch[0].float().to(device), batch[1].to(device)
            if n_faulted > 0:
                x, _ = inject_faults_batch(x, fault_type, n_faulted, rng)
                x = x.to(device)

            dmg, loc = model(x)
            probs = torch.softmax(loc / temperature, dim=-1)

            if ratio_alpha is not None:
                max_p = probs.max(dim=-1, keepdim=True).values
                pred_pres = (probs > ratio_alpha * max_p) & (max_p > ratio_beta)
            else:
                pred_pres = probs > 0.5

            if dmg_gate is not None:
                pred_pres = pred_pres & (dmg.squeeze(-1) > dmg_gate).unsqueeze(-1)

            y_pres = y > PRESENCE_NORM_THRESH
            k_true = y_pres.sum(-1)

            for b in range(x.size(0)):
                K = int(k_true[b].item())
                c = counters[_asce_subset_for(K)]
                pred_set = set(pred_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                fp_this = len(pred_set - true_set)
                c["d_tp"] += len(pred_set & true_set)
                c["d_fp"] += fp_this
                c["d_fn"] += len(true_set - pred_set)
                c["n_samples"]    += 1
                c["total_k_pred"] += len(pred_set)
                if fp_this > 0:
                    c["n_fp_samples"] += 1
                if K > 0:
                    top_locs = probs[b].topk(K).indices.tolist()
                    c["tkr_hits"]  += len(set(top_locs) & true_set)
                    c["tkr_total"] += K

    return {
        k: _make_row(c["d_tp"], c["d_fp"], c["d_fn"],
                     c["tkr_hits"], c["tkr_total"], 0, 0, 0, False,
                     n_samples=c["n_samples"], n_fp_samples=c["n_fp_samples"],
                     total_k_pred=c["total_k_pred"],
                     is_undamaged=(k == "undamaged"))
        for k, c in counters.items() if c["n_samples"] > 0
    }


@torch.inference_mode()
def _run_condition_asce_dr(
    model: torch.nn.Module,
    test_dl,
    fault_type: str,
    n_faulted: int,
    n_repeats: int,
    device: torch.device,
    threshold: float = 0.5,
    ratio_alpha: float | None = None,
    ratio_beta: float = 0.0,
) -> dict[str, dict[str, float]]:
    counters = {s: _new_asce_counter() for s in _ASCE_SUBSETS}
    repeats = max(1, n_repeats) if n_faulted > 0 else 1

    for repeat in range(repeats):
        rng = torch.Generator()
        rng.manual_seed(repeat * 997)

        for batch in test_dl:
            x, y = batch[0].float().to(device), batch[1].to(device)
            if n_faulted > 0:
                x, _ = inject_faults_batch(x, fault_type, n_faulted, rng)
                x = x.to(device)

            pred = model(x)

            if ratio_alpha is not None:
                max_p = pred.max(dim=-1, keepdim=True).values
                pred_pres = (pred > ratio_alpha * max_p) & (max_p > ratio_beta)
            else:
                pred_pres = pred > threshold

            y_pres = y > PRESENCE_NORM_THRESH
            k_true = y_pres.sum(-1)

            for b in range(x.size(0)):
                K = int(k_true[b].item())
                c = counters[_asce_subset_for(K)]
                pred_set = set(pred_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                fp_this = len(pred_set - true_set)
                c["d_tp"] += len(pred_set & true_set)
                c["d_fp"] += fp_this
                c["d_fn"] += len(true_set - pred_set)
                c["n_samples"]    += 1
                c["total_k_pred"] += len(pred_set)
                if fp_this > 0:
                    c["n_fp_samples"] += 1
                if K > 0:
                    top_locs = pred[b].topk(K).indices.tolist()
                    c["tkr_hits"]  += len(set(top_locs) & true_set)
                    c["tkr_total"] += K

    return {
        k: _make_row(c["d_tp"], c["d_fp"], c["d_fn"],
                     c["tkr_hits"], c["tkr_total"], 0, 0, 0, False,
                     n_samples=c["n_samples"], n_fp_samples=c["n_fp_samples"],
                     total_k_pred=c["total_k_pred"],
                     is_undamaged=(k == "undamaged"))
        for k, c in counters.items() if c["n_samples"] > 0
    }


def _run_sweep_asce(
    model_name: str,
    run_fn,
    model,
    test_dl,
    fault_types,
    n_faulted_list,
    n_repeats,
    device,
    out_stem,
) -> list[dict]:
    """Per-batch fault sweep for ASCE — {'undamaged','k1',...,'k5'} × (fault_type, n_faulted)."""
    rows: list[dict] = []

    if 0 in n_faulted_list:
        print(f"[{model_name}] clean baseline")
        results = run_fn(model, test_dl, "hard", 0, 1, device)
        for subset, row in results.items():
            row["model"]      = model_name
            row["fault_type"] = "clean"
            row["n_faulted"]  = 0
            row["subset"]     = subset
            rows.append(row)

    nf_nonzero = [n for n in n_faulted_list if n > 0]
    total = len(fault_types) * len(nf_nonzero)
    done  = 0
    for ft in fault_types:
        for nf in nf_nonzero:
            done += 1
            print(f"[{model_name}] ({done}/{total})  fault={ft}  n_faulted={nf}")
            results = run_fn(model, test_dl, ft, nf, n_repeats, device)
            for subset, row in results.items():
                row["model"]      = model_name
                row["fault_type"] = ft
                row["n_faulted"]  = nf
                row["subset"]     = subset
                rows.append(row)

    _print_table(rows, model_name)
    if out_stem is not None:
        _save_results(rows, out_stem)
    return rows


def _run_sweep_7story(
    model_name: str,
    run_fn,
    model,
    test_dl,
    fault_types,
    n_faulted_list,
    n_repeats,
    device,
    out_stem,
) -> list[dict]:
    """Like _run_sweep but run_fn returns {'single': row, 'double': row}."""
    rows: list[dict] = []

    nf_list_with_clean = [0] + [n for n in n_faulted_list if n > 0] if 0 in n_faulted_list \
        else [n for n in n_faulted_list if n > 0]

    if 0 in n_faulted_list:
        print(f"[{model_name}] clean baseline")
        results = run_fn(model, test_dl, "hard", 0, 1, device)
        for subset, row in results.items():
            row["model"]      = model_name
            row["fault_type"] = "clean"
            row["n_faulted"]  = 0
            row["subset"]     = subset
            rows.append(row)

    nf_nonzero = [n for n in n_faulted_list if n > 0]
    total = len(fault_types) * len(nf_nonzero)
    done  = 0
    for ft in fault_types:
        for nf in nf_nonzero:
            done += 1
            print(f"[{model_name}] ({done}/{total})  fault={ft}  n_faulted={nf}")
            results = run_fn(model, test_dl, ft, nf, n_repeats, device)
            for subset, row in results.items():
                row["model"]      = model_name
                row["fault_type"] = ft
                row["n_faulted"]  = nf
                row["subset"]     = subset
                rows.append(row)

    _print_table(rows, model_name)
    if out_stem is not None:
        _save_results(rows, out_stem)
    return rows


# ---------------------------------------------------------------------------
# Shared result helpers
# ---------------------------------------------------------------------------

def _make_row(
    d_tp, d_fp, d_fn, tkr_hits, tkr_total,
    f_tp, f_fp, f_fn, has_fault_head,
    n_samples=0, n_fp_samples=0, total_k_pred=0,
    is_undamaged=False,
) -> dict[str, float]:
    row: dict[str, float] = {}
    if is_undamaged:
        # K=0: F1/precision/recall/top_k_recall undefined (no true positives possible).
        # Useful K=0 metrics are sample-level: how often does the model flag anything?
        row["f1"]           = float("nan")
        row["precision"]    = float("nan")
        row["recall"]       = float("nan")
        row["top_k_recall"] = float("nan")
    else:
        f1, prec, rec = f1_from_counts(d_tp, d_fp, d_fn)
        row["f1"]           = f1
        row["precision"]    = prec
        row["recall"]       = rec
        row["top_k_recall"] = tkr_hits / max(tkr_total, 1)

    # sample_far / mean_k_pred need a non-zero per-sample count; callers that
    # do not track them (non-7story runners) pass n_samples=0 → emit NaN.
    if n_samples > 0:
        row["sample_far"]  = n_fp_samples / n_samples
        row["mean_k_pred"] = total_k_pred / n_samples
    else:
        row["sample_far"]  = float("nan")
        row["mean_k_pred"] = float("nan")

    if has_fault_head:
        ff1, fprec, frec = f1_from_counts(f_tp, f_fp, f_fn)
        row["fault_f1"]        = ff1
        row["fault_precision"] = fprec
        row["fault_recall"]    = frec
    else:
        row["fault_f1"]        = float("nan")
        row["fault_precision"] = float("nan")
        row["fault_recall"]    = float("nan")
    return row


def _print_table(rows: list[dict], model_name: str) -> None:
    has_subset = any("subset" in r for r in rows)
    print(f"\n=== {model_name} ===")
    col_w = 12
    fields = ["fault_type", "n_faulted"]
    if has_subset:
        fields.append("subset")
    fields += _ALL_METRICS
    header = "  ".join(f"{h:<{col_w}}" for h in fields)
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = [r["fault_type"], str(r["n_faulted"])]
        if has_subset:
            vals.append(r.get("subset", ""))
        vals += [
            f"{r.get(m, float('nan')):.4f}"
            if not math.isnan(r.get(m, float("nan"))) else "N/A"
            for m in _ALL_METRICS
        ]
        print("  ".join(f"{v:<{col_w}}" for v in vals))
    print()


def _save_results(rows: list[dict], stem) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    json_path = stem.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(
            [{k: (None if isinstance(v, float) and math.isnan(v) else v)
              for k, v in r.items()}
             for r in rows],
            f, indent=2,
        )

    csv_path = stem.with_suffix(".csv")
    has_subset = any("subset" in r for r in rows)
    fields = ["model", "fault_type", "n_faulted"]
    if has_subset:
        fields.append("subset")
    fields += _ALL_METRICS
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, float("nan")) for k in fields})

    print(f"[results] Saved → {json_path}")
    print(f"[results] Saved → {csv_path}")


# ---------------------------------------------------------------------------
# Per-model sweep
# ---------------------------------------------------------------------------

def _run_sweep(
    model_name: str,
    run_fn,
    model,
    recordings,
    fault_types,
    n_faulted_list,
    n_repeats,
    batch_size,
    device,
    out_stem: Path | None,
) -> list[dict]:
    rows: list[dict] = []

    if 0 in n_faulted_list:
        print(f"[{model_name}] clean baseline")
        row = run_fn(model, recordings, "hard", 0, 1, batch_size, device)
        row["model"]      = model_name
        row["fault_type"] = "clean"
        row["n_faulted"]  = 0
        rows.append(row)

    nf_nonzero = [n for n in n_faulted_list if n > 0]
    total = len(fault_types) * len(nf_nonzero)
    done  = 0
    for ft in fault_types:
        for nf in nf_nonzero:
            done += 1
            print(f"[{model_name}] ({done}/{total})  fault={ft}  n_faulted={nf}")
            row = run_fn(model, recordings, ft, nf, n_repeats, batch_size, device)
            row["model"]      = model_name
            row["fault_type"] = ft
            row["n_faulted"]  = nf
            rows.append(row)

    _print_table(rows, model_name)
    if out_stem is not None:
        _save_results(rows, out_stem)
    return rows


# ---------------------------------------------------------------------------
# Ensemble calibration helper
# ---------------------------------------------------------------------------

def _resolve_c_ensemble_cal(
    model, ckpt_path, dataset_name, args, device,
) -> dict[str, float]:
    """
    Return ensemble calibration {ensemble_alpha, ensemble_beta, ensemble_use_severity}.

    Priority:
      1. If calibration sidecar already has ensemble params → use them.
      2. Else run ``calibrate_c_ensemble`` on val loaders, save back to sidecar.
    """
    from lib.calibration import calibrate_c_ensemble, save_calibration

    cal = load_calibration(ckpt_path)
    if "ensemble_alpha" in cal:
        print(f"[c-ensemble] reusing sidecar params: "
              f"α={cal['ensemble_alpha']:.3f}  β={cal['ensemble_beta']:.3f}  "
              f"use_severity={cal.get('ensemble_use_severity', True)}")
        return cal

    dataset = get_dataset(dataset_name)
    dl_kwargs = dict(
        root=args.root or dataset.default_root,
        num_workers=0,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        seed=42,
    )
    if dataset_name in ("qatar", "lumo"):
        dl_kwargs.update(window_size=args.window_size,
                         overlap=args.overlap,
                         downsample=args.downsample)
    if dataset_name in ("7story", "7story-sparse", "asce"):
        dl_kwargs["norm_method"] = args.norm_method
    cal_loaders = dataset.get_calibration_val_loaders(**dl_kwargs)
    ens_cal = calibrate_c_ensemble(
        model, cal_loaders, device,
        use_severity=not args.c_ensemble_no_severity,
    )
    # Merge into existing sidecar and persist
    cal.update(ens_cal)
    save_calibration(cal, ckpt_path)
    return cal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fault robustness sweep for C, v1, and DR heads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",      default="qatar", choices=["qatar", "7story", "lumo", "asce", "asce-columns"],
                        help="Dataset to evaluate on (default: qatar).")
    parser.add_argument("--c",            default=None, metavar="CKPT",
                        help="C-head checkpoint path.")
    parser.add_argument("--v1",           default=None, metavar="CKPT",
                        help="v1 checkpoint path.")
    parser.add_argument("--dr",           default=None, metavar="CKPT",
                        help="DR checkpoint path.")
    parser.add_argument("--b",            default=None, metavar="CKPT",
                        help="Baseline B (PlainDR) checkpoint path.")
    parser.add_argument("--root",         default=None,
                        help="Dataset root (default: data/Qatar/processed for qatar, "
                             "data/7-story-frame/safetensors/unc=0 for 7story, "
                             "data/LUMO for lumo, data/asce_hammer for asce).")
    parser.add_argument("--fault-types",  nargs="+", default=FAULT_TYPES,
                        choices=FAULT_TYPES)
    parser.add_argument("--fault-ratio",  nargs="+", type=float,
                        default=[0.0, 0.2, 0.5, 0.8],
                        help="Fraction of sensors to fault (converted to absolute count per dataset).")
    parser.add_argument("--n-faulted",    nargs="+", type=int, default=None,
                        help="Override: absolute sensor counts (ignores --fault-ratio).")
    parser.add_argument("--n-repeats",    type=int, default=3)
    parser.add_argument("--batch-size",   type=int, default=64)
    # Qatar-specific windowing
    parser.add_argument("--window-size",  type=int, default=2048)
    parser.add_argument("--overlap",      type=float, default=0.5)
    parser.add_argument("--downsample",   type=int, default=4)
    # 7-story normalization — must match what the checkpoint was trained with
    parser.add_argument("--norm-method",  choices=["mean", "none"], default="none",
                        help="RMS normalization aggregation (7story only). Must match training config.")
    parser.add_argument("--out",          default=None, metavar="STEM",
                        help="Output path stem. Model name is appended automatically "
                             "when multiple models are evaluated, e.g. <stem>_c.json. "
                             "Defaults to saved_results/<dataset>/eval_fault_<ts>.")
    parser.add_argument("--c-ensemble",   action="store_true", default=False,
                        help="Use ensemble slot decoding for the C-head (soft-map + ratio threshold). "
                             "If calibration sidecar lacks ensemble params, calibrates on-the-fly from val.")
    parser.add_argument("--c-ensemble-no-severity", action="store_true", default=False,
                        help="C-head ensemble: drop severity factor (weight = is_obj only). "
                             "Only used when calibrating on-the-fly.")
    parser.add_argument("--c-label",      default="C",   metavar="LABEL",
                        help="Label recorded in results for the C-head model (e.g. 'C+fh', 'C+fh+sb').")
    parser.add_argument("--v1-label",     default="v1",  metavar="LABEL",
                        help="Label recorded in results for the v1 model.")
    parser.add_argument("--dr-label",     default="DR",  metavar="LABEL",
                        help="Label recorded in results for the DR model.")
    parser.add_argument("--b-label",      default="B",   metavar="LABEL",
                        help="Label recorded in results for the B model.")
    args = parser.parse_args()

    if args.c is None and args.v1 is None and args.dr is None and args.b is None:
        parser.error("At least one of --c, --v1, --dr, --b is required.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine output stems
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = Path("saved_results") / args.dataset / f"eval_fault_{ts}"
    base_stem = Path(args.out) if args.out else default_out
    n_models  = sum(x is not None for x in [args.c, args.v1, args.dr, args.b])

    def _stem(name: str) -> Path:
        if n_models == 1:
            return base_stem
        return base_stem.parent / f"{base_stem.name}_{name}"

    # Convert --fault-ratio to absolute sensor counts (unless --n-faulted given)
    n_sensors = DATASETS[args.dataset].n_sensors
    if args.n_faulted is not None:
        n_faulted_list = args.n_faulted
    else:
        n_faulted_list = sorted(set(max(0, round(r * n_sensors)) for r in args.fault_ratio))

    # -------------------------------------------------------------------------
    # Qatar branch (recording-consistent injection)
    # -------------------------------------------------------------------------
    if args.dataset == "qatar":
        root = args.root or "data/Qatar/processed"
        _wkw = dict(root=root, window_size=args.window_size,
                     overlap=args.overlap, downsample=args.downsample)

        print(f"[evaluate_fault] Loading Qatar single-damage test recordings from: {root}")
        single_recs = get_qatar_test_by_recording(**_wkw)
        n_s = sum(x.size(0) for x, _ in single_recs)
        print(f"[evaluate_fault] {len(single_recs)} single recordings, {n_s} windows")

        print(f"[evaluate_fault] Loading Qatar double-damage test recording (j23+j24)")
        double_recs = get_qatar_double_by_recording(**_wkw, held_out_index=QATAR_HELD_OUT_DOUBLE)
        n_d = sum(x.size(0) for x, _ in double_recs)
        print(f"[evaluate_fault] {len(double_recs)} double recordings, {n_d} windows")

        def _qatar_sweep(model_label, run_fn, model):
            """Run fault sweep on single + double, return combined rows with subset."""
            rows_s = _run_sweep(model_label, run_fn, model, single_recs,
                                args.fault_types, n_faulted_list, args.n_repeats,
                                args.batch_size, device, out_stem=None)
            for r in rows_s:
                r["subset"] = "single"

            rows_d = _run_sweep(model_label, run_fn, model, double_recs,
                                args.fault_types, n_faulted_list, args.n_repeats,
                                args.batch_size, device, out_stem=None)
            for r in rows_d:
                r["subset"] = "double"

            combined = rows_s + rows_d
            _print_table(combined, model_label)
            _save_results(combined, _stem(model_label.lower().replace("+", "-")))
            return combined

        if args.c is not None:
            print(f"\n[evaluate_fault] Loading C checkpoint: {args.c}")
            model_c = load_model_c_from_checkpoint(args.c, device=device)
            model_c.eval()
            if args.c_ensemble:
                cal = _resolve_c_ensemble_cal(model_c, args.c, "qatar", args, device)
                run_fn_c = lambda model, recs, ft, nf, nr, bs, dev: _run_condition_c(
                    model, recs, ft, nf, nr, bs, dev,
                    ensemble_alpha=cal.get("ensemble_alpha"),
                    ensemble_beta=cal.get("ensemble_beta", 0.0),
                    ensemble_use_severity=cal.get("ensemble_use_severity", True),
                )
            else:
                run_fn_c = _run_condition_c
            _qatar_sweep(args.c_label, run_fn_c, model_c)
            del model_c

        if args.v1 is not None:
            print(f"\n[evaluate_fault] Loading v1 checkpoint: {args.v1}")
            model_v1 = load_model_from_checkpoint(args.v1, device=device)
            model_v1.eval()
            cal = load_calibration(args.v1)
            run_fn = lambda model, recs, ft, nf, nr, bs, dev: _run_condition_v1(
                model, recs, ft, nf, nr, bs, dev,
                temperature=cal.get("temperature", 1.0),
                ratio_alpha=cal.get("ratio_alpha"),
                ratio_beta=cal.get("ratio_beta", 0.0),
                dmg_gate=cal.get("dmg_gate"),
            )
            _qatar_sweep(args.v1_label, run_fn, model_v1)
            del model_v1

        if args.dr is not None:
            print(f"\n[evaluate_fault] Loading DR checkpoint: {args.dr}")
            model_dr = load_model_dr_from_checkpoint(args.dr, device=device)
            model_dr.eval()
            cal = load_calibration(args.dr)
            run_fn = lambda model, recs, ft, nf, nr, bs, dev: _run_condition_dr(
                model, recs, ft, nf, nr, bs, dev,
                threshold=cal.get("threshold", 0.5),
                ratio_alpha=cal.get("ratio_alpha"),
                ratio_beta=cal.get("ratio_beta", 0.0),
            )
            _qatar_sweep(args.dr_label, run_fn, model_dr)
            del model_dr

        if args.b is not None:
            print(f"\n[evaluate_fault] Loading B checkpoint: {args.b}")
            model_b = load_model_dr_from_checkpoint(args.b, device=device)
            model_b.eval()
            cal = load_calibration(args.b)
            run_fn = lambda model, recs, ft, nf, nr, bs, dev: _run_condition_dr(
                model, recs, ft, nf, nr, bs, dev,
                threshold=cal.get("threshold", 0.5),
                ratio_alpha=cal.get("ratio_alpha"),
                ratio_beta=cal.get("ratio_beta", 0.0),
            )
            _qatar_sweep(args.b_label, run_fn, model_b)
            del model_b

    # -------------------------------------------------------------------------
    # LUMO branch (recording-consistent injection, same paradigm as Qatar)
    # -------------------------------------------------------------------------
    elif args.dataset == "lumo":
        root = args.root or "data/LUMO"
        print(f"[evaluate_fault] Loading LUMO test recordings from: {root}")
        recordings = get_lumo_test_by_recording(
            root=root,
            window_size=args.window_size,
            overlap=args.overlap,
            downsample=args.downsample,
        )
        n_windows = sum(x.size(0) for x, _ in recordings)
        print(f"[evaluate_fault] {len(recordings)} recordings, {n_windows} windows")

        if args.c is not None:
            print(f"\n[evaluate_fault] Loading C checkpoint: {args.c}")
            model_c = load_model_c_from_checkpoint(args.c, device=device)
            model_c.eval()
            if args.c_ensemble:
                cal = _resolve_c_ensemble_cal(model_c, args.c, "lumo", args, device)
                run_fn_c = lambda model, recs, ft, nf, nr, bs, dev: _run_condition_c(
                    model, recs, ft, nf, nr, bs, dev,
                    ensemble_alpha=cal.get("ensemble_alpha"),
                    ensemble_beta=cal.get("ensemble_beta", 0.0),
                    ensemble_use_severity=cal.get("ensemble_use_severity", True),
                )
            else:
                run_fn_c = _run_condition_c
            _run_sweep(args.c_label, run_fn_c, model_c, recordings,
                       args.fault_types, n_faulted_list, args.n_repeats,
                       args.batch_size, device, _stem("c"))
            del model_c

        if args.v1 is not None:
            print(f"\n[evaluate_fault] Loading v1 checkpoint: {args.v1}")
            model_v1 = load_model_from_checkpoint(args.v1, device=device)
            model_v1.eval()
            cal = load_calibration(args.v1)
            run_fn = lambda model, recs, ft, nf, nr, bs, dev: _run_condition_v1(
                model, recs, ft, nf, nr, bs, dev,
                temperature=cal.get("temperature", 1.0),
                ratio_alpha=cal.get("ratio_alpha"),
                ratio_beta=cal.get("ratio_beta", 0.0),
                dmg_gate=cal.get("dmg_gate"),
            )
            _run_sweep(args.v1_label, run_fn, model_v1, recordings,
                       args.fault_types, n_faulted_list, args.n_repeats,
                       args.batch_size, device, _stem("v1"))
            del model_v1

        if args.dr is not None:
            print(f"\n[evaluate_fault] Loading DR checkpoint: {args.dr}")
            model_dr = load_model_dr_from_checkpoint(args.dr, device=device)
            model_dr.eval()
            cal = load_calibration(args.dr)
            run_fn = lambda model, recs, ft, nf, nr, bs, dev: _run_condition_dr(
                model, recs, ft, nf, nr, bs, dev,
                threshold=cal.get("threshold", 0.5),
                ratio_alpha=cal.get("ratio_alpha"),
                ratio_beta=cal.get("ratio_beta", 0.0),
            )
            _run_sweep(args.dr_label, run_fn, model_dr, recordings,
                       args.fault_types, n_faulted_list, args.n_repeats,
                       args.batch_size, device, _stem("dr"))
            del model_dr

        if args.b is not None:
            print(f"\n[evaluate_fault] Loading B checkpoint: {args.b}")
            model_b = load_model_dr_from_checkpoint(args.b, device=device)
            model_b.eval()
            cal = load_calibration(args.b)
            run_fn = lambda model, recs, ft, nf, nr, bs, dev: _run_condition_dr(
                model, recs, ft, nf, nr, bs, dev,
                threshold=cal.get("threshold", 0.5),
                ratio_alpha=cal.get("ratio_alpha"),
                ratio_beta=cal.get("ratio_beta", 0.0),
            )
            _run_sweep(args.b_label, run_fn, model_b, recordings,
                       args.fault_types, n_faulted_list, args.n_repeats,
                       args.batch_size, device, _stem("b"))
            del model_b

    # -------------------------------------------------------------------------
    # 7-story branch (per-batch fault injection, split by K=1/K=2)
    # -------------------------------------------------------------------------
    elif args.dataset == "7story":
        from lib.data_7story import get_7story_dataloaders
        root = args.root or "data/7-story-frame/safetensors/unc=0"
        extra_unc1_undamaged = Path(root).parent / "unc=1" / "undamaged"
        print(f"[evaluate_fault] Loading 7-story test set from: {root}")
        _, _, test_dl = get_7story_dataloaders(
            ["single", "double", "undamaged"],
            root=root,
            extra_subset_roots=[extra_unc1_undamaged] if extra_unc1_undamaged.exists() else None,
            num_workers=0,
            eval_batch_size=args.batch_size,
            norm_method=args.norm_method,
        )
        n_samples = len(test_dl.dataset)
        print(f"[evaluate_fault] {n_samples} test samples (single + double + undamaged)")

        if args.c is not None:
            print(f"\n[evaluate_fault] Loading C checkpoint: {args.c}")
            model_c = load_model_c_from_checkpoint(args.c, device=device)
            model_c.eval()
            if args.c_ensemble:
                cal = _resolve_c_ensemble_cal(model_c, args.c, "7story", args, device)
                run_fn = lambda model, dl, ft, nf, nr, dev: _run_condition_7story_c(
                    model, dl, ft, nf, nr, dev,
                    ensemble_alpha=cal.get("ensemble_alpha"),
                    ensemble_beta=cal.get("ensemble_beta", 0.0),
                    ensemble_use_severity=cal.get("ensemble_use_severity", True),
                )
            else:
                run_fn = lambda model, dl, ft, nf, nr, dev: _run_condition_7story_c(
                    model, dl, ft, nf, nr, dev)
            _run_sweep_7story(args.c_label, run_fn, model_c, test_dl,
                              args.fault_types, n_faulted_list, args.n_repeats,
                              device, _stem("c"))
            del model_c

        if args.v1 is not None:
            print(f"\n[evaluate_fault] Loading v1 checkpoint: {args.v1}")
            model_v1 = load_model_from_checkpoint(args.v1, device=device)
            model_v1.eval()
            cal = load_calibration(args.v1)
            run_fn = lambda model, dl, ft, nf, nr, dev: _run_condition_7story_v1(
                model, dl, ft, nf, nr, dev,
                temperature=cal.get("temperature", 1.0),
                ratio_alpha=cal.get("ratio_alpha"),
                ratio_beta=cal.get("ratio_beta", 0.0),
                dmg_gate=cal.get("dmg_gate"),
            )
            _run_sweep_7story(args.v1_label, run_fn, model_v1, test_dl,
                              args.fault_types, n_faulted_list, args.n_repeats,
                              device, _stem("v1"))
            del model_v1

        if args.dr is not None:
            print(f"\n[evaluate_fault] Loading DR checkpoint: {args.dr}")
            model_dr = load_model_dr_from_checkpoint(args.dr, device=device)
            model_dr.eval()
            cal = load_calibration(args.dr)
            run_fn = lambda model, dl, ft, nf, nr, dev: _run_condition_7story_dr(
                model, dl, ft, nf, nr, dev,
                threshold=cal.get("threshold", 0.5),
                ratio_alpha=cal.get("ratio_alpha"),
                ratio_beta=cal.get("ratio_beta", 0.0),
            )
            _run_sweep_7story(args.dr_label, run_fn, model_dr, test_dl,
                              args.fault_types, n_faulted_list, args.n_repeats,
                              device, _stem("dr"))
            del model_dr

        if args.b is not None:
            print(f"\n[evaluate_fault] Loading B checkpoint: {args.b}")
            model_b = load_model_dr_from_checkpoint(args.b, device=device)
            model_b.eval()
            cal = load_calibration(args.b)
            run_fn = lambda model, dl, ft, nf, nr, dev: _run_condition_7story_dr(
                model, dl, ft, nf, nr, dev,
                threshold=cal.get("threshold", 0.5),
                ratio_alpha=cal.get("ratio_alpha"),
                ratio_beta=cal.get("ratio_beta", 0.0),
            )
            _run_sweep_7story(args.b_label, run_fn, model_b, test_dl,
                              args.fault_types, n_faulted_list, args.n_repeats,
                              device, _stem("b"))
            del model_b

    # -------------------------------------------------------------------------
    # ASCE branch (per-batch injection, single row per condition)
    # -------------------------------------------------------------------------
    elif args.dataset in ("asce", "asce-columns"):
        from lib.data_asce import get_asce_dataloaders
        if args.dataset == "asce-columns":
            root = args.root or "data/asce_hammer_columns"
            n_loc = 36
        else:
            root = args.root or "data/asce_hammer"
            n_loc = 32
        print(f"[evaluate_fault] Loading {args.dataset} test set from: {root}  (L={n_loc})")
        _, _, test_dl = get_asce_dataloaders(
            root=root,
            num_workers=0,
            eval_batch_size=args.batch_size,
            norm_method=args.norm_method,
            n_locations=n_loc,
        )
        n_samples = len(test_dl.dataset)
        print(f"[evaluate_fault] {n_samples} test samples")

        if args.c is not None:
            print(f"\n[evaluate_fault] Loading C checkpoint: {args.c}")
            model_c = load_model_c_from_checkpoint(args.c, device=device)
            model_c.eval()
            run_fn = lambda model, dl, ft, nf, nr, dev: _run_condition_asce_c(
                model, dl, ft, nf, nr, dev)
            _run_sweep_asce(args.c_label, run_fn, model_c, test_dl,
                            args.fault_types, n_faulted_list, args.n_repeats,
                            device, _stem("c"))
            del model_c

        if args.v1 is not None:
            print(f"\n[evaluate_fault] Loading v1 checkpoint: {args.v1}")
            model_v1 = load_model_from_checkpoint(args.v1, device=device)
            model_v1.eval()
            cal = load_calibration(args.v1)
            run_fn = lambda model, dl, ft, nf, nr, dev: _run_condition_asce_v1(
                model, dl, ft, nf, nr, dev,
                temperature=cal.get("temperature", 1.0),
                ratio_alpha=cal.get("ratio_alpha"),
                ratio_beta=cal.get("ratio_beta", 0.0),
                dmg_gate=cal.get("dmg_gate"),
            )
            _run_sweep_asce(args.v1_label, run_fn, model_v1, test_dl,
                            args.fault_types, n_faulted_list, args.n_repeats,
                            device, _stem("v1"))
            del model_v1

        if args.dr is not None:
            print(f"\n[evaluate_fault] Loading DR checkpoint: {args.dr}")
            model_dr = load_model_dr_from_checkpoint(args.dr, device=device)
            model_dr.eval()
            cal = load_calibration(args.dr)
            run_fn = lambda model, dl, ft, nf, nr, dev: _run_condition_asce_dr(
                model, dl, ft, nf, nr, dev,
                threshold=cal.get("threshold", 0.5),
                ratio_alpha=cal.get("ratio_alpha"),
                ratio_beta=cal.get("ratio_beta", 0.0),
            )
            _run_sweep_asce(args.dr_label, run_fn, model_dr, test_dl,
                            args.fault_types, n_faulted_list, args.n_repeats,
                            device, _stem("dr"))
            del model_dr

        if args.b is not None:
            print(f"\n[evaluate_fault] Loading B checkpoint: {args.b}")
            model_b = load_model_dr_from_checkpoint(args.b, device=device)
            model_b.eval()
            cal = load_calibration(args.b)
            run_fn = lambda model, dl, ft, nf, nr, dev: _run_condition_asce_dr(
                model, dl, ft, nf, nr, dev,
                threshold=cal.get("threshold", 0.5),
                ratio_alpha=cal.get("ratio_alpha"),
                ratio_beta=cal.get("ratio_beta", 0.0),
            )
            _run_sweep_asce(args.b_label, run_fn, model_b, test_dl,
                            args.fault_types, n_faulted_list, args.n_repeats,
                            device, _stem("b"))
            del model_b


if __name__ == "__main__":
    main()
