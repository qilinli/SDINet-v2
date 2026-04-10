"""
evaluate_fault.py — Fault robustness evaluation for C, v1, and DR heads.

Supports Qatar and 7-story datasets via --dataset flag.

Qatar (default): recording-consistent fault injection — for each recording,
n_faulted sensors are selected once and the same fault is applied to ALL
windows.  Matches physical reality (a disconnected cable affects the whole
test session).

7-story: per-batch fault injection — each simulation sample is independent,
so faults are injected fresh per batch.  Results are reported separately for
single-damage (K=1) and double-damage (K=2) ground-truth subsets.

Usage
-----
python evaluate_fault.py --dataset qatar --c <ckpt> [--v1 <ckpt>] [--dr <ckpt>] \\
  [--fault-types hard gain bias noise stuck partial] \\
  [--n-faulted 0 1 3 5 10 15] [--n-repeats 3] \\
  [--root data/Qatar/processed] [--out saved_results/qatar/eval_fault]

python evaluate_fault.py --dataset 7story --c <ckpt> [--v1 <ckpt>] [--dr <ckpt>] \\
  [--root data/7-story-frame/safetensors/unc=0] \\
  [--out saved_results/7story-fault/eval_fault]
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
from lib.data_qatar import (
    QATAR_N_SENSORS,
    get_qatar_test_by_recording,
)
from lib.faults import FAULT_TYPES, _apply_soft_fault, inject_faults_batch
from lib.metrics import (
    PRESENCE_NORM_THRESH,
    _c_slot_decode,
    f1_from_counts,
)
from lib.model import (
    load_model_c_from_checkpoint,
    load_model_dr_from_checkpoint,
    load_model_from_checkpoint,
)

_DMG_METRICS   = ["f1", "precision", "recall", "top_k_recall"]
_FAULT_METRICS = ["fault_f1", "fault_precision", "fault_recall"]
_ALL_METRICS   = _DMG_METRICS + _FAULT_METRICS


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
                for n in range(N):
                    _inject(x_f[n], sensor_idx, fault_type)

            all_loc, all_sev = [], []
            for i in range(0, N, batch_size):
                xb = x_f[i:i + batch_size].to(device)
                yf = y_flt[i:i + batch_size].to(device)
                loc_logits, severity, fault_prob = model(xb)
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
            active, pred_loc, is_obj = _c_slot_decode(loc_all)
            k_true = y_pres.sum(-1)

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
                    ts = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                    tkr_hits  += len(ps & ts)
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
                for n in range(N):
                    _inject(x_f[n], sensor_idx, fault_type)

            all_probs = []
            for i in range(0, N, batch_size):
                xb = x_f[i:i + batch_size].to(device)
                _dmg, loc = model(xb)
                probs = torch.softmax(loc / temperature, dim=-1).cpu()  # (B, L)
                all_probs.append(probs)

            probs_all = torch.cat(all_probs)  # (N, L)

            # ratio threshold → predicted presence per sample
            if ratio_alpha is not None:
                max_p = probs_all.max(dim=-1, keepdim=True).values
                pred_pres = (probs_all > ratio_alpha * max_p) & (max_p > ratio_beta)
            else:
                pred_pres = probs_all > 0.5

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
                for n in range(N):
                    _inject(x_f[n], sensor_idx, fault_type)

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
) -> dict[str, dict[str, float]]:
    """Returns {"single": row, "double": row} keyed by damage multiplicity."""
    repeats = max(1, n_repeats) if n_faulted > 0 else 1
    counters = {
        "single": dict(d_tp=0, d_fp=0, d_fn=0, tkr_hits=0, tkr_total=0,
                       f_tp=0, f_fp=0, f_fn=0, has_fault_head=False),
        "double": dict(d_tp=0, d_fp=0, d_fn=0, tkr_hits=0, tkr_total=0,
                       f_tp=0, f_fp=0, f_fn=0, has_fault_head=False),
    }

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

            loc_logits, severity, fault_prob = model(x)
            active, pred_loc, is_obj = _c_slot_decode(loc_logits)
            y_pres = y > PRESENCE_NORM_THRESH  # (B, L)
            k_true = y_pres.sum(-1)            # (B,)

            for b in range(x.size(0)):
                K = int(k_true[b].item())
                subset = "double" if K >= 2 else "single"
                c = counters[subset]

                pred_set = set(pred_loc[b, active[b]].tolist())
                true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                c["d_tp"] += len(pred_set & true_set)
                c["d_fp"] += len(pred_set - true_set)
                c["d_fn"] += len(true_set - pred_set)

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
                     c["has_fault_head"])
        for k, c in counters.items()
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
) -> dict[str, dict[str, float]]:
    repeats = max(1, n_repeats) if n_faulted > 0 else 1
    counters = {
        "single": dict(d_tp=0, d_fp=0, d_fn=0, tkr_hits=0, tkr_total=0),
        "double": dict(d_tp=0, d_fp=0, d_fn=0, tkr_hits=0, tkr_total=0),
    }

    for repeat in range(repeats):
        rng = torch.Generator()
        rng.manual_seed(repeat * 997)

        for batch in test_dl:
            x, y = batch[0].float().to(device), batch[1].to(device)
            if n_faulted > 0:
                x, _ = inject_faults_batch(x, fault_type, n_faulted, rng)
                x = x.to(device)

            _dmg, loc = model(x)
            probs = torch.softmax(loc / temperature, dim=-1)  # (B, L)

            if ratio_alpha is not None:
                max_p = probs.max(dim=-1, keepdim=True).values
                pred_pres = (probs > ratio_alpha * max_p) & (max_p > ratio_beta)
            else:
                pred_pres = probs > 0.5

            y_pres = y > PRESENCE_NORM_THRESH
            k_true = y_pres.sum(-1)

            for b in range(x.size(0)):
                K = int(k_true[b].item())
                subset = "double" if K >= 2 else "single"
                c = counters[subset]
                pred_set = set(pred_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                c["d_tp"] += len(pred_set & true_set)
                c["d_fp"] += len(pred_set - true_set)
                c["d_fn"] += len(true_set - pred_set)
                if K > 0:
                    top_locs = probs[b].topk(K).indices.tolist()
                    c["tkr_hits"]  += len(set(top_locs) & true_set)
                    c["tkr_total"] += K

    return {
        k: _make_row(c["d_tp"], c["d_fp"], c["d_fn"],
                     c["tkr_hits"], c["tkr_total"], 0, 0, 0, False)
        for k, c in counters.items()
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
    counters = {
        "single": dict(d_tp=0, d_fp=0, d_fn=0, tkr_hits=0, tkr_total=0),
        "double": dict(d_tp=0, d_fp=0, d_fn=0, tkr_hits=0, tkr_total=0),
    }

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
                subset = "double" if K >= 2 else "single"
                c = counters[subset]
                pred_set = set(pred_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
                c["d_tp"] += len(pred_set & true_set)
                c["d_fp"] += len(pred_set - true_set)
                c["d_fn"] += len(true_set - pred_set)
                if K > 0:
                    top_locs = pred[b].topk(K).indices.tolist()
                    c["tkr_hits"]  += len(set(top_locs) & true_set)
                    c["tkr_total"] += K

    return {
        k: _make_row(c["d_tp"], c["d_fp"], c["d_fn"],
                     c["tkr_hits"], c["tkr_total"], 0, 0, 0, False)
        for k, c in counters.items()
    }


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

    _print_table_7story(rows, model_name)
    if out_stem is not None:
        _save_results_7story(rows, out_stem)
    return rows


def _print_table_7story(rows: list[dict], model_name: str) -> None:
    print(f"\n=== {model_name} ===")
    col_w = 12
    fields = ["fault_type", "n_faulted", "subset"] + _ALL_METRICS
    header = "  ".join(f"{h:<{col_w}}" for h in fields)
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = [r["fault_type"], str(r["n_faulted"]), r["subset"]] + [
            f"{r.get(m, float('nan')):.4f}"
            if not math.isnan(r.get(m, float("nan"))) else "N/A"
            for m in _ALL_METRICS
        ]
        print("  ".join(f"{v:<{col_w}}" for v in vals))
    print()


def _save_results_7story(rows: list[dict], stem) -> None:
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
    fields   = ["model", "fault_type", "n_faulted", "subset"] + _ALL_METRICS
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, float("nan")) for k in fields})

    print(f"[results] Saved → {json_path}")
    print(f"[results] Saved → {csv_path}")


# ---------------------------------------------------------------------------
# Shared result helpers
# ---------------------------------------------------------------------------

def _make_row(
    d_tp, d_fp, d_fn, tkr_hits, tkr_total,
    f_tp, f_fp, f_fn, has_fault_head,
) -> dict[str, float]:
    f1, prec, rec = f1_from_counts(d_tp, d_fp, d_fn)
    row: dict[str, float] = {
        "f1":           f1,
        "precision":    prec,
        "recall":       rec,
        "top_k_recall": tkr_hits / max(tkr_total, 1),
    }
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
    print(f"\n=== {model_name} ===")
    col_w = 12
    fields = ["fault_type", "n_faulted"] + _ALL_METRICS
    header = "  ".join(f"{h:<{col_w}}" for h in fields)
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = [r["fault_type"], str(r["n_faulted"])] + [
            f"{r.get(m, float('nan')):.4f}"
            if not math.isnan(r.get(m, float("nan"))) else "N/A"
            for m in _ALL_METRICS
        ]
        print("  ".join(f"{v:<{col_w}}" for v in vals))
    print()


def _save_results(rows: list[dict], stem: Path) -> None:
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
    fields   = ["model", "fault_type", "n_faulted"] + _ALL_METRICS
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fault robustness sweep for C, v1, and DR heads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",      default="qatar", choices=["qatar", "7story"],
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
                             "data/7-story-frame/safetensors/unc=0 for 7story).")
    parser.add_argument("--fault-types",  nargs="+", default=FAULT_TYPES,
                        choices=FAULT_TYPES)
    parser.add_argument("--n-faulted",    nargs="+", type=int,
                        default=[0, 1, 3, 5, 10, 15])
    parser.add_argument("--n-repeats",    type=int, default=3)
    parser.add_argument("--batch-size",   type=int, default=64)
    # Qatar-specific windowing
    parser.add_argument("--window-size",  type=int, default=2048)
    parser.add_argument("--overlap",      type=float, default=0.5)
    parser.add_argument("--downsample",   type=int, default=4)
    parser.add_argument("--out",          default=None, metavar="STEM",
                        help="Output path stem. Model name is appended automatically "
                             "when multiple models are evaluated, e.g. <stem>_c.json. "
                             "Defaults to saved_results/<dataset>/eval_fault_<ts>.")
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

    # -------------------------------------------------------------------------
    # Qatar branch (original recording-consistent injection)
    # -------------------------------------------------------------------------
    if args.dataset == "qatar":
        root = args.root or "data/Qatar/processed"
        print(f"[evaluate_fault] Loading Qatar test recordings from: {root}")
        recordings = get_qatar_test_by_recording(
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
            _run_sweep(args.c_label, _run_condition_c, model_c, recordings,
                       args.fault_types, args.n_faulted, args.n_repeats,
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
            )
            _run_sweep(args.v1_label, run_fn, model_v1, recordings,
                       args.fault_types, args.n_faulted, args.n_repeats,
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
                       args.fault_types, args.n_faulted, args.n_repeats,
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
                       args.fault_types, args.n_faulted, args.n_repeats,
                       args.batch_size, device, _stem("b"))
            del model_b

    # -------------------------------------------------------------------------
    # 7-story branch (per-batch fault injection, split by K=1/K=2)
    # -------------------------------------------------------------------------
    else:
        from lib.data_7story import get_7story_dataloaders
        root = args.root or "data/7-story-frame/safetensors/unc=0"
        print(f"[evaluate_fault] Loading 7-story test set from: {root}")
        _, _, test_dl = get_7story_dataloaders(
            ["single", "double"],
            root=root,
            num_workers=0,
            eval_batch_size=args.batch_size,
        )
        n_samples = len(test_dl.dataset)
        print(f"[evaluate_fault] {n_samples} test samples")

        if args.c is not None:
            print(f"\n[evaluate_fault] Loading C checkpoint: {args.c}")
            model_c = load_model_c_from_checkpoint(args.c, device=device)
            model_c.eval()
            run_fn = lambda model, dl, ft, nf, nr, dev: _run_condition_7story_c(
                model, dl, ft, nf, nr, dev)
            _run_sweep_7story(args.c_label, run_fn, model_c, test_dl,
                              args.fault_types, args.n_faulted, args.n_repeats,
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
            )
            _run_sweep_7story(args.v1_label, run_fn, model_v1, test_dl,
                              args.fault_types, args.n_faulted, args.n_repeats,
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
                              args.fault_types, args.n_faulted, args.n_repeats,
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
                              args.fault_types, args.n_faulted, args.n_repeats,
                              device, _stem("b"))
            del model_b


if __name__ == "__main__":
    main()
