"""Per-K-damage F1 breakdown for ASCE checkpoints (clean test split)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lib.data_asce import get_asce_dataloaders
from lib.metrics import PRESENCE_NORM_THRESH, _c_slot_decode, f1_from_counts
from lib.model import (
    load_model_c_from_checkpoint,
    load_model_dr_from_checkpoint,
    load_model_from_checkpoint,
)
from lib.calibration import load_calibration


@torch.inference_mode()
def eval_c(model, dl, device):
    counts = {k: [0, 0, 0, 0, 0] for k in range(6)}  # k -> [d_tp, d_fp, d_fn, tkr_hits, tkr_tot]
    for x, y in dl:
        x, y = x.float().to(device), y.to(device)
        loc_logits, severity, fault_prob, _ = model(x)
        active, pred_loc, is_obj = _c_slot_decode(loc_logits)
        y_pres = y > PRESENCE_NORM_THRESH
        k_true = y_pres.sum(-1)
        for b in range(x.size(0)):
            K = int(k_true[b].item())
            pred_set = set(pred_loc[b, active[b]].tolist())
            true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
            counts[K][0] += len(pred_set & true_set)
            counts[K][1] += len(pred_set - true_set)
            counts[K][2] += len(true_set - pred_set)
            if K > 0:
                K_slots = min(K, is_obj.size(-1))
                top_slots = is_obj[b].topk(K_slots).indices
                ps = set(pred_loc[b, top_slots].tolist())
                counts[K][3] += len(ps & true_set)
                counts[K][4] += K
    return counts


@torch.inference_mode()
def eval_v1(model, dl, device, cal):
    temperature = cal.get("temperature", 1.0)
    ratio_alpha = cal.get("ratio_alpha")
    ratio_beta = cal.get("ratio_beta", 0.0)
    dmg_gate = cal.get("dmg_gate")
    counts = {k: [0, 0, 0, 0, 0] for k in range(6)}
    for x, y in dl:
        x, y = x.float().to(device), y.to(device)
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
            pred_set = set(pred_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
            true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
            counts[K][0] += len(pred_set & true_set)
            counts[K][1] += len(pred_set - true_set)
            counts[K][2] += len(true_set - pred_set)
            if K > 0:
                top_locs = probs[b].topk(K).indices.tolist()
                counts[K][3] += len(set(top_locs) & true_set)
                counts[K][4] += K
    return counts


@torch.inference_mode()
def eval_dr(model, dl, device, cal):
    threshold = cal.get("threshold", 0.5)
    ratio_alpha = cal.get("ratio_alpha")
    ratio_beta = cal.get("ratio_beta", 0.0)
    counts = {k: [0, 0, 0, 0, 0] for k in range(6)}
    for x, y in dl:
        x, y = x.float().to(device), y.to(device)
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
            pred_set = set(pred_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
            true_set = set(y_pres[b].nonzero(as_tuple=False)[:, 0].tolist())
            counts[K][0] += len(pred_set & true_set)
            counts[K][1] += len(pred_set - true_set)
            counts[K][2] += len(true_set - pred_set)
            if K > 0:
                top_locs = pred[b].topk(K).indices.tolist()
                counts[K][3] += len(set(top_locs) & true_set)
                counts[K][4] += K
    return counts


def report(name, counts):
    print(f"\n=== {name} ===")
    print(f"{'K':<4} {'n_samp':<8} {'F1':<8} {'prec':<8} {'recall':<8} {'top_k_rec':<10}")
    for K in sorted(counts.keys()):
        d_tp, d_fp, d_fn, tkr_hits, tkr_tot = counts[K]
        n_samp = (d_tp + d_fn) // K if K > 0 else d_fp  # rough sample count per K
        f1, prec, rec = f1_from_counts(d_tp, d_fp, d_fn)
        tkr = tkr_hits / max(tkr_tot, 1)
        print(f"{K:<4} {n_samp:<8} {f1:<8.3f} {prec:<8.3f} {rec:<8.3f} {tkr:<10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/asce_hammer")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_dl = get_asce_dataloaders(
        root=args.root, num_workers=0, eval_batch_size=args.batch_size
    )
    print(f"[eval] {len(test_dl.dataset)} test samples")

    ckpts = {
        "B":        ("states/asce-fault/b-cd4270df-1574-4a50-8c76-c53fe258e470.pt",       "dr"),
        "v1":       ("states/asce-fault/v1-c38816b6-d5e4-431d-ba99-38239b52ab4e.pt",      "v1"),
        "C":        ("states/asce-fault/c-2f1ecb34-1b14-44fa-9c77-412c671fd66b.pt",       "c"),
        "C+fh":     ("states/asce-fault/c-fh-1f32f253-89bd-4b0b-a23d-5c8905ce5efd.pt",    "c"),
        "C+fh+sb":  ("states/asce-fault/c-fh-sb-506d43b4-2861-4614-8474-c8599ae944f2.pt", "c"),
    }

    for label, (ckpt, kind) in ckpts.items():
        if not Path(ckpt).exists():
            print(f"[skip] {label}: {ckpt} not found")
            continue
        print(f"\n[load] {label}: {ckpt}")
        if kind == "c":
            model = load_model_c_from_checkpoint(ckpt, device=device).eval()
            counts = eval_c(model, test_dl, device)
        elif kind == "v1":
            model = load_model_from_checkpoint(ckpt, device=device).eval()
            cal = load_calibration(ckpt)
            counts = eval_v1(model, test_dl, device, cal)
        else:
            model = load_model_dr_from_checkpoint(ckpt, device=device).eval()
            cal = load_calibration(ckpt)
            counts = eval_dr(model, test_dl, device, cal)
        report(label, counts)
        del model


if __name__ == "__main__":
    main()
