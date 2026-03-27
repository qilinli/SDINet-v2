"""
inspect_predictions.py — Visualise DR / C model predictions vs ground truth.

Usage
-----
    # Tower dataset (default), latest DR checkpoint
    python inspect_predictions.py --model dr

    # Tower, explicit checkpoint
    python inspect_predictions.py --model dr --ckpt states/tower/dr-combined-<uuid>.pt

    # C model on tower
    python inspect_predictions.py --model c

    # Safetensors dataset
    python inspect_predictions.py --model dr --dataset safetensors
    python inspect_predictions.py --model c  --dataset safetensors

Tower output  → saved_results_{model}/tower/predictions_{grid,scatter}.png
Safetensors   → saved_results_{model}/safetensors/predictions_{damaged,scatter}.png

Tower plots (L=4 locations, 9 damage states)
--------------------------------------------
1. Grid: rows = damage states, columns = 4 locations.
   Box = predicted distribution over test windows.  Red dashed = ground truth.
2. Scatter: predicted vs true per window, one column per location, coloured by state.

Safetensors plots (L=70 locations, K=1 or K=2 damages per sample)
------------------------------------------------------------------
1. Damaged vs undamaged: violin of predictions at truly damaged locations vs all
   undamaged locations, split by single / double damage.  Shows separation quality.
2. Scatter: predicted value at each location vs true value (0 or 1), coloured by
   subset (single / double).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from lib.data_tower import (
    _load_arrays, _stratified_case_split,
    TOWER_DEFAULT_ROOT, TOWER_TIME_LEN, TOWER_N_SENSORS, TOWER_N_LOCATIONS,
)
from lib.model import ModelConfigDR, ModelConfigC, build_model

# ── constants ──────────────────────────────────────────────────────────────────

LOCATION_NAMES = ["B6", "B1", "Conn", "Floor3"]

DS_ORDER = ["healthy", "DS1", "DS2", "DS3", "DS4", "DS5", "DS6", "DS7", "DS8"]

# colour per damage state (matplotlib tab10 palette)
_CMAP = plt.get_cmap("tab10")
DS_COLOR = {ds: _CMAP(i) for i, ds in enumerate(DS_ORDER)}

EXCITATION_TYPES = ["EQ", "WN", "sine"]


# ── data loading with damage-state metadata ────────────────────────────────────

def load_test_split_with_labels(
    root: Path,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X_test, Y_test (raw [0,1]), state_labels (string per window).
    Uses the identical split logic as get_tower_dataloaders so test indices match.
    """
    Xs, Ys, case_ids, states = [], [], [], []
    case_idx = 0

    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        excitation = d.name.split("_")[-1]
        if excitation not in EXCITATION_TYPES:
            continue
        # parse damage state from folder name (e.g. "DS3_WN" → "DS3", "healthy_EQ" → "healthy")
        ds = d.name.rsplit("_", 1)[0]

        for npz in sorted(d.glob("*.npz")):
            f = np.load(npz)
            n = len(f["X"])
            Xs.append(f["X"])
            Ys.append(f["Y"])
            case_ids.extend([case_idx] * n)
            states.extend([ds] * n)
            case_idx += 1

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    case_ids_arr = np.array(case_ids)
    states_arr = np.array(states)

    unique_cases = np.unique(case_ids_arr)
    case_to_y = {c: Y[case_ids_arr == c][0] for c in unique_cases}
    _, _, test_cases = _stratified_case_split(unique_cases, case_to_y, seed)

    test_mask = np.isin(case_ids_arr, test_cases)
    return X[test_mask], Y[test_mask], states_arr[test_mask]


# ── model loading ──────────────────────────────────────────────────────────────

def load_dr_model(ckpt_path: Path) -> torch.nn.Module:
    cfg = ModelConfigDR(
        time_len=TOWER_TIME_LEN,
        n_sensors=TOWER_N_SENSORS,
        num_locations=TOWER_N_LOCATIONS,
    )
    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_c_model(ckpt_path: Path) -> torch.nn.Module:
    """Load a tower C model; infers architecture from checkpoint state dict."""
    from lib.model import ModelConfigC
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    arch = _infer_c_arch(sd, num_locations=TOWER_N_LOCATIONS)
    cfg = ModelConfigC(
        time_len=TOWER_TIME_LEN,
        n_sensors=TOWER_N_SENSORS,
        **arch,
    )
    model = build_model(cfg)
    model.load_state_dict(sd)
    model.eval()
    return model


# ── inference ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_dr(model: torch.nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Returns predictions in [0,1] space — same as raw Y labels."""
    device = next(model.parameters()).device
    preds = []
    x_t = torch.from_numpy(X).float()
    if x_t.ndim == 3:        # tower: (N, T, S) → (N, 1, T, S)
        x_t = x_t.unsqueeze(1)
    # safetensors: already (N, 1, T, S)
    for i in range(0, len(x_t), batch_size):
        preds.append(model(x_t[i:i+batch_size].to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)   # (N, L)


@torch.inference_mode()
def run_c(model: torch.nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    Returns per-location severity in [0,1] space for model C.
    MidnC outputs (slot_loc_logits, slot_sev) — we convert to a dense (N, L) map.
    """
    device = next(model.parameters()).device
    x_t = torch.from_numpy(X).float()
    if x_t.ndim == 3:
        x_t = x_t.unsqueeze(1)
    all_loc, all_sev = [], []
    for i in range(0, len(x_t), batch_size):
        loc_logits, sev = model(x_t[i:i+batch_size].to(device))
        # loc_logits: (B, K, L+1);  sev: (B, K)
        # Formula from MidnC docstring:
        #   is_obj[k]       = 1 − softmax(loc_logits[k])[∅]
        #   damage_map[l]   = Σ_k  is_obj[k] * sev[k] * loc_probs[k, l]
        loc_prob = loc_logits.softmax(-1)               # (B, K, L+1)
        is_obj   = 1.0 - loc_prob[..., -1]             # (B, K)
        weighted = (is_obj * sev).unsqueeze(-1)         # (B, K, 1)
        dense    = (loc_prob[..., :-1] * weighted).sum(dim=1)  # (B, L)
        all_loc.append(dense.cpu().numpy())
    return np.concatenate(all_loc, axis=0)


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_predictions(
    pred: np.ndarray,   # (N, L) in [0, 1]
    true: np.ndarray,   # (N, L) in [0, 1]
    states: np.ndarray, # (N,) strings
    out_path: Path,
) -> None:
    present_states = [ds for ds in DS_ORDER if ds in states]
    n_states = len(present_states)
    n_locs   = TOWER_N_LOCATIONS

    fig, axes = plt.subplots(
        n_states, n_locs,
        figsize=(n_locs * 3.0, n_states * 2.0),
        sharex=True, sharey=True,
    )
    axes = np.array(axes).reshape(n_states, n_locs)

    for row, ds in enumerate(present_states):
        mask = states == ds
        p = pred[mask]   # (n, L)
        t = true[mask]   # (n, L) — should be constant per location within a state
        color = DS_COLOR[ds]

        for col in range(n_locs):
            ax = axes[row, col]
            gt_val = float(t[0, col])

            bp = ax.boxplot(
                p[:, col],
                patch_artist=True,
                widths=0.5,
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker=".", markersize=2, alpha=0.4),
            )
            bp["boxes"][0].set_facecolor((*color[:3], 0.5))

            ax.axhline(gt_val, color="red", linewidth=1.5, linestyle="--", label=f"GT={gt_val:.2f}")
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(LOCATION_NAMES[col], fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ds, fontsize=8, rotation=0, labelpad=30, va="center")

    # legend
    gt_line = mpatches.Patch(color="red", label="Ground truth")
    pred_box = mpatches.Patch(facecolor="grey", alpha=0.5, label="Predicted (test windows)")
    fig.legend(handles=[gt_line, pred_box], loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Predicted vs Ground-Truth Severity per Damage State × Location", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[inspect] Saved grid plot → {out_path}")


def plot_scatter(
    pred: np.ndarray,
    true: np.ndarray,
    states: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, TOWER_N_LOCATIONS, figsize=(TOWER_N_LOCATIONS * 3.5, 3.5), sharey=True)

    for col, (ax, loc_name) in enumerate(zip(axes, LOCATION_NAMES)):
        for ds in DS_ORDER:
            mask = states == ds
            if not mask.any():
                continue
            ax.scatter(
                true[mask, col], pred[mask, col],
                color=DS_COLOR[ds], alpha=0.3, s=8, label=ds,
            )
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("True severity", fontsize=9)
        ax.set_title(loc_name, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

    axes[0].set_ylabel("Predicted severity", fontsize=9)

    handles = [mpatches.Patch(color=DS_COLOR[ds], label=ds)
               for ds in DS_ORDER if ds in states]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=7, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Predicted vs True Severity — scatter (test set, coloured by damage state)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[inspect] Saved scatter plot → {out_path}")


def print_table(pred: np.ndarray, true: np.ndarray, states: np.ndarray) -> None:
    """Print per-state mean prediction and GT for quick terminal check."""
    present_states = [ds for ds in DS_ORDER if ds in states]
    header = f"{'State':<10}" + "".join(f"  {loc:>12}" for loc in LOCATION_NAMES)
    print("\n" + header)
    print("-" * len(header))
    for ds in present_states:
        mask = states == ds
        gt  = true[mask][0]
        mean_p = pred[mask].mean(axis=0)
        std_p  = pred[mask].std(axis=0)
        row = f"{ds:<10}"
        for l in range(TOWER_N_LOCATIONS):
            row += f"  {mean_p[l]:+.3f}±{std_p[l]:.3f}"
        row += f"   GT: [{', '.join(f'{v:.2f}' for v in gt)}]"
        print(row)
    print()


# ── safetensors data loading ───────────────────────────────────────────────────

def load_safetensors_test(
    root: str = "data/7-story-frame/safetensors/unc=0",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X_test (N,1,500,65), Y_test_norm (N,70) in [-1,1], subset (N,) strings.
    Uses the identical split as get_combined_dataloaders.
    """
    from lib.data_safetensors import SafetensorsDataset, val_input_preprocess, target_preprocess
    from sklearn.model_selection import train_test_split
    from pathlib import Path as _Path
    from torch.utils.data import Subset

    root_p = _Path(root)
    Xs, Ys, subsets = [], [], []

    for name in ("single", "double"):
        ds_root = root_p / name
        if not ds_root.exists():
            continue
        ds = SafetensorsDataset(ds_root, [val_input_preprocess, target_preprocess])
        indices = np.arange(len(ds))
        _, valtest = train_test_split(indices, test_size=0.3, random_state=seed)
        _, test_idx = train_test_split(valtest, test_size=0.5, random_state=seed)

        for idx in test_idx:
            x, y = ds[idx]
            Xs.append(x)
            Ys.append(y)
            subsets.append(name)

    return np.stack(Xs), np.stack(Ys), np.array(subsets)


def load_dr_model_safetensors(ckpt_path: Path) -> torch.nn.Module:
    cfg = ModelConfigDR()   # defaults: time_len=500, n_sensors=65, num_locations=70
    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _infer_c_arch(state_dict: dict, num_locations: int) -> dict:
    """Infer num_decoder_layers, num_slots, nhead from checkpoint keys."""
    num_slots = state_dict["2.queries"].shape[0]
    layers = [k for k in state_dict if k.startswith("2.decoder.layers.")]
    layer_idxs = {int(k.split(".")[3]) for k in layers}
    num_decoder_layers = max(layer_idxs) + 1 if layer_idxs else 1
    # infer nhead from in_proj_weight: shape (3*embed, embed); embed from queries
    embed_dim = state_dict["2.queries"].shape[1]
    # pick any multihead_attn weight to derive nhead — can't directly, use default 8
    # (nhead only affects internal attention splits, doesn't change param count per se)
    # Use a heuristic: if embed_dim % 8 == 0 → nhead=8, else 4
    nhead = 8 if embed_dim % 8 == 0 else 4
    return dict(num_slots=num_slots, num_decoder_layers=num_decoder_layers,
                nhead=nhead, num_locations=num_locations)


def load_c_model_safetensors(ckpt_path: Path) -> torch.nn.Module:
    from lib.model import ModelConfigC
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    arch = _infer_c_arch(sd, num_locations=70)
    cfg = ModelConfigC(**arch)
    model = build_model(cfg)
    model.load_state_dict(sd)
    model.eval()
    return model


# ── safetensors plotting ───────────────────────────────────────────────────────

PRESENCE_THRESH = -1.0 + 1e-4   # y_norm > this → damaged

def plot_safetensors_damaged_vs_undamaged(
    pred: np.ndarray,    # (N, L) in [0,1]
    y_norm: np.ndarray,  # (N, L) in [-1,1]
    subsets: np.ndarray, # (N,) "single"/"double"
    out_path: Path,
) -> None:
    """
    Violin plot: predicted severity at damaged vs undamaged locations.
    One column per subset (single / double).
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    subset_list = [s for s in ("single", "double") if s in subsets]

    for ax, name in zip(axes, subset_list):
        mask = subsets == name
        p = pred[mask]      # (n, L)
        y = y_norm[mask]    # (n, L)

        damaged   = y > PRESENCE_THRESH   # (n, L) bool
        p_dam  = p[damaged].ravel()
        p_udam = p[~damaged].ravel()

        parts = ax.violinplot(
            [p_udam, p_dam],
            positions=[0, 1],
            showmedians=True,
            showextrema=True,
        )
        colors = ["#4878d0", "#ee854a"]
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Undamaged\nlocations", "Damaged\nlocations"], fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{name.capitalize()} damage (K={'1' if name=='single' else '2'})", fontsize=10)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="threshold=0.5")
        ax.set_ylabel("Predicted severity", fontsize=9)

        # annotate medians
        med_u = float(np.median(p_udam))
        med_d = float(np.median(p_dam))
        ax.text(0, med_u + 0.03, f"med={med_u:.3f}", ha="center", fontsize=7, color="navy")
        ax.text(1, med_d + 0.03, f"med={med_d:.3f}", ha="center", fontsize=7, color="darkorange")

    fig.suptitle(
        "Prediction distribution: damaged vs undamaged locations\n"
        "(safetensors test set, L=70)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[inspect] Saved damaged/undamaged plot → {out_path}")


def plot_safetensors_scatter(
    pred: np.ndarray,
    y_norm: np.ndarray,
    subsets: np.ndarray,
    out_path: Path,
) -> None:
    """
    Scatter: predicted value vs true value (mapped to [0,1]) for every
    location in every test sample.  Coloured by subset.
    """
    y_01 = (y_norm + 1.0) / 2.0   # map [-1,1] → [0,1]

    colors = {"single": "#4878d0", "double": "#ee854a"}
    fig, ax = plt.subplots(figsize=(5, 4))

    for name in ("single", "double"):
        mask = subsets == name
        if not mask.any():
            continue
        p_flat = pred[mask].ravel()
        y_flat = y_01[mask].ravel()
        ax.scatter(y_flat, p_flat, color=colors[name], alpha=0.03, s=2, label=name)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("True severity  (0 = undamaged, 1 = damaged)", fontsize=9)
    ax.set_ylabel("Predicted severity", fontsize=9)
    ax.set_title("Predicted vs True — all locations (safetensors test set)", fontsize=9)
    ax.set_aspect("equal")

    handles = [mpatches.Patch(color=colors[n], label=n) for n in ("single", "double") if n in subsets]
    ax.legend(handles=handles, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[inspect] Saved scatter plot → {out_path}")


def print_safetensors_table(pred: np.ndarray, y_norm: np.ndarray, subsets: np.ndarray) -> None:
    damaged = y_norm > PRESENCE_THRESH
    print(f"\n{'Subset':<10}  {'N samples':>10}  {'med@damaged':>12}  {'med@undamaged':>14}  {'separation':>10}")
    print("-" * 62)
    for name in ("single", "double"):
        mask = subsets == name
        if not mask.any():
            continue
        p = pred[mask]; d = damaged[mask]
        med_d = float(np.median(p[d]))
        med_u = float(np.median(p[~d]))
        print(f"{name:<10}  {mask.sum():>10}  {med_d:>12.4f}  {med_u:>14.4f}  {med_d - med_u:>10.4f}")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def latest_ckpt(model_type: str, dataset: str = "tower") -> Path:
    candidates = sorted(
        Path(f"states/{dataset}").glob(f"{model_type}-combined-*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        # fall back to root states/ for safetensors
        candidates = sorted(
            Path("states").glob(f"{model_type}-combined-*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
    if not candidates:
        raise FileNotFoundError(f"No {model_type} checkpoint found for dataset={dataset}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["dr", "c"], default="dr")
    parser.add_argument("--dataset", choices=["tower", "safetensors"], default="tower")
    parser.add_argument("--ckpt",    type=Path, default=None)
    parser.add_argument("--root",    type=Path, default=None)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    ckpt = args.ckpt or latest_ckpt(args.model, args.dataset)
    print(f"[inspect] Loading {args.model.upper()} checkpoint: {ckpt}")

    out_dir = Path(f"saved_results_{args.model}/{args.dataset}")
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Tower ──────────────────────────────────────────────────────────────────
    if args.dataset == "tower":
        if args.model == "dr":
            model = load_dr_model(ckpt)
            run_fn = run_dr
        else:
            model = load_c_model(ckpt)
            run_fn = run_c

        model = model.to(device)
        root = args.root or Path(TOWER_DEFAULT_ROOT)

        print("[inspect] Loading test split …")
        X_test, Y_test, states = load_test_split_with_labels(root, seed=args.seed)
        print(f"[inspect] Test windows: {len(X_test)}  states: {np.unique(states).tolist()}")

        pred = run_fn(model, X_test)
        print_table(pred, Y_test, states)
        plot_predictions(pred, Y_test, states, out_dir / "predictions_grid.png")
        plot_scatter(pred, Y_test, states,     out_dir / "predictions_scatter.png")

    # ── Safetensors ────────────────────────────────────────────────────────────
    else:
        if args.model == "dr":
            model = load_dr_model_safetensors(ckpt)
            run_fn = run_dr
        else:
            model = load_c_model_safetensors(ckpt)
            run_fn = run_c

        model = model.to(device)
        root = str(args.root) if args.root else "data/7-story-frame/safetensors/unc=0"

        print("[inspect] Loading safetensors test split …")
        X_test, Y_norm, subsets = load_safetensors_test(root, seed=args.seed)
        print(f"[inspect] Test samples: {len(X_test)}  subsets: {np.unique(subsets).tolist()}")

        pred = run_fn(model, X_test)
        print_safetensors_table(pred, Y_norm, subsets)
        plot_safetensors_damaged_vs_undamaged(pred, Y_norm, subsets,
                                              out_dir / "predictions_damaged.png")
        plot_safetensors_scatter(pred, Y_norm, subsets,
                                 out_dir / "predictions_scatter.png")


if __name__ == "__main__":
    main()
