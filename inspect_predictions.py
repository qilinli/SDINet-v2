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

Tower output  → saved_results/tower/{model}/predictions_{grid,scatter}.png
Safetensors   → saved_results/safetensors/{model}/predictions_{damaged,scatter}.png

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
from lib.data_qatar import (
    get_qatar_dataloaders, get_qatar_double_test_dataloader,
    QATAR_DEFAULT_ROOT, QATAR_N_SENSORS, QATAR_N_LOCATIONS,
)
from lib.data_7story import (
    load_real_test_tensors, DEFAULT_BENCHMARK, TWO_DAMAGE_BENCHMARK,
)
from lib.model import (
    ModelConfig, ModelConfigDR, ModelConfigC, build_model,
    load_model_from_checkpoint, load_model_dr_from_checkpoint, load_model_c_from_checkpoint,
)

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
def run_c(model: torch.nn.Module, X: np.ndarray, batch_size: int = 256,
          use_severity: bool = True) -> np.ndarray:
    """
    Returns per-location score in [0,1] space for model C.
    MidnC outputs (slot_loc_logits, slot_sev) — we convert to a dense (N, L) map.

    use_severity=True  (default, for 7-story):
        damage_map[l] = Σ_k  is_obj[k] * sev[k] * loc_probs[k, l]
    use_severity=False (for Qatar, where sev_weight=0.0 so sev head is untrained):
        damage_map[l] = Σ_k  is_obj[k] * loc_probs[k, l]
    """
    device = next(model.parameters()).device
    x_t = torch.from_numpy(X).float()
    if x_t.ndim == 3:
        x_t = x_t.unsqueeze(1)
    all_loc = []
    for i in range(0, len(x_t), batch_size):
        loc_logits, sev, _ = model(x_t[i:i+batch_size].to(device))
        # loc_logits: (B, K, L+1);  sev: (B, K)
        loc_prob = loc_logits.softmax(-1)               # (B, K, L+1)
        is_obj   = 1.0 - loc_prob[..., -1]             # (B, K)
        scale    = (is_obj * sev) if use_severity else is_obj  # (B, K)
        dense    = (loc_prob[..., :-1] * scale.unsqueeze(-1)).sum(dim=1)  # (B, L)
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
    Uses the identical split as get_7story_dataloaders.
    """
    from lib.data_7story import SevenStoryDataset, val_input_preprocess, target_preprocess
    from sklearn.model_selection import train_test_split
    from pathlib import Path as _Path
    from torch.utils.data import Subset

    root_p = _Path(root)
    Xs, Ys, subsets = [], [], []

    for name in ("single", "double"):
        ds_root = root_p / name
        if not ds_root.exists():
            continue
        ds = SevenStoryDataset(ds_root, [val_input_preprocess, target_preprocess])
        indices = np.arange(len(ds))
        _, valtest = train_test_split(indices, test_size=0.3, random_state=seed)
        _, test_idx = train_test_split(valtest, test_size=0.5, random_state=seed)

        for idx in test_idx:
            x, y = ds[idx]
            Xs.append(x)
            Ys.append(y)
            subsets.append(name)

    return np.stack(Xs), np.stack(Ys), np.array(subsets)


def load_dr_model_safetensors(ckpt_path: Path, n_sensors: int = 65) -> torch.nn.Module:
    return load_model_dr_from_checkpoint(ckpt_path, model_cfg=ModelConfigDR(n_sensors=n_sensors)).eval()


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


def load_c_model_safetensors(ckpt_path: Path, n_sensors: int = 65) -> torch.nn.Module:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    arch = _infer_c_arch(sd, num_locations=70)
    cfg = ModelConfigC(**arch, n_sensors=n_sensors)
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
    dataset_label: str = "safetensors test set, L=70",
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
        f"Prediction distribution: damaged vs undamaged locations\n({dataset_label})",
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
    dataset_label: str = "safetensors test set",
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
    ax.set_title(f"Predicted vs True — all locations ({dataset_label})", fontsize=9)
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


# ── Qatar data loading ────────────────────────────────────────────────────────

def load_qatar_test(
    root: str | Path,
    window_size: int = 2048,
    overlap: float = 0.5,
    downsample: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X_test (N,1,T,30), Y_norm (N,30) in {-1,+1}, subsets (N,) strings.
    Combines the single-damage test split (Dataset B last 50%) and the
    double-damage test set into one array, tagged by 'single' / 'double'.
    """
    dl_kwargs = dict(root=root, window_size=window_size, overlap=overlap,
                     downsample=downsample, eval_batch_size=256, num_workers=0)

    _, _, test_dl   = get_qatar_dataloaders(**dl_kwargs, train_batch_size=256)
    double_dl       = get_qatar_double_test_dataloader(**dl_kwargs)

    def _collect(dl):
        Xs, Ys = [], []
        for x, y in dl:
            Xs.append(x.numpy()); Ys.append(y.numpy())
        return np.concatenate(Xs), np.concatenate(Ys)

    X_s, Y_s = _collect(test_dl)
    X_d, Y_d = _collect(double_dl)

    X       = np.concatenate([X_s, X_d])
    Y       = np.concatenate([Y_s, Y_d])
    subsets = np.array(["single"] * len(X_s) + ["double"] * len(X_d))
    return X, Y, subsets


# ── Qatar model loading ────────────────────────────────────────────────────────

def load_v1_model_safetensors(ckpt_path: Path, n_sensors: int = 65) -> torch.nn.Module:
    return load_model_from_checkpoint(ckpt_path, model_cfg=ModelConfig(n_sensors=n_sensors)).eval()


def load_v1_model_qatar(ckpt_path: Path, time_len: int = 512) -> torch.nn.Module:
    cfg = ModelConfig(n_sensors=QATAR_N_SENSORS, time_len=time_len, out_channels=QATAR_N_LOCATIONS + 1)
    return load_model_from_checkpoint(ckpt_path, model_cfg=cfg).eval()


def load_dr_model_qatar(ckpt_path: Path, time_len: int = 512) -> torch.nn.Module:
    cfg = ModelConfigDR(n_sensors=QATAR_N_SENSORS, time_len=time_len, num_locations=QATAR_N_LOCATIONS)
    return load_model_dr_from_checkpoint(ckpt_path, model_cfg=cfg).eval()


def load_c_model_qatar(ckpt_path: Path) -> torch.nn.Module:
    return load_model_c_from_checkpoint(ckpt_path).eval()


# ── v1 inference ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_v1(model: torch.nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    Returns per-location score in [0,1] for the v1 head.
    Map = softmax(loc_pred) scaled by (dmg_pred + 1) / 2, so undamaged samples
    produce near-zero scores and damaged samples concentrate mass on the
    predicted location.
    """
    device = next(model.parameters()).device
    x_t = torch.from_numpy(X).float()
    if x_t.ndim == 3:
        x_t = x_t.unsqueeze(1)
    preds = []
    for i in range(0, len(x_t), batch_size):
        dmg, loc = model(x_t[i:i+batch_size].to(device))  # (B,1), (B,L)
        dmg_scale = (dmg + 1.0) / 2.0                      # (B,1) ∈ [0,1]
        pred = dmg_scale * loc.softmax(-1)                  # (B,L)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


# ── Real .mat benchmark ────────────────────────────────────────────────────────

def plot_real_benchmark(
    pred_single: np.ndarray,   # (70,) predicted scores in [0,1]
    pred_two:    np.ndarray,   # (70,) predicted scores in [0,1]
    model_name:  str,
    out_path:    Path,
) -> None:
    """
    Bar chart of predicted damage map vs ground truth for both real .mat benchmarks.

    Ground truth locations are highlighted in red; all others in steel blue.
    The ground truth label is in physical units (severity ∈ [0,1]); the normalised
    presence threshold maps to severity > 0 (i.e. any non-zero label is "damaged").

    Single-damage benchmark: joint 12 (0-indexed: 11), severity=0.125
    Two-damage benchmark:    joints 6 & 12 (0-indexed: 5 & 11), severity=0.125 each
    """
    _, y_single = load_real_test_tensors(spec=DEFAULT_BENCHMARK)
    _, y_two    = load_real_test_tensors(spec=TWO_DAMAGE_BENCHMARK)

    # Convert physical label to boolean damage presence
    gt_single = (y_single.squeeze().numpy() > 0)   # (70,) bool
    gt_two    = (y_two.squeeze().numpy()    > 0)   # (70,) bool

    # Raw physical severity for annotation (as stored in the .mat file)
    sev_single = y_single.squeeze().numpy()   # e.g. 0.125
    sev_two    = y_two.squeeze().numpy()

    L = 70
    xs = np.arange(L)

    fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

    for ax, pred, gt, sev, title_suffix in zip(
        axes,
        [pred_single, pred_two],
        [gt_single,   gt_two],
        [sev_single,  sev_two],
        ["Single-damage benchmark (joint 12, severity=0.125)",
         "Two-damage benchmark (joints 6 & 12, severity=0.125 each)"],
    ):
        colors = ["#e87171" if gt[i] else "#7eb5e8" for i in range(L)]
        ax.bar(xs, pred, color=colors, width=0.8, alpha=0.85, zorder=2)

        # mark ground-truth locations with vertical lines and severity label
        for loc in np.where(gt)[0]:
            ax.axvline(loc, color="red", linewidth=1.5, linestyle="--", alpha=0.7, zorder=3)
            ax.text(loc + 0.3, pred[loc] + 0.03,
                    f"GT={sev[loc]:.3f}", fontsize=7, color="red", va="bottom")

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Predicted score", fontsize=9)
        ax.set_title(title_suffix, fontsize=9)
        ax.set_xlim(-1, L)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4, zorder=0)

    axes[-1].set_xlabel("Location index (0-indexed, 0–69)", fontsize=9)

    fig.legend(
        handles=[mpatches.Patch(facecolor="#e87171", label="Damaged (GT)"),
                 mpatches.Patch(facecolor="#7eb5e8", label="Undamaged (GT)")],
        loc="upper right", fontsize=8,
    )
    v1_note = "  (V1: bar = softmax location probability, independent of global damage score)" if model_name == "v1" else ""
    fig.suptitle(
        f"{model_name.upper()} — Real physical benchmark\n"
        f"Bar height = model predicted score  |  Bar colour = ground truth (red=damaged, blue=undamaged){v1_note}",
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[inspect] Saved real benchmark plot → {out_path}")


def _run_real_inference(model, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Run model on both real .mat benchmarks, return (pred_single, pred_two) each (70,)."""
    device = next(model.parameters()).device

    def _infer(spec):
        x_raw, _ = load_real_test_tensors(spec=spec)   # (T, S_real)
        x = x_raw[None, None, ...].to(device)           # (1, 1, T, S_real)
        with torch.inference_mode():
            if model_name == "v1":
                dmg, loc = model(x)
                dmg_val = float(((dmg + 1.0) / 2.0).squeeze().item())
                # Distributed map: dmg_scale * softmax(loc) over L locations.
                # Consistent with distributed_map_v1 in metrics and run_v1.
                # Note: softmax here is post-hoc over locations (L dim); the
                # model's internal softmax is over sensors (S dim) for importance.
                pred = (dmg_val * loc.softmax(-1)).squeeze().cpu().numpy()
                print(f"  [v1] global dmg_scale={dmg_val:.4f}  "
                      f"({'damaged' if dmg_val > 0.5 else 'undamaged'})")
            elif model_name == "c":
                loc_logits, sev, _ = model(x)
                loc_prob = loc_logits.softmax(-1)
                is_obj   = 1.0 - loc_prob[..., -1]
                scale    = is_obj * sev
                pred = (loc_prob[..., :-1] * scale.unsqueeze(-1)).sum(dim=1).squeeze().cpu().numpy()
            else:  # dr
                pred = model(x).squeeze().cpu().numpy()
        return pred

    return _infer(DEFAULT_BENCHMARK), _infer(TWO_DAMAGE_BENCHMARK)


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
    parser.add_argument("--model",   choices=["v1", "dr", "c"], default="dr")
    parser.add_argument("--dataset", choices=["tower", "safetensors", "qatar", "7story-real", "7story-sparse-real"],
                        default="tower")
    parser.add_argument("--ckpt",    type=Path, default=None)
    parser.add_argument("--root",    type=Path, default=None)
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--window-size", type=int,   default=2048,
                        help="Qatar window size in samples (default 2048=2s @ 1024 Hz)")
    parser.add_argument("--downsample", type=int,   default=4,
                        help="Qatar decimation factor (default 4 → 256 Hz)")
    parser.add_argument("--overlap",    type=float, default=0.5,
                        help="Qatar window overlap fraction (default 0.5)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Override output directory (default: saved_results/{dataset}/{model}/)")
    args = parser.parse_args()

    # Maps "real" dataset aliases → (ckpt_dir_name, output_subpath, n_sensors)
    _REAL_MAP = {
        "7story-real":        ("7story",        "7story/real",        65),
        "7story-sparse-real": ("7story-sparse", "7story-sparse/real", 9),
    }
    if args.dataset in _REAL_MAP:
        ckpt_base, out_subpath, n_sensors = _REAL_MAP[args.dataset]
        dataset_for_ckpt = ckpt_base
        default_out_dir  = Path(f"saved_results/{out_subpath}/{args.model}")
    else:
        dataset_for_ckpt = args.dataset
        default_out_dir  = Path(f"saved_results/{args.dataset}/{args.model}")
    ckpt = args.ckpt or latest_ckpt(args.model, dataset_for_ckpt)
    print(f"[inspect] Loading {args.model.upper()} checkpoint: {ckpt}")

    out_dir = args.out_dir if args.out_dir is not None else default_out_dir
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Real .mat benchmark ────────────────────────────────────────────────────
    if args.dataset in _REAL_MAP:
        if args.model == "v1":
            model = load_v1_model_safetensors(ckpt, n_sensors=n_sensors)
        elif args.model == "c":
            model = load_c_model_safetensors(ckpt, n_sensors=n_sensors)
        else:
            model = load_dr_model_safetensors(ckpt, n_sensors=n_sensors)

        model = model.to(device)
        pred_single, pred_two = _run_real_inference(model, args.model)

        print(f"\n[real-single]  top-3 predicted locations: {np.argsort(pred_single)[-3:][::-1].tolist()}")
        print(f"               GT damaged location: 11 (1-indexed: 12), severity=0.125")
        print(f"               predicted score at GT loc: {pred_single[11]:.4f}")
        print(f"\n[real-two]     top-3 predicted locations: {np.argsort(pred_two)[-3:][::-1].tolist()}")
        print(f"               GT damaged locations: 5, 11 (1-indexed: 6, 12), severity=0.125 each")
        print(f"               predicted scores at GT locs: {pred_two[5]:.4f}, {pred_two[11]:.4f}")

        plot_real_benchmark(pred_single, pred_two, args.model,
                            out_dir / "real_benchmark.png")

    # ── Tower ──────────────────────────────────────────────────────────────────
    elif args.dataset == "tower":
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

    # ── Qatar ──────────────────────────────────────────────────────────────────
    elif args.dataset == "qatar":
        time_len = args.window_size // args.downsample
        if args.model == "v1":
            model  = load_v1_model_qatar(ckpt, time_len=time_len)
            run_fn = run_v1
        elif args.model == "dr":
            model  = load_dr_model_qatar(ckpt, time_len=time_len)
            run_fn = run_dr
        else:
            model  = load_c_model_qatar(ckpt)
            run_fn = lambda m, X: run_c(m, X, use_severity=False)

        model = model.to(device)
        root  = str(args.root) if args.root else QATAR_DEFAULT_ROOT

        print("[inspect] Loading Qatar test splits (single + double) …")
        X_test, Y_norm, subsets = load_qatar_test(
            root, window_size=args.window_size,
            overlap=args.overlap, downsample=args.downsample,
        )
        print(f"[inspect] single={( subsets=='single').sum()}  double={(subsets=='double').sum()} windows")

        pred = run_fn(model, X_test)
        print_safetensors_table(pred, Y_norm, subsets)
        plot_safetensors_damaged_vs_undamaged(pred, Y_norm, subsets,
                                              out_dir / "predictions_damaged.png",
                                              dataset_label="Qatar SHM Benchmark, L=30")
        plot_safetensors_scatter(pred, Y_norm, subsets,
                                 out_dir / "predictions_scatter.png",
                                 dataset_label="Qatar SHM Benchmark")

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
