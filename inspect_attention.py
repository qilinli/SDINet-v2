"""
inspect_attention.py — Visualise slot cross-attention of C+FH+SB on the Qatar test set.

Four plots are produced:

1. r_bias_comparison.png — Learned R_bias rows vs binary-init adjacency for 6 joints.
2. attn_clean.png        — Layer-0 vs layer-1 attention on sensor grid for clean samples.
3. attn_fault.png        — Attention with clean / near-fault / far-fault conditions.
4. attn_entropy.png      — Shannon entropy scatter: layer-0 vs layer-1, all test windows.

Usage
-----
    python inspect_attention.py \\
        --ckpt states/qatar-fault-sb/c-combined-85e69e27-0a5d-439d-b930-d687ca6c1e3b.pt

    python inspect_attention.py --ckpt <path> --out-dir saved_results/my_run/attention
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch import Tensor

from lib.data_qatar import (
    QATAR_GRID_ROWS, QATAR_GRID_COLS, QATAR_N_SENSORS, QATAR_N_LOCATIONS,
    QATAR_DEFAULT_ROOT,
    build_structural_affinity,
    get_qatar_test_by_recording,
    _apply_soft_fault,
)
from lib.metrics import PRESENCE_NORM_THRESH, _c_slot_decode
from lib.model import load_model_c_from_checkpoint

_ROWS = QATAR_GRID_ROWS   # 6
_COLS = QATAR_GRID_COLS   # 5
_S    = QATAR_N_SENSORS   # 30
_L    = QATAR_N_LOCATIONS # 30


# ---------------------------------------------------------------------------
# Attention-capturing forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def _decoder_forward_with_attn(
    head: torch.nn.Module,
    memory: Tensor,
) -> tuple[Tensor, list[Tensor]]:
    """
    Run MidnC's slot decoder manually, capturing cross-attention weights at every layer.

    PyTorch's TransformerDecoder calls MultiheadAttention with need_weights=False;
    we re-implement the norm_first=True layer loop to pass need_weights=True and
    recover (B, K, S) attention maps per layer.

    For layers ≥ 1 with structural bias the R_bias is included so the captured weights
    reflect the actual biased attention used in the forward pass.

    Args:
        head:    MidnC module (model[2])
        memory:  (B, S, embed_dim) sensor features from neck

    Returns:
        tgt:        (B, K, embed_dim) final slot states
        attn_list:  list of (B, K, S) tensors — one per decoder layer
    """
    B, S, E = memory.shape
    K = head.num_slots

    tgt = head.queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, E)
    attn_list: list[Tensor] = []

    for i, layer in enumerate(head.decoder.layers):
        # --- self-attention block ---
        tgt2 = layer.norm1(tgt)
        tgt2 = layer.self_attn(tgt2, tgt2, tgt2, need_weights=False)[0]
        tgt  = tgt + layer.dropout1(tgt2)

        # --- cross-attention block: capture weights ---
        tgt2 = layer.norm2(tgt)
        attn_mask = None
        if head.use_structural_bias and i > 0:
            loc_probs = head.loc_head(tgt).softmax(-1)                         # (B, K, L+1)
            bias      = loc_probs @ head.R_bias                                 # (B, K, S)
            bias      = bias.unsqueeze(1).expand(-1, head._nhead, -1, -1)
            attn_mask = bias.reshape(B * head._nhead, K, S)                    # (B*H, K, S)

        tgt2, w = layer.multihead_attn(
            tgt2, memory, memory,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True,
        )   # w: (B, K, S)
        attn_list.append(w.cpu())
        tgt = tgt + layer.dropout2(tgt2)

        # --- FFN block ---
        tgt2 = layer.norm3(tgt)
        tgt2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt2))))
        tgt  = tgt + layer.dropout3(tgt2)

    return tgt, attn_list


@torch.no_grad()
def _run_with_attn(
    model: torch.nn.Sequential,
    x: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor | None, list[Tensor]]:
    """
    Full model forward with attention capture.

    Returns:
        loc_logits:  (B, K, L+1)
        severity:    (B, K)
        fault_prob:  (B, S) or None
        attn_list:   list of (B, K, S) per decoder layer
    """
    x = x.to(device)
    backbone, neck, head = model[0], model[1], model[2]

    feats  = neck(backbone(x))          # (B, embed_dim, S)
    memory = feats.permute(0, 2, 1)     # (B, S, embed_dim)

    if head.use_spatial_layer:
        memory = head.pos_enc(memory)
        memory = head.spatial(memory)

    tgt, attn_list = _decoder_forward_with_attn(head, memory)

    loc_logits = head.loc_head(tgt)              # (B, K, L+1)
    severity   = head.sev_head(tgt).squeeze(-1)  # (B, K)
    fault_prob = None
    if head.use_fault_head:
        fault_prob = torch.sigmoid(head.fault_head(memory)).squeeze(-1).cpu()  # (B, S)

    return loc_logits.cpu(), severity.cpu(), fault_prob, attn_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_coords(idx: int) -> tuple[int, int]:
    """Sensor/joint index → (row, col) on 6×5 grid."""
    return divmod(idx, _COLS)


def _manhattan(a: int, b: int) -> int:
    ra, ca = _grid_coords(a)
    rb, cb = _grid_coords(b)
    return abs(ra - rb) + abs(ca - cb)


def _active_slot_attn(attn: Tensor, active: Tensor) -> Tensor:
    """
    Average cross-attention over active slots.

    Args:
        attn:   (K, S) attention weights for one sample
        active: (K,)   bool mask of active slots
    Returns:
        (S,) mean attention; if no active slot, uniform attention
    """
    if active.any():
        return attn[active].mean(0)
    return torch.ones(_S) / _S


def _entropy(w: Tensor) -> float:
    """Shannon entropy of a probability vector (nats)."""
    w = w.clamp(min=1e-9)
    return float(-(w * w.log()).sum())


def _imshow_grid(ax: plt.Axes, values: np.ndarray, vmin=None, vmax=None,
                 cmap="YlOrRd", title: str = "", colorbar: bool = False) -> None:
    """Plot a (6,5) array as a sensor-grid heatmap."""
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _mark_cell(ax: plt.Axes, idx: int, marker: str, color: str, size: int = 12) -> None:
    """Overlay a marker at grid cell corresponding to sensor/joint index."""
    row, col = _grid_coords(idx)
    ax.plot(col, row, marker, color=color, markersize=size,
            markeredgewidth=1.5, markeredgecolor="white")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[inspect_attention] Saved → {path}")


# ---------------------------------------------------------------------------
# Plot 1 — R_bias: learned vs binary init
# ---------------------------------------------------------------------------

def plot_r_bias(head: torch.nn.Module, out_dir: Path) -> None:
    """
    For 6 representative joints: compare binary 4-connected adjacency init
    to the learned R_bias row after training.
    Layout: 2 rows × 6 cols (row 0 = init, row 1 = learned).
    """
    if not head.use_structural_bias:
        print("[plot_r_bias] Model has no R_bias — skipping.")
        return

    # corners + 2 interior joints — spread across full 6×5 grid
    joints = [0, 4, 12, 17, 25, 29]   # (r0,c0) (r0,c4) (r2,c2) (r3,c2) (r5,c0) (r5,c4)
    R_init    = build_structural_affinity().numpy()   # (L+1, S)
    R_learned = head.R_bias.detach().cpu().numpy()    # (L+1, S)

    fig, axes = plt.subplots(2, len(joints), figsize=(2.4 * len(joints), 5.2))
    fig.suptitle(
        "R_bias: 4-connected adjacency init (top) vs learned after training (bottom)\n"
        "× marks the joint whose affinity row is shown",
        fontsize=9,
    )

    # diverging colormap for learned: blue=negative (suppress), white=0, red=positive (boost)
    abs_max = float(np.abs(R_learned[:_L]).max())
    for col_idx, l in enumerate(joints):
        row_l, col_l = _grid_coords(l)
        col_title = f"joint {l}  (row {row_l}, col {col_l})"

        # init row (binary 0/1)
        ax0 = axes[0, col_idx]
        _imshow_grid(ax0, R_init[l].reshape(_ROWS, _COLS),
                     vmin=0, vmax=1, cmap="Blues",
                     title=col_title, colorbar=True)
        _mark_cell(ax0, l, "x", "red", 11)
        if col_idx == 0:
            ax0.set_ylabel("init (binary)", fontsize=8)

        # learned row (continuous, may be negative)
        ax1 = axes[1, col_idx]
        _imshow_grid(ax1, R_learned[l].reshape(_ROWS, _COLS),
                     vmin=-abs_max, vmax=abs_max, cmap="bwr",
                     title="", colorbar=True)
        _mark_cell(ax1, l, "x", "black", 11)
        if col_idx == 0:
            ax1.set_ylabel("learned", fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir / "r_bias_comparison.png")


# ---------------------------------------------------------------------------
# Plot 2 — Clean attention: layer 0 vs layer 1
# ---------------------------------------------------------------------------

def plot_clean_attn(
    model: torch.nn.Sequential,
    recordings: list[tuple[Tensor, Tensor]],
    out_dir: Path,
    device: torch.device,
    n_samples: int = 8,
    seed: int = 0,
) -> None:
    """
    For N correctly predicted single-damage test samples:
    show [layer-0 attn | layer-1 attn | R_bias row | joint grid] per row.
    """
    head = model[2]
    rng  = np.random.default_rng(seed)

    # One candidate per recording — iterate ALL recordings so every damage location
    # is represented before we select for diversity.
    candidates: dict[int, dict] = {}   # keyed by true_loc; keeps one sample per location
    for x_rec, y_rec in recordings:
        true_locs = (y_rec > PRESENCE_NORM_THRESH).nonzero(as_tuple=False)
        unique_locs = true_locs[:, 1].unique().tolist() if len(true_locs) else []
        if len(unique_locs) != 1:
            continue   # skip healthy or multi-damage recordings
        true_loc = unique_locs[0]
        if true_loc in candidates:
            continue   # already have this location
        # use a random window from this recording for variety
        win_idx = int(rng.integers(0, x_rec.size(0)))
        loc_logits, _, _, attn_list = _run_with_attn(model, x_rec[win_idx:win_idx+1], device)
        active, pred_loc, _ = _c_slot_decode(loc_logits)
        pred_locs = pred_loc[0, active[0]].tolist()
        if pred_locs == [true_loc]:
            candidates[true_loc] = {
                "true_loc":  true_loc,
                "pred_loc":  pred_locs[0],
                "attn_list": [a[0] for a in attn_list],
                "active":    active[0],
            }

    if not candidates:
        print("[plot_clean_attn] No correctly predicted samples found — skipping.")
        return

    # pick n_samples spread across the grid
    all_locs = sorted(candidates.keys())
    step = max(1, len(all_locs) // n_samples)
    chosen = [candidates[all_locs[i]] for i in range(0, len(all_locs), step)][:n_samples]

    n_layers = len(chosen[0]["attn_list"])
    R_learned = head.R_bias.detach().cpu().numpy() if head.use_structural_bias else None

    # layout: n_chosen rows × (n_layers + 1 + 1) cols
    #   cols: layer-0, layer-1, ..., R_bias[pred], joint-grid
    n_cols = n_layers + 1 + 1
    fig, axes = plt.subplots(len(chosen), n_cols,
                             figsize=(2.6 * n_cols, 2.4 * len(chosen)),
                             constrained_layout=True)
    if len(chosen) == 1:
        axes = axes[np.newaxis, :]

    col_titles = [f"Layer-{i} attention" for i in range(n_layers)]
    col_titles += ["R_bias[pred loc]", "Joint grid"]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=10, fontweight="bold")

    for ri, s in enumerate(chosen):
        true_l  = s["true_loc"]
        pred_l  = s["pred_loc"]
        active  = s["active"]
        attn_layers = s["attn_list"]

        # attention: average over active slots → (S,)
        attn_maps = [_active_slot_attn(a, active).numpy() for a in attn_layers]
        vmax_attn = max(a.max() for a in attn_maps)

        for li, amap in enumerate(attn_maps):
            ax = axes[ri, li]
            _imshow_grid(ax, amap.reshape(_ROWS, _COLS),
                         vmin=0, vmax=vmax_attn, cmap="YlOrRd", colorbar=True)
            _mark_cell(ax, true_l, "x", "blue", 10)
            if true_l != pred_l:
                _mark_cell(ax, pred_l, "s", "cyan", 8)
            ax.set_ylabel(f"dmg={true_l}", fontsize=7)

        # R_bias row for predicted location
        ax_rb = axes[ri, n_layers]
        if R_learned is not None:
            abs_max = float(np.abs(R_learned[:_L]).max())
            rb = R_learned[pred_l].reshape(_ROWS, _COLS)
            _imshow_grid(ax_rb, rb, vmin=-abs_max, vmax=abs_max,
                         cmap="bwr", colorbar=True)
            _mark_cell(ax_rb, pred_l, "x", "black", 10)
        else:
            ax_rb.axis("off")
            ax_rb.text(0.5, 0.5, "no R_bias", ha="center", va="center", fontsize=7)

        # joint grid: true vs predicted
        ax_g = axes[ri, n_layers + 1]
        grid_map = np.zeros((_ROWS, _COLS))
        tr, tc = _grid_coords(true_l)
        pr, pc = _grid_coords(pred_l)
        grid_map[tr, tc] = 1.0
        if true_l != pred_l:
            grid_map[pr, pc] = 0.5
        _imshow_grid(ax_g, grid_map, vmin=0, vmax=1, cmap="Greens")
        _mark_cell(ax_g, true_l, "x", "red", 12)
        if true_l != pred_l:
            _mark_cell(ax_g, pred_l, "s", "orange", 8)

    legend = [
        mpatches.Patch(color="blue",  label="true damage location (×)"),
        mpatches.Patch(color="cyan",  label="predicted location (□, if wrong)"),
        mpatches.Patch(color="white", label="color: yellow=low attn, red=high attn"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Cross-attention on sensor grid (clean test samples)", fontsize=10)
    _save(fig, out_dir / "attn_clean.png")


# ---------------------------------------------------------------------------
# Plot 3 — Fault injection comparison
# ---------------------------------------------------------------------------

def plot_fault_attn(
    model: torch.nn.Sequential,
    recordings: list[tuple[Tensor, Tensor]],
    out_dir: Path,
    device: torch.device,
    n_samples: int = 4,
    fault_type: str = "bias",
    seed: int = 0,
) -> None:
    """
    For N test samples: compare layer-1 attention under
    [clean | near-fault | far-fault] conditions, all on the 6×5 grid.
    """
    rng = np.random.default_rng(seed)

    # One candidate per damage location across all recordings
    candidates: dict[int, dict] = {}
    for x_rec, y_rec in recordings:
        true_locs = (y_rec > PRESENCE_NORM_THRESH).nonzero(as_tuple=False)
        unique_locs = true_locs[:, 1].unique().tolist() if len(true_locs) else []
        if len(unique_locs) != 1:
            continue
        true_loc = unique_locs[0]
        if true_loc in candidates:
            continue
        win_idx = int(rng.integers(0, x_rec.size(0)))
        candidates[true_loc] = {
            "x":        x_rec[win_idx:win_idx+1].clone(),
            "true_loc": true_loc,
        }

    if not candidates:
        print("[plot_fault_attn] No suitable samples found — skipping.")
        return

    # pick n_samples spread across the grid
    all_locs = sorted(candidates.keys())
    step = max(1, len(all_locs) // n_samples)
    chosen_pool = [candidates[all_locs[i]] for i in range(0, len(all_locs), step)][:n_samples]

    fig, axes = plt.subplots(len(chosen_pool), 3,
                             figsize=(8.5, 2.8 * len(chosen_pool)),
                             constrained_layout=True)
    if len(chosen_pool) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Clean (no fault)",
                  f"Near-fault: adjacent sensor\nfaulted ({fault_type})",
                  f"Far-fault: distant sensor\nfaulted ({fault_type})"]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=10, fontweight="bold")

    def _get_attn(x_in: Tensor, last_layer: bool = True) -> np.ndarray:
        loc_logits, _, _, attn_list = _run_with_attn(model, x_in, device)
        active, _, _ = _c_slot_decode(loc_logits)
        layer_attn = attn_list[-1] if last_layer else attn_list[0]
        return _active_slot_attn(layer_attn[0], active[0]).numpy()

    for ri, s in enumerate(chosen_pool):
        x_clean  = s["x"]
        true_loc = s["true_loc"]

        near_candidates = [i for i in range(_S) if _manhattan(i, true_loc) <= 1 and i != true_loc]
        near_sensor = int(rng.choice(near_candidates)) if near_candidates else true_loc

        far_candidates = [i for i in range(_S) if _manhattan(i, true_loc) >= 4]
        far_sensor = int(rng.choice(far_candidates)) if far_candidates else (true_loc + 15) % _S

        # build fault versions — seed torch for reproducible magnitudes
        torch.manual_seed(seed + ri * 17)
        x_near = x_clean.clone()
        _apply_soft_fault(x_near[0], [near_sensor], fault_type)

        torch.manual_seed(seed + ri * 17 + 1)
        x_far = x_clean.clone()
        _apply_soft_fault(x_far[0], [far_sensor], fault_type)

        attn_clean = _get_attn(x_clean)
        attn_near  = _get_attn(x_near)
        attn_far   = _get_attn(x_far)

        vmax = max(attn_clean.max(), attn_near.max(), attn_far.max())

        for ci, (amap, fault_s) in enumerate([
            (attn_clean, None),
            (attn_near, near_sensor),
            (attn_far, far_sensor),
        ]):
            ax = axes[ri, ci]
            _imshow_grid(ax, amap.reshape(_ROWS, _COLS),
                         vmin=0, vmax=vmax, cmap="YlOrRd", colorbar=True)
            _mark_cell(ax, true_loc, "x", "blue", 12)
            if fault_s is not None:
                _mark_cell(ax, fault_s, "X", "black", 12)
            ax.set_ylabel(f"dmg={true_loc}", fontsize=7)

    legend = [
        mpatches.Patch(color="blue",  label="true damage location (×)"),
        mpatches.Patch(color="black", label="faulted sensor (✕)"),
        mpatches.Patch(color="white", label="color: yellow=low attn, red=high attn"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Layer-1 attention: fault injection ({fault_type})", fontsize=9)
    _save(fig, out_dir / "attn_fault.png")


# ---------------------------------------------------------------------------
# Plot 4 — Entropy scatter: layer 0 vs layer 1
# ---------------------------------------------------------------------------

def plot_entropy(
    model: torch.nn.Sequential,
    recordings: list[tuple[Tensor, Tensor]],
    out_dir: Path,
    device: torch.device,
    batch_size: int = 32,
) -> None:
    """
    Scatter plot of per-sample Shannon entropy at layer 0 vs layer 1.
    Points below y=x: R_bias sharpened attention.
    Colour encodes true damage location.
    """
    ent_l0: list[float] = []
    ent_l1: list[float] = []
    locs:   list[int]   = []

    for x_rec, y_rec in recordings:
        N = x_rec.size(0)
        for start in range(0, N, batch_size):
            xb = x_rec[start:start + batch_size]
            loc_logits, _, _, attn_list = _run_with_attn(model, xb, device)
            active, _, _ = _c_slot_decode(loc_logits)
            B = xb.size(0)
            for b in range(B):
                win_idx = start + b
                tl = (y_rec[win_idx] > PRESENCE_NORM_THRESH).nonzero(as_tuple=False)[:, 0].tolist()
                loc = tl[0] if len(tl) == 1 else -1
                a0 = _active_slot_attn(attn_list[0][b], active[b])
                a1 = _active_slot_attn(attn_list[-1][b], active[b])
                ent_l0.append(_entropy(a0))
                ent_l1.append(_entropy(a1))
                locs.append(loc)

    ent_l0_arr = np.array(ent_l0)
    ent_l1_arr = np.array(ent_l1)
    locs_arr   = np.array(locs)

    delta = ent_l1_arr - ent_l0_arr
    frac_lower = float((delta < 0).mean())
    print(f"[entropy] mean Δentropy (layer1 − layer0): {delta.mean():+.4f}")
    print(f"[entropy] fraction with lower entropy at layer 1: {frac_lower:.3f}")

    H_uniform = math.log(_S)   # entropy of uniform distribution over 30 sensors

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    cmap = plt.get_cmap("tab20", _L)
    for loc in range(_L):
        mask = locs_arr == loc
        if mask.any():
            ax.scatter(ent_l0_arr[mask], ent_l1_arr[mask],
                       c=[cmap(loc)], s=8, alpha=0.5, linewidths=0)

    lo = min(ent_l0_arr.min(), ent_l1_arr.min()) * 0.95
    hi = H_uniform * 1.05
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # y = x reference (no change between layers)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x  (no change)")

    # uniform entropy reference lines
    ax.axvline(H_uniform, color="gray", lw=0.8, ls=":",
               label=f"uniform ({H_uniform:.2f} nats)")
    ax.axhline(H_uniform, color="gray", lw=0.8, ls=":")
    ax.text(H_uniform + 0.02, lo + 0.05, "uniform", fontsize=7,
            color="gray", rotation=90, va="bottom")
    ax.text(lo + 0.02, H_uniform + 0.02, "uniform", fontsize=7, color="gray")

    # mean-value crosshair
    m0, m1 = ent_l0_arr.mean(), ent_l1_arr.mean()
    ax.axvline(m0, color="steelblue", lw=0.8, ls="--",
               label=f"layer-0 mean ({m0:.2f})")
    ax.axhline(m1, color="tomato",    lw=0.8, ls="--",
               label=f"layer-1 mean ({m1:.2f})")

    # shade regions
    ax.fill_between([lo, hi], [lo, hi], [hi, hi],
                    alpha=0.04, color="green",
                    label=f"below y=x: layer-1 sharper ({frac_lower*100:.1f}% of samples)")
    ax.fill_between([lo, hi], [lo, lo], [lo, hi],
                    alpha=0.04, color="orange",
                    label=f"above y=x: layer-1 more diffuse ({(1-frac_lower)*100:.1f}%)")

    ax.set_xlabel("Layer-0 entropy — nats  (unbiased, learned queries only)", fontsize=9)
    ax.set_ylabel("Layer-1 entropy — nats  (R_bias applied)", fontsize=9)
    ax.set_title(
        f"Cross-attention entropy: layer 0 vs layer 1\n"
        f"mean Δ = {delta.mean():+.4f} nats  |  "
        f"uniform = {H_uniform:.2f} nats  (perfectly spread attention)",
        fontsize=8,
    )
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    _save(fig, out_dir / "attn_entropy.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attention visualisation for C+FH+SB on Qatar.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt",        required=True,
                        help="C-head checkpoint (e.g. states/qatar-fault-sb/c-combined-*.pt)")
    parser.add_argument("--root",        default=QATAR_DEFAULT_ROOT,
                        help="Qatar processed data root")
    parser.add_argument("--out-dir",     default=None,
                        help="Output directory (default: saved_results/<run>/attention/)")
    parser.add_argument("--n-samples",   type=int, default=8,
                        help="Samples per panel in clean-attn and fault-attn plots")
    parser.add_argument("--fault-type",  default="gain",
                        choices=["gain", "bias", "noise", "partial"],
                        help="Fault type for the fault-injection comparison plot")
    parser.add_argument("--window-size", type=int,   default=2048)
    parser.add_argument("--overlap",     type=float, default=0.5)
    parser.add_argument("--downsample",  type=int,   default=4)
    parser.add_argument("--seed",        type=int,   default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        run_name = Path(args.ckpt).parent.name   # e.g. qatar-fault-sb
        out_dir  = Path("saved_results") / run_name / "attention"

    print(f"[inspect_attention] Checkpoint : {args.ckpt}")
    print(f"[inspect_attention] Output dir : {out_dir}")
    print(f"[inspect_attention] Device     : {device}")

    # load model
    model = load_model_c_from_checkpoint(args.ckpt, device=device)
    model.eval()
    head = model[2]

    n_layers = len(head.decoder.layers)
    print(f"[inspect_attention] Decoder layers: {n_layers}, "
          f"use_structural_bias: {head.use_structural_bias}")

    # load test data
    print(f"[inspect_attention] Loading test recordings from: {args.root}")
    recordings = get_qatar_test_by_recording(
        root=args.root,
        window_size=args.window_size,
        overlap=args.overlap,
        downsample=args.downsample,
    )
    n_windows = sum(x.size(0) for x, _ in recordings)
    print(f"[inspect_attention] {len(recordings)} recordings, {n_windows} windows")

    # --- Plot 1: R_bias ---
    plot_r_bias(head, out_dir)

    # --- Plot 2: clean attention ---
    plot_clean_attn(model, recordings, out_dir, device,
                    n_samples=args.n_samples, seed=args.seed)

    # --- Plot 3: fault injection ---
    plot_fault_attn(model, recordings, out_dir, device,
                    n_samples=min(args.n_samples, 6), fault_type=args.fault_type,
                    seed=args.seed)

    # --- Plot 4: entropy scatter ---
    plot_entropy(model, recordings, out_dir, device)

    print("[inspect_attention] Done.")


if __name__ == "__main__":
    main()
