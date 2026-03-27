"""
lib/data_tower.py — DataLoaders for the tower structural damage dataset.

Dataset layout::

    root/
        DS1_EQ/  DS1_WN/  DS2_EQ/  DS2_WN/ ... DS8_EQ/  DS8_WN/
        healthy_EQ/  healthy_WN/  healthy_sine/

Each directory contains .npz files with:
    X:  (N, 400, 6)  float32  — N windows × T time-steps × S sensors
    Y:  (N, 4)       float32  — damage severity ∈ [0, 1] per structural location

Label convention (identical to lib/data_safetensors.py):
    y_norm = Y * 2 − 1  ∈ [−1, 1]
    Undamaged (Y = 0)   →  y_norm = −1
    Max damaged (Y = 1) →  y_norm = +1

Dataset constants (passed to model configs):
    TOWER_TIME_LEN    = 400
    TOWER_N_SENSORS   = 6
    TOWER_N_LOCATIONS = 4
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

TOWER_DEFAULT_ROOT: str = "data/tower"

TOWER_TIME_LEN: int    = 400
TOWER_N_SENSORS: int   = 6
TOWER_N_LOCATIONS: int = 4


def _load_arrays(
    root: Path,
    excitation_types: list[str],
    include_healthy: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate all X / Y arrays and return a recording-level case ID per segment.

    Splitting is done at the .npz file level so all segments from the same
    physical test run (one .npz = one case) stay in the same partition,
    preventing leakage from overlapping 2-second windows.

    Returns
    -------
    X        : (N, T, S)
    Y        : (N, L)
    case_ids : (N,) int  — same value for all segments from the same .npz file
    """
    Xs, Ys, case_ids = [], [], []
    case_idx = 0

    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        excitation = d.name.split("_")[-1]          # "EQ", "WN", "sine", …
        is_healthy = d.name.startswith("healthy")
        if not include_healthy and is_healthy:
            continue
        if excitation not in excitation_types:
            continue

        for npz in sorted(d.glob("*.npz")):
            f = np.load(npz)
            n = len(f["X"])
            Xs.append(f["X"])
            Ys.append(f["Y"])
            case_ids.extend([case_idx] * n)
            case_idx += 1

    if not Xs:
        raise ValueError(
            f"No data found under {root} for excitation_types={excitation_types}. "
            "Check the path and excitation_types argument."
        )
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0), np.array(case_ids)


def _damage_category(y_first: np.ndarray) -> str:
    """Return 'healthy', 'single', or 'multi' for the first-segment label of a recording."""
    n = int((y_first > 0).sum())
    if n == 0:
        return "healthy"
    return "single" if n == 1 else "multi"


def _stratified_case_split(
    unique_cases: np.ndarray,
    case_to_y: dict[int, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split recordings into train / val / test with guaranteed representation of
    every damage category (healthy, single, multi) in every split.

    Strategy per category:
      - Allocate ~15 % to test  (at least 1, at most n-2 so train/val each keep ≥1)
      - Allocate ~15 % to val   (at least 1 if any remain, otherwise 0)
      - Rest to train

    This prevents the common failure where the two single-damage recordings both
    end up in train (probability ~69 % under a naive random split).
    """
    from collections import defaultdict

    rng = np.random.RandomState(seed)

    by_cat: dict[str, list[int]] = defaultdict(list)
    for c in unique_cases:
        by_cat[_damage_category(case_to_y[c])].append(int(c))

    train_list, val_list, test_list = [], [], []
    for cat in sorted(by_cat):          # deterministic order
        cases = by_cat[cat]
        rng.shuffle(cases)
        n = len(cases)
        n_test = min(max(1, round(n * 0.15)), n - 1)
        n_val  = min(max(1, round(n * 0.15)), n - n_test - 1)
        n_val  = max(n_val, 0)
        test_list.extend(cases[:n_test])
        val_list.extend(cases[n_test : n_test + n_val])
        train_list.extend(cases[n_test + n_val :])

    return np.array(train_list), np.array(val_list), np.array(test_list)


class _AugDataset(Dataset):
    """
    Training-only dataset with on-the-fly augmentation.

    Two augmentations applied independently per sample:
    1. Amplitude scaling — multiply all channels by Uniform(0.8, 1.2).
       Simulates different excitation gain settings without altering inter-channel ratios.
    2. Gaussian noise — add per-sensor noise with σ = 5 % of that sensor's RMS.
       Since C1 is normalised to RMS ≈ 1, this injects ~0.05 g of noise on C1 and
       proportionally less on the weaker channels (C3, C6), preserving relative SNR.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x   # (N, 1, T, S)
        self.y = y   # (N, L)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        x = self.x[idx].clone()   # (1, T, S)

        # 1. Amplitude scaling
        x = x * (0.8 + 0.4 * torch.rand(1))

        # 2. Per-sensor Gaussian noise (5 % of each sensor's RMS)
        sensor_rms = x.pow(2).mean(dim=1, keepdim=True).sqrt()   # (1, 1, S)
        x = x + torch.randn_like(x) * (0.05 * sensor_rms)

        return x, self.y[idx]


def get_tower_dataloaders(
    excitation_types: list[str],
    *,
    root: str | Path,
    num_workers: int = 0,
    train_batch_size: int = 256,
    eval_batch_size: int = 32,
    seed: int = 42,
    include_healthy: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from the tower dataset.

    Batches are ``(x, y)`` compatible with all ``main_*.py`` entry points:

        x: (B, 1, T=400, S=6)   float32 — add-channel input tensor
        y: (B, L=4)              float32 — normalised labels ∈ [−1, 1]

    Splitting is done at the **recording level** (one .npz = one physical test
    run) to prevent leakage from overlapping 2-second windows.  Within each
    damage category (healthy / single / multi) the split guarantees at least one
    recording in every partition, so the test set always contains all three
    damage types regardless of how few single-damage recordings exist.

    Args:
        excitation_types: Excitation signals to include, e.g. ``["EQ", "WN"]``,
                          ``["WN"]``, or ``["EQ", "WN", "sine"]``.
        root:             Path to tower dataset root directory.
        include_healthy:  Include healthy (Y = 0) samples (default ``True``).
    """
    root = Path(root)
    X, Y, case_ids = _load_arrays(root, excitation_types, include_healthy)

    # (N, T, S) → (N, 1, T, S);  Y ∈ [0,1] → y_norm ∈ [−1,1]
    x_t = torch.from_numpy(X).float().unsqueeze(1)
    y_t = torch.from_numpy(Y * 2.0 - 1.0).float()

    # Build mapping from case id → representative label (first segment)
    unique_cases = np.unique(case_ids)
    case_to_y = {c: Y[case_ids == c][0] for c in unique_cases}

    train_cases, val_cases, test_cases = _stratified_case_split(
        unique_cases, case_to_y, seed
    )

    train_idx = np.where(np.isin(case_ids, train_cases))[0]
    val_idx   = np.where(np.isin(case_ids, val_cases))[0]
    test_idx  = np.where(np.isin(case_ids, test_cases))[0]

    def _dl(idx, batch_size, shuffle, augment=False):
        ds = _AugDataset(x_t[idx], y_t[idx]) if augment else TensorDataset(x_t[idx], y_t[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers)

    # Report category breakdown per split
    def _cat_counts(cases):
        cats = [_damage_category(case_to_y[c]) for c in cases]
        return {k: cats.count(k) for k in ("healthy", "single", "multi") if k in cats}

    print(
        f"[tower] {len(unique_cases)} recordings → "
        f"train={len(train_cases)} {_cat_counts(train_cases)} ({len(train_idx)} segs), "
        f"val={len(val_cases)} {_cat_counts(val_cases)} ({len(val_idx)} segs), "
        f"test={len(test_cases)} {_cat_counts(test_cases)} ({len(test_idx)} segs)"
    )
    return _dl(train_idx, train_batch_size, shuffle=True,  augment=True), \
           _dl(val_idx,   eval_batch_size,  shuffle=False), \
           _dl(test_idx,  eval_batch_size,  shuffle=False)
