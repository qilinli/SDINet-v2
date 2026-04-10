"""
lib/data_lumo.py — DataLoaders for the LUMO long-term SHM dataset.

LUMO = *Leibniz University Test Structure for MOnitoring*, a 9 m outdoor
lattice tower instrumented for long-term ambient (wind-excited) structural
health monitoring.  Distinct from 7-story (simulated forced shaker) and
Qatar (lab white-noise): LUMO is the *field* benchmark of the
synthetic → lab → field trio.

Dataset layout (local subset, full benchmark)::

    root/
        01_Healthy/    SHMTS_*.mat  (5 files × 10 min)
        02_DAM6_111/   SHMTS_*.mat
        03_Healthy/
        04_DAM4_111/
        05_Healthy/
        06_DAM3_111/
        07_Healthy/
        08_DAM6_010/
        09_Healthy/
        10_DAM4_010/
        11_Healthy/
        12_DAM3_010/
        readme.pdf, lumo_fem_healthy.inp

Each .mat contains a single struct ``Dat`` with::

    Dat.Data         (990600, 22)  float32  — 10 min @ Fs=1651.61 Hz
    Dat.ChannelNames (22,) str                — 18 accel + 3 strain + 1 temp
    Dat.ChannelUnits (22,) str                — g / m·m⁻¹ / °C
    Dat.Fs           float                    — 1651.6129
    Dat.Time / Dat.Timestamps                 — UTC metadata

Channel layout (columns of ``Dat.Data``)::

    0 .. 17  accelerometers  accel01x, accel01y, …, accel09y  (9 pairs × xy)
    18 .. 20 strain gauges   strain01, strain02, strain03    (base only)
    21       temperature     temp01                           (base only)

Strain channels were verified to have ~10⁻⁶ m/m std in healthy recordings —
effectively zero-variance, unusable as SDI input.  Temperature is an EOV
covariate that the current parity framing does not use.  **We keep only the
18 accel channels.**

Label encoding (parity framing with binary {healthy, damaged} — per the
approved plan, severity 010 vs 111 is collapsed):

    L = 3 damage positions: {DAM3, DAM4, DAM6} mapped to indices {0, 1, 2}.
    y_norm ∈ {-1, +1}^3.  +1 iff *any* bracing at that position is unbolted.

    Healthy:       [-1, -1, -1]  (K = 0)
    *_DAM3_*  :   [+1, -1, -1]  (K = 1)
    *_DAM4_*  :   [-1, +1, -1]  (K = 1)
    *_DAM6_*  :   [-1, -1, +1]  (K = 1)

Dataset constants (passed to model configs)::

    LUMO_FS           = 1651.6129
    LUMO_N_SENSORS    = 18
    LUMO_N_LOCATIONS  = 3
    LUMO_TIME_LEN     = 3304   # window_size // downsample @ defaults

Windowing is done on-the-fly at load time (analogous to Qatar's ``window_size``
/ ``overlap`` kwargs), not pre-extracted .npz files like Tower.

See the approved plan at /home/jovyan/.claude/plans/jolly-tumbling-zephyr.md
and CLAUDE.md for more context on the benchmark role.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, TensorDataset

LUMO_DEFAULT_ROOT: str = "data/LUMO"

LUMO_FS: float       = 1651.6129032258063
LUMO_N_SENSORS: int  = 18
LUMO_N_LOCATIONS: int = 3

# Default window configuration matches train.py's CLI defaults
# (--window-size 2048 --overlap 0.5 --downsample 4) so direct calls to
# get_lumo_dataloaders() produce the same shapes as CLI-driven runs and
# stay within the parity framing.  At 1651.61 Hz and window_size=2048
# that's ≈1.24 s per window; 4× decimation → effective Fs ≈ 413 Hz,
# well above the ≤20 Hz modal band of interest.  Raising --window-size
# to 3304 gives the full 2 s window if desired.
LUMO_DEFAULT_WINDOW_SIZE: int = 2048
LUMO_DEFAULT_OVERLAP: float   = 0.5
LUMO_DEFAULT_DOWNSAMPLE: int  = 4
LUMO_TIME_LEN: int = LUMO_DEFAULT_WINDOW_SIZE // LUMO_DEFAULT_DOWNSAMPLE   # 512

# Fixed DAM → label-index mapping.  Only DAM3, DAM4, DAM6 exist in the
# full LUMO benchmark (DAM1, DAM2, DAM5 are not instrumented scenarios).
_DAM_TO_IDX: dict[str, int] = {"DAM3": 0, "DAM4": 1, "DAM6": 2}


# --------------------------------------------------------------------------- #
# Folder parsing                                                               #
# --------------------------------------------------------------------------- #

def _parse_scenario(folder_name: str) -> tuple[str, np.ndarray]:
    """
    Parse a LUMO scenario folder name into (category, y_binary).

    Folder format: ``<idx>_Healthy`` or ``<idx>_DAM<n>_<pattern>``
    where ``<pattern>`` ∈ {"010", "111", ...} — ignored under binary framing.

    Returns
    -------
    category : str
        "healthy" or e.g. "DAM6" (used for stratified splitting so all
        damage positions are represented in all splits).
    y        : (3,) float32
        Binary label in {-1, +1}, per the plan.
    """
    parts = folder_name.split("_")
    if len(parts) >= 2 and parts[1] == "Healthy":
        return "healthy", np.full(LUMO_N_LOCATIONS, -1.0, dtype=np.float32)

    if len(parts) < 3 or not parts[1].startswith("DAM"):
        raise ValueError(
            f"Unrecognised LUMO folder name {folder_name!r}. "
            f"Expected '<idx>_Healthy' or '<idx>_DAM<n>_<pattern>'."
        )

    dam = parts[1]   # e.g. "DAM6"
    if dam not in _DAM_TO_IDX:
        raise ValueError(
            f"LUMO folder {folder_name!r} references unknown DAM position {dam!r}. "
            f"Expected one of {sorted(_DAM_TO_IDX)}."
        )

    y = np.full(LUMO_N_LOCATIONS, -1.0, dtype=np.float32)
    y[_DAM_TO_IDX[dam]] = 1.0
    return dam, y


# --------------------------------------------------------------------------- #
# .mat loading + windowing                                                     #
# --------------------------------------------------------------------------- #

def _load_accel(mat_path: Path) -> np.ndarray:
    """
    Load the 18 accelerometer channels from one LUMO .mat file.

    Returns
    -------
    accel : (T_total, 18) float32
    """
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data = m["Dat"].Data                  # (T_total, 22) float32
    return np.ascontiguousarray(data[:, :LUMO_N_SENSORS])


def _window(
    signal: np.ndarray,
    window_size: int,
    overlap: float,
    downsample: int,
) -> np.ndarray:
    """
    Slice a raw (T_total, 18) signal into fixed-length, overlapped windows.

    Decimation is applied *after* windowing so window_size refers to raw
    sample indices (matches the Qatar convention).

    Returns
    -------
    windows : (N, T, 18)  float32  where T = window_size // downsample
    """
    assert window_size % downsample == 0, (
        f"window_size ({window_size}) must be divisible by downsample ({downsample})"
    )
    stride = max(1, int(round(window_size * (1.0 - overlap))))
    t_total = signal.shape[0]
    if t_total < window_size:
        return np.empty((0, window_size // downsample, LUMO_N_SENSORS), dtype=np.float32)

    n_windows = 1 + (t_total - window_size) // stride
    t_win = window_size // downsample
    out = np.empty((n_windows, t_win, LUMO_N_SENSORS), dtype=np.float32)
    for i in range(n_windows):
        start = i * stride
        chunk = signal[start : start + window_size]
        if downsample > 1:
            chunk = chunk[::downsample]
        out[i] = chunk[:t_win]
    return out


# --------------------------------------------------------------------------- #
# Dataset scan                                                                 #
# --------------------------------------------------------------------------- #

def _scan_recordings(
    root: Path,
    window_size: int,
    overlap: float,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Walk all LUMO scenario folders, load + window every .mat file, and
    attach a recording-level case id (one id per .mat file) for splitting.

    Returns
    -------
    X         : (N, T, 18)  float32
    Y         : (N, L=3)    float32  — already in {-1, +1}
    case_ids  : (N,)        int64    — unique id per .mat file
    case_meta : list[str]             — one entry per unique case id: folder/file
                (length == number of unique case ids)
    """
    if not root.exists():
        raise FileNotFoundError(f"LUMO root does not exist: {root}")

    Xs: list[np.ndarray] = []
    Ys: list[np.ndarray] = []
    case_ids: list[int] = []
    case_meta: list[str] = []
    case_idx = 0

    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        try:
            _, y_vec = _parse_scenario(folder.name)
        except ValueError:
            # Skip non-scenario folders (e.g. .ipynb_checkpoints).
            continue

        for mat_path in sorted(folder.glob("SHMTS_*.mat")):
            accel = _load_accel(mat_path)
            windows = _window(accel, window_size, overlap, downsample)
            if windows.shape[0] == 0:
                continue
            Xs.append(windows)
            Ys.append(np.broadcast_to(y_vec, (windows.shape[0], LUMO_N_LOCATIONS)).copy())
            case_ids.extend([case_idx] * windows.shape[0])
            case_meta.append(f"{folder.name}/{mat_path.name}")
            case_idx += 1

    if not Xs:
        raise ValueError(
            f"No LUMO .mat files found under {root}. "
            "Expected folders like '01_Healthy', '02_DAM6_111', ..."
        )

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    cids = np.asarray(case_ids, dtype=np.int64)
    return X, Y, cids, case_meta


# --------------------------------------------------------------------------- #
# Recording-level stratified split                                             #
# --------------------------------------------------------------------------- #

def _damage_category(y_vec: np.ndarray) -> str:
    """Return 'healthy' or e.g. 'dam0', 'dam1', 'dam2' for splitting."""
    pos = np.where(y_vec > 0)[0]
    if pos.size == 0:
        return "healthy"
    return f"dam{int(pos[0])}"


def _stratified_case_split(
    unique_cases: np.ndarray,
    case_to_y: dict[int, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split recordings (one .mat = one case) into train / val / test so that
    every damage category (healthy + each DAM position) is represented in
    every partition.  Same logic as ``lib/data_tower.py``.
    """
    from collections import defaultdict

    rng = np.random.RandomState(seed)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for c in unique_cases:
        by_cat[_damage_category(case_to_y[c])].append(int(c))

    train_list: list[int] = []
    val_list:   list[int] = []
    test_list:  list[int] = []

    for cat in sorted(by_cat):
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


# --------------------------------------------------------------------------- #
# Structured fault masks                                                       #
# --------------------------------------------------------------------------- #

def build_struct_masks_lumo() -> list[list[int]]:
    """11 structured mask groups for LUMO's 18 sensors (0-indexed).

    9 level groups (2 sensors each: X + Y at same floor)
    + 2 axis groups (9 sensors each: all X-axis or all Y-axis).
    """
    masks: list[list[int]] = [[2 * k, 2 * k + 1] for k in range(9)]
    masks.append(list(range(0, 18, 2)))   # X-axis: 0, 2, 4, ..., 16
    masks.append(list(range(1, 18, 2)))   # Y-axis: 1, 3, 5, ..., 17
    return masks


# --------------------------------------------------------------------------- #
# Augmented training dataset                                                   #
# --------------------------------------------------------------------------- #

_struct_masks_lumo = build_struct_masks_lumo()  # precompute once


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def get_lumo_dataloaders(
    *,
    root: str | Path = LUMO_DEFAULT_ROOT,
    window_size: int = LUMO_DEFAULT_WINDOW_SIZE,
    overlap: float = LUMO_DEFAULT_OVERLAP,
    downsample: int = LUMO_DEFAULT_DOWNSAMPLE,
    num_workers: int = 0,
    train_batch_size: int = 256,
    eval_batch_size: int = 32,
    seed: int = 42,
    p_hard: float = 0.0,
    p_struct_mask: float = 0.0,
    p_soft: float = 0.0,
    **_,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for the LUMO field SHM benchmark.

    Batches are ``(x, y)`` compatible with all SDINet-v2 training entry points:

        x: (B, 1, T, 18)   float32 — T = window_size // downsample
        y: (B, 3)          float32 — normalised labels ∈ {-1, +1}^3

    Splitting is done at the **recording level** (one .mat file = one case)
    with stratification across {healthy, DAM3, DAM4, DAM6} to guarantee each
    damage position appears in every split.

    Args:
        root:         Path to LUMO dataset root directory.
        window_size:  Raw samples per window (default 3304 ≈ 2 s @ 1651.61 Hz).
        overlap:      Fraction overlap ∈ [0, 1) between consecutive windows.
        downsample:   Decimation factor applied within each window (default 4
                      → effective Fs ≈ 413 Hz, well above the ~20 Hz modal band).
        num_workers:  DataLoader workers.
        train_batch_size / eval_batch_size: batch sizes.
        seed:         Split seed.
    """
    root = Path(root)
    X, Y, case_ids, case_meta = _scan_recordings(root, window_size, overlap, downsample)

    # (N, T, S) → (N, 1, T, S);  Y already in {-1, +1}
    x_t = torch.from_numpy(X).float().unsqueeze(1)
    y_t = torch.from_numpy(Y).float()

    unique_cases = np.unique(case_ids)
    case_to_y = {int(c): Y[case_ids == c][0] for c in unique_cases}

    train_cases, val_cases, test_cases = _stratified_case_split(
        unique_cases, case_to_y, seed
    )

    train_idx = np.where(np.isin(case_ids, train_cases))[0]
    val_idx   = np.where(np.isin(case_ids, val_cases))[0]
    test_idx  = np.where(np.isin(case_ids, test_cases))[0]

    def _dl(idx, batch_size, shuffle, augment=False):
        from lib.faults import AugDataset
        ds = (AugDataset(x_t[idx], y_t[idx], LUMO_N_SENSORS, _struct_masks_lumo,
                         p_hard=p_hard, p_struct=p_struct_mask, p_soft=p_soft)
              if augment else TensorDataset(x_t[idx], y_t[idx]))
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    def _cat_counts(cases):
        cats = [_damage_category(case_to_y[int(c)]) for c in cases]
        return {k: cats.count(k) for k in sorted(set(cats))}

    t_win = window_size // downsample
    print(
        f"[lumo] {len(unique_cases)} recordings, {X.shape[0]} windows "
        f"(T={t_win}, S={LUMO_N_SENSORS}, L={LUMO_N_LOCATIONS}) → "
        f"train={len(train_cases)} {_cat_counts(train_cases)} ({len(train_idx)} wins), "
        f"val={len(val_cases)} {_cat_counts(val_cases)} ({len(val_idx)} wins), "
        f"test={len(test_cases)} {_cat_counts(test_cases)} ({len(test_idx)} wins)"
    )

    return (
        _dl(train_idx, train_batch_size, shuffle=True,  augment=True),
        _dl(val_idx,   eval_batch_size,  shuffle=False),
        _dl(test_idx,  eval_batch_size,  shuffle=False),
    )
