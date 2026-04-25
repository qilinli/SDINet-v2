"""
lib/data_asce.py — DataLoaders for the ASCE 4-story braced-frame hammer benchmark.

Dataset layout (``data/asce_hammer/``)::

    asce_hammer/
        k0/  scenario_000001.mat … scenario_001000.mat   (1 000  undamaged)
        k1/  scenario_001001.mat … scenario_011000.mat   (10 000 K=1)
        k2/  scenario_011001.mat … scenario_021000.mat   (10 000 K=2)
        k3/  scenario_021001.mat … scenario_031000.mat   (10 000 K=3)
        k4/  scenario_031001.mat … scenario_041000.mat   (10 000 K=4)
        k5/  scenario_041001.mat … scenario_051000.mat   (10 000 K=5)
        dataset_index.mat
        README.md

Each .mat (MATLAB v7.3 / HDF5) contains:
    acc                   (16, 5001)  — 16 accelerometer channels @ 1000 Hz, 5.0 s
                                        (half-sine impact at roof + free decay)
    stiffness_reduction   (32, 1)     — ground truth ∈ [0, 1] per brace
    impact_force          scalar      — peak force amplitude (N), ~N(3000, 2%)

Windowing: we keep only the first **2 s** of each scenario (first 2001 raw
samples) — covers the impulse + ~2 fundamental-damping decay constants
(T₁_decay ≈ 1.67 s at 9.52 Hz, 1% damping) and matches the existing SDINet
pipeline's 2 s / 250 Hz convention.

Preprocessing (CLAUDE.md convention):
    1. Window to first 2 s (2001 samples)
    2. FIR anti-alias decimate 4× (1000 → 250 Hz, 2001 → 501, truncate to 500)
    3. Optional global RMS normalisation applied at **load time** (so
       ``--norm-method`` controls it — same regime as 7-story: v1/B default
       ``none``, C-variants pass ``mean`` explicitly; Insight #30).

Label mapping (stiffness_reduction ∈ [0, 1]):
    y_norm = 2·stiffness_reduction − 1  ∈  [−1, +1]
    Undamaged (0)  → −1,   complete break (1.0) → +1.

Split: scenario-level random stratified by k ∈ {0..5}, seeded.  Each class is
split 70/15/15 into train/val/test independently.

Dataset constants:
    ASCE_FS          = 1000 Hz  (raw)
    ASCE_N_SENSORS   = 16
    ASCE_N_LOCATIONS = 32
    ASCE_TIME_LEN    = 500      (after windowing to 2 s and 4× decimation)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import decimate as _scipy_decimate

from lib.preprocessing import normalize_rms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASCE_DEFAULT_ROOT: str = "data/asce_hammer"

ASCE_FS: int          = 1000
ASCE_DOWNSAMPLE: int  = 4           # → 250 Hz
ASCE_RAW_WINDOW: int  = 2001        # first 2.0 s of 5.0 s signal (kept, then decimated)
ASCE_TIME_LEN: int    = 500         # decimated length (truncated from 501)
ASCE_N_SENSORS: int   = 16
ASCE_N_LOCATIONS: int = 32
ASCE_K_MAX: int       = 5
ASCE_N_FLOORS: int    = 4

# Per-class scenario counts (from README)
_CLASS_COUNTS = {0: 1000, 1: 10000, 2: 10000, 3: 10000, 4: 10000, 5: 10000}

_CACHE_NAME = f"asce_T{ASCE_TIME_LEN}.npz"


# ---------------------------------------------------------------------------
# Structured mask groups (for --p-struct-mask fault augmentation)
# ---------------------------------------------------------------------------

def build_struct_masks_asce() -> list[list[int]]:
    """6 groups for ASCE's 16-sensor layout: 4 floors × 4 sensors + 2 axis groups.

    Channel order per README (1-indexed → 0-indexed here):
        Floor 1: ch 0,1,2,3
        Floor 2: ch 4,5,6,7
        Floor 3: ch 8,9,10,11
        Floor 4: ch 12,13,14,15
    Within each floor the channels alternate x,y,x,y.

    Models correlated failures (whole-floor wiring harness, x-axis DAQ card).
    """
    floors = [[4 * f + i for i in range(4)] for f in range(ASCE_N_FLOORS)]
    x_axis = [0, 2, 4, 6, 8, 10, 12, 14]
    y_axis = [1, 3, 5, 7, 9, 11, 13, 15]
    return floors + [x_axis, y_axis]


_struct_masks_asce = build_struct_masks_asce()


def build_structural_affinity_asce() -> torch.Tensor:
    """(L+1, S) = (33, 16) binary location-sensor affinity for the brace dataset.

    Each brace spans two adjacent floors; sensors on either of those floors
    receive 1.  Ground floor has no sensors so story-1 braces couple only to
    Floor-1 sensors.  Row 32 (null class) is all zeros.

    Channel groups (0-indexed):
        Floor 1 → ch 0–3
        Floor 2 → ch 4–7
        Floor 3 → ch 8–11
        Floor 4 → ch 12–15
    """
    R = torch.zeros(ASCE_N_LOCATIONS + 1, ASCE_N_SENSORS)
    R[0:8,    0:4]   = 1.0   # story 1  (ground↔F1): F1 only
    R[8:16,   0:8]   = 1.0   # story 2  (F1↔F2)
    R[16:24,  4:12]  = 1.0   # story 3  (F2↔F3)
    R[24:32,  8:16]  = 1.0   # story 4  (F3↔F4)
    return R


def build_structural_affinity_asce_columns() -> torch.Tensor:
    """(L+1, S) = (37, 16) affinity for the 36-column ASCE-hammer-columns dataset.

    Each story contains 9 columns (3×3 plan grid) spanning two adjacent floors.
    Coupling rules mirror the brace affinity (same rigid-floor sensor groups),
    only the "n_per_story" constant grows from 8 → 9.
    """
    N_LOC = 36
    R = torch.zeros(N_LOC + 1, ASCE_N_SENSORS)
    R[0:9,    0:4]   = 1.0   # story 1 columns (ground↔F1)
    R[9:18,   0:8]   = 1.0   # story 2 columns (F1↔F2)
    R[18:27,  4:12]  = 1.0   # story 3 columns (F2↔F3)
    R[27:36,  8:16]  = 1.0   # story 4 columns (F3↔F4)
    return R


# ---------------------------------------------------------------------------
# Cache build (one-time, ~5–10 min)
# ---------------------------------------------------------------------------

def _build_cache(root: Path, cache_path: Path, n_locations: int = ASCE_N_LOCATIONS) -> None:
    """Read all 51 000 .mat files, window + decimate, save one NPZ.

    Cached signal is **pre-normalisation**; RMS is applied at load time so
    ``--norm-method`` can toggle it.

    ``n_locations`` is the target-vector width in the .mat files; 32 for the
    brace dataset, 36 for the columns dataset. The cache shape follows it.
    """
    import h5py

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    N = sum(_CLASS_COUNTS.values())
    X = np.empty((N, ASCE_TIME_LEN, ASCE_N_SENSORS), dtype=np.float32)
    Y = np.empty((N, n_locations),                    dtype=np.float32)
    K = np.empty((N,),                                dtype=np.int8)

    i = 0
    for k in range(6):
        kdir = root / f"k{k}"
        mats = sorted(kdir.glob("scenario_*.mat"))
        print(f"[asce] reading k{k}: {len(mats)} files")
        for p in mats:
            with h5py.File(p, "r") as f:
                acc = np.array(f["acc"], dtype=np.float32)             # (16, 5001) in v7.3 layout
                sr  = np.array(f["stiffness_reduction"]).ravel().astype(np.float32)  # (32,)
            # h5py returns MATLAB columns as rows; acc is (16, 5001) — transpose to (5001, 16)
            acc = acc.T
            # Keep first 2 s (2001 samples), then 4× FIR anti-alias decimation → 501; truncate to 500
            acc = acc[:ASCE_RAW_WINDOW]
            dec = _scipy_decimate(acc, ASCE_DOWNSAMPLE, ftype="fir", axis=0, zero_phase=True)
            dec = dec[:ASCE_TIME_LEN].astype(np.float32)
            X[i] = dec
            Y[i] = sr
            K[i] = k
            i += 1

    assert i == N, f"expected {N} scenarios, got {i}"
    print(f"[asce] saving cache → {cache_path}  ({X.nbytes / 1e9:.2f} GB)")
    np.savez(cache_path, X=X, Y=Y, K=K)


def _load_cache(root: Path, n_locations: int = ASCE_N_LOCATIONS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = root / "processed" / _CACHE_NAME
    if not cache_path.exists():
        print(f"[asce] cache not found at {cache_path}; building (one-time, ~5–10 min)")
        _build_cache(root, cache_path, n_locations=n_locations)
    f = np.load(cache_path)
    return f["X"], f["Y"], f["K"]


# ---------------------------------------------------------------------------
# Stratified scenario-level split
# ---------------------------------------------------------------------------

def _stratified_split(
    K: np.ndarray, seed: int, val_frac: float = 0.15, test_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-k stratified split so every damage class appears in every partition."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for k in np.unique(K):
        idx = np.where(K == k)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val  = int(round(n * val_frac))
        test_idx.append(idx[:n_test])
        val_idx.append(idx[n_test : n_test + n_val])
        train_idx.append(idx[n_test + n_val:])
    return (np.concatenate(train_idx),
            np.concatenate(val_idx),
            np.concatenate(test_idx))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_asce_dataloaders(
    *,
    root: str | Path,
    num_workers: int = 0,
    train_batch_size: int = 128,
    eval_batch_size: int = 128,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    p_hard: float = 0.0,
    p_struct_mask: float = 0.0,
    p_soft: float = 0.0,
    norm_method: str = "none",
    n_locations: int = ASCE_N_LOCATIONS,
    **_,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return train / val / test DataLoaders for the ASCE hammer benchmark.

    Batch shapes:
        x: (B, 1, 500, 16)   float32
        y: (B, 32)           float32 ∈ [−1, +1]  (2·reduction − 1)

    Args:
        norm_method: {"none", "mean"}.  "mean" = global RMS over all sensors
            and time (matches 7-story default for C-variants).  "none" skips
            the divisor (matches v1/B original-paper preprocessing).
    """
    root = Path(root)
    X, Y, K = _load_cache(root, n_locations=n_locations)

    # Apply load-time normalisation so --norm-method controls it.
    X = normalize_rms(X, ref_channel=None, method=norm_method)

    x_t = torch.from_numpy(X).unsqueeze(1).float()           # (N, 1, T, S)
    y_t = torch.from_numpy(Y * 2.0 - 1.0).float()            # [0,1] → [−1,+1]

    train_idx, val_idx, test_idx = _stratified_split(K, seed, val_frac, test_frac)

    def _cat_counts(idx):
        return {int(k): int((K[idx] == k).sum()) for k in range(6)}

    print(
        f"[asce] {len(X)} scenarios → "
        f"train={len(train_idx)} {_cat_counts(train_idx)}, "
        f"val={len(val_idx)} {_cat_counts(val_idx)}, "
        f"test={len(test_idx)} {_cat_counts(test_idx)}  "
        f"[T={ASCE_TIME_LEN} @ {ASCE_FS // ASCE_DOWNSAMPLE} Hz, norm={norm_method}]"
    )

    def _dl(idx, batch_size, shuffle, augment=False):
        from lib.faults import AugDataset
        ds = (AugDataset(x_t[idx], y_t[idx], ASCE_N_SENSORS, _struct_masks_asce,
                         p_hard=p_hard, p_struct=p_struct_mask, p_soft=p_soft)
              if augment else TensorDataset(x_t[idx], y_t[idx]))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)

    return (
        _dl(train_idx, train_batch_size, shuffle=True,  augment=True),
        _dl(val_idx,   eval_batch_size,  shuffle=False),
        _dl(test_idx,  eval_batch_size,  shuffle=False),
    )
