"""
lib/data_qatar.py — DataLoaders for the Qatar SHM Benchmark dataset.

Pipeline overview
-----------------
NPZ cache (raw signal)  →  windowing  →  normalization  →  tensor  →  augmentation  →  batch

Stage 1 — build cache (run once):
    python data/Qatar/build_dataset.py
    Saves one NPZ per recording: X (262144, 30) float32 + damaged_joints metadata.
    No windowing or normalization — all those decisions live here.

Stage 2 — this file (called at training time):
    For each recording:
        (262144, 30)  →  window  →  (255, 2048, 30)   [default: 2s windows, 50% overlap]
                      →  normalize                     [divide by RMS of channel 0]
                      →  label replicate               [same (30,) label for every window]
    Concatenate across recordings → X (N, T, S),  Y (N, 30) binary {0,1}
    Convert to tensors, add channel dim: x (N, 1, T, S),  y_norm (N, 30) ∈ {-1,+1}
    Split at recording level (all windows of a recording stay together).
    Training DataLoader wraps x in _AugDataset for on-the-fly augmentation.

Splits:
    train       —  Dataset A (all 31 recordings: 1 healthy + 30 single-damage)
    val         —  Dataset B, first 50% of windows per recording (independent run)
    test        —  Dataset B, last 50% of windows per recording (1-window gap at boundary)
    double test —  Double Damage (5 recordings, 2 joints damaged each)

Why this split?
    Recording-level splits from A leave only ~4 unseen val scenarios → val_top_k_recall=0
    throughout training, making val useless as a convergence signal and poisoning
    calibration.  Using Dataset B (independent excitation run) for val/test gives a
    meaningful val signal (all 30 scenarios present) while keeping the evaluation
    independent from training data.

Label convention (shared with lib/data_safetensors.py):
    Y ∈ {0, 1}^30  →  y_norm = 2Y − 1 ∈ {−1, +1}
    Undamaged location → −1,  Damaged location → +1

Sensor grid (0-indexed, matching physical QUGS layout):
    Row r (0–5, front→back):  joints  5r … 5r+4
    Col c (0–4, right→left):  joints  c, c+5, c+10, c+15, c+20, c+25

Dataset constants:
    QATAR_FS          = 1024  Hz
    QATAR_N_SENSORS   = 30
    QATAR_N_LOCATIONS = 30
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from scipy.signal import decimate as _scipy_decimate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QATAR_DEFAULT_ROOT: str = "data/Qatar/processed"

QATAR_FS: int          = 1024
QATAR_N_SENSORS: int   = 30
QATAR_N_LOCATIONS: int = 30
QATAR_GRID_ROWS: int   = 6   # rows front→back
QATAR_GRID_COLS: int   = 5   # columns right→left


# ---------------------------------------------------------------------------
# Sensor grid helpers  (used for structured masking and fault evaluation)
# ---------------------------------------------------------------------------

def row_sensors(row: int) -> list[int]:
    """0-indexed sensor indices for a physical grid row (0=front, 5=back)."""
    if not 0 <= row < QATAR_GRID_ROWS:
        raise ValueError(f"row must be 0–{QATAR_GRID_ROWS - 1}, got {row}")
    base = row * QATAR_GRID_COLS
    return list(range(base, base + QATAR_GRID_COLS))


def col_sensors(col: int) -> list[int]:
    """0-indexed sensor indices for a physical grid column (0–4)."""
    if not 0 <= col < QATAR_GRID_COLS:
        raise ValueError(f"col must be 0–{QATAR_GRID_COLS - 1}, got {col}")
    return [col + QATAR_GRID_COLS * r for r in range(QATAR_GRID_ROWS)]


def mask_sensors(x: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Zero out sensor channels in x (B, 1, T, S) or (1, T, S). Returns a copy."""
    x = x.clone()
    x[..., indices] = 0.0
    return x


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _window_signal(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Slice a (N_samples, S) recording into overlapping windows.

    With default window_size=2048, stride=1024 (50% overlap):
        262144 samples  →  255 windows of shape (2048, 30)
        Adjacent windows share 1024 samples — they are NOT independent.
        Splitting must therefore happen at the recording level, not window level.

    Returns: (n_windows, window_size, S)
    """
    n_windows = max(0, (len(signal) - window_size) // stride + 1)
    return np.stack(
        [signal[i * stride: i * stride + window_size] for i in range(n_windows)],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(windows: np.ndarray, mode: str | None) -> np.ndarray:
    """
    Per-window normalization applied after windowing.

    "rms"    — divide all channels by the RMS of channel 0 (joint 1, front-row
               reference sensor near the shaker). Preserves cross-channel amplitude
               ratios, which carry damage location information. Channel 0 ends up
               with RMS ≈ 1; other channels scale proportionally.

    "zscore" — subtract mean and divide by std across all channels and time steps.
               Removes DC offset and normalises energy but destroys inter-channel
               amplitude relationships.

    None     — no normalization (use when inspecting raw values).
    """
    if mode is None:
        return windows
    if mode == "rms":
        rms = np.sqrt(np.mean(windows[:, :, 0] ** 2, axis=1, keepdims=True))  # (N,1)
        rms = rms[:, :, np.newaxis]   # (N,1,1) broadcasts over (N,T,S)
        return windows / (rms + 1e-8)
    if mode == "zscore":
        mean = windows.mean(axis=(1, 2), keepdims=True)
        std  = windows.std(axis=(1, 2), keepdims=True)
        return (windows - mean) / (std + 1e-8)
    raise ValueError(f"Unknown normalize mode: {mode!r}. Use 'rms', 'zscore', or None.")


# ---------------------------------------------------------------------------
# NPZ group loader
# ---------------------------------------------------------------------------

def _load_group(
    root: Path,
    group: str,
    window_size: int,
    stride: int,
    normalize: str | None,
    downsample: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all NPZ files in one group, window + normalize, return flat arrays.

    Each recording contributes n_windows rows, all with the same label vector.
    case_ids lets the caller keep track of which rows came from which recording
    (critical for recording-level train/val splitting).

    If downsample > 1, apply a FIR anti-alias filter and decimate the raw
    signal before windowing.  window_size and stride are divided by the same
    factor so the physical window duration and overlap fraction are preserved.

        downsample=4: model input ( 512, 30) at  256 Hz  (default — captures up to 128 Hz)
        downsample=8: model input ( 256, 30) at  128 Hz
        downsample=1: model input (2048, 30) at 1024 Hz  (raw, no decimation)
        downsample=8: model input ( 256, 30) at  128 Hz

    Returns
    -------
    X        : (N, T, S)  float32  — windowed, normalized; T = window_size // downsample
    Y        : (N, L)     float32  — binary {0, 1}; 1 at each damaged joint
    case_ids : (N,)       int      — same integer for all windows of one recording
    """
    group_dir = root / group
    if not group_dir.exists():
        raise FileNotFoundError(
            f"Processed directory not found: {group_dir}\n"
            "Run `python data/Qatar/build_dataset.py` first."
        )

    eff_window = window_size // downsample
    eff_stride = max(1, stride // downsample)

    Xs, Ys, case_ids = [], [], []
    case_idx = 0

    for npz_path in sorted(group_dir.glob("*.npz")):
        f      = np.load(npz_path)
        signal = f["X"]                        # (262144, 30) raw signal
        joints = f["damaged_joints"].tolist()  # e.g. [18] or [3, 26]

        if downsample > 1:
            # FIR filter (linear phase, no distortion for vibration signals)
            # then decimate; zero_phase=True applies filter twice for flatness
            signal = _scipy_decimate(
                signal, downsample, ftype="fir", axis=0, zero_phase=True,
            ).astype(np.float32)

        windows = _window_signal(signal, eff_window, eff_stride)  # (n_w, T, 30)
        if len(windows) == 0:
            continue
        windows = _normalize(windows, normalize)

        # Build label vector: 1.0 at each damaged joint (joints are 1-indexed)
        y = np.zeros(QATAR_N_LOCATIONS, dtype=np.float32)
        for j in joints:
            y[j - 1] = 1.0

        n = len(windows)
        Xs.append(windows)
        Ys.append(np.tile(y, (n, 1)))   # same label replicated for every window
        case_ids.extend([case_idx] * n)
        case_idx += 1

    if not Xs:
        raise ValueError(
            f"No NPZ files found in {group_dir}. "
            "Run `python data/Qatar/build_dataset.py` first."
        )

    return (
        np.concatenate(Xs, axis=0),
        np.concatenate(Ys, axis=0),
        np.array(case_ids),
    )


# ---------------------------------------------------------------------------
# Time-based val / test split within Dataset B
# ---------------------------------------------------------------------------

def _time_split_b(
    case_ids: np.ndarray,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split Dataset B windows by time position within each recording.

    For each recording: first val_frac of windows → val, rest → test.
    One window is dropped at the boundary to prevent overlapping-window bleed
    (adjacent windows share 50% of raw samples by default).

    Windows appear in time order because _load_group iterates sorted NPZ files
    and appends windows sequentially — no shuffling occurs before this split.

    Returns
    -------
    val_idx  : flat index array into the concatenated B array
    test_idx : flat index array into the concatenated B array
    """
    val_list, test_list = [], []
    for c in np.unique(case_ids):
        idxs = np.where(case_ids == c)[0]   # time-ordered indices for this recording
        n = len(idxs)
        split = max(1, int(n * val_frac))
        val_list.append(idxs[:split])
        if split + 1 < n:
            test_list.append(idxs[split + 1:])  # +1 gap eliminates overlap bleed
    return np.concatenate(val_list), np.concatenate(test_list)


# ---------------------------------------------------------------------------
# Augmentation dataset  (training only)
# ---------------------------------------------------------------------------

class _AugDataset(Dataset):
    """
    Wraps training tensors and applies four stochastic augmentations per sample.

    Applied fresh every epoch so the same window looks different each time.
    Val / test use plain TensorDataset — no augmentation.

    Augmentations (applied in order):
    1. Amplitude scaling  ×U(0.8, 1.2)
       Simulates different excitation intensity between recordings.

    2. Gaussian noise  σ = 5% of each sensor's RMS
       Simulates sensor noise and small environmental fluctuations.
       Per-sensor scaling preserves relative noise levels across channels.

    3. Random channel masking  (prob p_random)
       Zero out k ~ U(0, max_k) randomly chosen sensor channels.
       Trains robustness to individual sensor failures.

    4. Structured masking  (prob p_struct)
       Zero out one complete grid row (5 sensors) or column (6 sensors).
       Simulates a wiring harness or DAQ channel group going offline.
    """

    def __init__(
        self,
        x: torch.Tensor,   # (N, 1, T, S)
        y: torch.Tensor,   # (N, L)
        p_random: float = 0.5,
        max_k: int = 10,
        p_struct: float = 0.3,
    ) -> None:
        self.x = x
        self.y = y
        self.p_random = p_random
        self.max_k    = max_k
        self.p_struct = p_struct
        # All possible structured masks: 6 rows + 5 columns
        self._struct_masks: list[list[int]] = (
            [row_sensors(r) for r in range(QATAR_GRID_ROWS)]
            + [col_sensors(c) for c in range(QATAR_GRID_COLS)]
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        x = self.x[idx].clone()   # (1, T, S)

        # 1. Amplitude scaling
        x = x * (0.8 + 0.4 * torch.rand(1))

        # 2. Per-sensor noise
        sensor_rms = x.pow(2).mean(dim=1, keepdim=True).sqrt()   # (1, 1, S)
        x = x + torch.randn_like(x) * (0.05 * sensor_rms)

        # 3. Random channel masking
        if self.max_k > 0 and torch.rand(1).item() < self.p_random:
            k = int(torch.randint(0, self.max_k + 1, (1,)).item())
            if k > 0:
                x[:, :, torch.randperm(QATAR_N_SENSORS)[:k].tolist()] = 0.0

        # 4. Structured masking
        if torch.rand(1).item() < self.p_struct:
            mask_idx = int(torch.randint(0, len(self._struct_masks), (1,)).item())
            x[:, :, self._struct_masks[mask_idx]] = 0.0

        return x, self.y[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_qatar_dataloaders(
    *,
    root: str | Path,
    window_size: int = 2048,
    overlap: float = 0.5,
    downsample: int = 4,
    normalize: str | None = "rms",
    num_workers: int = 0,
    train_batch_size: int = 256,
    eval_batch_size: int = 32,
    seed: int = 42,         # kept for API compatibility; split is deterministic by time
    val_frac: float = 0.5,
    p_random_mask: float = 0.5,
    max_random_mask: int = 10,
    p_struct_mask: float = 0.3,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from the Qatar SHM Benchmark.

    Data sources:
        train  —  Dataset A (all 31 recordings: 1 healthy + 30 single-damage)
        val    —  Dataset B, first val_frac of windows per recording
        test   —  Dataset B, remaining windows per recording (1-window gap at boundary)

    Batch format:
        x : (B, 1, window_size // downsample, 30)   float32
        y : (B, 30)                                  float32  ∈ {−1, +1}

    Args:
        root:            Path to processed/ directory (output of build_dataset.py).
        window_size:     Physical window in original samples at 1024 Hz
                         (1024=1 s, 2048=2 s, 4096=4 s).
        overlap:         Fraction overlap between adjacent windows (0 = non-overlapping).
        downsample:      Decimation factor applied before windowing (default 4).
                         4 → 256 Hz, 512 samples: captures structural modes up to
                         128 Hz at 4× smaller model input than raw 1024 Hz.
                         Must divide window_size evenly; use 1 to disable.
        normalize:       "rms" | "zscore" | None  (see _normalize for details).
        val_frac:        Fraction of Dataset B windows per recording used for val
                         (time-ordered: first val_frac → val, rest → test).
        p_random_mask:   Training augmentation: prob of random channel dropout.
        max_random_mask: Training augmentation: max channels zeroed per sample.
        p_struct_mask:   Training augmentation: prob of row/column structured masking.
    """
    root   = Path(root)
    stride = max(1, int(window_size * (1.0 - overlap)))

    # Load Dataset A (all → train) and Dataset B (time-split → val/test)
    X_a, Y_a, _          = _load_group(root, "dataset_a", window_size, stride, normalize, downsample)
    X_b, Y_b, case_ids_b = _load_group(root, "dataset_b", window_size, stride, normalize, downsample)

    # Convert to tensors; add channel dim; map labels {0,1} → {-1,+1}
    x_a = torch.from_numpy(X_a).float().unsqueeze(1)   # (N, 1, T, 30)
    y_a = torch.from_numpy(Y_a * 2.0 - 1.0).float()
    x_b = torch.from_numpy(X_b).float().unsqueeze(1)
    y_b = torch.from_numpy(Y_b * 2.0 - 1.0).float()

    # Time-based split of Dataset B into val and test
    val_idx, test_idx = _time_split_b(case_ids_b, val_frac)

    eff_fs = QATAR_FS // downsample
    eff_T  = window_size // downsample
    print(
        f"[qatar] train=Dataset A ({len(X_a)} segs) | "
        f"val=Dataset B first {val_frac:.0%} ({len(val_idx)} segs) | "
        f"test=Dataset B last {1.0 - val_frac:.0%} ({len(test_idx)} segs)  "
        f"[window={window_size}→{eff_T} @ {eff_fs} Hz, norm={normalize!r}]"
    )

    def _dl(x, y, batch_size, shuffle, augment=False):
        ds = (
            _AugDataset(x, y, p_random=p_random_mask,
                        max_k=max_random_mask, p_struct=p_struct_mask)
            if augment
            else TensorDataset(x, y)
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers)

    return (
        _dl(x_a,              y_a,              train_batch_size, shuffle=True,  augment=True),
        _dl(x_b[val_idx],     y_b[val_idx],     eval_batch_size,  shuffle=False),
        _dl(x_b[test_idx],    y_b[test_idx],    eval_batch_size,  shuffle=False),
    )


def get_qatar_double_test_dataloader(
    *,
    root: str | Path,
    window_size: int = 2048,
    overlap: float = 0.5,
    downsample: int = 4,
    normalize: str | None = "rms",
    eval_batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """
    DataLoader for the 5 double-damage recordings.

    Used to evaluate multi-damage generalisation: the model is trained on
    single-damage data only, then tested here without any retraining.
    Each sample has exactly 2 entries of +1 in y (two damaged joints).

    Batches: x (B, 1, window_size // downsample, 30),  y (B, 30) ∈ {−1, +1}.
    """
    root   = Path(root)
    stride = max(1, int(window_size * (1.0 - overlap)))

    X, Y, _ = _load_group(root, "double_damage", window_size, stride, normalize, downsample)
    x_t = torch.from_numpy(X).float().unsqueeze(1)
    y_t = torch.from_numpy(Y * 2.0 - 1.0).float()

    eff_fs = QATAR_FS // downsample
    eff_T  = window_size // downsample
    print(
        f"[qatar-double] {len(x_t)} windows from 5 double-damage recordings  "
        f"[window={window_size}→{eff_T} @ {eff_fs} Hz, norm={normalize!r}]"
    )

    return DataLoader(TensorDataset(x_t, y_t), batch_size=eval_batch_size,
                      shuffle=False, num_workers=num_workers)
