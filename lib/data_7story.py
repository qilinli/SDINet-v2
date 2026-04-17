from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt
import torch
from safetensors import safe_open
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from lib.preprocessing import normalize_rms


# Sensor indices (0-indexed, out of 65) that match the 9 sensors used in the
# real physical benchmark .mat file.  Used by the "7story-sparse" dataset config
# to train models on the same sensor subset as the benchmark.
SPARSE_7STORY_SENSOR_INDICES: list[int] = [4, 7, 9, 11, 17, 47, 50, 53, 56]


class SevenStoryDataset(Dataset):
    def __init__(self, root: Path | str, getters: Sequence[Callable]) -> None:
        super().__init__()
        self.root = Path(root)
        self.files = list(self.root.glob("*.safetensors"))
        self.getters = list(getters)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        file = self.files[index]
        data = safe_open(file, framework="numpy")
        return [getter(data) for getter in self.getters]


def input_preprocess(
    data,
    n_sensors: int = 65,
    sensor_dim: int = 1,
    sensor_indices: list[int] | None = None,
) -> npt.NDArray[np.float32]:
    # Raw shape: (1000, S, 3) — first 500 samples are real signal,
    # last 500 are zero-padded.  Truncate, don't decimate.
    accel = data.get_tensor("acc").reshape(1000, n_sensors, 3).transpose(2, 0, 1)
    accel = accel[:sensor_dim, :500].astype(np.float32)  # (sensor_dim, 500, n_sensors)
    # Per-window global RMS normalization (no single ref — robust to sensor faults)
    accel = normalize_rms(accel, ref_channel=None)
    if sensor_indices is not None:
        accel = accel[:, :, sensor_indices]   # (sensor_dim, 500, len(sensor_indices))
    return accel


def target_preprocess(data) -> npt.NDArray[np.float32]:
    return data.get_tensor("target").astype(np.float32) / 0.15 - 1


def get_7story_dataloaders(
    subset_names: list[str],
    *,
    root: str | Path = "data/7-story-frame/safetensors/unc=0",
    num_workers: int = 8,
    train_batch_size: int = 128,
    eval_batch_size: int = 32,
    seed: int = 42,
    sensor_indices: list[int] | None = None,
    p_hard: float = 0.0,
    p_struct_mask: float = 0.0,
    p_soft: float = 0.0,
    **_,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from multiple subsets combined.

    Each subset is split independently (70 / 15 / 15) with the same seed
    before concatenation, so the splits are reproducible and there is no
    leakage between train and eval across subsets.

    Training set is always wrapped in ``_AugDataset`` which applies:
      - Signal augmentation (amplitude scaling + per-sensor noise) — always on.
      - Fault augmentation (random/structured/soft) — opt-in via flags.

    When any fault flag is > 0 the DataLoader returns (x, y, y_fault) triplets;
    otherwise (x, y) pairs.
    """
    from torch.utils.data import ConcatDataset

    root = Path(root)
    train_parts, val_parts, test_parts = [], [], []
    n_sensors = len(sensor_indices) if sensor_indices is not None else 65

    for name in subset_names:
        ds_root = root / name
        ds = SevenStoryDataset(
            ds_root, [lambda x, _si=sensor_indices: input_preprocess(x, sensor_indices=_si),
                      target_preprocess]
        )

        indices = np.arange(len(ds))
        train_idx, valtest_idx = train_test_split(indices, test_size=0.3, random_state=seed)
        val_idx, test_idx = train_test_split(valtest_idx, test_size=0.5, random_state=seed)

        train_parts.append(Subset(ds, train_idx))
        val_parts.append(Subset(ds, val_idx))
        test_parts.append(Subset(ds, test_idx))

    train_base = _AugDataset(
        ConcatDataset(train_parts),
        n_sensors=n_sensors,
        struct_masks=build_struct_masks_7story(),
        p_hard=p_hard,
        p_struct=p_struct_mask,
        p_soft=p_soft,
    )

    train_dl = DataLoader(
        train_base, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        ConcatDataset(val_parts), batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    test_dl = DataLoader(
        ConcatDataset(test_parts), batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    return train_dl, val_dl, test_dl


def get_7story_single_dataloaders(
    subset_name: str,
    *,
    root: str | Path = "data/7-story-frame/safetensors/unc=0",
    num_workers: int = 8,
    train_batch_size: int = 128,
    eval_batch_size: int = 32,
    seed: int = 42,
    sensor_indices: list[int] | None = None,
    **_,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(root)
    ds_root = root / subset_name
    ds = SevenStoryDataset(
        ds_root,
        [lambda x, _si=sensor_indices: input_preprocess(x, sensor_indices=_si),
         target_preprocess],
    )

    train, valtest = train_test_split(
        np.arange(len(ds)), test_size=0.3, random_state=seed
    )
    val, test = train_test_split(valtest, test_size=0.5, random_state=seed)

    train_dl = DataLoader(
        Subset(ds, train),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        Subset(ds, val),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_dl = DataLoader(
        Subset(ds, test),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Structural affinity (used by C-head with --use-structural-bias)
# ---------------------------------------------------------------------------

def build_structural_affinity_7story() -> torch.Tensor:
    """
    (L+1, S) = (71, 65) binary structural affinity for the 7-story frame.

    Row 0  = null / no-object slot — all-ones (unbiased, same as Qatar).
    Row l+1 has exactly 2 non-zero entries: the two sensors directly adjacent
    to damage location l in the structural graph.

    Structure (1-indexed sensors and locations):
      Left column  : sensors 1–22 (bottom→top).
                     Floor-k node = sensor 1+3k  (k=1..7 → 4,7,10,13,16,19,22)
      Right column : sensors 44–65 (top→bottom).
                     Floor-k node = sensor 65−3k (k=1..7 → 62,59,56,53,50,47,44)
      Beam k       : sensors {20+3k, 21+3k, 22+3k} (k=1..7)
                     beam 1 → {23,24,25},  beam 7 → {41,42,43}

      Left col  locations  1–21 : loc l between sensors l and l+1
      Beam locations       22–49: 4 per beam, beam k base = 22+4(k-1)
        base+0: left-col node (1+3k)   ↔ beam sensor (20+3k)
        base+1: beam sensor   (20+3k)  ↔ beam sensor (21+3k)
        base+2: beam sensor   (21+3k)  ↔ beam sensor (22+3k)
        base+3: beam sensor   (22+3k)  ↔ right-col node (65−3k)
      Right col locations 50–70 : loc 50+j between sensors 44+j and 45+j  (j=0..20)
    """
    A = torch.zeros(71, 65)
    A[0] = 1.0  # null slot: attend to all sensors equally

    def _adj(loc_1: int, s1_1: int, s2_1: int) -> None:
        A[loc_1, s1_1 - 1] = 1.0
        A[loc_1, s2_1 - 1] = 1.0

    # Left column (locations 1–21)
    for l in range(1, 22):
        _adj(l, l, l + 1)

    # Beams (locations 22–49, 4 per beam)
    for k in range(1, 8):
        s_lc  = 1 + 3 * k
        s_b1  = 20 + 3 * k
        s_b2  = 21 + 3 * k
        s_b3  = 22 + 3 * k
        s_rc  = 65 - 3 * k
        base  = 22 + 4 * (k - 1)
        _adj(base,     s_lc, s_b1)
        _adj(base + 1, s_b1, s_b2)
        _adj(base + 2, s_b2, s_b3)
        _adj(base + 3, s_b3, s_rc)

    # Right column (locations 50–70)
    for j in range(21):
        _adj(50 + j, 44 + j, 45 + j)

    return A


# ---------------------------------------------------------------------------
# Structured mask groups (used for structured fault augmentation)
# ---------------------------------------------------------------------------

def build_struct_masks_7story() -> list[list[int]]:
    """21 structured mask groups for the 7-story frame (0-indexed sensor indices).

    7 beam groups (3 sensors each on same horizontal member)
    + 7 left-column story segments (consecutive sensors along vertical riser)
    + 7 right-column story segments (same, opposite column).

    Parallels Qatar's row_sensors / col_sensors structured masking.
    """
    masks: list[list[int]] = []
    # Beams: beam k has sensors {20+3k, 21+3k, 22+3k} (1-indexed)
    for k in range(1, 8):
        masks.append([19 + 3 * k, 20 + 3 * k, 21 + 3 * k])
    # Left column (sensors 1–22, 1-indexed = 0–21, 0-indexed)
    # Story segments of 3 consecutive sensors; story 7 includes top node (sensor 22 = idx 21)
    for s in range(1, 7):
        masks.append([3 * (s - 1), 3 * (s - 1) + 1, 3 * (s - 1) + 2])
    masks.append([18, 19, 20, 21])  # story 7 + top node
    # Right column (sensors 44–65, 1-indexed = 43–64, 0-indexed)
    # Story 7 includes top node (sensor 65 = idx 64)
    for s in range(1, 7):
        masks.append([43 + 3 * (s - 1), 43 + 3 * (s - 1) + 1, 43 + 3 * (s - 1) + 2])
    masks.append([61, 62, 63, 64])  # story 7 + top node
    return masks


# ---------------------------------------------------------------------------
# Training augmentation dataset wrapper
# ---------------------------------------------------------------------------

class _AugDataset(Dataset):
    """Wraps a lazy (x, y) dataset with on-the-fly training augmentation.

    Always applies signal augmentation (amplitude scaling + per-sensor noise)
    via :func:`~lib.faults.apply_signal_aug`.

    Optionally applies fault augmentation (random/structured/soft) via
    :func:`~lib.faults.apply_fault_aug` when any fault flag is > 0.
    Returns (x, y, y_fault) triplets when faults are active, (x, y) otherwise.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_sensors: int,
        struct_masks: list[list[int]],
        p_hard: float = 0.0,
        p_struct: float = 0.0,
        p_soft: float = 0.0,
    ) -> None:
        self._dataset = dataset
        self._n_sensors = n_sensors
        self._use_fault = p_hard > 0 or p_struct > 0 or p_soft > 0
        self._fault_kwargs = dict(
            n_sensors=n_sensors, struct_masks=struct_masks,
            p_hard=p_hard, p_struct=p_struct, p_soft=p_soft,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        from lib.faults import apply_fault_aug, apply_signal_aug

        x, y = self._dataset[index]
        x = torch.as_tensor(x).float().clone()
        y = torch.as_tensor(y).float()

        x = apply_signal_aug(x)

        if self._use_fault:
            y_fault = apply_fault_aug(x, **self._fault_kwargs)
            return x, y, y_fault
        return x, y


# ---------------------------------------------------------------------------
# Real .mat benchmark (7-story frame physical experiment)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]

DAMAGE_PHYSICAL_SCALE: float = 0.15


@dataclass(frozen=True)
class RealMatBenchmarkSpec:
    mat_filename: str = "Testing_SingleEAcc9Sensor0.5sec.mat"
    time_len: int = 500   # raw data zero-padded to this length if shorter


DEFAULT_BENCHMARK    = RealMatBenchmarkSpec()
TWO_DAMAGE_BENCHMARK = RealMatBenchmarkSpec(
    mat_filename="Testing_TwoEAcc9Sensor0.45sec.mat",
    time_len=500,   # T=450 in file — zero-padded to 500 on load
)


def default_benchmark_mat_path(spec: RealMatBenchmarkSpec = DEFAULT_BENCHMARK) -> Path:
    return _REPO_ROOT / "data" / "7-story-frame" / spec.mat_filename


@lru_cache(maxsize=4)
def _load_benchmark_tensors_cached(mat_path_str: str) -> tuple[torch.Tensor, torch.Tensor]:
    from scipy.io import loadmat
    mat = loadmat(mat_path_str)
    test_data   = torch.from_numpy(mat["Testing_Data"]).float()
    test_target = torch.from_numpy(mat["Testing_label"]).float()
    return test_data, test_target


def load_real_test_tensors(
    mat_path: str | Path | None = None,
    *,
    spec: RealMatBenchmarkSpec = DEFAULT_BENCHMARK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load ``Testing_Data`` / ``Testing_label`` from the benchmark ``.mat`` file (cached).

    Zero-pads the time axis to ``spec.time_len`` if the raw data is shorter.
    """
    path = Path(mat_path) if mat_path is not None else default_benchmark_mat_path(spec)
    test_data, test_target = _load_benchmark_tensors_cached(str(path.resolve()))

    T = test_data.size(0)
    if T < spec.time_len:
        pad = torch.zeros(spec.time_len - T, test_data.size(1))
        test_data = torch.cat([test_data, pad], dim=0)
    elif T > spec.time_len:
        test_data = test_data[: spec.time_len]

    return test_data, test_target


def _print_eval_results(results: dict[str, float]) -> None:
    print(
        f"map_mse={results['map_mse']:.4e}  "
        f"top_k_recall={results['top_k_recall']:.3f}  "
        f"AP={results['ap']:.3f}  "
        f"F1={results['f1']:.3f}  "
        f"severity_mae={results['severity_mae']:.4e}  "
        f"mean_k_pred={results['mean_k_pred']:.1f} (true={results['mean_k_true']:.1f})"
    )
