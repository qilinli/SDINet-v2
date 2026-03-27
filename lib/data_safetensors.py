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


def add_noise(signal: npt.NDArray[np.float32], snr: float) -> npt.NDArray[np.float32]:
    linear_snr = 10.0 ** (snr / 10.0)
    noise_var = signal.var(1)[:, None, :] / linear_snr
    noise_var[np.abs(noise_var) < 1e-12] = linear_snr  # for near-null signals
    e = np.random.randn(*signal.shape) * (noise_var * 2.0) ** 0.5 / 2
    return signal + e


class SafetensorsDataset(Dataset):
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
    snr: float = -1.0,
) -> npt.NDArray[np.float32]:
    accel = data.get_tensor("acc").reshape(1000, n_sensors, 3).transpose(2, 0, 1)
    accel = accel[:sensor_dim, :500]
    if snr > 0.0:
        accel = add_noise(accel, snr)
    return accel.astype(np.float32)  # type: ignore


def val_input_preprocess(
    data,
    n_sensors: int = 65,
    sensor_dim: int = 1,
    snr: float = -1.0,
) -> npt.NDArray[np.float32]:
    accel = data.get_tensor("acc").reshape(1000, n_sensors, 3).transpose(2, 0, 1)
    accel = accel[:sensor_dim, :500]
    if snr > 0.0:
        accel = add_noise(accel, snr)
    return accel.astype(np.float32)


def target_preprocess(data) -> npt.NDArray[np.float32]:
    return data.get_tensor("target").astype(np.float32) / 0.15 - 1


def get_combined_dataloaders(
    subset_names: list[str],
    snr: float = 0.0,
    *,
    root: str | Path = "data/7-story-frame/safetensors/unc=0",
    num_workers: int = 0,
    train_batch_size: int = 128,
    eval_batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from multiple subsets combined.

    Each subset is split independently (70 / 15 / 15) with the same seed
    before concatenation, so the splits are reproducible and there is no
    leakage between train and eval across subsets.
    """
    from torch.utils.data import ConcatDataset

    root = Path(root)
    train_parts, val_parts, test_parts = [], [], []

    for name in subset_names:
        ds_root = root / name
        train_ds = SafetensorsDataset(
            ds_root, [lambda x: input_preprocess(x, snr=snr), target_preprocess]
        )
        val_ds = SafetensorsDataset(
            ds_root, [lambda x: val_input_preprocess(x, snr=snr), target_preprocess]
        )

        indices = np.arange(len(train_ds))
        train_idx, valtest_idx = train_test_split(indices, test_size=0.3, random_state=seed)
        val_idx, test_idx = train_test_split(valtest_idx, test_size=0.5, random_state=seed)

        train_parts.append(Subset(train_ds, train_idx))
        val_parts.append(Subset(val_ds, val_idx))
        test_parts.append(Subset(val_ds, test_idx))

    train_dl = DataLoader(
        ConcatDataset(train_parts), batch_size=train_batch_size, shuffle=True, num_workers=num_workers
    )
    val_dl = DataLoader(
        ConcatDataset(val_parts), batch_size=eval_batch_size, shuffle=False, num_workers=num_workers
    )
    test_dl = DataLoader(
        ConcatDataset(test_parts), batch_size=eval_batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dl, val_dl, test_dl


def get_dataloaders(
    subset_name: str,
    snr: float = 0.0,
    *,
    root: str | Path = "data/7-story-frame/safetensors/unc=0",
    num_workers: int = 12,
    train_batch_size: int = 128,
    eval_batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(root)
    ds_root = root / subset_name
    train_ds = SafetensorsDataset(
        ds_root,
        [lambda x: input_preprocess(x, snr=snr), target_preprocess],
    )
    val_ds = SafetensorsDataset(
        ds_root,
        [lambda x: val_input_preprocess(x, snr=snr), target_preprocess],
    )

    train, valtest = train_test_split(
        np.arange(len(train_ds)), test_size=0.3, random_state=seed
    )
    val, test = train_test_split(valtest, test_size=0.5, random_state=seed)

    train_dl = DataLoader(
        Subset(train_ds, train),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dl = DataLoader(
        Subset(val_ds, val),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_dl = DataLoader(
        Subset(val_ds, test),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dl, val_dl, test_dl


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
