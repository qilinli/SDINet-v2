from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt
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


def get_dataloaders(
    subset_name: str,
    snr: float = 0.0,
    *,
    root: str | Path = "data/safetensors/unc=0",
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

