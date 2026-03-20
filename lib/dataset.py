from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.io import loadmat
from torch.utils.data import Dataset


class MatlabDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        getters: list[
            Callable[[dict[str, npt.NDArray[np.float_]]], npt.NDArray[np.float_]]
        ],
        cache_files: bool = False
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.mat"))
        self.getters = getters

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[npt.NDArray[np.float_], ...]:
        file = self.files[index]
        data = loadmat(file)
        return tuple(getter(data) for getter in self.getters)


def frame_accel_1d(data: dict[str, npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
    accel = data["acc"].T[np.newaxis, ...]  # 1 x 1000 x 195
    return accel


def frame_accel_3d(data: dict[str, npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
    accel = data["acc"].reshape(3, 65, 1000).transpose(0, 2, 1)  # 3 x 1000 x 65
    return accel


def frame_damage(data: dict[str, npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
    return data["Dmg_Lev"].flatten()


def frame_integrity(data: dict[str, npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
    return 1.0 - data["Dmg_Lev"].flatten()


def tower_accel_1d(data: dict[str, npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
    accel = data["acc"][np.newaxis, ...]  # 1 x 2001 x 216
    return accel


def tower_accel_2d(data: dict[str, npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
    accel = data["acc"].reshape(2001, 2, 108).transpose(1, 0, 2)  # 2 x 2001 x 108
    return accel


def tower_damage(data: dict[str, npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
    return 1.0 - data["theta"].flatten()


def tower_integrity(data: dict[str, npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
    return data["theta"].flatten()


def subsample(
    element: npt.NDArray[np.float_], out_size: int, return_idxs: bool = False
) -> npt.NDArray[np.float_] | tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]:
    keep_idxs = np.random.choice(element.shape[-1], out_size, replace=False)
    element = element[..., keep_idxs]
    return (element, keep_idxs) if return_idxs else element
