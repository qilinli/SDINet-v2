"""
lib/datasets.py — dataset registry for SDINet-v2.

Encapsulates per-dataset metadata (shape, label type) and dataloader
factories so that train.py and evaluate.py contain no dataset-specific
if/elif logic beyond the one block that maps CLI flags to loader kwargs.

To add a new dataset:
  1. Write a ``_loader_<name>`` wrapper function below.
  2. Add a ``DatasetConfig(...)`` entry to ``DATASETS``.
  3. Add one ``elif dataset.name == "<name>":`` block in train.py's
     ``_build_dl_kwargs`` (3 lines).  Nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from lib.data_safetensors import (
    get_combined_dataloaders,
    get_dataloaders as _get_split_loader,
)
from lib.data_tower import (
    TOWER_DEFAULT_ROOT,
    TOWER_N_LOCATIONS,
    TOWER_N_SENSORS,
    TOWER_TIME_LEN,
    get_tower_dataloaders,
)
from lib.data_qatar import (
    QATAR_DEFAULT_ROOT,
    QATAR_N_LOCATIONS,
    QATAR_N_SENSORS,
    get_qatar_dataloaders,
    get_qatar_double_test_dataloader,
)


@dataclass
class DatasetConfig:
    """
    Dataset metadata + dataloader factories.

    Separates dataset concerns (input shape, label semantics, loader API)
    from model and training concerns.  The ``_model_cfg_overrides`` and
    ``_training_overrides`` dicts encode dataset-mandated defaults that
    differ per model — e.g. Qatar binary labels require ``sev_weight=0.0``
    for the C-head and ``pos_weight=L-1`` for the DR-head.  Explicit CLI
    flags in train.py always take priority over these registry defaults.
    """

    name: str
    n_sensors: int
    n_locations: int
    time_len: int | None      # None → computed at runtime (Qatar: window_size//downsample)
    label_type: Literal["continuous", "binary"]
    supports_real_benchmark: bool   # 7-story .mat file only
    default_root: str

    # Dataloader factories — use the public methods, not these directly
    _loader_fn: Callable
    _extra_test_fn: Callable | None = None   # Qatar double-damage only

    # Dataset × model integration overrides.
    # Keys are model names ("v1", "c", "dr").  CLI flags override these.
    _model_cfg_overrides: dict[str, dict] = field(default_factory=dict)
    _training_overrides:  dict[str, dict] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def model_config_overrides(self, model: str) -> dict:
        """Extra kwargs to inject into ModelConfig* construction."""
        return self._model_cfg_overrides.get(model, {})

    def training_overrides(self, model: str) -> dict:
        """Extra kwargs for do_training_* (e.g. pos_weight for DR on binary data)."""
        return self._training_overrides.get(model, {})

    def get_dataloaders(self, **kwargs):
        """Return ``(train_dl, val_dl, test_dl)``."""
        return self._loader_fn(**kwargs)

    def get_calibration_val_loaders(self, **kwargs) -> list:
        """
        Return the val DataLoader(s) used for post-training calibration.

        7-story pools single + double val sets (both were seen during training).
        All other datasets return a single-element list.
        """
        if self.name == "7story":
            return _7story_cal_loaders(**kwargs)
        _, val_dl, _ = self.get_dataloaders(**kwargs)
        return [val_dl]

    def get_test_loaders(self, **kwargs) -> list[tuple[str, object]]:
        """
        Return ``[(label, test_dl), ...]`` for final evaluation reporting.

        7-story returns separate single and double test loaders.
        Other datasets return a single ``[(dataset_name, test_dl)]`` pair.
        """
        if self.name == "7story":
            test_single, test_double = _7story_test_loaders(**kwargs)
            return [("single", test_single), ("double", test_double)]
        _, _, test_dl = self.get_dataloaders(**kwargs)
        return [(self.name, test_dl)]

    def get_extra_test_loader(self, **kwargs):
        """Return the double-damage test loader (Qatar only), else None."""
        if self._extra_test_fn is None:
            return None
        return self._extra_test_fn(**kwargs)


# --------------------------------------------------------------------------- #
# Wrapper functions                                                             #
# Translate the uniform dl_kwargs dict into each data module's call signature. #
# --------------------------------------------------------------------------- #

def _loader_7story(root, snr, num_workers, train_batch_size, eval_batch_size, seed, **_):
    return get_combined_dataloaders(
        ["single", "double"], snr,
        root=root,
        num_workers=num_workers,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
    )


def _7story_cal_loaders(root, snr, num_workers, eval_batch_size, seed, **_) -> list:
    """Separate single + double val loaders for 7-story calibration."""
    _, val_s, _ = _get_split_loader("single", snr, root=root, num_workers=num_workers,
                                    eval_batch_size=eval_batch_size, seed=seed)
    _, val_d, _ = _get_split_loader("double", snr, root=root, num_workers=num_workers,
                                    eval_batch_size=eval_batch_size, seed=seed)
    return [val_s, val_d]


def _7story_test_loaders(root, snr, num_workers, eval_batch_size, seed, **_):
    """Separate single + double test loaders for 7-story evaluation."""
    _, _, test_s = _get_split_loader("single", snr, root=root, num_workers=num_workers,
                                     eval_batch_size=eval_batch_size, seed=seed)
    _, _, test_d = _get_split_loader("double", snr, root=root, num_workers=num_workers,
                                     eval_batch_size=eval_batch_size, seed=seed)
    return test_s, test_d


def _loader_tower(root, tower_excitation, num_workers, train_batch_size, eval_batch_size, seed, **_):
    return get_tower_dataloaders(
        list(tower_excitation),
        root=root,
        num_workers=num_workers,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
    )


def _loader_qatar(root, window_size, overlap, downsample,
                  num_workers, train_batch_size, eval_batch_size, seed, **_):
    return get_qatar_dataloaders(
        root=root,
        window_size=window_size, overlap=overlap, downsample=downsample,
        num_workers=num_workers,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
    )


def _extra_test_qatar(root, window_size, overlap, downsample,
                      num_workers, eval_batch_size, **_):
    return get_qatar_double_test_dataloader(
        root=root,
        window_size=window_size, overlap=overlap, downsample=downsample,
        num_workers=num_workers,
        eval_batch_size=eval_batch_size,
    )


# --------------------------------------------------------------------------- #
# Dataset registry                                                              #
# --------------------------------------------------------------------------- #

DATASETS: dict[str, DatasetConfig] = {
    "7story": DatasetConfig(
        name="7story",
        n_sensors=65,
        n_locations=70,
        time_len=500,
        label_type="continuous",
        supports_real_benchmark=True,
        default_root="data/7-story-frame/safetensors/unc=0",
        _loader_fn=_loader_7story,
    ),
    "tower": DatasetConfig(
        name="tower",
        n_sensors=TOWER_N_SENSORS,
        n_locations=TOWER_N_LOCATIONS,
        time_len=TOWER_TIME_LEN,
        label_type="continuous",
        supports_real_benchmark=False,
        default_root=TOWER_DEFAULT_ROOT,
        _loader_fn=_loader_tower,
    ),
    "qatar": DatasetConfig(
        name="qatar",
        n_sensors=QATAR_N_SENSORS,
        n_locations=QATAR_N_LOCATIONS,
        time_len=None,          # computed: window_size // downsample
        label_type="binary",
        supports_real_benchmark=False,
        default_root=QATAR_DEFAULT_ROOT,
        _loader_fn=_loader_qatar,
        _extra_test_fn=_extra_test_qatar,
        # Qatar binary labels {-1,+1}: severity is always 1 for damaged slots,
        # so sev_loss is trivially learned and decouples val_loss from val_mse.
        # Force sev_weight=0 for the C-head (location CE only).
        _model_cfg_overrides={"c": {"sev_weight": 0.0}},
        # DR uses MSE on Qatar (pos_weight=None) — BCE with high pos_weight collapses
        # the MIL model (sigmoid-then-BCE is numerically unstable at this scale).
        _training_overrides={},
    ),
}


def get_dataset(name: str) -> DatasetConfig:
    """Return the DatasetConfig for *name*. Raises ValueError for unknown names."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}")
    return DATASETS[name]
