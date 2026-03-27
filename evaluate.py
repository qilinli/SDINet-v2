"""
evaluate.py — full model comparison table.

Evaluates v1, C, and/or DR checkpoints on the test set(s) for a given dataset
and (for 7-story) on the real-world MATLAB benchmarks.  Prints a side-by-side
metric table.

Usage::

    python evaluate.py --dataset 7story --v1 states/7story/v1-combined-<uuid>.pt
    python evaluate.py --dataset qatar  --c  states/qatar/c-combined-<uuid>.pt --dr states/qatar/dr-combined-<uuid>.pt
    python evaluate.py --dataset tower  --v1 states/tower/v1-combined-<uuid>.pt

Optional flags::

    --root        override dataset root path
    --snr         SNR for noise injection (7-story only; default -1.0 = off)
    --seed        dataloader split seed (default 42)
    --window-size Qatar window size in samples (default 2048)
    --overlap     Qatar window overlap fraction (default 0.5)
    --downsample  Qatar decimation factor (default 4)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lib.calibration import (
    do_real_test_c_calibrated,
    do_real_test_dr_calibrated,
    do_real_test_v1_calibrated,
    eval_on_loader_c_calibrated,
    eval_on_loader_dr_calibrated,
    eval_on_loader_v1_calibrated,
    load_calibration,
)
from lib.datasets import DATASETS, get_dataset
from lib.data_safetensors import DEFAULT_BENCHMARK, TWO_DAMAGE_BENCHMARK
from lib.model import (
    load_model_from_checkpoint,
    load_model_c_from_checkpoint,
    load_model_dr_from_checkpoint,
)

METRICS = [
    "map_mse",
    "top_k_recall",
    "ap",
    "f1",
    "precision",
    "recall",
    "severity_mae",
    "mean_k_pred",
    "mean_k_true",
]

_EVAL_FN  = {"v1": eval_on_loader_v1_calibrated,
             "c":  eval_on_loader_c_calibrated,
             "dr": eval_on_loader_dr_calibrated}
_REAL_FN  = {"v1": do_real_test_v1_calibrated,
             "c":  do_real_test_c_calibrated,
             "dr": do_real_test_dr_calibrated}
_LOAD_FN  = {"v1": load_model_from_checkpoint,
             "c":  load_model_c_from_checkpoint,
             "dr": load_model_dr_from_checkpoint}
_CAL_LABEL = {
    "v1": lambda cal: (
        f"v1 (T={cal['temperature']:.2f}, α={cal['ratio_alpha']:.2f}, β={cal['ratio_beta']:.2f})"
        if "ratio_alpha" in cal else "v1 (uncalibrated)"
    ),
    "c": lambda cal: f"C (obj_θ={cal.get('obj_threshold', 0.5):.3f})",
    "dr": lambda cal: (
        f"DR (α={cal['ratio_alpha']:.2f}, β={cal['ratio_beta']:.2f})"
        if "ratio_alpha" in cal else f"DR (θ={cal.get('threshold', 0.5):.2f})"
    ),
}


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def run_evaluation(
    v1_ckpt: str | Path | None,
    c_ckpt:  str | Path | None = None,
    dr_ckpt: str | Path | None = None,
    *,
    dataset_name: str = "7story",
    root: str | None = None,
    snr: float = -1.0,
    seed: int = 42,
    window_size: int = 2048,
    overlap: float = 0.5,
    downsample: int = 4,
    tower_excitation: tuple[str, ...] = ("EQ", "WN", "sine"),
) -> dict[str, dict[str, float]]:
    """
    Evaluate any combination of checkpoints and return a nested metric dict.

    Result keys follow the pattern ``<MODEL>/<split>`` e.g. ``v1/single``,
    ``C/double``, ``DR/tower``, ``v1/real-1dmg``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(dataset_name)
    actual_root = root if root is not None else dataset.default_root

    dl_kwargs: dict = dict(
        root=actual_root,
        num_workers=0,
        eval_batch_size=32,
        seed=seed,
        # dataset-specific (ignored by non-matching loaders via **_)
        snr=snr,
        tower_excitation=tower_excitation,
        window_size=window_size,
        overlap=overlap,
        downsample=downsample,
        train_batch_size=32,   # unused for test loaders
    )

    results: dict[str, dict[str, float]] = {}

    ckpts = {"v1": v1_ckpt, "c": c_ckpt, "dr": dr_ckpt}
    for model_name, ckpt in ckpts.items():
        if ckpt is None:
            continue

        model = _LOAD_FN[model_name](ckpt, device=device)
        cal   = load_calibration(ckpt)
        print(f"[calibration] {_CAL_LABEL[model_name](cal)}")

        eval_fn = _EVAL_FN[model_name]
        tag = model_name.upper()

        for split_label, test_dl in dataset.get_test_loaders(**dl_kwargs):
            results[f"{tag}/{split_label}"] = eval_fn(model, test_dl, device, **cal)

        # Extra test loader (Qatar double-damage)
        extra_dl = dataset.get_extra_test_loader(**dl_kwargs)
        if extra_dl is not None:
            results[f"{tag}/{dataset_name}_double"] = eval_fn(model, extra_dl, device, **cal)

        # Real benchmark (7-story only)
        if dataset.supports_real_benchmark:
            real_fn = _REAL_FN[model_name]
            results[f"{tag}/real-1dmg"] = real_fn(model, device=device, print_result=False,
                                                   spec=DEFAULT_BENCHMARK, **cal)
            results[f"{tag}/real-2dmg"] = real_fn(model, device=device, print_result=False,
                                                   spec=TWO_DAMAGE_BENCHMARK, **cal)

    return results


def print_table(results: dict[str, dict[str, float]]) -> None:
    cols  = list(results.keys())
    col_w = 14
    print(f"{'metric':<16}" + "".join(f"{c:>{col_w}}" for c in cols))
    print("-" * (16 + col_w * len(cols)))
    for m in METRICS:
        row = f"{m:<16}"
        for c in cols:
            row += f"{results[c].get(m, float('nan')):>{col_w}.4f}"
        print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate v1, C, DR checkpoints side-by-side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="7story", choices=list(DATASETS),
                        help="Dataset the checkpoints were trained on")
    parser.add_argument("--v1", default=None, metavar="CKPT",
                        help="v1 (Midn) checkpoint path")
    parser.add_argument("--c",  default=None, metavar="CKPT",
                        help="Approach-C (MidnC) checkpoint path")
    parser.add_argument("--dr", default=None, metavar="CKPT",
                        help="Approach-DR (MidnDR) checkpoint path")
    parser.add_argument("--root", default=None,
                        help="Override dataset root path")
    parser.add_argument("--snr",         type=float, default=-1.0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--window-size", type=int,   default=2048)
    parser.add_argument("--overlap",     type=float, default=0.5)
    parser.add_argument("--downsample",  type=int,   default=4)
    parser.add_argument("--tower-excitation", nargs="+", default=["EQ", "WN", "sine"])
    args = parser.parse_args()

    if args.v1 is None and args.c is None and args.dr is None:
        parser.error("At least one of --v1, --c, --dr is required.")

    results = run_evaluation(
        args.v1, args.c, args.dr,
        dataset_name=args.dataset,
        root=args.root,
        snr=args.snr,
        seed=args.seed,
        window_size=args.window_size,
        overlap=args.overlap,
        downsample=args.downsample,
        tower_excitation=tuple(args.tower_excitation),
    )
    print_table(results)


if __name__ == "__main__":
    main()
