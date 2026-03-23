"""
Real-world MATLAB benchmark helpers.

Loads input / label tensors from ``data/Testing_SingleEAcc9Sensor0.5sec.mat``
and runs a forward pass for either model head (v1 or Approach-B), returning
the standard ``evaluate_all`` / ``evaluate_all_v1`` metric dict.

CLI::

    python -m lib.testing --v1  states/single-damage-<uuid>.pt
    python -m lib.testing --b   states/multi-damage-B-<uuid>.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from scipy.io import loadmat

_REPO_ROOT = Path(__file__).resolve().parents[1]

DAMAGE_PHYSICAL_SCALE: float = 0.15


# ---------------------------------------------------------------------------
# Benchmark .mat file spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RealMatBenchmarkSpec:
    mat_filename: str = "Testing_SingleEAcc9Sensor0.5sec.mat"


DEFAULT_BENCHMARK = RealMatBenchmarkSpec()


def default_benchmark_mat_path(spec: RealMatBenchmarkSpec = DEFAULT_BENCHMARK) -> Path:
    return _REPO_ROOT / "data" / spec.mat_filename


@lru_cache(maxsize=4)
def _load_benchmark_tensors_cached(mat_path_str: str) -> tuple[torch.Tensor, torch.Tensor]:
    mat = loadmat(mat_path_str)
    test_data   = torch.from_numpy(mat["Testing_Data"]).float()
    test_target = torch.from_numpy(mat["Testing_label"]).float()
    return test_data, test_target


def load_real_test_tensors(
    mat_path: str | Path | None = None,
    *,
    spec: RealMatBenchmarkSpec = DEFAULT_BENCHMARK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ``Testing_Data`` / ``Testing_label`` from the benchmark ``.mat`` file (cached)."""
    path = Path(mat_path) if mat_path is not None else default_benchmark_mat_path(spec)
    return _load_benchmark_tensors_cached(str(path.resolve()))


# ---------------------------------------------------------------------------
# Checkpoint loaders
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    model_cfg=None,
) -> torch.nn.Module:
    """Load a v1 (Midn) checkpoint and return a ready-to-run model."""
    from lib.model import ModelConfig, build_model

    if model_cfg is None:
        model_cfg = ModelConfig()

    model = build_model(model_cfg)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state)

    if device is not None:
        model = model.to(device)
    return model


def load_model_b_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    model_cfg=None,
) -> torch.nn.Module:
    """Load an Approach-B (MidnB) checkpoint and return a ready-to-run model."""
    from lib.model import ModelConfigB, build_model

    if model_cfg is None:
        model_cfg = ModelConfigB()

    model = build_model(model_cfg)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state)

    if device is not None:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Real benchmark forward pass
# ---------------------------------------------------------------------------

def do_real_test(
    model: torch.nn.Module,
    *,
    device: str | torch.device | None = None,
    mat_path: str | Path | None = None,
    spec: RealMatBenchmarkSpec = DEFAULT_BENCHMARK,
    print_result: bool = True,
) -> dict[str, float]:
    """Forward pass + ``evaluate_all_v1`` metrics on the real .mat benchmark."""
    from lib.metrics import evaluate_all_v1

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    test_data, test_target = load_real_test_tensors(mat_path, spec=spec)
    x = test_data[None, None, ...].to(device)

    with torch.inference_mode():
        model = model.to(device)
        model.eval()
        dmg_pred, loc_pred = model(x)

    y_norm = (test_target.squeeze().to(device) / DAMAGE_PHYSICAL_SCALE - 1.0).unsqueeze(0)
    result = evaluate_all_v1(dmg_pred, loc_pred, y_norm)

    if print_result:
        _print_eval_results(result)

    return result


def do_real_test_b(
    model: torch.nn.Module,
    *,
    device: str | torch.device | None = None,
    mat_path: str | Path | None = None,
    spec: RealMatBenchmarkSpec = DEFAULT_BENCHMARK,
    print_result: bool = True,
) -> dict[str, float]:
    """Forward pass + ``evaluate_all`` metrics on the real .mat benchmark."""
    from lib.metrics import evaluate_all

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    test_data, test_target = load_real_test_tensors(mat_path, spec=spec)
    x = test_data[None, None, ...].to(device)

    with torch.inference_mode():
        model = model.to(device)
        model.eval()
        presence_logits, severity = model(x)

    y_norm = (test_target.squeeze().to(device) / DAMAGE_PHYSICAL_SCALE - 1.0).unsqueeze(0)
    result = evaluate_all(presence_logits, severity, y_norm)

    if print_result:
        _print_eval_results(result)

    return result


def _print_eval_results(results: dict[str, float]) -> None:
    print(
        f"map_mse={results['map_mse']:.4e}  "
        f"top_k_recall={results['top_k_recall']:.3f}  "
        f"AP={results['ap']:.3f}  "
        f"F1={results['f1']:.3f}  "
        f"severity_mae={results['severity_mae']:.4e}  "
        f"mean_k_pred={results['mean_k_pred']:.1f} (true={results['mean_k_true']:.1f})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SDINet real benchmark from a checkpoint.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--v1", metavar="CKPT", help="v1 (Midn) checkpoint path")
    group.add_argument("--b",  metavar="CKPT", help="Approach-B (MidnB) checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.v1:
        model = load_model_from_checkpoint(args.v1, device=device)
        do_real_test(model, device=device)
    else:
        model = load_model_b_from_checkpoint(args.b, device=device)
        do_real_test_b(model, device=device)


if __name__ == "__main__":
    main()
