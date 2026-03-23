"""
Evaluation helpers for the **single real-world MATLAB benchmark** shipped with this repo.

What this module does
---------------------
- Loads input / label tensors from ``data/Testing_SingleEAcc9Sensor0.5sec.mat``.
- Runs ``do_real_test(model, ...)`` forward pass + notebook-style metrics.

What this module also provides
------------------------------
- Checkpoint helpers: ``load_model_from_checkpoint`` and
  ``do_real_test_from_checkpoint``.
- A direct CLI entrypoint: ``python -m lib.testing <checkpoint_path>``.

Example::

    python -m lib.testing states/single-damage-sparse-<uuid>.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Damage scaling (must match ``lib.data_safetensors.target_preprocess``)
#
#   normalized = physical / DAMAGE_PHYSICAL_SCALE - 1
#   inverse:     physical = (normalized + 1) * DAMAGE_PHYSICAL_SCALE
# ---------------------------------------------------------------------------
DAMAGE_PHYSICAL_SCALE: float = 0.15
# Approach-B severity target: y_sev = (y_norm + 1) / 2 = raw / 0.30
# Inverse: raw_physical = severity * SEVERITY_PHYSICAL_SCALE
SEVERITY_PHYSICAL_SCALE: float = 2.0 * DAMAGE_PHYSICAL_SCALE  # 0.30


def normalized_damage_to_physical(normalized: torch.Tensor) -> torch.Tensor:
    """Map model damage output (normalized) back to physical damage units."""
    return (normalized + 1.0) * DAMAGE_PHYSICAL_SCALE


def severity_to_physical(severity: torch.Tensor) -> torch.Tensor:
    """Map Approach-B severity output (sigmoid ∈ [0,1]) to physical damage units."""
    return severity * SEVERITY_PHYSICAL_SCALE


# ---------------------------------------------------------------------------
# Benchmark file + notebook-aligned scalar metrics (fixed test setup)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RealMatBenchmarkSpec:
    """Constants tied to ``Testing_SingleEAcc9Sensor0.5sec.mat`` / notebook."""

    mat_filename: str = "Testing_SingleEAcc9Sensor0.5sec.mat"
    """File under ``<repo>/data/``."""

    reference_scalar_damage: float = 0.125
    """Scalar damage used for ``rw_mse`` in the original notebook."""

    location_class_for_nll: int = 11
    """Location class index for the reported NLL line in the notebook."""


DEFAULT_BENCHMARK = RealMatBenchmarkSpec()


def default_benchmark_mat_path(spec: RealMatBenchmarkSpec = DEFAULT_BENCHMARK) -> Path:
    return _REPO_ROOT / "data" / spec.mat_filename


@lru_cache(maxsize=4)
def _load_benchmark_tensors_cached(mat_path_str: str) -> tuple[torch.Tensor, torch.Tensor]:
    mat = loadmat(mat_path_str)
    test_data = torch.from_numpy(mat["Testing_Data"]).float()
    test_target = torch.from_numpy(mat["Testing_label"]).float()
    return test_data, test_target


def load_real_test_tensors(
    mat_path: str | Path | None = None,
    *,
    spec: RealMatBenchmarkSpec = DEFAULT_BENCHMARK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load ``Testing_Data`` / ``Testing_label`` from the benchmark ``.mat`` file.

    Cached per resolved path. Call this if you need the tensors without running
    the model.
    """
    path = Path(mat_path) if mat_path is not None else default_benchmark_mat_path(spec)
    return _load_benchmark_tensors_cached(str(path.resolve()))


def do_real_test(
    model: torch.nn.Module,
    *,
    device: str | torch.device | None = None,
    mat_path: str | Path | None = None,
    spec: RealMatBenchmarkSpec = DEFAULT_BENCHMARK,
    print_result: bool = True,
) -> dict[str, float | int]:
    """
    Run the benchmark forward pass and return metrics (same keys as the notebook).

    Parameters
    ----------
    model
        Trained network; **you** load weights before calling (this function does not).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    test_data, test_target = load_real_test_tensors(mat_path, spec=spec)
    x = test_data[None, None, ...].to(device)

    with torch.inference_mode():
        model = model.to(device)
        model.eval()
        dmg_pred, loc_pred = model(x)

    dmg_pred_norm = dmg_pred.squeeze()
    dmg_pred_f = normalized_damage_to_physical(dmg_pred_norm).item()

    loc_pred_sq = loc_pred.squeeze()
    loc_nll = -(
        loc_pred_sq.log_softmax(-1)[spec.location_class_for_nll].item()
    )
    max_nll = -(loc_pred_sq.log_softmax(-1).max().item())

    # Soft assignment: distribute predicted damage across all locations by probability.
    smooth = dmg_pred_f * loc_pred_sq.softmax(-1)
    # Hard assignment: place all predicted damage at the single most likely location.
    hard = torch.zeros_like(test_target.squeeze())
    hard[loc_pred_sq.argmax().item()] = dmg_pred_f

    test_target_sq = test_target.squeeze()
    gt_dmg = test_target_sq.max().item()
    gt_loc = test_target_sq.argmax().item()
    rw_mse = (dmg_pred_f - spec.reference_scalar_damage) ** 2
    rw_err_s = ((smooth.cpu() - test_target_sq.cpu()) ** 2).mean().item()
    rw_err_h = ((hard.cpu() - test_target_sq.cpu()) ** 2).mean().item()

    result: dict[str, float | int] = {
        "rw_mse": rw_mse,
        "rw_nll": loc_nll,
        "rw_err_s": rw_err_s,
        "rw_err_h": rw_err_h,
        "dmg_pred": dmg_pred_f,
        "loc_argmax": loc_pred_sq.argmax().item(),
        "gt_dmg": gt_dmg,
        "gt_loc": gt_loc,
        "best_nll": max_nll,
    }

    if print_result:
        print(
            f"Pred {dmg_pred_f:7.04f} @ {result['loc_argmax']:d} | "
            f"GT {gt_dmg:7.04f} @ {gt_loc:d} | "
            f"MSE: {rw_mse:10.04e}, NLL: {loc_nll:10.04e} "
            f"(best: {max_nll:10.04e})"
        )

    return result


def do_real_test_b(
    model: torch.nn.Module,
    *,
    device: str | torch.device | None = None,
    mat_path: str | Path | None = None,
    spec: RealMatBenchmarkSpec = DEFAULT_BENCHMARK,
    print_result: bool = True,
) -> dict[str, float | int]:
    """
    Run the benchmark forward pass for an Approach-B model (MidnB head).

    Approach-B outputs ``(presence_logits, severity)`` both of shape ``(B, L)``
    instead of v1's scalar damage + 70-dim location pair.  The combined damage
    map is ``sigmoid(presence_logits) * severity * SEVERITY_PHYSICAL_SCALE``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    test_data, test_target = load_real_test_tensors(mat_path, spec=spec)
    x = test_data[None, None, ...].to(device)

    with torch.inference_mode():
        model = model.to(device)
        model.eval()
        presence_logits, severity = model(x)

    # (1, L) → (L,)
    presence_logits_sq = presence_logits.squeeze()
    severity_sq = severity.squeeze()

    presence_prob = torch.sigmoid(presence_logits_sq)   # (L,)  ∈ [0, 1]
    # damage map in physical units: presence_prob * severity * 0.30
    damage_map = presence_prob * severity_to_physical(severity_sq)  # (L,)

    test_target_sq = test_target.squeeze()
    gt_dmg = test_target_sq.max().item()
    gt_loc = test_target_sq.argmax().item()

    loc_argmax = damage_map.argmax().item()
    dmg_pred_f = damage_map[loc_argmax].item()
    dmg_at_gt = damage_map[gt_loc].item()

    # Soft assignment MSE: damage map vs physical target
    rw_err_s = ((damage_map.cpu() - test_target_sq.cpu()) ** 2).mean().item()

    # Hard assignment: concentrate all predicted damage at argmax location
    hard = torch.zeros_like(test_target_sq)
    hard[loc_argmax] = dmg_pred_f
    rw_err_h = ((hard.cpu() - test_target_sq.cpu()) ** 2).mean().item()

    rw_mse = (dmg_at_gt - spec.reference_scalar_damage) ** 2

    result: dict[str, float | int] = {
        "rw_mse": rw_mse,
        "rw_err_s": rw_err_s,
        "rw_err_h": rw_err_h,
        "dmg_pred": dmg_pred_f,
        "dmg_at_gt_loc": dmg_at_gt,
        "loc_argmax": loc_argmax,
        "gt_dmg": gt_dmg,
        "gt_loc": gt_loc,
        "presence_at_gt": presence_prob[gt_loc].item(),
        "severity_at_gt": severity_sq[gt_loc].item(),
    }

    if print_result:
        print(
            f"Pred {dmg_pred_f:7.04f} @ {loc_argmax:d} | "
            f"GT {gt_dmg:7.04f} @ {gt_loc:d} | "
            f"Dmg@GT {dmg_at_gt:7.04f} (P={result['presence_at_gt']:.3f}, S={result['severity_at_gt']:.3f}) | "
            f"MSE: {rw_mse:10.04e} | rw_err_s: {rw_err_s:10.04e}"
        )

    return result


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    model_cfg=None,
) -> torch.nn.Module:
    """
    Load a saved `model.state_dict()` checkpoint and return a ready-to-run model.

    This matches the notebook usage where the checkpoint filename looks like:
    `states/single-damage-sparse-<uuid>.pt`.
    """
    # Local import to keep this module lightweight.
    from lib.model import ModelConfig, build_model

    if model_cfg is None:
        model_cfg = ModelConfig()

    model = build_model(model_cfg)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state)

    if device is not None:
        model = model.to(device)
    return model


def do_real_test_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    mat_path: str | Path | None = None,
    spec: RealMatBenchmarkSpec = DEFAULT_BENCHMARK,
    model_cfg=None,
    print_result: bool = True,
) -> dict[str, float | int]:
    """
    Convenience wrapper for ad-hoc evaluation:
    load checkpoint -> run `do_real_test`.
    """
    model = load_model_from_checkpoint(
        checkpoint_path, device=device, model_cfg=model_cfg
    )
    return do_real_test(
        model,
        device=device,
        mat_path=mat_path,
        spec=spec,
        print_result=print_result,
    )


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


def do_real_test_b_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    mat_path: str | Path | None = None,
    spec: RealMatBenchmarkSpec = DEFAULT_BENCHMARK,
    model_cfg=None,
    print_result: bool = True,
) -> dict[str, float | int]:
    """Convenience wrapper: load Approach-B checkpoint -> run `do_real_test_b`."""
    model = load_model_b_from_checkpoint(
        checkpoint_path, device=device, model_cfg=model_cfg
    )
    return do_real_test_b(
        model,
        device=device,
        mat_path=mat_path,
        spec=spec,
        print_result=print_result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SDINet real benchmark from a checkpoint."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to a saved model.state_dict() checkpoint (.pt).",
    )
    args = parser.parse_args()

    do_real_test_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
