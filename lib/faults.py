"""
lib/faults.py — Shared sensor fault injection utilities.

Used by:
  - lib/data_7story.py (training augmentation)
  - lib/data_qatar.py  (training augmentation)
  - lib/data_tower.py  (training augmentation)
  - lib/data_lumo.py   (training augmentation)
  - lib/training.py    (faulted validation pass)
  - evaluate_fault.py  (evaluation sweep)
"""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Fault type registry
# ---------------------------------------------------------------------------

_SOFT_FAULT_TYPES: list[str] = ["gain", "bias", "gain_bias", "noise", "stuck", "partial"]

FAULT_TYPES: list[str] = ["hard"] + _SOFT_FAULT_TYPES


def _apply_soft_fault(x: torch.Tensor, idx: list[int], fault_type: str) -> None:
    """Apply a soft sensor fault in-place to x (1, T, S) at sensor indices idx."""
    seg = x[:, :, idx]                                          # (1, T, k)
    rms = seg.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8   # (1, 1, k)

    if fault_type == "gain":
        # Amplifier drift / cable resistance: moderate gain error
        scale = torch.rand(1).item()
        scale = scale * 0.30 + 0.50 if torch.rand(1).item() < 0.5 else scale * 0.50 + 1.20
        x[:, :, idx] = seg * scale                               # U(0.5,0.8) or U(1.2,1.7)

    elif fault_type == "bias":
        # DC offset from ground loop / amplifier drift
        magnitude = (torch.rand(1).item() * 0.8 + 0.2) * rms   # U(0.2, 1.0) × rms
        sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0
        x[:, :, idx] = seg + sign * magnitude

    elif fault_type == "gain_bias":
        # Loose/corroded connector: gain error + DC offset
        scale = torch.rand(1).item() * 0.40 + 0.50              # U(0.5, 0.9)
        magnitude = (torch.rand(1).item() * 0.4 + 0.1) * rms   # U(0.1, 0.5) × rms
        sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0
        x[:, :, idx] = seg * scale + sign * magnitude

    elif fault_type == "noise":
        # EMI / cable degradation: additive noise on top of signal
        noise_scale = (torch.rand(1).item() * 1.5 + 0.5) * rms  # U(0.5, 2.0) × rms
        x[:, :, idx] = seg + torch.randn_like(seg) * noise_scale

    elif fault_type == "stuck":
        # Frozen DAQ: replace time series with its mean + small noise
        mean_val = seg.mean(dim=1, keepdim=True)                  # (1, 1, k)
        x[:, :, idx] = mean_val + torch.randn_like(seg) * (0.01 * rms)

    elif fault_type == "partial":
        # Partial attenuation: loose connector, signal reduced but present
        scale = torch.rand(1).item() * 0.40 + 0.30               # U(0.3, 0.7)
        x[:, :, idx] = seg * scale


def apply_signal_aug(x: torch.Tensor) -> torch.Tensor:
    """Always-on training regularisation: amplitude scaling + per-sensor noise.

    Applied to every training sample across all datasets for parity.

    1. Amplitude scaling — multiply all channels by U(0.8, 1.2).
    2. Per-sensor Gaussian noise — σ = 5% of each sensor's RMS.

    Args:
        x: (1, T, S) or (C, T, S) single-sample tensor (cloned by caller).
    Returns:
        Augmented tensor, same shape.
    """
    x.mul_(0.8 + 0.4 * torch.rand(1).item())
    sensor_rms = x.pow(2).mean(dim=1, keepdim=True).sqrt()  # (C, 1, S)
    x.add_(torch.randn_like(x) * (0.05 * sensor_rms))
    return x


def apply_fault_aug(
    x: torch.Tensor,
    n_sensors: int,
    struct_masks: list[list[int]],
    *,
    p_hard: float = 0.0,
    p_struct: float = 0.0,
    p_soft: float = 0.0,
) -> torch.Tensor:
    """Inject sensor faults into a single training sample.

    Three independent fault mechanisms applied in order:
      1. Per-sensor hard fault — each sensor independently zeroed with prob ``p_hard``.
      2. Structured masking    — with prob ``p_struct``, zero one pre-defined sensor group.
      3. Per-sensor soft fault — each non-hard-faulted sensor independently gets a
         random soft fault type with prob ``p_soft``.

    The number of faulted sensors is a natural binomial draw, scaling automatically
    with the sensor count S of each dataset.

    Args:
        x:             (1, T, S) or (C, T, S) single-sample tensor (modified in-place).
        n_sensors:     Number of sensors S.
        struct_masks:  List of sensor-index groups for structured masking.
        p_hard:        Per-sensor probability of hard fault (zero-out).
        p_struct:      Per-window probability of structured group masking.
        p_soft:        Per-sensor probability of soft fault.

    Returns:
        y_fault: (S,) binary tensor — 1.0 for faulted sensors, 0.0 otherwise.
    """
    import random as _random

    y_fault = torch.zeros(n_sensors)

    # 1. Per-sensor hard faults
    if p_hard > 0.0:
        hard_mask = torch.rand(n_sensors) < p_hard
        if hard_mask.any():
            idx = hard_mask.nonzero(as_tuple=True)[0].tolist()
            x[..., idx] = 0.0
            y_fault[idx] = 1.0

    # 2. Structured masking (per-window gate, then one random group)
    if p_struct > 0.0 and struct_masks and torch.rand(1).item() < p_struct:
        mask_idx = int(torch.randint(0, len(struct_masks), (1,)).item())
        struct_idx = struct_masks[mask_idx]
        x[..., struct_idx] = 0.0
        y_fault[struct_idx] = 1.0

    # 3. Per-sensor soft faults (skip already-hard-faulted sensors)
    if p_soft > 0.0:
        eligible = (y_fault == 0.0)
        soft_mask = (torch.rand(n_sensors) < p_soft) & eligible
        if soft_mask.any():
            idx = soft_mask.nonzero(as_tuple=True)[0].tolist()
            fault_type = _random.choice(_SOFT_FAULT_TYPES)
            _apply_soft_fault(x, idx, fault_type)
            y_fault[idx] = 1.0

    return y_fault


# ---------------------------------------------------------------------------
# Shared training dataset with on-the-fly augmentation
# ---------------------------------------------------------------------------

class AugDataset(torch.utils.data.Dataset):
    """Training-only dataset: signal aug + optional fault aug.

    Wraps pre-loaded (x, y) tensors.  Always applies :func:`apply_signal_aug`.
    When any fault flag > 0, also applies :func:`apply_fault_aug` and returns
    ``(x, y, y_fault)`` triplets; otherwise returns ``(x, y)`` pairs.

    Args:
        x:             (N, 1, T, S) pre-loaded input tensors.
        y:             (N, L) label tensors.
        n_sensors:     Number of sensors S (dataset-specific).
        struct_masks:  Pre-built structured mask groups for the dataset.
        p_hard:        Per-sensor hard-fault probability.
        p_struct:      Per-window structured masking probability.
        p_soft:        Per-sensor soft-fault probability.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_sensors: int,
        struct_masks: list[list[int]],
        p_hard: float = 0.0,
        p_struct: float = 0.0,
        p_soft: float = 0.0,
    ) -> None:
        self.x = x
        self.y = y
        self._use_fault = p_hard > 0 or p_struct > 0 or p_soft > 0
        self._fault_kwargs = dict(
            n_sensors=n_sensors, struct_masks=struct_masks,
            p_hard=p_hard, p_struct=p_struct, p_soft=p_soft,
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        x = apply_signal_aug(self.x[idx].clone())
        if self._use_fault:
            y_fault = apply_fault_aug(x, **self._fault_kwargs)
            return x, self.y[idx], y_fault
        return x, self.y[idx]


def inject_faults_batch(
    x: torch.Tensor,
    fault_type: str,
    n_faulted: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inject a single fault type into a clean batch for evaluation.

    Sensor selection is per-sample random (seeded via rng) so that the same
    rng state produces reproducible faulted batches.  Fault magnitudes for
    soft types use the global torch RNG — average over n_repeats in
    evaluate_fault.py for stable metrics.

    Args:
        x:          (B, 1, T, S) clean batch tensor.
        fault_type: one of FAULT_TYPES.  "hard" zeros selected sensors.
        n_faulted:  exact number of sensors to fault per sample.
        rng:        seeded Generator for reproducible sensor selection.

    Returns:
        x_faulted: (B, 1, T, S)
        y_fault:   (B, S) binary GT ∈ {0, 1}
    """
    B, _, T, S = x.shape
    x_out   = x.clone()
    y_fault = torch.zeros(B, S)
    for b in range(B):
        idx = torch.randperm(S, generator=rng)[:n_faulted].tolist()
        y_fault[b, idx] = 1.0
        if fault_type == "hard":
            x_out[b, :, :, idx] = 0.0
        else:
            _apply_soft_fault(x_out[b], idx, fault_type)
    return x_out, y_fault
