"""
lib/preprocessing.py — Shared signal preprocessing utilities.

Used by all dataset loaders (data_7story, data_qatar, data_lumo) to ensure
consistent preprocessing across the sim → lab → field trio.
"""
from __future__ import annotations

import numpy as np


def normalize_rms(
    windows: np.ndarray,
    ref_channel: int | None = None,
    method: str = "mean",
) -> np.ndarray:
    """Per-window RMS normalization.

    Divides all channels by a scalar RMS computed from either a single
    reference sensor or an aggregate across all sensors.  Preserves
    cross-channel amplitude ratios (which encode structural damage
    location) while providing excitation-amplitude invariance.

    When ``ref_channel`` is an integer, the RMS of that single sensor is
    used (transmissibility referencing).  When ``ref_channel`` is ``None``,
    the per-sensor RMS is aggregated according to ``method``:

      - ``"mean"`` (default): global RMS across all sensors and time steps
        jointly — equivalent to the L2 energy norm of the window divided by
        sqrt(T·S).  Fault-sensitive by design: the amplitude cue surfaced by
        fault contamination of the denominator is exploitable by a fault-aware
        head (see Insight #30).
      - ``"none"``: no normalization; returns ``windows`` unchanged.

    Args:
        windows:     (N, T, S) windowed signal array.
        ref_channel: Sensor index for single-sensor reference, or None for
                     aggregate RMS across all sensors.
        method:      Aggregation when ``ref_channel`` is None:
                     {"mean", "none"}.

    Returns:
        Normalized array, same shape.
    """
    if ref_channel is not None:
        rms = np.sqrt(np.mean(windows[:, :, ref_channel] ** 2, axis=1, keepdims=True))  # (N, 1)
        rms = rms[:, :, np.newaxis]  # (N, 1, 1)
        return windows / (rms + 1e-8)

    if method == "none":
        return windows
    if method == "mean":
        rms = np.sqrt(np.mean(windows ** 2, axis=(1, 2), keepdims=True))  # (N, 1, 1)
        return windows / (rms + 1e-8)
    raise ValueError(f"Unknown normalize_rms method: {method!r}  (choices: 'mean', 'none')")
