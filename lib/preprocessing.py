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
) -> np.ndarray:
    """Per-window RMS normalization.

    Divides all channels by a scalar RMS computed from either a single
    reference sensor or all sensors.  Preserves cross-channel amplitude
    ratios (which encode structural damage location) while providing
    excitation-amplitude invariance.

    When ``ref_channel`` is an integer, the RMS of that single sensor is used
    (transmissibility referencing — ideal when a known excitation input sensor
    exists, e.g. Qatar channel 0 near the shaker).

    When ``ref_channel`` is ``None``, the RMS is computed across all sensors
    jointly.  This is suitable when no natural reference sensor exists
    (e.g. 7-story simulation, LUMO ambient excitation).

    Args:
        windows:     (N, T, S) windowed signal array.
        ref_channel: Sensor index for single-sensor reference, or None for
                     global (all-sensor) RMS.

    Returns:
        Normalized array, same shape.
    """
    if ref_channel is not None:
        rms = np.sqrt(np.mean(windows[:, :, ref_channel] ** 2, axis=1, keepdims=True))  # (N, 1)
        rms = rms[:, :, np.newaxis]  # (N, 1, 1)
    else:
        rms = np.sqrt(np.mean(windows ** 2, axis=(1, 2), keepdims=True))  # (N, 1, 1)
    return windows / (rms + 1e-8)
