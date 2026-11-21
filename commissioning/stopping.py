"""Normal-only stopping rule. Delta is measured against the previous grid point, not N/2."""

from __future__ import annotations

import math

import numpy as np


def relative_deltas(medians: list[float] | np.ndarray) -> list[float | None]:
    values = [float(v) for v in medians]
    if any(not math.isfinite(v) for v in values):
        raise RuntimeError("Stopping medians must be finite.")
    deltas: list[float | None] = [None]
    for previous, current in zip(values, values[1:]):
        if previous == 0.0:
            deltas.append(0.0 if current == 0.0 else math.inf)
        else:
            deltas.append(abs(current - previous) / abs(previous))
    return deltas


def n_stop(grid: list[int], medians: list[float] | np.ndarray, delta_threshold: float) -> int | None:
    """Endpoint of the second consecutive grid transition whose relative change is below the threshold.

    Returns None when the rule never fires. There is no interpolation between grid points.
    """
    if len(grid) != len(medians):
        raise RuntimeError("Grid and calibration medians must have the same length.")
    if len(grid) < 2:
        return None
    deltas = relative_deltas(medians)
    streak = 0
    for index in range(1, len(grid)):
        if deltas[index] is not None and deltas[index] < delta_threshold:
            streak += 1
            if streak >= 2:
                return int(grid[index])
        else:
            streak = 0
    return None
