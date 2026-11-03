"""Explicit CW/CCW notation about a declared signed hinge axis."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

VIEW_CONVENTION = 'View from the positive hinge-axis end toward its origin: ccw is positive, cw negative.'
AXIS_CONVENTION = 'The largest-magnitude world component is positive (exact ties: X, then Y, then Z). This sets sign only; it does not align the hinge with a world axis.'


def canonical_axis(axis: Sequence[float]) -> np.ndarray:
    """Orient an already validated nonzero hinge line independently of point order."""
    value = np.asarray(axis, dtype=float)
    value = value / np.linalg.norm(value)
    component = int(np.argmax(np.abs(value)))
    return value if value[component] > 0 else -value


def signed_rotation(degrees: float | None, direction: str | None = None) -> float | None:
    """Convert a named direction and positive magnitude; preserve signed legacy calls."""
    if direction is None:
        return degrees
    if direction not in {'cw', 'ccw'}:
        raise ValueError('rotation_direction must be cw or ccw')
    if degrees is None or not math.isfinite(degrees) or degrees <= 0:
        raise ValueError('CW/CCW requires a positive angle magnitude in joint_degrees (degrees for plan_arc)')
    return degrees if direction == 'ccw' else -degrees
