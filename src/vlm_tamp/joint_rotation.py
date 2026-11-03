"""Explicit CW/CCW notation about a declared signed hinge axis."""
from __future__ import annotations

import math

VIEW_CONVENTION = 'View from the positive hinge-axis end toward its origin: ccw is positive, cw negative.'


def signed_rotation(degrees: float | None, direction: str | None = None) -> float | None:
    """Convert a named direction and positive magnitude; preserve signed legacy calls."""
    if direction is None:
        return degrees
    if direction not in {'cw', 'ccw'}:
        raise ValueError('rotation_direction must be cw or ccw')
    if degrees is None or not math.isfinite(degrees) or degrees <= 0:
        raise ValueError('CW/CCW requires a positive angle magnitude in joint_degrees (degrees for plan_arc)')
    return degrees if direction == 'ccw' else -degrees
