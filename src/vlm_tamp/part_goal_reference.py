"""Carry an explicitly selected reference using measured rigid-part motion."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .contact_skills import fit_object_motion


def carry_reference(
    reference_points: ArrayLike, current_points: ArrayLike, reference_world_m: ArrayLike
) -> dict[str, Any]:
    """Fit the whole selected part; never copy a previous pose after an unsupported fit."""
    anchor = np.asarray(reference_world_m, float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("The attached reference needs a finite observed world point")
    fit = fit_object_motion(reference_points, current_points)
    result = {
        "fit": fit,
        "estimated_reference_world_m": None,
        "reference_world_m": anchor.tolist(),
        "source": "Measured RGB-D rigid-part transform applied to the selected reference",
        "closing_direction_supplied": False,
        "simulator_geometry_used": False,
        "is_direct_reference_depth_measurement": False,
    }
    if fit["accepted"]:
        transform = np.asarray(fit["transform"])
        result["estimated_reference_world_m"] = (
            transform[:3, :3] @ anchor + transform[:3, 3]
        ).tolist()
    return result


def part_reference_gap(moving: dict, fixed: dict) -> dict[str, Any]:
    """Return a clearly labelled estimated gap only when both part poses are supported."""
    a, b = moving.get("estimated_reference_world_m"), fixed.get("estimated_reference_world_m")
    value = float(np.linalg.norm(np.asarray(a) - b)) if a is not None and b is not None else None
    return {
        "estimated_gap_m": value,
        "supported": value is not None,
        "source": "Two references carried by independently measured rigid-part transforms",
        "semantic_goal_verified": False,
        "native_state_used": False,
        "unsupported_reasons": {}
        if value is not None
        else {"moving": moving["fit"].get("reason"), "fixed": fixed["fit"].get("reason")},
    }

