"""Two-view rigid-part measurements for caller-selected semantic regions.

Distances describe visible selected surfaces. They do not assign opening/closing meaning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from .contact_skills import fit_object_motion


def fit_camera_groups(reference: dict[str, np.ndarray], current: dict[str, np.ndarray]) -> dict:
    """Fit one rigid motion from all supplied views in their calibrated world frame."""
    names = sorted(reference)
    first = np.concatenate([reference[name] for name in names])
    last = np.concatenate([current[name] for name in names])
    result = fit_object_motion(first, last)
    inliers = set(result.get("inlier_indices", []))
    offset = 0
    support = {}
    for name in names:
        count = len(reference[name])
        finite = np.isfinite(reference[name]).all(1) & np.isfinite(current[name]).all(1)
        support[name] = {
            "queries": count,
            "measured_pairs": int(finite.sum()),
            "inliers": sum(i in inliers for i in range(offset, offset + count)),
            "inlier_query_indices": [
                i - offset for i in range(offset, offset + count) if i in inliers
            ],
        }
        offset += count
    return {
        **result,
        "camera_support": support,
        "cameras_supplied": names,
        "closing_direction_supplied": False,
    }


def transformed_region(points: ArrayLike, fit: dict) -> np.ndarray | None:
    """Carry the registered measured region only when its current rigid fit is supported."""
    if not fit["accepted"]:
        return None
    transform = np.asarray(fit["transform"])
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def region_proximity(moving: ArrayLike | None, fixed: ArrayLike | None) -> dict[str, Any]:
    """Measure visible-region surface proximity without selecting mating pixels or a goal pose."""
    if moving is None or fixed is None:
        return {
            "available": False,
            "reason": "current_region_pose_unresolved",
            "closing_label": None,
        }
    first = np.asarray(moving, float).reshape(-1, 3)
    second = np.asarray(fixed, float).reshape(-1, 3)
    if not len(first) or not len(second):
        return {
            "available": False,
            "reason": "selected_region_has_no_measured_points",
            "closing_label": None,
        }
    distances = {
        "door_to_opening": cKDTree(second).query(first)[0],
        "opening_to_door": cKDTree(first).query(second)[0],
    }
    result = {
        "available": True,
        "closing_label": None,
        "semantic_match_verified": False,
        "definition": "Nearest-surface distances between the two selected visible RGB-D regions, carried by their measured poses. Smaller is not automatically closing.",
    }
    for name, values in distances.items():
        result[name] = {
            "mean_m": float(np.mean(values)),
            "median_m": float(np.median(values)),
            "p90_m": float(np.quantile(values, 0.9)),
        }
    result["symmetric_mean_m"] = float(np.mean([np.mean(v) for v in distances.values()]))
    return result
