"""Measure distances between selected, tracked RGB-D landmarks.

The caller supplies their semantic meaning. This module identifies no door,
closed configuration, hinge or preferred motion direction.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .part_perception import usable_pixels


def landmark_sample(view: dict | None, pixel_xy: ArrayLike, visible: bool = True) -> dict[str, Any]:
    """Record a literal depth sample and whether it supports the tracked landmark."""
    pixel = np.asarray(pixel_xy, dtype=float)
    if pixel.shape != (2,):
        raise ValueError('A landmark needs two original-image pixel coordinates')
    result: dict[str, Any] = {
        'pixel_xy': pixel.tolist() if np.isfinite(pixel).all() else None,
        'tracker_visible': bool(visible), 'sample_pixel_xy': None,
        'raw_world_m': None, 'depth_supported': False, 'supported': False,
    }
    if not np.isfinite(pixel).all():
        result['reason'] = 'nonfinite_tracker_prediction'
        return result
    if view is None:
        result['reason'] = 'recorded_depth_not_retained'
        return result
    height, width = np.asarray(view['depth']).shape
    x, y = map(int, np.rint(pixel))
    result['sample_pixel_xy'] = [x, y]
    if not (0 <= x < width and 0 <= y < height):
        result['reason'] = 'outside_image'
        return result
    world = np.asarray(view['world_points'])[y, x]
    depth = float(np.asarray(view['depth'])[y, x])
    finite_depth = bool(np.isfinite(world).all() and np.isfinite(depth) and depth > 0)
    if finite_depth:
        result['raw_world_m'] = world.tolist()
    result['depth_supported'] = bool(usable_pixels(view)[y, x])
    result['supported'] = bool(visible and result['depth_supported'])
    result['reason'] = ('measured' if result['supported'] else 'tracker_not_visible' if not visible
                        else 'invalid_depth' if not finite_depth else 'known_robot_depth')
    return result


def measured_gap(moving: dict, fixed: dict) -> dict[str, Any]:
    """Measure the two current landmarks; never substitute a held or predicted point."""
    raw = None
    if moving.get('raw_world_m') is not None and fixed.get('raw_world_m') is not None:
        raw = float(np.linalg.norm(np.asarray(moving['raw_world_m']) - fixed['raw_world_m']))
    supported = bool(moving.get('supported') and fixed.get('supported'))
    return {'gap_m': raw if supported else None,
            'raw_sample_distance_m': raw,
            'supported': supported,
            'reason': 'measured_selected_pair' if supported else {
                'moving': moving.get('reason'), 'fixed': fixed.get('reason')},
            'semantic_goal_verified': False,
            'scope': 'Euclidean distance between the two selected material-point hypotheses. '
                     'Both are tracked and depth-sampled at this capture. No closing sign or success label.'}
