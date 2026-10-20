"""Observed prismatic handle displacement, independent of hand commands."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from .geometry import transform_points
from .object_motion_v3 import observed_object_cloud


def measure_progress(reference, cameras, robot_volumes, axis, expected_progress,
                     previous_progress=0., minimum_progress=0., maximum_progress=.18):
    reference = np.asarray(reference, float)
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    predicted = np.eye(4)
    predicted[:3, 3] = float(expected_progress) * axis
    observed = observed_object_cloud(cameras, reference, predicted, robot_volumes)
    result = {'tracking_valid': False, 'source': 'RGB-D selected handle surface translated along measured panel normal',
              'simulator_object_state_used': False, 'observed_points': len(observed),
              'expected_hand_progress_m': float(expected_progress), 'previous_observed_progress_m': float(previous_progress)}
    if len(reference) < 75 or len(observed) < 75:
        return {**result, 'failure': 'insufficient_observed_handle_support'}
    tree = cKDTree(observed)
    low = max(float(minimum_progress), min(previous_progress, expected_progress) - .025)
    high = min(float(maximum_progress), max(previous_progress, expected_progress) + .03)
    if high <= low:
        return {**result, 'failure': 'invalid_observed_progress_search_interval'}

    def distances(progress):
        return tree.query(reference + progress * axis)[0]

    def cost(progress):
        return float(np.mean(np.minimum(distances(progress), .018)**2))

    grid = np.linspace(low, high, max(3, int(np.ceil((high - low) / .0015)) + 1))
    best = float(grid[np.argmin([cost(value) for value in grid])])
    fit = minimize_scalar(cost, bounds=(max(low, best - .002), min(high, best + .002)),
                          method='bounded', options={'xatol': 1e-5})
    progress = float(fit.x)
    distance = distances(progress)
    stationary = distances(previous_progress)
    p75 = float(np.quantile(distance, .75))
    coverage = float(np.mean(distance < .006))
    transform = np.eye(4)
    transform[:3, 3] = progress * axis
    result.update(tracking_valid=bool(coverage >= .6 and p75 <= .009),
                  observed_progress_m=progress, observed_increment_m=progress - previous_progress,
                  coverage_6mm=coverage, percentile75_m=p75,
                  previous_surface_percentile75_m=float(np.quantile(stationary, .75)),
                  following_better=bool(p75 + .0015 < np.quantile(stationary, .75)),
                  hand_surface_discrepancy_m=float(expected_progress - progress),
                  transform_closed_to_observed_world=transform.tolist(),
                  search_bounds_m=[low, high])
    return result


def retained_progress(measurement, gripper_qpos, *,maximum_hand_surface_error=.015):
    gap = float(np.sum(np.abs(gripper_qpos)))
    return bool(measurement['tracking_valid'] and .002 < gap < .079
                and abs(measurement['hand_surface_discrepancy_m']) <= maximum_hand_surface_error)


def closed_reference(initial_reference, closure_transform):
    return transform_points(initial_reference, closure_transform)
