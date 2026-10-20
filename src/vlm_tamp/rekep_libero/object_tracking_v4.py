"""Measure closure-induced object motion before testing retained transport."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import transform_points


def register_closed_surface(reference, observed, maximum_center_motion=.025, maximum_angle=.4):
    """Bounded local rigid registration of a selected surface at closure.

    No grasp attachment is inferred here. The subsequent lift/pull must still
    demonstrate retained transport against the stationary-surface hypothesis.
    """
    reference, observed = np.asarray(reference, float), np.asarray(observed, float)
    if len(reference) < 75 or len(observed) < 75:
        raise ValueError('Insufficient observed surface to track closure-induced motion')
    tree = cKDTree(observed)
    center = reference.mean(axis=0)
    transform = np.eye(4)
    for _ in range(35):
        moved = transform_points(reference, transform)
        distance, indices = tree.query(moved)
        keep = distance < .018
        if keep.sum() < 75:
            raise ValueError('Closure surface registration has insufficient correspondences')
        first, second = moved[keep], observed[indices[keep]]
        a, b = first.mean(axis=0), second.mean(axis=0)
        u, _, vt = np.linalg.svd((first - a).T @ (second - b))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        delta = np.eye(4)
        delta[:3, :3], delta[:3, 3] = rotation, b - rotation @ a
        proposed = delta @ transform
        motion = np.linalg.norm(transform_points([center], proposed)[0] - center)
        angle = Rotation.from_matrix(proposed[:3, :3]).magnitude()
        if motion > maximum_center_motion or angle > maximum_angle:
            break
        transform = proposed
        if np.linalg.norm(delta[:3, 3]) < 1e-5 and Rotation.from_matrix(rotation).magnitude() < 1e-4:
            break
    distance = tree.query(transform_points(reference, transform))[0]
    result = {'source': 'bounded RGB-D rigid surface registration at physical closure',
              'attachment_claimed': False, 'simulator_object_state_used': False,
              'transform_initial_to_closed_world': transform.tolist(),
              'coverage_6mm': float(np.mean(distance < .006)),
              'percentile75_m': float(np.quantile(distance, .75)),
              'center_motion_m': float(np.linalg.norm(transform_points([center], transform)[0] - center)),
              'angle_rad': float(Rotation.from_matrix(transform[:3, :3]).magnitude()),
              'maximum_center_motion_m': maximum_center_motion, 'maximum_angle_rad': maximum_angle}
    result['verified'] = result['coverage_6mm'] >= .6 and result['percentile75_m'] <= .009
    return result
