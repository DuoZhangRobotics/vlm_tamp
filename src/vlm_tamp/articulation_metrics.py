"""Observer-only geometric metrics for unbounded, unoriented hinge axes."""
import numpy as np


def axis_errors(origin, axis, reference_origin, reference_axis):
    p, q = np.asarray(origin, float), np.asarray(reference_origin, float)
    a, b = np.asarray(axis, float), np.asarray(reference_axis, float)
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in [p, q, a, b]):
        raise ValueError("Axis metrics require finite 3D origins and directions")
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-12:
        raise ValueError("Axis directions must be nonzero")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    cross = np.cross(a, b)
    cross_norm = np.linalg.norm(cross)
    point_line = np.linalg.norm(np.cross(p - q, b))
    shortest = point_line if cross_norm < 1e-8 else abs(np.dot(p - q, cross)) / cross_norm
    # A unique point on the predicted axis at the reference anchor's height
    # along the reference direction. Unlike point-to-line, this is invariant
    # to sliding the predicted origin along its own axis.
    plane_point = None if abs(dot) < 1e-8 else p + a * (np.dot(q - p, b) / dot)
    plane = None if plane_point is None else float(np.linalg.norm(plane_point - q) * 1000.)
    return {"axis_angle_deg": float(np.degrees(np.arccos(np.clip(abs(dot), 0., 1.)))),
            "origin_to_reference_line_mm": float(point_line * 1000.),
            "shortest_line_distance_mm": float(shortest * 1000.),
            "reference_plane_error_mm": plane,
            "axis_sign_evaluated": False,
            "plane_metric_defined": plane is not None}
