"""Numerical relations on poses; no generated code, robot command, or planner import.

Jacobians use [world translation, right/body rotation] perturbations:
p' = p + dp, R' = R exp(skew(dw)). A robot backend can apply its pose
Jacobian by the chain rule. These are not yet joint-space ST-pRRTC kernels.
"""

import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(vector).all() or length < 1e-9:
        raise ValueError("Expected a finite nonzero 3-vector")
    return vector / length


def skew(vector):
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def check_pose(position, rotation):
    position, rotation = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("Expected position (3,) and rotation (3,3)")
    if not np.isfinite(position).all() or not np.isfinite(rotation).all():
        raise ValueError("Nonfinite pose")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-7
    ):
        raise ValueError("Rotation must belong to SO(3)")
    return position, rotation


def perpendicular_basis(axis):
    axis = unit(axis)
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    first = unit(np.cross(axis, helper))
    return np.column_stack([first, np.cross(axis, first)])


def evaluate_relation(spec, position, rotation):
    """Return residual/Jacobian and separate inequality/Jacobian arrays.

point_target constrains the explicitly selected point, not whole-object On/In.
axis_alignment has two independent equality residuals plus a branch inequality
excluding an upside-down solution. axis_cone encodes an angular tolerance as
an inequality; it must not be silently substituted for the equality manifold.
"""
    p, r = check_pose(position, rotation)
    kind = spec["type"]
    h, j, g, jg = np.empty(0), np.empty((0, 6)), np.empty(0), np.empty((0, 6))
    if kind == "point_target":
        local = np.asarray(spec["point_local_m"], dtype=float)
        target = np.asarray(spec["target_world_m"], dtype=float)
        if local.shape != (3,) or target.shape != (3,) or not np.isfinite([local, target]).all():
            raise ValueError("Invalid point-target parameters")
        h = p + r @ local - target
        j = np.column_stack([np.eye(3), -r @ skew(local)])
    elif kind in {"axis_alignment", "axis_cone"}:
        local, target = unit(spec["axis_local"]), unit(spec["axis_world"])
        current = r @ local
        derivative = np.column_stack([np.zeros((3, 3)), -r @ skew(local)])
        if kind == "axis_alignment":
            basis = perpendicular_basis(target)
            h, j = basis.T @ current, basis.T @ derivative
            g, jg = np.array([-target @ current]), (-target @ derivative)[None, :]
        else:
            angle = spec["max_angle_degrees"]
            if isinstance(angle, bool) or not np.isfinite(angle) or not 0 < angle < 90:
                raise ValueError("Axis cone needs an angle strictly between 0 and 90 degrees")
            g = np.array([np.cos(np.deg2rad(angle)) - target @ current])
            jg = (-target @ derivative)[None, :]
    else:
        raise ValueError("Unsupported numerical relation: " + kind)
    return {"equality": h, "equality_jacobian": j,
            "inequality": g, "inequality_jacobian": jg}


def satisfied(spec, position, rotation, equality_tolerance=1e-7):
    result = evaluate_relation(spec, position, rotation)
    return bool(np.all(np.abs(result["equality"]) <= equality_tolerance)
                and np.all(result["inequality"] <= 1e-12))
