"""Signed surface-contact frames; no task direction or wrist-twist prior."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def normal_vector(value: Sequence[float], label: str = "surface normal") -> np.ndarray:
    """Return a unit vector without changing the caller's chosen side."""
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
        raise ValueError(f"A finite nonzero {label} is required")
    return vector / np.linalg.norm(vector)


def checked_surface_normal(
    constraint: str, surface_normal_world: Sequence[float] | None
) -> np.ndarray | None:
    """Require an explicit signed normal only for the selected surface model."""
    if constraint not in {"point", "surface_normal"}:
        raise ValueError("contact_constraint must be point or surface_normal")
    if constraint == "point":
        if surface_normal_world is not None:
            raise ValueError(
                "Select contact_constraint=surface_normal to constrain a supplied surface normal"
            )
        return None
    if surface_normal_world is None:
        raise ValueError("surface_normal requires the selected face's signed outward surface_normal_world")
    return normal_vector(surface_normal_world)


def frame_with_z(normal: Sequence[float]) -> np.ndarray:
    """Construct coordinates around a given axis; the x/y basis is not a wrist target.

    The least-parallel Cartesian axis keeps the coordinate construction nonsingular.
    cuRobo's goal-frame yaw weight is zero, so rotating this basis about z does not
    change which physical normal satisfies the requested contact constraint.
    """
    z_axis = normal_vector(normal)
    reference = np.eye(3)[np.argmin(np.abs(z_axis))]
    x_axis = np.cross(reference, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    return np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis))


def approach_geometry(point: Sequence[float], normal: Sequence[float],
                      hand_spheres_tcp: Sequence[Sequence[float]],
                      contact_normal_local: Sequence[float] | None) -> dict[str, Any]:
    """Use one hand extent as approach travel, with no obstacle inflation.

    A signed face normal selects the outside. A point-only request instead uses
    its explicit motion direction and an orientation-independent hand extent.
    """
    outward = normal_vector(normal)
    hand = np.asarray(hand_spheres_tcp, float)
    if contact_normal_local is not None:
        projection = hand[:,:3]@normal_vector(contact_normal_local)
        distance = float(np.max(projection+hand[:,3])-np.min(projection-hand[:,3]))
    else:
        distance = float(np.linalg.norm(np.max(hand[:,:3]+hand[:,3:4],axis=0)
                                        -np.min(hand[:,:3]-hand[:,3:4],axis=0)))
    return {'contact_point_world_m': np.asarray(point).tolist(),
            'precontact_point_world_m': (np.asarray(point)+distance*outward).tolist(),
            'outward_direction_world': outward.tolist(), 'approach_direction_world': (-outward).tolist(),
            'approach_distance_m': distance, 'distance_source': 'measured-aperture hand sphere extent',
            'collision_padding_m': 0., 'free_space_finger_collision_checked': True,
            'contact_motion_finger_collision_checked': False,
            'task_direction_verified': False}


def normal_angle(normal_a: Sequence[float], normal_b: Sequence[float]) -> float:
    """Return the signed-face separation angle, preserving opposite faces as pi."""
    dot = normal_vector(normal_a) @ normal_vector(normal_b)
    return float(np.arccos(np.clip(dot, -1.0, 1.0)))


def contact_orientation_report(
    rotation_world: np.ndarray,
    contact_normal_local: Sequence[float],
    surface_normal_world: Sequence[float],
    direction_world: Sequence[float] | None = None,
    current_contact_point: Sequence[float] | None = None,
    target_point: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Report normal opposition and requested motion, without inferring open/close."""
    outward = normal_vector(surface_normal_world)
    actual = np.asarray(rotation_world) @ normal_vector(contact_normal_local, "robot contact normal")
    report = {
        "surface_normal_world": outward.tolist(),
        "desired_robot_contact_normal_world": (-outward).tolist(),
        "robot_contact_normal_world": actual.tolist(),
        "normal_error_degrees": float(np.degrees(normal_angle(actual, -outward))),
        "wrist_twist": "free around the selected normal",
        "normal_sign_source": "caller-selected observed face; never flipped to infer task direction",
        "task_direction_verified": False,
        "physical_contact_verified": False,
    }
    if direction_world is not None:
        direction = normal_vector(direction_world, "motion direction")
        report["requested_motion_dot_outward_normal"] = float(direction @ outward)
        report["motion_component_meaning"] = (
            "negative enters the selected face; positive moves away; neither means open or close"
        )
    if current_contact_point is not None and target_point is not None:
        report["current_contact_signed_plane_offset_m"] = float(
            (np.asarray(current_contact_point) - np.asarray(target_point)) @ outward
        )
        report["plane_side_meaning"] = (
            "positive is on the selected outward side; negative is the other plane side; a path may still go around an edge"
        )
    return report


def joint_push_prediction(
    point: Sequence[float],
    origin: Sequence[float],
    axis: Sequence[float],
    direction: Sequence[float],
) -> dict[str, Any]:
    """Report the moment arm for an explicitly associated joint, never a task sign."""
    lever = np.asarray(point) - np.asarray(origin)
    moment = float(normal_vector(axis) @ np.cross(lever, normal_vector(direction)))
    return {
        "joint_axis_moment_arm_m": moment,
        "signed_rotation_tendency": int(np.sign(moment)),
        "assumption": "force follows requested motion at the selected point; contact and force are not measured",
        "open_or_close": "unverified; the signed axis must separately be related to the task goal",
    }


def selection_geometry(
    point: Sequence[float],
    outward_normal: Sequence[float],
    robot_contact_point: Sequence[float],
    direction: Sequence[float],
    cameras: dict[str, Any],
) -> dict[str, Any]:
    """Describe plane sides and candidate views without choosing a face or action."""
    point = np.asarray(point, dtype=float)
    normal = normal_vector(outward_normal)
    side = float((np.asarray(robot_contact_point) - point) @ normal)
    views = []
    for name, camera in cameras.items():
        if camera.get("camera_to_world") is None:
            continue
        ray = np.asarray(camera["camera_to_world"])[:3, 3] - point
        distance = float(np.linalg.norm(ray))
        signed = float(ray @ normal)
        views.append(
            {
                "camera": name,
                "signed_plane_offset_m": signed,
                "normal_view_alignment": abs(signed) / distance if distance else None,
                "same_plane_side_as_contact": side * signed > 0,
            }
        )
    views.sort(key=lambda row: -(row["normal_view_alignment"] or 0.0))
    component=float(normal_vector(direction) @ normal)
    effect=('into_selected_face' if component<0 else 'away_from_selected_face' if component>0 else 'tangent_to_selected_face')
    advice=('The requested motion enters this face. Approach from its outward side for a normal push.' if component<0 else
        'The requested motion leaves this face. A normal push in that direction needs contact on the opposite observed face; an engaged grasp/hook could pull. Do not just flip the normal or reuse a hidden-face pixel.' if component>0 else
        'The requested motion is tangent to this face; normal contact alone does not provide that push direction.')
    return {
        "surface_normal_world": normal.tolist(),
        "initial_robot_contact_signed_plane_offset_m": side,
        "requested_motion_dot_outward_normal": component,
        "contact_effect":{'motion_relative_to_face':effect,'explanation':advice,
            'opposite_side_camera_candidates':[row['camera'] for row in views if row['signed_plane_offset_m']<0],
            'candidate_scope':'Camera positions on the other side of the measured plane; visibility is not certified.',
            'execution_veto_applied':False,'task_direction_verified':False},
        "camera_plane_relations": views,
        "same_plane_side_camera_candidates": [
            row["camera"] for row in views if row["same_plane_side_as_contact"]
        ],
        "interpretation": "Plane-side and view-angle evidence only; visibility, reachability and task direction are not inferred.",
    }
