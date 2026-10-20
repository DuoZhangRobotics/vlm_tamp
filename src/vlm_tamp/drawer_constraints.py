"""Observed drawer-front geometry and stage-scoped handle path proposals.

The compiler accepts RGB-selected landmarks, calibrated depth, and numerical
policy settings. It never receives simulator instances, joints, object poses,
or segmentation. A visible front normal is only a slider-axis heuristic.

Pose Jacobians follow the existing [world translation, right/body rotation]
convention: p' = p + dp, R' = R exp(skew(dw)). The handle-tracking pose is
abstract until a robot TCP and verified grasp attachment are bound to it.
"""

import math

import numpy as np

from .pose_constraints import check_pose, perpendicular_basis, skew, unit
from .rgbd_numericals import lift, world_points


DRAWER_POINT_FIELDS = (
    "source_uv", "handle_uv", "face_a_uv", "face_b_uv", "face_c_uv",
)
DRAWER_GEOMETRY_DEFAULTS = {
    "depth_patch_radius_pixels": 1,
    "maximum_depth_jump_m": 0.03,
    "maximum_depth_m": 5.0,
    "plane_patch_radius_pixels": 7,
    "plane_depth_band_m": 0.02,
    "maximum_plane_rms_m": 0.003,
    "minimum_drawer_face_span_m": 0.02,
    "minimum_drawer_triangle_sine": 0.08,
    "minimum_drawer_view_normal_cosine": 0.15,
    "drawer_line_tolerance_m": 0.01,
}


def validate_drawer_blanks(blanks):
    required = set(DRAWER_POINT_FIELDS) | {"joint_type", "motion", "path_constraint"}
    if not isinstance(blanks, dict) or set(blanks) != required:
        raise ValueError("Fill exactly the drawer landmark and method blanks")
    for key in DRAWER_POINT_FIELDS:
        value = blanks[key]
        if value is not None and (
            not isinstance(value, list) or len(value) != 2 or any(
                isinstance(item, bool) or not isinstance(item, (float, int))
                or not math.isfinite(item) or not 0 <= item <= 1 for item in value
            )
        ):
            raise ValueError("Invalid normalized drawer landmark: " + key)
    choices = {
        "joint_type": {"prismatic", "revolute", "unresolved"},
        "motion": {"pull", "push", "unresolved"},
        "path_constraint": {"prismatic_pull", "none", "unresolved"},
    }
    if any(not isinstance(blanks[key], str) or blanks[key] not in values
           for key, values in choices.items()):
        raise ValueError("Invalid drawer joint, motion, or constraint choice")
    return blanks


def _geometry_config(config):
    settings = {**DRAWER_GEOMETRY_DEFAULTS, **config}
    for key in ("depth_patch_radius_pixels", "plane_patch_radius_pixels"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Invalid nonnegative pixel radius: " + key)
    for key in (
        "maximum_depth_jump_m", "maximum_depth_m", "plane_depth_band_m",
        "maximum_plane_rms_m", "minimum_drawer_face_span_m", "drawer_line_tolerance_m",
    ):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid positive geometry tolerance: " + key)
    for key in ("minimum_drawer_triangle_sine", "minimum_drawer_view_normal_cosine"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value < 1:
            raise ValueError("Invalid dimensionless geometry threshold: " + key)
    return settings


def _front_plane(landmarks, depth, points, calibration, config):
    keys = ("face_a_uv", "face_b_uv", "face_c_uv")
    if not all(landmarks[key]["status"] == "measured" for key in keys):
        return {"status": "unresolved", "reason": "Three accepted drawer-front landmarks are required"}
    anchors = np.asarray([landmarks[key]["position_m"] for key in keys])
    first, second = anchors[1] - anchors[0], anchors[2] - anchors[0]
    lengths = [np.linalg.norm(first), np.linalg.norm(second), np.linalg.norm(anchors[2] - anchors[1])]
    if min(lengths) < config["minimum_drawer_face_span_m"]:
        return {"status": "unresolved", "reason": "Drawer-front landmarks are insufficiently separated",
                "anchor_pair_distances_m": [float(value) for value in lengths]}
    sine = float(np.linalg.norm(np.cross(first, second)) / (lengths[0] * lengths[1]))
    if sine < config["minimum_drawer_triangle_sine"]:
        return {"status": "unresolved", "reason": "Drawer-front landmarks are nearly collinear",
                "anchor_triangle_sine": sine}

    # Local neighborhoods are measured around the unchanged VLM pixels. Depth
    # filtering has no instance mask, semantic labels, or nearest-object rescue.
    selected = np.zeros(depth.shape, dtype=bool)
    radius = config["plane_patch_radius_pixels"]
    for key in keys:
        x, y = landmarks[key]["pixel_xy"]
        ys = slice(max(0, y - radius), min(depth.shape[0], y + radius + 1))
        xs = slice(max(0, x - radius), min(depth.shape[1], x + radius + 1))
        selected[ys, xs] |= np.abs(depth[ys, xs] - depth[y, x]) <= config["plane_depth_band_m"]
    cloud = points[selected]
    if len(cloud) < 12:
        return {"status": "unresolved", "reason": "Insufficient observed drawer-front surface samples"}
    center = cloud.mean(axis=0)
    _, singular, vt = np.linalg.svd(cloud - center, full_matrices=False)
    if singular[1] < 1e-5:
        return {"status": "unresolved", "reason": "Observed drawer-front surface is degenerate"}
    normal = vt[-1]
    rms = float(np.sqrt(np.mean(((cloud - center) @ normal) ** 2)))
    anchor_deviation = float(np.max(np.abs((anchors - center) @ normal)))
    result = {
        "normal_world": normal.tolist(), "plane_point_world_m": center.tolist(),
        "plane_rms_m": rms, "maximum_anchor_plane_deviation_m": anchor_deviation,
        "sample_count": len(cloud), "anchor_triangle_sine": sine,
        "anchor_pair_distances_m": [float(value) for value in lengths],
        "semantic_drawer_front_verified": False,
        "source": "Depth patches at three unchanged RGB-selected front-face pixels",
    }
    if rms > config["maximum_plane_rms_m"] or anchor_deviation > config["maximum_plane_rms_m"]:
        return {**result, "status": "unresolved", "reason": "Selected surface is not sufficiently planar"}
    camera = np.asarray(calibration["camera_to_world"], dtype=float)[:3, 3]
    to_camera = camera - center
    if np.linalg.norm(to_camera) < 1e-9:
        return {**result, "status": "unresolved", "reason": "Camera coincides with the estimated front plane"}
    facing = float(normal @ unit(to_camera))
    if abs(facing) < config["minimum_drawer_view_normal_cosine"]:
        return {**result, "status": "unresolved", "reason": "Front normal sign is ambiguous from a grazing view"}
    if facing < 0:
        normal = -normal
    return {**result, "status": "estimated", "normal_world": normal.tolist(),
            "view_normal_cosine": abs(facing),
            "normal_sign_policy": "Orient the selected visible front normal toward the camera",
            "normal_sign_semantically_verified": False}


def compile_drawer(blanks, payload, depth, calibration, config):
    """Compile RGB-D proposals; no simulator joint or object identity is input."""
    validate_drawer_blanks(blanks)
    config = _geometry_config(config)
    points = world_points(depth, calibration["intrinsic"], calibration["camera_to_world"])
    depth = np.asarray(depth, dtype=float)
    landmarks = {key: lift(blanks[key], depth, points, config) for key in DRAWER_POINT_FIELDS}
    front = _front_plane(landmarks, depth, points, calibration, config)
    axis = {"status": "unresolved", "reason": "Selected drawer-front plane is unresolved"}
    if front["status"] == "estimated":
        axis = {
            "status": "estimated", "axis_world": front["normal_world"],
            "direction_policy": "Assume sliding direction is the visible front normal toward the camera",
            "canonical_joint_axis_verified": False, "drawer_front_binding_verified": False,
            "source": "RGB-selected visible front plane + calibrated depth; front-normal heuristic",
            "frame": "world", "units": "unit direction",
        }
    relations, unresolved = [], []
    selected_pull = (blanks["joint_type"] == "prismatic" and blanks["motion"] == "pull"
                     and blanks["path_constraint"] == "prismatic_pull")
    if selected_pull and axis["status"] == "estimated" and landmarks["handle_uv"]["status"] == "measured":
        relations.append({
            "id": "drawer_handle_pull_line", "type": "point_on_line", "scope": "path", "stage": "pull",
            "point_local_m": [0., 0., 0.],
            "line_point_world_m": landmarks["handle_uv"]["position_m"],
            "axis_world": axis["axis_world"], "minimum_travel_m": 0., "maximum_travel_m": None,
            "frame": "world", "units": "meters", "equality_residual_dimension": 2,
            "equality_tolerance_m": config["drawer_line_tolerance_m"],
            "point_semantics": "Abstract handle-tracking point; robot TCP and grasp offset are not bound",
            "semantic_handle_verified": False, "semantic_joint_axis_verified": False,
            "robot_joint_binding_status": "unresolved", "robot_tcp_transform_status": "unresolved",
            "attachment_transform_status": "unresolved", "joint_limits_status": "unresolved",
            "estimate_status": "candidate", "planner_ready": False,
        })
    elif not selected_pull:
        unresolved.append("VLM choices do not select the required prismatic pull relation")
    else:
        unresolved.append("Pull line geometry unresolved: handle or drawer-front axis unavailable")

    extension = payload.get("requested_extension_m")
    if extension is not None and (
        isinstance(extension, bool) or not isinstance(extension, (int, float))
        or not np.isfinite(extension) or extension <= 0
    ):
        raise ValueError("Explicit requested extension must be finite and positive")
    opening_goal = {"status": "unresolved", "requested_extension_m": extension}
    if extension is None:
        opening_goal["reason"] = "The instruction to open supplies no metric extension or stopping threshold"
    elif relations:
        opening_goal.update(
            status="estimated", source="Explicit requested extension + observed handle/axis proposal",
            target_handle_world_m=(np.asarray(relations[0]["line_point_world_m"])
                                   + extension * np.asarray(axis["axis_world"])).tolist(),
            joint_limits_verified=False,
        )
    else:
        opening_goal["reason"] = "Requested extension is known, but pull geometry is unresolved"
    orientation = {
        "type": "fixed_orientation", "scope": "path", "stage": "pull", "status": "unresolved",
        "reference": "Orientation at verified handle attachment",
        "reason": "Handle grasp pose and rigid grasp attachment have not been estimated or verified",
    }
    unresolved.append(orientation["reason"])
    if opening_goal["status"] == "unresolved":
        unresolved.append(opening_goal["reason"])
    stages = [
        {"id": "approach", "action": "Approach the selected handle", "path_constraint_ids": []},
        {"id": "grasp", "action": "Establish and verify handle attachment", "path_constraint_ids": []},
        {"id": "pull", "action": "Pull while holding the handle", "path_constraint_ids": [item["id"] for item in relations]},
        {"id": "release", "action": "Release after the opening goal is verified", "path_constraint_ids": []},
    ]
    return {
        "schema_version": "rgbd-drawer-1.0", "task_type": "open_middle_drawer",
        "landmarks": landmarks, "drawer_front_plane": front, "drawer_axis": axis,
        "relations": relations, "path_constraints": relations, "stages": stages,
        "orientation_policy": orientation, "opening_goal": opening_goal,
        "grasp": {"status": "unresolved", "position_proposal_m": landmarks["handle_uv"].get("position_m"),
                  "reason": "Handle landmark alone does not determine gripper orientation or attachment"},
        "unresolved": unresolved, "geometry_source": "calibrated sensor depth",
        "provided_inventory": False, "joint_space_compiled": False,
        "planner_ready": False, "robot_execution_tested": False,
    }


def evaluate_line_constraint(spec, position, rotation):
    """Two line equalities and optional forward/maximum travel inequalities."""
    if spec["type"] != "point_on_line":
        raise ValueError("Expected a point_on_line relation")
    position, rotation = check_pose(position, rotation)
    local = np.asarray(spec["point_local_m"], dtype=float)
    origin = np.asarray(spec["line_point_world_m"], dtype=float)
    if local.shape != (3,) or origin.shape != (3,) or not np.isfinite([local, origin]).all():
        raise ValueError("Invalid point-on-line parameters")
    axis = unit(spec["axis_world"])
    basis = perpendicular_basis(axis)
    delta = position + rotation @ local - origin
    derivative = np.column_stack([np.eye(3), -rotation @ skew(local)])
    equality, jacobian = basis.T @ delta, basis.T @ derivative
    inequalities, inequality_jacobians = [], []
    lower, upper = spec.get("minimum_travel_m"), spec.get("maximum_travel_m")
    for bound in (lower, upper):
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float)) or not np.isfinite(bound)):
            raise ValueError("Travel bounds must be finite numbers or null")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("Minimum travel cannot exceed maximum travel")
    if lower is not None:
        inequalities.append(lower - axis @ delta)
        inequality_jacobians.append(-axis @ derivative)
    if upper is not None:
        inequalities.append(axis @ delta - upper)
        inequality_jacobians.append(axis @ derivative)
    return {"equality": equality, "equality_jacobian": jacobian,
            "inequality": np.asarray(inequalities),
            "inequality_jacobian": np.asarray(inequality_jacobians).reshape(-1, 6)}
