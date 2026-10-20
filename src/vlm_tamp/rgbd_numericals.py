"""RGB landmark blanks + calibrated depth -> numerical proposals.

Inference and geometry functions have no record, instance mask, BDDL reference,
simulator identity, or object-pose input. Unknown landmarks remain unresolved.
"""

import json
import math

import numpy as np

from .pose_constraints import unit
from .rgb_role_choices import instruction_template
from .rgb_role_points import validate_points


POINT_FIELDS = (
    "source_uv", "grasp_uv", "closing_a_uv", "closing_b_uv",
    "axis_low_uv", "axis_high_uv", "target_uv",
)
UPRIGHT_SUFFIX = " while keeping the moved object upright throughout transport"


def template(base_instruction, upright=False):
    parsed = instruction_template(base_instruction)
    if len(parsed["goals"]) != 1 or parsed["goals"][0]["state"] != "on_top":
        raise ValueError("Numerical pilot supports a single rigid-object On placement")
    goal = parsed["goals"][0]
    if goal["object_part"] != "whole" or goal["target_part"] != "whole":
        raise ValueError("Part and articulated placements require a separate geometry contract")
    return {
        "instruction": base_instruction + (UPRIGHT_SUFFIX if upright else ""),
        "source_description": parsed["roles"][goal["object_slot"]],
        "target_description": parsed["roles"][goal["target_slot"]],
        "blanks": list(POINT_FIELDS) + ["grasp_mode", "path_constraint"],
        "coordinate_convention": "[x,y] normalized to [0,1], origin top-left",
    }


def blank_schema():
    point = {"anyOf": [{"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 2, "maxItems": 2}, {"type": "null"}]}
    properties = {key: point for key in POINT_FIELDS}
    properties.update(
        grasp_mode={"type": "string", "enum": ["top_down", "unresolved"]},
        path_constraint={"type": "string", "enum": ["none", "keep_upright", "unresolved"]},
    )
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def validate_blanks(blanks):
    if not isinstance(blanks, dict) or set(blanks) != set(blank_schema()["properties"]):
        raise ValueError("Fill exactly the numerical landmark and method blanks")
    for key in POINT_FIELDS:
        value = blanks[key]
        if value is not None and (not isinstance(value, list) or len(value) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in value
        )):
            raise ValueError("Invalid normalized landmark: " + key)
    if blanks["grasp_mode"] not in {"top_down", "unresolved"} or blanks["path_constraint"] not in {
        "none", "keep_upright", "unresolved"
    }:
        raise ValueError("Invalid method or constraint choice")
    return blanks


def semantic_points(blanks, base_instruction):
    parsed = instruction_template(base_instruction)
    goal = parsed["goals"][0]
    result = {goal["object_slot"]: blanks["source_uv"], goal["target_slot"]: blanks["target_uv"]}
    return validate_points(result, base_instruction)


def metric_depth(raw, near, far):
    raw = np.asarray(raw, dtype=float).squeeze()
    if not 0 < near < far or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise ValueError("Invalid rendered depth or near/far planes")
    return near / (1 - raw * (1 - near / far))


def world_points(depth, intrinsic, camera_to_world):
    depth = np.asarray(depth, dtype=float)
    intrinsic, transform = np.asarray(intrinsic, dtype=float), np.asarray(camera_to_world, dtype=float)
    if depth.ndim != 2 or intrinsic.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("Invalid depth/calibration dimensions")
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Depth must be positive and finite")
    if not np.isfinite(intrinsic).all() or not np.isfinite(transform).all():
        raise ValueError("Calibration must be finite")
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("Invalid homogeneous camera transform")
    yy, xx = np.mgrid[:depth.shape[0], :depth.shape[1]]
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(intrinsic).T
    return (rays * depth[..., None]) @ transform[:3, :3].T + transform[:3, 3]


def pixel_index(uv, shape):
    height, width = shape
    return min(width - 1, int(uv[0] * width)), min(height - 1, int(uv[1] * height))


def lift(uv, depth, points, config):
    if uv is None:
        return {"status": "unresolved", "reason": "VLM abstained"}
    x, y = pixel_index(uv, depth.shape)
    radius = config["depth_patch_radius_pixels"]
    patch = depth[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
    center = float(depth[y, x])
    spread = float(np.max(np.abs(patch - center)))
    result = {"uv": uv, "pixel_xy": [x, y], "depth_m": center,
              "local_depth_spread_m": spread, "frame": "world", "units": "meters"}
    if not np.isfinite(center) or not 0 < center < config["maximum_depth_m"]:
        return {**result, "status": "unresolved", "reason": "Invalid sensor depth"}
    # Never move an anchor to a more convenient surface or consult a mask.
    if spread > config["maximum_depth_jump_m"]:
        return {**result, "status": "unresolved", "reason": "Depth discontinuity at unchanged pixel"}
    return {**result, "status": "measured", "position_m": points[y, x].tolist(),
            "semantic_identity_verified": False, "source": "selected RGB pixel + measured depth"}


def fit_support(uv, depth, points, config):
    if uv is None:
        return {"status": "unresolved", "reason": "No target landmark"}
    x, y = pixel_index(uv, depth.shape)
    radius = config["plane_patch_radius_pixels"]
    ys, xs = slice(max(0, y-radius), y+radius+1), slice(max(0, x-radius), x+radius+1)
    z, cloud = depth[ys, xs].reshape(-1), points[ys, xs].reshape(-1, 3)
    cloud = cloud[np.abs(z-depth[y, x]) <= config["plane_depth_band_m"]]
    if len(cloud) < 12:
        return {"status": "unresolved", "reason": "Insufficient local surface samples"}
    center = cloud.mean(axis=0)
    _, singular, vt = np.linalg.svd(cloud-center, full_matrices=False)
    if singular[1] < 1e-5:
        return {"status": "unresolved", "reason": "Degenerate local surface"}
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    rms = float(np.sqrt(np.mean(((cloud-center) @ normal)**2)))
    tilt = float(np.rad2deg(np.arccos(np.clip(normal[2], -1, 1))))
    result = {"normal_world": normal.tolist(), "plane_point_world_m": center.tolist(),
              "plane_rms_m": rms, "sample_count": len(cloud), "tilt_degrees": tilt,
              "source": "local observed depth patch; no object model or mask"}
    if rms > config["maximum_plane_rms_m"] or tilt > config["maximum_support_tilt_degrees"]:
        return {**result, "status": "unresolved", "reason": "Patch is not a sufficiently planar support"}
    return {**result, "status": "estimated", "semantic_support_verified": False}


def compile_numericals(blanks, payload, depth, calibration, config):
    """Only sensor/calibration data and VLM blanks may enter this compiler."""
    validate_blanks(blanks)
    points = world_points(depth, calibration["intrinsic"], calibration["camera_to_world"])
    landmarks = {key: lift(blanks[key], depth, points, config) for key in POINT_FIELDS}
    support = fit_support(blanks["target_uv"], depth, points, config)
    grasp = {"status": "unresolved", "reason": "Missing geometric grasp anchors or mode"}
    measured = lambda key: landmarks[key]["status"] == "measured"
    if blanks["grasp_mode"] == "top_down" and all(measured(key) for key in (
        "grasp_uv", "closing_a_uv", "closing_b_uv"
    )):
        a = np.asarray(landmarks["closing_a_uv"]["position_m"])
        b = np.asarray(landmarks["closing_b_uv"]["position_m"])
        direction = b-a
        direction[2] = 0
        span = float(np.linalg.norm(direction))
        width = span + config["finger_clearance_m"]
        if span < config["minimum_closing_span_m"] or width > config["maximum_gripper_width_m"]:
            grasp = {"status": "unresolved", "reason": "Measured closing span exceeds primitive limits",
                     "closing_span_m": span, "required_opening_m": width}
        else:
            x_axis, z_axis = unit(direction), np.array([0., 0., -1.])
            rotation = np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])
            center = np.asarray(landmarks["grasp_uv"]["position_m"])
            grasp = {"status": "candidate", "position_m": center.tolist(),
                     "rotation_matrix": rotation.tolist(), "required_opening_m": width,
                     "closing_span_m": span,
                     "pregrasp_position_m": (center-z_axis*config["pregrasp_distance_m"]).tolist(),
                     "frame": "world", "pose_axes": "x finger-closing, z approach; right-handed",
                     "robot_tcp_transform_status": "unresolved", "ik_verified": False,
                     "collision_verified": False, "contact_stability_verified": False,
                     "source": "depth landmarks + configured top-down primitive"}
    axis = {"status": "unresolved", "reason": "Missing two observable object-axis landmarks"}
    if measured("axis_low_uv") and measured("axis_high_uv"):
        delta = np.asarray(landmarks["axis_high_uv"]["position_m"])-landmarks["axis_low_uv"]["position_m"]
        length = float(np.linalg.norm(delta))
        if length >= config["minimum_object_axis_length_m"]:
            axis = {"status": "estimated", "axis_world": unit(delta).tolist(),
                    "landmark_distance_m": length, "source": "depth surface chord selected bottom-to-top",
                    "canonical_object_axis_verified": False}
    relations, unresolved = [], []
    if support["status"] == "estimated" and measured("target_uv"):
        target = np.asarray(landmarks["target_uv"]["position_m"])
        normal = np.asarray(support["normal_world"])
        target = target + normal * config["placement_clearance_m"]
        relations.append({"id": "place_point", "type": "point_target", "scope": "goal", "stage": "place",
                          "point_local_m": [0., 0., 0.], "target_world_m": target.tolist(),
                          "frame": "world", "units": "meters", "equality_tolerance_m": config["point_tolerance_m"],
                          "point_semantics": "abstract selected placement point, not object or robot TCP origin",
                          "whole_object_placement_status": "unresolved: object-bottom/TCP offsets not estimated",
                          "clearance_source": "configured policy; not estimated by VLM",
                          "semantic_target_verified": False})
    else:
        unresolved.append("No numerical placement-point target: support surface unresolved")
    if blanks["path_constraint"] == "keep_upright":
        if grasp["status"] == "candidate" and axis["status"] == "estimated":
            rotation = np.asarray(grasp["rotation_matrix"])
            local = rotation.T @ np.asarray(axis["axis_world"])
            common = {"axis_local": local.tolist(), "axis_world": [0., 0., 1.],
                      "scope": "path", "stage": "transport", "frame": "world",
                      "attachment_assumption": "object rigidly attached to proposed grasp frame",
                      "required_by_instruction": UPRIGHT_SUFFIX.strip() in payload["instruction"],
                      "semantic_axis_verified": False}
            relations.extend([
                {**common, "id": "upright_manifold", "type": "axis_alignment",
                 "equality_residual_dimension": 2, "positive_axis_branch_inequality": True},
                {**common, "id": "upright_tolerance", "type": "axis_cone",
                 "max_angle_degrees": config["upright_tolerance_degrees"],
                 "tolerance_source": "configured policy; not inferred from RGB"},
            ])
        else:
            unresolved.append("Upright relation selected but grasp-frame/axis geometry unresolved")
    elif blanks["path_constraint"] == "unresolved":
        unresolved.append("VLM abstained on the path-constraint type")
    return {"schema_version": "rgbd-numerical-1.0", "landmarks": landmarks, "support": support,
            "grasp": grasp, "object_axis": axis, "relations": relations, "unresolved": unresolved,
            "geometry_source": "calibrated sensor depth", "provided_inventory": False,
            "joint_space_compiled": False, "robot_execution_tested": False}


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
