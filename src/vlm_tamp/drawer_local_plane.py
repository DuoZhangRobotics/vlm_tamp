"""Local RGB-D plane proposals using two unchanged RGB-selected anchors.

The source pixel anchors the surface fit. The handle pixel is excluded from
plane support because a handle can protrude from the front. Neither selected
pixel is moved, and no instance mask, joint, CAD model, or simulator pose enters
this module. A planar patch cannot identify a particular coplanar drawer.
"""

import math

import numpy as np

from .rgbd_numericals import pixel_index, world_points


LOCAL_POINT_FIELDS = ("source_uv", "handle_uv")
LOCAL_PLANE_DEFAULTS = {
    "depth_patch_radius_pixels": 1,
    "minimum_depth_m": 0.1,
    "maximum_depth_m": 5.0,
    "maximum_depth_jump_m": 0.03,
    "local_plane_radius_pixels": 32,
    "local_plane_handle_exclusion_radius_pixels": 8,
    "local_plane_depth_band_m": 0.06,
    "local_plane_depth_edge_m": 0.02,
    "local_plane_ransac_seed": 0,
    "local_plane_ransac_iterations": 128,
    "local_plane_inlier_tolerance_m": 0.003,
    "local_plane_minimum_inliers": 72,
    "local_plane_minimum_inlier_fraction": 0.6,
    "local_plane_minimum_minor_extent_m": 0.02,
    "local_plane_maximum_rms_m": 0.003,
    "local_plane_maximum_seed_deviation_m": 0.003,
    "local_plane_ambiguity_fraction": 0.2,
    "local_plane_ambiguity_angle_degrees": 10.0,
    "minimum_drawer_view_normal_cosine": 0.15,
    "drawer_line_tolerance_m": 0.01,
}


def validate_local_blanks(blanks):
    required = set(LOCAL_POINT_FIELDS) | {"joint_type", "motion", "path_constraint"}
    if not isinstance(blanks, dict) or set(blanks) != required:
        raise ValueError("Fill exactly the two landmark and three method blanks")
    for key in LOCAL_POINT_FIELDS:
        _validate_uv(blanks[key], key)
    choices = {
        "joint_type": {"prismatic", "revolute", "unresolved"},
        "motion": {"pull", "push", "unresolved"},
        "path_constraint": {"prismatic_pull", "none", "unresolved"},
    }
    for key, values in choices.items():
        if not isinstance(blanks[key], str) or blanks[key] not in values:
            raise ValueError("Invalid drawer method choice: " + key)
    return blanks


def _validate_uv(value, key):
    if value is not None and (
        not isinstance(value, list) or len(value) != 2 or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(item) or not 0 <= item <= 1 for item in value
        )
    ):
        raise ValueError("Invalid normalized landmark: " + key)


def local_plane_config(config):
    settings = {**LOCAL_PLANE_DEFAULTS, **config}
    nonnegative = ("depth_patch_radius_pixels", "local_plane_handle_exclusion_radius_pixels",
                   "local_plane_ransac_seed")
    positive_int = ("local_plane_radius_pixels", "local_plane_ransac_iterations",
                    "local_plane_minimum_inliers")
    for key in nonnegative + positive_int:
        value = settings[key]
        if type(value) is not int or value < (1 if key in positive_int else 0):
            raise ValueError("Invalid integer local-plane setting: " + key)
    for key in (
        "minimum_depth_m", "maximum_depth_m", "maximum_depth_jump_m",
        "local_plane_depth_band_m", "local_plane_depth_edge_m",
        "local_plane_inlier_tolerance_m", "local_plane_minimum_minor_extent_m",
        "local_plane_maximum_rms_m", "local_plane_maximum_seed_deviation_m",
        "local_plane_ambiguity_angle_degrees", "drawer_line_tolerance_m",
    ):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid positive local-plane setting: " + key)
    for key in ("local_plane_minimum_inlier_fraction", "local_plane_ambiguity_fraction",
                "minimum_drawer_view_normal_cosine"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("Invalid local-plane fraction: " + key)
    if settings["maximum_depth_m"] <= settings["minimum_depth_m"]:
        raise ValueError("Maximum depth must exceed minimum depth")
    if settings["local_plane_ambiguity_angle_degrees"] >= 90:
        raise ValueError("Plane ambiguity angle must be below 90 degrees")
    if settings["local_plane_minimum_inliers"] < 3:
        raise ValueError("At least three plane inliers are required")
    return settings


def _depth_and_points(depth, calibration, config):
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2 or not all(depth.shape):
        raise ValueError("Depth must be a nonempty two-dimensional image")
    valid = (np.isfinite(depth) & (depth >= config["minimum_depth_m"])
             & (depth < config["maximum_depth_m"]))
    # Invalid pixels are never used. A finite temporary depth merely permits
    # the shared calibrated backprojection to validate calibration globally.
    points = world_points(np.where(valid, depth, 1.0), calibration["intrinsic"],
                          calibration["camera_to_world"])
    return depth, points, valid


def _measure(uv, depth, points, valid, config):
    if uv is None:
        return {"status": "unresolved", "reason": "VLM abstained"}
    x, y = pixel_index(uv, depth.shape)
    result = {"uv": list(uv), "pixel_xy": [x, y], "frame": "world", "units": "meters",
              "semantic_identity_verified": False}
    if not valid[y, x]:
        return {**result, "status": "unresolved", "reason": "Invalid sensor depth at unchanged pixel"}
    radius = config["depth_patch_radius_pixels"]
    ys, xs = slice(max(0, y - radius), y + radius + 1), slice(max(0, x - radius), x + radius + 1)
    patch, valid_patch = depth[ys, xs], valid[ys, xs]
    result["depth_m"] = float(depth[y, x])
    if not valid_patch.all():
        return {**result, "status": "unresolved", "reason": "Invalid neighboring sensor depth"}
    spread = float(np.max(np.abs(patch - depth[y, x])))
    result["local_depth_spread_m"] = spread
    if spread > config["maximum_depth_jump_m"]:
        return {**result, "status": "unresolved", "reason": "Depth discontinuity at unchanged pixel"}
    return {**result, "status": "measured", "position_m": points[y, x].tolist(),
            "source": "unchanged RGB-selected pixel + measured depth"}


def _pixel_list(mask):
    rows, columns = np.nonzero(mask)
    return np.column_stack([columns, rows]).tolist()


def _depth_edges(depth, valid, threshold):
    edges = np.zeros(depth.shape, dtype=bool)
    horizontal = ((valid[:, 1:] != valid[:, :-1])
                  | (valid[:, 1:] & valid[:, :-1] & (np.abs(depth[:, 1:] - depth[:, :-1]) > threshold)))
    vertical = ((valid[1:] != valid[:-1])
                | (valid[1:] & valid[:-1] & (np.abs(depth[1:] - depth[:-1]) > threshold)))
    edges[:, 1:] |= horizontal
    edges[:, :-1] |= horizontal
    edges[1:] |= vertical
    edges[:-1] |= vertical
    return edges


def _refit(cloud, mask):
    center = cloud[mask].mean(axis=0)
    _, singular, vt = np.linalg.svd(cloud[mask] - center, full_matrices=False)
    return center, vt[-1], singular, vt


def _ransac(cloud, config, random_seed):
    if len(cloud) < 3:
        return None
    rng = np.random.default_rng(random_seed)
    tolerance = config["local_plane_inlier_tolerance_m"]
    best_mask, best_count, best_error = None, 0, float("inf")
    for _ in range(config["local_plane_ransac_iterations"]):
        a, b, c = cloud[rng.choice(len(cloud), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        distance = np.abs(np.einsum("ij,j->i", cloud - a, normal))
        mask = distance <= tolerance
        count = int(mask.sum())
        error = float(np.mean(distance[mask] ** 2))
        if count > best_count or (count == best_count and error < best_error):
            best_mask, best_count, best_error = mask, count, error
    if best_mask is None or best_count < 3:
        return None
    for _ in range(3):
        center, normal, singular, vt = _refit(cloud, best_mask)
        refined = np.abs(np.einsum("ij,j->i", cloud - center, normal)) <= tolerance
        if refined.sum() < 3:
            return None
        unchanged = np.array_equal(best_mask, refined)
        best_mask = refined
        if unchanged:
            break
    center, normal, singular, vt = _refit(cloud, best_mask)
    residuals = np.einsum("ij,j->i", cloud[best_mask] - center, normal)
    return {"center": center, "normal": normal, "singular": singular, "basis": vt,
            "inliers": best_mask, "rms": float(np.sqrt(np.mean(residuals ** 2)))}


def fit_local_drawer_plane(source_uv, handle_uv, depth, calibration, config):
    """Return (JSON plane diagnostics, observed inlier mask); no semantic rescue."""
    _validate_uv(source_uv, "source_uv")
    _validate_uv(handle_uv, "handle_uv")
    config = local_plane_config(config)
    depth, points, valid = _depth_and_points(depth, calibration, config)
    empty = np.zeros(depth.shape, dtype=bool)
    source = _measure(source_uv, depth, points, valid, config)
    base = {
        "status": "unresolved", "source_anchor": source,
        "source": "Robust local calibrated-depth plane around the unchanged RGB source pixel",
        "semantic_drawer_front_verified": False, "support_is_semantic_part_mask": False,
        "coplanar_drawer_identity_resolved": False,
        "policy": {key: config[key] for key in LOCAL_PLANE_DEFAULTS},
        "support_pixels": {key: [] for key in (
            "candidate_pixel_xy", "inlier_pixel_xy", "excluded_handle_pixel_xy",
            "depth_edge_pixel_xy", "rejected_candidate_pixel_xy",
        )},
    }
    if source["status"] != "measured":
        return {**base, "reason": "Source anchor is unresolved: " + source["reason"]}, empty
    x, y = source["pixel_xy"]
    radius = config["local_plane_radius_pixels"]
    roi = empty.copy()
    roi[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = True
    handle_mask = empty.copy()
    if handle_uv is not None:
        hx, hy = pixel_index(handle_uv, depth.shape)
        yy, xx = np.mgrid[:depth.shape[0], :depth.shape[1]]
        handle_mask = ((xx - hx) ** 2 + (yy - hy) ** 2
                       <= config["local_plane_handle_exclusion_radius_pixels"] ** 2)
    edge_mask = _depth_edges(depth, valid, config["local_plane_depth_edge_m"])
    selected = (roi & valid & (np.abs(depth - depth[y, x]) <= config["local_plane_depth_band_m"])
                & ~handle_mask & ~edge_mask)
    support = base["support_pixels"]
    support["candidate_pixel_xy"] = _pixel_list(selected)
    support["excluded_handle_pixel_xy"] = _pixel_list(roi & handle_mask)
    support["depth_edge_pixel_xy"] = _pixel_list(roi & edge_mask)
    base.update(roi_pixel_count=int(roi.sum()), candidate_sample_count=int(selected.sum()),
                seed_pixel_xy=[x, y], handle_exclusion_applied=handle_uv is not None)
    minimum = config["local_plane_minimum_inliers"]
    if selected.sum() < minimum:
        return {**base, "reason": "Insufficient valid local surface candidates"}, empty
    cloud = points[selected]
    model = _ransac(cloud, config, config["local_plane_ransac_seed"])
    if model is None:
        return {**base, "reason": "Observed local support is rank deficient or has no plane consensus"}, empty
    count = int(model["inliers"].sum())
    inlier_mask = empty.copy()
    inlier_mask[selected] = model["inliers"]
    support["inlier_pixel_xy"] = _pixel_list(inlier_mask)
    support["rejected_candidate_pixel_xy"] = _pixel_list(selected & ~inlier_mask)
    plane_coordinates = (cloud[model["inliers"]] - model["center"]) @ model["basis"][:2].T
    extents = np.quantile(plane_coordinates, .9, axis=0) - np.quantile(plane_coordinates, .1, axis=0)
    seed_deviation = float(abs((points[y, x] - model["center"]) @ model["normal"]))
    result = {
        **base, "normal_world": model["normal"].tolist(),
        "plane_point_world_m": model["center"].tolist(), "sample_count": count,
        "inlier_fraction": count / len(cloud), "plane_rms_m": model["rms"],
        "plane_extent_10_90_m": extents.tolist(), "minor_extent_m": float(min(extents)),
        "singular_values_m": model["singular"].tolist(),
        "source_plane_deviation_m": seed_deviation,
    }
    if count < minimum or result["inlier_fraction"] < config["local_plane_minimum_inlier_fraction"]:
        return {**result, "reason": "Insufficient robust local plane consensus"}, inlier_mask
    if model["singular"][1] < 1e-5 or min(extents) < config["local_plane_minimum_minor_extent_m"]:
        return {**result, "reason": "Local plane support has insufficient two-dimensional extent"}, inlier_mask
    if model["rms"] > config["local_plane_maximum_rms_m"]:
        return {**result, "reason": "Local plane RMS exceeds the fixed tolerance"}, inlier_mask
    if seed_deviation > config["local_plane_maximum_seed_deviation_m"]:
        return {**result, "reason": "Dominant local plane is inconsistent with the unchanged source anchor"}, inlier_mask
    remainder = cloud[~model["inliers"]]
    ambiguity_minimum = max(minimum, math.ceil(config["local_plane_ambiguity_fraction"] * len(cloud)))
    result["competing_plane_check"] = {"minimum_samples": ambiguity_minimum, "status": "no_significant_competing_plane"}
    if len(remainder) >= ambiguity_minimum:
        alternate = _ransac(remainder, config, config["local_plane_ransac_seed"] + 1)
        if alternate is not None and alternate["inliers"].sum() >= ambiguity_minimum:
            cosine = float(np.clip(abs(alternate["normal"] @ model["normal"]), 0, 1))
            angle = float(np.rad2deg(np.arccos(cosine)))
            separation = float(abs((alternate["center"] - model["center"]) @ model["normal"]))
            result["competing_plane_check"].update(
                sample_count=int(alternate["inliers"].sum()), angle_degrees=angle,
                separation_m=separation,
            )
            if (angle > config["local_plane_ambiguity_angle_degrees"]
                    or separation > 2 * config["local_plane_inlier_tolerance_m"]):
                result["competing_plane_check"]["status"] = "ambiguous"
                return {**result, "reason": "A substantial distinct competing local plane is present"}, inlier_mask
    camera = np.asarray(calibration["camera_to_world"], dtype=float)[:3, 3]
    to_camera = camera - model["center"]
    length = np.linalg.norm(to_camera)
    if length < 1e-9:
        return {**result, "reason": "Camera coincides with the estimated front plane"}, inlier_mask
    facing = float(model["normal"] @ (to_camera / length))
    if abs(facing) < config["minimum_drawer_view_normal_cosine"]:
        return {**result, "reason": "Front normal sign is ambiguous from a grazing view"}, inlier_mask
    normal = model["normal"] * (1 if facing > 0 else -1)
    return {
        **result, "status": "estimated", "normal_world": normal.tolist(),
        "view_normal_cosine": abs(facing),
        "normal_sign_policy": "Orient the observed local front normal toward the camera",
        "normal_sign_semantically_verified": False,
    }, inlier_mask


def compile_local_drawer(blanks, payload, depth, calibration, config):
    """Construct the same abstract pull relation with two RGB-selected anchors."""
    validate_local_blanks(blanks)
    config = local_plane_config(config)
    depth, points, valid = _depth_and_points(depth, calibration, config)
    landmarks = {key: _measure(blanks[key], depth, points, valid, config) for key in LOCAL_POINT_FIELDS}
    front, _ = fit_local_drawer_plane(blanks["source_uv"], blanks["handle_uv"], depth, calibration, config)
    support = front.pop("support_pixels")
    axis = {"status": "unresolved", "reason": "Observed local drawer-front plane is unresolved"}
    if front["status"] == "estimated":
        axis = {
            "status": "estimated", "axis_world": front["normal_world"],
            "direction_policy": "Assume sliding direction is the visible front normal toward the camera",
            "canonical_joint_axis_verified": False, "drawer_front_binding_verified": False,
            "source": "Unchanged RGB source seed + robust local calibrated-depth plane; front-normal heuristic",
            "frame": "world", "units": "unit direction",
        }
    relations, unresolved = [], []
    selected_pull = (blanks["joint_type"] == "prismatic" and blanks["motion"] == "pull"
                     and blanks["path_constraint"] == "prismatic_pull")
    if selected_pull and axis["status"] == "estimated" and landmarks["handle_uv"]["status"] == "measured":
        relations.append({
            "id": "drawer_handle_pull_line", "type": "point_on_line", "scope": "path", "stage": "pull",
            "point_local_m": [0., 0., 0.], "line_point_world_m": landmarks["handle_uv"]["position_m"],
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
        unresolved.append("Pull line geometry unresolved: handle or local drawer-front axis unavailable")
    extension = payload.get("requested_extension_m")
    if extension is not None and (
        isinstance(extension, bool) or not isinstance(extension, (int, float))
        or not math.isfinite(extension) or extension <= 0
    ):
        raise ValueError("Explicit requested extension must be finite and positive")
    goal = {"status": "unresolved", "requested_extension_m": extension}
    if extension is None:
        goal["reason"] = "The instruction to open supplies no metric extension or stopping threshold"
    elif relations:
        goal.update(status="estimated", source="Explicit requested extension + observed handle/axis proposal",
                    target_handle_world_m=(np.asarray(relations[0]["line_point_world_m"])
                                           + extension * np.asarray(axis["axis_world"])).tolist(),
                    joint_limits_verified=False)
    else:
        goal["reason"] = "Requested extension is known, but pull geometry is unresolved"
    orientation = {
        "type": "fixed_orientation", "scope": "path", "stage": "pull", "status": "unresolved",
        "reference": "Orientation at verified handle attachment",
        "reason": "Handle grasp pose and rigid grasp attachment have not been estimated or verified",
    }
    unresolved.append(orientation["reason"])
    if goal["status"] == "unresolved":
        unresolved.append(goal["reason"])
    stages = [
        {"id": "approach", "action": "Approach the selected handle", "path_constraint_ids": []},
        {"id": "grasp", "action": "Establish and verify handle attachment", "path_constraint_ids": []},
        {"id": "pull", "action": "Pull while holding the handle", "path_constraint_ids": [item["id"] for item in relations]},
        {"id": "release", "action": "Release after the opening goal is verified", "path_constraint_ids": []},
    ]
    return {
        "schema_version": "rgbd-drawer-local-plane-1.0", "task_type": "open_middle_drawer",
        "landmarks": landmarks, "drawer_front_plane": front, "drawer_axis": axis,
        "local_plane_support": support, "relations": relations, "path_constraints": relations,
        "stages": stages, "orientation_policy": orientation, "opening_goal": goal,
        "grasp": {"status": "unresolved", "position_proposal_m": landmarks["handle_uv"].get("position_m"),
                  "reason": "Handle landmark alone does not determine gripper orientation or attachment"},
        "unresolved": unresolved, "geometry_source": "calibrated sensor depth",
        "provided_inventory": False, "joint_space_compiled": False,
        "planner_ready": False, "robot_execution_tested": False,
    }
