"""Offline-only numerical pilot checks. Simulator labels never rescue predictions."""

import numpy as np
from PIL import Image

from .dataset import safe_path
from .pose_constraints import evaluate_relation, perpendicular_basis, satisfied, skew, unit
from .rgb_evaluation import evaluate_rgb, make_reference
from .rgb_role_diagnostics import template_contract
from .rgb_role_points import compile_points
from .rgbd_numericals import POINT_FIELDS, pixel_index, semantic_points


def exp_rotation(vector):
    length = np.linalg.norm(vector)
    if length < 1e-12:
        return np.eye(3)
    k = skew(vector/length)
    return np.eye(3)+np.sin(length)*k+(1-np.cos(length))*(k@k)


def align_rotation(local, target):
    local, target = unit(local), unit(target)
    left = np.column_stack([local, perpendicular_basis(local)])
    right = np.column_stack([target, perpendicular_basis(target)])
    return right @ left.T


def relation_checks(spec):
    """Executable equations tested on synthetic pose states, not robot motions."""
    if spec["type"] == "point_target":
        r = np.eye(3)
        p = np.asarray(spec["target_world_m"])-np.asarray(spec["point_local_m"])
        positive = satisfied(spec, p, r)
        negative = not satisfied(spec, p+np.array([.03, 0, 0]), r,
                                 equality_tolerance=spec["equality_tolerance_m"])
    else:
        p = np.zeros(3)
        r = align_rotation(spec["axis_local"], spec["axis_world"])
        positive = satisfied(spec, p, r)
        sideways = perpendicular_basis(spec["axis_world"])[:, 0]
        negative = not satisfied(spec, p, exp_rotation(sideways*np.deg2rad(20)) @ r)
    trial_p = p+np.array([.013, -.021, .017])
    trial_r = r @ exp_rotation(np.array([.2, -.3, .1]))
    analytical = evaluate_relation(spec, trial_p, trial_r)
    errors = []
    for key in ("equality", "inequality"):
        numeric = np.empty((len(analytical[key]), 6))
        for col in range(6):
            delta = np.eye(6)[col]*1e-6
            plus = evaluate_relation(spec, trial_p+delta[:3], trial_r@exp_rotation(delta[3:]))[key]
            minus = evaluate_relation(spec, trial_p-delta[:3], trial_r@exp_rotation(-delta[3:]))[key]
            numeric[:, col] = (plus-minus)/2e-6
        if numeric.size:
            errors.append(float(np.max(np.abs(numeric-analytical[key+"_jacobian"]))))
    error = max(errors, default=0.)
    return {"id": spec["id"], "type": spec["type"], "positive_state_accepted": positive,
            "violating_state_rejected": negative, "maximum_jacobian_error": error,
            "passed": positive and negative and error < 1e-7,
            "scope": "formula self-consistency; does not verify semantic bindings or execution"}


def capture_checks(depth, cloud, calibration):
    intrinsic = np.asarray(calibration["intrinsic"])
    transform = np.asarray(calibration["camera_to_world"])
    camera = (cloud-transform[:3, 3]) @ transform[:3, :3]
    projected = camera @ intrinsic.T
    pixels = projected[..., :2]/projected[..., 2, None]
    yy, xx = np.mgrid[:depth.shape[0], :depth.shape[1]]
    error = float(np.max(np.abs(pixels-np.stack([xx, yy], axis=-1))))
    depth_error = float(np.max(np.abs(camera[..., 2]-depth)))
    return {"finite_depth": bool(np.isfinite(depth).all()), "depth_shape": list(depth.shape),
            "reprojection_maximum_error_pixels": error, "depth_roundtrip_maximum_error_m": depth_error,
            "passed": error < 1e-7 and depth_error < 1e-8,
            "scope": "calibration/inverse consistency; ideal simulated depth, not real-sensor accuracy"}


def grade_case(record, root, case, row):
    """Use held-back labels after inference, independently of numerical compile status."""
    contract = template_contract(record)
    if not contract["supported"]:
        raise ValueError("Selected scene lacks an independently aligned role reference")
    parsed_roles = list(contract["role_owners"])
    if len(parsed_roles) != 2:
        raise ValueError("Numerical pilot expects exactly two goal roles")
    source_owner, target_owner = [contract["role_owners"][key] for key in parsed_roles]
    visual = {key: value for key, value in record["reference"]["visual"].items()
              if value.get("kind", "object") == "object"}
    masks = {key: np.asarray(Image.open(safe_path(root, value["mask"]))) > 0
             for key, value in visual.items()}
    shape = next(iter(masks.values())).shape
    blanks, numeric = row["blanks"], row["numericals"]
    anchors = []
    for key in POINT_FIELDS:
        expected = target_owner if key == "target_uv" else source_owner
        uv = blanks[key] if blanks else None
        matches = []
        if uv is not None:
            x, y = pixel_index(uv, shape)
            matches = [name for name, mask in masks.items() if mask[y, x]]
        status = ("abstained" if uv is None else "correct_instance" if matches == [expected]
                  else "wrong_instance" if len(matches) == 1 else "ambiguous" if matches else "background")
        anchors.append({"field": key, "uv": uv, "expected_owner_offline_only": expected,
                        "mask_matches": matches, "diagnostic": status,
                        "depth_status": numeric["landmarks"][key]["status"] if numeric else "unresolved"})
    goal_match = False
    try:
        prediction = compile_points(semantic_points(blanks, record["model_inputs"]["instruction"]),
                                    record["model_inputs"]["instruction"])
        goal_match = evaluate_rgb(record, prediction, make_reference(record), root)["semantic_goal"]["exact_match"]
    except (ValueError, TypeError):
        pass
    by_field = {anchor["field"]: anchor for anchor in anchors}
    axis_error, support_error = None, None
    if numeric:
        geometry = record["reference"]["geometry"]
        if numeric["object_axis"]["status"] == "estimated" and source_owner in geometry:
            up = np.asarray(geometry[source_owner]["rotation_matrix"]).reshape(3, 3)[:, 2]
            measured = np.asarray(numeric["object_axis"]["axis_world"])
            axis_error = float(np.rad2deg(np.arccos(np.clip(up @ measured, -1, 1))))
        if numeric["support"]["status"] == "estimated" and target_owner in geometry:
            up = np.asarray(geometry[target_owner]["rotation_matrix"]).reshape(3, 3)[:, 2]
            normal = np.asarray(numeric["support"]["normal_world"])
            support_error = float(np.rad2deg(np.arccos(np.clip(abs(up @ normal), -1, 1))))
    candidate = bool(numeric and numeric["grasp"]["status"] == "candidate")
    return {"semantic_complete_goal": goal_match, "anchors": anchors,
            "grasp_candidate": candidate,
            "candidate_landmarks_on_correct_object": candidate and all(
                by_field[key]["diagnostic"] == "correct_instance" for key in (
                    "source_uv", "grasp_uv", "closing_a_uv", "closing_b_uv")),
            "path_choice_correct": bool(blanks and blanks["path_constraint"] == (
                "keep_upright" if case["upright"] else "none")),
            "object_axis_canonical_z_proxy_error_degrees": axis_error,
            "support_normal_canonical_z_proxy_error_degrees": support_error,
            "axis_reference_scope": "held-back object-frame z proxy, not human-reviewed part/affordance labels",
            "relation_checks": [relation_checks(spec) for spec in numeric["relations"]] if numeric else [],
            "grasp_success_evaluated": False, "robot_execution_evaluated": False}
