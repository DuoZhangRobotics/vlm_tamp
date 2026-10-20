import json

import numpy as np
import pytest

from vlm_tamp.libero import ROOT
from vlm_tamp.pose_constraints import evaluate_relation, satisfied, skew
from vlm_tamp.rgbd_numericals import (
    POINT_FIELDS, compile_numericals, metric_depth, template, validate_blanks, world_points,
)
from vlm_tamp.rgbd_numeric_trial import sdk_arguments


def rotation(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    k = skew(vector/angle)
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


@pytest.mark.parametrize("spec", [
    {"type": "point_target", "point_local_m": [.03, -.01, .02], "target_world_m": [.4, .2, .8]},
    {"type": "axis_alignment", "axis_local": [1, 2, 3], "axis_world": [0, 0, 1]},
    {"type": "axis_cone", "axis_local": [1, 2, 3], "axis_world": [0, 0, 1], "max_angle_degrees": 5},
])
def test_pose_jacobians_match_independent_finite_differences(spec):
    rng = np.random.default_rng(24)
    for _ in range(8):
        p, r = rng.normal(size=3), rotation(rng.normal(size=3))
        result = evaluate_relation(spec, p, r)
        for key in ("equality", "inequality"):
            numeric = np.empty((len(result[key]), 6))
            for col in range(6):
                delta = np.eye(6)[col]*1e-6
                plus = evaluate_relation(spec, p+delta[:3], r@rotation(delta[3:]))[key]
                minus = evaluate_relation(spec, p-delta[:3], r@rotation(-delta[3:]))[key]
                numeric[:, col] = (plus-minus)/2e-6
            assert np.allclose(result[key+"_jacobian"], numeric, atol=1e-8)


def test_upright_manifold_rejects_upside_down_and_retains_yaw_freedom():
    spec = {"type": "axis_alignment", "axis_local": [0, 0, 1], "axis_world": [0, 0, 1]}
    assert satisfied(spec, [0, 0, 0], rotation([0, 0, 1.4]))
    assert not satisfied(spec, [0, 0, 0], rotation([np.pi, 0, 0]))
    assert not satisfied(spec, [0, 0, 0], rotation([.1, 0, 0]))
    assert np.linalg.matrix_rank(evaluate_relation(spec, [0, 0, 0], np.eye(3))["equality_jacobian"]) == 2
    cone = {**spec, "type": "axis_cone", "max_angle_degrees": 5}
    assert satisfied(cone, [0, 0, 0], rotation([np.deg2rad(2), 0, 0]))
    assert not satisfied(cone, [0, 0, 0], rotation([np.deg2rad(10), 0, 0]))


def test_depth_conversion_and_world_frame_are_consistent():
    near, far = .1, 10
    assert np.allclose(metric_depth(np.array([[0., 1.]]), near, far), [[near, far]])
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    depth = np.full((3, 4), 2.)
    points = world_points(depth, np.eye(3), transform)
    assert np.allclose(points[1, 2], [5, 4, 5])
    with pytest.raises(ValueError):
        world_points(-depth, np.eye(3), transform)


def test_inference_payload_contains_only_rgb_and_instruction_blanks():
    config = json.loads((ROOT / "configs/rgbd_numeric_trial.json").read_text())
    payload = template("put the bowl on the plate", True)
    args = sdk_arguments(config, payload, b"test image", "Fill numerical blanks")
    assert "upright" in payload["instruction"]
    assert [item["type"] for item in args["input"][0]["content"]] == ["input_text", "input_image"]
    assert not any(key in args for key in ("tools", "previous_response_id", "conversation"))
    with pytest.raises(ValueError):
        sdk_arguments(config, {**payload, "inventory": ["bowl_1"]}, b"image", "Fill blanks")
    with pytest.raises(ValueError):
        template("put the bowl in the drawer")


def test_unknown_and_invalid_landmarks_are_not_filled_from_reference_geometry():
    blanks = {**{key: None for key in POINT_FIELDS}, "grasp_mode": "unresolved", "path_constraint": "none"}
    assert validate_blanks(blanks) == blanks
    config = json.loads((ROOT / "configs/rgbd_numeric_trial.json").read_text())["geometry"]
    result = compile_numericals(blanks, template("put the bowl on the plate"), np.ones((20, 20)),
                                {"intrinsic": np.eye(3), "camera_to_world": np.eye(4)}, config)
    assert result["grasp"]["status"] == "unresolved" and not result["relations"]
    for bad in ([float("nan"), .2], [True, .2], [-.1, .2], [.2, 1.1], [.2]):
        with pytest.raises(ValueError):
            validate_blanks({**blanks, "source_uv": bad})


def test_depth_edge_rejection_preserves_the_selected_pixel():
    from vlm_tamp.rgbd_numericals import lift
    config = json.loads((ROOT / "configs/rgbd_numeric_trial.json").read_text())["geometry"]
    depth = np.ones((20, 20))
    depth[:, 10:] = 2
    cloud = world_points(depth, np.eye(3), np.eye(4))
    estimate = lift([.5, .5], depth, cloud, config)
    assert estimate["status"] == "unresolved"
    assert estimate["pixel_xy"] == [10, 10]
    assert "position_m" not in estimate


def test_measured_geometry_produces_proposals_and_rejects_oversized_grasps():
    config = json.loads((ROOT / "configs/rgbd_numeric_trial.json").read_text())["geometry"]
    depth = np.ones((100, 100))
    k = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
    blanks = {"source_uv": [.5, .5], "grasp_uv": [.5, .5], "closing_a_uv": [.48, .5],
              "closing_b_uv": [.52, .5], "axis_low_uv": None, "axis_high_uv": None,
              "target_uv": [.7, .7], "grasp_mode": "top_down", "path_constraint": "keep_upright"}
    result = compile_numericals(blanks, template("put the bowl on the plate", True), depth,
                                {"intrinsic": k, "camera_to_world": np.eye(4)}, config)
    assert result["grasp"]["status"] == "candidate"
    assert not result["grasp"]["ik_verified"]
    assert np.isclose(np.linalg.det(result["grasp"]["rotation_matrix"]), 1)
    assert [rel["type"] for rel in result["relations"]] == ["point_target"]
    assert result["unresolved"]
    oversized = compile_numericals({**blanks, "closing_a_uv": [.4, .5], "closing_b_uv": [.6, .5]},
                                   template("put the bowl on the plate", True), depth,
                                   {"intrinsic": k, "camera_to_world": np.eye(4)}, config)
    assert oversized["grasp"]["status"] == "unresolved"


def test_object_axis_constraints_do_not_require_a_grasp_pose():
    from vlm_tamp.rgbd_axis_constraints import compile_independent_axis
    from vlm_tamp.rgbd_numeric_evaluation import relation_checks
    config = json.loads((ROOT / "configs/rgbd_numeric_trial.json").read_text())["geometry"]
    # Two depth surfaces differ in camera z, providing a nonzero observed axis.
    depth = np.ones((100, 100))
    depth[70:90, :] = 1.1
    k = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
    blanks = {key: None for key in POINT_FIELDS}
    blanks.update(source_uv=[.5, .5], target_uv=[.4, .4], axis_low_uv=[.5, .8], axis_high_uv=[.5, .5],
                  grasp_mode="unresolved", path_constraint="keep_upright")
    result = compile_independent_axis(blanks, template("put the bowl on the plate", True), depth,
                                      {"intrinsic": k, "camera_to_world": np.eye(4)}, config)
    assert result["grasp"]["status"] == "unresolved"
    relations = [spec for spec in result["relations"] if spec["scope"] == "path"]
    assert {spec["type"] for spec in relations} == {"axis_alignment", "axis_cone"}
    assert all(relation_checks(spec)["passed"] for spec in relations)
    assert all(spec["robot_joint_binding_status"] == "unresolved" for spec in relations)
