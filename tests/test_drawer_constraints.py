import copy
import json

import numpy as np
import pytest

from vlm_tamp.drawer_constraints import (
    DRAWER_GEOMETRY_DEFAULTS, DRAWER_POINT_FIELDS, compile_drawer,
    evaluate_line_constraint, validate_drawer_blanks,
)
from vlm_tamp.pose_constraints import perpendicular_basis, skew, unit


def rotation(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    matrix = skew(vector / angle)
    return np.eye(3) + np.sin(angle) * matrix + (1 - np.cos(angle)) * matrix @ matrix


def scene():
    return (
        {"source_uv": [.5, .5], "handle_uv": [.5, .5], "face_a_uv": [.2, .2],
         "face_b_uv": [.8, .2], "face_c_uv": [.2, .8], "joint_type": "prismatic",
         "motion": "pull", "path_constraint": "prismatic_pull"},
        {"instruction": "open the middle drawer of the cabinet"},
        np.ones((100, 100)),
        {"intrinsic": [[100., 0, 50], [0, 100., 50], [0, 0, 1]],
         "camera_to_world": np.eye(4).tolist()},
        dict(DRAWER_GEOMETRY_DEFAULTS),
    )


def test_drawer_pull_line_is_stage_scoped_and_robot_binding_stays_unresolved():
    result = compile_drawer(*scene())
    assert result["drawer_axis"]["status"] == "estimated"
    assert np.allclose(result["drawer_axis"]["axis_world"], [0, 0, -1])
    assert not result["drawer_axis"]["canonical_joint_axis_verified"]
    assert result["path_constraints"] == result["relations"]
    relation, = result["relations"]
    assert relation["type"] == "point_on_line" and relation["stage"] == "pull"
    assert relation["robot_joint_binding_status"] == "unresolved"
    assert relation["attachment_transform_status"] == "unresolved"
    assert relation["point_semantics"].startswith("Abstract handle-tracking point")
    assert [stage["path_constraint_ids"] for stage in result["stages"]] == [
        [], [], ["drawer_handle_pull_line"], [],
    ]
    assert result["orientation_policy"]["status"] == "unresolved"
    assert "rotation_matrix" not in result["orientation_policy"]
    assert result["opening_goal"]["status"] == "unresolved"
    assert result["opening_goal"]["requested_extension_m"] is None
    assert not result["planner_ready"] and not result["joint_space_compiled"]
    assert not result["robot_execution_tested"] and not result["provided_inventory"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("key,value", [
    ("joint_type", "revolute"), ("joint_type", "unresolved"),
    ("motion", "push"), ("motion", "unresolved"),
    ("path_constraint", "none"), ("path_constraint", "unresolved"),
])
def test_abstention_or_incompatible_joint_does_not_generate_slider_relation(key, value):
    arguments = list(scene())
    arguments[0] = {**arguments[0], key: value}
    result = compile_drawer(*arguments)
    assert result["drawer_axis"]["status"] == "estimated"
    assert result["relations"] == []
    assert result["unresolved"]


def test_all_unknown_pixels_remain_unknown_without_geometry_rescue():
    arguments = list(scene())
    arguments[0] = {**arguments[0], **{key: None for key in DRAWER_POINT_FIELDS}}
    result = compile_drawer(*arguments)
    assert result["drawer_axis"]["status"] == "unresolved"
    assert result["relations"] == []
    assert all(item["status"] == "unresolved" for item in result["landmarks"].values())


def test_depth_discontinuity_preserves_handle_pixel_and_prevents_line_compilation():
    arguments = list(scene())
    arguments[2][:, 50:52] = 1.2
    result = compile_drawer(*arguments)
    assert result["drawer_front_plane"]["status"] == "estimated"
    handle = result["landmarks"]["handle_uv"]
    assert handle["status"] == "unresolved" and handle["pixel_xy"] == [50, 50]
    assert "position_m" not in handle
    assert result["relations"] == []


@pytest.mark.parametrize("face_c", [[.2, .2], [.5, .2], [.8001, .2001]])
def test_duplicate_or_collinear_front_points_are_rejected(face_c):
    arguments = list(scene())
    arguments[0] = {**arguments[0], "face_c_uv": face_c}
    result = compile_drawer(*arguments)
    assert result["drawer_front_plane"]["status"] == "unresolved"
    assert result["relations"] == []


def test_nonplanar_patches_are_rejected_even_when_three_centers_define_a_plane():
    arguments = list(scene())
    arguments[2][70:91, 10:31] = 1.3
    result = compile_drawer(*arguments)
    assert all(result["landmarks"][key]["status"] == "measured" for key in (
        "face_a_uv", "face_b_uv", "face_c_uv",
    ))
    assert result["drawer_front_plane"]["status"] == "unresolved"
    assert result["drawer_front_plane"]["plane_rms_m"] > arguments[4]["maximum_plane_rms_m"]
    assert result["relations"] == []


def test_grazing_view_does_not_silently_choose_a_normal_sign():
    arguments = list(scene())
    arguments[3]["intrinsic"][0][2] = 50000
    result = compile_drawer(*arguments)
    assert result["drawer_front_plane"]["status"] == "unresolved"
    assert "grazing" in result["drawer_front_plane"]["reason"]
    assert result["relations"] == []


def test_explicit_opening_distance_is_separate_from_unknown_joint_limits():
    arguments = list(scene())
    arguments[1] = {**arguments[1], "requested_extension_m": .12}
    result = compile_drawer(*arguments)
    goal = result["opening_goal"]
    assert goal["status"] == "estimated" and goal["requested_extension_m"] == .12
    assert np.allclose(goal["target_handle_world_m"], [0, 0, .88])
    assert not goal["joint_limits_verified"]
    assert result["relations"][0]["maximum_travel_m"] is None
    assert not result["planner_ready"]


@pytest.mark.parametrize("extension", [0, -.1, True, float("nan"), float("inf")])
def test_invalid_requested_extension_is_rejected(extension):
    arguments = list(scene())
    arguments[1] = {**arguments[1], "requested_extension_m": extension}
    with pytest.raises(ValueError):
        compile_drawer(*arguments)


@pytest.mark.parametrize("value", [[True, .2], [float("nan"), .2], [.2], [-.1, .3], [.2, 1.1]])
def test_invalid_uv_blanks_are_rejected(value):
    with pytest.raises(ValueError):
        validate_drawer_blanks({**scene()[0], "source_uv": value})


def test_blank_contract_rejects_unexpected_inventory_and_wrong_methods():
    with pytest.raises(ValueError):
        validate_drawer_blanks({**scene()[0], "inventory": ["middle drawer"]})
    with pytest.raises(ValueError):
        validate_drawer_blanks({**scene()[0], "motion": "rotate"})


@pytest.mark.parametrize("field", ["joint_type", "motion", "path_constraint"])
@pytest.mark.parametrize("value", [[], {}, True, 1, None])
def test_method_wrong_types_are_rejected_with_value_error(field, value):
    with pytest.raises(ValueError):
        validate_drawer_blanks({**scene()[0], field: value})


@pytest.mark.parametrize("configuration", [
    {"depth_patch_radius_pixels": True}, {"plane_patch_radius_pixels": -1},
    {"maximum_plane_rms_m": 0}, {"minimum_drawer_triangle_sine": 1},
    {"minimum_drawer_view_normal_cosine": float("nan")},
])
def test_invalid_geometry_tolerances_are_rejected(configuration):
    arguments = list(scene())
    arguments[4].update(configuration)
    with pytest.raises(ValueError):
        compile_drawer(*arguments)


def test_line_jacobian_matches_independent_pose_finite_differences_with_tcp_offset():
    spec = {"type": "point_on_line", "point_local_m": [.03, -.04, .02],
            "line_point_world_m": [.2, .3, .8], "axis_world": [1, 2, 3],
            "minimum_travel_m": 0., "maximum_travel_m": .4}
    rng = np.random.default_rng(95)
    for _ in range(8):
        position, attitude = rng.normal(size=3), rotation(rng.normal(size=3))
        result = evaluate_line_constraint(spec, position, attitude)
        assert np.linalg.matrix_rank(result["equality_jacobian"]) == 2
        for key in ("equality", "inequality"):
            numeric = np.empty((len(result[key]), 6))
            for col in range(6):
                delta = np.eye(6)[col] * 1e-6
                plus = evaluate_line_constraint(spec, position + delta[:3], attitude @ rotation(delta[3:]))[key]
                minus = evaluate_line_constraint(spec, position - delta[:3], attitude @ rotation(-delta[3:]))[key]
                numeric[:, col] = (plus - minus) / 2e-6
            assert np.allclose(result[key + "_jacobian"], numeric, atol=1e-8)


def test_line_keeps_forward_translation_free_and_detects_lateral_backward_and_overtravel():
    spec = {"type": "point_on_line", "point_local_m": [0., 0., 0.],
            "line_point_world_m": [.2, .3, .8], "axis_world": [1, 2, 3],
            "minimum_travel_m": 0., "maximum_travel_m": .4}
    origin, axis = np.array(spec["line_point_world_m"]), unit(spec["axis_world"])
    forward = evaluate_line_constraint(spec, origin + .2 * axis, np.eye(3))
    assert np.allclose(forward["equality"], 0) and np.all(forward["inequality"] <= 0)
    perpendicular = perpendicular_basis(axis)[:, 0]
    lateral = evaluate_line_constraint(spec, origin + .2 * axis + .01 * perpendicular, np.eye(3))
    assert np.isclose(np.linalg.norm(lateral["equality"]), .01)
    backward = evaluate_line_constraint(spec, origin - .1 * axis, np.eye(3))
    assert backward["inequality"][0] > 0
    overtravel = evaluate_line_constraint(spec, origin + .5 * axis, np.eye(3))
    assert overtravel["inequality"][1] > 0


def test_unknown_travel_bounds_are_not_fabricated():
    spec = {"type": "point_on_line", "point_local_m": [0., 0., 0.],
            "line_point_world_m": [0., 0., 0.], "axis_world": [1, 0, 0]}
    result = evaluate_line_constraint(spec, [5, 0, 0], np.eye(3))
    assert result["inequality"].shape == (0,)
    assert result["inequality_jacobian"].shape == (0, 6)
    bad = copy.deepcopy(spec)
    bad.update(minimum_travel_m=.2, maximum_travel_m=.1)
    with pytest.raises(ValueError):
        evaluate_line_constraint(bad, [0, 0, 0], np.eye(3))
