"""Independent geometry and abstention checks for the two-anchor compiler."""

import json

import numpy as np
import pytest

from vlm_tamp.drawer_constraints import evaluate_line_constraint
from vlm_tamp.drawer_local_plane import (
    LOCAL_PLANE_DEFAULTS, compile_local_drawer, fit_local_drawer_plane,
    local_plane_config, validate_local_blanks,
)


def scene():
    return (
        {"source_uv": [.5, .5], "handle_uv": [.7, .5], "joint_type": "prismatic",
         "motion": "pull", "path_constraint": "prismatic_pull"},
        {"instruction": "open the middle drawer of the cabinet"},
        np.ones((100, 100)),
        {"intrinsic": [[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]],
         "camera_to_world": np.eye(4).tolist()},
        dict(LOCAL_PLANE_DEFAULTS),
    )


def fit(arguments):
    blanks, _, depth, calibration, config = arguments
    return fit_local_drawer_plane(blanks["source_uv"], blanks["handle_uv"], depth, calibration, config)


def test_tilted_plane_with_outliers_has_accurate_normal_and_deterministic_support():
    arguments = list(scene())
    yy, xx = np.mgrid[:100, :100]
    arguments[2] = 1 / (1 + .15 * ((xx - 50) / 100) + .2 * ((yy - 50) / 100))
    rng = np.random.default_rng(106)
    indices = rng.choice(10000, 800, replace=False)
    arguments[2].reshape(-1)[indices] += .015
    arguments[2][49:52, 49:52] = 1 / (1 + .15 * ((xx[49:52, 49:52] - 50) / 100)
                                     + .2 * ((yy[49:52, 49:52] - 50) / 100))
    front, support = fit(arguments)
    again, again_support = fit(arguments)
    reference = -np.array([.15, .2, 1.]) / np.linalg.norm([.15, .2, 1.])
    assert front["status"] == "estimated"
    assert np.allclose(front["normal_world"], reference, atol=1e-10)
    assert front == again and np.array_equal(support, again_support)
    assert front["plane_rms_m"] < 1e-10
    assert front["sample_count"] > 1000
    assert front["support_pixels"]["rejected_candidate_pixel_xy"]
    assert not front["semantic_drawer_front_verified"]
    assert not front["support_is_semantic_part_mask"]


def test_handle_protrusion_is_excluded_without_moving_source_or_handle():
    arguments = list(scene())
    yy, xx = np.mgrid[:100, :100]
    handle = (xx - 70) ** 2 + (yy - 50) ** 2 <= 5 ** 2
    arguments[2][handle] = .98
    front, support = fit(arguments)
    result = compile_local_drawer(*arguments)
    assert front["status"] == "estimated"
    assert np.allclose(front["normal_world"], [0, 0, -1])
    assert not support[handle].any()
    assert result["landmarks"]["source_uv"]["pixel_xy"] == [50, 50]
    assert result["landmarks"]["handle_uv"]["pixel_xy"] == [70, 50]
    assert result["landmarks"]["handle_uv"]["position_m"] == pytest.approx([.196, 0, .98])
    assert set(result["landmarks"]) == {"source_uv", "handle_uv"}
    json.dumps(result, allow_nan=False)


def test_point_on_line_is_only_for_pull_and_attachment_stays_unresolved():
    result = compile_local_drawer(*scene())
    relation, = result["relations"]
    assert result["path_constraints"] == result["relations"]
    assert relation["type"] == "point_on_line" and relation["stage"] == "pull"
    assert relation["attachment_transform_status"] == "unresolved"
    assert relation["semantic_handle_verified"] is False
    assert result["drawer_axis"]["canonical_joint_axis_verified"] is False
    origin, axis = np.array(relation["line_point_world_m"]), np.array(relation["axis_world"])
    residual = evaluate_line_constraint(relation, origin + .1 * axis, np.eye(3))
    assert np.allclose(residual["equality"], 0)
    assert np.all(residual["inequality"] <= 0)
    assert [stage["path_constraint_ids"] for stage in result["stages"]] == [[], [], [relation["id"]], []]
    assert result["grasp"]["status"] == result["orientation_policy"]["status"] == "unresolved"
    assert result["opening_goal"]["status"] == "unresolved"
    assert not result["planner_ready"] and not result["joint_space_compiled"]


def test_wrong_semantic_plane_can_fit_but_is_never_marked_correct():
    arguments = list(scene())
    # A perfectly planar wrong drawer or table is geometrically indistinguishable
    # without semantic evidence. Uniform depth must not produce an identity claim.
    arguments[2][:] = .8
    result = compile_local_drawer(*arguments)
    assert result["drawer_front_plane"]["status"] == "estimated"
    assert result["drawer_front_plane"]["semantic_drawer_front_verified"] is False
    assert result["drawer_front_plane"]["coplanar_drawer_identity_resolved"] is False
    assert result["drawer_axis"]["drawer_front_binding_verified"] is False
    assert result["relations"][0]["semantic_joint_axis_verified"] is False


def test_dominant_plane_inconsistent_with_seed_does_not_rescue_the_seed():
    arguments = list(scene())
    arguments[2][49:52, 49:52] = 1.015
    front, _ = fit(arguments)
    assert front["status"] == "unresolved"
    assert "inconsistent" in front["reason"]
    assert front["source_anchor"]["pixel_xy"] == [50, 50]
    assert front["source_plane_deviation_m"] > .003
    assert compile_local_drawer(*arguments)["relations"] == []


def test_substantial_competing_parallel_plane_causes_abstention():
    arguments = list(scene())
    arguments[2][:, 62:] = 1.01
    front, _ = fit(arguments)
    assert front["status"] == "unresolved"
    assert "competing" in front["reason"]
    assert front["competing_plane_check"]["status"] == "ambiguous"
    assert front["competing_plane_check"]["separation_m"] == pytest.approx(.01)


def test_collinear_pixel_support_does_not_define_a_plane():
    arguments = list(scene())
    arguments[2][:] = np.nan
    arguments[2][50, :] = 1
    arguments[0]["handle_uv"] = None
    arguments[4].update(depth_patch_radius_pixels=0, local_plane_minimum_inliers=3)
    front, support = fit(arguments)
    assert front["status"] == "unresolved"
    assert not support.any()
    json.dumps(front, allow_nan=False)


def test_narrow_strip_abstains_on_two_dimensional_extent_even_with_many_pixels():
    arguments = list(scene())
    arguments[2][:] = 1.2
    arguments[2][49:53, :] = 1
    arguments[4]["local_plane_minimum_inliers"] = 20
    front, _ = fit(arguments)
    assert front["status"] == "unresolved"
    assert "extent" in front["reason"]


@pytest.mark.parametrize("source", [None, [.5, .5]])
def test_unknown_or_invalid_seed_abstains_without_search(source):
    arguments = list(scene())
    arguments[0]["source_uv"] = source
    arguments[2][50, 50] = np.nan
    front, support = fit(arguments)
    assert front["status"] == "unresolved"
    assert not support.any()
    assert front["support_pixels"]["candidate_pixel_xy"] == []
    json.dumps(front, allow_nan=False)


@pytest.mark.parametrize("value", [0, .09, 5, np.inf, np.nan])
def test_invalid_source_depth_is_not_lifted_or_emitted_as_nonfinite_json(value):
    arguments = list(scene())
    arguments[2][50, 50] = value
    result = compile_local_drawer(*arguments)
    assert result["landmarks"]["source_uv"]["status"] == "unresolved"
    assert result["drawer_axis"]["status"] == "unresolved"
    assert result["relations"] == []
    json.dumps(result, allow_nan=False)


def test_depth_edges_are_excluded_and_edge_handle_remains_unresolved():
    arguments = list(scene())
    arguments[2][:, 70:] = 1.15
    front, support = fit(arguments)
    result = compile_local_drawer(*arguments)
    assert front["status"] == "estimated"
    assert not support[:, 69:71].any()
    assert front["support_pixels"]["depth_edge_pixel_xy"]
    assert result["landmarks"]["handle_uv"]["status"] == "unresolved"
    assert result["landmarks"]["handle_uv"]["pixel_xy"] == [70, 50]
    assert result["relations"] == []


def test_insufficient_roi_and_grazing_view_abstain():
    arguments = list(scene())
    arguments[4]["local_plane_radius_pixels"] = 3
    front, _ = fit(arguments)
    assert front["status"] == "unresolved" and "Insufficient" in front["reason"]
    arguments = list(scene())
    arguments[3]["intrinsic"][0][2] = 50000
    front, _ = fit(arguments)
    assert front["status"] == "unresolved" and "grazing" in front["reason"]


@pytest.mark.parametrize("key,value", [
    ("joint_type", "revolute"), ("joint_type", "unresolved"),
    ("motion", "push"), ("path_constraint", "none"),
])
def test_method_abstention_or_incompatibility_keeps_geometry_but_no_pull_relation(key, value):
    arguments = list(scene())
    arguments[0][key] = value
    result = compile_local_drawer(*arguments)
    assert result["drawer_front_plane"]["status"] == "estimated"
    assert result["relations"] == []


def test_explicit_goal_does_not_invent_joint_limits():
    arguments = list(scene())
    arguments[1]["requested_extension_m"] = .12
    result = compile_local_drawer(*arguments)
    assert result["opening_goal"]["status"] == "estimated"
    assert result["opening_goal"]["target_handle_world_m"] == pytest.approx([.2, 0, .88])
    assert result["opening_goal"]["joint_limits_verified"] is False
    assert result["relations"][0]["maximum_travel_m"] is None


@pytest.mark.parametrize("extension", [True, 0, -.1, np.nan, np.inf])
def test_invalid_goal_extension_is_rejected(extension):
    arguments = list(scene())
    arguments[1]["requested_extension_m"] = extension
    with pytest.raises(ValueError):
        compile_local_drawer(*arguments)


@pytest.mark.parametrize("value", [[True, .5], [np.nan, .5], [.5], [1.01, .5]])
def test_bad_uv_or_legacy_face_fields_are_rejected(value):
    with pytest.raises(ValueError):
        validate_local_blanks({**scene()[0], "source_uv": value})
    with pytest.raises(ValueError):
        validate_local_blanks({**scene()[0], "face_a_uv": [.1, .2]})


@pytest.mark.parametrize("update", [
    {"local_plane_radius_pixels": True}, {"local_plane_ransac_seed": -1},
    {"local_plane_ransac_iterations": 0}, {"local_plane_minimum_inliers": 2},
    {"local_plane_inlier_tolerance_m": np.nan}, {"local_plane_minimum_inlier_fraction": 1},
    {"minimum_depth_m": 5}, {"local_plane_ambiguity_angle_degrees": 90},
])
def test_bad_policy_is_rejected(update):
    with pytest.raises(ValueError):
        local_plane_config(update)


def test_invalid_calibration_and_depth_dimensions_are_rejected():
    arguments = list(scene())
    arguments[3]["camera_to_world"][3][0] = 1
    with pytest.raises(ValueError):
        fit(arguments)
    arguments = list(scene())
    arguments[2] = np.ones((100,))
    with pytest.raises(ValueError):
        fit(arguments)
