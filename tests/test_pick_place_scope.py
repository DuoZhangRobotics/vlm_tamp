import hashlib
import json

import numpy as np
import pytest

from vlm_tamp import pick_place_scope as scope
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_numericals import pixel_index


def scene():
    return (
        {"source_uv": [.5, .5], "grasp_uv": [.5, .5], "closing_a_uv": [.48, .5],
         "closing_b_uv": [.52, .5], "target_uv": [.7, .7], "grasp_mode": "top_down"},
        scope.template("put the bowl on the plate"), np.ones((100, 100)),
        {"intrinsic": [[100., 0, 50], [0, 100., 50], [0, 0, 1]], "camera_to_world": np.eye(4).tolist()},
        json.loads((ROOT / "configs/rgbd_numeric_trial.json").read_text())["geometry"],
    )


def candidates():
    return [{"id": 0, "pixel_xy": [0, 0], "world_xyz_m": [1., 2., 3.], "reference_identity": "secret"},
            {"id": 1, "pixel_xy": [99, 50], "world_xyz_m": [4., 5., 6.], "reference_identity": "hidden"}]


def config():
    return {"model": "gpt-6-luna", "reasoning_effort": "none", "image_detail": "high", "max_output_tokens": 1000}


@pytest.mark.parametrize("condition", scope.CONDITIONS)
def test_future_schema_contains_five_landmarks_and_grasp_only(condition):
    schema = scope.blank_schema(condition, candidates() if condition == "dinov3_candidates" else None)
    fields = scope.POINT_FIELDS if condition == "direct_rgb" else scope.ID_FIELDS
    assert set(schema["properties"]) == set(fields + ("grasp_mode",))
    assert len(schema["required"]) == 6 and not schema["additionalProperties"]
    assert not any("axis" in field or "constraint" in field for field in schema["properties"])
    payload = (scope.template("put the bowl on the plate") if condition == "direct_rgb"
               else scope.candidate_payload("put the bowl on the plate", candidates(), (100, 100)))
    assert payload["blanks"] == list(fields + ("grasp_mode",))
    assert payload["task_type"] == "pick_place"
    assert payload["source_description"] == "the bowl" and payload["target_description"] == "the plate"


@pytest.mark.parametrize("instruction", [
    "open the middle drawer of the cabinet", "put the bowl in the drawer",
    "put the bowl on the plate while keeping the moved object upright throughout transport",
    "put the cup on the table without spilling", "",
])
def test_articulated_inside_or_additional_transport_requirement_needs_another_contract(instruction):
    with pytest.raises(ValueError):
        scope.template(instruction)


def test_direct_pixels_preserve_exact_user_selected_coordinates():
    blanks = scene()[0]
    blanks["source_uv"] = [.1234567, .9876543]
    resolved = scope.resolve_choices(blanks, "direct_rgb", None, (100, 100))
    assert resolved == blanks
    resolved["source_uv"][0] = .1
    assert blanks["source_uv"] == [.1234567, .9876543]


def test_id_zero_and_image_edges_resolve_exact_original_pixel_centers():
    choices = {**{key: 0 for key in scope.ID_FIELDS}, "grasp_mode": "unresolved"}
    choices.update(target_id=1, closing_a_id=None)
    resolved = scope.resolve_choices(choices, "dinov3_candidates", candidates(), (100, 100))
    assert resolved["source_uv"] == [.005, .005]
    assert resolved["target_uv"] == [.995, .505]
    assert pixel_index(resolved["target_uv"], (100, 100)) == (99, 50)
    assert resolved["closing_a_uv"] is None
    assert set(resolved) == set(scope.POINT_FIELDS + ("grasp_mode",))


@pytest.mark.parametrize("bad", [-1, 2, True, .0, "0", [], {}])
def test_invalid_id_does_not_snap_to_a_candidate(bad):
    choices = {**{key: 0 for key in scope.ID_FIELDS}, "grasp_mode": "unresolved", "source_id": bad}
    with pytest.raises(ValueError):
        scope.resolve_choices(choices, "dinov3_candidates", candidates(), (100, 100))


def test_empty_candidate_set_requires_null_landmarks():
    choices = {**{key: None for key in scope.ID_FIELDS}, "grasp_mode": "unresolved"}
    resolved = scope.resolve_choices(choices, "dinov3_candidates", [], (100, 100))
    assert all(resolved[field] is None for field in scope.POINT_FIELDS)
    assert scope.blank_schema("dinov3_candidates", [])["properties"]["source_id"] == {"type": "null"}


@pytest.mark.parametrize("field", ["axis_low_uv", "axis_high_uv", "path_constraint", "inventory"])
def test_caller_cannot_reintroduce_axis_or_path_fields(field):
    with pytest.raises(ValueError):
        scope.validate_blanks({**scene()[0], field: None})
    with pytest.raises(ValueError):
        scope.compile_pick_place({**scene()[0], field: None}, *scene()[1:])


@pytest.mark.parametrize("value", [[True, .2], [float("nan"), .2], [-.1, .2], [.2, 1.1], [.2]])
def test_invalid_normalized_uvs_are_rejected(value):
    with pytest.raises(ValueError):
        scope.validate_blanks({**scene()[0], "source_uv": value})


@pytest.mark.parametrize("value", [[], {}, True, 1, None, "front_grasp"])
def test_invalid_grasp_modes_raise_value_error(value):
    with pytest.raises(ValueError):
        scope.validate_blanks({**scene()[0], "grasp_mode": value})


@pytest.mark.parametrize("condition,image_count", [("direct_rgb", 1), ("dinov3_candidates", 2)])
def test_future_sdk_requests_have_no_axis_or_path_blanks_and_no_sensor_or_reference_metadata(condition, image_count):
    payload = (scope.template("put the bowl on the plate") if condition == "direct_rgb"
               else scope.candidate_payload("put the bowl on the plate", candidates(), (100, 100)))
    args = scope.sdk_arguments(config(), payload, b"original RGB", "Choose visible pick-place landmarks",
                               condition=condition, candidates=candidates() if image_count == 2 else None,
                               numbered_image=b"numbered RGB" if image_count == 2 else None)
    assert len(args["input"][0]["content"]) == image_count + 1
    text = json.dumps(args)
    assert all(value not in text for value in (
        "axis_low", "axis_high", "path_constraint", "world_xyz_m", "reference_identity",
        "secret", "hidden", "depth_meters", "camera_to_world", "inventory",
    ))
    assert args["text"]["format"]["strict"] and args["store"] is False
    assert not any(key in args for key in ("tools", "previous_response_id", "conversation"))


@pytest.mark.parametrize("extra", ["path_constraint", "inventory", "depth", "joint_axis"])
def test_future_sdk_rejects_extra_payload_inputs(extra):
    payload = {**scene()[1], extra: "oracle"}
    with pytest.raises(ValueError):
        scope.sdk_arguments(config(), payload, b"RGB", "prompt")


def test_future_sdk_rejects_changed_descriptions_and_wrong_image_method():
    payload = {**scene()[1], "source_description": "inventory-provided bowl_1"}
    with pytest.raises(ValueError):
        scope.sdk_arguments(config(), payload, b"RGB", "prompt")
    with pytest.raises(ValueError):
        scope.sdk_arguments(config(), scene()[1], b"RGB", "prompt", numbered_image=b"unexpected")


def test_compilation_keeps_placement_goal_without_extra_path_constraint_or_axis():
    result = scope.compile_pick_place(*scene())
    assert result["task_type"] == "pick_place" and result["path_constraints"] == []
    assert result["path_constraints_required"] is False
    assert "object_axis" not in result and set(result["landmarks"]) == set(scope.POINT_FIELDS)
    assert len(result["goals"]) == 1 and result["relations"] == result["goals"]
    goal = result["goals"][0]
    assert goal["type"] == "point_target" and goal["scope"] == "goal"
    assert "unresolved" in goal["whole_object_placement_status"]
    assert not goal["semantic_target_verified"] and not result["whole_object_placement_verified"]
    assert result["grasp"]["status"] == "candidate"
    assert not result["grasp"]["ik_verified"] and not result["grasp"]["collision_verified"]
    assert not result["grasp"]["contact_stability_verified"]
    assert all(stage["path_constraint_ids"] == [] for stage in result["stages"])
    assert not result["planner_ready"] and not result["joint_space_compiled"]
    assert not result["robot_execution_tested"]
    json.dumps(result, allow_nan=False)


def test_oversized_grasp_is_unresolved_without_destroying_placement_goal_or_adding_constraints():
    args = list(scene())
    args[0].update(closing_a_uv=[.4, .5], closing_b_uv=[.6, .5])
    result = scope.compile_pick_place(*args)
    assert result["grasp"]["status"] == "unresolved"
    assert result["grasp"]["required_opening_m"] > args[-1]["maximum_gripper_width_m"]
    assert len(result["goals"]) == 1 and result["path_constraints"] == []


def test_all_unknown_landmarks_remain_unresolved_and_emit_no_fake_goal():
    args = list(scene())
    args[0] = {**{field: None for field in scope.POINT_FIELDS}, "grasp_mode": "unresolved"}
    result = scope.compile_pick_place(*args)
    assert result["grasp"]["status"] == "unresolved"
    assert result["goals"] == result["relations"] == result["path_constraints"] == []
    assert all(point["status"] == "unresolved" for point in result["landmarks"].values())


def test_candidate_declaration_can_compile_after_local_pixel_resolution():
    args = list(scene())
    args[1] = scope.candidate_payload("put the bowl on the plate", candidates(), (100, 100))
    result = scope.compile_pick_place(*args)
    assert result["path_constraints"] == [] and len(result["goals"]) == 1


def test_compiler_rejects_changed_blank_declaration_and_extra_payload_constraints():
    args = list(scene())
    args[1]["blanks"].append("path_constraint")
    with pytest.raises(ValueError):
        scope.compile_pick_place(*args)
    args = list(scene())
    args[1]["path_constraint"] = "keep_upright"
    with pytest.raises(ValueError):
        scope.compile_pick_place(*args)


def test_old_numerical_compiler_hash_is_unchanged_from_frozen_parent():
    protocol = json.loads((ROOT / "configs/frozen_task_grounding_scope.json").read_text())
    path = "src/vlm_tamp/rgbd_numericals.py"
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == protocol["implementation_hashes"][path]
