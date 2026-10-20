import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from vlm_tamp.rgb_evaluation import evaluate_rgb, make_reference
from vlm_tamp.rgb_role_choices import compile_bindings
from vlm_tamp.rgb_role_points import build_point_request, compile_points, point_schema, validate_points


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "gray").save(stream, format="PNG")
    return stream.getvalue()


def test_direct_rgb_transport_never_calls_segmentation_or_accepts_inventory(monkeypatch):
    from vlm_tamp import rgb_object_choices, rgb_role_choices
    from vlm_tamp.rgb_role_point_trial import request_arguments

    def forbidden(*args, **kwargs):
        raise AssertionError("Segmentation/candidate builder was called")

    monkeypatch.setattr(rgb_role_choices, "object_request", forbidden)
    monkeypatch.setattr(rgb_object_choices, "object_bundle", forbidden)
    config = {
        "model": "gpt-5.4-mini-2026-03-17",
        "image_detail": "high",
        "reasoning_effort": "medium",
        "max_output_tokens": 3000,
    }
    request = build_point_request("put the apple in the bowl", image_bytes(), "Locate roles")
    args = request_arguments(config, request)
    assert set(request.payload) == {"instruction", "roles"}
    assert [item["type"] for item in args["input"][0]["content"]] == ["input_text", "input_image"]
    assert not any(key in args for key in ("tools", "previous_response_id", "conversation"))
    assert len(compile_points({"slot_0": [0.2, 0.3], "slot_1": [0.7, 0.8]},
                              request.payload["instruction"]).goals) == 1
    with pytest.raises(ValueError):
        request_arguments(config, replace(request, payload={**request.payload, "inventory": ["apple"]}))
    with pytest.raises(ValueError):
        request_arguments(config, replace(request, payload={**request.payload, "roles": {"slot_0": "oracle"}}))


def test_point_compilation_preserves_existing_multiple_goals_and_drawer_parts():
    instruction = "put the black bowl in the bottom drawer of the cabinet and close it"
    points = {"slot_0": [0.2, 0.4], "slot_1": [0.7, 0.3]}
    direct = compile_points(points, instruction)
    existing = compile_bindings(
        {"slot_0": 1, "slot_1": 2}, instruction,
        [{"choice": 1, "point": points["slot_0"]}, {"choice": 2, "point": points["slot_1"]}],
    )
    assert direct.model_dump() == existing.model_dump()
    assert {goal.relation for goal in direct.goals} == {"In", "Close"}
    assert any(entity.selector == "bottom_drawer" for entity in direct.entities)


@pytest.mark.parametrize("payload", [
    {}, {"slot_0": [0.2]}, {"slot_0": [0.2, 0.3, 0.4]}, {"slot_0": [True, 0.3]},
    {"slot_0": [float("nan"), 0.3]}, {"slot_0": [float("inf"), 0.3]},
    {"slot_0": [-0.01, 0.3]}, {"slot_0": [0.2, 1.01]}, {"slot_0": ["0.2", 0.3]},
    {"slot_0": [0.2, 0.3], "extra": [0.4, 0.5]},
])
def test_invalid_coordinates_and_undeclared_roles_are_rejected(payload):
    with pytest.raises(ValueError):
        validate_points(payload, "turn on the stove")


def test_unknown_point_is_valid_abstention_and_not_a_successful_goal():
    instruction = "turn on the stove"
    assert validate_points({"slot_0": None}, instruction) == {"slot_0": None}
    assert point_schema(instruction)["properties"]["slot_0"]["anyOf"][1] == {"type": "null"}
    with pytest.raises(ValueError):
        compile_points({"slot_0": None}, instruction)


def test_same_grader_requires_correct_instance_and_visible_pixel(tmp_path):
    visual = {}
    for identifier, left, right in (("stove_1", 2, 9), ("stove_2", 19, 27)):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, left:right] = 255
        Image.fromarray(mask).save(tmp_path / (identifier + ".png"))
        visual[identifier] = {"kind": "object", "mask": identifier + ".png"}
    record = {
        "model_inputs": {"instruction": "turn on the stove"},
        "reference": {
            "entities": [{"identifier": name, "category": "stove", "kind": "object"} for name in visual],
            "visual": visual,
            "goal_predicates": [{"name": "Turnon", "arguments": ["stove_1"]}],
        },
    }
    reference = make_reference(record)
    for point, matched in (([0.15, 0.5], True), ([0.7, 0.5], False), ([0.5, 0.5], False)):
        prediction = compile_points({"slot_0": point}, record["model_inputs"]["instruction"])
        metric = evaluate_rgb(record, prediction, reference, tmp_path)
        assert metric["semantic_goal"]["exact_match"] is matched
