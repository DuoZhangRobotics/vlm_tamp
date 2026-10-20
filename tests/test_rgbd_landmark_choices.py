import copy
import json

import numpy as np
import pytest

from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_axis_constraints import compile_independent_axis
from vlm_tamp.rgbd_landmark_choices import (
    ID_FIELDS, blank_schema, candidate_payload, resolve_blanks, sdk_arguments,
    validate_blanks, validate_candidates,
)
from vlm_tamp.rgbd_numericals import POINT_FIELDS, pixel_index, template


def candidates():
    return [
        {"id": 0, "pixel_xy": [0, 0], "predicted_region_id": 1,
         "position_world_m": [1., 2., 3.], "offline_owner": "never send me"},
        {"id": 1, "pixel_xy": [12, 6], "predicted_region_id": 2},
        {"id": 2, "pixel_xy": [6, 3], "predicted_region_id": 2},
    ]


def blanks():
    return {**{field: None for field in ID_FIELDS}, "grasp_mode": "unresolved", "path_constraint": "none"}


def test_original_pixel_centers_roundtrip_on_non_square_image_and_borders():
    data = candidates()
    selected = {**blanks(), "source_id": 0, "grasp_id": 2, "target_id": 1}
    before = copy.deepcopy(selected)
    resolved = resolve_blanks(selected, data, (7, 13))
    assert selected == before
    assert pixel_index(resolved["source_uv"], (7, 13)) == (0, 0)
    assert pixel_index(resolved["grasp_uv"], (7, 13)) == (6, 3)
    assert pixel_index(resolved["target_uv"], (7, 13)) == (12, 6)
    assert resolved["target_uv"] == [(12 + .5)/13, (6 + .5)/7]
    assert resolved["closing_a_uv"] is None


def test_empty_candidates_allow_only_null_and_preserve_abstention():
    schema = blank_schema([])
    assert all(schema["properties"][field] == {"type": "null"} for field in ID_FIELDS)
    selected = blanks()
    assert validate_blanks(selected, []) == selected
    resolved = resolve_blanks(selected, [], (10, 20))
    assert all(resolved[field] is None for field in POINT_FIELDS)
    with pytest.raises(ValueError):
        validate_blanks({**selected, "source_id": 0}, [])


@pytest.mark.parametrize("value", [True, False, 3, -1, 1.0, "1", [1]])
def test_invalid_or_unavailable_selections_are_rejected(value):
    with pytest.raises(ValueError):
        validate_blanks({**blanks(), "source_id": value}, candidates())


@pytest.mark.parametrize("data", [
    [{"id": 0, "pixel_xy": [0, 0]}, {"id": 0, "pixel_xy": [1, 1]}],
    [{"id": 1, "pixel_xy": [0, 0]}],
    [{"id": True, "pixel_xy": [0, 0]}],
    [{"id": 0, "pixel_xy": [np.nan, 0]}],
    [{"id": 0, "pixel_xy": [0.5, 0]}],
    [{"id": 0, "pixel_xy": [0, True]}],
    [{"id": 0, "pixel_xy": [-1, 0]}],
    [{"id": 0, "pixel_xy": [0, 0], "predicted_region_id": 0}],
])
def test_invalid_candidate_ids_pixels_and_background_regions_are_rejected(data):
    with pytest.raises(ValueError):
        validate_candidates(data, (7, 13))


def test_schema_requires_every_field_and_exact_available_ids():
    schema = blank_schema(candidates())
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(blanks())
    assert schema["properties"]["source_id"]["anyOf"] == [
        {"type": "integer", "minimum": 0, "maximum": 2}, {"type": "null"},
    ]
    with pytest.raises(ValueError):
        validate_blanks({**blanks(), "inventory": []}, candidates())
    with pytest.raises(ValueError):
        validate_blanks({key: value for key, value in blanks().items() if key != "source_id"}, candidates())
    with pytest.raises(ValueError):
        validate_blanks({**blanks(), "grasp_mode": "sideways"}, candidates())
    with pytest.raises(ValueError):
        validate_blanks({**blanks(), "path_constraint": "unknown"}, candidates())
    with pytest.raises(ValueError):
        validate_blanks({**blanks(), "grasp_mode": ["top_down"]}, candidates())
    with pytest.raises(ValueError):
        resolve_blanks(blanks(), candidates(), (6, 13))


def test_request_exposes_only_instruction_descriptions_ids_and_two_rgb_images():
    config = json.loads((ROOT / "configs/rgbd_numeric_trial.json").read_text())
    payload = candidate_payload("put the bowl on the plate", candidates(), (7, 13), upright=True)
    assert payload["available_candidate_ids"] == [0, 1, 2]
    assert "upright" in payload["instruction"]
    assert "position_world_m" not in json.dumps(payload)
    assert "offline_owner" not in json.dumps(payload)
    assert "predicted_region_id" not in json.dumps(payload)
    args = sdk_arguments(config, payload, b"original png", b"numbered png", "Select IDs")
    assert [item["type"] for item in args["input"][0]["content"]] == [
        "input_text", "input_image", "input_image",
    ]
    assert args["store"] is False
    assert not any(key in args for key in ("tools", "previous_response_id", "conversation"))
    assert args["text"]["format"]["strict"] is True
    for name, value in (("inventory", ["bowl_1"]), ("world_points", [[1, 2, 3]]),
                        ("reference_masks", "mask.png")):
        with pytest.raises(ValueError):
            sdk_arguments(config, {**payload, name: value}, b"rgb", b"numbered", "Select IDs")


def test_selected_depth_edge_is_rejected_by_existing_compiler_without_snapping():
    config = json.loads((ROOT / "configs/rgbd_numeric_trial.json").read_text())["geometry"]
    depth = np.ones((20, 30))
    depth[:, 15:] = 2.
    data = [{"id": 0, "pixel_xy": [15, 10], "predicted_region_id": 1}]
    selected = {**blanks(), "source_id": 0, "grasp_id": 0}
    resolved = resolve_blanks(selected, data, depth.shape)
    compiled = compile_independent_axis(
        resolved, template("put the bowl on the plate"), depth,
        {"intrinsic": np.eye(3), "camera_to_world": np.eye(4)}, config,
    )
    measured = compiled["landmarks"]["grasp_uv"]
    assert measured["pixel_xy"] == [15, 10]
    assert measured["status"] == "unresolved"
    assert "position_m" not in measured
    assert "Depth discontinuity" in measured["reason"]
    assert compiled["grasp"]["status"] == "unresolved"


def test_same_candidate_can_fill_source_and_grasp_without_inventing_missing_axis():
    selected = {**blanks(), "source_id": 2, "grasp_id": 2}
    resolved = resolve_blanks(selected, candidates(), (7, 13))
    assert resolved["source_uv"] == resolved["grasp_uv"]
    assert resolved["axis_low_uv"] is None and resolved["axis_high_uv"] is None
