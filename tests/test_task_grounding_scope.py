import copy
import json

import pytest
from PIL import Image

from vlm_tamp import task_grounding_scope as scope
from vlm_tamp.drawer_constraints import DRAWER_POINT_FIELDS
from vlm_tamp.experiment import BudgetExceeded, Ledger
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_numericals import pixel_index


def controls():
    return json.loads((ROOT / "configs/task_grounding_scope.json").read_text())


def methods():
    return {"joint_type": "prismatic", "motion": "pull", "path_constraint": "prismatic_pull"}


def numbered_choices(value=0):
    return {**{key: value for key in scope.ID_FIELDS}, **methods()}


def candidates():
    return [
        {"id": 0, "pixel_xy": [0, 0], "predicted_region_id": 1,
         "world_xyz_m": [1., 2., 3.], "reference_identity": "secret drawer"},
        {"id": 1, "pixel_xy": [511, 255], "predicted_region_id": 2,
         "world_xyz_m": [4., 5., 6.], "reference_identity": "secret handle"},
    ]


def test_candidate_ids_resolve_unchanged_original_pixel_centers_including_zero():
    chosen = numbered_choices(0)
    chosen.update(handle_id=1, face_c_id=None)
    result = scope.resolve_choices(chosen, "dinov3_candidates", candidates(), (512, 512))
    assert result["source_uv"] == [.5 / 512, .5 / 512]
    assert result["handle_uv"] == [511.5 / 512, 255.5 / 512]
    assert result["face_c_uv"] is None
    assert pixel_index(result["source_uv"], (512, 512)) == (0, 0)
    assert pixel_index(result["handle_uv"], (512, 512)) == (511, 255)
    assert set(result) == set(DRAWER_POINT_FIELDS + scope.METHOD_FIELDS)
    assert not any("xyz" in key or "identity" in key for key in result)


def test_direct_pixels_are_preserved_without_snapping_and_input_is_not_mutated():
    chosen = {**{key: [.123456, .654321] for key in DRAWER_POINT_FIELDS}, **methods()}
    chosen["face_c_uv"] = None
    before = copy.deepcopy(chosen)
    result = scope.resolve_choices(chosen, "direct_rgb", None, (512, 512))
    assert result == chosen and chosen == before
    result["source_uv"][0] = .9
    assert chosen == before


@pytest.mark.parametrize("bad", [-1, 2, True, .0, "0", [0], {}])
def test_bad_candidate_selection_is_rejected_without_rescue(bad):
    with pytest.raises(ValueError):
        scope.resolve_choices({**numbered_choices(), "handle_id": bad},
                              "dinov3_candidates", candidates(), (512, 512))


def test_empty_candidate_set_requires_abstention():
    schema = scope.blank_schema("dinov3_candidates", [])
    assert schema["properties"]["handle_id"] == {"type": "null"}
    result = scope.resolve_choices(numbered_choices(None), "dinov3_candidates", [], (512, 512))
    assert all(result[key] is None for key in DRAWER_POINT_FIELDS)
    with pytest.raises(ValueError):
        scope.resolve_choices(numbered_choices(), "dinov3_candidates", [], (512, 512))


def test_schema_declares_only_current_drawer_landmarks_and_methods():
    direct = scope.blank_schema("direct_rgb")
    discrete = scope.blank_schema("dinov3_candidates", candidates())
    assert set(direct["required"]) == set(DRAWER_POINT_FIELDS + scope.METHOD_FIELDS)
    assert set(discrete["required"]) == set(scope.ID_FIELDS + scope.METHOD_FIELDS)
    assert not direct["additionalProperties"] and not discrete["additionalProperties"]
    assert discrete["properties"]["handle_id"]["anyOf"][0] == {
        "type": "integer", "minimum": 0, "maximum": 1,
    }
    with pytest.raises(ValueError):
        scope.blank_schema("provided_inventory", candidates())


@pytest.fixture
def sensor_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(scope, "ROOT", tmp_path)
    config = controls()
    case = {"id": "test", "example_id": "scene", "condition": "direct_rgb"}
    capture = tmp_path / config["dataset"] / case["example_id"]
    capture.mkdir(parents=True)
    Image.new("RGB", (512, 512), (20, 30, 40)).save(capture / "rgb.png")
    original = (capture / "rgb.png").read_bytes()
    candidate_dir = tmp_path / config["candidate_dataset"] / case["example_id"]
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "original.png").write_bytes(original)
    Image.new("RGB", (1536, 1536), (50, 60, 70)).save(candidate_dir / "numbered.png")
    (candidate_dir / "candidates.json").write_text(json.dumps(candidates()))
    record = {"model_inputs": {"instruction": "open the middle drawer of the cabinet", "image": "rgb.png"},
              "reference": {"joint_limits": [.0, .3], "instance_inventory": ["secret drawer"]}}
    return config, case, record, capture, candidate_dir


@pytest.mark.parametrize("condition,image_count", [("direct_rgb", 1), ("dinov3_candidates", 2)])
def test_api_inputs_contain_rgb_and_instruction_without_depth_or_reference_metadata(sensor_inputs, condition, image_count):
    config, case, record, _, _ = sensor_inputs
    case["condition"] = condition
    points, original, numbered, shape, payload = scope.inputs(config, case, record)
    assert shape == (512, 512)
    assert set(payload) == {"instruction", "task_type", "blanks"} | (
        {"available_candidate_ids"} if condition == "dinov3_candidates" else set()
    )
    args = scope.sdk_arguments(config, case, payload, original, numbered, "Fill visible drawer blanks", points)
    text = json.dumps(args)
    assert all(secret not in text for secret in (
        "secret drawer", "secret handle", "world_xyz_m", "reference_identity", "joint_limits",
        "instance_inventory", "camera_to_world", "depth_meters",
    ))
    assert len(args["input"][0]["content"]) == image_count + 1
    assert args["reasoning"] == {"effort": "none"}
    assert args["model"] == "gpt-6-luna" and args["max_output_tokens"] == 1000
    assert args["store"] is False
    assert not any(key in args for key in ("tools", "previous_response_id", "conversation"))


@pytest.mark.parametrize("forbidden", ["inventory", "depth", "joint_axis", "reference"])
def test_api_rejects_additional_payload_fields(sensor_inputs, forbidden):
    config, case, record, _, _ = sensor_inputs
    points, original, numbered, _, payload = scope.inputs(config, case, record)
    with pytest.raises(ValueError):
        scope.sdk_arguments(config, case, {**payload, forbidden: "oracle"}, original, numbered, "prompt", points)


def test_model_input_inventory_is_rejected_before_building_requests(sensor_inputs):
    config, case, record, _, _ = sensor_inputs
    record["model_inputs"]["inventory"] = ["drawer"]
    with pytest.raises(ValueError):
        scope.inputs(config, case, record)


def test_candidate_original_image_must_match_capture_bytes(sensor_inputs):
    config, case, record, _, candidate_dir = sensor_inputs
    case["condition"] = "dinov3_candidates"
    (candidate_dir / "original.png").write_bytes(b"altered image")
    with pytest.raises(ValueError):
        scope.inputs(config, case, record)


def test_pick_place_derivation_reuses_eight_saved_cases_without_mutating_sources():
    paths = [ROOT / "results/rgbd-dino-luna-report/summary.json",
             ROOT / "results/rgbd-dino-luna-comparison/manifest.json"]
    before = {str(path): scope.file_sha(path) for path in paths}
    rows = scope.derive_pick_place()
    assert len(rows) == 8
    assert all(row["new_api_calls"] == 0 and row["reused_prediction"] for row in rows)
    assert all("upright" not in row["source_case_id"] for row in rows)
    for row in rows:
        numeric = row["numericals"]
        assert numeric["task_type"] == "pick_place" and numeric["path_constraints"] == []
        assert numeric["relations"] == numeric["goals"]
        assert all(relation["scope"] == "goal" for relation in numeric["relations"])
        assert "object_axis" not in numeric
        assert not any(key.startswith("axis_") for key in numeric["landmarks"])
        assert not any(key.startswith("axis_") or key == "path_constraint" for key in row["choices"])
        assert not any(key.startswith("axis_") or key == "path_constraint" for key in row["blanks"])
        assert all(stage["path_constraint_ids"] == [] for stage in numeric["stages"])
        assert not row["metrics"]["path_constraints_required"]
        assert row["metrics"]["path_constraints_emitted"] == 0
    rows[0]["numericals"]["path_constraints"].append({"type": "invented"})
    assert scope.derive_pick_place()[0]["numericals"]["path_constraints"] == []
    assert before == {str(path): scope.file_sha(path) for path in paths}


@pytest.fixture
def budget_config(tmp_path, monkeypatch):
    config = controls()
    monkeypatch.setattr(scope, "ROOT", tmp_path)
    prior_dir = tmp_path / config["pick_place_source_trial"]
    prior_dir.mkdir(parents=True)
    (prior_dir / "manifest.json").write_text(json.dumps({
        "ledger": {"estimated_usage_cost_usd": config["prior_usage_cost_usd"]},
    }))
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    return config, config_file


def test_authorized_budget_includes_previous_luna_usage(budget_config):
    config, config_file = budget_config
    loaded = scope.load_config(config_file)
    assert loaded["overall_budget_usd"] == 2
    assert loaded["max_budget_usd"] + loaded["prior_usage_cost_usd"] == 2
    ledger = Ledger(loaded)
    ledger.actual_cost = loaded["max_budget_usd"] - .001
    ledger.admit(.001)
    with pytest.raises(BudgetExceeded):
        ledger.admit(.00101)
    assert config["prior_usage_cost_usd"] > 0


@pytest.mark.parametrize("change", [
    {"overall_budget_usd": 2.01}, {"max_budget_usd": 2.},
    {"prior_usage_cost_usd": 0., "max_budget_usd": 2.},
    {"overall_budget_usd": True}, {"max_budget_usd": float("nan")},
    {"prior_usage_cost_usd": -1.}, {"requested_extension_m": .1},
    {"max_requests": 9}, {"max_retries": 1}, {"model": "gpt-5.4-mini"},
])
def test_unauthorized_budget_extension_or_transport_changes_are_rejected(budget_config, change):
    config, config_file = budget_config
    config.update(change)
    config_file.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        scope.load_config(config_file)


def test_reserved_cost_uses_cache_write_ceiling_for_all_input_tokens():
    config = controls()
    args = {"instructions": "Visible drawer selection", "input": [{"content": [{"text": "task"}]}],
            "text": {"format": {"schema": {"type": "object"}}}}
    encoded = len((args["instructions"] + "task" + json.dumps({"type": "object"})).encode())
    expected = ((encoded + config["image_token_allowance"] + 512) * .125 + 1000 * .50) / 1e6
    assert scope.request_cost_reservation(config, args) == pytest.approx(expected)
    ledger = Ledger(config)
    for _ in range(8):
        ledger.admit(scope.request_cost_reservation(config, args))
    with pytest.raises(BudgetExceeded):
        ledger.admit(scope.request_cost_reservation(config, args))
