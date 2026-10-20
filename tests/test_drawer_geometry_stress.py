"""Two-anchor API isolation, completion, budget, and geometry boundaries."""

import base64
import copy
import json

import numpy as np
import pytest

from vlm_tamp import drawer_geometry_stress as stress
from vlm_tamp.dataset import read_records
from vlm_tamp.drawer_local_plane import LOCAL_PLANE_DEFAULTS
from vlm_tamp.experiment import BudgetExceeded, Ledger, input_price_ceiling
from vlm_tamp.libero import ROOT


@pytest.fixture
def controls():
    return json.loads((ROOT / "configs/drawer_geometry_stress.json").read_text())


def response(blanks):
    return {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps(blanks)},
    ]}]}


def test_four_real_capture_requests_only_send_two_rgb_points_and_three_choices(controls):
    records, _ = read_records(ROOT / controls["dataset"])
    records = {record["example_id"]: record for record in records}
    reservations = []
    assert tuple(case["id"] for case in controls["cases"]) == stress.CASE_IDS
    for case in controls["cases"]:
        record = records[case["example_id"]]
        args = stress.request_arguments(controls, case, record)
        assert args["model"] == "gpt-6-luna"
        assert args["reasoning"] == {"effort": "high"}
        assert args["max_output_tokens"] == 6000
        assert args["service_tier"] == "default" and args["store"] is False
        assert set(args) == {"model", "instructions", "input", "text", "reasoning", "service_tier",
                             "max_output_tokens", "store"}
        assert len(args["input"]) == 1 and args["input"][0]["role"] == "user"
        content = args["input"][0]["content"]
        assert len(content) == 2 and content[0]["type"] == "input_text"
        payload = json.loads(content[0]["text"])
        assert payload == {"instruction": record["model_inputs"]["instruction"],
                           "task_type": "open_middle_drawer", "blanks": list(stress.FIELDS)}
        assert list(payload["blanks"][:2]) == ["source_uv", "handle_uv"]
        assert not any(case["id"] in text for text in (content[0]["text"], args["instructions"]))
        assert content[1]["type"] == "input_image" and content[1]["detail"] == "high"
        image = base64.b64decode(content[1]["image_url"].split(",", 1)[1])
        assert image == (ROOT / controls["dataset"] / case["example_id"] / "rgb.png").read_bytes()
        assert "depth" not in payload and "reference" not in payload and "inventory" not in payload
        schema = args["text"]["format"]["schema"]
        assert args["text"]["format"]["strict"] is True
        assert set(schema["properties"]) == set(stress.FIELDS)
        assert len(schema["properties"]) == 5 and not schema["additionalProperties"]
        reservations.append(stress.request_cost_reservation(controls, args))
    assert len(reservations) == 4
    assert sum(reservations) + controls["prior_usage_cost_usd"] <= controls["overall_budget_usd"]
    assert input_price_ceiling(controls) == .125


def test_model_input_contract_rejects_reference_inventory(controls):
    case = controls["cases"][0]
    record = {"model_inputs": {"instruction": "open the middle drawer", "image": "rgb.png",
                               "reference": {"joint_type": "prismatic"}}}
    with pytest.raises(ValueError, match="Only instruction and image"):
        stress.request_arguments(controls, case, record)


@pytest.mark.parametrize("key,value", [
    ("reasoning_effort", "none"), ("max_output_tokens", 1000), ("max_output_tokens", True),
    ("max_requests", 5), ("max_retries", 1), ("max_retries", False), ("model", "gpt-6-astra"),
])
def test_controls_reject_unapproved_inference_changes(controls, tmp_path, key, value):
    controls[key] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(controls))
    with pytest.raises(ValueError, match="approved two-point"):
        stress.load_config(path)


@pytest.mark.parametrize("section,key,value", [
    ("local_plane", "local_plane_radius_pixels", 31),
    ("local_plane", "local_plane_ransac_seed", 1),
    ("geometry", "maximum_depth_jump_m", .1),
    ("evaluation", "minimum_support_purity_diagnostic", .5),
])
def test_parent_local_policy_remains_immutable(controls, tmp_path, section, key, value):
    controls[section][key] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(controls))
    with pytest.raises(ValueError, match="Frozen local-plane policy changed"):
        stress.load_config(path)


@pytest.mark.parametrize("key,value", [
    ("max_budget_usd", float("nan")), ("max_budget_usd", float("inf")), ("max_budget_usd", True),
    ("prior_usage_cost_usd", float("nan")), ("prior_usage_cost_usd", True),
    ("overall_budget_usd", float("nan")), ("overall_budget_usd", True),
    ("max_budget_usd", 2.), ("max_budget_usd", 0),
])
def test_budget_rejects_nonfinite_boolean_and_unaccounted_available_funds(controls, tmp_path, key, value):
    controls[key] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(controls))
    with pytest.raises(ValueError, match="two-dollar"):
        stress.load_config(path)


def test_valid_budget_contains_all_prior_spending_and_unknown_usage_cannot_be_reused(controls):
    actual = stress.load_config(ROOT / "configs/drawer_geometry_stress.json")
    assert actual["prior_usage_cost_usd"] == pytest.approx(.152321)
    assert actual["prior_usage_cost_usd"] + actual["max_budget_usd"] == pytest.approx(2.)
    ledger = Ledger(controls)
    ledger.actual_cost = controls["max_budget_usd"] - .01
    ledger.unknown_cost_reservation = .006
    with pytest.raises(BudgetExceeded):
        ledger.admit(.004001)
    assert ledger.attempts == 0
    ledger.admit(.004)
    assert ledger.attempts == 1


@pytest.mark.parametrize("change", ["order", "example", "task", "count"])
def test_predetermined_camera_and_state_cases_cannot_be_substituted(controls, tmp_path, change):
    if change == "order":
        controls["cases"] = list(reversed(controls["cases"]))
    elif change == "example":
        controls["cases"][0]["example_id"] = controls["cases"][1]["example_id"]
    elif change == "task":
        controls["cases"][0]["task_type"] = "pick_place"
    else:
        controls["cases"].pop()
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(controls))
    with pytest.raises(ValueError, match="four predetermined"):
        stress.load_config(path)


@pytest.mark.parametrize("status,details,expected", [
    ("incomplete", {"reason": "max_output_tokens"}, "response_truncated"),
    ("incomplete", {"reason": "content_filter"}, "invalid_output"),
    ("failed", None, "invalid_output"),
])
def test_incomplete_response_never_reaches_depth_fit_or_grading(controls, monkeypatch, status, details, expected):
    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete response reached geometry or held-back grading")

    monkeypatch.setattr(stress, "compile_local_drawer", forbidden)
    monkeypatch.setattr(stress, "grade_local_drawer", forbidden)
    monkeypatch.setattr(stress, "observed_support_audit", forbidden)
    raw = {"status": status, "incomplete_details": details, "output": []}
    result = stress.compile_response(controls, controls["cases"][0], {}, raw)
    assert result["status"] == expected
    assert result["choices"] is result["blanks"] is result["numericals"] is None
    assert result["metrics"] is result["support_audit"] is None


def test_completed_legacy_five_point_answer_is_invalid_and_not_silently_projected(controls, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Legacy five-point response reached the new depth fit")

    monkeypatch.setattr(stress, "compile_local_drawer", forbidden)
    blanks = {"source_uv": [.5, .5], "handle_uv": [.7, .5], "face_a_uv": [.2, .2],
              "joint_type": "prismatic", "motion": "pull", "path_constraint": "prismatic_pull"}
    result = stress.compile_response(controls, controls["cases"][0], {}, response(blanks))
    assert result["status"] == "invalid_output" and result["numericals"] is None


def test_completed_response_fits_before_reference_grading_and_preserves_pixels(controls, tmp_path, monkeypatch):
    monkeypatch.setattr(stress, "ROOT", tmp_path)
    case = controls["cases"][0]
    folder = tmp_path / controls["dataset"] / case["example_id"]
    folder.mkdir(parents=True)
    np.save(folder / "depth_meters.npy", np.ones((100, 100)))
    calibration = {"intrinsic": [[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]],
                   "camera_to_world": np.eye(4).tolist()}
    (folder / "calibration.json").write_text(json.dumps(calibration))
    blanks = {"source_uv": [.5, .5], "handle_uv": [.7, .5], "joint_type": "prismatic",
              "motion": "pull", "path_constraint": "prismatic_pull"}
    record = {"model_inputs": {"instruction": "open the middle drawer", "image": "rgb.png"}}
    calls = []
    actual = stress.compile_local_drawer
    reference_correct = [True]

    def fit(blanks_arg, payload, depth, calib, policy):
        assert payload == {"instruction": "open the middle drawer"}
        assert set(blanks_arg) == set(stress.FIELDS)
        assert all(policy[key] == value for key, value in LOCAL_PLANE_DEFAULTS.items())
        calls.append("fit")
        return actual(blanks_arg, payload, depth, calib, policy)

    def grade(record_arg, root_arg, case_arg, row, config):
        assert calls[-1] == "fit"
        calls.append("grade")
        return {"role_binding_correct": reference_correct[0]}

    def audit(config, case_arg, numeric):
        assert calls[-1] == "grade"
        calls.append("audit")
        return {"reference_pure": reference_correct[0]}

    monkeypatch.setattr(stress, "compile_local_drawer", fit)
    monkeypatch.setattr(stress, "grade_local_drawer", grade)
    monkeypatch.setattr(stress, "observed_support_audit", audit)
    raw = response(blanks)
    original = copy.deepcopy(raw)
    first = stress.compile_response(controls, case, record, raw)
    reference_correct[0] = False
    second = stress.compile_response(controls, case, record, raw)
    assert calls == ["fit", "grade", "audit", "fit", "grade", "audit"]
    assert first["status"] == second["status"] == "valid"
    assert first["numericals"] == second["numericals"]
    assert first["metrics"] != second["metrics"] and first["support_audit"] != second["support_audit"]
    assert first["blanks"] == second["blanks"] == blanks
    assert first["numericals"]["landmarks"]["source_uv"]["pixel_xy"] == [50, 50]
    assert first["numericals"]["landmarks"]["handle_uv"]["pixel_xy"] == [70, 50]
    assert raw == original


def test_real_held_back_grader_changes_no_inference_and_null_points_remain_unknown(controls, tmp_path, monkeypatch):
    monkeypatch.setattr(stress, "ROOT", tmp_path)
    case = controls["cases"][0]
    folder = tmp_path / controls["dataset"] / case["example_id"]
    (folder / "evaluation").mkdir(parents=True)
    np.save(folder / "depth_meters.npy", np.ones((100, 100)))
    calibration = {"intrinsic": [[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]],
                   "camera_to_world": np.eye(4).tolist()}
    (folder / "calibration.json").write_text(json.dumps(calibration))
    reference = {"middle_drawer": {"joint_type": "prismatic", "axisopeningworld": [0, 0, -1]}, "masks": {}}
    for part in ("middle_drawer", "middle_panel", "middle_handle", "cabinet"):
        filename = "evaluation/" + part + ".npy"
        mask = np.zeros((100, 100), dtype=bool) if part == "middle_handle" else np.ones((100, 100), dtype=bool)
        if part == "middle_handle":
            mask[50, 70] = True
        if part == "middle_panel":
            mask[50, 70] = False
        np.save(folder / filename, mask)
        reference["masks"][part] = {"npy": filename}
    reference_path = folder / "evaluation/reference.json"
    reference_path.write_text(json.dumps(reference))
    blanks = {"source_uv": [.5, .5], "handle_uv": [.7, .5], "joint_type": "prismatic",
              "motion": "pull", "path_constraint": "prismatic_pull"}
    record = {"model_inputs": {"instruction": "open the middle drawer", "image": "rgb.png"}}
    first = stress.compile_response(controls, case, record, response(blanks))
    assert first["status"] == "valid" and first["metrics"]["two_point_geometry_criterion"]
    for value in reference["masks"].values():
        np.save(folder / value["npy"], np.zeros((100, 100), dtype=bool))
    reference["middle_drawer"]["axisopeningworld"] = [1, 0, 0]
    reference_path.write_text(json.dumps(reference))
    second = stress.compile_response(controls, case, record, response(blanks))
    assert first["numericals"] == second["numericals"]
    assert first["metrics"] != second["metrics"]
    assert first["support_audit"] != second["support_audit"]
    assert second["metrics"]["axis_error_degrees"] == pytest.approx(90)
    unknown = stress.compile_response(controls, case, record, response({**blanks, "source_uv": None, "handle_uv": None}))
    assert unknown["status"] == "valid"
    assert unknown["numericals"]["drawer_front_plane"]["status"] == "unresolved"
    assert unknown["numericals"]["relations"] == []
    assert unknown["metrics"]["two_point_geometry_criterion"] is False
    assert unknown["support_audit"]["local_depth"]["middle_panel_purity"] is None
