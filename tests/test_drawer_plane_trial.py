"""Cross-boundary replay, provenance, and held-back grading checks."""

import copy
import json

import numpy as np
import pytest

from vlm_tamp import drawer_plane_trial as trial
from vlm_tamp.dataset import read_records
from vlm_tamp.drawer_constraints import DRAWER_GEOMETRY_DEFAULTS, compile_drawer
from vlm_tamp.drawer_local_plane import LOCAL_PLANE_DEFAULTS
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_numericals import world_points


@pytest.fixture
def synthetic_replay(tmp_path, monkeypatch):
    """One local plane with a deliberately wrong legacy third face landmark."""
    monkeypatch.setattr(trial, "ROOT", tmp_path)
    dataset = "data/rgbd-drawer-frontview-pilot"
    example_id = "synthetic-drawer"
    folder = tmp_path / dataset / example_id
    evaluation = folder / "evaluation"
    evaluation.mkdir(parents=True)
    depth = np.ones((100, 100))
    calibration = {"intrinsic": [[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]],
                   "camera_to_world": np.eye(4).tolist()}
    np.save(folder / "depth_meters.npy", depth)
    np.save(folder / "world_points.npy", world_points(depth, calibration["intrinsic"],
                                                     calibration["camera_to_world"]))
    (folder / "calibration.json").write_text(json.dumps(calibration))
    masks = {name: np.ones(depth.shape, dtype=bool) for name in ("middle_drawer", "middle_panel", "cabinet")}
    masks["middle_handle"] = np.zeros(depth.shape, dtype=bool)
    masks["middle_handle"][50, 70] = True
    masks["middle_panel"][50, 70] = False
    masks["middle_panel"][10, 10] = False
    reference = {"masks": {}, "middle_drawer": {"joint_type": "prismatic", "axisopeningworld": [0, 0, -1]}}
    for name, mask in masks.items():
        filename = "evaluation/" + name + ".npy"
        np.save(folder / filename, mask)
        reference["masks"][name] = {"npy": filename}
    (evaluation / "reference.json").write_text(json.dumps(reference))
    geometry = dict(DRAWER_GEOMETRY_DEFAULTS)
    blanks = {"source_uv": [.5, .5], "handle_uv": [.7, .5],
              "face_a_uv": [.2, .2], "face_b_uv": [.8, .2], "face_c_uv": [.1, .1],
              "joint_type": "prismatic", "motion": "pull", "path_constraint": "prismatic_pull"}
    record = {"example_id": example_id, "model_inputs": {"instruction": "open the middle drawer"}}
    source = {"case_id": "synthetic", "status": "valid", "model_key": "luna", "model": "gpt-6-luna",
              "blanks": blanks,
              "numericals": compile_drawer(blanks, record["model_inputs"], depth, calibration, geometry)}
    config = {"dataset": dataset, "source_trial": "results/synthetic-source", "geometry": geometry,
              "local_plane": dict(LOCAL_PLANE_DEFAULTS),
              "evaluation": {"axis_tolerance_degrees": 5., "minimum_support_purity_diagnostic": .9}}
    case = {"id": "synthetic", "source_case_id": "synthetic", "example_id": example_id}
    source_folder = tmp_path / config["source_trial"]
    source_folder.mkdir(parents=True)
    (source_folder / "predictions.jsonl").write_text(json.dumps(source) + "\n")
    return config, case, record, source, folder, source_folder


def test_replay_projects_only_unchanged_two_anchors_and_method_choices(synthetic_replay, monkeypatch):
    config, case, record, source, _, _ = synthetic_replay
    original = copy.deepcopy(source)
    actual_compile = trial.compile_local_drawer
    seen = []

    def capture(blanks, payload, depth, calibration, settings):
        assert set(blanks) == set(trial.LOCAL_FIELDS)
        assert not any(key.startswith("face_") for key in blanks)
        assert payload == {"instruction": record["model_inputs"]["instruction"]}
        assert not any(key in payload for key in ("masks", "joint", "pose", "inventory"))
        seen.append(copy.deepcopy(blanks))
        return actual_compile(blanks, payload, depth, calibration, settings)

    monkeypatch.setattr(trial, "compile_local_drawer", capture)
    row = trial.replay_case(config, case, record, source)
    assert seen == [{key: source["blanks"][key] for key in trial.LOCAL_FIELDS}]
    assert row["blanks"]["source_uv"] == source["blanks"]["source_uv"]
    assert row["blanks"]["handle_uv"] == source["blanks"]["handle_uv"]
    assert row["source_blanks"] == source["blanks"]
    assert row["old_numericals"] == source["numericals"]
    assert source == original
    assert not row["semantic_pixels_changed"]
    assert row["new_api_calls"] == 0 and row["cost_usd"] == 0


def test_source_file_bytes_and_source_nested_lists_stay_unchanged(synthetic_replay):
    config, case, record, _, _, source_folder = synthetic_replay
    source_file = source_folder / "predictions.jsonl"
    before = source_file.read_bytes()
    source = trial.source_rows(config)[case["source_case_id"]]
    row = trial.replay_case(config, case, record, source)
    row["blanks"]["source_uv"][0] = .1
    row["source_blanks"]["handle_uv"][0] = .1
    row["old_numericals"]["landmarks"]["source_uv"]["pixel_xy"][0] = 0
    assert source["blanks"]["source_uv"] == [.5, .5]
    assert source["blanks"]["handle_uv"] == [.7, .5]
    assert source["numericals"]["landmarks"]["source_uv"]["pixel_xy"] == [50, 50]
    assert source_file.read_bytes() == before


def test_reference_changes_only_grading_not_sampling_or_numeric_estimates(synthetic_replay):
    config, case, record, source, folder, _ = synthetic_replay
    first = trial.replay_case(config, case, record, source)
    assert first["metrics"]["role_binding_correct"]
    reference_path = folder / "evaluation/reference.json"
    reference = json.loads(reference_path.read_text())
    for value in reference["masks"].values():
        np.save(folder / value["npy"], np.zeros((100, 100), dtype=bool))
    reference["middle_drawer"]["axisopeningworld"] = [1, 0, 0]
    reference_path.write_text(json.dumps(reference))
    second = trial.replay_case(config, case, record, source)
    assert first["numericals"] == second["numericals"]
    assert first["blanks"] == second["blanks"]
    assert first["metrics"] != second["metrics"]
    assert first["support_audit"] != second["support_audit"]
    assert second["metrics"]["role_binding_correct"] is False
    assert second["metrics"]["axis_error_degrees"] == pytest.approx(90)


def test_old_five_anchor_failure_is_preserved_when_reduced_contract_succeeds(synthetic_replay):
    config, case, record, source, _, _ = synthetic_replay
    row = trial.replay_case(config, case, record, source)
    assert row["old_metrics"]["role_binding_correct"]
    assert row["old_metrics"]["axis_within_tolerance"]
    assert row["old_metrics"]["front_body_landmarks_correct"] is False
    assert row["old_metrics"]["old_five_anchor_criterion"] is False
    assert row["metrics"]["two_point_geometry_criterion"] is True
    assert "not correction of the old five-anchor answer" in row["metrics"]["comparison_scope"]
    assert row["metrics"]["semantic_front_face_verified"] is False
    assert row["metrics"]["planner_ready"] is False


def test_support_purity_is_only_a_diagnostic_and_never_a_fit_gate(synthetic_replay):
    config, case, record, source, folder, _ = synthetic_replay
    row = trial.replay_case(config, case, record, source)
    assert row["metrics"]["two_point_geometry_criterion"]
    np.save(folder / "evaluation/middle_panel.npy", np.zeros((100, 100), dtype=bool))
    replay = trial.replay_case(config, case, record, source)
    assert replay["numericals"] == row["numericals"]
    assert replay["metrics"]["two_point_geometry_criterion"]
    assert replay["metrics"]["middle_panel_support_purity"] == 0
    assert not replay["metrics"]["two_point_geometry_and_reference_pure_support"]
    assert not replay["metrics"]["support_purity_is_fit_gate"]


def test_support_quality_counts_disjoint_parts_background_and_empty_support():
    mask = np.ones((2, 4), dtype=bool)
    masks = {name: np.zeros(mask.shape, dtype=bool) for name in ("middle_panel", "middle_handle", "cabinet")}
    masks["middle_panel"][0, :3] = True
    masks["middle_handle"][0, 3] = True
    masks["cabinet"][0] = True
    masks["cabinet"][1, 0] = True
    result = trial.support_quality(mask, masks)
    assert result["support_sample_count"] == 8
    assert result["middle_panel_samples"] == 3 and result["middle_panel_purity"] == 3 / 8
    assert result["middle_handle_samples"] == 1 and result["middle_handle_fraction"] == 1 / 8
    assert result["other_cabinet_samples"] == 1 and result["non_cabinet_samples"] == 3
    assert not result["part_labels_used_in_sampling"]
    empty = trial.support_quality(np.zeros(mask.shape, dtype=bool), masks)
    assert empty["support_sample_count"] == empty["middle_panel_samples"] == empty["middle_handle_samples"] == 0
    assert empty["other_cabinet_samples"] == empty["non_cabinet_samples"] == 0
    assert empty["middle_panel_purity"] is None and empty["middle_handle_fraction"] is None


@pytest.mark.parametrize("pixels", [[[True, 1]], [[1, -1]], [[4, 1]], [[1]], [[1.0, 1]]])
def test_support_pixels_are_integer_observed_image_coordinates(pixels):
    with pytest.raises(ValueError):
        trial.support_mask(pixels, (2, 4))


def test_duplicate_support_pixels_are_counted_once():
    result = trial.support_mask([[1, 0], [1, 0], [3, 1]], (2, 4))
    assert result.sum() == 2 and result[0, 1] and result[1, 3]


@pytest.mark.parametrize("changed", ["model_key", "status"])
def test_replay_rejects_non_luna_or_incomplete_predictions(synthetic_replay, changed):
    config, case, record, source, _, _ = synthetic_replay
    source[changed] = "astra" if changed == "model_key" else "response_truncated"
    with pytest.raises(ValueError, match="completed real Luna"):
        trial.replay_case(config, case, record, source)


@pytest.mark.parametrize("key,value", [
    ("local_plane_radius_pixels", 31), ("local_plane_ransac_seed", 1),
    ("local_plane_minimum_inlier_fraction", .5),
])
def test_predeclared_policy_tampering_is_rejected(tmp_path, key, value):
    config = json.loads((ROOT / "configs/drawer_plane_ablation.json").read_text())
    config["local_plane"][key] = value
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="predeclared blind"):
        trial.load_config(path)


def test_real_cached_luna_replay_preserves_prediction_bytes_and_selected_pixels():
    config = trial.load_config(ROOT / "configs/drawer_plane_ablation.json")
    case = config["cases"][0]
    records, _ = read_records(ROOT / config["dataset"])
    record = next(value for value in records if value["example_id"] == case["example_id"])
    path = ROOT / config["source_trial"] / "predictions.jsonl"
    before = path.read_bytes()
    source = trial.source_rows(config)[case["source_case_id"]]
    original = copy.deepcopy(source)
    row = trial.replay_case(config, case, record, source)
    for field in trial.LOCAL_FIELDS:
        assert row["blanks"][field] == source["blanks"][field]
    for field in ("source_uv", "handle_uv"):
        measured = row["numericals"]["landmarks"][field]
        assert measured["uv"] == source["blanks"][field]
        assert measured["pixel_xy"] == source["numericals"]["landmarks"][field]["pixel_xy"]
    assert set(row["numericals"]["landmarks"]) == {"source_uv", "handle_uv"}
    assert row["new_api_calls"] == 0
    assert row["numericals"]["drawer_front_plane"]["policy"] == LOCAL_PLANE_DEFAULTS
    assert source == original and path.read_bytes() == before
