import copy
import json

import numpy as np
import pytest
from PIL import Image

from vlm_tamp.dataset import read_records
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_candidate_evaluation import grade_choices, proposal_coverage
from vlm_tamp.rgbd_landmark_choices import ID_FIELDS
from vlm_tamp.rgbd_numericals import POINT_FIELDS


def scene(tmp_path, *, shared=False):
    masks = {owner: np.zeros((8, 12), dtype=np.uint8) for owner in ("bowl_1", "plate_1", "bowl_2")}
    masks["bowl_1"][1:4, 1:4] = 255
    masks["plate_1"][4:7, 5:8] = 255
    masks["bowl_2"][1:4, 9:12] = 255
    if shared:
        masks["bowl_2"][2, 2] = 255
    visual = {}
    for owner, mask in masks.items():
        Image.fromarray(mask).save(tmp_path / (owner + ".png"))
        visual[owner] = {"mask": owner + ".png", "kind": "object", "bbox": [0., 0., 1., 1.]}
    return {
        "model_inputs": {"instruction": "put the bowl on the plate"},
        "reference": {
            "entities": [{"identifier": owner, "category": owner, "kind": "object"} for owner in masks],
            "goal_predicates": [{"name": "On", "arguments": ["bowl_1", "plate_1"]}],
            "visual": visual, "geometry": {},
        },
    }


def candidates():
    return [{"id": index, "pixel_xy": pixel} for index, pixel in enumerate(
        ([2, 2], [6, 5], [10, 2], [0, 0])
    )]


def choices():
    return {**{field: 0 for field in ID_FIELDS}, "target_id": 1,
            "grasp_mode": "top_down", "path_constraint": "none"}


def by_field(rows):
    return {row["field"]: row for row in rows}


def test_coverage_does_not_establish_selection_or_part_quality(tmp_path):
    record = scene(tmp_path)
    coverage = proposal_coverage(record, tmp_path, candidates())
    assert coverage["complete_required_role_coverage"]
    assert coverage["required_roles_covered"] == 2
    assert coverage["required_fields_covered"] == len(POINT_FIELDS)
    assert coverage["roles"]["source"]["available_candidate_ids"] == [0]
    assert coverage["roles"]["target"]["available_candidate_ids"] == [1]
    assert coverage["wrong_instance_candidate_ids"] == [2]
    assert coverage["background_candidate_ids"] == [3]
    assert not any(coverage[key] for key in (
        "part_coverage_verified", "contact_coverage_verified", "axis_coverage_verified", "grasp_coverage_verified",
    ))
    bad = grade_choices(record, tmp_path, {"upright": False}, {**choices(), "source_id": 2}, candidates(), None)
    assert bad["choices_valid"] and not bad["semantic_complete_goal"]
    assert bad["proposal_coverage"]["complete_required_role_coverage"]
    assert by_field(bad["selection_diagnostics"])["source_uv"]["diagnostic"] == "selection_error"


def test_exact_id_resolution_uses_unchanged_semantic_grader(tmp_path):
    result = grade_choices(scene(tmp_path), tmp_path, {"upright": False}, choices(), candidates(), None)
    assert result["choices_valid"] and result["semantic_complete_goal"]
    assert result["resolved_blanks"]["source_uv"] == [2.5/12, 2.5/8]
    assert result["path_choice_correct"]
    assert all(row["diagnostic"] == "correct_selection" for row in result["selection_diagnostics"])
    assert not result["grasp_candidate"]
    assert not result["grasp_success_evaluated"]
    assert not result["robot_execution_evaluated"]


def test_null_distinguishes_missing_proposal_from_selection_abstention(tmp_path):
    record = scene(tmp_path)
    no_target = [{"id": 0, "pixel_xy": [2, 2]}]
    blank = {**{field: None for field in ID_FIELDS}, "grasp_mode": "unresolved", "path_constraint": "none"}
    result = grade_choices(record, tmp_path, {"upright": False}, blank, no_target, None)
    fields = by_field(result["selection_diagnostics"])
    assert fields["target_uv"]["diagnostic"] == "proposal_missing"
    assert fields["source_uv"]["diagnostic"] == "selection_abstention"
    assert not result["semantic_complete_goal"]
    assert result["case_in_denominator"]
    assert result["proposal_coverage"]["required_roles_covered"] == 1


def test_shared_expected_hits_are_explicit_and_not_strict_usable_coverage(tmp_path):
    record = scene(tmp_path, shared=True)
    coverage = proposal_coverage(record, tmp_path, candidates())
    assert coverage["shared_candidate_ids"] == [0]
    assert coverage["roles"]["source"]["expected_object_hit_ids"] == [0]
    assert coverage["roles"]["source"]["available_candidate_ids"] == []
    assert not coverage["complete_required_role_coverage"]
    result = grade_choices(record, tmp_path, {"upright": False}, choices(), candidates(), None)
    assert by_field(result["selection_diagnostics"])["source_uv"]["diagnostic"] == "selection_error"
    assert not result["semantic_complete_goal"]
    result = grade_choices(record, tmp_path, {"upright": False}, {**choices(), "source_id": None}, candidates(), None)
    assert by_field(result["selection_diagnostics"])["source_uv"]["diagnostic"] == "selection_abstention"


@pytest.mark.parametrize("form", [None, {}, {"source_id": 0}, {**choices(), "source_id": 40},
                                   {**choices(), "source_id": True}])
def test_missing_and_invalid_forms_remain_failed_cases(tmp_path, form):
    result = grade_choices(scene(tmp_path), tmp_path, {"upright": False}, form, candidates(), None)
    assert not result["choices_valid"]
    assert result["choice_validation_error"]
    assert not result["semantic_complete_goal"]
    assert not result["path_choice_correct"]
    assert result["case_in_denominator"]
    assert len(result["anchors"]) == len(POINT_FIELDS)
    if isinstance(form, dict) and form.get("source_id") is not None:
        assert by_field(result["selection_diagnostics"])["source_uv"]["diagnostic"] == "selection_error"


def test_empty_candidates_are_counted_and_inputs_and_masks_are_unchanged(tmp_path):
    record = scene(tmp_path)
    selected = {**{field: None for field in ID_FIELDS}, "grasp_mode": "unresolved", "path_constraint": "none"}
    original_record, original_choices = copy.deepcopy(record), copy.deepcopy(selected)
    mask_bytes = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = grade_choices(record, tmp_path, {"upright": False}, selected, [], None)
    assert result["proposal_coverage"]["candidate_count"] == 0
    assert result["proposal_coverage"]["required_roles_covered"] == 0
    assert all(row["diagnostic"] == "proposal_missing" for row in result["selection_diagnostics"])
    assert record == original_record and selected == original_choices
    assert mask_bytes == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    data = candidates()
    original = copy.deepcopy(data)
    grade_choices(record, tmp_path, {"upright": False}, choices(), data, None)
    assert data == original


def test_all_six_frozen_cases_remain_in_the_denominator_without_responses():
    config = json.loads((ROOT / "configs/rgbd_numeric_trial.json").read_text())
    records, root = read_records(ROOT / config["dataset"])
    by_id = {record["example_id"]: record for record in records}
    results = [grade_choices(by_id[case["example_id"]], root, case, None, [], None) for case in config["cases"]]
    assert len(results) == 6
    assert len({case["example_id"] for case in config["cases"]}) == 4
    assert all(result["case_in_denominator"] and not result["semantic_complete_goal"] for result in results)
    assert sum(result["choices_valid"] for result in results) == 0
