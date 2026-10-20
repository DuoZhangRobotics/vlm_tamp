import numpy as np
from PIL import Image

from vlm_tamp.rgb_evaluation import evaluate_rgb, make_reference
from vlm_tamp.rgb_role_diagnostics import diagnose_role_points, template_contract
from vlm_tamp.rgb_role_points import compile_points


def task_record(instruction, goals):
    return {"model_inputs": {"instruction": instruction}, "reference": {
        "entities": [{"identifier": name, "category": name, "kind": "object"}
                     for name in ("soup", "tomato", "basket", "stove", "pot")],
        "goal_predicates": [{"name": name, "arguments": args} for name, args in goals],
    }}


def test_coverage_audit_detects_compound_tasks_misparsed_as_one_role():
    both = task_record("put both the soup and the tomato in the basket",
                       [("In", ["soup", "basket"]), ("In", ["tomato", "basket"])])
    power = task_record("turn on the stove and put the pot on it",
                        [("Turnon", ["stove"]), ("On", ["pot", "stove"])])
    assert template_contract(both)["reason"] == "goal_structure_mismatch"
    assert template_contract(power)["reason"] == "goal_structure_mismatch"


def test_offline_role_alignment_follows_independent_reference():
    record = task_record("put the soup in the basket", [("In", ["soup", "basket"])])
    assert template_contract(record) == {
        "supported": True, "reason": "supported_structure",
        "role_owners": {"slot_0": "soup", "slot_1": "basket"},
    }
    record["model_inputs"]["instruction"] = "fold the shirt"
    assert template_contract(record)["reason"] == "instruction_grammar_rejected"


def test_diagnostics_retain_wrong_instance_and_background_failures(tmp_path):
    record = task_record("turn on the stove", [("Turnon", ["stove"])])
    record["metadata"] = {"camera": {"height": 32, "width": 32}}
    record["reference"]["visual"] = {}
    for identifier, x1, x2 in (("stove", 2, 10), ("pot", 20, 28)):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, x1:x2] = 255
        mask[12:20, x1 + 2:x2 - 2] = 0  # Visible-object box contains an empty hole.
        Image.fromarray(mask).save(tmp_path / (identifier + ".png"))
        record["reference"]["visual"][identifier] = {
            "mask": identifier + ".png", "bbox": [x1 / 32, 8 / 32, x2 / 32, 24 / 32],
        }
    owners = {"slot_0": "stove"}
    cases = [([0.08, 0.3], "correct_instance"), ([0.65, 0.3], "wrong_instance"),
             ([0.18, 0.5], "background_or_occluder")]
    for point, diagnostic in cases:
        row = diagnose_role_points(record, {"slot_0": point}, owners, tmp_path)[0]
        assert row["diagnostic"] == diagnostic
        prediction = compile_points({"slot_0": point}, record["model_inputs"]["instruction"])
        metric = evaluate_rgb(record, prediction, make_reference(record), tmp_path)
        assert metric["semantic_goal"]["exact_match"] is (diagnostic == "correct_instance")
        if diagnostic == "background_or_occluder":
            assert row["point_inside_expected_bbox"] and row["pixel_distance_to_expected_mask"] > 0
    assert diagnose_role_points(record, {"slot_0": None}, owners, tmp_path)[0]["diagnostic"] == "abstained"
    assert diagnose_role_points(record, None, owners, tmp_path)[0]["diagnostic"] == "invalid_or_missing_response"
