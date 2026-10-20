"""Check strict part-binding diagnostics and preserve unscored planned cases."""

import copy
import sys

from vlm_tamp.libero import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from report_drawer_models_high import aggregate_model, audit_metrics  # noqa: E402


def metrics():
    return {"role_binding_correct": True, "front_body_landmarks_correct": True,
            "constraint_template_correct": True, "axis_within_tolerance": True,
            "line_constraint_compiled": True, "numerical_candidate_correct": True,
            "correct_anchors": 5}


def test_parallel_wrong_face_cannot_pass_strict_binding_despite_correct_axis():
    value = metrics()
    value["front_body_landmarks_correct"] = False
    result = audit_metrics(value, True)
    assert result["all_five_landmarks_correct"] is False
    assert result["numerical_binding_correct"] is False
    assert result["unbound_axis_within_tolerance"] is True


def test_all_bound_points_still_need_a_compiled_line_and_template():
    value = metrics()
    assert audit_metrics(value, True)["numerical_binding_correct"] is True
    for name in ("constraint_template_correct", "line_constraint_compiled", "axis_within_tolerance"):
        changed = {**value, name: False}
        assert audit_metrics(changed, True)["all_five_landmarks_correct"] is True
        assert audit_metrics(changed, True)["numerical_binding_correct"] is False


def test_truncated_and_budget_unrequested_cases_remain_unscored_in_planned_denominator():
    value = metrics()
    valid = {"status": "valid", "metrics": value, "binding_audit": audit_metrics(value, True),
             "response_context": {"status": "completed", "selection_scored": True},
             "raw_row": {"attempts": 1, "cost_usd": .1}}
    truncated = copy.deepcopy(valid)
    truncated.update(status="response_truncated", binding_audit=audit_metrics(value, False),
                     response_context={"status": "incomplete", "selection_scored": False})
    skipped = copy.deepcopy(truncated)
    skipped.update(status="not_run", response_context={"status": "not_run", "selection_scored": False},
                   raw_row={"attempts": 0, "cost_usd": 0})
    result = aggregate_model([valid, truncated, skipped])
    assert result["planned_cases"] == 3 and result["scored_selection_cases"] == 1
    assert result["boolean_metric_counts"]["role_binding_correct"] == 1
    assert result["boolean_metric_counts"]["numerical_binding_correct"] == 1
    assert result["planned_anchor_denominator"] == 15 and result["scored_anchor_denominator"] == 5
    assert result["api_attempts"] == 2
    assert result["unscored_status_counts"] == {"response_truncated": 1, "not_run": 1}
