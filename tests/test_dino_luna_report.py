"""Ensure paired reports retain failures and reject mismatched historical inputs."""

import copy
import json
import sys

import pytest

from vlm_tamp.libero import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from report_dino_luna_comparison import comparison_data, read_response_context  # noqa: E402
from verify_dino_luna_comparison import _response_model_matches  # noqa: E402


def summaries(tmp_path):
    config = {key: "same" for key in (
        "dataset", "depth_dataset", "candidate_datasets", "conditions", "cases", "prompt",
        "segmentation", "proposal",
    )}
    config.update(model="gpt-5.4-mini-2026-03-17", reasoning_effort="medium", max_output_tokens=3500,
                  max_budget_usd=.15, pricing={"input_per_million": .75})
    task = {
        "id": "fixture", "condition": "dinov2", "instruction": "put the bowl on the plate",
        "status": "invalid_output", "response_context": {"status": "incomplete",
            "selection_diagnostic_applicable": False},
        "metrics": {"semantic_complete_goal": False}, "raw_row": {"cost_usd": .01},
    }
    history = {"config": config, "model": config["model"], "planned_cases": 1,
               "prompt_text": "same prompt", "candidate_metadata": {"scene": {"hash": "same"}},
               "tasks": [task], "ledger": {"api_attempts": 1}, "condition_summaries": {},
               "incomplete_comparison": True, "stopped": None}
    current = copy.deepcopy(history)
    current["config"].update(model="gpt-6-luna", reasoning_effort="none", max_output_tokens=1000,
                              max_budget_usd=2., pricing={"input_per_million": .1})
    current["model"] = "gpt-6-luna"
    current["tasks"][0].update(status="valid", response_context={"status": "completed",
        "selection_diagnostic_applicable": True}, metrics={"semantic_complete_goal": True})
    source = tmp_path / "mini.json"
    source.write_text(json.dumps(history))
    return current, history, source


def test_comparison_retains_incomplete_mini_as_unscored_and_records_control_changes(tmp_path):
    current, history, source = summaries(tmp_path)
    result = comparison_data(current, history, source)
    assert result["model_ranking_established"] is False
    assert result["paired_case_statuses"][0]["mini_selection_scored"] is False
    assert result["paired_case_statuses"][0]["mini_response_status"] == "incomplete"
    assert result["paired_case_statuses"][0]["luna_selection_scored"] is True
    assert result["changed_trial_controls"]["reasoning_effort"] == {"mini": "medium", "luna": "none"}


@pytest.mark.parametrize("field", ["prompt", "proposal", "candidate_metadata"])
def test_matched_comparison_rejects_changed_inputs(tmp_path, field):
    current, history, source = summaries(tmp_path)
    if field == "candidate_metadata":
        current[field]["scene"]["hash"] = "different"
    else:
        current["config"][field] = "different"
    with pytest.raises(ValueError):
        comparison_data(current, history, source)


def test_failed_charged_attempt_is_distinguished_from_an_unrequested_case(tmp_path):
    context = read_response_context(tmp_path, "fixture", {"status": "api_failure"})
    assert context["status"] == context["unscored_selection_reason"] == "api_failure"
    assert context["selection_diagnostic_applicable"] is False


@pytest.mark.parametrize("actual,expected", [
    ("gpt-6-luna", True), ("gpt-6-luna-2026-10-12", True),
    ("gpt-6-sol", False), ("gpt-6-luna-preview", False), (None, False),
])
def test_only_declared_model_family_is_accepted(actual, expected):
    assert _response_model_matches(actual, "gpt-6-luna") is expected
