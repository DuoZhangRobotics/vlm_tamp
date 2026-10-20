"""Budget and transport safeguards for the matched high-reasoning pilot."""

import copy
import json

import pytest

from vlm_tamp import drawer_model_comparison as comparison
from vlm_tamp import task_grounding_frontview as parent
from vlm_tamp.dataset import read_records
from vlm_tamp.experiment import BudgetExceeded, input_price_ceiling
from vlm_tamp.libero import ROOT
from vlm_tamp.providers import ProviderResponse


@pytest.fixture
def controls():
    return json.loads((ROOT / "configs/drawer_models_high.json").read_text())


def test_cumulative_admission_counts_both_models_and_unknown_usage(controls):
    ledger = comparison.SharedLedger(controls)
    ledger.ledgers["luna"].actual_cost = controls["max_budget_usd"] - .10
    ledger.ledgers["astra"].actual_cost = .06
    ledger.ledgers["astra"].unknown_cost_reservation = .01
    with pytest.raises(BudgetExceeded):
        ledger.admit("astra", .030001)
    assert sum(item.attempts for item in ledger.ledgers.values()) == 0
    ledger.admit("astra", .03)
    assert ledger.ledgers["astra"].attempts == 1
    assert ledger.ledgers["luna"].attempts == 0


def test_unknown_request_reservation_cannot_be_spent_by_the_other_model(controls):
    ledger = comparison.SharedLedger(controls)
    reservation = controls["max_budget_usd"] - .01
    ledger.admit("luna", reservation)
    ledger.reserve_unknown("luna", reservation)
    with pytest.raises(BudgetExceeded):
        ledger.admit("astra", .010001)
    assert ledger.ledgers["astra"].attempts == 0
    assert ledger.ledgers["luna"].unknown_cost_reservation == reservation


def test_each_arm_has_four_attempts_and_shares_eight_total_attempts(controls):
    ledger = comparison.SharedLedger(controls)
    for _ in range(4):
        ledger.admit("luna", .001)
    with pytest.raises(BudgetExceeded):
        ledger.admit("luna", .001)
    for _ in range(4):
        ledger.admit("astra", .001)
    with pytest.raises(BudgetExceeded):
        ledger.admit("astra", .001)
    assert ledger.as_dict()["api_attempts"] == 8


def test_mixed_model_usage_uses_its_own_cache_write_ceiling_without_double_counting(controls):
    ledger = comparison.SharedLedger(controls)
    response = ProviderResponse("{}", {"usage": {
        "input_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 1000},
        "output_tokens_details": {"reasoning_tokens": 200},
    }}, input_tokens=1000, output_tokens=200)
    costs = []
    for key in ("luna", "astra"):
        ledger.admit(key, .03)
        costs.append(ledger.account(key, response, .03))
    assert costs == pytest.approx([.000225, .0225])
    summary = ledger.as_dict()
    assert summary["estimated_usage_cost_usd"] == pytest.approx(.022725)
    assert summary["input_tokens"] == 2000
    assert summary["output_tokens"] == 400
    assert summary["reasoning_tokens"] == 400
    assert summary["unknown_usage_reserved_usd"] == 0


def test_matched_requests_preserve_parent_rgb_prompt_schema_and_local_geometry(controls):
    original = json.loads((ROOT / "configs/task_grounding_frontview.json").read_text())
    records, _ = read_records(ROOT / controls["dataset"])
    records = {record["example_id"]: record for record in records}
    parent_cases = {case["id"]: case for case in original["cases"]}
    inputs_by_scene = {}
    reservations = []
    for case in controls["cases"]:
        model_config = comparison.model_config(controls, case)
        assert model_config["reasoning_effort"] == "high"
        assert model_config["max_output_tokens"] == 6000
        assert model_config["max_requests"] == 4
        assert model_config["geometry"] == original["geometry"]
        request = parent.request_arguments(model_config, case, records[case["example_id"]])
        prior_request = parent.request_arguments(original, parent_cases[case["base_case_id"]],
                                                records[case["example_id"]])
        for allowed_change in ("model", "reasoning", "max_output_tokens"):
            prior_request[allowed_change] = request[allowed_change]
        assert request == prior_request
        previous = inputs_by_scene.setdefault(case["example_id"], copy.deepcopy(request["input"]))
        assert request["input"] == previous
        payload = json.loads(request["input"][0]["content"][0]["text"])
        assert set(payload) == {"instruction", "task_type", "blanks"}
        assert len(request["input"][0]["content"]) == 2
        assert not any(key in request for key in ("tools", "conversation", "previous_response_id"))
        assert input_price_ceiling(model_config) == (.125 if case["model_key"] == "luna" else 12.5)
        reservations.append(comparison.request_cost_reservation(controls, request))
    assert len(inputs_by_scene) == 4
    assert sum(reservations) + controls["prior_usage_cost_usd"] <= controls["overall_budget_usd"]


def test_token_truncation_is_a_completion_failure_before_parsing_any_landmarks(controls):
    case = controls["cases"][0]
    raw = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
           "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}
    result = comparison.compile_response(controls, case, {}, raw)
    assert result["status"] == "response_truncated"
    assert result.get("choices") is None
    assert result.get("blanks") is None
    assert result.get("numericals") is None


@pytest.mark.parametrize("field,value", [
    ("reasoning_effort", "low"), ("max_output_tokens", 1000), ("max_retries", 1),
    ("overall_budget_usd", 2.01), ("prior_usage_cost_usd", 0.),
])
def test_control_or_cumulative_budget_changes_are_rejected(controls, tmp_path, field, value):
    controls[field] = value
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(controls))
    with pytest.raises(ValueError):
        comparison.load_config(path)
