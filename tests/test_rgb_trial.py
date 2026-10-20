import hashlib
import json

import pytest

from vlm_tamp.providers import ProviderResponse
from vlm_tamp.rgb_evaluation import freeze_references
from vlm_tamp.rgb_grounding import rgb_output_schema
from vlm_tamp.rgb_trial import run_rgb_trial


@pytest.fixture
def rgb_config(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr("vlm_tamp.rgb_trial.ROOT", tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Identify objects in this image and express the requested relation.")
    reference = tmp_path / "references.json"
    freeze_references(dataset[0], reference)
    c = {
        "model": "fake-test-model",
        "conditions": ["rgb"],
        "example_ids": ["test-0"],
        "prompt": "prompt.txt",
        "references": "references.json",
        "frozen_protocol": "protocol.json",
        "temperature": None,
        "reasoning_effort": "medium",
        "service_tier": "default",
        "image_detail": "high",
        "capture_usage_details": True,
        "max_retries": 0,
        "max_requests": 1,
        "max_budget_usd": 0.2,
        "max_output_tokens": 8000,
        "image_token_allowance": 4096,
        "timeout_seconds": 10,
        "pricing": {"input_per_million": 0.75, "output_per_million": 4.5},
    }
    protocol = {
        **c,
        "schema_sha256": hashlib.sha256(json.dumps(rgb_output_schema(), sort_keys=True).encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256((dataset[0] / "records.jsonl").read_bytes()).hexdigest(),
        "references_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "implementation_hashes": {},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    return path, c, protocol


class FakeRGBProvider:
    def __init__(self):
        self.calls = 0

    def infer(self, request):
        self.calls += 1
        assert set(request.payload) == {"instruction"} and request.image is not None
        payload = {
            "entities": [
                {"id": "a", "category": "bowl", "description": "bowl", "point": [0.2, 0.4]},
                {"id": "b", "category": "plate", "description": "plate", "point": [0.8, 0.4]},
            ],
            "goals": [{"relation": "On", "arguments": ["a", "b"]}],
        }
        return ProviderResponse(
            json.dumps(payload),
            {
                "model": "fake-test-model",
                "status": "completed",
                "usage": {"output_tokens_details": {"reasoning_tokens": 5}},
            },
            100,
            20,
        )


def test_rgb_paid_guard_and_full_offline_workflow(rgb_config, dataset, tmp_path):
    path, _, _ = rgb_config
    provider = FakeRGBProvider()
    with pytest.raises(PermissionError, match="allow-paid"):
        run_rgb_trial(dataset[0], path, tmp_path / "forbidden", provider=provider)
    assert provider.calls == 0
    out = tmp_path / "fake-workflow"
    m = run_rgb_trial(dataset[0], path, out, allow_paid=True, provider=provider)
    assert provider.calls == m["ledger"]["api_attempts"] == 1
    assert m["offline_mock"] is True and m["model_input_inventory"] is False
    summary = json.loads((out / "summary.json").read_text())
    assert summary["semantic_complete_goals"] == 1
    assert summary["ledger"]["reasoning_tokens"] == 5
    with pytest.raises(FileExistsError):
        run_rgb_trial(dataset[0], path, out, allow_paid=True, provider=provider)


def test_rgb_budget_stop_prevents_generation(rgb_config, dataset, tmp_path):
    path, c, protocol = rgb_config
    c["max_budget_usd"] = protocol["max_budget_usd"] = 1e-9
    path.write_text(json.dumps(c))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    provider = FakeRGBProvider()
    m = run_rgb_trial(dataset[0], path, tmp_path / "budget", allow_paid=True, provider=provider)
    assert provider.calls == m["ledger"]["api_attempts"] == 0
    assert json.loads((tmp_path / "budget/predictions.jsonl").read_text())["status"] == "budget_stopped"


def test_rgb_api_failure_stops_and_redacts_error(rgb_config, dataset, tmp_path):
    class Failure(Exception):
        status_code = 400

    class FailedProvider(FakeRGBProvider):
        def infer(self, request):
            self.calls += 1
            raise Failure("secret must never be logged")

    provider = FailedProvider()
    out = tmp_path / "failure"
    m = run_rgb_trial(dataset[0], rgb_config[0], out, allow_paid=True, provider=provider)
    assert provider.calls == m["ledger"]["api_attempts"] == 1
    assert m["ledger"]["unknown_usage_reserved_usd"] > 0
    assert "secret must" not in (out / "predictions.jsonl").read_text()
