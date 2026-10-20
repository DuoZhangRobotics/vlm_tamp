import json

import pytest

from vlm_tamp.cli import load_credentials
from vlm_tamp.evaluation import evaluate_saved
from vlm_tamp.experiment import run_experiment
from vlm_tamp.providers import ProviderResponse
from vlm_tamp.visualization import gallery


def paid_config(config):
    path, c = config
    c.update(
        provider="openai",
        model="test-model",
        max_budget_usd=1,
        pricing={"input_per_million": 1, "output_per_million": 1},
    )
    path.write_text(json.dumps(c))
    return path


class FakeProvider:
    def __init__(self, text='{"schema_version":"1.2","entities":[],"goals":[]}'):
        self.text = text
        self.calls = 0

    def infer(self, request):
        self.calls += 1
        return ProviderResponse(self.text, {"output": self.text}, 100, 10)


def test_offline_end_to_end_caching_and_report(dataset, config, tmp_path):
    root, _ = dataset
    out = tmp_path / "first"
    a = run_experiment(root, config[0], out, cache=tmp_path / "cache")
    assert a["successful_predictions"] == 4
    b = run_experiment(root, config[0], tmp_path / "second", cache=tmp_path / "cache")
    cached = [json.loads(x) for x in (tmp_path / "second/predictions.jsonl").read_text().splitlines()]
    assert all(r["cached"] and r["cost_usd"] == 0 for r in cached)
    assert b["ledger"]["api_attempts"] == 0
    rows, summary = evaluate_saved(root, out)
    assert len(rows) == len(summary) == 4
    report = gallery(root, out)
    assert report.exists() and (out / "report.md").exists()
    assert "OFFLINE MOCK" in report.read_text()


def test_paid_guard_and_dry_run(dataset, config, tmp_path):
    path = paid_config(config)
    fake = FakeProvider()
    with pytest.raises(PermissionError, match="allow-paid"):
        run_experiment(dataset[0], path, tmp_path / "forbidden", provider=fake)
    run_experiment(dataset[0], path, tmp_path / "dry", dry_run=True, provider=fake)
    assert fake.calls == 0


def test_budget_and_actual_attempt_limit(dataset, config, tmp_path):
    path = paid_config(config)
    c = json.loads(path.read_text())
    c["max_requests"] = 1
    path.write_text(json.dumps(c))
    fake = FakeProvider()
    manifest = run_experiment(dataset[0], path, tmp_path / "limited", allow_paid=True, provider=fake)
    assert fake.calls == 1 and manifest["ledger"]["api_attempts"] == 1
    rows = [json.loads(x) for x in (tmp_path / "limited/predictions.jsonl").read_text().splitlines()]
    assert rows[-1]["status"] == "budget_stopped"
    c["max_requests"], c["max_budget_usd"] = 5, 0
    path.write_text(json.dumps(c))
    fake = FakeProvider()
    run_experiment(dataset[0], path, tmp_path / "zero", allow_paid=True, provider=fake)
    assert fake.calls == 0


def test_cache_write_ceiling_and_reasoning_are_not_double_billed(config, dataset):
    from vlm_tamp.dataset import build_request
    from vlm_tamp.experiment import Ledger, load_config, request_reservation

    c = load_config(config[0])
    c.update(
        capture_usage_details=True,
        pricing={"input_per_million": 10, "cache_write_per_million": 12.5, "output_per_million": 50},
    )
    request = build_request(dataset[1]["model_inputs"], dataset[0], "rgb", c["prompt_text"])
    inputs, reservation = request_reservation(c, request)
    assert reservation == (inputs * 12.5 + c["max_output_tokens"] * 50) / 1e6
    ledger = Ledger(c)
    response = ProviderResponse(
        "{}",
        {
            "usage": {
                "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 30},
                "output_tokens_details": {"reasoning_tokens": 800},
            }
        },
        100,
        1000,
    )
    cost = ledger.account(response, reservation)
    assert cost == (100 * 12.5 + 1000 * 50) / 1e6
    assert ledger.as_dict()["reasoning_tokens"] == 800
    assert ledger.as_dict()["cached_input_tokens"] == 40
    assert ledger.as_dict()["cache_write_tokens"] == 30
    old = dict(c, pricing={"input_per_million": 10, "output_per_million": 50})
    assert Ledger(old).account(response, reservation) == (100 * 10 + 1000 * 50) / 1e6


def test_validation_failure_retains_usage_and_raw(dataset, config, tmp_path):
    path = paid_config(config)
    manifest = run_experiment(
        dataset[0], path, tmp_path / "invalid", allow_paid=True, provider=FakeProvider("not JSON")
    )
    assert manifest["ledger"]["input_tokens"] == 400
    assert manifest["successful_predictions"] == 0
    rows = [json.loads(x) for x in (tmp_path / "invalid/predictions.jsonl").read_text().splitlines()]
    assert all(r["status"] == "validation_failure" for r in rows)
    assert len(list((tmp_path / "invalid/raw").glob("*.json"))) == 4


def test_api_failure_can_stop_without_extra_calls(dataset, config, tmp_path):
    path = paid_config(config)
    c = json.loads(path.read_text())
    c.update(stop_on_api_failure=True, max_retries=0)
    path.write_text(json.dumps(c))

    class Failure(Exception):
        status_code = 400

    class FailingProvider(FakeProvider):
        def infer(self, request):
            self.calls += 1
            raise Failure("must not expose credentials")

    provider = FailingProvider()
    out = tmp_path / "stop-after-failure"
    manifest = run_experiment(dataset[0], path, out, allow_paid=True, provider=provider)
    assert provider.calls == manifest["ledger"]["api_attempts"] == 1
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "api_failure"
    assert manifest["ledger"]["unknown_usage_reserved_usd"] > 0
    assert "must not expose" not in (out / "predictions.jsonl").read_text()


def test_rate_limit_retry_is_accounted(dataset, config, tmp_path, monkeypatch):
    path = paid_config(config)
    monkeypatch.setattr("vlm_tamp.experiment.time.sleep", lambda _: None)

    class RateLimit(Exception):
        status_code = 429

    class Provider(FakeProvider):
        def infer(self, request):
            if self.calls == 0:
                self.calls += 1
                raise RateLimit("credential-bearing messages must not be logged")
            return super().infer(request)

    provider = Provider()
    manifest = run_experiment(dataset[0], path, tmp_path / "retry", allow_paid=True, provider=provider)
    assert manifest["ledger"]["api_attempts"] == 5
    assert manifest["ledger"]["unknown_usage_reserved_usd"] > 0
    assert "credential-bearing" not in (tmp_path / "retry/predictions.jsonl").read_text()


def test_missing_credentials_fall_back_to_mock(dataset, config, tmp_path, monkeypatch):
    path = paid_config(config)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manifest = run_experiment(dataset[0], path, tmp_path / "fallback")
    assert manifest["offline_mock"] and manifest["requested_provider"] == "openai"
    assert manifest["ledger"]["api_attempts"] == 0


def test_credentials_are_literal_and_allowlisted(tmp_path, monkeypatch):
    keyfile = tmp_path / "key.env"
    keyfile.write_text("export OPENAI_API_KEY='literal-$(never_execute)'\nIGNORED_SECRET=no\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    load_credentials(keyfile)
    import os

    assert os.environ["OPENAI_API_KEY"] == "literal-$(never_execute)"
    assert "IGNORED_SECRET" not in os.environ


def test_changed_reference_dataset_is_rejected(dataset, config, tmp_path):
    root, record = dataset
    out = tmp_path / "original"
    run_experiment(root, config[0], out, cache=tmp_path / "cache")
    record["reference"]["goal_predicates"] = []
    (root / "records.jsonl").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Dataset changed"):
        evaluate_saved(root, out)


def test_changed_image_is_rejected_even_when_manifest_is_unchanged(dataset, config, tmp_path):
    from PIL import Image

    root, _ = dataset
    out = tmp_path / "image-check"
    run_experiment(root, config[0], out, cache=tmp_path / "cache")
    Image.new("RGB", (20, 20), "red").save(root / "rgb.png")
    with pytest.raises(ValueError, match="image changed"):
        evaluate_saved(root, out)


def test_cache_keys_track_actual_schema_and_generation(dataset, config):
    from vlm_tamp.experiment import load_config, request_key
    from vlm_tamp.dataset import build_request

    c = load_config(config[0])
    root, r = dataset
    request = build_request(r["model_inputs"], root, "rgb", c["prompt_text"])
    original = request_key(c, request)
    for key, value in [("temperature", 0), ("reasoning_effort", "none"), ("image_detail", "low")]:
        changed = dict(c)
        changed[key] = value
        assert request_key(changed, request) != original


def test_frozen_protocol_blocks_prompt_or_generation_changes(config, tmp_path):
    import hashlib
    from vlm_tamp.experiment import load_config
    from vlm_tamp.grounding import strict_output_schema

    path, c = config
    c.update(temperature=0, image_detail="high", example_ids=["test-0"])
    protocol = {
        "schema_sha256": hashlib.sha256(
            json.dumps(strict_output_schema(), sort_keys=True).encode()
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(__import__("pathlib").Path(c["prompt"]).read_bytes()).hexdigest(),
        "models": [c["model"]],
        **{
            k: c[k] for k in ["example_ids", "conditions", "temperature", "image_detail", "max_output_tokens"]
        },
    }
    frozen = tmp_path / "protocol.json"
    frozen.write_text(json.dumps(protocol))
    c["frozen_protocol"] = str(frozen)
    path.write_text(json.dumps(c))
    assert load_config(path)["frozen_protocol_sha256"]
    c["max_output_tokens"] += 1
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="frozen protocol"):
        load_config(path)
