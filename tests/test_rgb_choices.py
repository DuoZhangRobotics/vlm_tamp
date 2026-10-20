import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from vlm_tamp.providers import ProviderResponse
from vlm_tamp.rgb_choice_trial import ChoiceOpenAIProvider, audit_request, file_sha, run_choice_trial
from vlm_tamp.rgb_choices import (
    ChoicePrediction,
    build_choice_request,
    choice_schema,
    compile_choices,
    digest,
)
from vlm_tamp.rgb_evaluation import evaluate_rgb, freeze_references, make_reference


@pytest.fixture
def choice_dataset(dataset):
    root, record = dataset
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb[:10, :10] = [230, 20, 10]
    rgb[10:, :10] = [10, 20, 220]
    rgb[:10, 10:] = [10, 220, 30]
    rgb[10:, 10:] = [200, 180, 10]
    Image.fromarray(rgb).save(root / "rgb.png")
    return root, record


def form(a, b):
    return {
        "schema_version": "rgb-choice-1.0",
        "goals": [
            {
                "state": "supported_on",
                "object_choice": a,
                "object_part": "whole",
                "target_choice": b,
                "target_part": "whole",
            }
        ],
    }


def side_choices(request):
    left = next(r["choice"] for r in request.regions if r["point"][0] < 0.5)
    right = next(r["choice"] for r in request.regions if r["point"][0] >= 0.5)
    return left, right


def test_choices_depend_on_pixels_not_instruction_or_reference(choice_dataset):
    root, record = choice_dataset
    image = (root / "rgb.png").read_bytes()
    q = build_choice_request("put the bowl on the plate", image, "test")
    other = build_choice_request("open an unrelated drawer", image, "test")
    assert (
        q.regions == other.regions
        and q.scene_image == other.scene_image
        and q.cards_image == other.cards_image
    )
    assert set(q.payload) == {"instruction", "visual_choices"}
    assert "DO_NOT_SEND_123" not in q.text and "bowl_1" not in q.text
    assert all(set(region) == {"choice", "point", "area", "pixel", "box"} for region in q.regions)
    schema = choice_schema(len(q.regions))
    for variant in schema["$defs"]["ChoiceGoal"]["anyOf"]:
        assert variant["properties"]["object_choice"]["enum"] == list(range(len(q.regions) + 1))


def test_same_evaluator_accepts_correct_choices_and_rejects_wrong_identity(choice_dataset):
    root, record = choice_dataset
    q = build_choice_request("put the bowl on the plate", (root / "rgb.png").read_bytes(), "test")
    left, right = side_choices(q)
    reference = make_reference(record)
    correct = compile_choices(ChoicePrediction.model_validate(form(left, right)), q.regions)
    assert evaluate_rgb(record, correct, reference, root)["semantic_goal"]["exact_match"]
    wrong = compile_choices(ChoicePrediction.model_validate(form(right, left)), q.regions)
    assert not evaluate_rgb(record, wrong, reference, root)["semantic_goal"]["exact_match"]
    assert "reference_id" not in json.dumps(correct.model_dump())


@pytest.mark.parametrize("failure", ["missing_choice", "unknown_part", "unknown_state", "extra_goal"])
def test_abstentions_and_extra_goals_never_become_success(choice_dataset, failure):
    root, record = choice_dataset
    q = build_choice_request("test", (root / "rgb.png").read_bytes(), "test")
    left, right = side_choices(q)
    payload = form(left, right)
    if failure == "missing_choice":
        payload["goals"][0]["object_choice"] = 0
    elif failure == "unknown_part":
        payload["goals"][0]["target_part"] = "unknown"
    elif failure == "unknown_state":
        payload["goals"].append(
            {
                "state": "unknown",
                "object_choice": 0,
                "object_part": "unknown",
                "target_choice": 0,
                "target_part": "whole",
            }
        )
    else:
        payload["goals"].append(
            {
                "state": "open",
                "object_choice": right,
                "object_part": "whole",
                "target_choice": 0,
                "target_part": "whole",
            }
        )
    parsed = ChoicePrediction.model_validate(payload)
    if failure != "extra_goal":
        with pytest.raises(ValueError):
            compile_choices(parsed, q.regions)
    else:
        assert not evaluate_rgb(record, compile_choices(parsed, q.regions), make_reference(record), root)[
            "semantic_goal"
        ]["exact_match"]


def test_part_selector_and_multiple_final_goals_survive_compilation(choice_dataset):
    root, _ = choice_dataset
    q = build_choice_request("test", (root / "rgb.png").read_bytes(), "test")
    a, b = side_choices(q)
    payload = form(a, b)
    payload["goals"][0].update(state="inside", target_part="bottom_drawer")
    payload["goals"].append(
        {
            "state": "closed",
            "object_choice": b,
            "object_part": "bottom_drawer",
            "target_choice": 0,
            "target_part": "whole",
        }
    )
    compiled = compile_choices(ChoicePrediction.model_validate(payload), q.regions)
    assert [g.relation for g in compiled.goals] == ["In", "Close"]
    assert compiled.goals[0].arguments[1] == compiled.goals[1].arguments[0]
    assert not compiled.stages and not compiled.constraints


def test_provider_rejects_tampered_rgb_choices_before_api(choice_dataset):
    root, _ = choice_dataset
    q = build_choice_request("test", (root / "rgb.png").read_bytes(), "test")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text="{}", usage=None, model_dump=lambda mode: {"status": "completed"})

    config = {
        "model": "fake",
        "image_detail": "high",
        "reasoning_effort": "medium",
        "max_output_tokens": 8000,
    }
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    provider = ChoiceOpenAIProvider(config, client=client)
    bad = replace(q, payload={**q.payload, "candidates": [{"identifier": "bowl_1"}]})
    with pytest.raises(ValueError):
        provider.infer(bad)
    tampered = copy.deepcopy(q.regions)
    tampered[0]["point"] = [0.01, 0.99]
    with pytest.raises(ValueError):
        provider.infer(replace(q, regions=tampered))
    assert not calls
    provider.infer(q)
    sent = calls[0]
    assert sum(item["type"] == "input_image" for item in sent["input"][0]["content"]) == 3
    assert "bowl_1" not in json.dumps(sent) and "DO_NOT_SEND_123" not in json.dumps(sent)


@pytest.fixture
def choice_config(tmp_path, choice_dataset, monkeypatch):
    monkeypatch.setattr("vlm_tamp.rgb_choice_trial.ROOT", tmp_path)
    root, record = choice_dataset
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Fill these final-goal choices.")
    reference = tmp_path / "references.json"
    freeze_references(root, reference)
    config = {
        "model": "fake",
        "conditions": ["rgb_choices"],
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
        "image_token_allowance": 16384,
        "timeout_seconds": 10,
        "pricing": {"input_per_million": 0.75, "output_per_million": 4.5},
    }
    request = build_choice_request(
        record["model_inputs"]["instruction"], (root / "rgb.png").read_bytes(), prompt.read_text()
    )
    protocol = {
        "config": config,
        "prompt_sha256": digest(prompt.read_bytes()),
        "dataset_sha256": file_sha(root / "records.jsonl"),
        "references_sha256": file_sha(reference),
        "requests": {"test-0": audit_request(request)},
        "implementation_hashes": {},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    return path, config, protocol


class FakeChoiceProvider:
    def __init__(self):
        self.calls = 0

    def infer(self, request):
        self.calls += 1
        a, b = side_choices(request)
        return ProviderResponse(
            json.dumps(form(a, b)),
            {
                "status": "completed",
                "model": "fake",
                "usage": {"output_tokens_details": {"reasoning_tokens": 5}},
            },
            100,
            20,
        )


def test_paid_guard_saved_request_and_full_workflow(choice_config, choice_dataset, tmp_path):
    provider = FakeChoiceProvider()
    path, _, _ = choice_config
    with pytest.raises(PermissionError, match="allow-paid"):
        run_choice_trial(choice_dataset[0], path, tmp_path / "forbidden", provider=provider)
    assert provider.calls == 0
    out = tmp_path / "run"
    manifest = run_choice_trial(choice_dataset[0], path, out, allow_paid=True, provider=provider)
    assert manifest["offline_mock"] and manifest["ledger"]["api_attempts"] == provider.calls == 1
    assert json.loads((out / "summary.json").read_text())["semantic_complete_goals"] == 1
    assert (out / "input-00/request.json").exists()
    with pytest.raises(FileExistsError):
        run_choice_trial(choice_dataset[0], path, out, allow_paid=True, provider=provider)


def test_budget_and_unknown_cost_stop_before_further_calls(choice_config, choice_dataset, tmp_path):
    path, config, protocol = choice_config
    config["max_budget_usd"] = 1e-9
    path.write_text(json.dumps(config))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    provider = FakeChoiceProvider()
    manifest = run_choice_trial(
        choice_dataset[0], path, tmp_path / "stopped", allow_paid=True, provider=provider
    )
    assert provider.calls == manifest["ledger"]["api_attempts"] == 0


def test_api_failure_is_redacted_and_reserved(choice_config, choice_dataset, tmp_path):
    class FailingProvider(FakeChoiceProvider):
        def infer(self, request):
            self.calls += 1
            raise RuntimeError("credential material must not be printed")

    provider = FailingProvider()
    out = tmp_path / "failed"
    manifest = run_choice_trial(choice_dataset[0], choice_config[0], out, allow_paid=True, provider=provider)
    assert manifest["ledger"]["api_attempts"] == provider.calls == 1
    assert manifest["ledger"]["unknown_usage_reserved_usd"] > 0
    assert "credential material" not in (out / "predictions.jsonl").read_text()


def test_invalid_form_cannot_insert_hidden_fields_or_unused_targets():
    payload = form(1, 2)
    payload["goals"][0]["reference_id"] = "bowl_1"
    with pytest.raises(ValidationError):
        ChoicePrediction.model_validate(payload)
    payload = form(1, 2)
    payload["goals"][0]["state"] = "closed"
    with pytest.raises(ValidationError):
        ChoicePrediction.model_validate(payload)
