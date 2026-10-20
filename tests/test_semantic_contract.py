import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vlm_tamp.dataset import build_request, read_records
from vlm_tamp.evaluation import evaluate_example
from vlm_tamp.experiment import load_config, request_key
from vlm_tamp.grounding import GroundingPrediction, strict_output_schema


def modern_prediction():
    return {
        "schema_version": "1.2",
        "entities": [
            {"id": "bowl", "category": "bowl", "description": "bowl", "reference_id": "bowl_1"},
            {"id": "plate", "category": "plate", "description": "plate", "reference_id": "plate_1"},
        ],
        "goals": [
            {
                "name": "On",
                "arguments": ["bowl", "plate"],
                "intent": "task_outcome",
                "evidence_text": "put the bowl on the plate",
            }
        ],
        "geometry_targets": [
            {
                "id": "plate_surface",
                "entity": "plate",
                "role": "support_surface",
                "description": "supporting contact surface",
                "parameters": [{"name": "surface_frame", "status": "unresolved", "value": None}],
            }
        ],
    }


def test_geometry_description_preserves_object_goal_identity(dataset):
    root, record = dataset
    request = build_request(record["model_inputs"], root, "candidates", "prompt")
    prediction = GroundingPrediction.model_validate(modern_prediction(), context=request.validation_context)
    assert prediction.goals[0].arguments == ["bowl", "plate"]
    assert prediction.geometry_targets[0].entity == "plate"
    assert "DO_NOT_SEND_123" not in json.dumps(request.validation_context)
    payload = modern_prediction()
    payload["goals"][0]["arguments"][1] = "plate_surface"
    with pytest.raises(ValidationError, match="declared semantic"):
        GroundingPrediction.model_validate(payload, context=request.validation_context)


@pytest.mark.parametrize(
    "change",
    [
        "invented_reference",
        "wrong_kind",
        "unsupported_region",
        "evidence",
        "arity",
        "missing_intent",
        "missing_geometry_owner",
        "geometry_namespace",
    ],
)
def test_invalid_modern_contract(dataset, change):
    root, record = dataset
    request = build_request(record["model_inputs"], root, "candidates", "prompt")
    payload = modern_prediction()
    if change == "invented_reference":
        payload["entities"][1]["reference_id"] = "plate_region_not_supplied"
    elif change == "wrong_kind":
        payload["entities"][1].update(kind="region", parent="bowl")
    elif change == "unsupported_region":
        payload["entities"].append(
            {
                "id": "invented",
                "category": "surface",
                "description": "surface",
                "kind": "region",
                "parent": "plate",
            }
        )
        payload["goals"][0]["arguments"][1] = "invented"
    elif change == "evidence":
        payload["goals"][0]["evidence_text"] = "a benchmark answer that was not provided"
    elif change == "arity":
        payload["goals"][0]["arguments"] = ["bowl"]
    elif change == "missing_intent":
        payload["goals"][0].pop("intent")
    elif change == "missing_geometry_owner":
        payload["geometry_targets"][0]["entity"] = "missing"
    else:
        payload["geometry_targets"][0]["id"] = "plate"
    with pytest.raises(ValidationError):
        GroundingPrediction.model_validate(payload, context=request.validation_context)


def test_real_supplied_region_is_preserved_and_parent_checked(dataset):
    root, record = dataset
    record["model_inputs"]["candidates"].append(
        {"identifier": "plate_support_region", "category": "region", "kind": "region", "parent": "plate_1"}
    )
    request = build_request(record["model_inputs"], root, "candidates", "prompt")
    payload = modern_prediction()
    payload["entities"].append(
        {
            "id": "support",
            "category": "region",
            "description": "supplied support site",
            "kind": "region",
            "parent": "plate",
            "reference_id": "plate_support_region",
        }
    )
    payload["goals"][0]["arguments"][1] = "support"
    assert (
        GroundingPrediction.model_validate(payload, context=request.validation_context)
        .entities[-1]
        .reference_id
        == "plate_support_region"
    )
    payload["entities"][-1]["parent"] = "bowl"
    with pytest.raises(ValidationError, match="supplied parent identity"):
        GroundingPrediction.model_validate(payload, context=request.validation_context)


def test_parent_cycles_and_missing_local_parents_are_rejected():
    payload = modern_prediction()
    payload["entities"][0]["parent"] = "plate"
    payload["entities"][1]["parent"] = "bowl"
    with pytest.raises(ValidationError, match="acyclic"):
        GroundingPrediction.model_validate(payload)
    payload = modern_prediction()
    payload["entities"][1]["kind"] = "part"
    with pytest.raises(ValidationError, match="declared local parent"):
        GroundingPrediction.model_validate(payload)


def test_image_and_identity_claims_are_request_bound(dataset):
    root, record = dataset
    payload = modern_prediction()
    for entity in payload["entities"]:
        entity["reference_id"] = None
    GroundingPrediction.model_validate(
        payload, context=build_request(record["model_inputs"], root, "rgb", "prompt").validation_context
    )
    payload["entities"][0].update(point=[0.2, 0.2], camera="agentview")
    with pytest.raises(ValidationError, match="requires an image"):
        GroundingPrediction.model_validate(
            payload,
            context=build_request(record["model_inputs"], root, "language", "prompt").validation_context,
        )
    with pytest.raises(ValidationError, match="not supplied"):
        GroundingPrediction.model_validate(
            modern_prediction(),
            context=build_request(record["model_inputs"], root, "rgb", "prompt").validation_context,
        )


def test_uncertain_terminal_intent_keeps_strict_scoring_separate(dataset):
    root, record = dataset
    payload = modern_prediction()
    payload["goals"][0]["intent"] = "uncertain"
    with pytest.raises(ValidationError, match="explanatory abstention"):
        GroundingPrediction.model_validate(payload)
    payload["abstentions"] = ["Whether this relation is meant to persist is unclear."]
    row = {
        "prediction": payload,
        "status": "ok",
        "condition": "candidates",
        "provider": "mock",
        "model": "mock",
        "offline_mock": True,
        "latency_seconds": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0,
        "cached": False,
    }
    result = evaluate_example(record, row, root)
    assert not result["symbolic"]["exact_match"] and result["symbolic_content"]["exact_match"]
    assert result["reliability"]["terminal_intent_abstentions"] == 1


def test_access_stage_and_explicit_terminal_open_are_distinct():
    base = {
        "schema_version": "1.2",
        "entities": [
            {"id": "cabinet", "category": "cabinet", "description": "cabinet"},
            {
                "id": "drawer",
                "category": "drawer",
                "description": "drawer",
                "kind": "region",
                "parent": "cabinet",
            },
            {"id": "bowl", "category": "bowl", "description": "bowl"},
        ],
        "goals": [
            {
                "name": "In",
                "arguments": ["bowl", "drawer"],
                "intent": "task_outcome",
                "evidence_text": "put the bowl inside",
            }
        ],
        "stages": [
            {
                "id": "open",
                "action": "open",
                "entities": ["drawer"],
                "subgoals": [{"name": "Open", "arguments": ["drawer"]}],
            }
        ],
    }
    parsed = GroundingPrediction.model_validate(base)
    assert [g.name for g in parsed.goals] == ["In"] and parsed.stages[0].subgoals[0].name == "Open"
    base["goals"].append(
        {
            "name": "Open",
            "arguments": ["drawer"],
            "intent": "explicit_terminal",
            "evidence_text": "leave the drawer open",
        }
    )
    assert len(GroundingPrediction.model_validate(base).goals) == 2


def test_actual_saved_plate_region_failure_is_detected_without_goal_answers():
    project = Path(__file__).resolve().parents[1]
    predictions = project / "results/logic-markers-54mini/predictions.jsonl"
    if not predictions.exists():
        pytest.skip("Saved local pilot not available in a fresh checkout")
    rows = [json.loads(s) for s in predictions.read_text().splitlines()]
    row = next(
        r for r in rows if r["condition"] == "markers" and r["example_id"].startswith("libero_spatial")
    )
    records, root = read_records(project / "data/pilot")
    record = next(r for r in records if r["example_id"] == row["example_id"])
    payload = copy.deepcopy(row["prediction"])
    assert GroundingPrediction.model_validate(payload).schema_version == "1.1"
    payload["schema_version"] = "1.2"
    for goal in payload["goals"]:
        goal.update(intent="task_outcome", evidence_text=record["model_inputs"]["instruction"])
    request = build_request(record["model_inputs"], root, "candidates", "prompt")
    with pytest.raises(ValidationError, match="Unsupported derived region"):
        GroundingPrediction.model_validate(payload, context=request.validation_context)


def test_legacy_frozen_schema_and_cache_keys_remain_replayable():
    project = Path(__file__).resolve().parents[1]
    manifest_file = project / "results/logic-markers-54mini/manifest.json"
    if not manifest_file.exists():
        pytest.skip("Saved local pilot not available")
    from vlm_tamp.libero import ROOT
    import vlm_tamp.experiment as experiment

    # Config fixtures normally isolate ROOT; this check explicitly restores the real project root.
    old_root = experiment.ROOT
    try:
        experiment.ROOT = ROOT
        config = load_config(project / "configs/logic_markers_54mini.json")
    finally:
        experiment.ROOT = old_root
    assert config["schema_version"] == "1.1"
    frozen = json.loads((project / "configs/frozen_logic_markers_protocol.json").read_text())
    assert (
        hashlib.sha256(json.dumps(strict_output_schema("1.1"), sort_keys=True).encode()).hexdigest()
        == frozen["schema_sha256"]
    )
    records, root = read_records(project / "data/pilot")
    record = next(r for r in records if r["example_id"] == config["example_ids"][0])
    request = build_request(record["model_inputs"], root, "rgb", config["prompt_text"])
    row = next(
        json.loads(s)
        for s in (manifest_file.parent / "predictions.jsonl").read_text().splitlines()
        if json.loads(s)["example_id"] == record["example_id"] and json.loads(s)["condition"] == "rgb"
    )
    assert request_key(config, request) == row["request_hash"]
