import copy
import hashlib
import json
from pathlib import Path

from vlm_tamp.grounding import GroundingPrediction
from vlm_tamp.linking import link_prediction
from vlm_tamp.linking import link_saved_outputs
from vlm_tamp.experiment import run_experiment
from vlm_tamp.evaluation import evaluate_saved
from vlm_tamp.providers import ProviderResponse
from vlm_tamp.visualization import gallery


def test_linking_preserves_goals_geometry_and_declared_identity():
    context = {
        "instruction": "put the bowl on the plate",
        "image_available": True,
        "candidates": [
            {"identifier": "bowl_1", "kind": "object", "parent": None},
            {"identifier": "plate_1", "kind": "object", "parent": None},
        ],
    }
    payload = {
        "schema_version": "1.2",
        "entities": [
            {"id": "bowl", "category": "bowl", "description": "bowl", "reference_id": "bowl_1"},
            {
                "id": "plate",
                "category": "plate",
                "description": "plate",
                "reference_id": "plate_1",
                "parent": "bowl_1",
            },
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
            {"id": "surface", "entity": "plate", "role": "support_surface", "description": "contact surface"}
        ],
        "stages": [
            {
                "id": "place",
                "action": "place",
                "entities": ["bowl_1", "surface"],
                "subgoals": [{"name": "On", "arguments": ["bowl_1", "surface"]}],
            }
        ],
    }
    original = copy.deepcopy(payload)
    linked, events = link_prediction(payload, context)
    assert payload == original and linked["goals"] == original["goals"]
    assert linked["geometry_targets"] == original["geometry_targets"]
    assert linked["stages"][0]["entities"] == ["bowl", "plate"]
    assert linked["entities"][1]["parent"] is None
    assert len(events) == 3
    GroundingPrediction.model_validate(linked, context=context)


def test_ambiguity_becomes_abstention_and_never_forced_resolution():
    payload = {
        "schema_version": "1.2",
        "entities": [
            {
                "id": "bowl",
                "category": "bowl",
                "description": "bowl",
                "ambiguous": True,
                "reference_id": "bowl_1",
            }
        ],
        "goals": [],
    }
    linked, events = link_prediction(payload, {"candidates": []})
    assert linked["entities"][0]["ambiguous"] is True
    assert linked["entities"][0]["reference_id"] is None
    assert events[0]["rule"] == "abstain_conflicting_identity"
    assert linked["abstentions"]


def test_linker_does_not_invent_missing_parents_or_rewrite_final_geometry_arguments():
    payload = {
        "schema_version": "1.2",
        "entities": [{"id": "plate", "reference_id": "plate_1", "parent": None}],
        "geometry_targets": [{"id": "surface", "entity": "plate"}],
        "goals": [{"name": "On", "arguments": ["plate", "surface"]}],
        "stages": [{"id": "stage", "entities": ["unknown"]}],
    }
    linked, _ = link_prediction(
        payload, {"candidates": [{"identifier": "plate_1", "parent": "missing_parent"}]}
    )
    assert linked["goals"] == payload["goals"]
    assert linked["stages"][0]["entities"] == ["unknown"]
    assert linked["entities"][0]["parent"] is None and len(linked["entities"]) == 1


def test_derived_experiment_preserves_raw_failures_costs_and_mock_label(dataset, config, tmp_path):
    root, _ = dataset
    path, c = config
    c["conditions"] = ["candidates"]
    path.write_text(json.dumps(c))

    class Provider:
        def infer(self, request):
            payload = {
                "schema_version": "1.2",
                "entities": [
                    {"id": "a", "category": "bowl", "description": "bowl", "reference_id": "bowl_1"},
                    {"id": "b", "category": "plate", "description": "plate", "reference_id": "plate_1"},
                ],
                "goals": [
                    {
                        "name": "On",
                        "arguments": ["a", "b"],
                        "intent": "task_outcome",
                        "evidence_text": request.payload["instruction"],
                    }
                ],
                "geometry_targets": [
                    {"id": "surface", "entity": "b", "role": "support_surface", "description": "surface"}
                ],
                "stages": [{"id": "place", "action": "place", "entities": ["a", "surface"]}],
            }
            return ProviderResponse(json.dumps(payload), {"test": True}, 0, 0)

    source = tmp_path / "source"
    run_experiment(root, path, source, provider=Provider())
    raw_bytes = (source / "predictions.jsonl").read_bytes()
    assert json.loads(raw_bytes)["status"] == "validation_failure"
    import vlm_tamp.linking as linking

    frozen = tmp_path / "protocol.json"
    frozen.write_text(
        json.dumps({"linker_sha256": hashlib.sha256(Path(linking.__file__).read_bytes()).hexdigest()})
    )
    output = tmp_path / "linked"
    manifest = link_saved_outputs(root, source, output, frozen)
    assert manifest["ledger"]["api_attempts"] == 0
    assert manifest["ledger"]["estimated_usage_cost_usd"] == 0
    assert (
        manifest["successful_predictions"] == 1
        and manifest["postprocessing"]["benchmark_goals_accessed"] is False
    )
    assert (source / "predictions.jsonl").read_bytes() == raw_bytes
    metrics, _ = evaluate_saved(root, output)
    assert metrics[0]["symbolic"]["exact_match"]
    assert "OFFLINE MOCK" in gallery(root, output).read_text()
