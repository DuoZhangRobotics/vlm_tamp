import hashlib
import json

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from vlm_tamp.evaluation import evaluate_articulation_facts, evaluate_example
from vlm_tamp.grounding import GroundingPrediction
from vlm_tamp.references import PartFact, SimulatorReference, attach_reference, check_reference


@pytest.fixture
def verified_reference(dataset):
    root, record = dataset
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:10, :10] = 255
    Image.fromarray(mask).save(root / "part.png")
    entity = {"identifier": "bowl_1_slide_region", "category": "region", "kind": "region", "parent": "bowl_1"}
    record["reference"]["entities"].append(entity)
    record["metadata"].update(bddl_sha256="fixture-bddl", versions={"libero_commit": "fixture-commit"})
    part = PartFact(
        entity_id=entity["identifier"],
        parent_id="bowl_1",
        body="slide_body",
        joint_id="slide_joint",
        joint_type="prismatic",
        axis_local=[1, 0, 0],
        axis_world=[0, 1, 0],
        joint_position_world=[0, 0, 0],
        goal_related=True,
        mask="part.png",
        mask_sha256=hashlib.sha256((root / "part.png").read_bytes()).hexdigest(),
        visible_pixels=100,
        bbox=[0, 0, 0.5, 0.5],
        verification="site_joint_body_mask_and_state_crosscheck",
        open_predicate_at_limits=[False, True],
    )
    ref = SimulatorReference(
        example_id=record["example_id"],
        bddl_sha256="fixture-bddl",
        initial_state_sha256=record["metadata"]["initial_state_sha256"],
        image_sha256=hashlib.sha256((root / "rgb.png").read_bytes()).hexdigest(),
        libero_commit="fixture-commit",
        verifier_sha256="fixture-verifier",
        camera="agentview",
        parts=[part],
    )
    return root, record, ref


def _row(prediction, condition="rgb"):
    return {
        "prediction": prediction,
        "status": "ok",
        "condition": condition,
        "provider": "mock",
        "model": "mock",
        "offline_mock": True,
        "latency_seconds": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0,
        "cached": False,
    }


def test_verified_masks_do_not_mutate_model_inputs_or_collection(verified_reference):
    root, record, ref = verified_reference
    inputs = json.dumps(record["model_inputs"], sort_keys=True)
    enriched = attach_reference(record, root, ref)
    assert ref.parts[0].entity_id not in record["reference"]["visual"]
    assert enriched["reference"]["visual"][ref.parts[0].entity_id]["kind"] == "part"
    assert json.dumps(enriched["model_inputs"], sort_keys=True) == inputs


def test_state_or_mask_changes_invalidate_reference(verified_reference):
    root, record, ref = verified_reference
    check_reference(record, root, ref)
    record["metadata"]["initial_state_sha256"] = "modified"
    with pytest.raises(ValueError, match="initial_state"):
        check_reference(record, root, ref)
    record["metadata"]["initial_state_sha256"] = ref.initial_state_sha256
    Image.new("L", (20, 20), 0).save(root / "part.png")
    with pytest.raises(ValueError, match="mask changed"):
        check_reference(record, root, ref)


def test_nested_part_and_whole_object_pixels_are_not_confused(verified_reference):
    root, record, ref = verified_reference
    part_id = ref.parts[0].entity_id
    record["reference"]["goal_predicates"] = [{"name": "Open", "arguments": [part_id]}]
    p = {
        "entities": [
            {
                "id": "part",
                "category": "drawer",
                "description": "sliding part",
                "kind": "part",
                "point": [0.2, 0.2],
                "camera": "agentview",
            }
        ],
        "goals": [{"name": "Open", "arguments": ["part"]}],
    }
    evaluated = evaluate_example(record, _row(p), root, simulator_reference=ref)
    assert evaluated["symbolic"]["exact_match"]
    assert evaluated["articulation_facts"]["goal_part_identification_recall"] == 1
    p["entities"][0]["kind"] = "object"
    assert not evaluate_example(record, _row(p), root, simulator_reference=ref)["symbolic"]["exact_match"]


def test_missing_articulation_type_does_not_get_kinematic_credit(verified_reference):
    _, _, ref = verified_reference
    p = GroundingPrediction.model_validate(
        {
            "schema_version": "1.0",
            "entities": [
                {
                    "id": "drawer",
                    "category": "drawer",
                    "description": "drawer",
                    "kind": "part",
                    "reference_id": ref.parts[0].entity_id,
                }
            ],
            "constraints": [
                {
                    "id": "c",
                    "category": "articulation",
                    "type": "drawer_open",
                    "entities": ["drawer"],
                    "scope": "path",
                    "requirement": "required",
                    "evidence": "observation",
                }
            ],
        }
    )
    result = evaluate_articulation_facts(p, ref, {"drawer": ref.parts[0].entity_id})
    assert result["conditional_joint_type_accuracy"] is None
    assert result["goal_part_joint_type_recall"] == 0
    assert result["unclassified_articulation_declarations"] == 1


def test_declared_joint_types_and_frames_are_verified(verified_reference):
    _, _, ref = verified_reference
    p = GroundingPrediction.model_validate(
        {
            "schema_version": "1.0",
            "entities": [
                {
                    "id": "drawer",
                    "category": "drawer",
                    "description": "drawer",
                    "kind": "part",
                    "reference_id": ref.parts[0].entity_id,
                }
            ],
            "constraints": [
                {
                    "id": "c",
                    "category": "articulation",
                    "type": "follow_prismatic_joint",
                    "entities": ["drawer"],
                    "scope": "path",
                    "requirement": "required",
                    "evidence": "observation",
                    "parameters": [
                        {"name": "axis", "status": "known", "value": [0, -1, 0], "frame": "world"}
                    ],
                }
            ],
        }
    )
    result = evaluate_articulation_facts(p, ref, {"drawer": ref.parts[0].entity_id})
    assert result["conditional_joint_type_accuracy"] == 1
    assert result["axis_angular_errors_degrees"] == [0]
    p.constraints[0].parameters[0].frame = "camera"
    result = evaluate_articulation_facts(p, ref, {"drawer": ref.parts[0].entity_id})
    assert result["axis_angular_errors_degrees"] == [] and result["incompatible_axis_frames"] == 1
    p.constraints[0].type = "revolute"
    assert (
        evaluate_articulation_facts(p, ref, {"drawer": ref.parts[0].entity_id})[
            "conditional_joint_type_accuracy"
        ]
        == 0
    )


def test_false_open_crosscheck_is_rejected(verified_reference):
    _, _, ref = verified_reference
    data = ref.parts[0].model_dump()
    data["open_predicate_at_limits"] = [False, False]
    with pytest.raises(ValidationError, match="toggle"):
        PartFact.model_validate(data)


def test_explicit_joint_type_and_unknown_are_not_legacy_action_inferences(verified_reference):
    _, _, ref = verified_reference
    p = GroundingPrediction.model_validate(
        {
            "schema_version": "1.1",
            "entities": [
                {
                    "id": "drawer",
                    "category": "drawer",
                    "description": "drawer",
                    "kind": "part",
                    "reference_id": ref.parts[0].entity_id,
                }
            ],
            "constraints": [
                {
                    "id": "c",
                    "category": "articulation",
                    "type": "drawer_open",
                    "joint_type": "unknown",
                    "articulated_part": "drawer",
                    "entities": ["drawer"],
                    "scope": "path",
                    "requirement": "required",
                    "evidence": "observation",
                }
            ],
        }
    )
    result = evaluate_articulation_facts(p, ref, {"drawer": ref.parts[0].entity_id})
    assert result["explicit_unknown_joint_types"] == 1 and result["matched_type_claims"] == 0
    p.constraints[0].joint_type = "prismatic"
    result = evaluate_articulation_facts(p, ref, {"drawer": ref.parts[0].entity_id})
    assert result["explicit_known_joint_types"] == 1 and result["conditional_joint_type_accuracy"] == 1
