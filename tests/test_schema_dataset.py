import json

import pytest
from pydantic import ValidationError

from vlm_tamp.dataset import build_request, safe_path, validate_dataset
from vlm_tamp.grounding import Entity, GroundingPrediction, strict_output_schema


@pytest.mark.parametrize(
    "condition,images,oracle",
    [("language", False, False), ("rgb", True, False), ("candidates", True, True), ("geometry", True, True)],
)
def test_input_boundary_and_distractors(dataset, condition, images, oracle):
    root, record = dataset
    record["model_inputs"]["candidates"][0]["goal_predicates"] = "INJECTED_GOAL"
    request = build_request(record["model_inputs"], root, condition, "prompt")
    assert (request.image is not None) == images
    assert ("candidates" in request.payload) == oracle
    assert "DO_NOT_SEND_123" not in request.text
    assert "INJECTED_GOAL" not in request.text
    assert "test-0" not in request.text and "rgb.png" not in request.text
    if oracle:
        assert len(request.payload["candidates"]) == 3
        assert "distractor_1" in request.text


def test_dataset_integrity(dataset):
    root, record = dataset
    assert validate_dataset(root)["examples"] == 1
    record["metadata"]["initial_state_sha256"] = "wrong"
    (root / "records.jsonl").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_dataset(root)
    with pytest.raises(ValueError, match="escapes"):
        safe_path(root, "../hidden.txt")


@pytest.mark.parametrize(
    "extra",
    [
        {"point": [1.1, 0.3], "camera": "agentview"},
        {"bbox": [0.5, 0.5, 0.4, 0.8], "camera": "agentview"},
        {"point": [0.2, 0.2]},
        {"ambiguous": True, "reference_id": "fake"},
    ],
)
def test_invalid_geometry_and_silent_disambiguation(extra):
    with pytest.raises(ValidationError):
        Entity(id="mug", category="mug", description="mug", **extra)


def test_schema_closure_and_no_hallucinated_verification():
    schema = strict_output_schema()

    def check(node):
        if isinstance(node, dict):
            if "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            assert "prefixItems" not in node
            for v in node.values():
                check(v)
        if isinstance(node, list):
            for v in node:
                check(v)

    check(schema)
    with pytest.raises(ValidationError):
        GroundingPrediction.model_validate(
            {
                "constraints": [
                    {
                        "id": "x",
                        "category": "path",
                        "type": "grasp",
                        "entities": [],
                        "scope": "path",
                        "requirement": "required",
                        "evidence": "observation",
                        "verified": True,
                    }
                ]
            }
        )
    with pytest.raises(ValidationError):
        GroundingPrediction.model_validate({"goals": [{"name": "Open", "arguments": ["undeclared"]}]})


def articulation_payload(joint_type="unknown"):
    return {
        "schema_version": "1.1",
        "entities": [{"id": "drawer", "category": "drawer", "description": "drawer", "kind": "part"}],
        "constraints": [
            {
                "id": "motion",
                "category": "articulation",
                "type": "follow_joint",
                "joint_type": joint_type,
                "articulated_part": "drawer",
                "entities": ["drawer"],
                "scope": "path",
                "requirement": "required",
                "evidence": "observation",
            }
        ],
    }


@pytest.mark.parametrize("joint_type", ["prismatic", "revolute", "unknown"])
def test_articulation_mechanism_is_explicit(joint_type):
    p = GroundingPrediction.model_validate(articulation_payload(joint_type))
    assert p.constraints[0].joint_type == joint_type and p.constraints[0].articulated_part == "drawer"
    assert strict_output_schema("1.1")["properties"]["schema_version"] == {"type": "string", "const": "1.1"}
    assert strict_output_schema()["properties"]["schema_version"] == {"type": "string", "const": "1.2"}


@pytest.mark.parametrize(
    "change", ["missing_type", "missing_part", "unknown_part", "whole_object", "invalid_type"]
)
def test_invalid_articulation_contract_is_rejected(change):
    p = articulation_payload()
    if change == "missing_type":
        p["constraints"][0].pop("joint_type")
    if change == "missing_part":
        p["constraints"][0].pop("articulated_part")
    if change == "unknown_part":
        p["constraints"][0]["articulated_part"] = "missing"
    if change == "whole_object":
        p["entities"][0]["kind"] = "object"
    if change == "invalid_type":
        p["constraints"][0]["joint_type"] = "drawer_open"
    with pytest.raises(ValidationError):
        GroundingPrediction.model_validate(p)


def test_legacy_predictions_and_parent_reference_contract():
    p = articulation_payload()
    p["schema_version"] = "1.0"
    p["constraints"][0].pop("joint_type")
    p["constraints"][0].pop("articulated_part")
    assert GroundingPrediction.model_validate(p).constraints[0].joint_type is None
    p = articulation_payload()
    p["entities"][0]["parent"] = "inventory_name_not_declared"
    with pytest.raises(ValidationError, match="Parent"):
        GroundingPrediction.model_validate(p)
