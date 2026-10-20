import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from vlm_tamp.comparison import compare_conditions
from vlm_tamp.dataset import build_request
from vlm_tamp.evaluation import evaluate_saved
from vlm_tamp.experiment import run_experiment
from vlm_tamp.markers import generate_marker_bundle, load_marker_bundle
from vlm_tamp.providers import ProviderResponse
from vlm_tamp.visualization import gallery


def test_markers_preserve_originals_include_distractors_and_exclude_goals(dataset, tmp_path):
    root, record = dataset
    original_manifest = (root / "records.jsonl").read_bytes()
    original_image = (root / "rgb.png").read_bytes()
    bundle = generate_marker_bundle(root, tmp_path / "markers")
    assert (root / "records.jsonl").read_bytes() == original_manifest
    assert (root / "rgb.png").read_bytes() == original_image
    sample = load_marker_bundle(tmp_path / "markers", root)["test-0"]
    assert {m["identifier"] for m in sample["mappings"]} == {"bowl_1", "plate_1"}
    sample["mappings"][0]["goals"] = "INJECTED_GOAL"
    request = build_request(record["model_inputs"], root, "markers", "prompt", sample)
    assert "distractor_1" in request.text
    assert "INJECTED_GOAL" not in request.text and "DO_NOT_SEND_123" not in request.text
    assert "test-0" not in request.text and "anchor_pixel" not in request.text
    assert "mask" not in request.text and "geometry" not in request.text
    assert request.source_image_sha256 == hashlib.sha256(original_image).hexdigest()
    assert request.image_hash != request.source_image_sha256
    assert Image.open(tmp_path / "markers" / bundle["examples"]["test-0"]["image"]).size == (20, 20)
    with pytest.raises(ValueError, match="separate oracle"):
        build_request(record["model_inputs"], root, "markers", "prompt")


def test_marker_numbering_and_image_are_independent_of_instruction_and_goal(dataset, tmp_path):
    root, record = dataset
    a = generate_marker_bundle(root, tmp_path / "first")["examples"]["test-0"]
    record["model_inputs"]["instruction"] = "a completely different instruction"
    record["reference"]["goal_predicates"] = [{"name": "SECRET", "arguments": ["distractor_1"]}]
    record["reference"]["goal_expression"] = ["SECRET"]
    (root / "records.jsonl").write_text(json.dumps(record))
    b = generate_marker_bundle(root, tmp_path / "second")["examples"]["test-0"]
    assert a == b


@pytest.mark.parametrize("change", ["drop_object", "anchor", "label", "image", "mask", "source"])
def test_marker_corruption_is_rejected(dataset, tmp_path, change):
    root, _ = dataset
    output = tmp_path / "markers"
    bundle = generate_marker_bundle(root, output)
    sample = bundle["examples"]["test-0"]
    if change == "drop_object":
        sample["mappings"].pop()
    elif change == "anchor":
        sample["mappings"][0]["anchor_pixel"] = [-1, 0]
    elif change == "label":
        sample["mappings"][0]["label"] = "GOAL_TARGET"
    elif change == "image":
        Image.new("RGB", (20, 20), "red").save(output / sample["image"])
    elif change == "mask":
        Image.new("L", (20, 20), 0).save(root / sample["mappings"][0]["mask"])
    elif change == "source":
        Image.new("RGB", (20, 20), "red").save(root / "rgb.png")
    (output / "markers.json").write_text(json.dumps(bundle))
    with pytest.raises(ValueError):
        load_marker_bundle(output, root)


def test_marker_anchors_are_actual_mask_pixels_including_visible_distractors(dataset, tmp_path):
    root, record = dataset
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 10] = 255
    Image.fromarray(mask).save(root / "distractor.png")
    record["reference"]["visual"]["distractor_1"] = {
        "mask": "distractor.png",
        "visible_pixels": 1,
        "bbox": [0.5, 0.5, 0.55, 0.55],
    }
    (root / "records.jsonl").write_text(json.dumps(record))
    bundle = generate_marker_bundle(root, tmp_path / "markers")
    sample = bundle["examples"]["test-0"]
    assert len(sample["mappings"]) == 3
    for entry in sample["mappings"]:
        x, y = entry["anchor_pixel"]
        assert np.array(Image.open(root / entry["mask"]))[y, x] > 0
    assert any(e["identifier"] == "distractor_1" for e in sample["mappings"])
    load_marker_bundle(tmp_path / "markers", root)


def test_marker_run_evaluation_gallery_and_paired_assistance(dataset, config, tmp_path):
    root, _ = dataset
    marker_dir = tmp_path / "markers"
    generate_marker_bundle(root, marker_dir)
    path, c = config
    c.update(conditions=["candidates", "markers"], marker_bundle=str(marker_dir))
    path.write_text(json.dumps(c))

    class Provider:
        def infer(self, request):
            ids = ["bowl_1", "plate_1"] if request.condition == "markers" else ["distractor_1", "plate_1"]
            prediction = {
                "schema_version": "1.2",
                "entities": [
                    {"id": str(i), "category": category, "description": category, "reference_id": ref}
                    for i, (category, ref) in enumerate(zip(["bowl", "plate"], ids))
                ],
                "goals": [
                    {
                        "name": "On",
                        "arguments": ["0", "1"],
                        "intent": "task_outcome",
                        "evidence_text": request.payload["instruction"],
                    }
                ],
            }
            return ProviderResponse(json.dumps(prediction), {"test": True}, 0, 0)

    experiment = tmp_path / "run"
    run_experiment(root, path, experiment, provider=Provider())
    metrics, _ = evaluate_saved(root, experiment)
    assert metrics[1]["symbolic"]["exact_match"]
    assert metrics[1]["visual"]["setting"] == "oracle_assisted"
    assert metrics[1]["visual"]["visible_whole_object_goal_roles"]["correct"] == 2
    assert "Actual model-visible oracle-labelled input" in gallery(root, experiment).read_text()
    comparison = compare_conditions(experiment, tmp_path / "comparison")
    assert comparison["pairs"] == 1
    assert comparison["baseline_object_roles_correct"] == 1
    assert comparison["assisted_object_roles_correct"] == 2
    rows = [json.loads(line) for line in (experiment / "predictions.jsonl").read_text().splitlines()]
    assert rows[0]["source_image_sha256"] == rows[1]["source_image_sha256"]
    assert rows[0]["image_sha256"] != rows[1]["image_sha256"]
    sample = load_marker_bundle(marker_dir, root)["test-0"]
    private = tmp_path / "private.env"
    private.write_text("CANARY_PRIVATE_CONTENT")
    rows[1]["input_provenance"]["image_path"] = str(private)
    (experiment / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    gallery(root, experiment)
    assert (experiment / "gallery-assets/model-input-0001.png").read_bytes() == sample["image"]
    rows[1]["candidate_inventory_sha256"] = "changed"
    (experiment / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="inventories"):
        compare_conditions(experiment, tmp_path / "invalid-comparison")
    Image.new("RGB", (20, 20), "red").save(sample["provenance"]["image_path"])
    with pytest.raises(ValueError, match="Marker image changed"):
        evaluate_saved(root, experiment)


def test_unknown_geometry_does_not_force_logical_abstention():
    from vlm_tamp.grounding import GroundingPrediction

    payload = {
        "goal_logic": "and",
        "entities": [
            {"id": "drawer", "category": "drawer", "description": "drawer", "kind": "part", "ambiguous": True}
        ],
        "goals": [{"name": "Open", "arguments": ["drawer"]}],
        "constraints": [
            {
                "id": "c",
                "category": "articulation",
                "type": "follow_joint",
                "entities": ["drawer"],
                "articulated_part": "drawer",
                "joint_type": "unknown",
                "scope": "path",
                "requirement": "conditional",
                "evidence": "planning_hypothesis",
                "parameters": [{"name": "axis", "value": None, "status": "unresolved"}],
            }
        ],
    }
    prediction = GroundingPrediction.model_validate(payload)
    assert prediction.goal_logic == "and" and prediction.constraints[0].parameters[0].value is None
    assert prediction.entities[0].reference_id is None


def test_frozen_protocol_rejects_changed_marker_bundle(dataset, config, tmp_path):
    from vlm_tamp.experiment import load_config
    from vlm_tamp.grounding import strict_output_schema

    root, _ = dataset
    marker_dir = tmp_path / "markers"
    generate_marker_bundle(root, marker_dir)
    path, c = config
    c.update(
        conditions=["markers"],
        marker_bundle=str(marker_dir),
        temperature=0,
        image_detail="high",
        example_ids=["test-0"],
    )
    path.write_text(json.dumps(c))
    loaded = load_config(path)
    protocol = {
        "schema_sha256": hashlib.sha256(
            json.dumps(strict_output_schema(), sort_keys=True).encode()
        ).hexdigest(),
        "prompt_sha256": loaded["prompt_sha256"],
        "models": [c["model"]],
        "marker_bundle_sha256": loaded["marker_bundle_sha256"],
        **{
            k: c[k] for k in ("conditions", "temperature", "image_detail", "max_output_tokens", "example_ids")
        },
    }
    frozen = tmp_path / "protocol.json"
    frozen.write_text(json.dumps(protocol))
    c["frozen_protocol"] = str(frozen)
    path.write_text(json.dumps(c))
    assert load_config(path)["frozen_protocol_sha256"]
    bundle = json.loads((marker_dir / "markers.json").read_text())
    bundle["oracle_assisted"] = False
    (marker_dir / "markers.json").write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="frozen experiment"):
        load_config(path)
