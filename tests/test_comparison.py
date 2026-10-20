import json

import pytest

from vlm_tamp.comparison import compare_experiments
from vlm_tamp.evaluation import evaluate_saved
from vlm_tamp.experiment import run_experiment


def test_paired_comparison_rejects_changed_references(dataset, config, tmp_path):
    root, _ = dataset
    outputs = [tmp_path / "before", tmp_path / "after"]
    for output in outputs:
        run_experiment(root, config[0], output, cache=tmp_path / "cache")
        evaluate_saved(root, output)
    report = compare_experiments(*outputs, tmp_path / "comparison")
    assert report["paired_predictions"] == 4 and report["offline_mock"]
    assert all(r["schema_valid"]["regressed"] == 0 for r in report["groups"])
    assert "OFFLINE MOCK" in (tmp_path / "comparison/comparison.md").read_text()
    manifest = outputs[1] / "evaluation_manifest.json"
    content = json.loads(manifest.read_text())
    content["simulator_references_sha256"] = "different"
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="identical"):
        compare_experiments(*outputs, tmp_path / "incompatible")


def test_controlled_backbone_requires_frozen_prompt_schema_and_generation(dataset, config, tmp_path):
    root, _ = dataset
    outputs = [tmp_path / "baseline", tmp_path / "comparison"]
    for output in outputs:
        run_experiment(root, config[0], output, cache=tmp_path / "cache")
        evaluate_saved(root, output)
    assert compare_experiments(*outputs, tmp_path / "valid", controlled_backbone=True)["controlled_backbone"]
    path = outputs[1] / "manifest.json"
    content = json.loads(path.read_text())
    content["postprocessing"] = {"method": "different_linker"}
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="postprocessing"):
        compare_experiments(*outputs, tmp_path / "different-pipeline", controlled_backbone=True)
    content.pop("postprocessing")
    content["config"]["prompt_sha256"] = "modified"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="frozen prompt"):
        compare_experiments(*outputs, tmp_path / "bad", controlled_backbone=True)
