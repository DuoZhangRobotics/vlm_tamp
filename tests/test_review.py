import json

import pytest

from vlm_tamp.annotations import load_annotations
from vlm_tamp.evaluation import evaluate_track_b
from vlm_tamp.grounding import GroundingPrediction
from vlm_tamp.review import generate_review_set


def test_review_pack_is_evidence_backed_and_cannot_create_expert_credit(dataset, tmp_path):
    root, record = dataset
    record["metadata"].update(bddl_sha256="test-bddl", versions={"libero_commit": "test-commit"})
    (root / "records.jsonl").write_text(json.dumps(record))
    original = (root / "records.jsonl").read_bytes()
    output = tmp_path / "review"
    drafts = generate_review_set(root, output)
    assert len(drafts) == 1 and drafts[0].status == "draft"
    assert drafts[0].alternatives[0].stages
    assert all(not p.reviewer for p in drafts[0].provenance)
    assert all(c.verified is False for c in drafts[0].alternatives[0].constraints)
    assert not drafts[0].coverage.stages and not drafts[0].coverage.constraints
    assert not evaluate_track_b(GroundingPrediction(), drafts[0])["evaluable"]
    assert load_annotations(output / "annotations")["test-0"].status == "draft"
    page = (output / "review.html").read_text()
    assert "Human reviewer name" in page and "Confirm that you checked" in page
    assert "exportReview" in page and "annotation.status = 'reviewed'" in page
    assert (root / "records.jsonl").read_bytes() == original
    evidence = json.loads((output / "evidence/test-0.json").read_text())
    assert evidence["published_goal_predicates"] == record["reference"]["goal_predicates"]
    with pytest.raises(FileExistsError):
        generate_review_set(root, output)


def test_review_unknown_examples_are_rejected(dataset, tmp_path):
    with pytest.raises(ValueError, match="Unknown review"):
        generate_review_set(dataset[0], tmp_path / "review", example_ids=["unknown"])
