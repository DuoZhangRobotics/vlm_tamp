from vlm_tamp.annotations import ReferenceAnnotation
from vlm_tamp.evaluation import (
    axis_error_degrees,
    evaluate_example,
    evaluate_track_b,
    reachability,
    score_sets,
)
from vlm_tamp.grounding import GroundingPrediction


def prediction():
    return {
        "entities": [
            {"id": "a", "category": "bowl", "description": "bowl", "reference_id": "bowl_1"},
            {"id": "b", "category": "plate", "description": "plate", "reference_id": "plate_1"},
        ],
        "goals": [{"name": "On", "arguments": ["a", "b"]}],
    }


def row(p):
    return {
        "prediction": p,
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


def test_order_identity_and_category_are_distinct(dataset):
    root, record = dataset
    p = prediction()
    assert evaluate_example(record, row(p), root)["symbolic"]["exact_match"]
    p["goals"][0]["arguments"].reverse()
    assert evaluate_example(record, row(p), root)["symbolic"]["f1"] == 0
    p = prediction()
    p["entities"][0]["reference_id"] = None
    evaluated = evaluate_example(record, row(p), root)
    assert evaluated["symbolic"]["exact_match"] is False
    assert evaluated["category_goals"]["exact_match"] is True


def test_visual_localization_is_not_identity_credit(dataset):
    root, record = dataset
    p = prediction()
    p["entities"][0].update(point=[0.1, 0.4], bbox=[0, 0, 0.5, 1], camera="agentview")
    result = evaluate_example(record, row(p), root)
    assert result["visual"]["bbox_iou_mean"] == 1
    assert result["visual"]["point_in_mask_accuracy"] == 1
    p["entities"][0]["point"] = [0.9, 0.4]
    assert evaluate_example(record, row(p), root)["visual"]["point_in_mask_accuracy"] == 0


def test_unconstrained_grounding_resolves_only_measured_pixels(dataset):
    root, record = dataset
    p = prediction()
    for entity, point in zip(p["entities"], [[0.1, 0.4], [0.9, 0.4]]):
        entity.update(reference_id=None, point=point, camera="agentview")
    result = evaluate_example(record, row(p), root)
    assert result["symbolic"]["exact_match"] is True
    assert result["visual"]["point_in_mask_accuracy"] == 1
    p["entities"][0]["point"] = [0.9, 0.4]
    assert evaluate_example(record, row(p), root)["symbolic"]["exact_match"] is False


def test_axis_frames_and_sign():
    assert axis_error_degrees([1, 0, 0], [-1, 0, 0], "world", "world") == 0
    assert axis_error_degrees([1, 0, 0], [0, 1, 0], "world", "world") == 90
    assert axis_error_degrees([1, 0, 0], [1, 0, 0], "camera", "world") is None
    assert axis_error_degrees([0, 0, 0], [1, 0, 0], "world", "world") is None


def test_partial_annotations_and_alternative_valid_plans():
    p = GroundingPrediction.model_validate(
        {"stages": [{"id": "s1", "action": "open"}, {"id": "s2", "action": "unlisted"}]}
    )
    a = ReferenceAnnotation.model_validate(
        {
            "example_id": "x",
            "status": "reviewed",
            "provenance": [{"source": "human", "evidence": "test-only reference", "reviewer": "fixture"}],
            "coverage": {"stages": True, "stages_complete": False},
            "alternatives": [
                {"id": "one", "stages": [{"id": "x", "action": "close"}]},
                {"id": "two", "stages": [{"id": "y", "action": "open"}]},
            ],
        }
    )
    result = evaluate_track_b(p, a)
    assert result["alternative"] == "two"
    assert result["stages"]["recall"] == 1
    assert result["stages"]["precision"] is None
    assert result["stages"]["unjudged_predictions"] == 1
    a.status = "draft"
    assert not evaluate_track_b(p, a)["evaluable"]


def test_dependencies_are_partial_orders():
    assert reachability([("a", "b"), ("b", "c")]) == {("a", "b"), ("b", "c"), ("a", "c")}
    assert ("a", "a") in reachability([("a", "b"), ("b", "a")])
    assert score_sets(["unlisted"], ["required"], complete=False)["fp"] is None


def test_failures_remain_in_symbolic_denominator(dataset):
    root, record = dataset
    r = row(None)
    r["status"] = "validation_failure"
    result = evaluate_example(record, r, root)
    assert result["symbolic"]["fn"] == 1 and result["reliability"]["validation_failure"]


def test_language_only_cannot_receive_instance_or_pixel_credit(dataset):
    root, record = dataset
    p = prediction()
    p["entities"][0].update(point=[0.1, 0.4], camera="agentview")
    r = row(p)
    r["condition"] = "language"
    result = evaluate_example(record, r, root)
    assert not result["symbolic"]["exact_match"]
    assert result["reliability"]["unsupported_identity_claims"] == 2
    assert result["reliability"]["unsupported_language_localization"] == 1
    assert result["visual"]["points_evaluable"] == 0


def test_language_only_cannot_receive_entity_linked_stage_credit(dataset):
    root, record = dataset
    p = prediction()
    p["stages"] = [{"id": "s", "action": "place", "entities": ["a", "b"]}]
    annotation = ReferenceAnnotation.model_validate(
        {
            "example_id": record["example_id"],
            "status": "reviewed",
            "provenance": [{"source": "human", "evidence": "test-only annotation", "reviewer": "fixture"}],
            "coverage": {"stages": True},
            "alternatives": [
                {
                    "id": "valid",
                    "stages": [{"id": "ref", "action": "place", "entities": ["bowl_1", "plate_1"]}],
                }
            ],
        }
    )
    r = row(p)
    assert evaluate_example(record, r, root, annotation)["track_b"]["stages"]["recall"] == 1
    r["condition"] = "language"
    assert evaluate_example(record, r, root, annotation)["track_b"]["stages"]["recall"] == 0


def test_logical_abstention_diagnostic_preserves_strict_instance_scoring(dataset):
    root, record = dataset
    p = prediction()
    p["goal_logic"] = "unresolved"
    result = evaluate_example(record, row(p), root)
    assert result["symbolic"]["exact_match"] is False
    assert result["symbolic_content"]["exact_match"] is True
    assert result["reliability"]["logical_status_abstention"] is True
    assert result["visual"]["visible_whole_object_goal_roles"]["correct"] == 2
    p["goals"][0]["arguments"].reverse()
    r = row(p)
    r["condition"] = "markers"
    result = evaluate_example(record, r, root)
    assert result["visual"]["visible_whole_object_goal_roles"]["correct"] == 0
