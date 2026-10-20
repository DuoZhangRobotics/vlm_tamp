import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vlm_tamp.dataset import build_request
from vlm_tamp.rgb_evaluation import evaluate_rgb, make_reference, normalize_goal
from vlm_tamp.rgb_grounding import ARITIES, RGBPrediction, rgb_output_schema
from vlm_tamp.rgb_trial import RGBOpenAIProvider


def image_prediction():
    return {
        "entities": [
            {"id": "item", "category": "bowl", "description": "bowl", "point": [0.2, 0.4]},
            {"id": "target", "category": "plate", "description": "plate", "point": [0.8, 0.4]},
        ],
        "goals": [{"relation": "On", "arguments": ["item", "target"]}],
    }


def test_rgb_schema_matches_supported_relation_and_arity():
    goals = rgb_output_schema()["$defs"]["RGBGoal"]["anyOf"]
    assert {g["properties"]["relation"]["const"] for g in goals} == set(ARITIES)
    for goal in goals:
        name = goal["properties"]["relation"]["const"]
        assert (
            goal["properties"]["arguments"]["minItems"]
            == goal["properties"]["arguments"]["maxItems"]
            == ARITIES[name]
        )
    prediction = image_prediction()
    prediction["goals"][0]["relation"] = "InFrontOf"
    RGBPrediction.model_validate(prediction)
    prediction["goals"][0]["relation"] = "UnspecifiedCustomPredicate"
    with pytest.raises(ValidationError):
        RGBPrediction.model_validate(prediction)


@pytest.mark.parametrize(
    "failure", ["reference_id", "wrong_goal_id", "cycle", "empty_box", "wrong_arity", "self_verification"]
)
def test_invalid_rgb_contract(failure):
    p = image_prediction()
    if failure == "reference_id":
        p["entities"][0]["reference_id"] = "hidden_sim_id"
    elif failure == "wrong_goal_id":
        p["goals"][0]["arguments"][0] = "missing"
    elif failure == "cycle":
        p["entities"][0].update(kind="region", selector="interior", parent="item")
    elif failure == "empty_box":
        p["entities"][0]["bbox"] = [0.5, 0.5, 0.4, 0.7]
    elif failure == "wrong_arity":
        p["goals"][0]["arguments"] = ["item"]
    else:
        p["constraints"] = [
            {
                "description": "grasp",
                "entities": ["item"],
                "scope": "path",
                "requirement": "required",
                "verified": True,
            }
        ]
    with pytest.raises(ValidationError):
        RGBPrediction.model_validate(p)


def test_rgb_identity_requires_image_not_category_or_name(dataset):
    root, record = dataset
    reference = make_reference(record)
    p = image_prediction()
    assert evaluate_rgb(record, RGBPrediction.model_validate(p), reference, root)["semantic_goal"][
        "exact_match"
    ]
    p["entities"][0]["point"] = [0.8, 0.4]
    assert not evaluate_rgb(record, RGBPrediction.model_validate(p), reference, root)["semantic_goal"][
        "exact_match"
    ]
    p["entities"][0]["point"] = None
    p["entities"][0]["id"] = "bowl_1"
    p["goals"][0]["arguments"][0] = "bowl_1"
    assert not evaluate_rgb(record, RGBPrediction.model_validate(p), reference, root)["semantic_goal"][
        "exact_match"
    ]


def test_region_id_not_required_for_containment_and_wrong_owner_fails(dataset):
    root, record = dataset
    record = copy.deepcopy(record)
    record["reference"]["entities"].append(
        {"identifier": "plate_1_contain_region", "category": "region", "kind": "region", "parent": "plate_1"}
    )
    record["reference"]["goal_predicates"] = [
        {"name": "In", "arguments": ["bowl_1", "plate_1_contain_region"]}
    ]
    reference = make_reference(record)
    p = image_prediction()
    p["goals"][0]["relation"] = "In"
    assert evaluate_rgb(record, RGBPrediction.model_validate(p), reference, root)["semantic_goal"][
        "exact_match"
    ]
    p["entities"].append(
        {
            "id": "interior",
            "category": "region",
            "description": "interior",
            "kind": "region",
            "parent": "item",
            "selector": "interior",
        }
    )
    p["goals"][0]["arguments"][1] = "interior"
    assert not evaluate_rgb(record, RGBPrediction.model_validate(p), reference, root)["semantic_goal"][
        "exact_match"
    ]


def test_wrong_drawer_level_and_extra_goal_are_not_forgiven(dataset):
    root, record = dataset
    record = copy.deepcopy(record)
    record["reference"]["entities"][1]["category"] = "wooden_cabinet"
    for level in ("top", "bottom"):
        record["reference"]["entities"].append(
            {
                "identifier": "plate_1_" + level + "_region",
                "category": "region",
                "kind": "region",
                "parent": "plate_1",
            }
        )
    record["reference"]["goal_predicates"] = [
        {"name": "In", "arguments": ["bowl_1", "plate_1_bottom_region"]}
    ]
    reference = make_reference(record)
    p = image_prediction()
    p["entities"].append(
        {
            "id": "drawer",
            "category": "drawer",
            "description": "bottom drawer",
            "kind": "part",
            "parent": "target",
            "selector": "bottom_drawer",
        }
    )
    p["goals"] = [{"relation": "In", "arguments": ["item", "drawer"]}]
    assert evaluate_rgb(record, RGBPrediction.model_validate(p), reference, root)["semantic_goal"][
        "exact_match"
    ]
    p["entities"][-1]["selector"] = "top_drawer"
    assert not evaluate_rgb(record, RGBPrediction.model_validate(p), reference, root)["semantic_goal"][
        "exact_match"
    ]
    p["entities"][-1]["selector"] = "bottom_drawer"
    p["goals"].append({"relation": "Open", "arguments": ["drawer"]})
    assert not evaluate_rgb(record, RGBPrediction.model_validate(p), reference, root)["semantic_goal"][
        "exact_match"
    ]


def test_bowl_intent_follows_published_instruction_without_global_on_in_equivalence(dataset):
    _, record = dataset
    record = copy.deepcopy(record)
    record["reference"]["entities"][1]["category"] = "akita_black_bowl"
    record["model_inputs"]["instruction"] = "put the cheese in the bowl"
    assert make_reference(record)["goals"][0][0] == "In"
    record["model_inputs"]["instruction"] = "put the cheese on the bowl"
    assert make_reference(record)["goals"][0][0] == "On"
    assert normalize_goal("On", [("a", "whole"), ("b", "whole")]) != normalize_goal(
        "In", [("a", "whole"), ("b", "whole")]
    )


def test_localization_resolver_does_not_use_goal_answers(dataset):
    from vlm_tamp.rgb_evaluation import resolve_entities

    root, record = dataset
    p = RGBPrediction.model_validate(image_prediction())
    before = resolve_entities(p, record, root)[0]
    altered = copy.deepcopy(record)
    altered["reference"]["goal_predicates"] = [{"name": "Close", "arguments": ["plate_1"]}]
    assert resolve_entities(p, altered, root)[0] == before


def test_provider_only_sends_rgb_instruction_and_enum_schema(dataset):
    root, record = dataset
    captured = []

    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(output_text="{}", usage=None, model_dump=lambda **_: {})

    provider = RGBOpenAIProvider(
        {"model": "fake", "image_detail": "high", "max_output_tokens": 8000},
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
    provider.infer(build_request(record["model_inputs"], root, "rgb", "semantic instructions"))
    request = captured[0]
    text = json.loads(request["input"][0]["content"][0]["text"])
    assert set(text) == {"instruction"}
    assert "DO_NOT_SEND_123" not in json.dumps(request)
    assert "candidates" not in text and "geometry" not in text
    assert request["text"]["format"]["schema"]["$defs"]["RGBGoal"]["anyOf"]
    with pytest.raises(ValueError, match="original image only"):
        provider.infer(build_request(record["model_inputs"], root, "candidates", "prompt"))
