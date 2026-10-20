import pytest

from vlm_tamp.rgb_role_choices import binding_schema, instruction_template, validate_bindings


def test_multicolor_and_multiple_placement_cardinality_come_from_instruction():
    one = instruction_template("put the yellow and white mug in the microwave and close it")
    assert list(one["roles"].values()) == ["the yellow and white mug", "the microwave"]
    assert [goal["state"] for goal in one["goals"]] == ["inside", "closed"]
    two = instruction_template(
        "put the white mug on the left plate and put the yellow and white mug on the right plate"
    )
    assert len(two["roles"]) == 4 and len(two["goals"]) == 2


def test_initial_qualifier_is_preserved_and_drawer_owner_is_not_a_hidden_id():
    retrieved = instruction_template(
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
    )
    assert "top drawer of the wooden cabinet" in retrieved["roles"]["slot_0"]
    drawer = instruction_template("put the apple in the bottom drawer of the wooden cabinet and close it")
    assert drawer["roles"]["slot_1"] == "the wooden cabinet"
    assert [goal["target_part"] for goal in drawer["goals"]] == ["bottom_drawer", "whole"]
    assert drawer["goals"][1]["object_part"] == "bottom_drawer"


def test_unsupported_language_and_invented_bindings_are_rejected():
    with pytest.raises(ValueError):
        instruction_template("fold the shirt")
    text = "turn on the kitchen light"
    assert instruction_template(text)["goals"][0]["state"] == "powered_on"
    assert binding_schema(text, 5)["properties"]["slot_0"]["enum"] == list(range(6))
    for payload in ({"slot_0": 6}, {"slot_0": True}, {"slot_0": 2, "slot_1": 3}, {}):
        with pytest.raises(ValueError):
            validate_bindings(payload, text, 5)
