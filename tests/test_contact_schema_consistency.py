"""Published contact arguments must agree with the existing runtime requirements."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
import pytest

from vlm_tamp.robot_contracts import operation_schemas, validate_operation


def contact_arguments(operation: str) -> dict[str, Any]:
    """Build a request with an explicit contact but no selected target region."""
    result: dict[str, Any] = {"camera": "agentview", "contact_pixel_xy": [5, 5]}
    if operation != "select_surface_contact":
        result.update(
            observation_id=1,
            contact_name="gripper0_finger1_pad_collision",
            direction_world=[1, 0, 0],
        )
    return result


@pytest.mark.parametrize("operation", ["push", "hook_pull", "select_surface_contact"])
def test_exported_contact_schema_expresses_existing_target_requirement(operation: str) -> None:
    """An input rejected for absent selection must not pass the schema shown to Luna."""
    validator = Draft202012Validator(operation_schemas()[operation]["function"]["parameters"])
    for target in ({}, {"region_xyxy": None}, {"contact_part_id": None}, {"contact_part_id": ""}):
        arguments = {**contact_arguments(operation), **target}
        expected_error = (
            "contact_part_id"
            if target == {"contact_part_id": ""}
            else "Supply an original-pixel region or a current selected part ID"
        )
        with pytest.raises(ValueError, match=expected_error):
            validate_operation(operation, arguments)
        assert list(validator.iter_errors(arguments)), arguments


@pytest.mark.parametrize("operation", ["push", "hook_pull", "select_surface_contact"])
def test_existing_contact_target_alternatives_remain_valid(operation: str) -> None:
    """A region and the already supported part handles remain interchangeable selections."""
    validator = Draft202012Validator(operation_schemas()[operation]["function"]["parameters"])
    selections = [{"region_xyxy": [1, 1, 9, 9]}, {"contact_part_id": "part001-0"}]
    if operation != "select_surface_contact":
        selections.append({"moving_part_id": "part001-0"})
    for target in selections:
        arguments = {**contact_arguments(operation), **target}
        validator.validate(arguments)
        validate_operation(operation, arguments)
