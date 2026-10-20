"""Future ordinary pick-place blanks with no task-specific path constraint.

Choose either normalized pixels from original RGB or IDs from a numbered RGB
proposal image. Both forms ask only for source, grasp, two closing points,
target, and a grasp method. Calibrated depth remains local to the compiler.
Collision avoidance, attachment verification and robot feasibility are backend
requirements; this contract adds no upright/axis/path choice to transport.

This new adapter composes the frozen numerical compiler without editing it.
An estimated placement-point goal does not certify whole-object stability.
"""

import base64
import copy
import math
import re

from .rgbd_landmark_choices import validate_candidates
from .rgbd_numericals import compile_numericals, json_text, template as numerical_template


POINT_FIELDS = ("source_uv", "grasp_uv", "closing_a_uv", "closing_b_uv", "target_uv")
ID_FIELDS = tuple(field.removesuffix("_uv") + "_id" for field in POINT_FIELDS)
METHOD_FIELDS = ("grasp_mode",)
CONDITIONS = ("direct_rgb", "dinov3_candidates")
MODEL_FIELDS = {"instruction", "task_type", "source_description", "target_description", "blanks"}


def template(base_instruction):
    """Instruction-only ordinary whole-object On placement; no upright variant."""
    if not isinstance(base_instruction, str) or not base_instruction.strip():
        raise ValueError("A nonempty ordinary placement instruction is required")
    if re.search(r"\bwhile\b|\bwithout spilling\b|\bkeep(?:ing)? (?:it|the .+?) upright\b", base_instruction.lower()):
        raise ValueError("Task-specific transport requirements need a separate constraint contract")
    declared = numerical_template(base_instruction, upright=False)
    return {key: declared[key] for key in ("instruction", "source_description", "target_description")} | {
        "task_type": "pick_place", "blanks": list(POINT_FIELDS + METHOD_FIELDS),
    }


def blank_schema(condition="direct_rgb", candidates=None):
    if condition not in CONDITIONS:
        raise ValueError("Unknown pick-place input method")
    if condition == "direct_rgb":
        names = POINT_FIELDS
        choice = {"anyOf": [
            {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
             "minItems": 2, "maxItems": 2}, {"type": "null"},
        ]}
    else:
        names = ID_FIELDS
        ids = validate_candidates(candidates)
        choice = ({"anyOf": [{"type": "integer", "minimum": 0, "maximum": len(ids) - 1}, {"type": "null"}]}
                  if ids else {"type": "null"})
    properties = {name: copy.deepcopy(choice) for name in names}
    properties["grasp_mode"] = {"type": "string", "enum": ["top_down", "unresolved"]}
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def validate_blanks(blanks):
    if not isinstance(blanks, dict) or set(blanks) != set(POINT_FIELDS + METHOD_FIELDS):
        raise ValueError("Fill exactly five ordinary pick-place landmarks and the grasp method")
    for field in POINT_FIELDS:
        value = blanks[field]
        if value is not None and (
            not isinstance(value, list) or len(value) != 2 or any(
                isinstance(item, bool) or not isinstance(item, (float, int))
                or not math.isfinite(item) or not 0 <= item <= 1 for item in value
            )
        ):
            raise ValueError("Invalid normalized pick-place landmark: " + field)
    if not isinstance(blanks["grasp_mode"], str) or blanks["grasp_mode"] not in {"top_down", "unresolved"}:
        raise ValueError("Invalid pick-place grasp method")
    return blanks


def candidate_payload(base_instruction, candidates, image_shape):
    declared = template(base_instruction)
    ids = validate_candidates(candidates, image_shape)
    return {**declared, "blanks": list(ID_FIELDS + METHOD_FIELDS), "available_candidate_ids": ids}


def resolve_choices(choices, condition, candidates, image_shape):
    schema = blank_schema(condition, candidates)
    if not isinstance(choices, dict) or set(choices) != set(schema["properties"]):
        raise ValueError("Fill exactly the declared ordinary pick-place blanks")
    if condition == "direct_rgb":
        return validate_blanks(copy.deepcopy(choices))
    ids = validate_candidates(candidates, image_shape)
    height, width = image_shape
    by_id = {point["id"]: point for point in candidates}
    result = {"grasp_mode": choices["grasp_mode"]}
    for id_field, uv_field in zip(ID_FIELDS, POINT_FIELDS):
        selected = choices[id_field]
        if selected is not None and (type(selected) is not int or selected not in ids):
            raise ValueError("Choose an available integer candidate ID or null")
        if selected is None:
            result[uv_field] = None
        else:
            x, y = by_id[selected]["pixel_xy"]
            result[uv_field] = [(x + .5) / width, (y + .5) / height]
    return validate_blanks(result)


def _validate_payload(payload, condition, candidates=None):
    numbered = condition == "dinov3_candidates"
    expected_fields = MODEL_FIELDS | ({"available_candidate_ids"} if numbered else set())
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("Only instruction-derived descriptions, blank declarations and optional IDs are allowed")
    expected = template(payload["instruction"])
    if numbered:
        expected.update(blanks=list(ID_FIELDS + METHOD_FIELDS), available_candidate_ids=validate_candidates(candidates))
    if payload != expected:
        raise ValueError("Ordinary pick-place payload differs from its instruction-only declaration")


def sdk_arguments(config, payload, original_image, prompt, *, condition="direct_rgb", candidates=None, numbered_image=None):
    """Build a future Responses request; this function makes no API call."""
    if condition not in CONDITIONS or (condition == "dinov3_candidates") != (numbered_image is not None):
        raise ValueError("Pick-place input method and numbered RGB must agree")
    _validate_payload(payload, condition, candidates)
    images = [original_image] + ([] if numbered_image is None else [numbered_image])
    if any(not isinstance(image, bytes) or not image for image in images):
        raise ValueError("Original and optional numbered RGB must be nonempty encoded image bytes")
    return {
        "model": config["model"], "instructions": prompt,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": json_text(payload)}] + [
            {"type": "input_image", "detail": config["image_detail"],
             "image_url": "data:image/png;base64," + base64.b64encode(image).decode()} for image in images
        ]}],
        "text": {"format": {"type": "json_schema", "name": "OrdinaryPickPlaceBlanks",
                            "schema": blank_schema(condition, candidates), "strict": True}},
        "reasoning": {"effort": config["reasoning_effort"]}, "service_tier": "default",
        "max_output_tokens": config["max_output_tokens"], "store": False,
    }


def compile_pick_place(blanks, payload, depth, calibration, config):
    """Use calibrated local depth, retaining placement goals and no path constraint."""
    validate_blanks(blanks)
    # A direct or candidate payload carries the same instruction-derived roles.
    if not isinstance(payload, dict) or payload.get("task_type") != "pick_place":
        raise ValueError("An ordinary pick-place payload is required")
    declared = template(payload["instruction"])
    if "available_candidate_ids" in payload:
        ids = payload["available_candidate_ids"]
        if not isinstance(ids, list) or any(type(value) is not int for value in ids) or ids != list(range(len(ids))):
            raise ValueError("Candidate IDs must be consecutive integers beginning at zero")
        declared.update(blanks=list(ID_FIELDS + METHOD_FIELDS), available_candidate_ids=ids)
    if payload != declared:
        raise ValueError("Placement payload must contain only its ordinary instruction-derived declaration")
    promoted = {**copy.deepcopy(blanks), "axis_low_uv": None, "axis_high_uv": None, "path_constraint": "none"}
    result = compile_numericals(promoted, declared, depth, calibration, config)
    goals = [item for item in result["relations"] if item["scope"] == "goal"]
    result.pop("object_axis", None)
    result["landmarks"] = {key: value for key, value in result["landmarks"].items() if key in POINT_FIELDS}
    result.update(
        schema_version="pick-place-scope-1.0", task_type="pick_place",
        relations=goals, goals=copy.deepcopy(goals), path_constraints=[],
        stages=[{"id": stage, "path_constraint_ids": []}
                for stage in ("approach", "grasp", "transport", "place", "release")],
        path_constraints_required=False, whole_object_placement_verified=False,
        backend_requirements=["Collision avoidance", "Verified grasp attachment", "Robot kinematics and feasibility"],
        planner_ready=False,
    )
    return result
