"""Numbered RGB landmark choices resolved to unchanged sensor pixels.

Candidate metadata stays local. The VLM sees instruction-derived descriptions,
available IDs, and RGB images; depth and reference identities never enter its
payload. Resolved blanks can be passed to ``compile_independent_axis``.
"""

import base64

from .rgbd_numericals import POINT_FIELDS, json_text, template


ID_FIELDS = tuple(field.removesuffix("_uv") + "_id" for field in POINT_FIELDS)
METHOD_FIELDS = ("grasp_mode", "path_constraint")
MODEL_FIELDS = {
    "instruction", "source_description", "target_description", "blanks", "available_candidate_ids",
}


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _shape(image_shape):
    if (not isinstance(image_shape, (list, tuple)) or len(image_shape) != 2
            or any(not _integer(value) or value <= 0 for value in image_shape)):
        raise ValueError("Image shape must be positive integer (height, width)")
    return tuple(image_shape)


def _validate_ids(ids):
    if (not isinstance(ids, list) or any(not _integer(value) for value in ids)
            or ids != list(range(len(ids)))):
        raise ValueError("Candidate IDs must be unique consecutive integers starting at zero")
    return ids


def validate_candidates(candidates, image_shape=None):
    """Check IDs and original integer pixels without interpreting sensor metadata."""
    if not isinstance(candidates, list) or any(not isinstance(item, dict) for item in candidates):
        raise ValueError("Candidates must be a list of candidate records")
    ids = _validate_ids([item.get("id") for item in candidates])
    shape = _shape(image_shape) if image_shape is not None else None
    for item in candidates:
        pixel = item.get("pixel_xy")
        if (not isinstance(pixel, (list, tuple)) or len(pixel) != 2
                or any(not _integer(value) or value < 0 for value in pixel)):
            raise ValueError("Candidate pixel_xy must contain original nonnegative integer x/y")
        if shape is not None and (pixel[0] >= shape[1] or pixel[1] >= shape[0]):
            raise ValueError("Candidate pixel is outside the original RGB/depth image")
        if "predicted_region_id" in item and (
            not _integer(item["predicted_region_id"]) or item["predicted_region_id"] <= 0
        ):
            raise ValueError("Predicted region IDs must be positive; label zero is unassigned")
    return ids


def _schema_from_ids(ids):
    _validate_ids(ids)
    # All IDs are consecutive; a bounded integer avoids the API's total enum
    # limit when the same candidate set is used for seven landmark blanks.
    choice = ({"anyOf": [{"type": "integer", "minimum": 0, "maximum": len(ids)-1}, {"type": "null"}]}
              if ids else {"type": "null"})
    properties = {field: choice for field in ID_FIELDS}
    properties.update(
        grasp_mode={"type": "string", "enum": ["top_down", "unresolved"]},
        path_constraint={"type": "string", "enum": ["none", "keep_upright", "unresolved"]},
    )
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def blank_schema(candidates):
    return _schema_from_ids(validate_candidates(candidates))


def validate_blanks(blanks, candidates):
    ids = validate_candidates(candidates)
    if not isinstance(blanks, dict) or set(blanks) != set(ID_FIELDS + METHOD_FIELDS):
        raise ValueError("Fill exactly the declared landmark-ID and method blanks")
    for field in ID_FIELDS:
        value = blanks[field]
        if value is not None and (not _integer(value) or value not in ids):
            raise ValueError("Landmark selection must use an available integer ID or null: " + field)
    if (not isinstance(blanks["grasp_mode"], str) or not isinstance(blanks["path_constraint"], str)
            or blanks["grasp_mode"] not in {"top_down", "unresolved"}
            or blanks["path_constraint"] not in {"none", "keep_upright", "unresolved"}):
        raise ValueError("Invalid grasp method or path-constraint choice")
    return blanks


def candidate_payload(base_instruction, candidates, image_shape, upright=False):
    declared = template(base_instruction, upright)
    ids = validate_candidates(candidates, image_shape)
    return {key: declared[key] for key in ("instruction", "source_description", "target_description")} | {
        "blanks": list(ID_FIELDS + METHOD_FIELDS), "available_candidate_ids": ids,
    }


def resolve_blanks(blanks, candidates, image_shape):
    """Resolve IDs at original pixel centers, with no nearest-point rescue or snapping."""
    height, width = _shape(image_shape)
    validate_candidates(candidates, image_shape)
    validate_blanks(blanks, candidates)
    by_id = {item["id"]: item for item in candidates}
    resolved = {field: blanks[field] for field in METHOD_FIELDS}
    for id_field, uv_field in zip(ID_FIELDS, POINT_FIELDS):
        selected = blanks[id_field]
        if selected is None:
            resolved[uv_field] = None
        else:
            x, y = by_id[selected]["pixel_xy"]
            resolved[uv_field] = [(x + 0.5) / width, (y + 0.5) / height]
    return resolved


def sdk_arguments(config, payload, original_image, numbered_image, prompt):
    """Build a Responses request without creating a client or making a paid call."""
    if not isinstance(payload, dict) or set(payload) != MODEL_FIELDS:
        raise ValueError("Only instruction descriptions, declared blanks, and candidate IDs may enter payload")
    if payload["blanks"] != list(ID_FIELDS + METHOD_FIELDS):
        raise ValueError("Model payload must declare the candidate-ID blanks exactly")
    ids = _validate_ids(payload["available_candidate_ids"])
    if any(not isinstance(payload[key], str) or not payload[key].strip()
           for key in ("instruction", "source_description", "target_description")):
        raise ValueError("Instruction-derived descriptions must be nonempty strings")
    if any(not isinstance(image, bytes) or not image for image in (original_image, numbered_image)):
        raise ValueError("Original RGB and numbered RGB must be nonempty encoded image bytes")
    images = [{"type": "input_image", "detail": config["image_detail"],
               "image_url": "data:image/png;base64," + base64.b64encode(image).decode()}
              for image in (original_image, numbered_image)]
    return {
        "model": config["model"], "instructions": prompt,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": json_text(payload)}, *images,
        ]}],
        "text": {"format": {"type": "json_schema", "name": "RGBDLandmarkChoices",
                            "schema": _schema_from_ids(ids), "strict": True}},
        "reasoning": {"effort": config["reasoning_effort"]},
        "service_tier": "default", "max_output_tokens": config["max_output_tokens"], "store": False,
    }
