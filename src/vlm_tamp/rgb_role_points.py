"""Match instruction roles to points in untouched RGB, without candidate proposals."""

import hashlib
import json
import math
from dataclasses import dataclass

from .rgb_role_choices import compile_bindings, instruction_template


def digest(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class PointRequest:
    prompt: str
    payload: dict
    source_image: bytes

    @property
    def text(self):
        return json.dumps(self.payload, ensure_ascii=False)


def build_point_request(instruction, image_bytes, prompt):
    return PointRequest(
        prompt,
        {"instruction": instruction, "roles": instruction_template(instruction)["roles"]},
        image_bytes,
    )


def point_schema(instruction):
    roles = instruction_template(instruction)["roles"]
    return {
        "type": "object",
        "properties": {
            key: {
                "description": "Visible image point [x,y] for " + phrase + "; null if uncertain or invisible",
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    {"type": "null"},
                ],
            }
            for key, phrase in roles.items()
        },
        "required": list(roles),
        "additionalProperties": False,
    }


def validate_points(payload, instruction):
    roles = instruction_template(instruction)["roles"]
    if not isinstance(payload, dict) or set(payload) != set(roles):
        raise ValueError("Fill exactly the declared visual roles")
    for point in payload.values():
        if point is None:
            continue
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
                for value in point
            )
        ):
            raise ValueError("A role needs a normalized [x,y] point or null")
    return payload


def compile_points(payload, instruction):
    """Feed VLM-selected points through the unchanged role/goal compiler.

    The temporary numbers below are internal adapter IDs, not candidates shown
    to the model. Coordinates are passed unchanged; no mask lookup or snapping.
    """
    validate_points(payload, instruction)
    roles = instruction_template(instruction)["roles"]
    regions, bindings = [], {}
    for number, key in enumerate(roles, 1):
        point = payload[key]
        if point is None:
            bindings[key] = 0
        else:
            regions.append({"choice": number, "point": point})
            bindings[key] = number
    return compile_bindings(bindings, instruction, regions)
