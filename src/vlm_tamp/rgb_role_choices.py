"""Instruction-only templates; Mini fills only visual role bindings."""

import copy
import re

from .rgb_object_choices import build_choice_request as object_request
from .rgb_choices_v2 import ChoicePrediction, compile_choices as compile_form, digest as digest


def instruction_template(instruction):
    text = re.sub(r"\s+", " ", instruction.strip().lower()).rstrip(".")
    roles, goals = {}, []

    def role(phrase):
        phrase = phrase.strip()
        for key, value in roles.items():
            if value == phrase:
                return key
        key = f"slot_{len(roles)}"
        roles[key] = phrase
        return key

    def placement(subject, relation, destination, close=False):
        part = "whole"
        drawer = re.fullmatch(r"(?:the )?(top|middle|bottom) drawer of (.+)", destination)
        if drawer:
            part = drawer[1] + "_drawer"
            destination = drawer[2]
        source, target = role(subject), role(destination)
        state = "inside" if relation == "in" else "on_top"
        goals.append(
            {
                "state": state,
                "object_slot": source,
                "object_part": "whole",
                "target_slot": target,
                "target_part": part,
            }
        )
        if close:
            goals.append(
                {
                    "state": "closed",
                    "object_slot": target,
                    "object_part": part,
                    "target_slot": None,
                    "target_part": "whole",
                }
            )

    power = re.fullmatch(r"turn (on|off) (.+)", text)
    push = re.fullmatch(r"push (.+) to (?:the )?front of (.+)", text)
    pickup = re.fullmatch(r"pick up (.+) and place (?:it|them) (in|on) (.+)", text)
    if power:
        goals.append(
            {
                "state": "powered_" + power[1],
                "object_slot": role(power[2]),
                "object_part": "whole",
                "target_slot": None,
                "target_part": "whole",
            }
        )
    elif push:
        goals.append(
            {
                "state": "in_front_of",
                "object_slot": role(push[1]),
                "object_part": "whole",
                "target_slot": role(push[2]),
                "target_part": "whole",
            }
        )
    elif pickup:
        placement(pickup[1], pickup[2], pickup[3])
    else:
        clauses = text.split(" and put ")
        for index, clause in enumerate(clauses):
            if index:
                clause = "put " + clause
            close = clause.endswith(" and close it")
            if close:
                clause = clause[: -len(" and close it")]
            match = re.fullmatch(r"put (.+?) (in|on) (.+)", clause)
            if not match:
                raise ValueError(
                    "Instruction is outside the supported placement/pushing/activation template grammar"
                )
            placement(match[1], match[2], match[3], close)
    return {"roles": roles, "goals": goals}


def build_choice_request(instruction, image_bytes, prompt):
    request = object_request(instruction, image_bytes, prompt)
    template = instruction_template(instruction)
    return type(request)(
        request.prompt,
        {**request.payload, "roles": template["roles"]},
        request.source_image,
        request.scene_image,
        request.cards_image,
        request.regions,
    )


def binding_schema(instruction, count):
    template = instruction_template(instruction)
    return {
        "type": "object",
        "properties": {
            key: {
                "type": "integer",
                "enum": list(range(count + 1)),
                "description": "Image choice for " + phrase,
            }
            for key, phrase in template["roles"].items()
        },
        "required": list(template["roles"]),
        "additionalProperties": False,
    }


def validate_bindings(payload, instruction, count):
    roles = instruction_template(instruction)["roles"]
    if set(payload) != set(roles) or any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= count
        for value in payload.values()
    ):
        raise ValueError("Role binding must fill every declared slot with an available image number or zero")
    return payload


def compile_bindings(bindings, instruction, regions):
    validate_bindings(bindings, instruction, len(regions))
    template = instruction_template(instruction)
    goals = []
    for declared in template["goals"]:
        goal = copy.deepcopy(declared)
        goal["object_choice"] = bindings[goal.pop("object_slot")]
        target = goal.pop("target_slot")
        goal["target_choice"] = bindings[target] if target is not None else 0
        goals.append(goal)
    return compile_form(ChoicePrediction(schema_version="rgb-choice-2.0", goals=goals), regions)
