"""Task-independent RGB regions and a finite-choice final-goal form.

The region builder accepts image bytes only. Its color components are visual
patches, not labeled object detections or simulator instance masks.
"""

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field, StrictInt, model_validator
from scipy import ndimage

from .grounding import Model
from .rgb_grounding import RGBEntity, RGBGoal, RGBPrediction


VERSION = "rgb-choice-1.0"
REGION_SETTINGS = {
    "spatial_radius": 6,
    "color_radius": 18,
    "lab_bin_width": 20,
    "min_pixels": 18,
    "max_area_fraction": 0.30,
    "max_choices": 48,
}
STATES = {
    "inside": "In",
    "supported_on": "On",
    "in_front_of": "InFrontOf",
    "open": "Open",
    "closed": "Close",
    "turned_on": "Turnon",
    "turned_off": "Turnoff",
    "stacked_on": "Stack",
}
BINARY_STATES = {"inside", "supported_on", "in_front_of", "stacked_on"}
PARTS = ("whole", "interior", "top_drawer", "middle_drawer", "bottom_drawer", "door", "unknown")
ChoiceState = Literal[
    "inside",
    "supported_on",
    "in_front_of",
    "open",
    "closed",
    "turned_on",
    "turned_off",
    "stacked_on",
    "unknown",
]
ChoicePart = Literal["whole", "interior", "top_drawer", "middle_drawer", "bottom_drawer", "door", "unknown"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def png(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def image_regions(image_bytes):
    """Mean-shift color components; parameters and ranking never read the task."""
    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.asarray(source)
    height, width = rgb.shape[:2]
    smooth = cv2.pyrMeanShiftFiltering(
        rgb, REGION_SETTINGS["spatial_radius"], REGION_SETTINGS["color_radius"]
    )
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).astype(np.int32) // REGION_SETTINGS["lab_bin_width"]
    radix = math.ceil(256 / REGION_SETTINGS["lab_bin_width"])
    bins = lab[:, :, 0] * radix**2 + lab[:, :, 1] * radix + lab[:, :, 2]
    components = []
    for value in np.unique(bins):
        labeled, _ = ndimage.label(bins == value)
        for label, box in enumerate(ndimage.find_objects(labeled), 1):
            if box is None:
                continue
            component = labeled[box] == label
            area = int(component.sum())
            if (
                not REGION_SETTINGS["min_pixels"]
                <= area
                <= height * width * REGION_SETTINGS["max_area_fraction"]
            ):
                continue
            distance = ndimage.distance_transform_edt(np.pad(component, 1))[1:-1, 1:-1]
            dy, dx = np.unravel_index(np.argmax(distance), distance.shape)
            y, x = int(dy + box[0].start), int(dx + box[1].start)
            components.append(
                {"area": area, "pixel": [x, y], "box": [box[1].start, box[0].start, box[1].stop, box[0].stop]}
            )
    # Largest components first, then image order. Both choices are independent of
    # task language, benchmark objects and offline masks.
    components.sort(key=lambda c: (-c["area"], c["pixel"][1], c["pixel"][0], c["box"]))
    components = components[: REGION_SETTINGS["max_choices"]]
    components.sort(key=lambda c: (c["pixel"][1], c["pixel"][0]))
    return [
        {"choice": index, "point": [(c["pixel"][0] + 0.5) / width, (c["pixel"][1] + 0.5) / height], **c}
        for index, c in enumerate(components, 1)
    ]


def render_choices(image_bytes, regions):
    """Full-scene labels plus context crops, with dots at the exact chosen pixel."""
    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = source.size
    font = ImageFont.load_default(size=17)
    scene = source.resize((768, 768))
    draw = ImageDraw.Draw(scene)
    columns, card_width, card_height = 8, 130, 150
    cards = Image.new(
        "RGB", (columns * card_width, max(1, math.ceil(len(regions) / columns)) * card_height), "#152030"
    )
    card_draw = ImageDraw.Draw(cards)
    for region in regions:
        choice = region["choice"]
        x, y = region["pixel"]
        px, py = (x + 0.5) * 768 / width, (y + 0.5) * 768 / height
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="yellow", outline="black")
        draw.text(
            (px + 5, py - 8), str(choice), fill="yellow", font=font, stroke_width=2, stroke_fill="black"
        )
        left, top, right, bottom = max(0, x - 25), max(0, y - 25), min(width, x + 26), min(height, y + 26)
        crop = source.crop((left, top, right, bottom)).resize((126, 126))
        cx, cy = ((choice - 1) % columns) * card_width, ((choice - 1) // columns) * card_height
        cards.paste(crop, (cx, cy + 23))
        card_draw.text((cx + 5, cy + 2), str(choice), fill="white", font=font)
        # Edge crops are asymmetric: keep the dot on its real source pixel.
        dx = cx + (x + 0.5 - left) * 126 / (right - left)
        dy = cy + 23 + (y + 0.5 - top) * 126 / (bottom - top)
        card_draw.ellipse((dx - 3, dy - 3, dx + 3, dy + 3), fill="yellow", outline="black")
    return png(scene), png(cards)


@dataclass(frozen=True)
class ChoiceRequest:
    prompt: str
    payload: dict
    source_image: bytes
    scene_image: bytes
    cards_image: bytes
    regions: list[dict]

    @property
    def text(self):
        import json

        return json.dumps(self.payload, sort_keys=True)


def build_choice_request(instruction, image_bytes, prompt):
    regions = image_regions(image_bytes)
    if not regions:
        raise ValueError("No RGB regions available")
    scene, cards = render_choices(image_bytes, regions)
    return ChoiceRequest(
        prompt,
        {"instruction": instruction, "visual_choices": list(range(1, len(regions) + 1))},
        image_bytes,
        scene,
        cards,
        regions,
    )


class ChoiceGoal(Model):
    state: ChoiceState
    object_choice: StrictInt = Field(ge=0)
    object_part: ChoicePart
    target_choice: StrictInt = Field(ge=0)
    target_part: ChoicePart

    @model_validator(mode="after")
    def unused_target(self):
        if self.state not in BINARY_STATES and (self.target_choice != 0 or self.target_part != "whole"):
            raise ValueError("Single-object states have no placement target")
        return self


class ChoicePrediction(Model):
    schema_version: Literal["rgb-choice-1.0"]
    goals: list[ChoiceGoal] = Field(min_length=1, max_length=8)


def choice_schema(count):
    if not isinstance(count, int) or not 1 <= count <= REGION_SETTINGS["max_choices"]:
        raise ValueError("Invalid visual-choice count")
    schema = ChoicePrediction.model_json_schema()
    for definition in schema["$defs"].values():
        definition["required"] = list(definition["properties"])
        definition["additionalProperties"] = False
    goal = schema["$defs"]["ChoiceGoal"]["properties"]
    for field in ("object_choice", "target_choice"):
        goal[field] = {"type": "integer", "enum": list(range(count + 1))}
    import copy

    variants = []
    for states, unary in (
        (sorted(BINARY_STATES), False),
        (sorted(set(STATES) - BINARY_STATES) + ["unknown"], True),
    ):
        properties = copy.deepcopy(goal)
        properties["state"] = {"type": "string", "enum": states}
        if unary:
            properties["target_choice"] = {"type": "integer", "const": 0}
            properties["target_part"] = {"type": "string", "const": "whole"}
        variants.append(
            {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }
        )
    schema["$defs"]["ChoiceGoal"] = {"anyOf": variants}
    schema["required"] = list(schema["properties"])
    schema["additionalProperties"] = False
    return schema


def compile_choices(prediction, regions):
    """Expand finite choices without any reference goals, labels or physics claims."""
    by_choice = {r["choice"]: r for r in regions}
    entities, goals = {}, []

    def entity(choice, selector):
        if selector == "unknown":
            raise ValueError("Unknown object part")
        if choice not in by_choice:
            raise ValueError("Missing or unavailable visual choice")
        owner = "choice_" + str(choice)
        if owner not in entities:
            entities[owner] = RGBEntity(
                id=owner,
                category="image referent",
                description="Selected RGB choice " + str(choice),
                point=by_choice[choice]["point"],
            )
        if selector == "whole":
            return owner
        part = owner + "_" + selector
        entities[part] = RGBEntity(
            id=part,
            category="selected part",
            description=selector,
            kind="region" if selector == "interior" else "part",
            parent=owner,
            selector=selector,
        )
        return part

    for goal in prediction.goals:
        if goal.state not in STATES:
            raise ValueError("Unsupported or unknown desired state")
        arguments = [entity(goal.object_choice, goal.object_part)]
        if goal.state in BINARY_STATES:
            arguments.append(entity(goal.target_choice, goal.target_part))
        goals.append(RGBGoal(relation=STATES[goal.state], arguments=arguments))
    return RGBPrediction(
        entities=list(entities.values()),
        goals=goals,
        uncertainties=[
            "Final semantic goals only; stages, metric geometry and constraints remain unverified."
        ],
    )
