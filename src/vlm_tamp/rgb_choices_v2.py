"""RGB-only small-region coverage and a non-overlapping final-state form.

Version 1 remains frozen. Neither proposal generation nor goal compilation reads
benchmark identities, masks, expected goals or simulator geometry.
"""

import copy
import io
import math
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field, StrictInt, model_validator
from scipy import ndimage

from .grounding import Model
from .rgb_choices import ChoicePrediction as LegacyChoicePrediction
from .rgb_choices import ChoiceRequest, compile_choices as legacy_compile, digest as digest, png


VERSION = "rgb-choice-2.0"
REGION_SETTINGS = {
    "reference_width": 256,
    "spatial_radius": 6,
    "color_radius": 18,
    "lab_bin_width": 20,
    "min_reference_pixels": 6,
    "max_area_fraction": 0.30,
    "score_area_saturation": 256,
    "context_padding": 10,
    "grid_side": 4,
    "max_per_grid_cell": 6,
    "minimum_point_separation": 0.018,
    "max_choices": 64,
}
STATES = {
    "inside": "In",
    "on_top": "On",
    "in_front_of": "InFrontOf",
    "open": "Open",
    "closed": "Close",
    "powered_on": "Turnon",
    "powered_off": "Turnoff",
}
BINARY_STATES = {"inside", "on_top", "in_front_of"}
STATE_TO_V1 = {"on_top": "supported_on", "powered_on": "turned_on", "powered_off": "turned_off"}
ChoiceState = Literal[
    "inside", "on_top", "in_front_of", "open", "closed", "powered_on", "powered_off", "unknown"
]
ChoicePart = Literal["whole", "interior", "top_drawer", "middle_drawer", "bottom_drawer", "door", "unknown"]


def image_regions(image_bytes):
    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.asarray(source)
    height, width = rgb.shape[:2]
    scale = width / REGION_SETTINGS["reference_width"]
    smooth = cv2.pyrMeanShiftFiltering(
        rgb, max(1, round(REGION_SETTINGS["spatial_radius"] * scale)), REGION_SETTINGS["color_radius"]
    )
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    quantized = lab.astype(np.int32) // REGION_SETTINGS["lab_bin_width"]
    radix = math.ceil(256 / REGION_SETTINGS["lab_bin_width"])
    bins = quantized[:, :, 0] * radix**2 + quantized[:, :, 1] * radix + quantized[:, :, 2]
    components = []
    grid = REGION_SETTINGS["grid_side"]
    for value in np.unique(bins):
        labels, _ = ndimage.label(bins == value)
        for label, box in enumerate(ndimage.find_objects(labels), 1):
            if box is None:
                continue
            component = labels[box] == label
            area = int(component.sum())
            if (
                not REGION_SETTINGS["min_reference_pixels"] * scale**2
                <= area
                <= height * width * REGION_SETTINGS["max_area_fraction"]
            ):
                continue
            distance = ndimage.distance_transform_edt(np.pad(component, 1))[1:-1, 1:-1]
            dy, dx = np.unravel_index(np.argmax(distance), distance.shape)
            y, x = int(dy + box[0].start), int(dx + box[1].start)
            padding = max(1, round(REGION_SETTINGS["context_padding"] * scale))
            y0, y1 = max(0, box[0].start - padding), min(height, box[0].stop + padding)
            x0, x1 = max(0, box[1].start - padding), min(width, box[1].stop + padding)
            local = lab[y0:y1, x0:x1].astype(float)
            inside = labels[y0:y1, x0:x1] == label
            outside = ~inside
            contrast = (
                float(np.linalg.norm(local[inside].mean(axis=0) - local[outside].mean(axis=0)))
                if outside.any()
                else 0.0
            )
            compactness = area / (component.shape[0] * component.shape[1])
            score = (
                math.sqrt(min(area, REGION_SETTINGS["score_area_saturation"] * scale**2))
                * (1 + contrast)
                * math.sqrt(compactness)
            )
            components.append(
                {
                    "pixel": [x, y],
                    "area": area,
                    "contrast": contrast,
                    "score": score,
                    "box": [box[1].start, box[0].start, box[1].stop, box[0].stop],
                    "grid_cell": [y * grid // height, x * grid // width],
                }
            )
    components.sort(key=lambda region: (-region["score"], region["pixel"][1], region["pixel"][0]))
    selected, per_cell = [], {}

    def add(region):
        cell = tuple(region["grid_cell"])
        if (
            len(selected) >= REGION_SETTINGS["max_choices"]
            or per_cell.get(cell, 0) >= REGION_SETTINGS["max_per_grid_cell"]
        ):
            return
        if any(
            math.dist(region["pixel"], other["pixel"]) < width * REGION_SETTINGS["minimum_point_separation"]
            for other in selected
        ):
            return
        selected.append(region)
        per_cell[cell] = per_cell.get(cell, 0) + 1

    for cell in sorted({tuple(region["grid_cell"]) for region in components}):
        add(next(region for region in components if tuple(region["grid_cell"]) == cell))
    for region in components:
        add(region)
    selected.sort(key=lambda region: (region["pixel"][1], region["pixel"][0]))
    return [
        {
            "choice": number,
            "point": [(region["pixel"][0] + 0.5) / width, (region["pixel"][1] + 0.5) / height],
            **region,
        }
        for number, region in enumerate(selected, 1)
    ]


def render_choices(image_bytes, regions):
    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = source.size
    scale = width / REGION_SETTINGS["reference_width"]
    scene = source.resize((768, 768))
    draw = ImageDraw.Draw(scene)
    font = ImageFont.load_default(size=18)
    columns, card_width, card_height = 8, 160, 192
    cards = Image.new(
        "RGB", (columns * card_width, max(1, math.ceil(len(regions) / columns)) * card_height), "#152030"
    )
    card_draw = ImageDraw.Draw(cards)
    radius = max(12, round(17 * scale))
    for region in regions:
        number = region["choice"]
        x, y = region["pixel"]
        px, py = (x + 0.5) * 768 / width, (y + 0.5) * 768 / height
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="yellow", outline="black")
        draw.text(
            (px + 5, py - 8), str(number), font=font, fill="yellow", stroke_width=2, stroke_fill="black"
        )
        left, top, right, bottom = (
            max(0, x - radius),
            max(0, y - radius),
            min(width, x + radius + 1),
            min(height, y + radius + 1),
        )
        crop = source.crop((left, top, right, bottom)).resize((156, 156))
        cx, cy = ((number - 1) % columns) * card_width, ((number - 1) // columns) * card_height
        cards.paste(crop, (cx + 2, cy + 28))
        card_draw.text((cx + 5, cy + 3), str(number), font=font, fill="white")
        # A hollow, small cross preserves the central packaging text. The
        # original image remains unmarked for reading labels and appearance.
        dx = cx + 2 + (x + 0.5 - left) * 156 / (right - left)
        dy = cy + 28 + (y + 0.5 - top) * 156 / (bottom - top)
        for x0, y0, x1, y1 in (
            (dx - 6, dy, dx - 3, dy),
            (dx + 3, dy, dx + 6, dy),
            (dx, dy - 6, dx, dy - 3),
            (dx, dy + 3, dx, dy + 6),
        ):
            card_draw.line((x0, y0, x1, y1), fill="yellow", width=1)
    return png(scene), png(cards)


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
    schema_version: Literal["rgb-choice-2.0"]
    goals: list[ChoiceGoal] = Field(min_length=1, max_length=8)


def choice_schema(count):
    if not isinstance(count, int) or not 1 <= count <= REGION_SETTINGS["max_choices"]:
        raise ValueError("Invalid visual-choice count")
    schema = ChoicePrediction.model_json_schema()
    properties = schema["$defs"]["ChoiceGoal"]["properties"]
    for field in ("object_choice", "target_choice"):
        properties[field] = {"type": "integer", "enum": list(range(count + 1))}
    variants = []
    for states, unary in (
        (sorted(BINARY_STATES), False),
        (sorted(set(STATES) - BINARY_STATES) + ["unknown"], True),
    ):
        fields = copy.deepcopy(properties)
        fields["state"] = {"type": "string", "enum": states}
        if unary:
            fields["target_choice"] = {"type": "integer", "const": 0}
            fields["target_part"] = {"type": "string", "const": "whole"}
        variants.append(
            {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
        )
    schema["$defs"]["ChoiceGoal"] = {"anyOf": variants}
    schema["required"] = list(schema["properties"])
    schema["additionalProperties"] = False
    return schema


def compile_choices(prediction, regions):
    payload = prediction.model_dump(mode="json")
    payload["schema_version"] = "rgb-choice-1.0"
    for goal in payload["goals"]:
        goal["state"] = STATE_TO_V1.get(goal["state"], goal["state"])
    return legacy_compile(LegacyChoicePrediction.model_validate(payload), regions)
