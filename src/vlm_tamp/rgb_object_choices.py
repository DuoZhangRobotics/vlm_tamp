"""Object-sized choices from class-agnostic MobileSAM, using RGB-only prompts."""

import copy
import io
import math
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .libero import ROOT
from .rgb_choices import ChoiceRequest, png
from .rgb_choices_v2 import ChoicePrediction as ChoicePrediction, choice_schema as choice_schema
from .rgb_choices_v2 import (
    compile_choices as compile_choices,
    digest as digest,
    image_regions as color_regions,
)


SETTINGS = {
    "grid_side": 8,
    "pred_iou_thresh": 0.75,
    "stability_score_thresh": 0.88,
    "minimum_area": 90,
    "maximum_area_fraction": 0.40,
    "containment_threshold": 0.98,
    "maximum_choices": 32,
}


@lru_cache(maxsize=1)
def segmentation_model():
    for path in (ROOT / ".runtime/rgb-object-deps", ROOT / "third_party/MobileSAM"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    from mobile_sam import sam_model_registry

    torch.set_num_threads(4)
    model = sam_model_registry["vit_t"](checkpoint=str(ROOT / "third_party/MobileSAM/weights/mobile_sam.pt"))
    return model.eval()


@lru_cache(maxsize=64)
def object_bundle(image_bytes):
    model = segmentation_model()
    from mobile_sam import SamAutomaticMaskGenerator

    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.asarray(source)
    height, width = rgb.shape[:2]
    anchors = color_regions(image_bytes)
    side = SETTINGS["grid_side"]
    grid = [[(x + 0.5) / side, (y + 0.5) / side] for y in range(side) for x in range(side)]
    prompts = np.asarray([region["point"] for region in anchors] + grid)
    generator = SamAutomaticMaskGenerator(
        model,
        points_per_side=None,
        point_grids=[prompts],
        points_per_batch=32,
        pred_iou_thresh=SETTINGS["pred_iou_thresh"],
        stability_score_thresh=SETTINGS["stability_score_thresh"],
        crop_n_layers=0,
        min_mask_region_area=30,
    )
    candidates = []
    for item in generator.generate(rgb):
        mask = item["segmentation"]
        area = int(mask.sum())
        if not SETTINGS["minimum_area"] <= area <= height * width * SETTINGS["maximum_area_fraction"]:
            continue
        ys, xs = np.where(mask)
        candidates.append(
            {
                "mask": mask,
                "area": area,
                "box": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                "predicted_iou": float(item["predicted_iou"]),
                "stability": float(item["stability_score"]),
            }
        )
    candidates.sort(key=lambda item: (-item["area"], -item["predicted_iou"], item["box"]))
    kept = []
    for candidate in candidates:
        if any(
            np.logical_and(candidate["mask"], other["mask"]).sum() / min(candidate["area"], other["area"])
            >= SETTINGS["containment_threshold"]
            for other in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) == SETTINGS["maximum_choices"]:
            break
    kept.sort(key=lambda item: ((item["box"][0] + item["box"][2]) / 2, (item["box"][1] + item["box"][3]) / 2))
    regions, masks = [], []
    for number, item in enumerate(kept, 1):
        mask = item["mask"]
        distance = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        in_mask = [anchor for anchor in anchors if mask[anchor["pixel"][1], anchor["pixel"][0]]]
        if in_mask:
            anchor = max(in_mask, key=lambda anchor: distance[anchor["pixel"][1], anchor["pixel"][0]])
            x, y = anchor["pixel"]
        else:
            y, x = [int(value) for value in np.unravel_index(np.argmax(distance), distance.shape)]
        regions.append(
            {
                "choice": number,
                "point": [(x + 0.5) / width, (y + 0.5) / height],
                "pixel": [x, y],
                "box": item["box"],
                "area": item["area"],
                "predicted_iou": item["predicted_iou"],
                "stability": item["stability"],
                "mask_sha256": digest(mask.tobytes()),
                "source": "RGB-only MobileSAM",
            }
        )
        masks.append(mask)
    scene = source.resize((768, 768))
    draw = ImageDraw.Draw(scene)
    font = ImageFont.load_default(size=20)
    columns, card_width, card_height = 6, 192, 224
    cards = Image.new(
        "RGB", (columns * card_width, max(1, math.ceil(len(regions) / columns)) * card_height), "#152030"
    )
    card_draw = ImageDraw.Draw(cards)
    for region, mask in zip(regions, masks):
        number = region["choice"]
        x, y = region["pixel"]
        box = [
            region["box"][0] * 768 / width,
            region["box"][1] * 768 / height,
            region["box"][2] * 768 / width,
            region["box"][3] * 768 / height,
        ]
        draw.rectangle(box, outline="yellow", width=2)
        draw.text(
            (box[0], box[1]), str(number), font=font, fill="yellow", stroke_width=2, stroke_fill="black"
        )
        px, py = (x + 0.5) * 768 / width, (y + 0.5) * 768 / height
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="yellow", outline="black")
        left, top, right, bottom = region["box"]
        padding = 4
        left, top, right, bottom = (
            max(0, left - padding),
            max(0, top - padding),
            min(width, right + padding),
            min(height, bottom + padding),
        )
        local_rgb = rgb[top:bottom, left:right]
        local_mask = mask[top:bottom, left:right]
        highlighted = (local_rgb.astype(float) * np.where(local_mask[..., None], 1.0, 0.20)).astype(np.uint8)
        crop = Image.fromarray(highlighted)
        crop.thumbnail((184, 184))
        cx, cy = ((number - 1) % columns) * card_width, ((number - 1) // columns) * card_height
        cards.paste(crop, (cx + (card_width - crop.width) // 2, cy + 28 + (184 - crop.height) // 2))
        card_draw.text((cx + 5, cy + 3), str(number), font=font, fill="white")
    return regions, png(scene), png(cards)


def build_choice_request(instruction, image_bytes, prompt):
    regions, scene, cards = object_bundle(image_bytes)
    regions = copy.deepcopy(regions)
    if not regions:
        raise ValueError("No RGB object candidates available")
    width, height = Image.open(io.BytesIO(image_bytes)).size
    choices = [
        {
            "number": region["choice"],
            "image_point": region["point"],
            "image_box": [
                value / (width if index % 2 == 0 else height) for index, value in enumerate(region["box"])
            ],
        }
        for region in regions
    ]
    return ChoiceRequest(
        prompt, {"instruction": instruction, "visual_choices": choices}, image_bytes, scene, cards, regions
    )
