"""All-scene simulator-assisted object labels, prepared outside normal model inputs."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .dataset import read_records, safe_path


SCOPE = "all visible whole objects; no part, region, joint, goal or plan labels"


def _visible_objects(record):
    candidates = {e["identifier"]: e for e in record["model_inputs"]["candidates"]}
    visible = {
        identifier: visual
        for identifier, visual in record["reference"]["visual"].items()
        if visual.get("kind", "object") == "object"
    }
    if any(
        identifier not in candidates or candidates[identifier]["kind"] != "object" for identifier in visible
    ):
        raise ValueError("Whole-object marker mask has no matching inventory object")
    return visible


def _mask(root, visual, size):
    path = safe_path(root, visual["mask"])
    mask = np.array(Image.open(path).convert("L")) > 0
    if mask.shape != size or int(mask.sum()) != visual["visible_pixels"] or not mask.any():
        raise ValueError("Marker source mask has inconsistent dimensions or visible pixels")
    return mask, hashlib.sha256(path.read_bytes()).hexdigest()


def _anchor(mask):
    # Choose an actual interior pixel; a mask centroid can fall in background.
    current = Image.fromarray((mask * 255).astype(np.uint8))
    interior = mask
    for _ in range(max(mask.shape)):
        eroded = current.filter(ImageFilter.MinFilter(3))
        if not np.array(eroded).any():
            break
        current = eroded
        interior = np.array(eroded) > 0
    ys, xs = np.where(interior)
    idx = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
    return int(xs[idx]), int(ys[idx])


def _draw(image, entries, masks, font):
    draw = ImageDraw.Draw(image)
    placed = []
    for entry in entries:
        mask = masks[entry["identifier"]]
        eroded = np.array(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))) > 0
        ys, xs = np.where(mask & ~eroded)
        for x, y in zip(xs.tolist(), ys.tolist()):
            draw.point((x, y), fill="#ffda65")
    for entry in entries:
        x, y = entry["anchor_pixel"]
        label = entry["label"]
        box = draw.textbbox((0, 0), label, font=font)
        width, height = box[2] - box[0] + 6, box[3] - box[1] + 6
        options = []
        for dx, dy in (
            (5, -height - 4),
            (5, 4),
            (-width - 5, -height - 4),
            (-width - 5, 4),
            (5, -height // 2),
            (-width - 5, -height // 2),
            (-width // 2, -height - 4),
            (-width // 2, 4),
        ):
            left = min(max(0, x + dx), max(0, image.width - width))
            top = min(max(0, y + dy), max(0, image.height - height))
            candidate = (left, top, left + width, top + height)
            overlap = sum(
                max(0, min(candidate[2], old[2]) - max(candidate[0], old[0]))
                * max(0, min(candidate[3], old[3]) - max(candidate[1], old[1]))
                for old in placed
            )
            options.append((overlap, candidate))
        _, chosen = min(options, key=lambda option: option[0])
        placed.append(chosen)
        left, top, right, bottom = chosen
        end = ((left + right) / 2, (top + bottom) / 2)
        draw.line((x, y, *end), fill="black", width=3)
        draw.line((x, y, *end), fill="#ffda65", width=1)
        draw.rectangle(chosen, fill="white", outline="black", width=1)
        draw.text((left + 3 - box[0], top + 3 - box[1]), label, font=font, fill="black")
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#ffda65", outline="black")


def generate_marker_bundle(dataset, output, example_ids=None):
    """Generate all visible object correspondences without inspecting task goals.

    The original RGB dimensions, manifest, inventory and references are preserved.
    Numbering is shuffled using the image hash, independent of instruction/goals.
    No part masks or kinematic/expert reference files are accepted here.
    """
    records, root = read_records(dataset)
    wanted = set(example_ids) if example_ids else {r["example_id"] for r in records}
    if not wanted <= {r["example_id"] for r in records}:
        raise ValueError("Unknown marker example IDs")
    output = Path(output).resolve()
    if (output / "markers.json").exists():
        raise FileExistsError("Marker bundle already exists; use a new --output")
    output.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 11) if font_path.exists() else ImageFont.load_default()
    examples = {}
    for record in records:
        if record["example_id"] not in wanted:
            continue
        source = safe_path(root, record["model_inputs"]["image"])
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        image = Image.open(source).convert("RGB")
        visible = _visible_objects(record)
        order = sorted(visible)
        random.Random(int(source_hash, 16)).shuffle(order)
        entries, masks = [], {}
        for index, identifier in enumerate(order, start=1):
            mask, mask_hash = _mask(root, visible[identifier], (image.height, image.width))
            masks[identifier] = mask
            entries.append(
                {
                    "label": f"M{index:02d}",
                    "identifier": identifier,
                    "anchor_pixel": list(_anchor(mask)),
                    "mask": visible[identifier]["mask"],
                    "mask_sha256": mask_hash,
                }
            )
        _draw(image, entries, masks, font)
        relative = f"images/{record['example_id']}.png"
        destination = safe_path(output, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        examples[record["example_id"]] = {
            "source_image_sha256": source_hash,
            "image": relative,
            "image_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "width": image.width,
            "height": image.height,
            "mappings": entries,
        }
    bundle = {
        "schema_version": "1.0",
        "oracle_assisted": True,
        "scope": SCOPE,
        "dataset_sha256": hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "font_sha256": hashlib.sha256(font_path.read_bytes()).hexdigest() if font_path.exists() else None,
        "examples": examples,
    }
    (output / "markers.json").write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle


def load_marker_bundle(path, dataset, records=None):
    path = Path(path).resolve()
    path = path / "markers.json" if path.is_dir() else path
    bundle = json.loads(path.read_text())
    all_records, root = read_records(dataset)
    records = all_records if records is None else records
    dataset_hash = hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest()
    if (
        bundle.get("schema_version") != "1.0"
        or bundle.get("oracle_assisted") is not True
        or bundle.get("scope") != SCOPE
    ):
        raise ValueError("Unsupported or mislabelled oracle marker bundle")
    if bundle["dataset_sha256"] != dataset_hash:
        raise ValueError("Marker bundle dataset hash mismatch")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {}
    for record in records:
        eid = record["example_id"]
        if eid not in bundle["examples"]:
            raise ValueError("Marker bundle does not cover selected observation")
        sample = bundle["examples"][eid]
        source = safe_path(root, record["model_inputs"]["image"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != sample["source_image_sha256"]:
            raise ValueError("Marker source image changed")
        image_path = safe_path(path.parent, sample["image"])
        image_bytes = image_path.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != sample["image_sha256"]:
            raise ValueError("Marker image changed")
        source_size = Image.open(source).size
        if Image.open(image_path).size != source_size or (sample["width"], sample["height"]) != source_size:
            raise ValueError("Marker image must retain original dimensions")
        visible = _visible_objects(record)
        entries = sample["mappings"]
        if (
            len(entries) != len(visible)
            or {e["identifier"] for e in entries} != set(visible)
            or len({e["label"] for e in entries}) != len(entries)
        ):
            raise ValueError("Markers must label all visible whole objects once, including distractors")
        order = sorted(visible)
        random.Random(int(sample["source_image_sha256"], 16)).shuffle(order)
        for index, entry in enumerate(entries, start=1):
            if entry["label"] != f"M{index:02d}" or entry["identifier"] != order[index - 1]:
                raise ValueError("Marker numbering must be independent of task goals and instructions")
            visual = visible[entry["identifier"]]
            mask, mask_hash = _mask(root, visual, (sample["height"], sample["width"]))
            if entry["mask"] != visual["mask"] or entry["mask_sha256"] != mask_hash:
                raise ValueError("Marker source mask changed")
            x, y = entry["anchor_pixel"]
            if (
                not isinstance(x, int)
                or not isinstance(y, int)
                or not 0 <= x < mask.shape[1]
                or not 0 <= y < mask.shape[0]
                or not mask[y, x]
            ):
                raise ValueError("Marker anchor must lie inside the actual object mask")
        result[eid] = {
            "source_image_sha256": sample["source_image_sha256"],
            "image_sha256": sample["image_sha256"],
            "image": image_bytes,
            "mappings": [{k: entry[k] for k in ("label", "identifier")} for entry in entries],
            "provenance": {
                "marker_bundle": str(path),
                "marker_bundle_sha256": digest,
                "image_path": str(image_path),
                "oracle_scope": SCOPE,
            },
        }
    return result
