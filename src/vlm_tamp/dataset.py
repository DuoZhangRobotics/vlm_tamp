"""Dataset validation and a single allowlisted model-input boundary."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


CONDITIONS = {
    "language": "Language only",
    "rgb": "Language + RGB",
    "candidates": "Language + RGB + inventory (oracle-assisted)",
    "geometry": "Language + RGB + geometry (oracle-assisted)",
    "markers": "Language + labelled RGB + inventory (oracle-assisted)",
}
ORACLE_CONDITIONS = {"candidates", "geometry", "markers"}


def read_records(path):
    path = Path(path)
    manifest = path / "records.jsonl" if path.is_dir() else path
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return records, manifest.parent.resolve()


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Dataset reference escapes dataset root")
    return path


def validate_dataset(path):
    records, root = read_records(path)
    ids = set()
    for record in records:
        eid = record["example_id"]
        if eid in ids:
            raise ValueError(f"Duplicate example ID: {eid}")
        ids.add(eid)
        if set(record["model_inputs"]) != {"instruction", "image", "candidates", "geometry"}:
            raise ValueError("Unexpected model-input field (possible reference leakage)")
        rgb = np.array(Image.open(safe_path(root, record["model_inputs"]["image"])).convert("RGB"))
        if rgb.std() < 5 or not record["model_inputs"]["instruction"]:
            raise ValueError(f"Blank image or instruction: {eid}")
        for visual in record["reference"]["visual"].values():
            mask = np.array(Image.open(safe_path(root, visual["mask"]))) > 0
            if mask.shape != rgb.shape[:2] or mask.sum() != visual["visible_pixels"]:
                raise ValueError(f"Inconsistent segmentation mask: {eid}")
            ys, xs = np.where(mask)
            expected = [
                xs.min() / rgb.shape[1],
                ys.min() / rgb.shape[0],
                (xs.max() + 1) / rgb.shape[1],
                (ys.max() + 1) / rgb.shape[0],
            ]
            if not np.allclose(expected, visual["bbox"]):
                raise ValueError(f"Bounding box does not match simulator mask: {eid}")
        state = np.load(safe_path(root, record["metadata"]["initial_state_path"]), allow_pickle=False)
        if hashlib.sha256(state.tobytes()).hexdigest() != record["metadata"]["initial_state_sha256"]:
            raise ValueError(f"Initial state hash mismatch: {eid}")
        known = {e["identifier"] for e in record["reference"]["entities"]}
        for goal in record["reference"]["goal_predicates"]:
            if not set(goal["arguments"]) <= known:
                raise ValueError(f"Unknown BDDL reference argument: {goal}")
    return {
        "examples": len(records),
        "suites": sorted({r["metadata"]["suite"] for r in records}),
        "visible_instance_masks": sum(len(r["reference"]["visual"]) for r in records),
        "symbolic_evaluable": sum(r["reference"]["coverage"]["symbolic_goals"] for r in records),
    }


@dataclass(frozen=True)
class ModelRequest:
    prompt: str
    payload: dict
    image: bytes | None
    condition: str
    source_image_sha256: str | None = None
    input_provenance: dict | None = None

    @property
    def text(self):
        return json.dumps(self.payload, sort_keys=True)

    @property
    def image_hash(self):
        return hashlib.sha256(self.image).hexdigest() if self.image is not None else None

    @property
    def validation_context(self):
        return {
            "instruction": self.payload["instruction"],
            "candidates": self.payload.get("candidates"),
            "image_available": self.image is not None,
        }


def build_request(inputs, root, condition, prompt, marker_sample=None):
    """Only this function exposes dataset content to a provider.

    It accepts the model_inputs compartment, not a complete dataset record.
    Nested oracle fields are allowlisted too: injected reference fields are ignored.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}")
    payload = {"instruction": inputs["instruction"]}
    if condition in ORACLE_CONDITIONS:
        payload["candidates"] = [
            {key: entity.get(key) for key in ("identifier", "category", "kind", "parent")}
            for entity in inputs["candidates"]
        ]
    if condition == "geometry":
        payload["geometry"] = {
            identifier: {key: value.get(key) for key in ("position", "rotation_matrix", "frame", "units")}
            for identifier, value in inputs["geometry"].items()
        }
    image = None if condition == "language" else safe_path(root, inputs["image"]).read_bytes()
    source_hash = hashlib.sha256(image).hexdigest() if image is not None else None
    provenance = None
    if condition == "markers":
        if marker_sample is None:
            raise ValueError("Markers require a validated, separate oracle input bundle")
        if marker_sample["source_image_sha256"] != source_hash:
            raise ValueError("Marker source image differs from the observation")
        payload["markers"] = [
            {key: mapping[key] for key in ("label", "identifier")} for mapping in marker_sample["mappings"]
        ]
        image = marker_sample["image"]
        if hashlib.sha256(image).hexdigest() != marker_sample["image_sha256"]:
            raise ValueError("Marker image hash mismatch")
        provenance = marker_sample["provenance"]
    return ModelRequest(
        prompt=prompt,
        payload=payload,
        image=image,
        condition=condition,
        source_image_sha256=source_hash,
        input_provenance=provenance,
    )
