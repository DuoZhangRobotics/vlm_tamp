import hashlib
import json

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def isolated_default_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("vlm_tamp.experiment.ROOT", tmp_path)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :10] = [200, 50, 20]
    image[:, 10:] = [50, 200, 220]
    Image.fromarray(image).save(root / "rgb.png")
    state = np.array([0.0, 1.0, 2.0])
    np.save(root / "initial.npy", state)
    visual = {}
    for identifier, side in (("bowl_1", slice(0, 10)), ("plate_1", slice(10, 20))):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[:, side] = 255
        Image.fromarray(mask).save(root / f"{identifier}.png")
        visual[identifier] = {
            "mask": f"{identifier}.png",
            "visible_pixels": 200,
            "bbox": [0.0, 0.0, 0.5, 1.0] if identifier == "bowl_1" else [0.5, 0.0, 1.0, 1.0],
        }
    entities = [
        {"identifier": "bowl_1", "category": "bowl", "kind": "object", "parent": None},
        {"identifier": "plate_1", "category": "plate", "kind": "object", "parent": None},
        {"identifier": "distractor_1", "category": "mug", "kind": "object", "parent": None},
    ]
    record = {
        "example_id": "test-0",
        "model_inputs": {
            "instruction": "put the bowl on the plate",
            "image": "rgb.png",
            "candidates": entities,
            "geometry": {},
        },
        "reference": {
            "goal_expression": ["And", ["On", "bowl_1", "plate_1"]],
            "goal_predicates": [{"name": "On", "arguments": ["bowl_1", "plate_1"], "normalized_name": "on"}],
            "entities": entities,
            "visual": visual,
            "coverage": {"symbolic_goals": True},
            "SECRET_REFERENCE": "DO_NOT_SEND_123",
        },
        "metadata": {
            "suite": "test",
            "task_index": 0,
            "state_index": 0,
            "initial_state_path": "initial.npy",
            "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            "camera": {
                "name": "agentview",
                "width": 20,
                "height": 20,
                "world_rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "world_position": [0, 0, 0],
                "fovy_degrees": 45,
            },
        },
    }
    (root / "records.jsonl").write_text(json.dumps(record) + "\n")
    return root, record


@pytest.fixture
def config(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return structured grounding using a partial-order graph.")
    path = tmp_path / "config.json"
    contents = {
        "provider": "mock",
        "model": "mock",
        "prompt": str(prompt),
        "conditions": ["language", "rgb", "candidates", "geometry"],
        "seed": 0,
        "timeout_seconds": 10,
        "max_retries": 2,
        "max_output_tokens": 1000,
        "max_requests": 10,
        "max_budget_usd": 0,
        "image_token_allowance": 4096,
        "pricing": {"input_per_million": 0, "output_per_million": 0},
    }
    path.write_text(json.dumps(contents))
    return path, contents
