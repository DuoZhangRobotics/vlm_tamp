"""Run released ReKep constraint generation on sensor-backed, RGB-derived keypoints."""

import ast
import importlib.util
import json

import numpy as np
from PIL import Image

from vlm_tamp.baseline_api import BoundedChat
from vlm_tamp.cli import load_credentials
from vlm_tamp.libero import ROOT


def main():
    load_credentials("/home/duo/.codex/keys/vlm-tamp.env")
    api = BoundedChat(ROOT / "results/external-component-api")
    repo = ROOT / "third_party/ReKep"
    spec = importlib.util.spec_from_file_location("rekep_native_constraints", repo / "constraint_generation.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    folder = ROOT / "results/rekep-keypoint-smoke"
    metadata = json.loads((folder / "metadata.json").read_text())
    keypoints = np.load(folder / "keypoints_world.npy")
    generator = native.ConstraintGenerator({"model": "gpt-4o-2024-11-20", "temperature": 0.0, "max_tokens": 2048})
    generator.client = api
    generator.base_dir = str(ROOT / "results/rekep-constraint-smoke")
    generated = generator.generate(np.asarray(Image.open(folder / "numbered.png")), metadata["instruction"],
                                   {"num_keypoints": len(keypoints), "init_keypoint_positions": keypoints})
    output = ROOT / "results/rekep-component-summary.json"
    result = {"status": "component_generation_completed", "directory": generated,
              "input_modality": "RGB-D plus RGB-predicted masks", "source_model_alias_retired": "chatgpt-4o-latest",
              "model_used": "gpt-4o-2024-11-20", "generated_code_executed": False,
              "physical_constraint_accuracy_verified": False, "robot_execution_tested": False}
    saved = json.loads((ROOT / generated / "metadata.json").read_text())
    result["num_stages"] = saved["num_stages"]
    result["grasp_keypoints"] = saved["grasp_keypoints"]
    result["release_keypoints"] = saved["release_keypoints"]
    functions, indices = [], []
    for path in (ROOT / generated).glob("*_constraints.txt"):
        tree = ast.parse(path.read_text())
        functions.extend(node.name for node in tree.body if isinstance(node, ast.FunctionDef))
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "keypoints":
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                    indices.append(node.slice.value)
    result["functions"] = functions
    result["static_keypoint_indices_in_range"] = all(0 <= index < len(keypoints) for index in indices)
    result["parsed_python_functions"] = len(functions)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
