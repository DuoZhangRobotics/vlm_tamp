"""Run the original ReKep keypoint proposer without its unused simulator imports."""

import ast
import importlib.util
import json
import sys
import types

import numpy as np
import torch
from PIL import Image

from vlm_tamp.libero import ROOT


def main():
    repo = ROOT / "third_party/ReKep"
    # Import exactly the pure bounds helper needed by keypoint_proposal.py,
    # avoiding unrelated Open3D/OmniGibson/robot-controller dependencies.
    tree = ast.parse((repo / "utils.py").read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "filter_points_by_bounds")
    module = types.ModuleType("utils")
    module.__dict__["np"] = np
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(repo / "utils.py"), "exec"), module.__dict__)
    sys.modules["utils"] = module
    torch.hub.set_dir(str(ROOT / ".runtime/torch-hub"))
    source = ROOT / "third_party/dinov2"
    original_load = torch.hub.load

    def pinned_load(repository, name, *args, **kwargs):
        if repository == "facebookresearch/dinov2":
            return original_load(str(source), name, *args, source="local", **kwargs)
        return original_load(repository, name, *args, **kwargs)

    torch.hub.load = pinned_load
    if not hasattr(torch.amp, "autocast"):
        torch.amp.autocast = torch.autocast
    spec = importlib.util.spec_from_file_location("rekep_keypoint_proposal", repo / "keypoint_proposal.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    config = {"device": "cuda", "bounds_min": [-1.5, -1.5, 0.65], "bounds_max": [1.5, 1.5, 2.0],
              "min_dist_bt_keypoints": 0.06, "num_candidates_per_mask": 5, "max_mask_ratio": 0.5, "seed": 0}
    proposer = native.KeypointProposer(config)
    proposer.mean_shift.n_jobs = 4
    captured = {}
    original_project = proposer._project_keypoints_to_img

    def record_projection(rgb, pixels, groups, masks, features):
        captured["pixels_yx"] = pixels.copy()
        captured["groups"] = groups.copy()
        return original_project(rgb, pixels, groups, masks, features)

    proposer._project_keypoints_to_img = record_projection
    folder = ROOT / "results/rekep-component-input"
    output = ROOT / "results/rekep-keypoint-smoke"
    if output.exists():
        raise FileExistsError("Preserve previous keypoint output")
    output.mkdir(parents=True)
    rgb = np.asarray(Image.open(folder / "rgb.png"))
    keypoints, marked = proposer.get_keypoints(rgb, np.load(folder / "world_points.npy"),
                                             np.load(folder / "rgb_predicted_masks.npy"))
    assert len(keypoints) > 0 and np.isfinite(keypoints).all()
    Image.fromarray(marked).save(output / "numbered.png")
    np.save(output / "keypoints_world.npy", keypoints)
    metadata = {**json.loads((folder / "metadata.json").read_text()), "num_keypoints": len(keypoints),
                "pixels_yx": captured["pixels_yx"].tolist(), "config": config,
                "native_proposer_source_unchanged": True, "robot_execution_tested": False}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"num_keypoints": len(keypoints), "modality": metadata["modality"],
                      "mask_source": metadata["mask_source"]}, indent=2))


if __name__ == "__main__":
    main()
