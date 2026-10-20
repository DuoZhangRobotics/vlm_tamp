"""Prepare a real RGB-D observation and RGB-predicted masks for ReKep."""

import json
import random

import numpy as np
from PIL import Image

from vlm_tamp.dataset import read_records, safe_path
from vlm_tamp.libero import ROOT, configure_libero
from vlm_tamp.rgb_object_choices import segmentation_model


def main():
    output = ROOT / "results/rekep-component-input"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve the existing ReKep input")
    output.mkdir(parents=True, exist_ok=True)
    configure_libero(gl="egl")
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.macros as macros

    records, root = read_records(ROOT / "data/rgb-role-generalization")
    record = next(record for record in records if record["example_id"] == "libero_goal-08-state005-seed0")
    np.random.seed(0)
    random.seed(0)
    spec = ROOT / "third_party/LIBERO" / record["metadata"]["bddl"]
    env = OffScreenRenderEnv(bddl_file_name=str(spec), camera_names=["agentview"],
                            camera_heights=512, camera_widths=512, camera_depths=True)
    try:
        env.seed(0)
        env.reset()
        state = np.load(safe_path(root, record["metadata"]["initial_state_path"]))
        observation = env.set_init_state(state)
        rgb, depth = observation["agentview_image"], observation["agentview_depth"]
        if macros.IMAGE_CONVENTION == "opengl":
            rgb, depth = rgb[::-1], depth[::-1]
        expected = np.asarray(Image.open(safe_path(root, record["model_inputs"]["image"])))
        assert np.array_equal(rgb, expected), "Depth collection must reproduce the frozen RGB"
        # The same pinhole/axis/depth formulas as robosuite.camera_utils,
        # without importing its unrelated optional HDF5 playback dependency.
        sim = env.sim
        cam = sim.model.camera_name2id("agentview")
        near = sim.model.vis.map.znear * sim.model.stat.extent
        far = sim.model.vis.map.zfar * sim.model.stat.extent
        depth = (near / (1 - depth * (1 - near / far))).squeeze()
        focal = 256 / np.tan(np.deg2rad(sim.model.cam_fovy[cam]) / 2)
        intrinsic = np.array([[focal, 0, 256], [0, focal, 256], [0, 0, 1]])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = sim.data.cam_xmat[cam].reshape(3, 3) @ np.diag([1, -1, -1])
        extrinsic[:3, 3] = sim.data.cam_xpos[cam]
        yy, xx = np.mgrid[:512, :512]
        rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(intrinsic).T
        camera_points = rays * depth[..., None]
        points = camera_points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        assert np.isfinite(points).all()
    finally:
        env.close()
    # Only the RGB enters this segmentation model. No simulator instance masks
    # or object categories are used to construct candidate regions.
    sam = segmentation_model()
    from mobile_sam import SamAutomaticMaskGenerator
    generator = SamAutomaticMaskGenerator(sam, points_per_side=8,
                                         pred_iou_thresh=0.75, stability_score_thresh=0.88,
                                         crop_n_layers=0, min_mask_region_area=30)
    candidates = generator.generate(rgb)
    candidates = [candidate for candidate in candidates if 90 <= candidate["area"] <= rgb.shape[0] * rgb.shape[1] * 0.45]
    candidates.sort(key=lambda candidate: (-candidate["area"], candidate["bbox"]))
    labels = np.zeros(depth.shape, dtype=np.int32)
    for number, candidate in enumerate(candidates[:24], 1):
        labels[candidate["segmentation"]] = number
    Image.fromarray(rgb).save(output / "rgb.png")
    np.save(output / "depth_meters.npy", depth)
    np.save(output / "world_points.npy", points)
    np.save(output / "rgb_predicted_masks.npy", labels)
    metadata = {"example_id": record["example_id"], "instruction": record["model_inputs"]["instruction"],
                "modality": "RGB-D; not the RGB-only Mini input condition",
                "mask_source": "RGB-only MobileSAM; no simulator segmentation or categories",
                "rgb_matches_frozen_image": True, "camera_intrinsic": intrinsic.tolist(),
                "camera_to_world": extrinsic.tolist(), "candidate_segments": int(len(np.unique(labels)) - 1),
                "frame": "world", "units": "meters", "robot_execution_tested": False}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
