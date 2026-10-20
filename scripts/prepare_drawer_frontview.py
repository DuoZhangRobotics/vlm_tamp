"""Capture a new fixed front camera view while preserving the original pilot.

The nominal camera was chosen by RGB inspection of the workspace. Camera
selection never queries object poses, joint axes or evaluation segmentation.
The existing capture implementation is reused through a process-local camera
adapter; its source and all original data remain unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import prepare_drawer_grounding as original_capture
from vlm_tamp.libero import ROOT, configure_libero


VIEW_POLICY = {
    "version": "drawer-frontview-camera-1.0",
    "camera": "agentview",
    "position_world_m": [0.04, 0.55, 1.55],
    "look_at_world_m": [0.04, -0.23, 0.96],
    "up_world": [0.0, 0.0, 1.0],
    "field_of_view": "unchanged built-in agentview 45 degrees",
    "image_size": [512, 512],
    "selection": "fixed nominal workspace camera selected through RGB-only development inspection",
    "intervention": "development camera change following original oblique-view localization failures",
    "object_pose_used_to_choose_camera": False,
    "joint_axis_used_to_choose_camera": False,
    "simulator_masks_used_to_choose_camera": False,
    "cropping_or_mask_conditioned_zoom": False,
    "held_out_view": False,
}


def configure_camera(env):
    from robosuite.utils.transform_utils import mat2quat

    model = env.sim.model
    cam = model.camera_name2id(VIEW_POLICY["camera"])
    # The agentview camera has a fixed world-body parent, which makes these
    # nominal positions independent of any cabinet or joint coordinate frame.
    if int(model.cam_bodyid[cam]) != 0:
        raise ValueError("Camera policy requires a world-body camera")
    position = np.asarray(VIEW_POLICY["position_world_m"], dtype=float)
    target = np.asarray(VIEW_POLICY["look_at_world_m"], dtype=float)
    backward = position-target
    backward /= np.linalg.norm(backward)
    right = np.cross(VIEW_POLICY["up_world"], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    rotation = np.column_stack([right, up, backward])
    # Robosuite returns float32 here. Normalize in float64 before writing the
    # runtime MuJoCo model: otherwise its camera matrix is slightly non-rigid
    # and the strict depth/backprojection round-trip check correctly fails.
    quat_xyzw = np.asarray(mat2quat(rotation), dtype=np.float64)
    quat_xyzw /= np.linalg.norm(quat_xyzw)
    model.cam_pos[cam] = position
    model.cam_quat[cam] = quat_xyzw[[3, 0, 1, 2]]
    env.sim.forward()
    assert np.allclose(env.sim.data.cam_xpos[cam], position)
    assert np.allclose(env.sim.data.cam_xmat[cam].reshape(3, 3), rotation, atol=1e-7)


def probe(path):
    """Render only RGB at the planned pose; no evaluator or robot action."""
    configure_libero(gl="egl")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.macros as macros

    suite = benchmark.get_benchmark_dict()["libero_goal"](task_order_index=0)
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(0), camera_names=["agentview"],
        camera_heights=512, camera_widths=512, camera_depths=False,
    )
    try:
        env.seed(0)
        env.reset()
        configure_camera(env)
        observation = env.set_init_state(suite.get_task_init_states(0)[0])
        rgb = observation["agentview_image"]
        if macros.IMAGE_CONVENTION == "opengl":
            rgb = rgb[::-1]
        Image.fromarray(rgb).save(path)
        print(json.dumps({"rgb_probe": str(path), "view_policy": VIEW_POLICY}))
    finally:
        env.close()


def capture(output):
    configure_libero(gl="egl")
    from libero.libero import envs

    previous_factory = envs.OffScreenRenderEnv
    previous_writer = original_capture.write_json

    class FixedCameraEnv:
        def __init__(self, *args, **kwargs):
            self.wrapped = previous_factory(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def reset(self):
            result = self.wrapped.reset()
            configure_camera(self.wrapped)
            return result

    def new_capture_writer(path, value):
        if Path(path).name == "selection_before_capture.json":
            value = dict(value, view_policy=VIEW_POLICY)
        previous_writer(path, value)

    # Process-local adapters avoid editing the frozen shared implementation.
    # They are restored even if capture fails; files are written only under
    # the new output directory selected by the caller.
    envs.OffScreenRenderEnv = FixedCameraEnv
    original_capture.write_json = new_capture_writer
    try:
        manifest = original_capture.capture(output)
    finally:
        envs.OffScreenRenderEnv = previous_factory
        original_capture.write_json = previous_writer

    output = Path(output).resolve()
    records = []
    for line in (output / "records.jsonl").read_text().splitlines():
        record = json.loads(line)
        image = output / record["model_inputs"]["image"]
        old_image = ROOT / "data/rgbd-drawer-pilot" / record["model_inputs"]["image"]
        assert old_image.is_file()
        old_pixels = np.asarray(Image.open(old_image))
        pixels = np.asarray(Image.open(image))
        assert pixels.shape == old_pixels.shape == (512, 512, 3)
        assert not np.array_equal(pixels, old_pixels)
        comparison = {
            "original_rgb_sha256": original_capture.sha(old_image),
            "new_rgb_sha256": original_capture.sha(image),
            "original_rgb_pixels_sha256": hashlib.sha256(old_pixels.tobytes()).hexdigest(),
            "new_rgb_pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "original_shape": list(old_pixels.shape), "new_shape": list(pixels.shape),
            "pixels_changed": True,
            "original_capture_unchanged": True,
        }
        record["metadata"]["view_policy"] = VIEW_POLICY
        record["metadata"]["original_view_comparison"] = comparison
        previous_writer(output / record["example_id"] / "record.json", record)
        records.append(record)
    (output / "records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    manifest["version"] = "drawer-rgbd-frontview-capture-1.0"
    manifest["plan"]["view_policy"] = VIEW_POLICY
    manifest["view_policy"] = VIEW_POLICY
    manifest["reused_capture_script_sha256"] = manifest["capture_script_sha256"]
    manifest["capture_script_sha256"] = original_capture.sha(__file__)
    manifest["original_view_comparisons"] = {
        record["example_id"]: record["metadata"]["original_view_comparison"] for record in records
    }
    manifest["files"] = {
        str(path.relative_to(output)): original_capture.sha(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"
    }
    manifest["scope"] = (
        "four correlated official states with a fixed development camera intervention; "
        "not held-out task or articulation-axis generalization; no robot execution"
    )
    previous_writer(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/rgbd-drawer-frontview-pilot")
    parser.add_argument("--probe-rgb", help="Save an RGB-only probe and stop before final capture")
    args = parser.parse_args()
    if args.probe_rgb:
        probe(Path(args.probe_rgb))
    else:
        result = capture(ROOT / args.output)
        print(json.dumps({"captures": len(result["captures"]), "distinct_rgb": result["distinct_rgb_observations"],
                          "paid_api_calls": 0, "robot_actions": 0}))
