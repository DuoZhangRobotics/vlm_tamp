"""Capture predetermined drawer camera and fixture interventions, without actions.

The original frozen capture implementation supplies the RGB-D calibration,
repeat checks and evaluation-only references. A process-local adapter changes
only the camera, fixed cabinet yaw and middle-slider initial coordinate. It
never feeds simulator identities, poses, joints or segmentation to grounding.
These are correlated development interventions, not held-out benchmark tasks.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import prepare_drawer_grounding as original_capture
from vlm_tamp.libero import ROOT, configure_libero


CASES = (
    {"name": "camera_left", "id": "drawer_geometry_camera_left", "position_world_m": [-0.22, 0.55, 1.55],
     "cabinet_yaw_delta_degrees": 0.0, "middle_q_m": None},
    {"name": "camera_right", "id": "drawer_geometry_camera_right", "position_world_m": [0.30, 0.55, 1.55],
     "cabinet_yaw_delta_degrees": 0.0, "middle_q_m": None},
    {"name": "cabinet_yaw_plus20_partial", "id": "drawer_geometry_yaw_plus20_open", "position_world_m": [0.04, 0.55, 1.55],
     "cabinet_yaw_delta_degrees": 20.0, "middle_q_m": -0.06},
    {"name": "cabinet_yaw_minus20_partial", "id": "drawer_geometry_yaw_minus20_open", "position_world_m": [0.04, 0.55, 1.55],
     "cabinet_yaw_delta_degrees": -20.0, "middle_q_m": -0.06},
)
LOOK_AT = [0.04, -0.23, 0.96]


def multiply_quaternions(left, right):
    """Hamilton product in MuJoCo's w,x,y,z convention."""
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    result = np.asarray([
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
    ], dtype=np.float64)
    return result / np.linalg.norm(result)


def configure_camera(env, case):
    from robosuite.utils.transform_utils import mat2quat

    model = env.sim.model
    cam = model.camera_name2id(original_capture.CAMERA)
    if int(model.cam_bodyid[cam]) != 0:
        raise ValueError("Predetermined camera policy requires a world-body camera")
    position = np.asarray(case["position_world_m"], dtype=np.float64)
    backward = position - np.asarray(LOOK_AT, dtype=np.float64)
    backward /= np.linalg.norm(backward)
    right = np.cross([0.0, 0.0, 1.0], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    rotation = np.column_stack([right, up, backward])
    quat = np.asarray(mat2quat(rotation), dtype=np.float64)
    quat /= np.linalg.norm(quat)
    model.cam_pos[cam] = position
    model.cam_quat[cam] = quat[[3, 0, 1, 2]]
    env.sim.forward()
    assert np.allclose(env.sim.data.cam_xpos[cam], position, atol=1e-12)
    assert np.allclose(env.sim.data.cam_xmat[cam].reshape(3, 3), rotation, atol=1e-7)


def capture_case(output, case):
    """Use the frozen helper for one official source state plus interventions."""
    from libero.libero import envs

    previous_factory = envs.OffScreenRenderEnv
    runtime = {}

    class IntervenedEnv:
        def __init__(self, *args, **kwargs):
            self.wrapped = previous_factory(*args, **kwargs)
            self.base_quaternion = None

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def reset(self):
            observation = self.wrapped.reset()
            configure_camera(self.wrapped, case)
            body = self.wrapped.sim.model.body_name2id("wooden_cabinet_1_main")
            model = self.wrapped.sim.model
            if int(model.body_parentid[body]) != 0 or int(model.body_jntnum[body]) != 0:
                raise ValueError("Cabinet intervention requires the existing fixed world fixture")
            self.base_quaternion = model.body_quat[body].copy()
            return observation

        def set_init_state(self, state):
            self.wrapped.set_init_state(state)
            model, data = self.wrapped.sim.model, self.wrapped.sim.data
            body = model.body_name2id("wooden_cabinet_1_main")
            half_angle = np.deg2rad(case["cabinet_yaw_delta_degrees"]) / 2
            yaw = np.asarray([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
            model.body_quat[body] = multiply_quaternions(yaw, self.base_quaternion)
            joint = model.joint_name2id("wooden_cabinet_1_middle_level")
            if case["middle_q_m"] is not None:
                lower, upper = model.jnt_range[joint]
                coordinate = float(case["middle_q_m"])
                if not lower <= coordinate <= upper:
                    raise ValueError("Requested simulation coordinate is outside joint limits")
                data.qpos[int(model.jnt_qposadr[joint])] = coordinate
                data.qvel[int(model.jnt_dofadr[joint])] = 0.0
            self.wrapped.sim.forward()
            captured_state = self.wrapped.get_sim_state().copy()
            observation = self.wrapped.regenerate_obs_from_state(captured_state)
            runtime.update({
                "source_cabinet_quaternion_wxyz": self.base_quaternion.tolist(),
                "captured_cabinet_quaternion_wxyz": model.body_quat[body].tolist(),
                "captured_middle_q_m": float(data.qpos[int(model.jnt_qposadr[joint])]),
                "captured_sim_state": captured_state,
            })
            return observation

    staging = output / ("_capture_" + case["name"])
    envs.OffScreenRenderEnv = IntervenedEnv
    try:
        captured = original_capture.capture(staging, states=(0,))
    finally:
        envs.OffScreenRenderEnv = previous_factory
    source_id = captured["captures"][0]
    identifier = case["id"]
    source_folder = staging / source_id
    folder = output / identifier
    source_folder.rename(folder)
    record = json.loads((folder / "record.json").read_text())
    record["example_id"] = identifier
    for field in ("model_inputs", "local_sensor_inputs", "evaluation"):
        for key, value in record[field].items():
            if isinstance(value, str) and value.startswith(source_id + "/"):
                record[field][key] = identifier + value[len(source_id):]
    record["metadata"]["initial_state_path"] = identifier + "/initial_state.npy"
    np.save(folder / "captured_sim_state.npy", runtime.pop("captured_sim_state"), allow_pickle=False)
    intervention = {
        "version": "drawer-geometry-stress-interventions-1.0", "case": case,
        "official_source_state_index": 0, "official_initial_state_after_intervention": False,
        "camera_look_at_world_m": LOOK_AT, "camera_up_world": [0.0, 0.0, 1.0],
        "camera_selected_using_object_pose_or_axis_or_mask": False,
        "camera_selection": "predetermined nominal workspace values before model responses",
        "cabinet_pose_override_kind": "fixed fixture world-z yaw relative to restored source pose",
        "drawer_override_kind": "simulation slider coordinate; zero robot actions or policy execution",
        "fixture_and_joint_names_use": "capture intervention and held-back evaluation only",
        "source_initial_state_file": identifier + "/initial_state.npy",
        "captured_sim_state_file": identifier + "/captured_sim_state.npy",
        "model_inventory": False, "robot_actions": 0, "paid_api_calls": 0,
        **runtime,
    }
    record["metadata"]["intervention"] = intervention
    record["metadata"]["capture_script_sha256"] = original_capture.sha(__file__)
    record["metadata"]["scope"] = (
        "one cabinet and official source state; predetermined development camera/yaw/slider interventions"
    )
    original_capture.write_json(folder / "intervention.json", intervention)
    original_capture.write_json(folder / "record.json", record)
    # Only these newly created staging summaries are removed. Their useful
    # checks are retained in the combined record and top-level manifest.
    for path in staging.iterdir():
        if not path.is_file():
            raise ValueError("Unexpected directory remains in the private capture staging area")
        path.unlink()
    staging.rmdir()
    return record


def capture(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing captures; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    configure_libero(gl="egl")
    plan = {
        "task_name": original_capture.TASK_NAME, "instruction": original_capture.INSTRUCTION,
        "cases": list(CASES), "look_at_world_m": LOOK_AT, "official_source_state_index": 0,
        "api_input_fields": ["instruction", "image"], "robot_actions": 0, "paid_api_calls": 0,
        "selection": "predetermined camera and fixture interventions before new model responses",
        "scope": "correlated development stress cases; not a held-out benchmark or motion execution",
    }
    original_capture.write_json(output / "selection_before_capture.json", plan)
    records = [capture_case(output, case) for case in CASES]
    rgb_hashes = [original_capture.sha(output / record["model_inputs"]["image"]) for record in records]
    assert len(set(rgb_hashes)) == len(records)
    (output / "records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    manifest = {
        "version": "drawer-rgbd-geometry-stress-1.0", "plan": plan,
        "captures": [record["example_id"] for record in records],
        "distinct_rgb_observations": len(set(rgb_hashes)),
        "checks": {record["example_id"]: record["metadata"]["capture_checks"] for record in records},
        "capture_script_sha256": original_capture.sha(__file__),
        "reused_capture_script_sha256": original_capture.sha(original_capture.__file__),
        "files": {str(path.relative_to(output)): original_capture.sha(path)
                  for path in sorted(output.rglob("*")) if path.is_file()},
        "robot_actions": 0, "paid_api_calls": 0, "scope": plan["scope"],
    }
    original_capture.write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/rgbd-drawer-geometry-stress")
    args = parser.parse_args()
    result = capture(ROOT / args.output)
    print(json.dumps({"captures": len(result["captures"]),
                      "distinct_rgb": result["distinct_rgb_observations"],
                      "paid_api_calls": 0, "robot_actions": 0}))
