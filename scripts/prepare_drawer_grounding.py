"""Capture the official middle-drawer task without executing robot actions.

Only instruction and RGB are declared model inputs. Depth/calibration are local
sensor inputs; simulator joint identities, articulation parameters and per-geom
masks are stored separately for evaluation and must never enter grounding.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_tamp.bddl import parse_bddl
from vlm_tamp.libero import ROOT, configure_libero, version_info
from vlm_tamp.rgbd_numeric_evaluation import capture_checks
from vlm_tamp.rgbd_numericals import metric_depth, world_points


TASK_NAME = "open_the_middle_drawer_of_the_cabinet"
INSTRUCTION = "Open the middle layer of the drawer"
STATES = (0, 10, 20, 30)
CAMERA = "agentview"
SIZE = 512


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def segmentation_reference(env, raw_geom_ids, folder):
    """Bind visible masks to actual compiled visual meshes, not image guesses."""
    model, data = env.sim.model, env.sim.data
    cabinet_geoms = []
    mesh_geoms = {}
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body = model.body_id2name(body_id) or ""
        if body.startswith("wooden_cabinet_1_"):
            cabinet_geoms.append(geom_id)
        if int(model.geom_type[geom_id]) == 7:
            mesh = model.mesh_id2name(int(model.geom_dataid[geom_id]))
            if mesh:
                mesh_geoms.setdefault(mesh, []).append(geom_id)

    panel_geoms = mesh_geoms["wooden_cabinet_1_wooden_cabinet_middle_vis"]
    handle_geoms = mesh_geoms["wooden_cabinet_1_wooden_cabinet_middle_handle_vis"]
    assert len(panel_geoms) == len(handle_geoms) == 1
    panel = np.isin(raw_geom_ids, panel_geoms)
    handle = np.isin(raw_geom_ids, handle_geoms)
    masks = {
        "cabinet": np.isin(raw_geom_ids, cabinet_geoms),
        "middle_drawer": panel | handle,
        "middle_handle": handle,
        "middle_panel": panel,
    }
    assert not np.any(panel & handle)
    assert all(mask.sum() > 0 for mask in masks.values()), "Required drawer part is occluded"
    description = {}
    for name, mask in masks.items():
        np.save(folder / f"{name}_mask.npy", mask, allow_pickle=False)
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"{name}_mask.png")
        yy, xx = np.nonzero(mask)
        description[name] = {
            "npy": f"evaluation/{name}_mask.npy",
            "png": f"evaluation/{name}_mask.png",
            "visible_pixels": int(mask.sum()),
            "bbox_xyxy_pixels": [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)],
            "provenance": "actual MuJoCo per-geom segmentation and compiled visual mesh identities",
        }
    description["middle_panel"]["scope"] = (
        "visible wooden_cabinet_middle_vis mesh excluding handle; mesh includes the drawer body, "
        "so this is not a separately certified front-face-only mask"
    )
    description["middle_handle"]["scope"] = (
        "visible middle_handle_vis mesh; physical grasp/contact/clearance is not certified"
    )
    mapping = []
    for geom_id in cabinet_geoms:
        mesh_id = int(model.geom_dataid[geom_id]) if int(model.geom_type[geom_id]) == 7 else None
        mapping.append({
            "geom_id": geom_id, "geom_name": model.geom_id2name(geom_id),
            "body_name": model.body_id2name(int(model.geom_bodyid[geom_id])),
            "group": int(model.geom_group[geom_id]),
            "mesh_name": model.mesh_id2name(mesh_id) if mesh_id is not None else None,
        })
    write_json(folder / "geom_identity_map.json", mapping)
    region = env.env.object_sites_dict["wooden_cabinet_1_middle_region"]
    joint_names = list(region.joints)
    assert joint_names == ["wooden_cabinet_1_middle_level"], joint_names
    joint_name = joint_names[0]
    joint_id = model.joint_name2id(joint_name)
    assert int(model.jnt_type[joint_id]) == 2
    body_id = int(model.jnt_bodyid[joint_id])
    rotation = data.body_xmat[body_id].reshape(3, 3)
    local_axis = model.jnt_axis[joint_id].copy()
    world_axis = rotation @ local_axis
    qpos = float(data.qpos[int(model.jnt_qposadr[joint_id])])
    cabinet = env.env.get_object("wooden_cabinet_1")
    open_ranges = cabinet.object_properties["articulation"]["default_open_ranges"]
    threshold = float(max(open_ranges))
    assert threshold == -0.14
    assert cabinet.is_open(threshold-1e-6) and not cabinet.is_open(threshold+1e-6)
    assert not cabinet.is_open(threshold), "Official Open comparison must remain strict"
    axis_opening = -world_axis
    joint = {
        "joint_name": joint_name, "joint_type": "prismatic", "type_id": 2,
        "moving_body_name": model.body_id2name(body_id),
        "axis_local": local_axis.tolist(), "jointworldaxis": world_axis.tolist(),
        "axis_positive_q_world": world_axis.tolist(),
        "openingsign": -1, "axisopeningworld": axis_opening.tolist(),
        "opening_direction_world": axis_opening.tolist(),
        "currentq": qpos, "current_q_m": qpos,
        "limits": model.jnt_range[joint_id].tolist(),
        "joint_limits_m": model.jnt_range[joint_id].tolist(),
        "official_open_threshold_m": threshold, "official_open_comparison": "q < threshold",
        "official_open_ranges_m": list(open_ranges),
        "currently_open": bool(cabinet.is_open(qpos)),
        "minimum_opening_displacement_m": max(0.0, qpos-threshold),
        "minimum_opening_displacement_note": "strictly greater than this value is required from current q",
        "frame": "world", "units": "meters",
        "body_position_world": data.body_xpos[body_id].tolist(),
        "body_rotation_world": rotation.tolist(),
        "joint_origin_body_local": model.jnt_pos[joint_id].tolist(),
        "provenance": "compiled simulator joint, body rotation and official WoodenCabinet.is_open predicate",
        "evaluation_only": True,
    }
    return {
        "middle_drawer": joint, "masks": description,
        "reference_visible_geoms": {"middle_panel": panel_geoms, "middle_handle": handle_geoms},
        "evaluation_only": True, "model_inventory": False,
        "grasp_feasibility_verified": False,
        "robot_execution_tested": False,
    }


def capture(output, states=STATES):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing captures; select a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "task_name": TASK_NAME, "suite": "libero_goal", "task_order_index": 0,
        "state_indices": list(states), "seed": 0, "camera": CAMERA, "size": SIZE,
        "instruction": INSTRUCTION,
        "selection": "four predetermined official initial states before model responses",
        "robot_actions": 0, "paid_api_calls": 0,
        "api_input_fields": ["instruction", "image"],
    }
    write_json(output / "selection_before_capture.json", plan)
    libero_root = configure_libero(gl="egl")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.macros as macros

    suite = benchmark.get_benchmark_dict()["libero_goal"](task_order_index=0)
    matches = [i for i in range(suite.n_tasks) if suite.get_task(i).name == TASK_NAME]
    assert matches == [0], matches
    task = suite.get_task(0)
    bddl = Path(suite.get_task_bddl_file_path(0)).resolve()
    assert bddl.name == TASK_NAME + ".bddl"
    parsed = parse_bddl(bddl)
    assert parsed["instruction"] == INSTRUCTION
    assert parsed["goal_predicates"][0]["arguments"] == ["wooden_cabinet_1_middle_region"]
    initial_states = suite.get_task_init_states(0)
    versions = version_info(libero_root)
    records = []
    all_checks = {}
    for index in states:
        if not 0 <= index < len(initial_states):
            raise ValueError("Invalid official initial state index")
        identifier = f"libero_goal-00-state{index:03d}-seed0"
        folder = output / identifier
        evaluation = folder / "evaluation"
        evaluation.mkdir(parents=True)
        np.random.seed(0)
        random.seed(0)
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_names=[CAMERA],
            camera_heights=SIZE, camera_widths=SIZE, camera_depths=True,
            camera_segmentations="element",
        )
        try:
            env.seed(0)
            env.reset()
            state = np.asarray(initial_states[index])
            observation = env.set_init_state(state)
            image = np.asarray(observation[f"{CAMERA}_image"])
            raw_depth = np.asarray(observation[f"{CAMERA}_depth"])
            geom_ids = np.asarray(observation[f"{CAMERA}_segmentation_element"]).squeeze()
            raw_segmentation = env.sim.render(camera_name=CAMERA, width=SIZE, height=SIZE,
                                              depth=False, segmentation=True)
            if macros.IMAGE_CONVENTION == "opengl":
                image, raw_depth, geom_ids = image[::-1], raw_depth[::-1], geom_ids[::-1]
            raw_segmentation = raw_segmentation[::-1]
            assert image.shape == (SIZE, SIZE, 3) and image.dtype == np.uint8 and image.std() > 5
            assert np.array_equal(raw_segmentation[..., 1], geom_ids)
            sim = env.sim
            cam = sim.model.camera_name2id(CAMERA)
            near = float(sim.model.vis.map.znear * sim.model.stat.extent)
            far = float(sim.model.vis.map.zfar * sim.model.stat.extent)
            depth = metric_depth(raw_depth, near, far)
            focal = SIZE/2 / np.tan(np.deg2rad(sim.model.cam_fovy[cam])/2)
            intrinsic = np.array([[focal, 0., SIZE/2], [0., focal, SIZE/2], [0., 0., 1.]])
            transform = np.eye(4)
            transform[:3, :3] = sim.data.cam_xmat[cam].reshape(3, 3) @ np.diag([1., -1., -1.])
            transform[:3, 3] = sim.data.cam_xpos[cam]
            cloud = world_points(depth, intrinsic, transform)
            calibration = {
                "intrinsic": intrinsic.tolist(), "camera_to_world": transform.tolist(),
                "near_m": near, "far_m": far, "camera": CAMERA, "height": SIZE, "width": SIZE,
                "frame": "world", "units": "meters", "image_origin": "top_left",
                "depth_convention": "camera optical-axis z, not Euclidean range",
                "source": "camera configuration/calibration only",
            }
            reference = segmentation_reference(env, geom_ids, evaluation)
            reference["official_goal_expression"] = parsed["goal_expression"]
            reference["initial_official_task_success"] = bool(env.check_success())
            assert not reference["initial_official_task_success"]
            second = env.set_init_state(state)
            repeated_image, repeated_raw_depth = second[f"{CAMERA}_image"], second[f"{CAMERA}_depth"]
            repeated_seg = second[f"{CAMERA}_segmentation_element"].squeeze()
            if macros.IMAGE_CONVENTION == "opengl":
                repeated_image, repeated_raw_depth, repeated_seg = (
                    repeated_image[::-1], repeated_raw_depth[::-1], repeated_seg[::-1]
                )
            repeat = {
                "same_restored_state_rgb_exact": bool(np.array_equal(image, repeated_image)),
                "same_restored_state_depth_exact": bool(np.array_equal(raw_depth, repeated_raw_depth)),
                "same_restored_state_segmentation_exact": bool(np.array_equal(geom_ids, repeated_seg)),
                "robot_actions": 0,
            }
            assert all(value for key, value in repeat.items() if key != "robot_actions")
            checks = capture_checks(depth, cloud, calibration)
            assert checks["passed"]
            checks["determinism"] = repeat
            Image.fromarray(image).save(folder / "rgb.png")
            np.save(folder / "depth_meters.npy", depth, allow_pickle=False)
            np.save(folder / "world_points.npy", cloud, allow_pickle=False)
            np.save(folder / "initial_state.npy", state, allow_pickle=False)
            np.save(evaluation / "geom_segmentation.npy", geom_ids, allow_pickle=False)
            np.save(evaluation / "raw_segmentation_object_type_and_geom_id.npy", raw_segmentation, allow_pickle=False)
            (evaluation / "compiled_model.xml").write_text(sim.model.get_xml())
            write_json(evaluation / "reference.json", reference)
            write_json(folder / "calibration.json", calibration)
            record = {
                "example_id": identifier,
                "model_inputs": {"instruction": INSTRUCTION, "image": f"{identifier}/rgb.png"},
                "local_sensor_inputs": {
                    "depth": f"{identifier}/depth_meters.npy",
                    "world_points": f"{identifier}/world_points.npy",
                    "calibration": f"{identifier}/calibration.json",
                },
                "evaluation": {"reference": f"{identifier}/evaluation/reference.json"},
                "metadata": {
                    "suite": "libero_goal", "task_index": 0, "task_name": TASK_NAME,
                    "state_index": index, "seed": 0, "benchmark_instruction": task.language,
                    "bddl_instruction": INSTRUCTION, "bddl": str(bddl.relative_to(libero_root)),
                    "bddl_sha256": sha(bddl), "initial_state_path": f"{identifier}/initial_state.npy",
                    "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                    "camera": {"name": CAMERA, "height": SIZE, "width": SIZE},
                    "versions": versions, "capture_checks": checks,
                    "segmentation_use": "evaluation only; never model input or landmark proposal input",
                    "robot_execution_tested": False,
                },
            }
            write_json(folder / "record.json", record)
            records.append(record)
            all_checks[identifier] = checks
            print(json.dumps({"captured": identifier, "mask_pixels": {
                name: item["visible_pixels"] for name, item in reference["masks"].items()
            }, "calibration_passed": checks["passed"], "deterministic": repeat}), flush=True)
        finally:
            env.close()
    (output / "records.jsonl").write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    rgb_hashes = [sha(output / item["model_inputs"]["image"]) for item in records]
    assert len(set(rgb_hashes)) == len(records), "Selected initial states must produce distinct observations"
    files = {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {
        "version": "drawer-rgbd-capture-1.0", "plan": plan,
        "captures": [record["example_id"] for record in records],
        "distinct_rgb_observations": len(set(rgb_hashes)), "checks": all_checks,
        "files": files, "capture_script_sha256": sha(__file__),
        "paid_api_calls": 0, "robot_actions": 0,
        "scope": "four correlated development scenes from one official task; no motion or grasp execution",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/rgbd-drawer-pilot")
    args = parser.parse_args()
    result = capture(ROOT / args.output)
    print(json.dumps({"captures": len(result["captures"]), "distinct_rgb": result["distinct_rgb_observations"],
                      "paid_api_calls": 0, "robot_actions": 0}))
