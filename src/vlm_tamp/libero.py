"""Official LIBERO loading and headless collection; no policy or planner code."""

import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from .bddl import parse_bddl


ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_goal", "libero_spatial", "libero_object", "libero_10")


def configure_libero(root=None, gl=None):
    """Configure paths before importing LIBERO to avoid its interactive prompt."""
    root = Path(root or os.environ.get("LIBERO_ROOT", ROOT / "third_party/LIBERO")).resolve()
    base = root / "libero/libero"
    if not (base / "bddl_files").is_dir():
        raise FileNotFoundError(f"LIBERO checkout missing at {root}; run scripts/setup.sh")
    config = ROOT / ".runtime/libero"
    config.mkdir(parents=True, exist_ok=True)
    import yaml

    paths = {
        "benchmark_root": str(base),
        "bddl_files": str(base / "bddl_files"),
        "init_states": str(base / "init_files"),
        "assets": str(base / "assets"),
        "datasets": str(ROOT / "data/demonstrations_unused"),
    }
    Path(paths["datasets"]).mkdir(parents=True, exist_ok=True)
    (config / "config.yaml").write_text(yaml.safe_dump(paths))
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    # Upstream uses a namespace outer directory omitted by its find_packages().
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("MUJOCO_GL", gl or "egl")
    if gl:
        os.environ["MUJOCO_GL"] = gl
    return root


def version_info(root):
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return {
        "libero_commit": commit,
        "collection_version": "0.1.0",
        "collection_sha256": hashlib.sha256(
            Path(__file__).read_bytes() + Path(__file__).with_name("bddl.py").read_bytes()
        ).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "torch", "mujoco", "robosuite")
        },
    }


def collect(output, suites=("libero_goal",), tasks=(0,), states=(0,), seed=0, size=256, gl=None, root=None):
    root = configure_libero(root, gl)
    import numpy as np
    from PIL import Image
    import robosuite.macros as macros
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    versions = version_info(root)
    for suite_name in suites:
        if suite_name not in SUITES:
            raise ValueError(f"Supported suites: {SUITES}")
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        for task_index in tasks:
            if not 0 <= task_index < suite.n_tasks:
                raise ValueError(f"Invalid task index {task_index}")
            task = suite.get_task(task_index)
            spec_path = Path(suite.get_task_bddl_file_path(task_index))
            parsed = parse_bddl(spec_path)
            initial_states = suite.get_task_init_states(task_index)
            np.random.seed(seed)
            random.seed(seed)
            env = OffScreenRenderEnv(
                bddl_file_name=str(spec_path),
                camera_names=["agentview"],
                camera_heights=size,
                camera_widths=size,
                camera_segmentations="instance",
            )
            try:
                for state_index in states:
                    if not 0 <= state_index < len(initial_states):
                        raise ValueError(f"Invalid initial state index {state_index}")
                    env.seed(seed)
                    env.reset()
                    state = np.asarray(initial_states[state_index])
                    obs = env.set_init_state(state)
                    flip = macros.IMAGE_CONVENTION == "opengl"
                    rgb = obs["agentview_image"]
                    seg = obs.get("agentview_segmentation_instance")
                    if flip:
                        rgb = rgb[::-1]
                        if seg is not None:
                            seg = seg[::-1]
                    if rgb.shape != (size, size, 3) or rgb.dtype != np.uint8 or rgb.std() < 5:
                        raise ValueError("Invalid or nearly blank rendered RGB image")
                    example_id = f"{suite_name}-{task_index:02d}-state{state_index:03d}-seed{seed}"
                    folder = output / example_id
                    folder.mkdir(exist_ok=True)
                    Image.fromarray(rgb).save(folder / "rgb.png")
                    np.save(folder / "initial_state.npy", state)
                    entities = parsed["entities"]
                    masks = {}
                    # Robosuite emits 1 + the insertion-order instance ID, verified below.
                    instances = list(env.env.model.instances_to_ids)
                    if seg is not None:
                        seg = seg.squeeze()
                        if seg.shape != (size, size):
                            raise ValueError("Unexpected segmentation dimensions")
                        np.save(folder / "segmentation.npy", seg)
                        for entity in entities:
                            identifier = entity["identifier"]
                            if identifier not in instances:
                                continue
                            mask = seg == instances.index(identifier) + 1
                            ys, xs = np.where(mask)
                            if len(xs):
                                mask_path = f"mask-{identifier}.png"
                                Image.fromarray((mask * 255).astype(np.uint8)).save(folder / mask_path)
                                masks[identifier] = {
                                    "mask": mask_path,
                                    "visible_pixels": int(mask.sum()),
                                    "bbox": [
                                        float(xs.min() / size),
                                        float(ys.min() / size),
                                        float((xs.max() + 1) / size),
                                        float((ys.max() + 1) / size),
                                    ],
                                    "provenance": "simulator_instance_segmentation",
                                }
                    sim = env.sim
                    geometry, joints = {}, []
                    for entity in entities:
                        identifier = entity["identifier"]
                        body_id = getattr(env.env, "obj_body_id", {}).get(identifier)
                        if body_id is not None:
                            geometry[identifier] = {
                                "position": sim.data.body_xpos[body_id].tolist(),
                                "rotation_matrix": sim.data.body_xmat[body_id].tolist(),
                                "frame": "world",
                                "units": "meters",
                            }
                    for j in range(sim.model.njnt):
                        name = sim.model.joint_id2name(j)
                        if not name or name.startswith(("robot", "gripper")):
                            continue
                        joints.append(
                            {
                                "identifier": name,
                                "type": {0: "free", 1: "ball", 2: "prismatic", 3: "revolute"}[
                                    int(sim.model.jnt_type[j])
                                ],
                                "axis": sim.model.jnt_axis[j].tolist(),
                                "frame": sim.model.body_id2name(int(sim.model.jnt_bodyid[j])),
                                "position": sim.model.jnt_pos[j].tolist(),
                                "range": sim.model.jnt_range[j].tolist(),
                                "provenance": "mujoco_model_body_local",
                            }
                        )
                    cam = sim.model.camera_name2id("agentview")
                    camera = {
                        "name": "agentview",
                        "height": size,
                        "width": size,
                        "fovy_degrees": float(sim.model.cam_fovy[cam]),
                        "world_position": sim.data.cam_xpos[cam].tolist(),
                        "world_rotation": sim.data.cam_xmat[cam].tolist(),
                        "image_origin": "top_left",
                        "source_image_convention": macros.IMAGE_CONVENTION,
                    }
                    record = {
                        "example_id": example_id,
                        "model_inputs": {
                            "instruction": task.language,
                            "image": f"{example_id}/rgb.png",
                            "candidates": entities,
                            "geometry": geometry,
                        },
                        "reference": {
                            "goal_expression": parsed["goal_expression"],
                            "goal_predicates": parsed["goal_predicates"],
                            "entities": entities,
                            "visual": {
                                k: dict(v, mask=f"{example_id}/{v['mask']}") for k, v in masks.items()
                            },
                            "geometry": geometry,
                            "joints": joints,
                            "unsupported_goal_expressions": parsed["unsupported_goal_expressions"],
                            "provenance": {
                                "goals": "official_bddl",
                                "entities": "official_bddl_scene_inventory",
                                "visual": "simulator_instance_segmentation",
                                "joints": "simulator_model",
                            },
                            "coverage": {
                                "symbolic_goals": not parsed["unsupported_goal_expressions"],
                                "instance_masks": bool(masks),
                                "task_relevant_entities": False,
                                "stages": False,
                                "constraints": False,
                                "articulation": False,
                            },
                        },
                        "metadata": {
                            "suite": suite_name,
                            "task_index": task_index,
                            "task_name": task.name,
                            "state_index": state_index,
                            "seed": seed,
                            "camera": camera,
                            "bddl": str(spec_path.relative_to(root)),
                            "bddl_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                            "bddl_instruction": parsed["instruction"],
                            "initial_state_path": f"{example_id}/initial_state.npy",
                            "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                            "segmentation_instances": instances,
                            "versions": versions,
                        },
                    }
                    (folder / "record.json").write_text(json.dumps(record, indent=2))
                    records.append(record)
                    print(f"Collected {example_id}: {task.language}", flush=True)
            finally:
                env.close()
    (output / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return records
