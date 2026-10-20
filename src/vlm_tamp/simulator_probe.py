"""Independent kinematic and static placement probes; never robot execution."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import read_records, safe_path
from .libero import configure_libero


def observation_only(record):
    """Drop the benchmark-answer compartment before physical verification."""
    return {key: record[key] for key in ("example_id", "model_inputs", "metadata")}


def contact_penetrations(contacts, object_geoms):
    selected = [c for c in contacts if (int(c.geom1) in object_geoms) != (int(c.geom2) in object_geoms)]
    return max((max(0.0, -float(c.dist)) for c in selected), default=0.0)


def geometry_vertices(sim, geom_ids):
    """Exact box vertices (the bowl uses boxes); other shapes remain unsupported."""
    signs = np.array(list(itertools.product((-1, 1), repeat=3)))
    vertices = []
    for geom in geom_ids:
        if int(sim.model.geom_type[geom]) != 6:
            return None
        local = signs * np.asarray(sim.model.geom_size[geom])
        rotation = np.asarray(sim.data.geom_xmat[geom]).reshape(3, 3)
        vertices.append(local @ rotation.T + np.asarray(sim.data.geom_xpos[geom]))
    return np.concatenate(vertices) if vertices else None


def contained_vertices(vertices, center, rotation, half_extents, tolerance=1e-7):
    if vertices is None:
        return None
    local = (vertices - center) @ rotation
    return bool(np.all(np.abs(local) <= np.asarray(half_extents) + tolerance))


def _image(env, camera):
    import robosuite.macros as macros

    env.env._update_observables(force=True)
    pixels = env.env._get_observations()[f"{camera}_image"]
    return pixels[::-1].copy() if macros.IMAGE_CONVENTION == "opengl" else pixels.copy()


def _joint_fact(env, identifier, site, samples):
    model, data = env.sim.model, env.sim.data
    joint = site.joints[0]
    jid = model.joint_name2id(joint)
    kind = {2: "prismatic", 3: "revolute"}.get(int(model.jnt_type[jid]))
    if kind is None or not model.jnt_limited[jid]:
        return None
    body = int(model.jnt_bodyid[jid])
    site_body = int(model.site_bodyid[model.site_name2id(identifier)])
    cursor = site_body
    while cursor > 0 and cursor != body:
        cursor = int(model.body_parentid[cursor])
    if cursor != body:
        raise ValueError("Region site is not attached to its declared joint subtree")
    original_q = float(data.get_joint_qpos(joint))
    rotation = np.asarray(data.body_xmat[body]).reshape(3, 3).copy()
    axis_local = np.asarray(model.jnt_axis[jid]).copy()
    axis_world = rotation @ axis_local
    anchor = np.asarray(data.body_xpos[body]).copy() + rotation @ np.asarray(model.jnt_pos[jid])
    sweep = []
    try:
        for q in np.linspace(*model.jnt_range[jid], samples):
            data.set_joint_qpos(joint, float(q))
            env.sim.forward()
            sweep.append(
                {
                    "q": float(q),
                    "open": bool(env.env.object_states_dict[identifier].is_open()),
                    "body_position_world": data.body_xpos[body].tolist(),
                    "body_rotation_world": data.body_xmat[body].tolist(),
                }
            )
    finally:
        data.set_joint_qpos(joint, original_q)
        env.sim.forward()
    displacement = np.asarray(sweep[-1]["body_position_world"]) - sweep[0]["body_position_world"]
    dq = sweep[-1]["q"] - sweep[0]["q"]
    max_rotation_change = max(
        float(np.max(np.abs(np.asarray(s["body_rotation_world"]).reshape(3, 3) - rotation))) for s in sweep
    )
    slide_error = float(np.linalg.norm(displacement - axis_world * dq)) if kind == "prismatic" else None
    return {
        "entity_id": identifier,
        "parent_id": site.parent_name,
        "joint_id": joint,
        "body": model.body_id2name(body),
        "joint_type": kind,
        "axis_local": axis_local.tolist(),
        "axis_world": axis_world.tolist(),
        "axis_point_world": anchor.tolist(),
        "limits": model.jnt_range[jid].tolist(),
        "coordinate_units": "m" if kind == "prismatic" else "rad",
        "initial_q": original_q,
        "initial_open": bool(env.env.object_states_dict[identifier].is_open()),
        "sweep": sweep,
        "translation_axis_residual_m": slide_error,
        "rotation_residual": max_rotation_change,
        "open_toggled_in_sweep": len({s["open"] for s in sweep}) == 2,
        "scope": "model/site/joint identity and forward-kinematic sweep; no robot actuation or task-necessity proof",
    }


def collect_physical_facts(dataset, example_id, output, moving_object, target_region, grid_size=5, gl="egl"):
    if not isinstance(grid_size, int) or not 2 <= grid_size <= 9:
        raise ValueError("Placement grid_size must be between 2 and 9")
    records, root = read_records(dataset)
    matches = [r for r in records if r["example_id"] == example_id]
    if len(matches) != 1:
        raise ValueError("Unknown or duplicate probe example")
    record = observation_only(matches[0])
    candidates = {e["identifier"]: e for e in record["model_inputs"]["candidates"]}
    if moving_object not in candidates or target_region not in candidates:
        raise ValueError("Probe entities must be supplied scene candidates")
    output = Path(output).resolve()
    if (output / "facts.json").exists():
        raise FileExistsError("Physical facts already exist; use a new --output")
    output.mkdir(parents=True, exist_ok=True)
    configure_libero(gl=gl)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.predicates import eval_predicate_fn

    meta, inputs = record["metadata"], record["model_inputs"]
    suite = benchmark.get_benchmark_dict()[meta["suite"]](task_order_index=0)
    spec = Path(suite.get_task_bddl_file_path(meta["task_index"]))
    if hashlib.sha256(spec.read_bytes()).hexdigest() != meta["bddl_sha256"]:
        raise ValueError("Benchmark environment changed since collection")
    state = np.load(safe_path(root, meta["initial_state_path"]), allow_pickle=False)
    if hashlib.sha256(state.tobytes()).hexdigest() != meta["initial_state_sha256"]:
        raise ValueError("Initial state changed since collection")
    original_pixels = np.array(Image.open(safe_path(root, inputs["image"])).convert("RGB"))
    camera = meta["camera"]
    env = OffScreenRenderEnv(
        bddl_file_name=str(spec),
        camera_names=[camera["name"]],
        camera_heights=camera["height"],
        camera_widths=camera["width"],
    )
    try:
        env.seed(meta["seed"])
        env.reset()
        # Avoid the wrapper's check_success() call: only physical predicates are queried here.
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        env.env._post_process()
        if not np.array_equal(_image(env, camera["name"]), original_pixels):
            raise ValueError("Probe re-render differs from the collected observation")
        Image.fromarray(original_pixels).save(output / "original.png")
        joints = {}
        for identifier, site in env.env.object_sites_dict.items():
            if identifier in candidates and site.joints and len(site.joints) == 1:
                fact = _joint_fact(env, identifier, site, 9)
                if fact is not None:
                    joints[identifier] = fact
        if target_region not in joints:
            raise ValueError("This trial needs a verified articulated target region")
        fact = joints[target_region]
        if not fact["open_toggled_in_sweep"]:
            raise ValueError("Target joint sweep did not toggle the official Open predicate")
        site = env.env.object_sites_dict[target_region]
        moving = env.env.get_object(moving_object)
        moving_joints = [
            j for j in moving.joints if int(env.sim.model.jnt_type[env.sim.model.joint_name2id(j)]) == 0
        ]
        if len(moving_joints) != 1:
            raise ValueError("Placement probe requires one free-object joint")
        free_joint = moving_joints[0]
        pose = env.sim.data.get_joint_qpos(free_joint).copy()
        object_geoms = {env.sim.model.geom_name2id(name) for name in moving.contact_geoms}
        original_in = bool(
            eval_predicate_fn(
                "in", env.env.object_states_dict[moving_object], env.env.object_states_dict[target_region]
            )
        )
        closed_q = min(
            (s["q"] for s in fact["sweep"] if not s["open"]), key=lambda q: abs(q - fact["initial_q"])
        )
        open_q = min((s["q"] for s in fact["sweep"] if s["open"]))
        probes = {}
        tolerance = 0.0005
        for label, q in (("closed", closed_q), ("open", open_q)):
            env.sim.set_state_from_flattened(state)
            env.sim.data.set_joint_qpos(fact["joint_id"], q)
            env.sim.forward()
            center = env.sim.data.get_site_xpos(target_region).copy()
            rotation = env.sim.data.get_site_xmat(target_region).copy()
            samples = []
            for offset in itertools.product(np.linspace(-0.65, 0.65, grid_size), repeat=3):
                trial_pose = pose.copy()
                trial_pose[:3] = center + rotation @ (np.asarray(offset) * site.size)
                env.sim.data.set_joint_qpos(free_joint, trial_pose)
                env.sim.forward()
                membership = bool(
                    eval_predicate_fn(
                        "in",
                        env.env.object_states_dict[moving_object],
                        env.env.object_states_dict[target_region],
                    )
                )
                penetration = contact_penetrations(env.sim.data.contact, object_geoms)
                inside = contained_vertices(
                    geometry_vertices(env.sim, object_geoms), center, rotation, site.size
                )
                samples.append(
                    {
                        "position_world": trial_pose[:3].tolist(),
                        "quaternion_wxyz": trial_pose[3:].tolist(),
                        "official_in_predicate": membership,
                        "max_penetration_m": penetration,
                        "collision_tolerance_pass": penetration <= tolerance,
                        "all_collision_box_vertices_in_site": inside,
                    }
                )
            accepted = [s for s in samples if s["official_in_predicate"] and s["collision_tolerance_pass"]]
            fully_contained = [s for s in accepted if s["all_collision_box_vertices_in_site"] is True]
            chosen = min(fully_contained or accepted or samples, key=lambda s: s["max_penetration_m"])
            trial_pose = np.array(chosen["position_world"] + chosen["quaternion_wxyz"])
            env.sim.data.set_joint_qpos(free_joint, trial_pose)
            env.sim.forward()
            Image.fromarray(_image(env, camera["name"])).save(output / f"{label}-probe.png")
            probes[label] = {
                "joint_q": q,
                "open_predicate": bool(env.env.object_states_dict[target_region].is_open()),
                "site_position_world": center.tolist(),
                "site_rotation_world": rotation.tolist(),
                "site_half_extents": np.asarray(site.size).tolist(),
                "tested": len(samples),
                "official_in_count": sum(s["official_in_predicate"] for s in samples),
                "nonpenetrating_in_count": len(accepted),
                "fully_contained_collision_box_count": len(fully_contained),
                "chosen_candidate": chosen,
                "samples": samples,
                "image": f"{label}-probe.png",
            }
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        restored = np.array_equal(_image(env, camera["name"]), original_pixels)
        if not restored:
            raise ValueError("Original scene did not restore after probes")
        result = {
            "schema_version": "1.0",
            "example_id": example_id,
            "source": "independently_probed_official_simulator",
            "benchmark_goal_answers_used": False,
            "initial_state_sha256": meta["initial_state_sha256"],
            "bddl_sha256": meta["bddl_sha256"],
            "image_sha256": hashlib.sha256(safe_path(root, inputs["image"]).read_bytes()).hexdigest(),
            "libero_commit": meta["versions"]["libero_commit"],
            "probe_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "moving_object": moving_object,
            "target_region": target_region,
            "joints": joints,
            "initial_in_predicate": original_in,
            "placement_probes": probes,
            "collision_penetration_tolerance_m": tolerance,
            "original_scene_restored": restored,
            "containment_semantics": "Official site In checks the object reference point; site contact is always true. Full collision-box containment is an additional conservative site-proxy check.",
            "scope": "Finite static-pose and forward-kinematic probes. Objects are repositioned directly; no robot grasp, dynamic stability or executable trajectory is established.",
        }
        (output / "facts.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()
