"""Independently checked simulator facts, separate from expert task-plan labels."""

import hashlib
import copy
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field, model_validator

from .dataset import read_records, safe_path
from .grounding import Model, Unit


class PartFact(Model):
    entity_id: str
    parent_id: str
    body: str
    joint_id: str
    joint_type: Literal["prismatic", "revolute"]
    axis_local: list[float] = Field(min_length=3, max_length=3)
    axis_world: list[float] = Field(min_length=3, max_length=3)
    joint_position_world: list[float] = Field(min_length=3, max_length=3)
    axis_sign_invariant: bool = True
    goal_related: bool
    mask: str | None = None
    mask_sha256: str | None = None
    visible_pixels: int = Field(ge=0)
    bbox: list[Unit] | None = Field(default=None, min_length=4, max_length=4)
    geometric_scope: str = "visible articulated body associated with the BDDL site, not the interior volume"
    verification: Literal["site_joint_body_mask_and_state_crosscheck"]
    open_predicate_at_limits: list[bool] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def valid_fact(self):
        if len(set(self.open_predicate_at_limits)) != 2:
            raise ValueError("Joint-limit perturbation must independently toggle the official Open predicate")
        if not np.isclose(np.linalg.norm(self.axis_local), 1) or not np.isclose(
            np.linalg.norm(self.axis_world), 1
        ):
            raise ValueError("Reference joint axes must be normalized")
        if self.visible_pixels and (not self.mask or not self.mask_sha256 or not self.bbox):
            raise ValueError("Visible parts require measured mask and box references")
        return self


class SimulatorReference(Model):
    schema_version: Literal["1.0"] = "1.0"
    example_id: str
    source: Literal["verified_official_simulator"] = "verified_official_simulator"
    bddl_sha256: str
    initial_state_sha256: str
    image_sha256: str
    libero_commit: str
    verifier_sha256: str
    camera: str
    parts: list[PartFact] = Field(default_factory=list)
    excluded_labels: list[str] = Field(
        default_factory=lambda: [
            "required task stages",
            "grasp strategy",
            "contact requirements",
            "path requirements",
        ]
    )


def load_simulator_references(path):
    if path is None:
        return {}
    files = sorted(Path(path).glob("*.json")) if Path(path).is_dir() else [Path(path)]
    result = {}
    for file in files:
        ref = SimulatorReference.model_validate_json(file.read_text())
        if ref.example_id in result:
            raise ValueError("Duplicate simulator reference")
        result[ref.example_id] = ref
    return result


def check_reference(record, root, reference):
    if record["example_id"] != reference.example_id:
        raise ValueError("Simulator reference example mismatch")
    for key in ("bddl_sha256", "initial_state_sha256"):
        if record["metadata"][key] != getattr(reference, key):
            raise ValueError(f"Simulator reference {key} mismatch")
    image = safe_path(root, record["model_inputs"]["image"])
    if hashlib.sha256(image.read_bytes()).hexdigest() != reference.image_sha256:
        raise ValueError("Simulator reference image mismatch")
    if record["metadata"]["versions"]["libero_commit"] != reference.libero_commit:
        raise ValueError("Simulator reference benchmark version mismatch")
    known = {e["identifier"] for e in record["reference"]["entities"]}
    for part in reference.parts:
        if part.entity_id not in known or part.parent_id not in known:
            raise ValueError("Simulator reference points to unknown entity")
        if part.mask:
            path = safe_path(root, part.mask)
            if hashlib.sha256(path.read_bytes()).hexdigest() != part.mask_sha256:
                raise ValueError("Simulator part mask changed since verification")
            mask = np.array(Image.open(path)) > 0
            if mask.shape != (record["metadata"]["camera"]["height"], record["metadata"]["camera"]["width"]):
                raise ValueError("Simulator part mask camera dimensions mismatch")
            if int(mask.sum()) != part.visible_pixels:
                raise ValueError("Simulator part pixel count mismatch")
            height, width = mask.shape
            ys, xs = np.where(mask)
            expected = [xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height]
            if not np.allclose(part.bbox, expected):
                raise ValueError("Simulator part box is inconsistent with measured mask")
            parent_visual = record["reference"]["visual"].get(part.parent_id)
            if parent_visual is None:
                raise ValueError("Visible part must have a visible parent instance")
            parent_mask = np.array(Image.open(safe_path(root, parent_visual["mask"]))) > 0
            if np.any(mask & ~parent_mask):
                raise ValueError("Simulator part mask escapes the verified parent mask")
    return reference


def attach_reference(record, root, reference):
    """Attach measured part masks in memory, preserving the collection manifest."""
    check_reference(record, root, reference)
    record = copy.deepcopy(record)
    for part in reference.parts:
        if part.mask:
            if part.entity_id in record["reference"]["visual"]:
                raise ValueError("Derived part reference must not overwrite an existing object mask")
            record["reference"]["visual"][part.entity_id] = {
                "mask": part.mask,
                "visible_pixels": part.visible_pixels,
                "bbox": part.bbox,
                "kind": "part",
                "parent_id": part.parent_id,
                "provenance": "verified_simulator_articulated_body_segmentation",
                "geometric_scope": part.geometric_scope,
            }
    return record


def _subtree_bodies(model, root_body):
    bodies = {root_body}
    for body in range(model.nbody):
        current = body
        while current > 0 and current not in bodies:
            current = int(model.body_parentid[current])
        if current in bodies:
            bodies.add(body)
    return bodies


def generate_simulator_references(dataset, output, example_ids=None, gl="egl"):
    """Re-render original states, then independently verify each eligible drawer part.

    New mask files are added under derived/, but original manifests/images are preserved.
    No candidate inventory or model-visible geometric metadata is changed.
    """
    from .libero import configure_libero

    configure_libero(gl=gl)
    import robosuite.macros as macros
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    records, root = read_records(dataset)
    selected = set(example_ids) if example_ids else {r["example_id"] for r in records}
    if not selected <= {r["example_id"] for r in records}:
        raise ValueError("Unknown simulator-reference example IDs")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    namespace = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    references = []
    for record in records:
        if record["example_id"] not in selected:
            continue
        destination = output / f"{record['example_id']}.json"
        if destination.exists():
            raise FileExistsError("Simulator references already exist; use a new --output")
        meta = record["metadata"]
        suite = benchmark.get_benchmark_dict()[meta["suite"]](task_order_index=0)
        spec = Path(suite.get_task_bddl_file_path(meta["task_index"]))
        if hashlib.sha256(spec.read_bytes()).hexdigest() != meta["bddl_sha256"]:
            raise ValueError("Official BDDL changed since collection")
        state = np.asarray(suite.get_task_init_states(meta["task_index"])[meta["state_index"]])
        if hashlib.sha256(state.tobytes()).hexdigest() != meta["initial_state_sha256"]:
            raise ValueError("Official initial state changed since collection")
        camera = meta["camera"]
        env = OffScreenRenderEnv(
            bddl_file_name=str(spec),
            camera_names=[camera["name"]],
            camera_heights=camera["height"],
            camera_widths=camera["width"],
            camera_segmentations=[["instance", "element"]],
        )
        try:
            env.seed(meta["seed"])
            env.reset()
            observation = env.set_init_state(state)
            rgb = observation[f"{camera['name']}_image"]
            element = observation[f"{camera['name']}_segmentation_element"].squeeze()
            instance = observation[f"{camera['name']}_segmentation_instance"].squeeze()
            if macros.IMAGE_CONVENTION == "opengl":
                rgb, element, instance = rgb[::-1], element[::-1], instance[::-1]
            original = np.array(Image.open(safe_path(root, record["model_inputs"]["image"])).convert("RGB"))
            if not np.array_equal(rgb, original):
                raise ValueError("Re-rendered reference image differs from the original observation")
            model, data = env.sim.model, env.sim.data
            instances = list(env.env.model.instances_to_ids)
            goal_entities = {a for g in record["reference"]["goal_predicates"] for a in g["arguments"]}
            parts = []
            known = {e["identifier"] for e in record["reference"]["entities"]}
            for identifier, site in env.env.object_sites_dict.items():
                if identifier not in known or not site.joints or len(site.joints) != 1:
                    continue
                joint_name = site.joints[0]
                joint_id = model.joint_name2id(joint_name)
                joint_type = {2: "prismatic", 3: "revolute"}.get(int(model.jnt_type[joint_id]))
                if joint_type is None or not model.jnt_limited[joint_id]:
                    continue
                body_id = int(model.jnt_bodyid[joint_id])
                subtree = _subtree_bodies(model, body_id)
                site_body = int(model.site_bodyid[model.site_name2id(identifier)])
                if site_body not in subtree or site.parent_name not in instances:
                    raise ValueError("Official site/joint/body association disagrees")
                geometry_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in subtree]
                mask = np.isin(element, geometry_ids)
                parent_mask = instance == instances.index(site.parent_name) + 1
                if np.any(mask & ~parent_mask):
                    raise ValueError("Part pixels are inconsistent with parent instance segmentation")
                axis_local = np.asarray(model.jnt_axis[joint_id]).copy()
                axis_world = np.asarray(data.body_xmat[body_id]).reshape(3, 3) @ axis_local
                anchor_world = np.asarray(data.body_xpos[body_id]) + np.asarray(
                    data.body_xmat[body_id]
                ).reshape(3, 3) @ np.asarray(model.jnt_pos[joint_id])
                openness = []
                try:
                    for limit in model.jnt_range[joint_id]:
                        data.set_joint_qpos(joint_name, float(limit))
                        env.sim.forward()
                        openness.append(bool(env.env.object_states_dict[identifier].is_open()))
                finally:
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                if len(set(openness)) != 2:
                    continue
                relative_mask, mask_hash, bbox = None, None, None
                if mask.any():
                    relative_mask = f"derived/{namespace}/{record['example_id']}/mask-{identifier}.png"
                    mask_file = safe_path(root, relative_mask)
                    mask_file.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_file)
                    mask_hash = hashlib.sha256(mask_file.read_bytes()).hexdigest()
                    ys, xs = np.where(mask)
                    bbox = [
                        float(xs.min() / mask.shape[1]),
                        float(ys.min() / mask.shape[0]),
                        float((xs.max() + 1) / mask.shape[1]),
                        float((ys.max() + 1) / mask.shape[0]),
                    ]
                parts.append(
                    PartFact(
                        entity_id=identifier,
                        parent_id=site.parent_name,
                        body=model.body_id2name(body_id),
                        joint_id=joint_name,
                        joint_type=joint_type,
                        axis_local=axis_local.tolist(),
                        axis_world=axis_world.tolist(),
                        joint_position_world=anchor_world.tolist(),
                        goal_related=identifier in goal_entities,
                        mask=relative_mask,
                        mask_sha256=mask_hash,
                        visible_pixels=int(mask.sum()),
                        bbox=bbox,
                        verification="site_joint_body_mask_and_state_crosscheck",
                        open_predicate_at_limits=openness,
                    )
                )
            reference = SimulatorReference(
                example_id=record["example_id"],
                bddl_sha256=meta["bddl_sha256"],
                initial_state_sha256=meta["initial_state_sha256"],
                image_sha256=hashlib.sha256(
                    safe_path(root, record["model_inputs"]["image"]).read_bytes()
                ).hexdigest(),
                libero_commit=meta["versions"]["libero_commit"],
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                camera=camera["name"],
                parts=parts,
            )
            check_reference(record, root, reference)
            destination.write_text(reference.model_dump_json(indent=2))
            references.append(reference)
            print(f"Verified {record['example_id']}: {len(parts)} articulated parts", flush=True)
        finally:
            env.close()
    return references
