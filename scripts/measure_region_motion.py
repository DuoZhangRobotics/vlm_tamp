"""Compare single-camera and fused CoTracker region poses on frozen recorded inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from measure_recorded_goal_gap import (
    camera_sequence,
    capture_catalog,
    dense_camera_sequence,
    load_depth,
    write,
)
from vlm_tamp.part_motion import CoTrackerBackend, lift_tracks, query_points
from vlm_tamp.region_motion import fit_camera_groups, region_proximity, transformed_region

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ["agentview", "wrist"]
ROLES = ["door", "opening"]


def read(path: Path) -> dict:
    """Read a frozen public record."""
    return json.loads(path.read_text())


def selected_region(
    work: Path, registry: dict, selection: dict, role: str, camera: str, view: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Load exactly the model-selected masks and their measured surface cloud."""
    mask = np.zeros(view["depth"].shape, bool)
    for identifier in selection[role]["views"][camera]["part_ids"]:
        part = registry[identifier]
        path = work / part["mask_file"]
        if part["observation_id"] != selection["observation_id"] or part["camera"] != camera:
            raise ValueError("Selected mask does not belong to the declared reference/camera")
        if hashlib.sha256(path.read_bytes()).hexdigest() != part["mask_sha256"]:
            raise ValueError("Selected mask changed after the model saw it")
        with np.load(path, allow_pickle=False) as data:
            mask |= data["usable_mask"]
    return mask, np.asarray(view["world_points"])[mask]


def interval_evidence(first: dict, last: dict) -> dict:
    """Describe measured region motion without assigning a task direction."""
    result = {
        "start_observation_id": first["observation_id"],
        "end_observation_id": last["observation_id"],
        "start_control": first["control_step"],
        "end_control": last["control_step"],
    }
    before, after = first["conditions"]["both"], last["conditions"]["both"]
    result.update(before_proximity=before["proximity"], after_proximity=after["proximity"])
    for role in ROLES:
        a, b = before[role], after[role]
        if a["fit"]["accepted"] and b["fit"]["accepted"]:
            relative = np.asarray(b["fit"]["transform"]) @ np.linalg.inv(
                np.asarray(a["fit"]["transform"])
            )
            result[role] = {
                "pose_available": True,
                "rotation_magnitude_deg": float(
                    np.degrees(Rotation.from_matrix(relative[:3, :3]).magnitude())
                ),
                "centroid_displacement_world_m": (
                    np.asarray(b["centroid_world_m"]) - a["centroid_world_m"]
                ).tolist(),
                "before_camera_support": a["fit"]["camera_support"],
                "after_camera_support": b["fit"]["camera_support"],
            }
        else:
            result[role] = {
                "pose_available": False,
                "before_reason": a["fit"]["reason"],
                "after_reason": b["fit"]["reason"],
            }
    result["interpretation"] = (
        "Unsigned motion and selected-region proximity only. Determine opening/closing from the visible semantic relationship; no supplied sign or orbit."
    )
    return result


def measure(study: Path, state: int, tracker: CoTrackerBackend) -> None:
    """Track both views separately and fuse their observed world correspondences directly."""
    started = time.perf_counter()
    folder = study / f"state{state:03d}"
    work = folder / "agent-workspace"
    selection = read(work / "references.json")
    frozen = read(folder / "reference-freeze.json")
    if hashlib.sha256((work / "references.json").read_bytes()).hexdigest() != frozen["sha256"]:
        raise ValueError("Region selections changed after freezing")
    registry = read(work / "part_registry.json")
    output = folder / "tracking"
    output.mkdir()
    run = (
        ROOT
        / f"results/luna-molmo2-integration-v3-20261103/trials/baseline-r1-close_microwave-state{state:03d}-600"
    )
    source = run / "agent-workspace/robot_data/attempt01"
    catalog = capture_catalog(source)
    reference_id = selection["observation_id"]
    reference_folder = source / f"obs{reference_id:03d}"
    references = {role: {} for role in ROLES}
    clouds = {role: {} for role in ROLES}
    observations = {
        r["observation_id"]: {
            "observation_id": r["observation_id"],
            "control_step": r["control_step"],
            "depth_retained": (r["folder"] / "sensors.npz").exists(),
            "views": {},
        }
        for r in catalog
    }
    targets = {r["observation_id"]: {role: {} for role in ROLES} for r in catalog}
    timing = []
    for camera in CAMERAS:
        reference = load_depth(reference_folder, camera)
        if reference is None:
            raise ValueError("Register a region in a retained-depth capture")
        reference["rgb"] = np.asarray(
            Image.open(reference_folder / (camera + ".png")).convert("RGB")
        ).copy()
        query_sets = {}
        ranges = {}
        offset = 0
        for role in ROLES:
            mask, cloud = selected_region(work, registry, selection, role, camera, reference)
            query_sets[role] = query_points(reference, mask, maximum=64)
            ranges[role] = (offset, offset + len(query_sets[role]))
            offset += len(query_sets[role])
            clouds[role][camera] = cloud
            world, valid = lift_tracks(
                [reference], query_sets[role][None], np.ones((1, len(query_sets[role])), bool)
            )
            references[role][camera] = world[0]
            np.savez_compressed(
                output / f"{camera}-{role}-reference.npz",
                mask=mask,
                cloud=cloud,
                queries=query_sets[role],
                world_points=world[0],
                valid=valid[0],
            )
        queries = np.concatenate([query_sets[role] for role in ROLES])
        views, frames = (
            dense_camera_sequence(run, catalog, output)
            if camera == "agentview"
            else camera_sequence(run, catalog, camera, output)
        )
        indices = {
            f["observation_id"]: i for i, f in enumerate(frames) if f["observation_id"] is not None
        }
        if len(queries):
            tracks, visible, profile = tracker.predict(
                views, queries, reference_index=indices[reference_id]
            )
        else:
            tracks = np.empty((len(views), 0, 2))
            visible = np.empty((len(views), 0), bool)
            profile = {
                "model_load": 0.0,
                "inference": 0.0,
                "status": "no_selected_visible_material",
            }
        np.savez_compressed(
            output / f"{camera}-tracks.npz", tracks=tracks, visible=visible, queries=queries
        )
        timing.append(
            {"camera": camera, "rgb_frames": len(views), "queries": len(queries), **profile}
        )
        del views
        for capture in catalog:
            identifier = capture["observation_id"]
            index = indices[identifier]
            view = load_depth(capture["folder"], camera)
            current = {}
            for role in ROLES:
                start, stop = ranges[role]
                pixels = tracks[index, start:stop]
                flags = visible[index, start:stop]
                if view is None:
                    world = np.full((len(pixels), 3), np.nan)
                    valid = np.zeros(len(pixels), bool)
                else:
                    lifted, valid_rows = lift_tracks([view], pixels[None], flags[None])
                    world = lifted[0]
                    valid = valid_rows[0]
                targets[identifier][role][camera] = world
                current[role] = {
                    "query_pixels_xy": np.round(pixels, 3).tolist(),
                    "tracker_visible": flags.tolist(),
                    "depth_valid": valid.tolist(),
                    "world_points": [
                        point.tolist() if good else None for point, good in zip(world, valid)
                    ],
                    "query_count": len(pixels),
                    "visible_queries": int(flags.sum()),
                    "depth_supported_queries": int(valid.sum()),
                }
            observations[identifier]["views"][camera] = current
    combined_clouds = {
        role: np.concatenate([clouds[role][camera] for camera in CAMERAS]) for role in ROLES
    }
    rows = []
    for capture in catalog:
        row = observations[capture["observation_id"]]
        conditions = {}
        for condition, names in [
            ("agentview", ["agentview"]),
            ("wrist", ["wrist"]),
            ("both", CAMERAS),
        ]:
            fits = {
                role: fit_camera_groups(
                    {name: references[role][name] for name in names},
                    {name: targets[row["observation_id"]][role][name] for name in names},
                )
                for role in ROLES
            }
            current_clouds = {
                role: transformed_region(combined_clouds[role], fits[role]) for role in ROLES
            }
            conditions[condition] = {
                role: {
                    "fit": fits[role],
                    "reference_centroid_world_m": combined_clouds[role].mean(0).tolist()
                    if len(combined_clouds[role])
                    else None,
                    "centroid_world_m": current_clouds[role].mean(0).tolist()
                    if current_clouds[role] is not None and len(current_clouds[role])
                    else None,
                }
                for role in ROLES
            }
            conditions[condition]["proximity"] = region_proximity(
                current_clouds["door"], current_clouds["opening"]
            )
        row["conditions"] = conditions
        rows.append(row)
    retained = [r for r in rows if r["depth_retained"]]
    selected = [
        retained[i]
        for i in np.unique(
            np.rint(np.linspace(0, len(retained) - 1, min(4, len(retained)))).astype(int)
        )
    ]
    intervals = [
        {"interval_id": i + 1, **interval_evidence(a, b)}
        for i, (a, b) in enumerate(zip(selected, selected[1:]))
    ]
    result = {
        "selection": selection,
        "samples": rows,
        "tracking": timing,
        "seconds": time.perf_counter() - started,
        "comparison_scope": "Same selected regions and tracker outputs for both camera ablations; both is the primary estimate",
        "fit_thresholds_changed": False,
        "pose_fit_frame": "Absolute current pose relative to the common reference capture",
        "robot_motion_executed": False,
        "motion_circle_supplied": False,
        "native_state_used": False,
        "closing_direction_hardcoded": False,
        "intervals": intervals,
    }
    write(output / "measurements.json", result)
    print(
        json.dumps(
            {
                "state": state,
                "seconds": result["seconds"],
                "tracking": timing,
                "available_proximity": {
                    name: sum(r["conditions"][name]["proximity"]["available"] for r in rows)
                    for name in ["agentview", "wrist", "both"]
                },
            }
        ),
        flush=True,
    )


def main() -> None:
    """Run on the selected GPU after model-authored regions are frozen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    if not (study / "selection-complete.json").exists():
        raise RuntimeError("Finish both region selections before tracking")
    from supervise_luna_dashboard_study import wait_for_capacity

    wait_for_capacity(study / "tracking-resources", "two-view-region-tracking")
    tracker = CoTrackerBackend(device="cuda")
    try:
        for state in [17, 49]:
            measure(study, state, tracker)
    finally:
        tracker.close()
    write(study / "measurement-complete.json", {"complete": True})


if __name__ == "__main__":
    main()
