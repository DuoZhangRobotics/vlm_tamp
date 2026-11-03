"""Estimate selected-reference gaps from CoTracker whole-part RGB-D correspondences."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from measure_recorded_goal_gap import (
    capture_catalog,
    camera_sequence,
    dense_camera_sequence,
    load_depth,
    write,
)
from vlm_tamp.goal_gap import landmark_sample, measured_gap
from vlm_tamp.part_motion import CoTrackerBackend, lift_tracks, query_points
from vlm_tamp.part_goal_reference import carry_reference, part_reference_gap

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    """Read an already saved declaration."""
    return json.loads(path.read_text())


def role_support(work: Path, source: Path, reference: dict, registry: dict) -> dict:
    """Bind the actor's masks and reference to the same recorded object pose."""
    camera = reference["camera"]
    identifier = reference["observation_id"]
    capture = source / f"obs{identifier:03d}"
    view = load_depth(capture, camera)
    if view is None:
        raise ValueError("A whole-part model needs retained depth at its reference capture")
    view["rgb"] = np.asarray(Image.open(capture / (camera + ".png")).convert("RGB")).copy()
    mask = np.zeros(view["depth"].shape, bool)
    for part_id in reference["part_ids"]:
        part = registry[part_id]
        if part["observation_id"] != identifier or part["camera"] != camera:
            raise ValueError("Tracking masks must match their declared reference capture/camera")
        path = work / part["mask_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != part["mask_sha256"]:
            raise ValueError("Selected mask changed after the model proposal was recorded")
        with np.load(path, allow_pickle=False) as data:
            mask |= data["usable_mask"]
    queries = query_points(view, mask, maximum=64)
    anchor = np.asarray(reference["pixel_xy"], float)
    queries = queries[np.linalg.norm(queries - anchor, axis=1) > 1e-6]
    sample = landmark_sample(view, anchor)
    return {
        "camera": camera,
        "observation_id": identifier,
        "view": view,
        "mask": mask,
        "queries": queries,
        "reference_sample": sample,
        "reference_pixel_xy": anchor.tolist(),
    }


def measure_case(
    study: Path, state: int, tracker: CoTrackerBackend | None, *, use_saved_tracks: bool = False
) -> None:
    """Fit each retained-depth capture directly to the reference, avoiding pose accumulation."""
    started = time.perf_counter()
    folder = study / f"state{state:03d}"
    work = folder / "agent-workspace"
    frozen = read(folder / "reference-freeze.json")
    path = work / "references.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != frozen["sha256"]:
        raise ValueError("References changed after freeze")
    refs = read(path)
    registry = read(work / "part_registry.json")
    output = folder / "whole-part"
    output.mkdir(exist_ok=use_saved_tracks)
    if any(refs.get(role) is None for role in ["moving", "fixed_closed"]):
        write(
            output / "gap.json",
            {"status": "unresolved_references", "samples": [], "references": refs},
        )
        return
    run = ROOT / (
        f"results/luna-molmo2-integration-v3-20261103/trials/"
        f"baseline-r1-close_microwave-state{state:03d}-600"
    )
    source = run / "agent-workspace/robot_data/attempt01"
    catalog = capture_catalog(source)
    supports = {
        role: role_support(work, source, refs[role], registry)
        for role in ["moving", "fixed_closed"]
    }
    groups = defaultdict(list)
    for role, support in supports.items():
        groups[support["camera"], support["observation_id"]].append(role)
    rows = {
        r["observation_id"]: {
            "observation_id": r["observation_id"],
            "control_step": r["control_step"],
        }
        for r in catalog
    }
    sequences = {}
    profiles = []
    for (camera, reference_id), roles in groups.items():
        if camera not in sequences:
            if use_saved_tracks:
                frames = read(output / (camera + "-tracking-inputs.json"))
                checked_videos = set()
                for frame in frames:
                    if (
                        frame.get("path")
                        and hashlib.sha256(Path(frame["path"]).read_bytes()).hexdigest()
                        != frame["sha256"]
                    ):
                        raise ValueError("Saved RGB capture changed")
                    if frame.get("source_video") and frame["source_video"] not in checked_videos:
                        if (
                            hashlib.sha256(Path(frame["source_video"]).read_bytes()).hexdigest()
                            != frame["source_video_sha256"]
                        ):
                            raise ValueError("Saved RGB video changed")
                        checked_videos.add(frame["source_video"])
                sequences[camera] = (None, frames)
            else:
                sequences[camera] = (
                    dense_camera_sequence(run, catalog, output)
                    if camera == "agentview"
                    else camera_sequence(run, catalog, camera, output)
                )
        views, frames = sequences[camera]
        indices = {
            r["observation_id"]: i for i, r in enumerate(frames) if r["observation_id"] is not None
        }
        reference_index = indices[reference_id]
        combined = []
        ranges = {}
        for role in roles:
            start = len(combined)
            combined.extend(supports[role]["queries"].tolist())
            stop = len(combined)
            combined.append(supports[role]["reference_pixel_xy"])
            ranges[role] = {"start": start, "stop": stop, "anchor_index": stop}
        queries = np.asarray(combined, np.float32)
        prefix = f"{camera}-ref{reference_id:03d}"
        if use_saved_tracks:
            with np.load(output / (prefix + "-tracks.npz"), allow_pickle=False) as stored:
                tracks, visible = stored["tracks"], stored["visible"]
                if not np.array_equal(stored["queries"], queries):
                    raise ValueError("Frozen query set changed")
            profile = read(output / (prefix + "-tracking.json"))
        else:
            tracks, visible, timing = tracker.predict(
                views, queries, reference_index=reference_index
            )
            np.savez_compressed(
                output / (prefix + "-tracks.npz"), tracks=tracks, visible=visible, queries=queries
            )
            profile = {
                "camera": camera,
                "reference_observation_id": reference_id,
                "rgb_frames": len(frames),
                "query_count": len(queries),
                "roles": ranges,
                **timing,
            }
            write(output / (prefix + "-tracking.json"), profile)
        profiles.append(profile)
        for role in roles:
            support = supports[role]
            scope = ranges[role]
            points = tracks[reference_index, scope["start"] : scope["stop"]]
            flags = np.ones((1, len(points)), bool)
            world, valid = lift_tracks([support["view"]], points[None], flags)
            support["reference_points"] = world[0]
            support["reference_valid"] = valid[0]
            np.savez_compressed(
                output / (role + "-reference.npz"),
                queries=support["queries"],
                world_points=world[0],
                valid=valid[0],
                mask=support["mask"],
            )
        for capture in catalog:
            frame_index = indices[capture["observation_id"]]
            current = load_depth(capture["folder"], camera)
            for role in roles:
                support = supports[role]
                scope = ranges[role]
                pixels = tracks[frame_index, scope["start"] : scope["stop"]]
                flags = visible[frame_index, scope["start"] : scope["stop"]]
                direct = landmark_sample(
                    current,
                    tracks[frame_index, scope["anchor_index"]],
                    bool(visible[frame_index, scope["anchor_index"]]),
                )
                direct.update(camera=camera)
                anchor = support["reference_sample"]
                if current is None:
                    estimate = {
                        "fit": {"accepted": False, "reason": "recorded_depth_not_retained"},
                        "estimated_reference_world_m": None,
                    }
                    valid = np.zeros(len(pixels), bool)
                else:
                    world, valid_rows = lift_tracks([current], pixels[None], flags[None])
                    valid = valid_rows[0]
                    target = world[0].copy()
                    target[~support["reference_valid"]] = np.nan
                    if anchor["supported"]:
                        estimate = carry_reference(
                            support["reference_points"], target, anchor["raw_world_m"]
                        )
                    else:
                        estimate = {
                            "fit": {"accepted": False, "reason": "reference_depth_unsupported"},
                            "estimated_reference_world_m": None,
                        }
                fit = estimate["fit"]
                row = {
                    **estimate,
                    "camera": camera,
                    "reference_observation_id": reference_id,
                    "query_count": len(pixels),
                    "visible_queries": int(flags.sum()),
                    "depth_supported_queries": int(valid.sum()),
                    "query_pixels_xy": pixels.tolist(),
                    "tracker_visible": flags.tolist(),
                    "depth_valid": valid.tolist(),
                    "direct_reference_sample": direct,
                    "reference_sample": anchor,
                    "inlier_indices": fit.get("inlier_indices", []),
                }
                rows[capture["observation_id"]][role] = row
    samples = []
    for capture in catalog:
        row = rows[capture["observation_id"]]
        row.update(part_reference_gap(row["moving"], row["fixed_closed"]))
        row["direct_point_gap"] = measured_gap(
            row["moving"]["direct_reference_sample"],
            row["fixed_closed"]["direct_reference_sample"],
        )
        samples.append(row)
    result = {
        "status": "whole_part_estimates",
        "references": refs,
        "samples": samples,
        "supported_captures": sum(r["supported"] for r in samples),
        "capture_count": len(samples),
        "tracking": profiles,
        "seconds": time.perf_counter() - started,
        "reference_fit_mode": "Each capture is fitted directly to its registered reference; no sequential pose propagation",
        "fit_method": "Existing fit_object_motion: 4 mm rigid residual, 8 inliers, strict majority and 25 mm span",
        "reference_track_included_in_fit": False,
        "unsupported_pose_replaced": False,
        "gap_is_estimated_from_part_pose": True,
        "robot_motion_executed": False,
        "used_saved_tracks": use_saved_tracks,
        "native_state_used_for_estimation": False,
        "closing_direction_supplied": False,
    }
    write(output / "gap.json", result)
    print(
        json.dumps(
            {
                "state": state,
                "supported": result["supported_captures"],
                "captures": len(samples),
                "seconds": result["seconds"],
            }
        ),
        flush=True,
    )


def main() -> None:
    """Run only after the new reference/mask selections have been frozen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--states", nargs="+", type=int, choices=[17, 49], default=[17, 49])
    parser.add_argument(
        "--use-saved-tracks",
        action="store_true",
        help="Explicit postprocessing of saved, hash-checked CoTracker outputs",
    )
    args = parser.parse_args()
    if not (args.study / "selection-complete.json").exists():
        raise RuntimeError("Complete both selection sessions first")
    if not args.use_saved_tracks:
        from supervise_luna_dashboard_study import wait_for_capacity

        wait_for_capacity(args.study / "tracking-resources", "whole-part-goal-tracking")
    tracker = None if args.use_saved_tracks else CoTrackerBackend(device="cuda")
    try:
        for state in args.states:
            measure_case(args.study, state, tracker, use_saved_tracks=args.use_saved_tracks)
    finally:
        if tracker is not None:
            tracker.close()
    if all((args.study / f"state{state:03d}/whole-part/gap.json").exists() for state in [17, 49]):
        write(args.study / "measurement-complete.json", {"complete": True})


if __name__ == "__main__":
    main()
