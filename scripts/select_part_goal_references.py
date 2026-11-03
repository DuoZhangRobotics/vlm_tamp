"""Select goal references and rigid-part tracking regions with explicit geometry feedback."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np

from select_recorded_goal_references import RecordedService, model_worker, select_references, write
from vlm_tamp.goal_gap import landmark_sample
from vlm_tamp.observation_completion import CompletionEvidence

ROOT = Path(__file__).resolve().parents[1]

PROMPT = """The recorded task is to close a microwave door. Select a goal-reference pair and the rigid parts that carry those references. This is read-only: no robot actions, native object state, prior successful coordinates or expected motion labels are available.
Read tools.txt. Inspect the saved sequence and both original cameras as needed. Use Molmo2 point proposals as assistance, and inspect them; you decide what to retain. Choose a material point on the moving closing/free edge and its corresponding visible point on the fixed appliance opening where it should meet when closed. A nearby point on a housing top or side is not automatically the matching closure point. If the first view is ambiguous, inspect another saved capture/camera before finalizing.
Use segment_parts to select spatially distributed visible material on the whole moving door and on the fixed appliance. You may include several masks from the same camera/capture for a rigid part. These are tracking supports, not just a tiny patch around the reference. Include frame corners and other identifiable material where visible. Background seen through an opening/transparency does not move with the part; a depth-supported pixel alone does not establish part identity. Inspect mask overlays and revise their prompts if necessary.
Use reference_geometry on your candidate pair, with a surface_polygon_xy for each role covering a visible material face near that reference. Read the 3D coordinates, coordinate differences, measured surface normals and support. Same image row does not imply same 3D height. Explain why the points/surfaces could have the claimed closed relationship under the observed articulation; no world-axis hinge or closing sign is supplied. If the geometry or surface identity contradicts that interpretation, revise your own selections and inspect again. Geometry feedback is not a semantic certificate.
Each reference must identify the camera/capture of its tracking masks. The chosen point need not be inside every tracking patch, but you must explain why the patches and point belong to the same rigid part. Captures without retained depth are listed and can still be inspected as RGB; choose a capture with depth to register a 3D part.
Write references.json with:
{"can_select":true|false,"moving":{"observation_id":1,"camera":"agentview|wrist","pixel_xy":[x,y],"part_ids":["selected part ID"],"surface_polygon_xy":[[x,y],...],"description":"material point and tracking supports","selection_source":"Molmo2 proposal or your selection","evidence":"why these belong to one moving rigid part"},"fixed_closed":{"observation_id":1,"camera":"agentview|wrist","pixel_xy":[x,y],"part_ids":["selected part ID"],"surface_polygon_xy":[[x,y],...],"description":"matching fixed frame reference and supports","selection_source":"Molmo2 proposal or your selection","evidence":"why these belong to the fixed appliance"},"geometry_file":"file returned by the final reference_geometry call","why_the_pair_expresses_closure":"reason using visible evidence and geometry","uncertainties":"limitations"}.
Use integer original-image pixels. You may retain a tentative visible pair with can_select=false and explicit uncertainty. Leave a role null if it truly cannot be identified; do not invent a hidden point. Freeze the final choices before whole-part tracking and gap evaluation. Do not give a push direction. Stay in this workspace.
"""


def build_service(source: Path, work: Path, output: Path, endpoint: Path) -> RecordedService:
    """Compose existing segmentation/surface tools with the recorded pointing service."""
    service = RecordedService(source, work, output, endpoint)
    for identifier, record in service.records.items():
        original = record["source_sensors"]
        destination = Path(record["sensors"]).parent / "sensors.npz"
        if original.exists():
            shutil.copyfile(original, destination)
            record.update(
                sensors=destination,
                sensor_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                calibration_sha256=hashlib.sha256(
                    Path(record["calibration"]).read_bytes()
                ).hexdigest(),
            )
        for entry in service.catalog:
            if entry["observation_id"] == identifier:
                entry["depth_retained"] = original.exists()
    write(work / "captures.json", service.catalog)
    write(service.public / "episode-status.json", service.episode_status())
    surfaces = CompletionEvidence(service.public / "surfaces")

    def segment_parts(**arguments) -> dict:
        """Keep the model-selected part IDs and their immutable recorded masks."""
        record = service.perception._record(arguments["observation_id"])
        if not Path(record["sensors"]).exists():
            raise ValueError("This capture has RGB but its depth export was not retained")
        result = service.perception.segment_parts(service.calls, **arguments)
        registry = {
            key: {
                **value["summary"],
                "mask_sha256": hashlib.sha256(
                    (work / value["summary"]["mask_file"]).read_bytes()
                ).hexdigest(),
            }
            for key, value in service.perception.parts.items()
        }
        write(work / "part_registry.json", registry)
        return result

    def surface_summary(reference: dict) -> dict | None:
        """Summarize a caller-selected measured patch without a semantic label."""
        polygon = reference.get("surface_polygon_xy")
        if polygon is None:
            return None
        record = service.perception._record(reference["observation_id"])
        result = surfaces.surface(
            record, reference["observation_id"], reference["camera"], polygon
        )
        keys = [
            "supported",
            "normal_world",
            "normal_toward_camera_world",
            "centroid_world_m",
            "selected_points",
            "inlier_points",
            "inlier_fraction",
            "rms_m",
            "surface_spans_m",
            "measurement_id",
        ]
        return {
            **{k: result[k] for k in keys},
            "overlay": str(Path(result["overlay"]).relative_to(work)),
            "meaning": "Observed patch geometry only; no object or closing-goal label.",
        }

    def reference_geometry(moving: dict, fixed_closed: dict) -> dict:
        """Expose literal reference geometry before the actor freezes its choices."""
        references = {"moving": moving, "fixed_closed": fixed_closed}
        result = {}
        images = []
        for role, ref in references.items():
            record = service.perception._record(ref["observation_id"])
            if not Path(record["sensors"]).exists():
                raise ValueError("Select a retained-depth capture for a 3D reference")
            view = service.perception._view(ref["observation_id"], ref["camera"])
            sample = landmark_sample(view, ref["pixel_xy"])
            parts = []
            for identifier in ref["part_ids"]:
                part = service.perception.parts[identifier]
                if (
                    part["observation_id"] != ref["observation_id"]
                    or part["camera"] != ref["camera"]
                ):
                    raise ValueError(
                        "A reference and its tracking masks need the same capture/camera"
                    )
                parts.append(part["summary"])
            surface = surface_summary(ref)
            if surface:
                images.append(surface["overlay"])
            result[role] = {"sample": sample, "surface": surface, "tracking_parts": parts}
        a = result["moving"]["sample"]["raw_world_m"]
        b = result["fixed_closed"]["sample"]["raw_world_m"]
        difference = (np.asarray(b) - a) if a is not None and b is not None else None
        result.update(
            reference_parameters=references,
            fixed_minus_moving_world_m=difference.tolist() if difference is not None else None,
            vertical_difference_mm=float(difference[2] * 1000) if difference is not None else None,
            coordinate_convention="Metres in the calibrated world frame; world Z is upward. No hinge orientation or closing sign is assumed.",
            semantic_match_verified=False,
            images=images,
            interpretation="Check the actual material surfaces and whether the 3D separation is compatible with their intended relationship when closed. Depth support and a plane fit do not establish that relationship.",
        )
        path = service.public / f"reference-geometry-{service.calls:03d}.json"
        result["geometry_file"] = str(path.relative_to(work))
        write(path, result)
        return result

    service.extra_operations.update(
        segment_parts=segment_parts, reference_geometry=reference_geometry
    )
    service.additional_help = """
Whole-part tracking preparation:
segment_parts(observation_id, camera, points_xy=null, point_labels=null, box_xyxy=null, label=null) uses the existing MobileSAM model. Supply positive/negative pixel clicks and/or a box. Inspect the returned masks and choose part IDs. Multiple selected masks from one camera/capture can represent one rigid part.
reference_geometry(moving, fixed_closed) takes the two proposed reference dictionaries with observation_id, camera, pixel_xy, part_ids and surface_polygon_xy. It returns literal 3D coordinates, differences and measured local plane normals, plus overlays. It does not choose a point, hinge or motion. Use a retained-depth capture.
Whole-part estimation will use multiple landmarks on your selected material masks. The marked reference is carried by the fitted part motion rather than relying on its individual track after occlusion. No prior or repaired pose will replace an unsupported fit.
"""
    return service


def main() -> None:
    """Freeze model-authored references and masks before the recorded-data test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    study.mkdir(parents=True, exist_ok=True)
    from supervise_luna_dashboard_study import wait_for_capacity

    wait_for_capacity(study, "whole-part-reference-selection")
    write(
        study / "protocol.json",
        {
            "states": [17, 49],
            "actor": "gpt-6-luna",
            "reasoning": "high",
            "fresh_agent_per_state": True,
            "selection_seconds": 900,
            "pointing_backend": "allenai/Molmo2-8B",
            "segmentation_backend": "existing MobileSAM",
            "geometry_feedback": "observed coordinates, differences and caller-selected surface patches",
            "tracking_backend": "existing CoTracker3 scaled_offline; all recorded external RGB",
            "queries_per_rigid_part_maximum": 64,
            "motion_fit": "existing fit_object_motion 3D rigid consensus",
            "inlier_threshold_m": 0.004,
            "minimum_inliers": 8,
            "majority_required": True,
            "minimum_spatial_span_m": 0.025,
            "absolute_reference_fits": True,
            "single_reference_track_used_in_pose_fit": False,
            "missing_depth": "No current 3D part pose; no interpolated depth or held pose",
            "native_state_supplied_to_models_or_fitter": False,
            "closing_direction_hardcoded": False,
            "robot_motion_executed": False,
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        },
    )
    with model_worker(study) as endpoint:
        for state in [17, 49]:
            select_references(
                study, state, endpoint, prompt=PROMPT, service_factory=build_service, seconds=900
            )
    write(study / "selection-complete.json", {"complete": True})


if __name__ == "__main__":
    main()
