"""Fresh Luna/Molmo2 region selection in both recorded cameras; no motion-circle input."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from select_part_goal_references import build_service
from select_recorded_goal_references import RecordedService, model_worker, select_references, write
from vlm_tamp.region_motion import region_proximity

PROMPT = """The task being studied is closing a microwave. Select the physical moving door region and the fixed opening/rim region from these recorded observations. This is read-only perception preparation, not robot execution.
Read tools.txt and captures.json. Inspect BOTH native cameras, agentview and wrist. Use Molmo2 proposals to help locate the door and the fixed opening/rim; inspect their overlays. Use segment_parts to select visible physical material on each part in BOTH views. A hole/window/background seen through the door is not door material. The fixed opening/rim is the boundary against which the door closes, not automatically the whole roof or outer housing. Select several masks if useful. If a part is hidden in one camera, record an empty part_ids list and explain that limitation; do not invent visibility. You may inspect different captures before choosing a common retained-depth reference capture for all four selections. The numerical tracker combines both cameras in one calibrated world frame.
You do not need to find two exact mating pixels, a hinge, an orbit, a goal angle, a push direction or a closing sign. Identify regions and explain their semantic relationship. Inspect region_geometry before finalizing; geometry does not certify identity. Be explicit about uncertainty. Do not infer any observed movement merely from the requested task.
Write references.json with this schema:
{"observation_id":1,"can_identify_regions":true|false,"door":{"description":"physical region","views":{"agentview":{"part_ids":["part ID"],"evidence":"visible material"},"wrist":{"part_ids":["part ID"],"evidence":"visible material"}}},"opening":{"description":"fixed opening or rim","views":{"agentview":{"part_ids":["part ID"],"evidence":"why this is the opening boundary"},"wrist":{"part_ids":["part ID"],"evidence":"why this is the opening boundary"}}},"closure_relation":"visible relation that would indicate the door is closed","uncertainties":"limitations"}.
Use original 512x512 pixels for segmentation prompts. Every selected part ID must belong to its declared camera at the common observation_id. Keep tentative visible regions with can_identify_regions=false when identity remains uncertain; use empty IDs for a region that cannot be located. No prior successful coordinates, native object state, measured-motion circle, expected motion label or notebook is supplied. Stay in this workspace.
"""

TOOLS = """Recorded RGB-D only; no robot actions. Read captures.json for camera timestamps and retained depth.
python3 robot.py observe '{"observation_id":1}' returns both original camera images.
python3 robot.py point '{"observation_id":1,"camera":"agentview","query":"your visible-region description"}' requests Molmo2 proposals. Inspect the result; it is not a semantic guarantee.
python3 robot.py segment_parts '{"observation_id":1,"camera":"agentview","points_xy":[[x,y]],"point_labels":[1],"label":"your region"}' uses MobileSAM. Positive/negative clicks and/or box_xyxy are supported. Inspect the masks and retain their part IDs.
python3 robot.py region_geometry '{"regions":YOUR_REFERENCES_OBJECT}' reports selected clouds and current region proximity. It does not label motion or choose a closed configuration.
"""


def build_region_service(
    source: Path, work: Path, output: Path, endpoint: Path
) -> RecordedService:
    """Reuse observation, pointing and segmentation; expose no point-pair or orbit API."""
    service = build_service(source, work, output, endpoint)
    service.extra_operations.pop("reference_geometry")
    service.additional_help = ""

    def region_geometry(regions: dict) -> dict:
        """Describe the actor's selected material using measured depth only."""
        identifier = regions["observation_id"]
        clouds = {}
        result = {}
        images = []
        for role in ["door", "opening"]:
            rows = []
            points = []
            for camera in ["agentview", "wrist"]:
                view = service.perception._view(identifier, camera)
                mask = np.zeros(view["depth"].shape, bool)
                for part_id in regions[role]["views"][camera]["part_ids"]:
                    part = service.perception.parts[part_id]
                    if part["observation_id"] != identifier or part["camera"] != camera:
                        raise ValueError(
                            "All selected masks must match the declared common capture and camera"
                        )
                    mask |= part["mask"]
                    images.append(part["summary"]["overlay"])
                cloud = np.asarray(view["world_points"])[mask]
                points.append(cloud)
                rows.append(
                    {
                        "camera": camera,
                        "points": len(cloud),
                        "centroid_world_m": cloud.mean(0).tolist() if len(cloud) else None,
                        "world_span_m": np.ptp(cloud, axis=0).tolist() if len(cloud) else None,
                    }
                )
            clouds[role] = np.concatenate(points)
            result[role] = {"views": rows, "selected_points": len(clouds[role])}
        result.update(
            proximity=region_proximity(clouds["door"], clouds["opening"]),
            images=images,
            semantic_identity_verified=False,
            closing_direction_supplied=False,
            scope="Selected observed material only. No automatic snapping, hidden surface completion or goal-direction label.",
        )
        write(service.public / f"region-geometry-{service.calls:03d}.json", result)
        return result

    service.extra_operations["region_geometry"] = region_geometry
    return service


def main() -> None:
    """Freeze fresh model selections before any tracking or native evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    study.mkdir(parents=True, exist_ok=True)
    from supervise_luna_dashboard_study import wait_for_capacity

    wait_for_capacity(study, "two-view-region-selection")
    write(
        study / "protocol.json",
        {
            "states": [17, 49],
            "actor": "gpt-6-luna",
            "reasoning": "high",
            "fresh_agent_per_state": True,
            "pointing": "allenai/Molmo2-8B",
            "segmentation": "existing MobileSAM",
            "cameras": ["agentview", "wrist"],
            "selection": "door and opening/rim regions; no mating-point requirement",
            "queries_per_part_per_camera": 64,
            "tracking": "existing CoTracker3; all saved external RGB and all saved wrist captures",
            "fit": "existing general 3D rigid consensus with unchanged thresholds",
            "conditions": ["agentview only", "wrist only", "both cameras; primary result"],
            "judge_intervals": "Three intervals spanning four uniformly selected retained-depth captures, chosen without native labels",
            "distance_semantics": "Descriptive surface proximity only; no automatic closing label",
            "motion_circle_supplied": False,
            "native_state_supplied": False,
            "closing_direction_hardcoded": False,
            "robot_motion_executed": False,
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        },
    )
    with model_worker(study) as endpoint:
        for state in [17, 49]:
            select_references(
                study,
                state,
                endpoint,
                prompt=PROMPT,
                service_factory=build_region_service,
                seconds=900,
                tools_description=TOOLS,
            )
    write(study / "selection-complete.json", {"complete": True})


if __name__ == "__main__":
    main()
