"""Let Luna reconsider fixed references with the measured whole-part orbit visible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from select_part_goal_references import build_service, PROMPT
from select_recorded_goal_references import RecordedService, model_worker, select_references, write


def observed_reference_orbit(
    transforms: ArrayLike, reference_world_m: ArrayLike
) -> dict[str, Any]:
    """Describe the full revolute orbit measured from part poses, without choosing a goal angle.

    This is geometry for reference selection, not a verified articulation model or a motion plan.
    All rotations determine the axis; no vertical-axis prior is supplied.
    """
    poses = np.asarray(transforms, float).reshape(-1, 4, 4)
    anchor = np.asarray(reference_world_m, float)
    vectors = Rotation.from_matrix(poses[:, :3, :3]).as_rotvec()
    moving = np.linalg.norm(vectors, axis=1) > 1e-6
    if not moving.any():
        return {"supported": False, "reason": "no_observed_rotation"}
    _, _, basis = np.linalg.svd(vectors[moving], full_matrices=False)
    axis = basis[0]
    _, _, perpendicular = np.linalg.svd(axis[None], full_matrices=True)
    plane = perpendicular[1:].T
    matrices = poses[moving, :3, :3] - np.eye(3)
    target = -poses[moving, :3, 3].reshape(-1)
    system = (matrices @ plane).reshape(-1, 2)
    center = plane @ np.linalg.lstsq(system, target, rcond=None)[0]
    radial = anchor - center - axis * np.dot(anchor - center, axis)
    circle_center = anchor - radial
    angles = np.linspace(-np.pi, np.pi, 181)
    circle = Rotation.from_rotvec(angles[:, None] * axis).apply(radial) + circle_center
    return {
        "supported": True,
        "source": "Observed rigid-part transforms only",
        "axis_world_unsigned": axis.tolist(),
        "axis_origin_world_m": center.tolist(),
        "circle_center_world_m": circle_center.tolist(),
        "radius_m": float(np.linalg.norm(radial)),
        "points_world_m": circle.tolist(),
        "pose_equation_rms_m": float(
            np.sqrt(np.mean((system @ (plane.T @ center) - target) ** 2))
        ),
        "measured_rotation_degrees": np.degrees(np.linalg.norm(vectors[moving], axis=1)).tolist(),
        "goal_angle_selected": False,
        "native_state_used": False,
        "interpretation": "Every direction around the measured circle is shown. Compatibility with this circle does not establish that a fixed point is the closing rim.",
    }


def orbit_distance(orbit: dict, point_world_m: ArrayLike) -> dict[str, float]:
    """Measure mismatch with the observed point orbit; do not project or replace a selection."""
    point = np.asarray(point_world_m, float)
    offset = point - np.asarray(orbit["circle_center_world_m"])
    axial = float(np.dot(offset, orbit["axis_world_unsigned"]))
    radius = float(np.linalg.norm(offset - axial * np.asarray(orbit["axis_world_unsigned"])))
    radial = radius - orbit["radius_m"]
    return {
        "axial_difference_m": axial,
        "radial_difference_m": radial,
        "minimum_distance_to_orbit_m": float(np.hypot(axial, radial)),
    }


def refinement_service(previous: Path) -> Callable[[Path, Path, Path, Path], RecordedService]:
    """Add observed-motion evidence, without native labels or an automatic replacement point."""

    def build(source: Path, work: Path, output: Path, endpoint: Path) -> RecordedService:
        """Compose recorded tools and measured-orbit feedback for this selection session."""
        service = build_service(source, work, output, endpoint)
        original = json.loads((previous / "references.json").read_text())
        measured = json.loads((previous / "whole-part/gap.json").read_text())
        reference = next(
            r
            for r in measured["samples"]
            if r["observation_id"] == original["moving"]["observation_id"]
        )
        transforms = [
            r["moving"]["fit"]["transform"]
            for r in measured["samples"]
            if r["moving"]["fit"]["accepted"]
        ]
        orbit = observed_reference_orbit(transforms, reference["moving"]["reference_world_m"])
        if not orbit["supported"]:
            raise ValueError("The recorded part poses do not resolve a rotation orbit")
        write(work / "previous-references.json", original)
        registry = json.loads((previous / "agent-workspace/part_registry.json").read_text())
        for part_id in set(original["moving"]["part_ids"] + original["fixed_closed"]["part_ids"]):
            part = registry[part_id]
            for key in ["mask_file", "overlay"]:
                target = work / part[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(previous / "agent-workspace" / part[key], target)
            service.perception.parts[part_id] = {
                "summary": part,
                "observation_id": part["observation_id"],
                "camera": part["camera"],
            }
        write(work / "part_registry.json", registry)
        service.calls = 100  # Preserve the previous model-authored IDs without overwriting them.
        orbit["previous_fixed_reference_compatibility"] = orbit_distance(
            orbit, reference["fixed_closed"]["reference_world_m"]
        )
        orbit["images"] = []
        for role in ["moving", "fixed_closed"]:
            ref = original[role]
            capture = work / "captures" / f"obs{ref['observation_id']:03d}"
            for camera in ["agentview", "wrist"]:
                calibration = json.loads((capture / "calibration.json").read_text())[camera]
                points = np.asarray(orbit["points_world_m"])
                matrix = np.linalg.inv(np.asarray(calibration["camera_to_world"]))
                local = points @ matrix[:3, :3].T + matrix[:3, 3]
                projected = local @ np.asarray(calibration["intrinsic"]).T
                uv = projected[:, :2] / projected[:, 2, None]
                image = Image.open(capture / (camera + ".png")).convert("RGB")
                draw = ImageDraw.Draw(image)
                for a, b, z1, z2 in zip(uv[:-1], uv[1:], local[:-1, 2], local[1:, 2]):
                    if z1 > 0 and z2 > 0 and np.isfinite([a, b]).all():
                        draw.line([tuple(a), tuple(b)], fill="#da2dcb", width=2)
                draw.rectangle((0, 0, 512, 23), fill="white")
                draw.text((5, 5), "Measured full orbit; neither direction is a goal", fill="black")
                path = service.public / f"orbit-{role}-{camera}.png"
                image.save(path)
                orbit["images"].append(str(path.relative_to(work)))
        write(work / "observed-orbit.json", orbit)
        base = service.extra_operations["reference_geometry"]

        def reference_geometry(moving: dict, fixed_closed: dict) -> dict:
            """Expose the chosen point's geometric mismatch without replacing it."""
            if moving != original["moving"]:
                raise ValueError(
                    "Keep the recorded moving reference unchanged for this comparison"
                )
            result = base(moving, fixed_closed)
            point = result["fixed_closed"]["sample"]["raw_world_m"]
            result["measured_orbit_compatibility"] = (
                orbit_distance(orbit, point) if point else None
            )
            result["orbit_interpretation"] = (
                "Distance from the measured full orbit, not a semantic guarantee of closure. No point was moved or selected for you."
            )
            write(work / result["geometry_file"], result)
            return result

        service.extra_operations["reference_geometry"] = reference_geometry
        service.additional_help += "\nRead previous-references.json and observed-orbit.json. Inspect the orbit overlays. Keep the moving reference dictionary unchanged; reconsider the fixed visible reference. reference_geometry additionally reports its distance from the measured orbit. The orbit comes from recorded part poses, not simulator state.\n"
        return service

    return build


def main() -> None:
    """Run independent, explicitly informed reference revisions without moving the robot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    output = study / "orbit-refinement"
    output.mkdir()
    from supervise_luna_dashboard_study import wait_for_capacity

    wait_for_capacity(output, "orbit-reference-refinement")
    prompt = (
        PROMPT
        + """
This revision additionally receives the earlier model-authored hypotheses and the whole-part motion measured from these recordings. Read previous-references.json and observed-orbit.json and view the full-orbit overlays. The circle is the path the moving reference can follow under the measured revolute motion. It shows all directions and contains no selected closing angle. The fitted orbit may be uncertain; its measured rotations and residual are reported. A fixed reference far from that circle cannot be the matching contact for this moving point. A point close to it still needs a visible semantic justification.
Keep the moving reference dictionary exactly unchanged to isolate the fixed-reference revision. Use the existing fixed-part masks if appropriate; your fixed point may be elsewhere on that same rigid appliance. You can choose another retained-depth view/capture and new masks if needed. Inspect both cameras and ask Molmo2 for a proposal. Use reference_geometry to examine the geometric compatibility of your chosen point; revise it yourself if necessary. Do not silently snap to the circle or infer a hidden point. If a correct visible fixed reference cannot be established, state that explicitly. Write the same references.json schema. No native labels, expected closing direction or operator-selected pixels are supplied.
"""
    )
    write(
        output / "protocol.json",
        {
            "source": "Earlier model selections and measured whole-part transforms",
            "native_state_supplied": False,
            "closing_direction_supplied": False,
            "fixed_reference_revision_only": True,
            "actor": "gpt-6-luna",
            "reasoning": "high",
        },
    )
    with model_worker(output) as endpoint:
        for state in [17, 49]:
            select_references(
                output,
                state,
                endpoint,
                prompt=prompt,
                service_factory=refinement_service(study / f"state{state:03d}"),
                seconds=900,
            )
    write(output / "selection-complete.json", {"complete": True})


if __name__ == "__main__":
    main()
