"""Ask fresh Luna agents to interpret two-view region motion without native labels or orbits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from select_recorded_goal_references import RecordedService, select_references, write

PROMPT = """Determine what the microwave door actually does in each of the three recorded intervals. The requested task was closing, but the robot can move it the wrong way. Do not infer actual motion from the task request, arm movement, a previous label or a presumed hinge sign.
Read tools.txt, regions.json and intervals.json. These regions were selected by a separate fresh Luna/Molmo2 session. Inspect reference-regions-agentview.png and reference-regions-wrist.png. Their semantic identities may still be uncertain; you may disagree with the selected regions.
For EACH interval, call inspect_interval and inspect both camera images before and after. The annotated pairs have original-resolution frames and tracked region points; raw images are also available. The wrist camera moves with the robot, so compare the door to the fixed opening/rim, not to image left/right. Measurements combine both calibrated cameras; single-camera results are diagnostic comparisons, not automatic substitutes.
The feedback contains unsigned rigid motion and nearest-surface distances between the selected regions. No orbit, closing sign, exact mating point, native object angle or expected label is provided. Smaller region distance is not automatically closing, and missing measurements do not imply stationary motion. Use the visible semantic relationship between the door and fixed opening. If that relationship cannot be established from the available evidence, say unclear and explain what is missing. You may inspect other provided public captures if needed.
Write judgments.json with {"assessments":[{"interval_id":1,"label":"opening|closing|unchanged|unclear","evidence":"brief visible before/after evidence and any useful measured support","uncertainties":"occlusion, region identity, small motion or conflicting evidence"}, ...],"region_identity_assessment":"whether the selected door and opening regions are appropriate"}. Include all interval IDs exactly once. No robot motion is available. Stay inside this public workspace.
"""

TOOLS = """Read-only motion interpretation. No planning, pointing, orbit estimation or robot action is enabled.
python3 robot.py inspect_interval '{"interval_id":1}' returns before/after pairs from BOTH cameras, raw image paths and descriptive region-motion measurements.
python3 robot.py observe '{"observation_id":1}' returns both original camera images from that saved public capture.
Read regions.json for the model-authored region selection; read intervals.json for interval IDs/timestamps.
"""


def pair_image(path: Path, work: Path, camera: str, first: dict, last: dict) -> None:
    """Render original-resolution paired observations and measured tracking queries."""
    canvas = Image.new("RGB", (1024, 540), "white")
    label = ImageDraw.Draw(canvas)
    for column, row in enumerate([first, last]):
        frame = Image.open(
            work / "captures" / f"obs{row['observation_id']:03d}" / (camera + ".png")
        ).convert("RGB")
        draw = ImageDraw.Draw(frame)
        for role, color in [("door", "#1b72d4"), ("opening", "#e78516")]:
            for pixel, visible in zip(
                row["views"][camera][role]["query_pixels_xy"],
                row["views"][camera][role]["tracker_visible"],
            ):
                if not visible:
                    continue
                x, y = pixel
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline="white", width=1)
        canvas.paste(frame, (column * 512, 28))
        label.text(
            (column * 512 + 8, 8),
            f"{'BEFORE' if column == 0 else 'AFTER'} | {camera} | control {row['control_step']} | blue door, orange opening",
            fill="black",
        )
    canvas.save(path)


def judge_service(previous: Path) -> Callable:
    """Expose compact measured feedback and source RGB, with no privileged evaluator data."""

    def build(source: Path, work: Path, output: Path, endpoint: Path | None) -> RecordedService:
        """Prepare a fresh public workspace from frozen selection and tracking artifacts."""
        service = RecordedService(source, work, output, None)
        measurements = json.loads((previous / "tracking/measurements.json").read_text())
        selection = measurements["selection"]
        rows = {r["observation_id"]: r for r in measurements["samples"]}
        write(work / "regions.json", selection)
        write(
            work / "intervals.json",
            [
                {
                    k: v
                    for k, v in r.items()
                    if k
                    in [
                        "interval_id",
                        "start_control",
                        "end_control",
                        "start_observation_id",
                        "end_observation_id",
                    ]
                }
                for r in measurements["intervals"]
            ],
        )
        for camera in ["agentview", "wrist"]:
            picture = np.asarray(
                Image.open(
                    work / "captures" / f"obs{selection['observation_id']:03d}" / (camera + ".png")
                ).convert("RGB")
            ).copy()
            for role, color in [("door", [20, 98, 173]), ("opening", [220, 125, 13])]:
                with np.load(
                    previous / f"tracking/{camera}-{role}-reference.npz", allow_pickle=False
                ) as data:
                    mask = data["mask"]
                picture[mask] = np.rint(0.65 * picture[mask] + 0.35 * np.asarray(color)).astype(
                    np.uint8
                )
            Image.fromarray(picture).save(work / f"reference-regions-{camera}.png")
        replies = {}
        for interval in measurements["intervals"]:
            identifier = interval["interval_id"]
            first = rows[interval["start_observation_id"]]
            last = rows[interval["end_observation_id"]]
            images = []
            raw = []
            for camera in ["agentview", "wrist"]:
                path = service.public / f"interval{identifier:02d}-{camera}.png"
                pair_image(path, work, camera, first, last)
                images.append(str(path.relative_to(work)))
                raw.extend(
                    f"captures/obs{r['observation_id']:03d}/{camera}.png" for r in [first, last]
                )
            replies[identifier] = {
                **interval,
                "paired_images": images,
                "raw_images": raw,
                "camera_ablation": {
                    name: {
                        "before": first["conditions"][name]["proximity"],
                        "after": last["conditions"][name]["proximity"],
                    }
                    for name in ["agentview", "wrist"]
                },
            }
            write(service.public / f"interval{identifier:02d}.json", replies[identifier])

        def inspect_interval(interval_id: int) -> dict:
            """Return one exact chronological interval without a task-motion label."""
            return replies[interval_id]

        service.extra_operations["inspect_interval"] = inspect_interval
        return service

    return build


def main() -> None:
    """Freeze judgments before independent native scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    output = study / "judgment"
    output.mkdir()
    if not (study / "measurement-complete.json").exists():
        raise RuntimeError("Finish region tracking before interpretation")
    for state in [17, 49]:
        select_references(
            output,
            state,
            None,
            prompt=PROMPT,
            service_factory=judge_service(study / f"state{state:03d}"),
            seconds=600,
            tools_description=TOOLS,
            output_name="judgments.json",
            freeze_metadata={
                "phase": "two-view-motion-interpretation",
                "frozen_before_gap_measurement": False,
                "gap_trends_supplied": True,
                "frozen_before_native_evaluation": True,
                "motion_circle_supplied": False,
            },
        )
    write(study / "judgment-complete.json", {"complete": True})


if __name__ == "__main__":
    main()
