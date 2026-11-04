"""Publish two-camera region tracking and separately score frozen Luna motion judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from report_part_goal_gap import resources, selection_timing

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["agentview", "wrist", "both"]


def read(path: Path) -> dict:
    """Read a finished local artifact."""
    return json.loads(path.read_text())


def check_frozen(folder: Path, name: str) -> dict:
    """Check that the public model output predates independent scoring."""
    path = folder / "agent-workspace" / name
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != read(folder / "reference-freeze.json")["sha256"]
    ):
        raise ValueError("Model output changed after freezing")
    return read(path)


def case_report(study: Path, state: int) -> dict:
    """Keep native truth confined to this report, after both model sessions finish."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = study / f"state{state:03d}"
    judge = study / "judgment" / folder.name
    selection = check_frozen(folder, "references.json")
    judgments = check_frozen(judge, "judgments.json")
    data = read(folder / "tracking/measurements.json")
    run = (
        ROOT
        / f"results/luna-molmo2-integration-v3-20261103/trials/baseline-r1-close_microwave-state{state:03d}-600"
    )
    native_file = read(run / "attempt01/initial/evaluation_contact_diagnostics.json")
    if native_file["excluded_from_policy_input"] is not True:
        raise ValueError("Expected evaluator-only export")
    native = {
        r["step"]: next(
            j
            for j in r["evaluation_only_articulation_state"]
            if j["joint_name"] == "microwave_1_microjoint"
        )
        for r in native_file["steps"]
    }
    reference = next(
        r for r in data["samples"] if r["observation_id"] == selection["observation_id"]
    )
    reference_native = native[reference["control_step"]]
    origin = np.asarray(reference_native["anchor_world_m"])
    axis = np.asarray(reference_native["axis_world"])
    initial = {}
    for camera in ["agentview", "wrist"]:
        initial[camera] = {}
        for role in ["door", "opening"]:
            with np.load(
                folder / f"tracking/{camera}-{role}-reference.npz", allow_pickle=False
            ) as arrays:
                initial[camera][role] = arrays["world_points"]
    rows = []
    for row in data["samples"]:
        rotation = Rotation.from_rotvec(
            axis * (native[row["control_step"]]["position"] - reference_native["position"])
        )
        row["evaluator"] = {
            "native_opening_deg": float(abs(np.degrees(native[row["control_step"]]["position"]))),
            "conditions": {},
        }
        for condition in CONDITIONS:
            estimate = row["conditions"][condition]["door"]
            point = estimate["reference_centroid_world_m"]
            expected = (
                rotation.apply(np.asarray(point) - origin) + origin if point is not None else None
            )
            predicted = estimate["centroid_world_m"]
            row["evaluator"]["conditions"][condition] = {
                "door_centroid_consistency_error_mm": float(
                    np.linalg.norm(np.asarray(predicted) - expected) * 1000
                )
                if predicted is not None and expected is not None
                else None
            }
        for camera in ["agentview", "wrist"]:
            for role in ["door", "opening"]:
                sample = row["views"][camera][role]
                points = initial[camera][role]
                expected = (
                    rotation.apply(points - origin) + origin
                    if role == "door" and len(points)
                    else points
                )
                errors = [
                    float(np.linalg.norm(np.asarray(point) - target) * 1000)
                    if point is not None and np.isfinite(target).all()
                    else None
                    for point, target in zip(sample["world_points"], expected)
                ]
                sample["native_motion_residual_mm"] = errors
                sample.pop("world_points")
        rows.append(row)
    assessments = judgments["assessments"]
    ids = [r["interval_id"] for r in assessments]
    if sorted(ids) != [r["interval_id"] for r in data["intervals"]]:
        raise ValueError("Judgments must include every interval exactly once")
    intervals = []
    for interval in data["intervals"]:
        answer = next(r for r in assessments if r["interval_id"] == interval["interval_id"])
        if answer["label"] not in ["opening", "closing", "unchanged", "unclear"]:
            raise ValueError("Unknown motion label")
        before = abs(native[interval["start_control"]]["position"])
        after = abs(native[interval["end_control"]]["position"])
        delta = float(np.degrees(after - before))
        truth = "opening" if delta > 0 else "closing" if delta < 0 else "unchanged"
        intervals.append(
            {
                **interval,
                "judgment": answer,
                "native_opening_change_deg": delta,
                "native_label": truth,
                "grade": "unclear"
                if answer["label"] == "unclear"
                else "correct"
                if answer["label"] == truth
                else "incorrect",
            }
        )
    summary = {}
    for condition in CONDITIONS:
        errors = [
            r["evaluator"]["conditions"][condition]["door_centroid_consistency_error_mm"]
            for r in rows
            if r['observation_id']!=selection['observation_id']
        ]
        errors = [v for v in errors if v is not None]
        summary[condition] = {
            "door_pose_captures": sum(
                r["conditions"][condition]["door"]["fit"]["accepted"] for r in rows
            ),
            "non_reference_door_pose_captures": sum(
                r['conditions'][condition]['door']['fit']['accepted'] for r in rows
                if r['observation_id']!=selection['observation_id']
            ),
            "opening_pose_captures": sum(
                r["conditions"][condition]["opening"]["fit"]["accepted"] for r in rows
            ),
            "proximity_captures": sum(
                r["conditions"][condition]["proximity"]["available"] for r in rows
            ),
            "maximum_door_centroid_error_mm": max(errors) if errors else None,
            "median_door_centroid_error_mm": float(np.median(errors)) if errors else None,
        }
    steps = [r["control_step"] for r in rows]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for condition, color, label in [
        ("agentview", "#8799ac", "Agentview"),
        ("wrist", "#b69b86", "Wrist"),
        ("both", "#1462ad", "Both cameras"),
    ]:
        distances = [
            r["conditions"][condition]["proximity"].get("symmetric_mean_m", np.nan) * 1000
            for r in rows
        ]
        axs[0].plot(steps, distances, "o-", color=color, label=label)
        errors = [
            r["evaluator"]["conditions"][condition]["door_centroid_consistency_error_mm"]
            for r in rows
        ]
        axs[2].plot(
            steps, [v if v is not None else np.nan for v in errors], "o-", color=color, label=label
        )
    axs[0].set(
        ylabel="Region proximity (mm)",
        title="Selected visible surfaces; this is not an automatic closing label",
    )
    axs[0].legend()
    if not any(
        r["conditions"][name]["proximity"]["available"] for r in rows for name in CONDITIONS
    ):
        axs[0].text(
            0.5,
            0.5,
            "No complete door + opening region measurement",
            ha="center",
            transform=axs[0].transAxes,
        )
    axs[1].plot(
        list(native), [abs(np.degrees(j["position"])) for j in native.values()], color="#a15617"
    )
    axs[1].set(
        ylabel="Native opening (degrees)",
        title="Evaluator only — withheld from both model sessions and tracking",
    )
    axs[2].set(
        ylabel="Door centroid error (mm)",
        xlabel="Control step in the earlier recording (20 steps = 1 s)",
    )
    axs[2].legend()
    for ax in axs:
        ax.grid(alpha=0.2)
    fig.savefig(folder / "tracking/curves.png", dpi=120)
    plt.close(fig)
    evaluation = {
        "native_state_used_only_after_frozen_judgments": True,
        "summary": summary,
        "intervals": [
            {
                k: r[k]
                for k in [
                    "interval_id",
                    "start_control",
                    "end_control",
                    "native_opening_change_deg",
                    "native_label",
                    "judgment",
                    "grade",
                ]
            }
            for r in intervals
        ],
    }
    (folder / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    return {
        "state": state,
        "folder": folder.name,
        "judgment_folder": str(judge.relative_to(study)),
        "selection": selection,
        "region_identity_assessment": judgments["region_identity_assessment"],
        "samples": rows,
        "intervals": intervals,
        "summary": summary,
        "tracking": data["tracking"],
        "selection_timing": selection_timing(folder),
        "judgment_timing": selection_timing(judge),
        "measurement_seconds": data["seconds"],
        "capture_count": len(rows),
        "depth_captures": sum(r["depth_retained"] for r in rows),
        "plot": str((folder / "tracking/curves.png").relative_to(study)),
        "video": "../"
        + str((run / "attempt01/initial/rollout.mp4").relative_to(ROOT / "results")),
        "original_task_result": "The earlier robot episode failed at 600 controls. This test only reads its recordings; no new task execution or planning occurred.",
    }


def main() -> None:
    """Generate a small static dashboard without periodic refresh or repeated media writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    if not (study / "judgment-complete.json").exists():
        raise RuntimeError("Freeze both judgments before native evaluation")
    cases = [case_report(study, state) for state in [17, 49]]
    audit = resources(
        study,
        [
            ("selection", study),
            ("tracking", study / "tracking-resources"),
            ("judgment", study / "judgment-resources"),
        ],
    )
    data = {
        "cases": cases,
        "resources": audit,
        "protocol": read(study / "protocol.json"),
        "summary": "Both cameras, model-selected door/opening regions, no motion-circle input and no supplied closing direction. Region identity, tracking availability and Luna’s interpretation are reported separately.",
    }
    (study / "summary.json").write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"
    )
    (study / "resource-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    shutil.copyfile(ROOT / "templates/region_motion.html", study / "watch.html")
    shutil.copyfile(ROOT / "templates/luna_documents.js", study / "document-viewer.js")
    print(
        json.dumps(
            {
                c["state"]: {
                    "tracking": c["summary"],
                    "judgments": [r["grade"] for r in c["intervals"]],
                }
                for c in cases
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
