"""Report whole-part tracking, model-selected references and held-out native evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    """Read a completed local record."""
    return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
    """Write finite, inspectable report data."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def selection_timing(folder: Path) -> dict:
    """Separate tool wall time from the surrounding actor session without calling it pure inference."""
    elapsed=read(folder/'agent-exit.json')['wall_seconds']
    tools: Counter=Counter()
    for line in (folder/'tool-events.jsonl').read_text().splitlines():
        event=json.loads(line)
        tools[event['request']['operation']]+=event['wall_seconds']
    return {'actor_session_including_tools_seconds':elapsed,
            'tool_wall_seconds_by_operation':dict(tools),
            'actor_and_orchestration_outside_tool_service_seconds':elapsed-sum(tools.values())}


def project(point: list | None, calibration: dict) -> list | None:
    """Project an estimated world reference for display, not measurement."""
    if point is None:
        return None
    matrix = np.linalg.inv(np.asarray(calibration["camera_to_world"]))
    local = matrix[:3, :3] @ point + matrix[:3, 3]
    if local[2] <= 0:
        return None
    uv = np.asarray(calibration["intrinsic"]) @ local
    return np.round(uv[:2] / uv[2], 2).tolist()


def condition(study: Path, folder: Path, state: int, native: dict[int, dict], label: str) -> dict:
    """Evaluate only after model selections and RGB-D fits have been frozen."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gap = read(folder / "whole-part/gap.json")
    refs = gap["references"]
    samples = []
    evaluation = []
    registry = read(folder / "agent-workspace/part_registry.json")
    for role, color in [("moving", [20, 98, 173]), ("fixed_closed", [176, 91, 13])]:
        if refs[role] is None:
            continue
        masks = []
        for identifier in refs[role]["part_ids"]:
            with np.load(
                folder / "agent-workspace" / registry[identifier]["mask_file"], allow_pickle=False
            ) as arrays:
                masks.append(arrays["usable_mask"])
        mask = np.logical_or.reduce(masks)
        rgba = np.zeros((*mask.shape, 4), np.uint8)
        rgba[mask, :3] = color
        rgba[mask, 3] = 85
        Image.fromarray(rgba).save(folder / "whole-part" / (role + "-mask.png"))
    if not gap["samples"]:
        orbit = read(folder / "agent-workspace/observed-orbit.json")
        axis = np.asarray(next(iter(native.values()))["axis_world"])
        axis_error = float(
            np.degrees(np.arccos(np.clip(abs(np.dot(axis, orbit["axis_world_unsigned"])), -1, 1)))
        )
        audit = {
            "native_labels_withheld_from_model": True,
            "axis_error_degrees": axis_error,
            "measured_rotation_degrees": orbit["measured_rotation_degrees"],
            "note": "This orbit is an experimental geometric hypothesis. A small equation residual does not establish an accurate axis, especially with very small observed rotations. Its candidate rejection is not proof of visual impossibility.",
        }
        return {
            "label": label,
            "folder": str(folder.relative_to(study)),
            "references": refs,
            "samples": [],
            "status": "unresolved_references",
            "orbit_audit": audit,
            "summary": {
                "supported_captures": 0,
                "total_captures": 15,
                "depth_captures": sum(
                    (p / "sensors.npz").exists()
                    for p in (folder / "agent-workspace/captures").glob("obs*")
                ),
                "maximum_carried_point_error_mm": None,
                "maximum_direct_point_error_mm": None,
                "reference_height_difference_mm": None,
                "trend_disagreements": [],
                "fit_reasons": {},
            },
            "tracking": [],
            "measurement_seconds": 0,
            "selection_seconds": read(folder / "agent-exit.json")["wall_seconds"],
            "selection_timing":selection_timing(folder),
            "plot": None,
            "gap_file": str((folder / "whole-part/gap.json").relative_to(study)),
        }
    reference = next(
        r for r in gap["samples"] if r["observation_id"] == refs["moving"]["observation_id"]
    )
    native_reference = native[reference["control_step"]]
    anchor = np.asarray(reference["moving"]["reference_world_m"])
    axis = np.asarray(native_reference["axis_world"])
    origin = np.asarray(native_reference["anchor_world_m"])
    for source in gap["samples"]:
        row = {
            k: source[k]
            for k in ["observation_id", "control_step", "estimated_gap_m", "supported"]
        }
        calibration = read(
            folder
            / "agent-workspace/captures"
            / f"obs{row['observation_id']:03d}"
            / "calibration.json"
        )
        true = native[row["control_step"]]
        expected = (
            Rotation.from_rotvec(axis * (true["position"] - native_reference["position"])).apply(
                anchor - origin
            )
            + origin
        )
        predicted = source["moving"]["estimated_reference_world_m"]
        direct = source["moving"]["direct_reference_sample"]
        evaluation.append(
            {
                "control_step": row["control_step"],
                "native_open_deg": abs(float(np.degrees(true["position"]))),
                "carried_reference_error_m": float(
                    np.linalg.norm(np.asarray(predicted) - expected)
                )
                if predicted is not None
                else None,
                "direct_reference_error_m": float(
                    np.linalg.norm(np.asarray(direct["raw_world_m"]) - expected)
                )
                if direct.get("supported")
                else None,
            }
        )
        row["evaluator"] = evaluation[-1]
        for role in ["moving", "fixed_closed"]:
            item = source[role]
            row[role] = {
                k: item[k]
                for k in [
                    "fit",
                    "camera",
                    "query_count",
                    "visible_queries",
                    "depth_supported_queries",
                    "inlier_indices",
                    "reference_sample",
                    "direct_reference_sample",
                    "estimated_reference_world_m",
                ]
            }
            row[role]["query_pixels_xy"] = np.round(item["query_pixels_xy"], 1).tolist()
            row[role]["estimated_pixel_xy"] = project(
                item["estimated_reference_world_m"], calibration[item["camera"]]
            )
        samples.append(row)
    write(
        folder / "whole-part/evaluator-validation.json",
        {"withheld_from_models_and_estimator": True, "rows": evaluation},
    )
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    steps = [r["control_step"] for r in samples]
    values = [r["estimated_gap_m"] * 1000 if r["supported"] else np.nan for r in samples]
    axs[0].plot(
        steps, values, "o-", color="#1462ad", label="References carried by whole-part poses"
    )
    axs[0].set(ylabel="Estimated gap (mm)", title=f"State {state} · {label}")
    axs[0].legend()
    axs[1].plot(
        list(native), [abs(np.degrees(r["position"])) for r in native.values()], color="#a15617"
    )
    axs[1].set(
        ylabel="Native opening (degrees)",
        title="Evaluator only — withheld from selection and estimation",
    )
    for key, color, title in [
        ("carried_reference_error_m", "#1462ad", "Whole-part carried point"),
        ("direct_reference_error_m", "#ad3a30", "Single point on the same new reference"),
    ]:
        axs[2].plot(
            steps,
            [r[key] * 1000 if r[key] is not None else np.nan for r in evaluation],
            "o-",
            color=color,
            label=title,
        )
    axs[2].set(
        ylabel="Reference error (mm)",
        xlabel="Control step in the earlier recording (20 steps = 1 s)",
    )
    axs[2].legend()
    for ax in axs:
        ax.grid(alpha=0.2)
    fig.savefig(folder / "whole-part/curves.png", dpi=120)
    plt.close(fig)
    supported = [r for r in samples if r["supported"]]
    disagreements = []
    for first, last in zip(supported, supported[1:]):
        dg = last["estimated_gap_m"] - first["estimated_gap_m"]
        da = last["evaluator"]["native_open_deg"] - first["evaluator"]["native_open_deg"]
        if dg * da < 0:
            disagreements.append(
                {
                    "start_control": first["control_step"],
                    "end_control": last["control_step"],
                    "gap_change_mm": dg * 1000,
                    "native_opening_change_deg": da,
                }
            )
    errors = [
        r["carried_reference_error_m"]
        for r in evaluation
        if r["carried_reference_error_m"] is not None
    ]
    direct_errors = [
        r["direct_reference_error_m"]
        for r in evaluation
        if r["direct_reference_error_m"] is not None
    ]
    part_samples = gap["samples"][0]
    summary = {
        "supported_captures": len(supported),
        "total_captures": len(samples),
        "depth_captures": sum(
            r["moving"]["fit"]["reason"] != "recorded_depth_not_retained" for r in samples
        ),
        "maximum_carried_point_error_mm": max(errors) * 1000 if errors else None,
        "maximum_direct_point_error_mm": max(direct_errors) * 1000 if direct_errors else None,
        "reference_height_difference_mm": abs(
            part_samples["moving"]["reference_world_m"][2]
            - part_samples["fixed_closed"]["reference_world_m"][2]
        )
        * 1000,
        "trend_disagreements": disagreements,
        "fit_reasons": {
            role: dict(Counter(r[role]["fit"]["reason"] for r in samples))
            for role in ["moving", "fixed_closed"]
        },
    }
    return {
        "label": label,
        "folder": str(folder.relative_to(study)),
        "references": refs,
        "samples": samples,
        "summary": summary,
        "tracking": gap["tracking"],
        "measurement_seconds": gap["seconds"],
        "selection_seconds": read(folder / "agent-exit.json")["wall_seconds"],
        "selection_timing":selection_timing(folder),
        "plot": str((folder / "whole-part/curves.png").relative_to(study)),
        "gap_file": str((folder / "whole-part/gap.json").relative_to(study)),
    }


def resources(study: Path) -> list[dict]:
    """Publish compact resource samples, including the explicitly failed first reader pass."""
    result = []
    for name, folder in [
        ("selection", study),
        ("first_tracker_reader", study / "tracking-resources"),
        ("distributed_tracking", study / "distributed-tracking-resources"),
        ("reference_refinement", study / "refinement-resources"),
        ("refined_tracking", study / "refined-tracking-resources"),
    ]:
        if not (folder / "resource-monitor.json").exists():
            continue
        location = read(folder / "resource-monitor.json")
        rows = [json.loads(line) for line in Path(location["samples"]).read_text().splitlines()]
        probes = {
            r["login_probe"]["checked_at"]: r["login_probe"] for r in rows if "login_probe" in r
        }
        gpu = read(folder / "gpu-summary.json")
        result.append(
            {
                "phase": name,
                "peak_job_memory_gib": max(r["job_memory_peak_gib"] for r in rows),
                "peak_anon_and_shmem_gib": max(r["job_anon_and_shmem_gib"] for r in rows),
                "minimum_host_available_gib": min(r["host_available_memory_gib"] for r in rows),
                "peak_gpu_gib": gpu["peak_gpu_memory_mib"] / 1024,
                "oom_kills": max(r["job_memory_oom_kills"] for r in rows),
                "maximum_io_full_avg10": max(r["io_full_avg10"] for r in rows),
                "ssh_checks": len(probes),
                "ssh_successes": sum(p["healthy"] for p in probes.values()),
                "status": read(folder / "supervisor-state.json")["status"],
            }
        )
        keys = [
            "time",
            "job_memory_gib",
            "job_anon_and_shmem_gib",
            "host_available_memory_gib",
            "io_full_avg10",
            "memory_full_avg10",
            "job_memory_oom_kills",
            "login_probe",
        ]
        (study / (name + "-telemetry.jsonl")).write_text(
            "".join(json.dumps({k: r.get(k) for k in keys}) + "\n" for r in rows)
        )
    return result


def main() -> None:
    """Render a small static page; it never refreshes or rewrites during inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    cases = []
    for state in [17, 49]:
        run = (
            ROOT
            / f"results/luna-molmo2-integration-v3-20261103/trials/baseline-r1-close_microwave-state{state:03d}-600"
        )
        native_export = read(run / "attempt01/initial/evaluation_contact_diagnostics.json")
        if native_export["excluded_from_policy_input"] is not True:
            raise ValueError("Evaluator export is not isolated")
        native = {
            r["step"]: next(
                j
                for j in r["evaluation_only_articulation_state"]
                if j["joint_name"] == "microwave_1_microjoint"
            )
            for r in native_export["steps"]
        }
        conditions = {}
        for key, folder, label in [
            (
                "initial",
                study / f"state{state:03d}",
                "Whole-part tracking / first reference selection",
            ),
            (
                "refined",
                study / f"orbit-refinement/state{state:03d}",
                "Whole-part tracking / motion-informed reference revision",
            ),
        ]:
            if (folder / "whole-part/gap.json").exists():
                conditions[key] = condition(study, folder, state, native, label)
        cases.append(
            {
                "state": state,
                "conditions": conditions,
                "video": "../"
                + str((run / "attempt01/initial/rollout.mp4").relative_to(ROOT / "results")),
                "original_task_result": "Failed at the unchanged 600-control-step limit. This analysis executes no robot motion.",
                "original_controls": 600,
                "native_initial_deg": abs(np.degrees(next(iter(native.values()))["position"])),
                "native_final_deg": abs(np.degrees(native[600]["position"])),
            }
        )
    data = {
        "cases": cases,
        "resources": resources(study),
        "protocol": read(study / "protocol.json"),
        "summary": "Whole-door CoTracker tracking and model-selected closure references, tested on the original recorded cases. Unsupported fits remain missing; no closing direction or simulator geometry enters selection or tracking.",
    }
    (study / "summary.json").write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
    write(study / "resource-audit.json", data["resources"])
    shutil.copyfile(ROOT / "templates/whole_part_gap.html", study / "watch.html")
    shutil.copyfile(ROOT / "templates/luna_documents.js", study / "document-viewer.js")
    print(
        json.dumps(
            {c["state"]: {k: v["summary"] for k, v in c["conditions"].items()} for c in cases},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
