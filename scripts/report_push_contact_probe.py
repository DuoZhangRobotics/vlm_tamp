"""Report intended contact, measured push phases and task success separately."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

from vlm_tamp.coupled_storage import read_json
from report_part_goal_gap import resources
from report_contact_skills import brief_evidence

ROOT = Path(__file__).resolve().parents[1]


def optional(path: Path, default: object = None) -> object:
    """Read only an existing artifact; missing evidence remains explicit."""
    return read_json(path) if path.exists() or path.with_suffix(".json.gz").exists() else default


def control_at(work: Path, observation: int | None) -> int | None:
    """Resolve a saved public capture to its actual control timestamp."""
    if observation is None:
        return None
    state = optional(work / f"robot_data/attempt01/obs{observation:03d}/robot.json")
    return state["control_steps"] if state is not None else None


def case_report(study: Path, entry: dict) -> dict:
    """Read evaluator-only object motion after the actor has frozen its outcome."""
    run = study / "trials" / entry["run"]
    work = run / "agent-workspace"
    events = [json.loads(line) for line in (run / "tool-events.jsonl").read_text().splitlines()]
    native = optional(run / "attempt01/initial/evaluation_contact_diagnostics.json")
    angles = {}
    if native is not None:
        if native.get("excluded_from_policy_input") is not True:
            raise ValueError("Expected evaluator-only contact export")
        angles = {
            r["step"]: abs(
                math.degrees(
                    next(
                        j["position"]
                        for j in r["evaluation_only_articulation_state"]
                        if j["joint_name"] == "microwave_1_microjoint"
                    )
                )
            )
            for r in native["steps"]
        }
    calls = []
    pointing = []
    tool_time = Counter()
    search_time = Counter()
    for event in events:
        request = event["request"]
        operation = request["operation"]
        args = request["arguments"]
        tool_time[operation] += event["wall_seconds"]
        value = event["response"].get("result", {})
        if operation == "point":
            pointing.append(
                {
                    "event": event["sequence"],
                    "request": args,
                    "raw_text": value.get("raw_text"),
                    "points": value.get("points", []),
                    "images": value.get("images", []),
                    "seconds": event["wall_seconds"],
                }
            )
        if operation not in ["push", "push_stroke", "continue_push"]:
            continue
        phases = []
        for phase in value.get("phase_visual_comparisons", []):
            before = control_at(work, phase.get("before_observation_id"))
            after = control_at(work, phase.get("after_observation_id"))
            change = (
                angles[after] - angles[before] if before in angles and after in angles else None
            )
            execution = next(
                (
                    s.get("execution", {})
                    for s in value.get("stages", [])
                    if s["stage"] == phase["stage"]
                ),
                {},
            )
            phases.append(
                {
                    **phase,
                    "start_control": before,
                    "end_control": after,
                    "native_opening_change_deg": change,
                    "native_motion": (
                        "closing" if change < 0 else "opening" if change > 0 else "unchanged"
                    )
                    if change is not None
                    else "unavailable",
                    "execution": brief_evidence(execution),
                }
            )
        whole = value.get("visual_comparison", {})
        before = control_at(work, whole.get("before_observation_id"))
        after = control_at(work, whole.get("after_observation_id"))
        change = angles[after] - angles[before] if before in angles and after in angles else None
        for key, seconds in value.get("timing_seconds", {}).items():
            search_time[key] += seconds
        calls.append(
            {
                "event": event["sequence"],
                "request": args,
                "ok": event["response"]["ok"],
                "error": event["response"].get("error"),
                "preview": bool(args.get("preview", False)),
                "status": value.get("status"),
                "stop_reason": value.get("stop_reason"),
                "robot_motion_executed": value.get("robot_motion_executed", False),
                "stroke_executed": any(p["stage"].startswith("push-stroke") for p in phases),
                "phases": phases,
                "whole_call_native_opening_change_deg": change,
                "contact": value.get("contact"),
                "selection_geometry": value.get("selection_geometry"),
                "contact_orientation": value.get("contact_orientation"),
                "approach_arrival": value.get("approach_arrival"),
                "motion_arrival": value.get("motion_arrival"),
                "images": value.get("images", []),
                "timing_seconds": value.get("timing_seconds", {}),
                "search": brief_evidence(value.get("search")),
                "diagnostic": brief_evidence(value.get("diagnostic")),
                "planning_stages": brief_evidence(value.get("planning_stages", [])),
                "motion_settings": value.get("motion_settings"),
                "remaining_skill_seconds": value.get("remaining_skill_seconds"),
                "recorded_object_progress_is_execution_gate":value.get('object_progress_is_execution_gate'),
                "seconds": event["wall_seconds"],
            }
        )
    executed = [r for r in calls if not r["preview"]]
    actor = read_json(run / "agent-exit.json")
    initial_step=control_at(work,1)
    final_step=entry['attempt']['control_steps']
    return {
        "state": entry["state"],
        "run": entry["run"],
        "folder": str(run.relative_to(study)),
        "attempt": entry["attempt"],
        "initial_opening_deg":angles.get(initial_step),
        "final_opening_deg":angles.get(final_step),
        "other_motion_requests":[{'event':e['sequence'],'operation':e['request']['operation'],
                                  'arguments':e['request']['arguments']}
                                 for e in events if e['request']['operation'] in ['execute','execute_sequence','gripper','hook_pull']],
        "intent": optional(work / "intent.json"),
        "actor_outcome": optional(work / "outcome.json"),
        "findings": (work / "findings.txt").read_text()
        if (work / "findings.txt").exists()
        else None,
        "nonpreview_push_requests": len(executed),
        "one_push_request_scope_followed": len(executed) <= 1,
        "calls": calls,
        "pointing": pointing,
        "native_opening_timeline": [{"control": k, "opening_deg": v} for k, v in angles.items()],
        "timing": {
            "actor_session_including_tools_seconds": actor["wall_seconds"],
            "tool_wall_seconds_by_operation": dict(tool_time),
            "actor_and_orchestration_outside_tools_seconds": actor["wall_seconds"]
            - sum(tool_time.values()),
            "nested_push_timing_seconds": dict(search_time),
        },
        "video": str((run / "attempt01/initial/rollout.mp4").relative_to(study)),
        "native_evidence_supplied_to_actor": False,
    }


def render(study: Path, *, with_resources: bool = False) -> None:
    """Publish only at phase boundaries; never reload the user's page automatically."""
    study = Path(study)
    manifest = optional(study / "probe-manifest.json")
    if manifest is None:
        return
    cases = [case_report(study, entry) for entry in manifest["completed_runs"]]
    audit = resources(study, [("live_push_probe", study)]) if with_resources else []
    data = {
        "cases": cases,
        "active_run": manifest["active_run"],
        "complete": manifest["complete"],
        "protocol": {
            k: v for k, v in manifest.items() if k not in ["source_sha256", "completed_runs"]
        },
        "resources": audit,
        "operator_diagnosis":optional(study/'diagnosis.json',{}),
        "summary": "One live push per case: Luna chooses rotation and contact; Molmo2 proposes the point; measured face geometry informs the preview. Approach and stroke effects are scored separately after the actor finishes.",
    }
    (study / "summary.json").write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"
    )
    (study / "resource-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    for source, target in [
        (ROOT / "templates/push_contact_probe.html", study / "watch.html"),
        (ROOT / "templates/luna_documents.js", study / "document-viewer.js"),
    ]:
        if not target.exists() or source.read_bytes() != target.read_bytes():
            shutil.copyfile(source, target)


def main() -> None:
    """Optionally append the completed supervisor's resource audit after workers exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--resources", action="store_true")
    args = parser.parse_args()
    render(args.study.resolve(), with_resources=args.resources)


if __name__ == "__main__":
    main()
