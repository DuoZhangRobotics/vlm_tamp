"""Small manual-refresh dashboard for direct actions, visual inputs and anchored memory."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from report_part_goal_gap import resources
from report_contact_skills import brief_evidence
from report_restriction_fixes import cached_profile_run
from vlm_tamp.coupled_storage import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def case_summary(study: Path, entry: dict) -> dict:
    """Report native completion separately from tool successes and learner notes."""
    run = study / "trials" / entry["run"]
    completed = "attempt" in entry
    result = {
        **entry,
        "folder": str(run.relative_to(study)),
        "progress_file": str((run / "decision-progress.json").relative_to(study)),
        "completed": completed,
        "video": str((run / "attempt01/initial/rollout.mp4").relative_to(study)),
        "has_video": (run / "attempt01/initial/rollout.mp4").is_file(),
        "interruption": read_json(run / "operator-interruption.json")
        if (run / "operator-interruption.json").exists()
        else None,
    }
    if not completed:
        return result
    progress = read_json(run / "decision-progress.json")
    help_requests = Counter(
        row["help_operation"] for row in progress["decisions"] if row.get("help_operation")
    )
    help_targets_recorded = all(
        row.get("help_operation") for row in progress["decisions"] if row["operation"] == "help"
    )
    result["documentation_usage"] = {
        "retained_between_decisions": entry.get("retain_tool_documentation", False),
        "help_requests_by_tool": dict(help_requests),
        "repeated_help_requests": sum(max(0, count - 1) for count in help_requests.values())
        if help_targets_recorded
        else None,
    }
    events = [json.loads(line) for line in (run / "tool-events.jsonl").read_text().splitlines()]
    tool_time = Counter()
    selections = Counter()
    issues = []
    for event in events:
        op = event["request"]["operation"]
        tool_time[op] += event["wall_seconds"]
        selections[op] += 1
        response = event["response"]
        value = response.get("result", {})
        reason = response.get("error") if not response.get("ok") else value.get("stop_reason")
        if reason and reason not in {"preview_only", "stroke_executed", "native_task_success"}:
            issues.append(
                {
                    "event": event["sequence"],
                    "operation": op,
                    "reason": reason,
                    "diagnostic": brief_evidence(
                        response.get("planning_diagnostic", value.get("diagnostic"))
                    ),
                }
            )
    model_time = sum(
        read_json(p)["model_and_client_seconds"] for p in (run / "decisions").glob("*/timing.json")
    )
    memory = read_json(run / "memory-final.json")
    result.update(
        decisions=progress["decisions"],
        issues=issues,
        timing={
            "actor_session_seconds": entry["actor_wall_seconds"],
            "model_and_client_seconds": model_time,
            "robot_tools_seconds": dict(tool_time),
        },
        tool_selections=dict(selections),
        luna_selections=dict(Counter(row["operation"] for row in progress["decisions"])),
        memory={
            "mode": memory["mode"],
            "retained": memory["retained_executed_transitions"],
            "executed": memory["total_executed_transitions"],
            "limit": memory["slot_limit"],
        },
        learner_findings=(run / "agent-workspace/findings.txt").read_text(),
        numerical_backend_unchanged=read_json(run / "source-integrity.json")["numerical_backend_unchanged"],
    )
    result["timing_details"] = cached_profile_run(run, events, entry["actor_wall_seconds"])
    usages = [
        event["usage"]
        for file in sorted((run / "decisions").glob("*/events.jsonl"))
        for event in map(json.loads, file.read_text().splitlines())
        if event.get("type") == "turn.completed"
    ]
    result["model_usage"] = {
        "scope": "CLI decision totals; these counters are not individual prompt context lengths.",
        "calls_with_usage": len(usages),
        "input_tokens": sum(row["input_tokens"] for row in usages),
        "maximum_input_tokens": max((row["input_tokens"] for row in usages), default=0),
        "output_tokens": sum(row["output_tokens"] for row in usages),
    }
    write_json(run / "case-summary.json", result)
    return result


def render(study: Path, *, with_resources: bool = False) -> None:
    """Publish at case boundaries; live per-decision files are already small."""
    study = Path(study)
    manifest = read_json(study / "visual-memory-manifest.json")
    completed = {r["run"]: r for r in manifest["completed_runs"]}
    cases = [case_summary(study, completed.get(entry["run"], entry)) for entry in manifest["runs"]]
    input_audit = read_json(study / "study-audit.json") if (study / "study-audit.json").exists() else None
    if input_audit:
        for case in cases:
            case["input_audit"] = input_audit["cases"].get(case["run"])
    phases = [(folder.name, folder) for folder in sorted(study.glob("resource-session-*"))]
    probe = study / "diagnostics/bowl-release-counterfactual"
    phases += [("current", study)]
    if (probe / "resource-monitor.json").exists():
        phases.append(("operator-release-diagnostic", probe))
    audit = resources(study, phases) if with_resources else []
    write_json(
        study / "summary.json",
        {
            "cases": cases,
            "complete": manifest["complete"],
            "active_run": manifest["active_run"],
            "config": manifest["config"],
            "resources": audit,
            "interruption": manifest.get("interruption"),
            "replacement_study": manifest.get("replacement_study"),
            "input_audit": input_audit,
            "scope": manifest["scope"],
            "recorded_decision_comparison": read_json(study / "recorded-decisions/summary.json")
            if (study / "recorded-decisions/summary.json").exists()
            else None,
            "operator_diagnosis": read_json(study / "diagnosis.json")
            if (study / "diagnosis.json").exists()
            else {},
        },
    )
    if audit:
        write_json(study / "resource-audit.json", audit)
    for source, name in [
        (ROOT / "templates/visual_memory.html", "watch.html"),
        (ROOT / "templates/luna_documents.js", "document-viewer.js"),
    ]:
        target = study / name
        if not target.exists() or target.read_bytes() != source.read_bytes():
            shutil.copyfile(source, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--resources", action="store_true")
    args = parser.parse_args()
    render(args.study, with_resources=args.resources)
