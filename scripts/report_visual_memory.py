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


def executed_behavior(events: list[dict], actions: list[dict], initial_control: int) -> dict:
    """Separate requested tool names from issued controls, motor commands and observed arrival."""
    rows = []
    gaps = []
    automatic_closures = 0
    close_requests = push_requests = push_without_motion = push_reached_stroke = previews = 0
    for event in events:
        operation = event['request']['operation']
        arguments = event['request']['arguments']
        value = event['response'].get('result', {})
        stages = value.get('stages', [])
        if not stages and 'execution' in value:
            stages = [{'stage': operation, 'execution': value['execution'],
                       'control_steps': value.get('observation', {}).get('control_steps')}]
        issued = False
        reached_stroke = False
        for stage in stages:
            execution = stage.get('execution', {})
            count = execution.get('issued_commands', 0)
            if not count:
                continue
            issued = True
            name = stage['stage']
            automatic_closures += name == 'gripper-close' and bool(value.get('grasp_execution', {}).get('automatic_closure'))
            reached_stroke |= name.startswith('push-stroke')
            end = stage.get('control_steps')
            start = end - count if end is not None else None
            commands = sorted({row['control'][-1] for row in actions
                               if start is not None and start < row['step'] <= end})
            goal = execution.get('arrival', {}).get('goal', {})
            rows.append({'event': event['sequence'], 'operation': operation, 'stage': name,
                         'before_control': start, 'after_control': end, 'issued_controls': count,
                         'gripper_commands': commands, 'goal_reached': goal.get('reached'),
                         'goal_error_mm': 1000 * goal['position_error_m'] if 'position_error_m' in goal else None,
                         'stop_reason': execution.get('termination_reason')})
        if operation == 'gripper' and arguments.get('close', True) and issued:
            close_requests += 1
        if operation == 'push':
            if arguments.get('preview', False):
                previews += 1
            else:
                push_requests += 1
                push_without_motion += not issued
                push_reached_stroke += reached_stroke
        gap = value.get('grasp_evidence', {}).get('pad_gap_mm')
        step = value.get('observation', {}).get('control_steps')
        if gap is not None and step is not None:
            gaps.append({'control': step, 'gap_mm': gap})
    return {
        'scope': 'Actual issued motion and gripper commands; none of these counts proves object manipulation.',
        'explicit_close_requests_executed': close_requests,
        'automatic_grasp_closures_executed': automatic_closures,
        'controls_commanding_close': sum(row['step'] > initial_control and row['control'][-1] > 0
                                        for row in actions),
        'push_requests_for_motion': push_requests, 'push_previews': previews,
        'push_requests_without_robot_motion': push_without_motion,
        'push_requests_that_reached_a_stroke': push_reached_stroke,
        'direct_hand_stroke_stages': sum(row['stage'].startswith('push-stroke')
                                        and row['operation'] in {'push_stroke', 'continue_push'} for row in rows),
        'jaw_gap_samples': gaps, 'motion_stages': rows,
    }


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
        if reason and reason not in {"preview_only", "stroke_executed", "native_task_success", "closure_completed",
                                     "simulator_success", "object_transfer_released"}:
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
    presentations = [read_json(path) for path in (run / 'decisions').glob('*/presentation.json')]
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
        feedback_presentation={
            'compact_transitions': entry.get('compact_transitions', False),
            'recorded_contexts': len(presentations),
            'original_payload_bytes': sum(row['original_payload_bytes'] for row in presentations),
            'presented_payload_bytes': sum(row['presented_payload_bytes'] for row in presentations),
            'all_image_manifests_unchanged': all(row['image_manifest_unchanged'] for row in presentations),
        },
    )
    result["timing_details"] = cached_profile_run(run, events, entry["actor_wall_seconds"])
    result['executed_behavior'] = executed_behavior(
        events, read_json(run / 'attempt01/initial/actions.json'),
        read_json(run / 'attempt01/opened/metadata.json')['control_steps'])
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
    phases += [("current", study)]
    phases.extend(("operator-" + path.parent.name, path.parent)
                  for path in sorted((study / "diagnostics").glob("*/resource-monitor.json")))
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
            "related_reports": read_json(study / "related-reports.json")
            if (study / "related-reports.json").exists() else [],
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
