"""A completed actor does not have to precompute the dashboard's timing artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_visual_memory import case_summary


def test_completed_case_generates_detailed_timing_before_linking_it(tmp_path: Path) -> None:
    """Build the first report from raw events, with no pre-existing profile JSON."""
    run = tmp_path / "trials/example"
    run.mkdir(parents=True)
    (run / "agent-workspace").mkdir()
    (run / "agent-workspace/findings.txt").write_text("No task success.")
    values = {
        "decision-progress.json": {
            "decisions": [
                {"operation": "help", "help_operation": "push"},
                {"operation": "push", "help_operation": None},
                {"operation": "help", "help_operation": "push"},
                {"operation": "help", "help_operation": "gripper"},
            ]
        },
        "memory-final.json": {
            "mode": "recent",
            "retained_executed_transitions": 0,
            "total_executed_transitions": 0,
            "slot_limit": 8,
        },
        "source-integrity.json": {"numerical_backend_unchanged": True},
        "agent-exit.json": {"wall_seconds": 12.0},
        "attempt01/initial/actions.json": [],
        "attempt01/opened/metadata.json": {"control_steps": 60},
    }
    for name, value in values.items():
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text(json.dumps(value))
    event = {
        "request": {"operation": "observe", "arguments": {}},
        "wall_seconds": 0.25,
        "response": {"ok": True, "result": {}},
    }
    (run / "tool-events.jsonl").write_text(json.dumps(event) + "\n")
    assert not (run / "timing-breakdown.json").exists()
    result = case_summary(tmp_path, {"run": "example", "attempt": {}, "actor_wall_seconds": 12.0})
    assert result["timing_details"]["service_wall_seconds"] == 0.25
    assert result["timing_details"]["outside_service_seconds"] == 11.75
    assert (run / "timing-breakdown.json").is_file()
    assert not result["has_video"]
    assert result["documentation_usage"]["help_requests_by_tool"] == {"push": 2, "gripper": 1}
    assert result["documentation_usage"]["repeated_help_requests"] == 1
