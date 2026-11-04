"""Run one model-chosen live push per case with a persistent Molmo2 pointer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_restriction_study import source_hashes
from run_varied_contact_practice import actor_command, audit_memory_input
from select_recorded_goal_references import model_worker
from report_push_contact_probe import render

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, value: object) -> None:
    """Publish a small phase-boundary record."""
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    """Keep diagnostic early finish distinct from a full-task evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    study.mkdir(exist_ok=True)
    (study / "trials").mkdir()
    from supervise_luna_dashboard_study import wait_for_capacity

    wait_for_capacity(study, "live-push-model-load")
    frozen = source_hashes()
    manifest = {
        "scope": "One live push diagnostic per previously used state; not a full-task success-rate comparison",
        "states": [17, 49],
        "model": "gpt-6-luna",
        "reasoning": "high",
        "molmo2": "allenai/Molmo2-8B",
        "control_steps": 600,
        "agent_seconds": 1800,
        "one_push_requested_per_actor": True,
        "voluntary_finish_after_diagnostic": True,
        "prior_memory_supplied": False,
        "closing_direction_supplied": False,
        "native_geometry_supplied": False,
        "motion_circle_supplied": False,
        "source_sha256": frozen,
        "gpu": os.environ["VLM_TAMP_GPU"],
        "started_at": time.time(),
        "active_run": None,
        "completed_runs": [],
        "complete": False,
    }
    write(study / "probe-manifest.json", manifest)
    render(study)
    with model_worker(study) as endpoint:
        for state in [17, 49]:
            if source_hashes() != frozen:
                raise RuntimeError("Frozen runtime sources changed")
            name = f"push-close_microwave-state{state:03d}"
            out = study / "trials" / name
            wait_for_capacity(study, name)
            command = actor_command(
                out,
                [
                    "--task",
                    "close_microwave",
                    "--state",
                    str(state),
                    "--attempts",
                    "1",
                    "--agent-seconds",
                    "1800",
                    "--evaluation-case",
                    "--clean-start",
                    "--pointing-socket",
                    str(endpoint),
                    "--prompt-file",
                    str(ROOT / "prompts/push_contact_probe.txt"),
                ],
                control_steps=600,
                episode_seconds=1800,
            )
            command.remove("--persistent-episode")
            manifest["active_run"] = name
            write(study / "probe-manifest.json", manifest)
            render(study)
            with (study / (name + ".log")).open("x") as log:
                process = subprocess.Popen(
                    command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT
                )
                write(
                    study / "active-process.json",
                    {"pid": process.pid, "run": name, "status": "running", "command": command},
                )
                try:
                    while process.poll() is None:
                        time.sleep(2)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=60)
                write(
                    study / "active-process.json",
                    {
                        "pid": process.pid,
                        "run": name,
                        "status": "completed",
                        "returncode": process.returncode,
                    },
                )
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
            if source_hashes() != frozen:
                raise RuntimeError("Frozen runtime sources changed during the probe")
            audit_memory_input(out, None)
            evaluation = json.loads((out / "evaluation-private.json").read_text())
            if len(evaluation["attempts"]) != 1:
                raise RuntimeError("The diagnostic did not complete exactly one attempt")
            write(
                out / "source-integrity.json",
                {"sources_unchanged_during_run": True, "checked_files": len(frozen)},
            )
            manifest["completed_runs"].append(
                {"state": state, "run": name, "attempt": evaluation["attempts"][0]}
            )
            manifest["active_run"] = None
            write(study / "probe-manifest.json", manifest)
            render(study)
    manifest.update(complete=True, finished_at=time.time())
    write(study / "probe-manifest.json", manifest)
    render(study)


if __name__ == "__main__":
    main()
