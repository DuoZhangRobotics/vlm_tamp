"""Run paired, two-turn Luna assessments of frozen recorded robot evidence.

No live robot, teacher, reflection agent or cross-case learning is involved.
The code and protocol are frozen before the first model request.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vlm_tamp.checkpoint_diagnostic import read_json, sha256, verify_packet, write_json
from vlm_tamp.recorded_evidence import (
    COMMON_BEFORE, COMMON_AFTER, BEFORE_INSTRUCTIONS, AFTER_INSTRUCTIONS,
    attached_images, build_stages, score_labels, validate_note, validate_decision,
)
from vlm_tamp.subscription_agent import (
    clean_environment, codex_arguments, codex_resume_arguments, verify_isolation,
)

SOURCE = ROOT / "results/luna-visual-inspection-ab-20261025"
CONDITIONS = ["baseline", "predict-inspect-revise"]
CASES = [
    ("state020-r1-visual-inspection", 31, "stationary_door_after_hand_motion"),
    ("state035-r1-baseline", 32, "stationary_door_after_hand_motion"),
    ("state039-r1-visual-inspection", 8, "measured_pose_unaccepted_hinge"),
    ("state039-r1-visual-inspection", 10, "visible_motion_without_accepted_fit"),
    ("state039-r1-baseline", 17, "native_success_early_stop"),
    ("state020-r1-visual-inspection", 23, "grasp_reserve_rejection"),
]
SOURCES = [
    "scripts/run_luna_evidence_diagnostic.py", "src/vlm_tamp/recorded_evidence.py",
    "src/vlm_tamp/checkpoint_diagnostic.py", "src/vlm_tamp/subscription_agent.py",
    "src/vlm_tamp/public_analysis.py", "src/vlm_tamp/compact_robot_interface.py",
    "src/vlm_tamp/robot_contracts.py", "tests/test_recorded_evidence.py",
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    write_json(temp, value)
    temp.replace(path)


def expected(category):
    normal = ["inspect_existing", "request_observation", "reestimate", "continue_motion", "replan"]
    result = {"motion_labels": ["moved"], "pose_update_basis": "hand_prediction",
              "task_success": False, "next_step_kinds": normal}
    if category == "stationary_door_after_hand_motion":
        result.update(motion_labels=["no_visible_motion"], contact_loss_case=True,
                      next_step_kinds=["inspect_existing", "request_observation", "reestimate", "recover_contact"])
    elif category == "measured_pose_unaccepted_hinge":
        result["pose_update_basis"] = "measured_rgbd"
    elif category == "visible_motion_without_accepted_fit":
        result["next_step_kinds"] = normal + ["recover_contact"]
    elif category == "native_success_early_stop":
        # The successful terminal movement is <1 degree. It is not necessary
        # to visually resolve it to obey an authoritative successful stop.
        result.update(motion_labels=["moved", "no_visible_motion", "uncertain"],
                      pose_update_basis="none", task_success=True, next_step_kinds=["stop_success"])
    elif category == "grasp_reserve_rejection":
        result.update(motion_labels=["not_executed"], pose_update_basis="none",
                      next_step_kinds=["inspect_existing", "request_observation", "reestimate", "recover_contact", "replan"])
    return result


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20261026)
    ids = rng.sample(range(100000, 999999), len(CASES))
    shared = SOURCE / "trials/state035-r1-baseline/agent-workspace"
    common = {name: shared / name for name in ["tools.txt", "experience/guidelines.json", "example/readme.txt"]}
    cases = []
    diagnoses = read_json(SOURCE / "failure-diagnoses-private.json")["trials"]
    for identifier, (name, event, category) in zip(ids, CASES):
        key = f"c{identifier}"
        manifests = build_stages(SOURCE / "trials" / name, event, out / "packets" / key, shared_sources=common)
        native = next(r for r in diagnoses if r["id"] == name)
        arc = next((r for r in native["executed_arcs"] if r["execution_event"] == event), None)
        cases.append({"case_id": key, "source_trial": name, "event": event,
                      "private_category": category, "expected": expected(category),
                      "private_native_arc": arc, "stages": manifests})
    save(out / "cases-private.json", cases)
    rubric = {
        "visibility": "Private evaluator only. Never sent to Luna.",
        "primary": "Within-case/repetition paired difference in all registered label checks passing.",
        "report_separately": ["Motion discrimination on the four ordinary executed arcs", "pose provenance",
                            "contact caution", "success stop", "budget-rejection interpretation",
                            "public evidence support and request feasibility", "timing and raw CLI token aggregates"],
        "review": [
            "Label agreement is a screen, not sufficient evidence of correct diagnosis or action feasibility.",
            "Review observed_changes/evidence/working_note and next_step independently of condition labels.",
            "Do not force a specific motion plan or hidden native hinge/contact measurement.",
            "A rejected visual fit does not prove zero motion or lost contact. RGB pairs can still show motion.",
            "A measured pose with rejected hinge fit is not a valid hinge estimate.",
            "A grasp reserve rejection is not evidence that cuRobo searched and failed or that all motion is impossible.",
            "A native-success-terminated trajectory must not be treated as an execution failure.",
            "Uncertain motion is allowed but does not count as discrimination on the four visibly clear ordinary arcs.",
            "A generic uncertainty label without relevant evidence is not a successful diagnosis.",
            "Native motion/contact values corroborate observer diagnosis only, never supply actor knowledge.",
            "Two samples per checkpoint are not independent environments; repeated state39 and state20 checkpoints are correlated.",
            "Any gain is evidence-use performance on selected replay checkpoints, not physical recovery or learned transfer.",
        ],
    }
    save(out / "rubric-private.json", rubric)
    for condition in CONDITIONS:
        for stage, common_prompt, additions in [("before", COMMON_BEFORE, BEFORE_INSTRUCTIONS),
                                                ("after", COMMON_AFTER, AFTER_INSTRUCTIONS)]:
            path = out / "prompts" / f"{condition}-{stage}.txt"
            path.parent.mkdir(exist_ok=True)
            path.write_text(common_prompt + "\n" + additions[condition])
    jobs = []
    # Same case/repetition adjacent, alternating order; reversed for repeat 2.
    for repeat in [1, 2]:
        for i, case in enumerate(cases if repeat == 1 else reversed(cases)):
            order = CONDITIONS if (i + repeat) % 2 else CONDITIONS[::-1]
            for condition in order:
                jobs.append({"run_id": f"run{len(jobs)+1:03d}", "case_id": case["case_id"],
                             "condition": condition, "repeat": repeat})
    save(out / "schedule-private.json", jobs)
    hashes = {}
    for name in SOURCES:
        path = ROOT / name
        hashes[name] = sha256(path)
        target = out / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    protocol = {
        "created_at": utc(), "model": "gpt-6-luna", "reasoning": "high",
        "scope": "Selected development replay checkpoints from one microwave asset, not held out.",
        "planned_sessions": len(jobs), "turns_per_session": 2, "seconds_per_turn": 240,
        "models_per_session": 1, "automatic_retries": 0, "concurrency": 1,
        "conditions": CONDITIONS, "source_study": str(SOURCE),
        "guide_sha256": sha256(shared / "experience/guidelines.json"),
        "intervention": "Generic expectation before a recorded request, followed by explicit comparison/revision. No task-specific hint or evaluator answer.",
        "common_to_both": "Same complete public prefixes, compact history, robot manuals, guide, numerical libraries, final schema and four before/eight paired original images delivered as prompt attachments. Two turns and equal wall limits.",
        "image_delivery": "CLI --image at each turn; no optional image-opening step required. No resizing/cropping/overlays in actor attachments.",
        "temporal_isolation": "Only before packet exists in actor workspace in turn 1. After packet is copied by host only after turn 1 exits and its note is archived.",
        "memory": "Within-session public before note only. No cross-case memory, teacher, external model, guide update or promotion.",
        "physical_robot_calls": 0, "simulator_started": False,
        "limits": "Bundled direct image delivery and reduced replay context are shared controls. Comparing to previous live failures is descriptive, not an estimate of delivery benefit. No statement about physical success from this diagnostic.",
        "source_sha256": hashes,
        "registered_sha256": {str(p.relative_to(out)): sha256(p) for p in
                              [out / "cases-private.json", out / "rubric-private.json", out / "schedule-private.json",
                               *sorted((out / "prompts").glob("*.txt"))]},
    }
    save(out / "protocol.json", protocol)
    save(out / "status.json", {"phase": "prepared", "completed": 0, "planned": len(jobs)})
    return protocol


def verify(out, protocol):
    for name, digest in protocol["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Frozen source changed: " + name)
    for name, digest in protocol["registered_sha256"].items():
        if sha256(out / name) != digest:
            raise ValueError("Registered input changed: " + name)


def turn(work, folder, stage, command, prompt, images, seconds):
    image_args = [item for path in images for item in ["--image", str(path)]]
    idx = command.index("-o")
    command[idx:idx] = image_args
    save(folder / f"{stage}-command.json", command)
    (folder / f"{stage}-prompt.txt").write_text(prompt)
    save(folder / f"{stage}-attachments.json", [{"path": str(p), "sha256": sha256(p)} for p in images])
    started = time.perf_counter()
    timed_out = False
    receipts = []
    offset = 0
    pending = b""
    with (folder / f"{stage}-events.jsonl").open("w") as log, (folder / f"{stage}-stderr.log").open("w") as err:
        process = subprocess.Popen(command, cwd=work, env=clean_environment(), stdin=subprocess.PIPE,
                                   stdout=log, stderr=err, text=True, start_new_session=True)
        save(folder / f"{stage}-process.json", {"pid": process.pid, "started_at": utc()})
        process.stdin.write(prompt)
        process.stdin.close()
        while True:
            with (folder / f"{stage}-events.jsonl").open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read()
                offset += len(chunk)
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                item = event.get("item", {})
                receipts.append({"seconds": time.perf_counter() - started, "type": event.get("type"),
                                 "item_id": item.get("id"), "item_type": item.get("type")})
            if process.poll() is not None:
                break
            if time.perf_counter() - started > seconds:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                break
            time.sleep(.05)
    elapsed = time.perf_counter() - started
    save(folder / f"{stage}-event-clock.json", receipts)
    starts, intervals = {}, []
    for receipt in receipts:
        if receipt["item_type"] != "command_execution":
            continue
        if receipt["type"] == "item.started":
            starts[receipt["item_id"]] = receipt["seconds"]
        elif receipt["type"] == "item.completed" and receipt["item_id"] in starts:
            intervals.append((starts.pop(receipt["item_id"]), receipt["seconds"]))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    command_seconds = sum(end - start for start, end in merged)
    events = []
    for line in (folder / f"{stage}-events.jsonl").read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    threads = [e["thread_id"] for e in events if e.get("type") == "thread.started"]
    usages = [e["usage"] for e in events if e.get("type") == "turn.completed"]
    return {"returncode": process.returncode, "wall_seconds": elapsed, "timed_out": timed_out,
            "thread_ids": threads, "usage_records": usages, "pid": process.pid,
            "observed_command_seconds": command_seconds, "outside_observed_commands_seconds": elapsed - command_seconds,
            "timing_scope": "50ms CLI event-receipt sampling. Outside command intervals includes inference, image tools, scheduling and transport; not pure reasoning.",
            "tool_calls": [e["item"] for e in events if e.get("type") == "item.completed"
                           and e.get("item", {}).get("type") == "command_execution"]}


def run_one(out, job, case, protocol):
    folder = out / "responses" / job["run_id"]
    folder.mkdir(parents=True, exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix="vlm-evidence-"))
    started = time.perf_counter()
    record = {**job, "started_at": utc(), "completed": False, "valid_decision": False, "workspace": str(work)}
    try:
        (work / "inputs").mkdir()
        shutil.copytree(out / "packets" / case["case_id"] / "before", work / "inputs/before")
        readonly = [work / "inputs"]
        forbidden = [out / "rubric-private.json", out / "cases-private.json", SOURCE / "failure-diagnoses-private.json",
                     out / "packets" / case["case_id"] / "after/latest-result.json", ROOT / "src/vlm_tamp/subscription_agent.py",
                     Path.home() / ".codex/auth.json"]
        link = work / "private-canary-link"
        link.symlink_to(out / "rubric-private.json")
        check_start = time.perf_counter()
        record["isolation"] = verify_isolation(work, [*forbidden, link], readonly_paths=readonly)
        record["preflight_seconds"] = time.perf_counter() - check_start
        link.unlink()
        record["before_packet_check"] = verify_packet(work / "inputs/before", case["stages"]["before"])
        record["after_absent_before_turn"] = not (work / "inputs/after").exists()
        if not record["after_absent_before_turn"]:
            raise ValueError("Future evidence exposed before prediction")
        args = codex_arguments(work, folder / "before-answer.txt", model=protocol["model"],
                               reasoning=protocol["reasoning"], readonly_paths=readonly, ephemeral=False)
        before = turn(work, folder, "before", args,
                      (out / "prompts" / f"{job['condition']}-before.txt").read_text(),
                      attached_images(work / "inputs", case["stages"], "before"), protocol["seconds_per_turn"])
        record["before"] = before
        if (work / "before.json").exists():
            shutil.copyfile(work / "before.json", folder / "before.json")
            try:
                validate_note(folder / "before.json")
                record["valid_before_note"] = True
            except (ValueError, TypeError) as error:
                record["note_error"] = str(error)
        else:
            record["note_error"] = "No before.json produced"
        if before["returncode"] or len(before["thread_ids"]) != 1:
            raise RuntimeError("Before turn did not complete; no automatic retry")
        # Host reveals only this one recorded result after the model turn ends.
        shutil.copytree(out / "packets" / case["case_id"] / "after", work / "inputs/after")
        immutable = [*readonly, *([work / "before.json"] if (work / "before.json").exists() else [])]
        args = codex_resume_arguments(work, folder / "after-answer.txt", before["thread_ids"][0],
                                      model=protocol["model"], reasoning=protocol["reasoning"], readonly_paths=immutable)
        after = turn(work, folder, "after", args,
                     (out / "prompts" / f"{job['condition']}-after.txt").read_text(),
                     attached_images(work / "inputs", case["stages"], "after"), protocol["seconds_per_turn"])
        record["after"] = after
        record["same_thread"] = before["thread_ids"] == after["thread_ids"]
        if not record["same_thread"]:
            raise RuntimeError("Second turn did not resume the same context")
        record["completed"] = after["returncode"] == 0
        if (work / "decision.json").exists():
            shutil.copyfile(work / "decision.json", folder / "decision.json")
            try:
                decision = validate_decision(folder / "decision.json")
                record["valid_decision"] = True
                record["score"] = score_labels(decision, case["expected"])
            except Exception as error:
                record["decision_error"] = str(error)
        else:
            record["decision_error"] = "No decision.json produced"
        for stage in ["before", "after"]:
            record[f"{stage}_packet_final_check"] = verify_packet(work / "inputs" / stage, case["stages"][stage])
        if (folder / "before.json").exists():
            record["before_note_unchanged"] = sha256(folder / "before.json") == sha256(work / "before.json")
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        archive_start = time.perf_counter()
        # All public case inputs already have immutable copies in packets/.
        # Preserve only actor-created files here, avoiding 24 copies of arrays.
        for path in work.iterdir():
            if path.name in {"inputs", ".analysis_site", "allowed-canary.txt"}:
                continue
            destination = folder / "workspace-output" / path.name
            destination.parent.mkdir(exist_ok=True)
            if path.is_dir() and not path.is_symlink():
                shutil.copytree(path, destination, symlinks=True)
            else:
                shutil.copy2(path, destination, follow_symlinks=False)
        shutil.rmtree(work)
        record["archive_seconds"] = time.perf_counter() - archive_start
        record["total_seconds"] = time.perf_counter() - started
        record["finished_at"] = utc()
        save(folder / "result.json", record)
    return record


def report(out):
    command = [sys.executable, str(ROOT / "scripts/report_luna_evidence_diagnostic.py"), "--output", str(out)]
    with (out / "report.log").open("a") as stream:
        result = subprocess.run(command, stdout=stream, stderr=stream, timeout=60)
    if result.returncode:
        raise RuntimeError("Report build failed; see report.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "results/luna-evidence-loop-20261026"))
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    protocol = read_json(out / "protocol.json") if (out / "protocol.json").exists() else prepare(out)
    verify(out, protocol)
    if args.prepare:
        report(out)
        print(json.dumps({"prepared": str(out), "sessions": protocol["planned_sessions"]}))
        return
    with (out / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        auth = subprocess.run(["codex", "login", "status"], env=clean_environment(), capture_output=True, text=True, check=True)
        if "Logged in using ChatGPT" not in auth.stdout + auth.stderr:
            raise RuntimeError("ChatGPT subscription required; no API fallback")
        save(out / "authentication.json", {"chatgpt_subscription": True, "api_fallback": False,
             "client_version": subprocess.check_output(["codex", "--version"], text=True).strip()})
        cases = {c["case_id"]: c for c in read_json(out / "cases-private.json")}
        jobs = read_json(out / "schedule-private.json")
        started = time.perf_counter()
        rows = []
        for job in jobs:
            verify(out, protocol)
            path = out / "responses" / job["run_id"] / "result.json"
            if path.exists():
                row = read_json(path)
            else:
                save(out / "status.json", {"phase": "running", "completed": len(rows), "planned": len(jobs), "current": job})
                report(out)
                row = run_one(out, job, cases[job["case_id"]], protocol)
            rows.append(row)
            save(out / "results.json", rows)
            report(out)
            print(json.dumps({key: row.get(key) for key in
                              ["run_id", "case_id", "condition", "completed", "valid_decision", "total_seconds", "error"]}), flush=True)
            if row.get("error"):
                save(out / "status.json", {"phase": "stopped_for_inspection", "completed": len(rows), "current": job, "error": row["error"]})
                report(out)
                raise RuntimeError(row["error"])
        verify(out, protocol)
        save(out / "status.json", {"phase": "complete", "completed": len(rows), "planned": len(jobs),
                                   "runner_seconds": time.perf_counter() - started})
        report(out)


if __name__ == "__main__":
    main()
