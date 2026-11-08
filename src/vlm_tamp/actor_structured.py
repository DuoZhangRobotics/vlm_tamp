"""One declarative robot request per bounded visual context; no policy-code execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from .compact_robot_interface import compact_response
from .coupled_storage import write_json
from .interaction_memory import InteractionMemory
from .robot_contracts import operation_schemas
from .subscription_agent import codex_arguments
from .visual_scene import CAMERAS

VISUAL_TOOLS = {"scene_views", "mark_scene_point", "preview_hand"}
OPERATIONS = (
    "observe",
    "episode_status",
    "inspect_video",
    "point",
    "segment_parts",
    "measure_surface",
    "select_surface_contact",
    "plan_contact_pose",
    "plan_pose",
    "plan_grasp",
    "execute",
    "gripper",
    "push",
    "continue_push",
    "push_stroke",
    "hook_pull",
    "ground",
    "inspect_scene",
    "plan_arc",
    "start_tracking",
    "measure_tracking",
    "apply_tracking",
    "track_part_motion",
    "track_video_motion",
    "estimate_and_verify_joint",
    "apply_joint_estimate",
    "apply_part_motion",
    "associate_views",
    "motion_settings",
    "finish",
)

INSTRUCTIONS = """You control a robot by selecting ONE structured tool request per decision.
Return the requested JSON envelope. arguments_json is a JSON OBJECT encoded as a string;
it is passed unchanged to the existing robot client and validated by the tool's schema.
Use help with {"operation":"tool_name"} to inspect a tool contract before an unfamiliar call.
Do not use shell/editor/MCP discovery tools, write programs, or generate executable policies.
You have no robot service connection in this decision workspace. Only your returned tool
request can produce an action. Targets, strategies, directions and recovery are your choices.

Current images and proprioception follow your task. Before/after memory records are measured
transitions from THIS episode; request success and robot movement are not task completion.
TRAJECTORY_GAP means omitted interactions, not adjacent frames. Notes are your hypotheses,
not independently verified facts. Keep the note concise and revise it when observations disagree.
No examples, successful coordinates, task-specific strategy or closing sign are provided.
The episode ends at native success, 600 controls or its wall deadline. Do not reset or issue
meaningless actions to use the budget. A failed plan does not end the episode.

Each native camera image is attached separately, unmodified and without a header or montage.
Use that image's own pixel coordinates for robot tools: x right and y down, origin at its
top-left corner. The image manifest names the camera. Scene projections instead use the
full labelled 640-by-640 image coordinates.
You may inspect earlier recorded frames with inspect_video. Every attached image is indexed below.
Geometry and perception tools are read-only unless their contract says they install a model.
For free pose motion, first select observed scene geometry with select_surface_contact or ground.
Motion planning and control use the existing numerical tools. Do not infer task completion from
an accepted plan or hand arrival. For a terminal finish request, set arguments_json to "{}"
and put supported findings and limits in the note field.
"""


def decision_schema(operations: list[str]) -> dict[str, Any]:
    """Constrain the action envelope, leaving arguments to the existing tool validator."""
    arguments: dict[str, Any] = {"type": "string"}
    if operations == ["finish"]:
        arguments["enum"] = ["{}"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["operation", "arguments_json", "expected_effect", "note"],
        "properties": {
            "operation": {"type": "string", "enum": operations},
            "arguments_json": arguments,
            "expected_effect": {"type": "string"},
            "note": {"type": "string"},
        },
    }


def tool_call(work: Path, operation: str, arguments_json: str, environment: dict) -> dict:
    """Use the supplied client; never evaluate model text as Python or shell code."""
    reply = subprocess.run(
        ["/usr/bin/python3", str(work / "robot.py"), "--full", operation, arguments_json],
        cwd=work,
        env=environment,
        text=True,
        capture_output=True,
        timeout=920,
    )
    if reply.returncode:
        return {
            "ok": False,
            "error_type": "tool_client_error",
            "error": reply.stderr[-4000:],
            "execution_status": "unknown; inspect episode_status before retrying",
            "client_returncode": reply.returncode,
        }
    return json.loads(reply.stdout)


def latest_observation(work: Path) -> dict[str, Any]:
    """Read only the public observation/proprioception export."""
    folders = [
        p
        for p in (work / "robot_data/attempt01").glob("obs*")
        if p.is_dir() and re.fullmatch(r"obs[0-9]+", p.name)
    ]
    folder = max(folders, key=lambda p: int(p.name[3:]))
    state = json.loads((folder / "robot.json").read_text())
    return {"observation_id": int(folder.name[3:]), "attempt": 1, **state}


def observation_images(work: Path, observation: dict) -> dict[str, str]:
    """Attach each original camera image without introducing a pixel-coordinate transform."""
    folder = (
        work
        / "robot_data"
        / f"attempt{observation.get('attempt', 1):02d}"
        / f"obs{observation['observation_id']:03d}"
    )
    paths = {camera: str((folder / f"{camera}.png").relative_to(work)) for camera in CAMERAS}
    for relative in paths.values():
        if not (work / relative).is_file():
            raise FileNotFoundError(work / relative)
    return paths


def image_paths(value: Any, work: Path) -> list[str]:
    """Find explicitly returned public images without reading private paths."""
    result = []
    keys = {"images", "image", "comparison_image", "approach_images", "preview_image", "annotated_image"}

    def visit(item: Any, selected: bool = False) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, selected or key in keys)
        elif isinstance(item, list):
            for child in item:
                visit(child, selected)
        elif selected and isinstance(item, str) and item.endswith((".png", ".jpg", ".jpeg")):
            path = work / item
            if path.resolve().is_relative_to(work.resolve()) and path.is_file() and not path.is_symlink():
                relative = str(path.relative_to(work))
                if relative not in result:
                    result.append(relative)

    visit(value, isinstance(value, str))
    return result


def feedback_record(response: dict, operation: str) -> dict:
    """Preserve compact existing outcomes; visual-tool metadata is already small."""
    result = (
        response
        if operation in VISUAL_TOOLS or operation in {"help", "inspect_image"}
        else compact_response(response)
    )
    preview = response.get("result", {}).get("hand_preview")
    if preview is not None:
        result.setdefault("result", {})["hand_preview"] = preview
    return result


def episode_status(work: Path) -> dict:
    """Use the existing public terminal state and wall clock."""
    result = json.loads((work / "robot_data/episode-status.json").read_text())
    deadline = json.loads((work / "robot_data/episode-deadline.json").read_text())["deadline_monotonic"]
    result["remaining_wall_seconds"] = max(0.0, deadline - time.monotonic())
    if not result.get("task_success") and result["remaining_wall_seconds"] == 0:
        result.update(episode_done=True, termination_reason="wall_budget_exhausted")
    return result


def publish_images(work: Path, out: Path, paths: list[str]) -> None:
    """Copy each small UI image once; never copy depth arrays or rewrite videos live."""
    for relative in paths:
        source = work / relative
        target = out / "live-images" / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def run_decision(
    prompt: str,
    images: list[Path],
    output: Path,
    environment: dict,
    schema: dict,
    model: str,
    reasoning: str,
    deadline: float,
) -> dict | None:
    """Run a fresh model context containing exactly the selected visual memory."""
    output.mkdir()
    with tempfile.TemporaryDirectory(prefix="luna-decision-") as temporary:
        work = Path(temporary)
        linked = []
        for index, source in enumerate(images, 1):
            target = work / f"image{index:02d}{source.suffix}"
            if source.stat().st_dev == work.stat().st_dev:
                os.link(source, target)
            else:
                shutil.copyfile(source, target)
            linked.append(target)
        write_json(work / "schema.json", schema)
        command = codex_arguments(
            work,
            work / "answer.json",
            model=model,
            reasoning=reasoning,
            readonly_paths=[*linked, work / "schema.json"],
        )[:-1]
        command += ["--disable", "shell_tool", "--disable", "view_image"]
        for target in linked:
            command += ["--image", str(target)]
        command += ["--output-schema", str(work / "schema.json"), "-"]
        write_json(output / "command.json", command)
        (output / "prompt.txt").write_text(prompt)
        began = time.perf_counter()
        with (output / "events.jsonl").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                text=True,
                cwd=work,
                env=environment,
            )
            try:
                process.stdin.write(prompt)
                process.stdin.close()
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.2)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
        timed_out = time.monotonic() >= deadline
        write_json(
            output / "timing.json",
            {
                "model_and_client_seconds": time.perf_counter() - began,
                "returncode": process.returncode,
                "wall_deadline_reached": timed_out,
                "input_images": len(images),
                "fresh_context": True,
                "robot_transport_access": False,
            },
        )
        if timed_out:
            return None
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        decision = json.loads((work / "answer.json").read_text())
        write_json(output / "decision.json", decision)
        return decision


def run_structured_actor(
    work: Path,
    out: Path,
    environment: dict,
    initial_prompt: str,
    seconds: int,
    *,
    model: str,
    reasoning: str,
    memory_mode: str,
    memory_slots: int,
    visual_interface: bool,
    retain_tool_documentation: bool = False,
) -> int:
    """Select and execute direct tool calls, with bounded within-episode evidence."""
    work, out = Path(work), Path(out)
    started = time.monotonic()
    deadline = started + seconds
    clock = {
        "deadline_monotonic": deadline,
        "wall_budget_seconds": seconds,
        "scope": "one physical episode; all model decisions share this deadline",
    }
    write_json(out / "episode-deadline.json", clock)
    write_json(work / "robot_data/episode-deadline.json", clock)
    (out / "agent-prompt.txt").write_text(INSTRUCTIONS + "\n" + initial_prompt)
    enabled = json.loads((work / "tools/schema-index.json").read_text())["operations"]
    operations = [
        *[name for name in OPERATIONS if name in enabled],
        "help",
        "inspect_image",
        *sorted(VISUAL_TOOLS if visual_interface else []),
    ]
    schema = decision_schema(operations)
    feedback = tool_call(work, "reset", "{}", environment)
    if not feedback.get("ok"):
        raise RuntimeError("Robot reset failed: " + json.dumps(feedback))
    current = latest_observation(work)
    memory = InteractionMemory(
        memory_mode, memory_slots, current["control_steps"], current["maximum_control_steps"]
    )
    instruction = feedback["result"]["instruction"]
    landmarks = feedback["result"].get("contact_landmarks_in_tcp", [])
    if visual_interface:
        feedback = tool_call(work, "scene_views", "{}", environment)
    last_operation = "scene_views" if visual_interface else "reset"
    note = ""
    turn = 0
    reason = None
    rows = []
    definitions = operation_schemas()
    catalog = {
        name: definitions[name]["function"]["description"] for name in operations if name in definitions
    }
    consulted_tool_contracts: dict[str, dict[str, Any]] = {}
    (out / "decisions").mkdir()
    (out / "agent-events.jsonl").touch()
    while time.monotonic() < deadline:
        status = episode_status(work)
        terminal = bool(status.get("episode_done"))
        current = latest_observation(work)
        current_images = observation_images(work, current)
        context = memory.context()
        paths = []
        for record in memory.selected():
            for key in ("before_images", "after_images"):
                for relative in record[key].values():
                    if relative not in paths:
                        paths.append(relative)
        for relative in current_images.values():
            if relative not in paths:
                paths.append(relative)
        extra_images = image_paths(feedback, work)
        for pending in reversed(memory.pending):
            extra_images.extend(p for p in image_paths(pending["receipt"], work) if p not in extra_images)
        omitted = []
        for relative in extra_images:
            if relative not in paths:
                if len(paths) < 48:
                    paths.append(relative)
                else:
                    omitted.append(relative)
        turn += 1
        folder = out / "decisions" / f"{turn:03d}"
        attachments = [
            {
                "image_number": i,
                "path": relative,
                "native_camera": Path(relative).stem if Path(relative).stem in CAMERAS else None,
                "sha256": hashlib.sha256((work / relative).read_bytes()).hexdigest(),
            }
            for i, relative in enumerate(paths, 1)
        ]
        payload = {
            "task": instruction,
            "tools": catalog,
            "visual_tools_available": visual_interface,
            "robot_contact_landmarks_at_reset": landmarks,
            "memory": context,
            "current_observation": current,
            "current_images": current_images,
            "episode_status": status,
            "latest_feedback": feedback_record(feedback, last_operation),
            "learner_note_unverified": note,
            "images": attachments,
            "help_arguments": {"operation": "tool_name"},
            "inspect_image_arguments": {"path": "public image path"},
            "images_omitted_by_declared_budget": omitted,
            "terminal_summary_requested": terminal,
        }
        if retain_tool_documentation:
            payload["previously_consulted_tool_contracts"] = consulted_tool_contracts
        prompt = INSTRUCTIONS + "\n" + initial_prompt + "\n" + json.dumps(payload, separators=(",", ":"))
        decision = run_decision(
            prompt,
            [work / p for p in paths],
            folder,
            environment,
            decision_schema(["finish"]) if terminal else schema,
            model,
            reasoning,
            deadline,
        )
        with (out / "agent-events.jsonl").open("a") as stream:
            stream.write((folder / "events.jsonl").read_text())
        write_json(folder / "input.json", payload)
        publish_images(work, out, paths)
        if decision is None:
            reason = "wall_budget_exhausted"
            break
        operation = decision["operation"]
        note = decision["note"]
        if operation not in operations:
            raise ValueError("Decision named a tool outside the declared interface")
        before = current
        begin = time.perf_counter()
        if operation == "help":
            args = json.loads(decision["arguments_json"])
            name = args["operation"]
            feedback = {
                "ok": name in catalog,
                "result": definitions[name]["function"] if name in catalog else {},
                "robot_motion_issued": False,
            }
            if retain_tool_documentation and feedback["ok"]:
                consulted_tool_contracts[name] = feedback["result"]
        elif operation == "inspect_image":
            args = json.loads(decision["arguments_json"])
            paths_requested = image_paths(args["path"], work)
            feedback = {
                "ok": bool(paths_requested),
                "result": {"images": paths_requested},
                "error": None if paths_requested else "No public image at the requested path",
                "robot_motion_issued": False,
            }
        else:
            feedback = tool_call(work, operation, decision["arguments_json"], environment)
        after = latest_observation(work)
        after_images = observation_images(work, after)
        record = {
            "decision": turn,
            "operation": operation,
            "arguments_json": decision["arguments_json"],
            "expected_effect": decision["expected_effect"],
            "learner_note_unverified": note,
            "before_control": before["control_steps"],
            "after_control": after["control_steps"],
            "before_observation": before["observation_id"],
            "after_observation": after["observation_id"],
            "before_images": current_images,
            "after_images": after_images,
            "receipt": feedback_record(feedback, operation),
            "tool_wall_seconds": time.perf_counter() - begin,
        }
        memory.add(record)
        write_json(folder / "result.json", record)
        rows.append(
            {
                "decision": turn,
                "operation": operation,
                "before_control": record["before_control"],
                "after_control": record["after_control"],
                "ok": feedback.get("ok"),
                "tool_wall_seconds": record["tool_wall_seconds"],
                "note": note,
                "memory_record_ids": [r["decision"] for r in memory.selected()],
                "help_operation": json.loads(decision["arguments_json"]).get("operation")
                if operation == "help"
                else None,
            }
        )
        publish_images(work, out, [*after_images.values(), *image_paths(feedback, work)])
        write_json(
            out / "decision-progress.json",
            {
                "decisions": rows,
                "current_observation": after,
                "condition": {
                    "visual_interface": visual_interface,
                    "memory_mode": memory_mode,
                    "retain_tool_documentation": retain_tool_documentation,
                },
                "complete": terminal,
            },
        )
        last_operation = operation
        if terminal:
            reason = status["termination_reason"]
            break
    reason = reason or "wall_budget_exhausted"
    write_json(out / "stop-service-reason.json", {"reason": reason})
    (work / "findings.txt").write_text(
        "Learner-authored terminal note (not an independent success label):\n" + note + "\n"
    )
    (out / "agent-last.txt").write_text(note + "\n")
    write_json(out / "memory-final.json", memory.context())
    write_json(
        out / "agent-exit.json",
        {
            "returncode": 0,
            "requested_model": model,
            "requested_reasoning": reasoning,
            "wall_seconds": time.monotonic() - started,
            "turns": turn,
            "termination_reason": reason,
            "normal_terminal": True,
            "structured_tool_calls": True,
            "generated_policy_execution": False,
            "retain_tool_documentation": retain_tool_documentation,
            "consulted_tool_contracts": sorted(consulted_tool_contracts),
        },
    )
    return 0
