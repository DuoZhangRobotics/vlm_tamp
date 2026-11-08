"""The bounded actor dispatches data requests and preserves actual transition evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest

import vlm_tamp.actor_structured as actor


def test_tool_argument_text_is_never_a_shell_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        actor.subprocess,
        "run",
        lambda command, **kw: (
            calls.append((command, kw)) or SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")
        ),
    )
    literal = '{"query":"$(never_execute_this)"}'
    assert actor.tool_call(tmp_path, "point", literal, {})["ok"]
    assert calls[0][0] == ["/usr/bin/python3", str(tmp_path / "robot.py"), "--full", "point", literal]
    assert "shell" not in calls[0][1]


def test_image_selection_excludes_private_paths_and_duplicate_source_crops(tmp_path: Path) -> None:
    Image.new("RGB", (4, 4)).save(tmp_path / "shown.png")
    Image.new("RGB", (4, 4)).save(tmp_path / "source.png")
    value = {
        "images": ["shown.png"],
        "metadata": {"source_image": "source.png"},
        "private": {"image": "../hidden.png"},
    }
    assert actor.image_paths(value, tmp_path) == ["shown.png"]


@pytest.mark.parametrize("retain_tool_documentation", [False, True])
@pytest.mark.parametrize("compact_transitions", [False, True])
@pytest.mark.parametrize("help_tool", ["push_stroke", "refresh_contact"])
def test_episode_uses_one_structured_action_and_complete_before_after_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retain_tool_documentation: bool, help_tool: str,
    compact_transitions: bool,
) -> None:
    work = tmp_path / "public"
    out = tmp_path / "run"
    work.mkdir()
    out.mkdir()
    (work / "robot_data").mkdir()
    (work / "tools").mkdir()
    (work / "tools/schema-index.json").write_text(
        json.dumps({"operations": {"push_stroke": {}, "refresh_contact": {}, "observe": {}, "finish": {}}})
    )
    state = {"observation_id": 1, "attempt": 1, "control_steps": 60, "maximum_control_steps": 600}
    choices = iter(
        [
            {
                "operation": "help",
                "arguments_json": json.dumps({"operation": help_tool}),
                "expected_effect": "read contract",
                "note": "unverified",
            },
            {
                "operation": "push_stroke",
                "arguments_json": '{"direction_world":[1,0,0],"distance_m":0.01}',
                "expected_effect": "test a direction",
                "note": "await outcome",
            },
            {
                "operation": "finish",
                "arguments_json": "{}",
                "expected_effect": "record completion",
                "note": "observed native success",
            },
        ]
    )
    dispatched = []
    inputs = []
    supplied_schemas = []

    def tool(_work: Path, operation: str, args: str, environment: dict) -> dict:
        dispatched.append(operation)
        if operation == "reset":
            return {"ok": True, "result": {"instruction": "fixture task"}}
        if operation == "push_stroke":
            state.update(control_steps=70, observation_id=2)
        return {
            "ok": True,
            "result": {"observation": dict(state), "task_success": state["control_steps"] == 70},
        }

    def images(_work: Path, observation: dict) -> dict[str, str]:
        result = {}
        for camera in ("agentview", "wrist"):
            relative = f"obs{observation['observation_id']}-{camera}.png"
            path = work / relative
            if not path.exists():
                Image.new("RGB", (10, 10)).save(path)
            result[camera] = relative
        return result

    def decision(prompt: str, images: list, folder: Path, *args: Any) -> dict:
        inputs.append(json.loads(prompt.rsplit("\n", 1)[1]))
        supplied_schemas.append(args[1])
        folder.mkdir()
        (folder / "events.jsonl").write_text("")
        return next(choices)

    monkeypatch.setattr(actor, "tool_call", tool)
    monkeypatch.setattr(actor, "latest_observation", lambda _: dict(state))
    monkeypatch.setattr(actor, "observation_images", images)
    monkeypatch.setattr(actor, "run_decision", decision)
    monkeypatch.setattr(
        actor,
        "episode_status",
        lambda _: {
            "episode_done": state["control_steps"] == 70,
            "task_success": state["control_steps"] == 70,
            "termination_reason": "simulator_success",
        },
    )
    assert (
        actor.run_structured_actor(
            work,
            out,
            {},
            "fixture",
            60,
            model="gpt-6-luna",
            reasoning="high",
            memory_mode="anchored",
            memory_slots=8,
            visual_interface=False,
            retain_tool_documentation=retain_tool_documentation,
            compact_transitions=compact_transitions,
        )
        == 0
    )
    assert dispatched == ["reset", "push_stroke", "finish"]
    assert help_tool in supplied_schemas[0]["properties"]["operation"]["enum"]
    assert inputs[1]["latest_feedback"]["ok"]
    assert inputs[1]["latest_feedback"]["result"]["name"] == help_tool
    assert inputs[1]["memory"]["retained_executed_transitions"] == 0
    transition = inputs[2]["memory"]["events"][0]
    assert transition["before_control"] == 60 and transition["after_control"] == 70
    assert transition["before_images"] == {"agentview": "obs1-agentview.png", "wrist": "obs1-wrist.png"}
    assert transition["after_images"] == {"agentview": "obs2-agentview.png", "wrist": "obs2-wrist.png"}
    assert inputs[1]["memory"]["recent_nonmotion_feedback"][0]["operation"] == "help"
    assert "preceding_nonmotion" not in transition
    assert inputs[2]["terminal_summary_requested"]
    assert len(inputs[2]["images"]) == 4
    assert "scene_views" not in inputs[0]["tools"]
    assert inputs[2]["memory"]["recent_nonmotion_feedback"] == []
    if compact_transitions:
        assert inputs[2]['latest_feedback']['executed_transition_decision'] == 2
        assert transition['receipt']['task_success'] is True
    else:
        assert 'feedback_format' not in inputs[2]
    if retain_tool_documentation:
        assert inputs[0]["previously_consulted_tool_contracts"] == {}
        expected = {help_tool: actor.operation_schemas()[help_tool]["function"]}
        assert inputs[1]["previously_consulted_tool_contracts"] == expected
        assert inputs[2]["previously_consulted_tool_contracts"] == expected
    else:
        assert all("previously_consulted_tool_contracts" not in value for value in inputs)
    assert json.loads((out / "agent-exit.json").read_text())["generated_policy_execution"] is False


def test_native_camera_attachments_preserve_tool_pixel_coordinates(tmp_path: Path) -> None:
    """A marked pixel is identical in model input and depth-source coordinates, in both cameras."""
    folder = tmp_path / "robot_data/attempt01/obs003"
    folder.mkdir(parents=True)
    for camera, color in [("agentview", (255, 0, 0)), ("wrist", (0, 255, 0))]:
        image = Image.new("RGB", (512, 512))
        image.putpixel((276, 270), color)
        image.save(folder / f"{camera}.png")
    paths = actor.observation_images(tmp_path, {"observation_id": 3, "attempt": 1})
    for camera, color in [("agentview", (255, 0, 0)), ("wrist", (0, 255, 0))]:
        source = folder / f"{camera}.png"
        supplied = tmp_path / paths[camera]
        assert supplied.read_bytes() == source.read_bytes()
        with Image.open(supplied) as image:
            assert image.size == (512, 512)
            assert image.getpixel((276, 270)) == color
            assert image.getpixel((276, 306)) == (0, 0, 0)


def test_terminal_request_uses_empty_finish_arguments_and_separate_findings() -> None:
    """Final notes cannot become invented arguments to the existing finish operation."""
    schema = actor.decision_schema(["finish"])
    assert schema["properties"]["arguments_json"]["enum"] == ["{}"]
    assert "note" in schema["required"]
    assert "enum" not in actor.decision_schema(["push", "finish"])["properties"]["arguments_json"]
