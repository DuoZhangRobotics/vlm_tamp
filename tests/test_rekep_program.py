"""Security, numerical and durable-cost checks for the ReKep adaptation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from vlm_tamp.rekep_libero.api import (
    ApiError, BudgetError, DurableLedger, UncertainRequest, request_arguments, reservation_usd,
)
from vlm_tamp.rekep_libero.program import ProgramError, build_prompt, parse_program
from vlm_tamp.rekep_libero.report import build_report


PROGRAM = '''```python
num_stages = 2
def stage1_subgoal_constraint1(end_effector, keypoints):
    """Reach the selected bowl keypoint."""
    return np.linalg.norm(end_effector - keypoints[0])
def stage2_subgoal_constraint1(end_effector, keypoints):
    target = np.mean(keypoints[[1, 2]], axis=0) + np.array([0, 0, 0.05])
    return np.linalg.norm(keypoints[0] - target)
def stage2_path_constraint1(end_effector, keypoints):
    return get_grasping_cost_by_keypoint_idx(0)
grasp_keypoints = [0, -1]
release_keypoints = [-1, 0]
```'''


def _replace_return(expression):
    return PROGRAM.replace("return np.linalg.norm(end_effector - keypoints[0])", "return " + expression)


def test_stage_costs_use_observed_coordinates_and_live_attachment():
    attached = {0: False}
    program = parse_program(PROGRAM, 3, "pick_place", lambda index: 0.0 if attached[index] else 1.0)
    points = np.array([[0.0, 0.0, 0.05], [-0.1, 0, 0], [0.1, 0, 0]])
    assert program.stages[0].subgoal_constraints[0](points[0], points) == 0
    assert program.stages[1].subgoal_constraints[0](points[0], points) == 0
    assert program.stages[1].path_constraints[0](points[0], points) == 1
    attached[0] = True
    assert program.stages[1].path_constraints[0](points[0], points) == 0
    program.validate_values(points[0], points)


@pytest.mark.parametrize("expression", [
    "open('/tmp/key').read()", "__import__('os').system('id')", "np.load('/tmp/key')",
    "np.linalg.__dict__", "keypoints.__class__", "np.array([1], dtype=object)",
    "np.asarray(keypoints, out=keypoints)", "[x for x in keypoints]", "keypoints[3]",
    "keypoints[-1]", "keypoints[int(end_effector[0])]", "2 ** 10000000",
])
def test_rejects_io_dynamic_code_and_unbounded_indices(expression):
    with pytest.raises(ProgramError):
        parse_program(_replace_return(expression), 3, "pick_place")


@pytest.mark.parametrize("prefix", ["import os\n", "from numpy import load\n", "print('hello')\n",
                                     "global x\n", "x = np.load('/tmp/key')\n"])
def test_rejects_top_level_execution(prefix):
    with pytest.raises(ProgramError):
        parse_program(PROGRAM.replace("num_stages = 2", prefix + "num_stages = 2"), 3, "pick_place")


def test_rejects_mutation_and_branches():
    for content in ("keypoints[0] = np.array([0, 0, 0])", "if True:\n        return 0",
                    "end_effector = np.array([0, 0, 0])"):
        text = PROGRAM.replace("    return np.linalg.norm(end_effector - keypoints[0])", "    " + content)
        with pytest.raises(ProgramError):
            parse_program(text, 3, "pick_place")


@pytest.mark.parametrize("replacement", [
    "num_stages = True", "num_stages = 13", "num_stages = 1", "num_stages = '2'",
])
def test_rejects_invalid_stage_metadata(replacement):
    with pytest.raises(ProgramError):
        parse_program(PROGRAM.replace("num_stages = 2", replacement), 3, "pick_place")


def test_release_requires_matching_earlier_grasp():
    with pytest.raises(ProgramError):
        parse_program(PROGRAM.replace("release_keypoints = [-1, 0]", "release_keypoints = [-1, 1]"), 3, "pick_place")
    with pytest.raises(ProgramError):
        parse_program(PROGRAM.replace("release_keypoints = [-1, 0]", "release_keypoints = [0, -1]"), 3, "pick_place")


@pytest.mark.parametrize("expression", ["1 / 0", "np.sqrt(-1)", "np.array([1, 2])",
                                        "np.linalg.norm(np.array([1e308, 1e308]))"])
def test_evaluation_requires_finite_scalar(expression):
    program = parse_program(_replace_return(expression), 3, "pick_place")
    with pytest.raises(ProgramError):
        program.validate_values(np.zeros(3), np.zeros((3, 3)))


def test_pickplace_rejects_geometric_path_and_drawer_allows_pull_stage():
    changed = PROGRAM.replace("return get_grasping_cost_by_keypoint_idx(0)",
                              "return np.linalg.norm(keypoints[0][0:2] - keypoints[1][0:2])")
    with pytest.raises(ProgramError, match="geometric path"):
        parse_program(changed, 3, "pick_place")
    program = parse_program(changed, 3, "open_middle_drawer")
    assert len(program.stages[1].path_constraints) == 1


def test_drawer_line_uses_static_anchors_and_matching_attachment_helper():
    changed = PROGRAM.replace("grasp_keypoints = [0, -1]", '''def stage2_path_constraint2(end_effector, keypoints):
    direction = keypoints[2] - keypoints[1]
    displacement = keypoints[0] - keypoints[1]
    return np.linalg.norm(np.cross(displacement, direction)) / np.linalg.norm(direction)
grasp_keypoints = [0, -1]''')
    program = parse_program(changed, 3, "open_middle_drawer", lambda index: 0)
    points = np.array([[0, 0.2, 0], [0, 0, 0], [0, 0.1, 0]], dtype=float)
    assert program.stages[1].path_constraints[1](points[0], points) == 0
    with pytest.raises(ProgramError, match="matching grasp"):
        parse_program(changed.replace("get_grasping_cost_by_keypoint_idx(0)",
                                      "get_grasping_cost_by_keypoint_idx(1)"), 3, "open_middle_drawer")


def test_native_prompt_preserved_with_task_policy(tmp_path):
    template = tmp_path / "prompt.txt"
    template.write_text("Native instructions\nTask: {instruction}")
    prompt = build_prompt("put the bowl on the plate", "pick_place", template)
    assert prompt.startswith("Native instructions\nTask: put the bowl on the plate")
    assert "do not add geometric path constraints" in prompt
    assert "provided inventory" not in prompt


class FakeClient:
    def __init__(self, *, fail=False, usage=True):
        self.calls = 0
        self.fail = fail
        self.usage = usage
        self.responses = self

    def create(self, **arguments):
        self.calls += 1
        if self.fail:
            raise TimeoutError("uncertain remote attempt")
        raw = {"status": "completed", "model": "gpt-6-luna", "output": []}
        if self.usage:
            raw["usage"] = {"input_tokens": 100, "output_tokens": 200}
        return SimpleNamespace(model_dump=lambda mode: raw)


def _args(tmp_path):
    image = tmp_path / "numbered.png"
    image.write_bytes(b"test image bytes")
    return request_arguments(image, "Only an instruction and numbered RGB scene")


def test_uncertain_attempt_retains_reservation_and_never_repeats(tmp_path):
    ledger = DurableLedger(tmp_path / "ledger")
    args = _args(tmp_path)
    client = FakeClient(fail=True)
    with pytest.raises(UncertainRequest):
        ledger.perform(args, "scene", allow_paid=True, client=client)
    snapshot = ledger.snapshot()
    assert snapshot["api_attempts"] == 1
    assert snapshot["requests"]["scene"]["status"] == "uncertain"
    assert snapshot["experiment_conservative_spend_usd"] == reservation_usd(args)
    restarted = DurableLedger(tmp_path / "ledger")
    with pytest.raises(UncertainRequest):
        restarted.perform(args, "scene", allow_paid=True, client=client)
    assert client.calls == 1


def test_completed_hash_cache_does_not_bill_again_and_rejects_changed_scene(tmp_path):
    ledger = DurableLedger(tmp_path / "ledger")
    args = _args(tmp_path)
    client = FakeClient()
    first = ledger.perform(args, "scene", allow_paid=True, client=client)
    assert ledger.perform(args, "scene", allow_paid=False, client=client) == first
    assert ledger.perform(args, "equivalent-image-scene", client=client) == first
    assert ledger.snapshot()["api_attempts"] == 1
    assert client.calls == 1
    changed = json.loads(json.dumps(args))
    changed["input"][0]["content"][0]["text"] += " modified"
    with pytest.raises(ApiError, match="different inputs"):
        ledger.perform(changed, "equivalent-image-scene", allow_paid=True, client=client)
    assert client.calls == 1


def test_response_saved_before_accounting_recovers_after_crash(tmp_path):
    ledger = DurableLedger(tmp_path / "ledger")
    args = _args(tmp_path)
    client = FakeClient()
    ledger.perform(args, "scene", allow_paid=True, client=client)
    saved = json.loads(ledger.path.read_text())
    saved["requests"]["scene"].pop("cost_usd")
    saved["requests"]["scene"]["status"] = "pending"
    ledger.path.write_text(json.dumps(saved))
    restarted = DurableLedger(tmp_path / "ledger")
    restarted.perform(args, "scene", client=client)
    assert restarted.snapshot()["requests"]["scene"]["status"] == "completed"
    assert client.calls == 1


def test_pending_before_remote_call_and_missing_usage_conservatively_accounted(tmp_path):
    ledger = DurableLedger(tmp_path / "ledger")
    args = _args(tmp_path)

    class InspectClient(FakeClient):
        def create(self, **arguments):
            saved = json.loads(ledger.path.read_text())
            assert saved["requests"]["scene"]["status"] == "pending"
            assert saved["remaining_usd"] < 2 - 0.154725
            return super().create(**arguments)

    client = InspectClient(usage=False)
    ledger.perform(args, "scene", allow_paid=True, client=client)
    assert ledger.snapshot()["experiment_conservative_spend_usd"] == reservation_usd(args)


def test_budget_rejection_happens_before_client_call(tmp_path):
    ledger = DurableLedger(tmp_path / "ledger")
    args = _args(tmp_path)
    args["input"][0]["content"][0]["text"] = "x" * 16000000
    client = FakeClient()
    with pytest.raises(BudgetError):
        ledger.perform(args, "scene", allow_paid=True, client=client)
    assert client.calls == 0
    assert ledger.snapshot()["api_attempts"] == 0
    with pytest.raises(BudgetError):
        DurableLedger(tmp_path / "other", prior_spend=0)


def test_twelve_thousand_token_allowance_is_reserved_before_call_and_old_cache_survives(tmp_path):
    ledger = DurableLedger(tmp_path / "ledger")
    old_args = _args(tmp_path)
    image = tmp_path / "numbered.png"
    new_args = request_arguments(image, "New held-out scene", max_output_tokens=12000)
    same_text_new_allowance = request_arguments(image, old_args["input"][0]["content"][0]["text"], 12000)
    assert reservation_usd(same_text_new_allowance) - reservation_usd(old_args) == pytest.approx(.003)
    old_client = FakeClient()
    old_response = ledger.perform(old_args, "development-scene", allow_paid=True, client=old_client)
    restarted = DurableLedger(tmp_path / "ledger")
    assert restarted.perform(old_args, "development-scene", client=old_client) == old_response
    assert old_client.calls == 1

    class InspectClient(FakeClient):
        def create(self, **arguments):
            saved = json.loads(ledger.path.read_text())
            assert saved["requests"]["evaluation-scene"]["reservation_usd"] == reservation_usd(new_args)
            assert saved["requests"]["evaluation-scene"]["status"] == "pending"
            assert arguments["max_output_tokens"] == 12000
            assert arguments["reasoning"] == {"effort": "high"}
            assert arguments["store"] is False
            return super().create(**arguments)

    restarted.perform(new_args, "evaluation-scene", allow_paid=True, client=InspectClient())
    assert restarted.snapshot()["api_attempts"] == 2


def test_request_forwards_explicit_output_allowance_without_loading_secrets(tmp_path, monkeypatch):
    import vlm_tamp.rekep_libero.api as api
    image = tmp_path / "numbered.png"
    image.write_bytes(b"numbered scene")
    captured = {}

    def fake_perform(self, arguments, scene_id, **kwargs):
        captured.update(arguments)
        return {"status": "completed", "output": [], "model": "gpt-6-luna"}

    monkeypatch.setattr(api.DurableLedger, "perform", fake_perform)
    api.request(image, "instruction", tmp_path,
                {"ledger_dir": tmp_path / "ledger", "max_output_tokens": 12000, "allow_paid": False})
    assert captured["max_output_tokens"] == 12000
    assert captured["store"] is False
    assert (tmp_path / "response.json").exists()


@pytest.mark.parametrize("limit", [True, 6000.0, 6001, 12001, 100000])
def test_output_allowance_above_policy_or_noninteger_never_admitted(tmp_path, limit):
    image = tmp_path / "numbered.png"
    image.write_bytes(b"image")
    with pytest.raises(ApiError):
        request_arguments(image, "instruction", limit)
    arguments = _args(tmp_path)
    arguments["max_output_tokens"] = limit
    client = FakeClient()
    with pytest.raises(ApiError):
        DurableLedger(tmp_path / "ledger").perform(arguments, "scene", allow_paid=True, client=client)
    assert client.calls == 0


def test_report_uses_all_planned_predicates_and_excludes_development(tmp_path):
    evaluation = tmp_path / "libero_goal-08-state001-seed0"
    development = tmp_path / "dev-libero_goal-08-state005-seed0"
    evaluation.mkdir()
    development.mkdir()
    (evaluation / "episode.json").write_text(json.dumps({"success": False, "task_type": "pick_place",
                                                          "status": "failed", "failure_stage": "grasp"}))
    (development / "episode.json").write_text(json.dumps({"success": True, "task_type": "pick_place"}))
    report = build_report([evaluation, development], tmp_path / "report", expected_episodes=10)
    assert report["evaluation_planned"] == 10
    assert report["evaluation_successes"] == 0
    assert report["development_count"] == 1
    assert report["failures"] == {"grasp": 1, "not_run": 9}
    assert "0/10" in (tmp_path / "report/index.html").read_text()


def test_report_rejects_plausibility_as_success(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "episode.json").write_text(json.dumps({"success": "optimizer_success"}))
    with pytest.raises(ValueError, match="boolean LIBERO success"):
        build_report([case], tmp_path / "report", expected_episodes=1)
