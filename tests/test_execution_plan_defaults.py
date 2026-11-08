"""Planning estimates and omitted execution parameters must describe the same command."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from serve_subscription_robot import RobotService
from vlm_tamp.timed_execution import schedule


def planned_service(tmp_path: Path, speed: float = 1.0, hold: int = 0) -> tuple[RobotService, dict, list]:
    """Build a real stored grasp plan with a deterministic numerical-selector fixture."""
    service = RobotService.__new__(RobotService)
    service.workspace = tmp_path
    service.public = tmp_path / 'robot_data'
    (service.public / 'attempt01').mkdir(parents=True)
    service.attempt = 1
    service.calls = 1
    service.epoch = 0
    service.plans = {}
    service.state = {'control_steps': 557, 'maximum_control_steps': 600}
    service.config = {}
    service.contact_scene_kind = 'surface'
    service.episode_status = lambda: {'remaining_wall_seconds': 120.}
    path = np.zeros((2, 7))
    path[:, 3] = -1.5
    path[1, 0] = 1.2
    count = len(schedule(path, maximum_joint_speed=speed, endpoint_hold_steps=hold))
    service._select_contact = lambda *args: {
        'approach': path, 'pose_world_xyzw': [0, 0, 1, 0, 0, 0, 1], 'source': 'fixture',
        'joints': path[-1].tolist(), 'budget_timing_assumptions': {
            'speed': speed, 'hold': hold, 'approach_controls': count},
    }
    result = service.plan_grasp(speed=speed, approach_speed=speed, hold=hold, close_steps=1)
    sent = []
    service.issue = lambda refs, gripper, label, **kwargs: sent.append((refs, gripper, kwargs))
    return service, result, sent


def test_omitted_execution_timing_matches_the_accepted_plan_budget(tmp_path: Path) -> None:
    """A 24-control approach must not become 75 controls because defaults changed silently."""
    service, plan, sent = planned_service(tmp_path)
    assert plan['budget_timing_assumptions']['approach_controls'] == 24
    service.execute_plan(plan['plan_id'], close_gripper=False, gripper=-1)
    assert len(sent[0][0]) == 24
    assert len(sent[0][0]) <= 600 - service.state['control_steps']


def test_grasp_approach_defaults_to_its_declared_open_gripper(tmp_path: Path) -> None:
    """Omitting a command must not close a gripper whose approach was planned open."""
    service, plan, sent = planned_service(tmp_path)
    service.execute_plan(plan['plan_id'], close_gripper=False)
    assert sent[0][1] == plan['grasp_stage_contract']['approach_gripper_command'] == -1


def test_explicit_execution_overrides_remain_available(tmp_path: Path) -> None:
    """Caller-selected timing and gripper changes remain explicit; add no new veto."""
    service, plan, sent = planned_service(tmp_path)
    service.execute_plan(plan['plan_id'], close_gripper=False, speed=.5, hold=3, gripper=0)
    assert len(sent[0][0]) == 51
    assert sent[0][1] == 0


def test_plans_without_timing_keep_the_existing_executor_defaults() -> None:
    """The repair does not silently retime unrelated free or articulated plans."""
    path = np.zeros((2, 7))
    path[1, 0] = .35
    sent = []
    service = SimpleNamespace(plans={'p': {'kind': 'arc', 'path': path, 'control_steps': 60,
        'epoch': 0, 'extra': None}}, state={'control_steps': 60}, epoch=0,
        issue=lambda refs, gripper, label, **kwargs: sent.append((refs, gripper)))
    RobotService.execute_plan(service, 'p')
    assert len(sent[0][0]) == 26 and sent[0][1] == 1


@pytest.mark.parametrize('speed,hold', [(None, 2), (.5, None)])
def test_one_override_does_not_discard_the_other_planned_setting(
    tmp_path: Path, speed: float | None, hold: int | None
) -> None:
    """Speed and endpoint hold inherit independently."""
    service, plan, sent = planned_service(tmp_path, hold=4)
    overrides = {name: value for name, value in [('speed', speed), ('hold', hold)] if value is not None}
    service.execute_plan(plan['plan_id'], close_gripper=False, gripper=-1, **overrides)
    assert len(sent[0][0]) == (26 if speed is None else 52)


def test_existing_budget_rejection_reports_requested_and_remaining_controls() -> None:
    """Keep the original admission rule but make its numerical reason visible."""
    service = SimpleNamespace(state={'control_steps': 557, 'maximum_control_steps': 600})
    with pytest.raises(ValueError, match='74 references requested, 43 controls remain'):
        RobotService.issue(service, np.zeros((74, 7)), -1, 'fixture')


def grasp_execution_service(tmp_path: Path, *, reached: bool = True) -> tuple[RobotService, dict, list]:
    """Use real planning/execution composition with measured controller replies."""
    service, plan, sent = planned_service(tmp_path)
    service.state['metadata'] = {'joint_servo': {'reference_joint_positions': [0.] * 7}}
    service.plans[plan['plan_id']]['extra']['grasp_close_steps'] = 20

    def issue(refs: np.ndarray, gripper: int, label: str, **kwargs: object) -> dict:
        """Advance controls and expose arrival independently of finger closure."""
        sent.append({'gripper': gripper, 'label': label, 'count': len(refs), 'artifact': service.calls})
        service.state['control_steps'] += len(refs)
        execution = {'planned_commands': len(refs), 'issued_commands': len(refs),
                     'all_commands_issued': True, 'termination_reason': 'trajectory_complete'}
        if 'arrival' in kwargs:
            execution['arrival'] = {'goal_reached': reached, 'goal': {'reached': reached}}
        return {'execution': execution, 'observation': {'control_steps': service.state['control_steps']},
                'grasp_evidence': {'status': 'unverified'}, 'task_success': False}

    service.issue = issue
    return service, plan, sent


def test_execute_grasp_closes_after_arrival_without_another_tool_call(tmp_path: Path) -> None:
    """The composed executor preserves an open approach and distinct physical closure."""
    service, plan, sent = grasp_execution_service(tmp_path)
    service.state['maximum_control_steps'] = 620
    result = service.execute_plan(plan['plan_id'])
    assert [stage['gripper'] for stage in sent] == [-1, 1]
    assert sent[1]['count'] == 20
    assert sent[0]['artifact'] != sent[1]['artifact']
    assert result['execution']['issued_commands'] == 44
    assert result['execution']['all_commands_issued'] is True
    assert result['grasp_execution']['closure_completed'] is True
    assert result['grasp_execution']['object_retention'] == 'unverified'
    assert result['task_success'] is False
    assert [stage['control_steps'] for stage in result['stages']] == [581, 601]
    assert result['observation']['control_steps'] == 601


def test_blocked_approach_does_not_issue_closure(tmp_path: Path) -> None:
    """An accepted path is not measured arrival at its grasp endpoint."""
    service, plan, sent = grasp_execution_service(tmp_path, reached=False)
    result = service.execute_plan(plan['plan_id'])
    assert len(sent) == 1
    assert result['stop_reason'] == 'approach_not_reached'
    assert result['grasp_execution']['closure_issued'] is False
    assert result['execution']['all_commands_issued'] is False


def test_closure_uses_remaining_budget_and_reports_partial_execution(tmp_path: Path) -> None:
    """Do not reserve extra controls or silently exceed the episode limit."""
    service, plan, sent = grasp_execution_service(tmp_path)
    result = service.execute_plan(plan['plan_id'])
    assert sent[1]['count'] == 19
    assert service.state['control_steps'] == 600
    assert result['grasp_execution']['executed_close_controls'] == 19
    assert result['grasp_execution']['closure_completed'] is False
    assert result['stop_reason'] == 'control_budget_exhausted'


@pytest.mark.parametrize('reason', ['simulator_success', 'wall_budget_exhausted', 'control_budget_exhausted'])
def test_terminal_approach_never_starts_another_stage(tmp_path: Path, reason: str) -> None:
    """Task completion and episode deadlines apply between the two stages."""
    service, plan, sent = grasp_execution_service(tmp_path)
    service.episode_status = lambda: {'episode_done': True, 'termination_reason': reason}
    result = service.execute_plan(plan['plan_id'])
    assert len(sent) == 1
    assert result['stop_reason'] == reason
    assert result['grasp_execution']['closure_issued'] is False


def test_non_grasp_plan_cannot_silently_gain_a_closure_duration(tmp_path: Path) -> None:
    """Manual gripper control remains the explicit interface for ordinary pose plans."""
    service, plan, sent = grasp_execution_service(tmp_path)
    del service.plans[plan['plan_id']]['extra']['grasp_close_steps']
    with pytest.raises(ValueError, match='requires a plan_grasp plan'):
        service.execute_plan(plan['plan_id'], close_gripper=True)
    assert not sent
