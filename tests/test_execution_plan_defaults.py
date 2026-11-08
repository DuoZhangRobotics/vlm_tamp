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
    service.execute_plan(plan['plan_id'], gripper=-1)
    assert len(sent[0][0]) == 24
    assert len(sent[0][0]) <= 600 - service.state['control_steps']


def test_grasp_approach_defaults_to_its_declared_open_gripper(tmp_path: Path) -> None:
    """Omitting a command must not close a gripper whose approach was planned open."""
    service, plan, sent = planned_service(tmp_path)
    service.execute_plan(plan['plan_id'])
    assert sent[0][1] == plan['grasp_stage_contract']['approach_gripper_command'] == -1


def test_explicit_execution_overrides_remain_available(tmp_path: Path) -> None:
    """Caller-selected timing and gripper changes remain explicit; add no new veto."""
    service, plan, sent = planned_service(tmp_path)
    service.execute_plan(plan['plan_id'], speed=.5, hold=3, gripper=0)
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
    service.execute_plan(plan['plan_id'], gripper=-1, **overrides)
    assert len(sent[0][0]) == (26 if speed is None else 52)


def test_existing_budget_rejection_reports_requested_and_remaining_controls() -> None:
    """Keep the original admission rule but make its numerical reason visible."""
    service = SimpleNamespace(state={'control_steps': 557, 'maximum_control_steps': 600})
    with pytest.raises(ValueError, match='74 references requested, 43 controls remain'):
        RobotService.issue(service, np.zeros((74, 7)), -1, 'fixture')
