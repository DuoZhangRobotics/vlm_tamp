"""Withheld native feedback is unknown, not a negative task evaluation."""

from pathlib import Path

import pytest

from serve_subscription_robot import RobotService


@pytest.mark.parametrize('native_feedback', [False, True])
def test_episode_status_distinguishes_withheld_feedback(tmp_path: Path, native_feedback: bool) -> None:
    """Success telemetry must respect the declared information contract."""
    service=RobotService.__new__(RobotService)
    service.output=tmp_path
    service.state={'control_steps':200,'maximum_control_steps':600}
    service.stop_on_success=native_feedback
    service.episode_success=True
    service.attempt=1
    result=service.episode_status()
    assert result['task_success_available'] is native_feedback
    assert result['task_success'] is (True if native_feedback else None)
    assert result['episode_done'] is native_feedback


def test_budget_exhaustion_does_not_reveal_withheld_native_outcome(tmp_path: Path) -> None:
    """An exhausted control budget terminates a blind episode without inventing failure."""
    service=RobotService.__new__(RobotService)
    service.output=tmp_path
    service.state={'control_steps':600,'maximum_control_steps':600}
    service.stop_on_success=False
    service.episode_success=False
    service.attempt=1
    result=service.episode_status()
    assert result['episode_done']
    assert result['termination_reason']=='control_budget_exhausted'
    assert result['task_success'] is None
