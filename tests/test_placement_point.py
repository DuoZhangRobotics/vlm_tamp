"""An object goal must preserve its offset without becoming a robot-contact or success claim."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vlm_tamp.placement_point import object_point_binding, plan_object_point
from vlm_tamp.rekep_libero.geometry import matrix_pose, pose_matrix, transform_points


def measured_state() -> dict:
    """Use a translated, rotated hand so treating the object point as the TCP fails."""
    return {'eef_pose_world_xyzw': [.2, -.1, 1., *Rotation.from_euler('xyz', [20, 50, 70], degrees=True).as_quat()],
            'joint_positions': [0.] * 7}


def test_point_binding_is_equivariant_under_world_frame_changes() -> None:
    """Changing world coordinates cannot change a rigid material point's TCP offset."""
    metadata = measured_state()
    source = [.3, -.05, .95]
    target = [.4, .25, .8]
    bound = object_point_binding(metadata, source, target, 'rigid')
    change = np.eye(4)
    change[:3, :3] = Rotation.from_euler('y', 31, degrees=True).as_matrix()
    change[:3, 3] = [-.3, .8, .2]
    shifted = {**metadata, 'eef_pose_world_xyzw': matrix_pose(change @ pose_matrix(metadata['eef_pose_world_xyzw']))}
    other = object_point_binding(shifted, *transform_points([source, target], change), 'rigid')
    np.testing.assert_allclose(other['contact_local_m'], bound['contact_local_m'], atol=1e-12)
    np.testing.assert_allclose(transform_points([bound['contact_local_m']], pose_matrix(metadata['eef_pose_world_xyzw']))[0], source)
    assert bound['attachment_verified'] is False
    assert bound['payload_collision_checked'] is False


def test_goal_point_requires_offset_correction_when_hand_rotates() -> None:
    """The correct hand translation is destination minus the rotated object offset."""
    bound = object_point_binding(measured_state(), [.3, -.05, .95], [.4, .25, .8], 'rigid')
    rotation = Rotation.from_euler('z', 130, degrees=True).as_matrix()
    local = np.asarray(bound['contact_local_m'])
    target = np.asarray(bound['target_point_world_m'])
    translation = target - rotation @ local
    np.testing.assert_allclose(translation + rotation @ local, target)
    assert np.linalg.norm(translation - target) > .05


def test_attachment_model_and_finite_points_are_explicit() -> None:
    """Do not silently repair an unsupported attachment or nonfinite geometry."""
    with pytest.raises(ValueError, match='attachment_model'):
        object_point_binding(measured_state(), [0, 0, 0], [0, 0, 1], 'sliding')
    with pytest.raises(ValueError, match='finite'):
        object_point_binding(measured_state(), [np.nan, 0, 0], [0, 0, 1], 'rigid')


@pytest.mark.parametrize('solved', [True, False])
def test_uses_existing_point_ik_and_native_path_without_alternate_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, solved: bool,
) -> None:
    """Pass the object offset unchanged; retain native failure instead of substituting a pose."""
    import vlm_tamp.surface_contact_tools as shared

    calls = []
    class Context:
        def __init__(self, *args: object, **kwargs: object) -> None:
            """Stand in for the already-tested native collision context."""
            pass
        def __enter__(self) -> 'Context':
            """Keep the test context alive for the query."""
            return self
        def __exit__(self, *args: object) -> None:
            """Leave numerical failures visible."""
            pass
        def plan(self, start: np.ndarray, end: list, **kwargs: object) -> tuple[np.ndarray, dict]:
            """Return the native path result without a replacement route."""
            calls.append('path')
            return np.array([start, end]), {'exact_solution': True}

    metadata = measured_state()
    def solve(target: list, local: list, seed: np.ndarray, rotation: np.ndarray) -> list:
        """Record the exact point request delivered to native IK."""
        calls.append({'target': target, 'local': local})
        return [{'joints': [.1] * 7}] if solved else []

    ik = SimpleNamespace(events=[], configure_scene=lambda context: None, solve_contact_many=solve,
                         chain=SimpleNamespace(forward=lambda joints: np.asarray(metadata['eef_pose_world_xyzw'])))
    monkeypatch.setattr(shared, 'NativeScene', Context)
    arguments = (ik, object(), {}, metadata, [.3, -.05, .95], [.4, .25, .8], 'rigid', tmp_path/'plan')
    if solved:
        _, report = plan_object_point(*arguments)
        assert calls[-1] == 'path'
        assert report['grasp_required'] is True
        assert report['object_point_goal']['attachment_verified'] is False
    else:
        with pytest.raises(ValueError) as caught:
            plan_object_point(*arguments)
        assert caught.value.planning_diagnostic['status'] == 'object_point_ik_failed'
        assert len(calls) == 1
    expected = object_point_binding(metadata, [.3, -.05, .95], [.4, .25, .8], 'rigid')
    np.testing.assert_array_equal(calls[0]['local'], expected['contact_local_m'])


def test_registered_tool_and_compact_feedback_preserve_attachment_uncertainty() -> None:
    """Luna can call the real schema and sees the assumption in its compact result."""
    from vlm_tamp.actor_structured import OPERATIONS
    from vlm_tamp.compact_robot_interface import compact_response
    from vlm_tamp.robot_contracts import validate_operation

    args = {'source_point_world_m': [0, 0, 1], 'target_point_world_m': [.2, 0, 1], 'attachment_model': 'rigid'}
    validate_operation('plan_object_point', args)
    assert 'plan_object_point' in OPERATIONS
    binding = object_point_binding(measured_state(), **args)
    reply = compact_response({'ok': True, 'result': {'object_point_goal': binding}})
    assert reply['result']['object_point_goal'] == binding


def test_execution_reports_predicted_object_point_and_holds_gripper(tmp_path: Path) -> None:
    """Robot FK arrival must not be mislabeled as observed object arrival."""
    from serve_subscription_robot import RobotService
    service = RobotService.__new__(RobotService)
    service.state = {'control_steps': 60}
    service.epoch = 0
    binding = object_point_binding(measured_state(), [.3, -.05, .95], [.4, .25, .8], 'rigid')
    service.plans = {'p': {'kind': 'free', 'control_steps': 60, 'epoch': 0,
        'path': np.zeros((2, 7)), 'extra': {'required_gripper_command': 0, 'object_point_goal': binding,
        'arrival': {'contact_local_m': binding['contact_local_m'], 'maximum_wait_steps': 20}}}}
    sent = []
    def issue(refs: np.ndarray, gripper: int, label: str, **kwargs: object) -> dict:
        """Expose robot-only arrival evidence without simulated object sensing."""
        sent.append((gripper, kwargs))
        return {'execution': {'arrival': {'goal': {'reached': True}}}, 'task_success': False}
    service.issue = issue
    result = service.execute_plan('p')
    assert sent[0][0] == 0
    assert sent[0][1]['arrival']['contact_local_m'] == binding['contact_local_m']
    assert 'assumed_attachment' in result['execution']['arrival']['goal']['tracked_point']
    assert result['task_success'] is False
