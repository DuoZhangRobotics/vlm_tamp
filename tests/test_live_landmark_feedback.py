"""Live feedback must preserve estimation and execution semantics."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from vlm_tamp.compact_robot_interface import compact_response
from vlm_tamp.rigid_hinge_tracking import RGBDHingeTracker
from vlm_tamp.subscription_tasks import task_spec
from test_scene_motion_observation import setup
from test_robot_sequence import service


def empty_tracker():
    tracker = RGBDHingeTracker.__new__(RGBDHingeTracker)
    tracker.references = {}
    tracker.reference_step = 12
    tracker.prior = np.zeros(3)
    return tracker


def test_optional_summary_preserves_exact_fit_and_reports_empty_evidence():
    tracker = empty_tracker()
    baseline = tracker.observe({}, control_step=20)
    enhanced = tracker.observe({}, control_step=20, include_displacement_evidence=True)
    summary = enhanced.pop('landmark_displacements')
    assert enhanced == baseline
    assert summary['finite_pairs'] == 0
    assert summary['xy_displacement_mm']['median'] is None
    assert summary['reference_control_step'] == 12 and summary['control_step'] == 20
    assert not summary['simulator_object_state_used']


def test_summary_failure_does_not_discard_original_fit(monkeypatch):
    tracker = empty_tracker()
    baseline = tracker.observe({}, control_step=20)
    def broken(*args):
        raise ValueError('Synthetic report failure')
    monkeypatch.setattr('vlm_tamp.rigid_hinge_tracking.summarize_pairs', broken)
    result = tracker.observe({}, control_step=20, include_displacement_evidence=True)
    evidence = result.pop('landmark_displacements')
    assert result == baseline
    assert evidence['available'] is False and evidence['error_type'] == 'ValueError'


@pytest.mark.parametrize('accepted', [True, False])
def test_live_execution_delivers_summary_without_changing_motion_contract(tmp_path, monkeypatch, accepted):
    module, robot, scene, transform = setup(tmp_path, monkeypatch, accepted=accepted)
    calls = []
    summary = {'finite_pairs': 90, 'near_stationary_xy_pairs': 88, 'xy_displacement_mm': {'median': 0},
               'simulator_object_state_used': False}
    fit = {'pose_accepted': accepted, 'rotation_xy': transform[:2, :2],
           'translation_xy_m': transform[:2, 3], 'landmark_displacements': summary}
    def observe(*args, **kwargs):
        calls.append(kwargs)
        return fit
    monkeypatch.setattr(module, 'RGBDHingeTracker', lambda *args, **kwargs:
                        SimpleNamespace(observe=observe, last_displacement_evidence_seconds=.001))
    monkeypatch.setattr(module, 'hand_motion', lambda *args,**kwargs: (transform, .12))
    robot.landmark_evidence = True
    result = robot.issue(np.zeros((5, 7)), 0, 'arc', articulated=True)
    assert calls[0]['include_displacement_evidence']
    assert result['execution']['issued_commands'] == 5
    key='scene_pose_update' if accepted else 'scene_pose_observation'
    assert result[key]['source'] == ('rgbd_rigid_motion' if accepted else 'unresolved_rgbd_motion')
    assert result[key]['fit']['landmark_displacements'] == summary
    raw = copy.deepcopy(result)
    display = compact_response({'ok': True, 'result': result})
    assert display['result'][key]['fit']['landmark_displacements'] == summary
    assert result.keys() == raw.keys()


def test_explicit_tracking_summary_survives_compact_display():
    summary = {'finite_pairs': 12, 'near_stationary_xy_pairs': 12, 'xy_displacement_mm': {'median': 0}}
    response = {'ok': True, 'result': {'measurement': {'pose_accepted': False, 'landmark_displacements': summary}}}
    assert compact_response(response)['result']['measurement']['landmark_displacements'] == summary


def test_sequence_stage_summary_is_not_dropped_by_compact_display():
    summary = {'finite_pairs': 90, 'near_stationary_xy_pairs': 88}
    response = {'ok': True, 'result': {'stages': [
        {'stage': 'approach', 'task_success': False},
        {'stage': 'manipulation', 'task_success': False, 'scene_pose_update': {
            'source': 'hand_motion_prediction', 'fit': {'landmark_displacements': summary}}}]}}
    stages = compact_response(response)['result']['stages']
    assert 'landmark_displacements' not in stages[0]
    assert stages[1]['landmark_displacements'] == summary
    assert stages[1]['scene_pose_source'] == 'hand_motion_prediction'


def test_task_selection_changes_instruction_and_motion_without_geometry():
    opened, closed = task_spec(), task_spec('close_microwave')
    assert opened['task_name'] == 'KITCHEN_SCENE7_open_the_microwave'
    assert closed['task_name'] == 'KITCHEN_SCENE6_close_the_microwave'
    assert opened['motion'] == 'open' and closed['motion'] == 'close'
    assert set(opened) == set(closed) == {'suite', 'task_name', 'instruction', 'motion'}
    closed['motion'] = 'changed'
    assert task_spec('close_microwave')['motion'] == 'close'
    with pytest.raises(ValueError, match='Unsupported'):
        task_spec('drawer')


def test_closing_grounding_uses_task_direction_and_selected_sensor_regions(tmp_path, monkeypatch):
    module, robot = service(tmp_path)
    robot.task_spec = task_spec('close_microwave')
    robot.sim = object(); robot.cameras = {'frontview': {'rgb': np.zeros((512, 512, 3), dtype=np.uint8)}}
    robot.rebuild = lambda: None
    robot.export_geometry = lambda evidence: robot.spec
    seen = []
    def fit(cameras, choices, template, metadata):
        seen.append(choices)
        return SimpleNamespace(spec=template), np.zeros((10, 3)), None, [], {'cylinder_radius_m': .01}
    monkeypatch.setattr(module, 'ground_handle', fit)
    result = robot.ground({'camera': 'frontview', 'contact_box_xyxy': [70, 20, 80, 80],
        'moving_panel_box_xyxy': [10, 10, 90, 90], 'hinge_pixels_xy': [[10, 10], [10, 90]],
        'initial_image_direction_xy': None})
    assert seen[0]['motion'] == result['motion'] == 'close'
    assert result['instruction'] == 'close the microwave'
