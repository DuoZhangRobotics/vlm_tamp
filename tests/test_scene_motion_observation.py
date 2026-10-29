"""Scene estimates must account for observed motion outside a planned arc."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_robot_sequence import service
from vlm_tamp.coupled_scene import Scene


def setup(tmp_path, monkeypatch, *, accepted):
    module, robot = service(tmp_path)
    scene = Scene.from_points([[3, 2, 1]], [[1, 0, 1], [1, .1, 1]], [[1.1, 0, 1]],
        {'joint_origin_world_m': [0, 0, 0], 'joint_axis_world': [0, 0, 1],
         'front_plane': {'point_world_m': [1, 0, 1]}, 'panel_normal_world': [1, 0, 0],
         'source_anchor_world_m': [1.1, 0, 1]})
    transform = scene.transform(.12)
    fit = {'pose_accepted': accepted, 'rotation_xy': transform[:2, :2],
           'translation_xy_m': transform[:2, 3], 'reason': 'accepted' if accepted else 'insufficient_moving_landmarks'}
    robot.config = {'scene_pose_sync': 'rgbd'}
    robot.scene = scene
    robot.cameras = {}
    robot.sim = object()
    robot.events, robot.executions, robot.anchors = [], [], [[1.1, 0, 1]]
    robot.tracker = None
    robot.ik = SimpleNamespace(chain=None)
    robot.stop_on_success = False
    robot.capture = lambda: {'control_steps': robot.state['control_steps']}
    robot.safe_scene_evidence = lambda: {'available': True}
    monkeypatch.setattr(module, 'RGBDHingeTracker', lambda *a, **k:
                        SimpleNamespace(observe=lambda *a, **k: fit))
    monkeypatch.setattr(module, 'execute_batch', lambda *a, **k:
                        ({'control_steps': 5, 'metadata': {'joint_positions': [0.] * 7}},
                         {'issued_commands': 5, 'planned_commands': 5, 'all_commands_issued': True}))
    return module, robot, scene, transform


@pytest.mark.parametrize('gripper', [-1, 0, 1])
def test_observed_object_motion_after_ordinary_commands_updates_only_moving_geometry(tmp_path, monkeypatch, gripper):
    module, robot, scene, transform = setup(tmp_path, monkeypatch, accepted=True)
    monkeypatch.setattr(module, 'hand_motion', lambda *a,**k: pytest.fail('Ordinary motion cannot assume attachment'))
    result = robot.issue(np.zeros((5, 7)), gripper, 'ordinary')
    expected = scene.moving_reference @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(robot.scene.moving_reference, expected)
    np.testing.assert_array_equal(robot.scene.static_points, scene.static_points)
    assert robot.epoch == 1
    assert result['scene_pose_update']['source'] == 'rgbd_rigid_motion'
    assert not result['scene_pose_update']['assumes_contact_retention']


def test_unconfirmed_ordinary_motion_never_moves_scene_by_hand_prediction(tmp_path, monkeypatch):
    module, robot, scene, _ = setup(tmp_path, monkeypatch, accepted=False)
    monkeypatch.setattr(module, 'hand_motion', lambda *a,**k: pytest.fail('No unobserved attachment assumption'))
    result = robot.issue(np.zeros((5, 7)), -1, 'ordinary')
    assert robot.scene is scene and robot.epoch == 0
    assert not result['scene_pose_observation']['scene_changed']
    assert not result['scene_pose_observation']['assumes_contact_retention']
    assert result['execution']['issued_commands'] == 5


def test_unconfirmed_contact_arc_keeps_observed_scene_instead_of_assuming_attachment(tmp_path, monkeypatch):
    module, robot, scene, transform = setup(tmp_path, monkeypatch, accepted=False)
    monkeypatch.setattr(module, 'hand_motion', lambda *a,**k: (transform, .12))
    result = robot.issue(np.zeros((5, 7)), 0, 'arc', articulated=True)
    assert robot.scene is scene and robot.epoch == 0
    assert not result['scene_pose_observation']['scene_changed']
    assert result['motion_effect']['status']=='object_motion_unresolved'
    assert result['execution']['issued_commands']==5


def test_surface_contact_scene_does_not_acquire_an_articulation_model(tmp_path, monkeypatch):
    module, robot, scene, _ = setup(tmp_path, monkeypatch, accepted=True)
    robot.contact_scene_kind = 'surface'
    monkeypatch.setattr(module, 'RGBDHingeTracker', lambda *a, **k: pytest.fail('Surface has no hinge model'))
    result = robot.issue(np.zeros((5, 7)), 0, 'ordinary')
    assert robot.scene is scene and robot.epoch == 0
    assert 'scene_pose_update' not in result


def test_stale_part_geometry_does_not_seed_background_tracking(tmp_path,monkeypatch):
    module,robot,scene,transform=setup(tmp_path,monkeypatch,accepted=True)
    robot.scene_reference_control_step=-1
    monkeypatch.setattr(module,'RGBDHingeTracker',lambda *a,**k:pytest.fail('Stale object pose must not select current tracking pixels'))
    monkeypatch.setattr(module,'hand_motion',lambda *a,**k:(transform,.12))
    result=robot.issue(np.zeros((5,7)),0,'arc',articulated=True)
    assert robot.scene is scene and robot.scene_reference_control_step==-1
    assert result['motion_effect']['status']=='object_motion_unresolved'
