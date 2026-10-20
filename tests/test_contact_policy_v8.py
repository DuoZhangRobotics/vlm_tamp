"""Behavioral checks for task-specific touch pairs, including negative controls."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull

from vlm_tamp.rekep_libero.contact_policy_v8 import ContactPolicy
from vlm_tamp.rekep_libero.scene_state_v7 import hull_volume
from vlm_tamp.rekep_libero.scene_state_v8 import ObservedScene


TCP = [0, 0, 1, 0, 0, 0, 1]
DRAWER = {'task_type': 'open_middle_drawer', 'front_plane': {'point_world_m': [0, 0, 1]},
          'pull_axis_world': [0, 1, 0]}
BOWL = {'task_type': 'pick_place'}


def box(low, high, name):
    vertices = np.array([[x, y, z] for x in [low[0], high[0]] for y in [low[1], high[1]]
                         for z in [low[2], high[2]]])
    return {'type': 'convex', 'fixed': False, 'robot_geom_name': name,
            'vertices_world': vertices, 'planes_world': ConvexHull(vertices).equations}


FINGER = box([-.01, -.01, .99], [.01, .01, 1.01], 'gripper0_finger1_collision')
PAD = box([-.01, -.01, .99], [0, .01, 1.01], 'gripper0_finger1_pad_collision')
PALM = box([-.01, -.01, .99], [.01, .01, 1.01], 'gripper0_hand_collision')
ARM = box([-.01, -.01, .99], [.01, .01, 1.01], 'robot0_link4_collision')


@pytest.mark.parametrize('stage', ['grasp', 'carry', 'pull', 'place', 'release', 'retreat_contact'])
def test_selected_bowl_contact_includes_finger_mesh_not_only_pad(stage):
    policy = ContactPolicy([FINGER, PAD], BOWL, {})
    # On the functional finger outside the rubber pad, still on the selected bowl.
    target = np.array([[.008, 0, 1.]])
    assert policy.robot_allowed(target, FINGER, TCP, target, stage).all()


@pytest.mark.parametrize('volume', [PALM, ARM])
def test_selected_object_does_not_grant_palm_or_arm_permission(volume):
    policy = ContactPolicy([FINGER, PAD, PALM, ARM], BOWL, {})
    target = np.array([[0., 0., 1.]])
    assert not policy.robot_allowed(target, volume, TCP, target, 'grasp').any()


@pytest.mark.parametrize('stage', ['approach', 'released'])
def test_target_permission_requires_contact_phase(stage):
    policy = ContactPolicy([FINGER], BOWL, {})
    target = np.array([[0., 0., 1.]])
    assert not policy.robot_allowed(target, FINGER, TCP, target, stage).any()


def test_adjacent_handle_or_object_does_not_inherit_selected_target_permission():
    policy = ContactPolicy([FINGER], DRAWER, {})
    target = np.array([[0., .02, 1.]])
    other = np.array([[0., .02, 1.08], [.04, .02, 1.]])
    assert not policy.robot_allowed(other, FINGER, TCP, target, 'grasp').any()


def test_local_drawer_panel_bracing_does_not_allow_neighbor_handle_or_far_panel():
    policy = ContactPolicy([FINGER, PALM, ARM], DRAWER, {})
    selected_handle = np.array([[0., .02, 1.]])
    points = np.array([[.015, 0, 1.025], [0, .02, 1.07], [.15, 0, 1.], [0, -.03, 1.]])
    assert policy.local_panel_allowed(points, FINGER, selected_handle, 'grasp', np.eye(4)).tolist() == [True, False, False, False]
    assert not policy.local_panel_allowed(points, PALM, selected_handle, 'grasp', np.eye(4)).any()
    assert not policy.local_panel_allowed(points, ARM, selected_handle, 'pull', np.eye(4)).any()
    assert not policy.local_panel_allowed(points, FINGER, selected_handle, 'approach', np.eye(4)).any()


def test_local_bracing_follows_observed_drawer_displacement():
    policy = ContactPolicy([FINGER], DRAWER, {})
    transform = np.eye(4)
    transform[1, 3] = .10
    handle = np.array([[0., .12, 1.]])
    points = np.array([[.01, .10, 1.01], [.01, 0, 1.01]])
    assert policy.local_panel_allowed(points, FINGER, handle, 'pull', transform).tolist() == [True, False]


def test_support_permission_is_pair_specific_and_phase_specific():
    policy = ContactPolicy([FINGER], BOWL, {})
    selected_plate = np.array([[0., 0., 1.], [.03, 0., 1.01]])
    another_object = np.array([[.06, 0., 1.]])
    points = np.concatenate([selected_plate, another_object])
    assert policy.support_allowed(points, [], selected_plate, 'place').tolist() == [True, True, False]
    assert not policy.support_allowed(points, [], selected_plate, 'carry').any()
    assert policy.support_allowed(points, [], selected_plate, 'carry', initial=True).tolist() == [True, True, False]


def scene_fixture(static, moving, spec=BOWL):
    scene = ObservedScene.__new__(ObservedScene)
    scene.region = SimpleNamespace(contact_reference=np.asarray(moving), task_type=spec['task_type'],
        axis=np.array([0., 1., 0.]), origin=np.array([0., 0., 1.]), side=np.array([1., 0., 0.]),
        z_low=.99, z_high=1.01, side_low=-.01, side_high=.01, initial_support=np.empty((0, 3)))
    scene.static_points, scene.static_tree = np.asarray(static), cKDTree(static)
    scene.moving_reference = np.asarray(moving)
    scene.policy = ContactPolicy([FINGER, PAD, PALM, ARM], spec, {})
    scene.config = {}
    scene.required_robot_parts = frozenset()
    scene.observed_transform, scene.margin, scene.revision = np.eye(4), .001, 0
    scene.target_points = np.empty((0, 3))
    scene.payload_volume = hull_volume(moving, 'source')
    return scene


def moving_box():
    return np.array([[x, y, z] for x in [-.003, .009] for y in [-.003, .003] for z in [.997, 1.003]])


def test_real_overlap_with_selected_bowl_is_allowed_but_unrelated_stationary_surface_blocks():
    scene = scene_fixture([[.008, .002, 1.]], moving_box())
    result = scene.check([FINGER], TCP, stage='grasp')
    assert not result['accepted']
    assert all(row['scene_part'] == 'stationary_scene' for row in result['collisions'])
    assert any(row['rule'] == 'finger_selected_target' for row in result['contact_permissions_applied'])


def test_arm_collision_with_selected_bowl_remains_blocking():
    scene = scene_fixture([[.20, 0, 1.]], moving_box())
    result = scene.check([ARM], TCP, stage='grasp')
    assert not result['accepted']
    assert result['collisions'][0]['scene_part'] == 'moving_source'


def test_positive_finger_clearance_is_not_reported_as_overlap():
    scene = scene_fixture([[.0105, 0, 1.]], moving_box())
    result = scene.check([FINGER], TCP, stage='grasp')
    assert result['accepted']
    assert len(result['nonblocking_positive_clearance']) == 1
    # The same point penetrating the unrelated object still blocks.
    scene.static_points = np.array([[.0095, 0, 1.]])
    scene.static_tree = cKDTree(scene.static_points)
    result = scene.check([FINGER], TCP, stage='grasp')
    assert not result['accepted']
    assert result['collisions'][0]['minimum_signed_point_distance_m'] < 0


def test_rotated_or_lifted_bowl_does_not_use_a_rotated_support_plane(monkeypatch):
    import vlm_tamp.rekep_libero.scene_state_v8 as module
    scene = scene_fixture([[.2, 0, 1.]], moving_box())
    scene.spacing, scene.depth_tolerance = .002, .0001
    # The initial source model did not see the lowest 2 mm of the bowl base.
    reference = np.array([[x, y, z] for x in np.linspace(-.008, .008, 10)
                          for y in np.linspace(-.008, .008, 10) for z in [1.003, 1.01]])
    scene.region.body_tree = cKDTree(reference)
    scene.region.support_height = 1.
    base = reference[:100].copy()
    base[:, 2] = 1.0008
    transform = np.eye(4)
    transform[2, 3] = .019
    observed_source = base + [0, 0, .019]
    floor = np.array([[0., 0., 1.]])
    monkeypatch.setattr(module, 'observed_points', lambda *args, **kwargs: (np.vstack([observed_source, floor]), {}))
    result = scene.refresh({}, {}, transform)
    assert result['current_moving_supported_points'] == len(observed_source)
    assert np.array_equal(scene.static_points, floor)
    assert result['support_plane_frame'] == 'stationary world'


def test_payload_collision_is_not_covered_by_robot_touch_link_permission():
    scene = scene_fixture([[0, 0, 1.]], moving_box())
    result = scene.check([], TCP, stage='carry', check_payload=True)
    assert not result['accepted']
    assert result['collisions'][0]['robot_geom_name'] == 'predicted_moving_source_envelope'


def test_positive_payload_clearance_is_logged_but_does_not_block():
    scene = scene_fixture([[.0095, 0, 1.]], moving_box())
    result = scene.check([], TCP, stage='carry', check_payload=True)
    assert result['accepted']
    assert result['nonblocking_positive_clearance'][0]['minimum_positive_clearance_m'] > 0


def test_payload_can_rest_on_selected_initial_support_then_leave_it():
    point = np.array([[0., 0., 1.]])
    scene = scene_fixture(point, moving_box())
    scene.region.initial_support = point
    assert scene.check([], TCP, stage='carry', check_payload=True)['accepted']
    assert not scene.check([], TCP, stage='place', check_payload=True)['accepted']


def test_missing_arm_geometry_still_raises():
    scene = scene_fixture([[.2, 0, 1.]], moving_box())
    scene.required_robot_parts = frozenset([FINGER['robot_geom_name'], ARM['robot_geom_name']])
    with pytest.raises(ValueError, match='missing configured'):
        scene.check([FINGER], TCP, stage='grasp')
