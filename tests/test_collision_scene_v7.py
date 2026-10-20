"""Contact selectivity, moving-scene and depth-provenance regressions."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from vlm_tamp.rekep_libero.contact_policy_v7 import ContactPolicy
from vlm_tamp.rekep_libero.scene_state_v7 import ObservedScene, hull_volume, observed_points, transformed_volume


def box(low, high, name):
    vertices = np.array([[x, y, z] for x in [low[0], high[0]] for y in [low[1], high[1]]
                         for z in [low[2], high[2]]])
    return {'type': 'convex', 'fixed': False, 'robot_geom_name': name,
            'vertices_world': vertices, 'planes_world': ConvexHull(vertices).equations}


PAD = box([.018, -.01, .99], [.025, .01, 1.01], 'gripper0_finger1_pad_collision')
TCP = [0, 0, 1, 0, 0, 0, 1]


@pytest.mark.parametrize('stage,expected', [('approach', False), ('grasp', True), ('pull', True),
                                         ('carry', True), ('place', True), ('released', False)])
def test_contact_permission_expires_and_is_local(stage, expected):
    policy = ContactPolicy([PAD])
    contact = np.array([[.018, 0, 1.]])
    assert bool(policy.robot_allowed(contact, PAD, TCP, contact, stage)[0]) == expected
    outside = np.array([[.024, 0, 1.]])
    assert not policy.robot_allowed(outside, PAD, TCP, outside, stage).any()
    another_handle = contact + [0, 0, .075]
    assert not policy.robot_allowed(contact, PAD, TCP, another_handle, stage).any()


@pytest.mark.parametrize('name', ['gripper0_hand_collision', 'robot0_link4_collision', 'gripper0_finger1_collision'])
def test_arm_palm_and_outer_fingers_never_receive_pad_permission(name):
    policy = ContactPolicy([PAD])
    point = np.array([[.018, 0, 1.]])
    volume = {**PAD, 'robot_geom_name': name}
    assert not policy.robot_allowed(point, volume, TCP, point, 'pull').any()


def test_bottom_support_permission_cannot_clear_side_collisions():
    policy = ContactPolicy([PAD])
    payload = np.array([[0, 0, 1.], [.04, 0, 1.04]])
    points = np.array([[0, 0, 1.], [0, 0, 1.02]])
    assert policy.support_allowed(points, payload, points, 'place').tolist() == [True, False]
    assert not policy.support_allowed(points, payload, points, 'carry').any()


def camera(points, depth=None, prior=None):
    points = np.asarray(points).reshape(2, 2, 3)
    data = {'world_points': points, 'depth': np.ones((2, 2)) if depth is None else depth,
            'intrinsic': np.eye(3), 'camera_to_world': np.eye(4)}
    if prior is not None:
        data.update(known_robot_depth_prior=prior, robot_depth_prior={
            'source': 'configured robot visual meshes projected at current proprioception only',
            'scene_geometry_used': False, 'units': 'meters', 'depth_convention': 'optical-axis z'})
    return data


def test_render_roundoff_filter_preserves_closer_obstacles():
    points = [[0, 0, 1.], [.01, 0, 1.], [.02, 0, 1.], [.03, 0, 1.]]
    depth = np.array([[1.000065, .999935], [.998, .99]])
    cloud, report = observed_points({'view': camera(points, depth, np.ones((2, 2)))}, {})
    assert len(cloud) == 2
    assert np.allclose(cloud[:, 0], [.02, .03])
    assert report['additional_numerically_matched_robot_endpoints'] == 2
    assert not report['future_robot_sweep_used_for_self_filter']


def test_simulator_instance_depth_prior_is_rejected():
    cam = camera([[0, 0, 1.]] * 4, prior=np.ones((2, 2)))
    cam['robot_depth_prior']['scene_geometry_used'] = True
    with pytest.raises(ValueError, match='only from configured'):
        observed_points({'view': cam}, {})


def artificial_scene(static, moving):
    from scipy.spatial import cKDTree
    from types import SimpleNamespace
    scene = ObservedScene.__new__(ObservedScene)
    scene.region = SimpleNamespace(contact_reference=np.asarray(moving), task_type='open_middle_drawer',
                                   axis=np.array([0., 1., 0.]))
    scene.static_points, scene.static_tree = np.asarray(static), cKDTree(static)
    scene.moving_reference = np.asarray(moving)
    scene.policy = ContactPolicy([PAD])
    scene.observed_transform, scene.margin, scene.revision = np.eye(4), .001, 0
    scene.target_points = np.empty((0, 3))
    scene.payload_volume = hull_volume(moving, 'source')
    return scene


def test_moving_drawer_vacates_old_surface_but_cabinet_stays():
    moving = np.array([[x, y, z] for x in [-.02, .02] for y in [0, .015] for z in [.99, 1.01]])
    cabinet = np.array([[.1, 0, 1.]])
    scene = artificial_scene(cabinet, moving)
    arm = box([-.03, -.005, .98], [.03, .02, 1.02], 'robot0_link4_collision')
    assert not scene.check([arm], TCP)['accepted']
    transform = scene.drawer_transform(.1)
    assert scene.check([arm], TCP, moving_transform=transform)['accepted']
    cabinet_arm = box([.09, -.005, .98], [.11, .02, 1.02], 'robot0_link4_collision')
    result = scene.check([cabinet_arm], TCP, moving_transform=transform)
    assert not result['accepted']
    assert result['collisions'][0]['scene_part'] == 'stationary_scene'


def test_payload_collision_is_checked_even_if_robot_misses_obstacle():
    moving = np.array([[x, y, z] for x in [-.02, .02] for y in [0, .015] for z in [.99, 1.01]])
    scene = artificial_scene([[0., .105, 1.]], moving)
    transform = scene.drawer_transform(.1)
    result = scene.check([], TCP, stage='pull', moving_transform=transform, check_payload=True)
    assert not result['accepted']
    assert result['collisions'][0]['robot_geom_name'] == 'predicted_moving_source_envelope'


def test_transformed_payload_planes_match_rotated_vertices():
    points = np.array([[x, y, z] for x in [-.02, .02] for y in [0, .015] for z in [.99, 1.01]])
    transform = np.array([[0., -1, 0, .1], [1., 0, 0, .2], [0, 0, 1, 0], [0, 0, 0, 1.]])
    volume = transformed_volume(hull_volume(points, 'source'), transform)
    planes, vertices = volume['planes_world'], volume['vertices_world']
    assert np.max(vertices @ planes[:, :3].T + planes[:, 3]) < 1e-9


def test_invalid_drawer_displacement_is_not_silently_clamped():
    points = np.array([[x, y, z] for x in [-.02, .02] for y in [0, .015] for z in [.99, 1.01]])
    scene = artificial_scene([[.1, 0, 1.]], points)
    with pytest.raises(ValueError, match='planning bounds'):
        scene.drawer_transform(.3)


def test_only_finger_mesh_overlap_with_inward_pad_patch_is_exempted():
    policy = ContactPolicy([PAD])
    finger = {**PAD, 'robot_geom_name': 'gripper0_finger1_collision'}
    points = np.array([[.018, 0, 1.], [.018, .02, 1.], [.024, 0, 1.]])
    allowed = policy.robot_allowed(points, finger, TCP, points, 'grasp', [PAD])
    assert allowed.tolist() == [True, False, False]


def test_stationary_obstacle_near_selected_pad_contact_is_not_ignored():
    moving = np.array([[x, y, z] for x in [.018, .03] for y in [-.003, .003] for z in [.998, 1.002]])
    scene = artificial_scene([[.019, 0, 1.]], moving)
    result = scene.check([PAD], TCP, stage='grasp')
    assert not result['accepted']
    assert any(row['scene_part'] == 'stationary_scene' for row in result['collisions'])


def test_missing_arm_geometry_cannot_silently_pass():
    points = np.array([[x, y, z] for x in [-.02, .02] for y in [0, .015] for z in [.99, 1.01]])
    scene = artificial_scene([[.1, 0, 1.]], points)
    scene.required_robot_parts = frozenset(['robot0_link4_collision', PAD['robot_geom_name']])
    with pytest.raises(ValueError, match='missing configured'):
        scene.check([PAD], TCP)


def test_contact_aperture_uses_gripper_joint_order_not_spatial_side():
    from vlm_tamp.rekep_libero.collision_paths_v7 import estimated_contact_aperture
    class Geometry:
        def fk(self, joints, include_volumes, gripper_qpos):
            a, b = gripper_qpos
            return {'eef_pose_world_xyzw': [0, 0, 0, 0, 0, 0, 1], 'robot_collision_volumes_world': [
                box([-a - .008, -.008, -.008], [-a, .008, .008], 'gripper0_finger1_pad_collision'),
                box([-b, -.008, -.008], [-b + .008, .008, .008], 'gripper0_finger2_pad_collision')]}
    cloud = np.array([[x, y, z] for x in [-.006, .012] for y in [-.003, 0, .003] for z in [-.003, 0, .003]])
    result = estimated_contact_aperture(Geometry(), np.zeros(7), cloud, [.04, -.04])
    assert np.allclose(result, [.0065, -.0125])


def test_pad_clearance_shell_does_not_allow_outer_finger_penetration():
    policy = ContactPolicy([PAD])
    finger = {**PAD, 'robot_geom_name': 'gripper0_finger1_collision'}
    points = np.array([[.018, .0112, 1.], [.018, .0112, 1.]])
    result = policy.robot_allowed(points, finger, TCP, points, 'grasp', [PAD],
                                  collision_values=np.array([.0006, -.0006]), clearance_margin=.001)
    assert result.tolist() == [True, False]


def test_planner_state_requires_explicit_drawer_configuration():
    points = np.array([[x, y, z] for x in [-.02, .02] for y in [0, .015] for z in [.99, 1.01]])
    scene = artificial_scene([[.1, 0, 1.]], points)
    with pytest.raises(ValueError, match='explicit prismatic'):
        scene.planner_state(None, np.zeros(7), gripper_qpos=[.04, -.04], stage='pull')


def test_joint_edge_sampling_detects_obstacle_between_clear_endpoints():
    from types import SimpleNamespace
    from vlm_tamp.rekep_libero.collision_paths_v7 import check_joint_path
    moving = np.array([[x, y, z] for x in [-.02, .02] for y in [.2, .215] for z in [.99, 1.01]])
    scene = artificial_scene([[.01, 0, 1.]], moving)
    class Geometry:
        def fk(self, q, include_volumes, gripper_qpos):
            return {'eef_pose_world_xyzw': [q[0], 0, 1, 0, 0, 0, 1], 'robot_collision_volumes_world': [
                box([q[0] - .001, -.002, .998], [q[0] + .001, .002, 1.002], 'robot0_link4_collision')]}
    ik = SimpleNamespace(self_valid=lambda q: True, chain=SimpleNamespace(forward=lambda q: [q[0], 0, 1, 0, 0, 0, 1]))
    path = np.array([[0, 0, 0, 0, 0, 0, 0], [.02, 0, 0, 0, 0, 0, 0]])
    geometry = Geometry()
    for q in path:
        assert scene.joints(geometry, q, gripper_qpos=[.04, -.04])['accepted']
    result = check_joint_path(scene, geometry, ik, path, gripper_qpos=[.04, -.04],
        transform_at_pose=lambda pose: np.eye(4), stage_at_pose=lambda pose: 'approach',
        check_payload=False, config={'collision_joint_sampling_rad': .006})
    assert not result['accepted']
    assert 0 < result['fraction'] < 1
