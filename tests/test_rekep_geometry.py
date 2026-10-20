"""Physical interface tests independent of API access and model downloads."""

from types import SimpleNamespace

import numpy as np
import pytest

from vlm_tamp.rekep_libero.geometry import (
    ObservedSDF, evaluate_collision, graspgen_to_tcp, interpolate_path,
    pose_matrix, rekep_to_tcp, tcp_to_rekep,
)
from vlm_tamp.rekep_libero.perception import NATIVE_PROPOSER_CONFIG, Perception
from vlm_tamp.rekep_libero.planning import ClientIK, PlanningAdapter, _native_modules, select_grasp


def plane_camera():
    return {'depth': np.full((32, 32), 2.), 'intrinsic': [[24, 0, 16], [0, 24, 16], [0, 0, 1]],
            'camera_to_world': np.eye(4)}


@pytest.fixture
def sdf():
    return ObservedSDF.from_rgbd([plane_camera()], [-.3, -.3, 1], [.3, .3, 3], voxel_size=.05)


def calibration():
    base = np.eye(4)
    base[2, 3] = .097
    tool = np.eye(4)
    tool[:3, :3] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    return {'graspgen_base_to_tcp': base.tolist(), 'tcp_to_rekep': tool.tolist(),
            'tool_collision_points_tcp': [[0, 0, -.08], [0, 0, 0]],
            'tool_collision_finger_flags': [False, True]}


def test_sdf_signed_inside_outside_unknown_and_out_of_bounds(sdf):
    distances = sdf.query([[0, 0, 1.5], [0, 0, 2], [0, 0, 2.5], [1, 0, 1.5]])
    assert distances[0] < 0
    assert (distances[1:] > 0).all()
    states = sdf.states([[0, 0, 1.5], [0, 0, 2], [0, 0, 2.5]])
    assert states['free'].tolist() == [True, False, False]
    assert states['occupied'].tolist() == [False, True, False]
    assert states['unknown'].tolist() == [False, False, True]


def test_opposite_depth_view_carves_only_observed_space(sdf):
    reverse = plane_camera()
    reverse['camera_to_world'] = np.diag([1, -1, -1, 1])
    reverse['camera_to_world'][2, 3] = 4
    fused = ObservedSDF.from_rgbd([plane_camera(), reverse], [-.3, -.3, 1], [.3, .3, 3], .05)
    assert fused.query([[0, 0, 2.5]])[0] < 0
    assert fused.query([[0, 0, 2]])[0] > 0
    assert sdf.query([[0, 0, 2.5]])[0] > 0


def test_contact_exception_is_finger_and_observed_target_only(sdf):
    points = [[0, 0, 2], [0, 0, 2], [0, 0, 2.1]]
    values, allowed = sdf.collision_values(points, finger_flags=[True, False, True],
                                          target_points=[[0, 0, 2], [0, 0, 2.1]])
    assert allowed.tolist() == [True, False, False]
    assert values[0] < 0 and (values[1:] > 0).all()
    assert not evaluate_collision(sdf, points, [True, False, True], [[0, 0, 2]])['accepted']


def test_robot_exclusion_does_not_carve_unknown(sdf):
    before = sdf.unknown.copy()
    count = sdf.exclude_known_robot([[0, 0, 2], [0, 0, 2.5]], .06)
    assert count > 0
    assert np.array_equal(before, sdf.unknown)
    assert sdf.query([[0, 0, 2]])[0] < 0
    assert sdf.query([[0, 0, 2.5]])[0] > 0
    with pytest.raises(ValueError, match='known robot'):
        sdf.exclude_observed_region([[0, 0, 2.5]], source='simulator_scene')


def test_exact_known_robot_volume_excludes_only_its_interior(sdf):
    from scipy.spatial import ConvexHull

    vertices = np.array([[x, y, z] for x in [-.1, .1] for y in [-.1, .1] for z in [2.2, 2.7]])
    volume = {'type': 'convex', 'source': 'configured robot collision geometry',
              'vertices_world': vertices.tolist(), 'planes_world': ConvexHull(vertices).equations.tolist()}
    before = sdf.unknown.copy()
    assert sdf.exclude_known_robot_volumes([volume]) > 0
    assert sdf.query([[0, 0, 2.5]])[0] < 0
    assert sdf.query([[.2, 0, 2.5]])[0] > 0
    assert np.all(~(before & ~sdf.unknown) | sdf.known_robot_prior)
    assert sdf.unknown[before & ~sdf.known_robot_prior].all()
    volume['source'] = 'simulator environment geometry'
    with pytest.raises(ValueError, match='configured convex robot'):
        sdf.exclude_known_robot_volumes([volume])


def test_grasp_base_tcp_and_rekep_frames_are_explicit():
    config = calibration()
    base = np.eye(4)
    base[:3, 3] = [1, 2, 3]
    tcp = graspgen_to_tcp(base, config)
    assert np.allclose(tcp[:3], [1, 2, 3.097])
    rekep = tcp_to_rekep(tcp, config)
    assert np.allclose(pose_matrix(rekep)[:3, 0], pose_matrix(tcp)[:3, 2])
    assert np.allclose(pose_matrix(rekep)[:3, 1], pose_matrix(tcp)[:3, 0])
    assert np.allclose(pose_matrix(rekep_to_tcp(rekep, config)), pose_matrix(tcp))


def test_pose_validation_and_interpolation():
    invalid = np.eye(4)
    invalid[0, 0] = 2
    with pytest.raises(ValueError, match='orthonormal'):
        pose_matrix(invalid)
    path = interpolate_path([[0, 0, 0, 0, 0, 0, 1], [.03, 0, 0, 0, 0, 1, 0]], .01, .1)
    assert np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1).max() <= .010001
    assert np.allclose(path[0], [0, 0, 0, 0, 0, 0, 1])
    assert np.allclose(pose_matrix(path[-1]), pose_matrix([.03, 0, 0, 0, 0, 1, 0]))


def test_client_ik_converts_tool_frame_and_keeps_all_panda_joints():
    poses = []

    class Client:
        def ik(self, target, seed_joints=None, max_iterations=120):
            poses.append(target)
            return dict(success=True, num_descents=2, position_error=.001, cspace_position=list(range(7)))

    adapter = ClientIK(Client(), calibration())
    tcp = [0, 0, 1, 0, 0, 0, 1]
    result = adapter.solve(pose_matrix(tcp_to_rekep(tcp, calibration())))
    assert np.allclose(pose_matrix(poses[0]), pose_matrix(tcp))
    assert result.cspace_position.shape == (7,) and result.cspace_position[6] == 6


def test_detector_selection_uses_tcp_and_rejects_palm_collision(sdf):
    base = np.eye(4)
    base[2, 3] = 2 - .097
    other = base.copy()
    other[0, 3] = .15
    generated = {'poses_world_base': np.array([base, other]), 'scores': np.array([.4, .99]),
                 'retained_indices': np.array([1, 0])}

    def ik(pose, initial_joint_pos=None, max_iterations=120):
        return dict(success=True, num_descents=1, position_error=0, cspace_position=np.zeros(7))

    result = select_grasp(generated, np.array([0, 0, 2]), np.array([[0, 0, 2]]), calibration(), ik, sdf)
    assert result['accepted'] and result['selected_index'] == 0
    assert result['rejected_candidates'][0]['reason'] == 'target_membership_or_keypoint_proximity'
    palm_blocked = calibration()
    palm_blocked['tool_collision_points_tcp'][0] = [0, 0, 0]
    blocked = select_grasp(generated, np.array([0, 0, 2]), np.array([[0, 0, 2]]), palm_blocked, ik, sdf)
    assert not blocked['accepted']
    assert blocked['failure'] == 'grasp_no_feasible_detector_candidates'


def test_native_proposer_configuration_and_seventh_joint_regularization():
    assert NATIVE_PROPOSER_CONFIG['num_candidates_per_mask'] == 5
    assert NATIVE_PROPOSER_CONFIG['min_dist_bt_keypoints'] == .06
    (subgoal, _path), metadata = _native_modules(lambda *args: 0.)

    class IK:
        def solve(self, *args, **kwargs):
            return SimpleNamespace(success=True, num_descents=0, position_error=0,
                                   cspace_position=np.r_[np.zeros(6), .7])

    bounds = [(-1., 1.)] * 6
    _cost, debug = subgoal.objective(np.zeros(6), bounds, np.zeros((1, 3)), np.array([True]),
        [], [], lambda xyz: np.zeros(len(xyz)), np.empty((0, 3)), np.eye(4), IK(), np.zeros(7), np.zeros(7),
        False, return_debug_dict=True)
    assert debug['reset_reg_cost'] == pytest.approx(.14)
    assert 'all seven Panda joints regularized' in metadata['adaptations']


def test_tracking_without_depth_support_fails_closed():
    # A fake encoder isolates the real sensor-validation branch.
    perception = object.__new__(Perception)
    perception._capture = {}

    class Proposer:
        def _preprocess(self, rgb, points, masks):
            return rgb, rgb, points, masks, {}

        def _get_features(self, *args):
            perception._capture['features'] = np.ones((4, 4, 2))

    perception.proposer = Proposer()
    obs = {'agentview_rgb': np.zeros((4, 4, 3), np.uint8), 'agentview_depth': np.zeros((4, 4)),
           'agentview_world_points': np.zeros((4, 4, 3))}
    meta = {'main_camera': 'agentview', 'cameras': {'agentview': {'intrinsic': np.eye(3), 'camera_to_world': np.eye(4)}}}
    bundle = {'main_camera': 'agentview', 'points_world': np.array([[0, 0, 0]]), 'features': np.ones((1, 2))}
    with pytest.raises(RuntimeError, match='tracking_lost_no_depth_support'):
        perception.track(bundle, obs, meta)


def test_native_path_optimizer_and_post_validation_use_calibrated_tcp(sdf):
    config = calibration()
    config['joint_limits'] = np.tile([-3.2, 3.2], (7, 1)).tolist()

    def ik(pose, initial_joint_pos=None, max_iterations=120):
        return dict(success=True, num_descents=0, position_error=0, cspace_position=np.zeros(7))

    planner = PlanningAdapter(ik, config, [-.3, -.3, 1], [.3, .3, 1.8], sampling_maxfun=20, maxiter=5)
    result = planner.solve_stage(SimpleNamespace(subgoal_constraints=[], path_constraints=[]),
        np.array([0, 0, 1.4, 0, 0, 0, 1]), np.zeros(7), np.array([[0, 0, 1.6]]), np.array([False]), sdf,
        fixed_grasp_tcp_pose=np.array([0, 0, 1.5, 0, 0, 0, 1]))
    assert result['accepted'], result['failure']
    assert result['subgoal']['type'] == 'detector_fixed_grasp'
    assert result['path']['type'] == 'path_solver'
    assert len(result['validation']['checks']) == len(result['poses_world_xyzw'])
    assert np.allclose(pose_matrix(result['poses_world_xyzw'][-1]), pose_matrix([0, 0, 1.5, 0, 0, 0, 1]))
