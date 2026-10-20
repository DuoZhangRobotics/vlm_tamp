"""Version 3 admission regressions requiring neither API nor simulator."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt

from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.rekep_libero.grasp_v3 import (
    AnalyticRobotSDF, LocalObservedField, contact_face_flags,
    filter_candidates, validate_pose_path,
)
from vlm_tamp.rekep_libero.strict_geometry import StrictObservedSDF


def uniform_field(free=True, bounds=(-.3, .3), spacing=.1, cls=StrictObservedSDF):
    size = round((bounds[1] - bounds[0]) / spacing) + 1
    shape = (size,) * 3
    return cls(np.full(3, bounds[0]), np.full(3, bounds[1]), np.zeros(shape, bool),
               np.full(shape, free, bool), np.full(shape, -1. if free else .1), np.full(3, spacing))


def calibration():
    transform = np.eye(4)
    transform[2, 3] = .097
    return {'graspgen_base_to_tcp': transform.tolist(),
            'tool_collision_points_tcp': [[0, 0, -.08], [.04, 0, 0], [-.04, 0, 0]],
            'tool_collision_finger_flags': [False, True, True],
            'tool_collision_contact_face_flags': [False, True, True],
            'joint_limits': [[-3, 3]] * 7}


def generated():
    base = np.eye(4)
    base[2, 3] = -.097
    return {'poses_world_base': np.asarray([base]), 'scores': [.8], 'retained_indices': [0]}


def solution(joints=None, robot_points=None):
    return {'success': True, 'position_error': .0001, 'orientation_error': .001,
            'num_descents': 4, 'cspace_position': [0.] * 7 if joints is None else joints,
            'robot_collision_points_world': [[0, 0, -.1]] if robot_points is None else robot_points,
            'robot_collision_finger_flags': [False], 'robot_collision_contact_face_flags': [False]}


def test_exact_robot_boundary_clears_without_clearing_one_mm_outside():
    field = uniform_field(False, (0., .02), .01, AnalyticRobotSDF)
    vertices = np.asarray([[x, y, z] for x in [.003, .0049]
                          for y in [.003, .0049] for z in [.003, .0049]])
    volume = {'type': 'convex', 'source': 'configured robot collision geometry',
              'vertices_world': vertices.tolist(), 'planes_world': ConvexHull(vertices).equations.tolist()}
    assert field.exclude_known_robot_volumes([volume]) == 0  # No coarse voxel center inside.
    assert field.states(vertices)['free'].all()
    values, allowed = field.collision_values(vertices, .005)
    assert (values <= 0).all() and not allowed.any()
    outside = [[.0059, .004, .004]]
    assert field.states(outside)['unknown'][0]
    assert field.query(outside)[0] > 0
    assert field.collision_values(outside)[0][0] > 0


def test_analytic_fine_robot_prior_does_not_fail_five_mm_margin():
    field = uniform_field(False, (0., .012), .003, AnalyticRobotSDF)
    vertices = np.asarray([[x, y, z] for x in [.003, .006]
                          for y in [.003, .006] for z in [.003, .006]])
    field.exclude_known_robot_volumes([{'type': 'convex', 'source': 'configured robot collision geometry',
        'vertices_world': vertices.tolist(), 'planes_world': ConvexHull(vertices).equations.tolist()}])
    assert (field.collision_values(vertices, .005)[0] <= 0).all()
    assert field.collision_values([[.007, .004, .004]], .005)[0][0] > 0


def test_saved_field_replays_exact_boundary_prior_without_pickle(tmp_path):
    field = uniform_field(False, (0., .02), .01, AnalyticRobotSDF)
    vertices = np.asarray([[x, y, z] for x in [.003, .0049]
                          for y in [.003, .0049] for z in [.003, .0049]])
    field.exclude_known_robot_volumes([{'type': 'convex', 'source': 'configured robot collision geometry',
        'vertices_world': vertices.tolist(), 'planes_world': ConvexHull(vertices).equations.tolist()}])
    output = tmp_path / 'field.npz'
    field.save(output)
    saved = AnalyticRobotSDF.load(output)
    assert saved.states(vertices)['free'].all()
    assert (saved.collision_values(vertices)[0] < 0).all()
    assert saved.states([[.0059, .004, .004]])['unknown'][0]


def test_local_unknown_is_never_excused_by_coarse_free():
    coarse = uniform_field(True)
    fine = uniform_field(False, (-.1, .1), .02)
    field = LocalObservedField(coarse, fine)
    assert field.states([[0, 0, 0]])['unknown'][0]
    assert field.query([[0, 0, 0]])[0] > 0
    values, allowed = field.collision_values([[0, 0, 0]], finger_flags=[True], target_points=[[0, 0, 0]])
    assert values[0] > 0 and not allowed[0]
    assert field.query([[.2, .2, .2]])[0] < 0


def test_occupied_corner_interpolation_cannot_hide_a_collision():
    free = np.ones((3, 3, 3), bool)
    free[1, 1, 1] = False
    occupied = ~free
    signed = (distance_transform_edt(~free) - distance_transform_edt(free)) * .01
    frozen = StrictObservedSDF(np.zeros(3), np.full(3, .02), occupied, free, signed, np.full(3, .01))
    fixed = AnalyticRobotSDF(np.zeros(3), np.full(3, .02), occupied, free, signed, np.full(3, .01))
    point = [[.0149, .0149, .0149]]
    assert frozen.states(point)['occupied'][0]
    assert frozen.query(point)[0] < -.005
    assert fixed.query(point)[0] > 0
    assert fixed.collision_values(point)[0][0] > 0
    values, allowed = fixed.collision_values(point, finger_flags=[True], target_points=point)
    assert allowed[0] and values[0] < 0
    values, allowed = fixed.collision_values(point, finger_flags=[False], target_points=point)
    assert not allowed[0] and values[0] > 0


def test_legacy_contact_labels_allow_inner_pad_and_reject_finger_back():
    config = {'tool_collision_points_tcp': [[-.04, 0, 0], [-.05, 0, 0],
              [.04, 0, 0], [.05, 0, 0], [0, 0, 0], [-.04, .01, 0]],
              'tool_collision_finger_flags': [True, True, True, True, False, True]}
    faces, source = contact_face_flags(config)
    assert faces.tolist() == [True, False, True, False, False, False]
    assert 'legacy' in source
    config['tool_collision_contact_face_flags'] = [True] * 6
    with pytest.raises(ValueError, match='subset'):
        contact_face_flags(config)


def test_filter_uses_alternate_collision_free_ik_and_records_all_solutions():
    class Sim:
        def multi_ik(self, *args, **kwargs):
            return {'solutions': [solution(robot_points=[[.4, 0, 0]]), solution(joints=[.1] * 7)],
                    'attempts': [{}, {}]}

    result = filter_candidates(generated(), [0, 0, 0], [[0, 0, 0]], calibration(), Sim(), uniform_field())
    assert result['accepted']
    assert result['selected_solution_joints'] == [.1] * 7
    assert result['attempted_pose_variants'] == 2
    assert len(result['feasible_candidates']) == 2
    rejected = result['feasible_candidates'][0]['solution_rejections']
    assert len(rejected) == 1 and rejected[0]['arm_collision']['blocked_unknown_points'] == 1


def test_half_turn_preserves_contact_origin_and_can_change_ik_reachability():
    received = []

    class Sim:
        def multi_ik(self, tcp, **kwargs):
            matrix = pose_matrix(tcp)
            received.append(matrix)
            return {'solutions': [solution()] if matrix[0, 0] < 0 else [], 'attempts': []}

    result = filter_candidates(generated(), [0, 0, 0], [[0, 0, 0]], calibration(), Sim(), uniform_field())
    assert result['accepted'] and result['selected_variant'] == 'antipodal_half_turn'
    assert np.allclose(received[0][:3, 3], received[1][:3, 3])
    assert np.allclose(received[0][:3, 2], received[1][:3, 2])
    assert result['rejected_candidates'][0]['reason'] == 'ik_unreachable'


def test_measured_asset_alignment_is_separate_and_keeps_97mm_tcp_transform():
    class Sim:
        def multi_ik(self, *args, **kwargs):
            return {'solutions': [solution()], 'attempts': []}

    config = calibration()
    config.update(graspgen_finger_attach_z_m=.0584, physical_finger_attach_z_m=.0524)
    original = generated()
    saved = original['poses_world_base'].copy()
    result = filter_candidates(original, [0, 0, 0], [[0, 0, 0]], config, Sim(), uniform_field())
    assert result['attempted_pose_variants'] == 4
    raw = next(row for row in result['feasible_candidates'] if row['variant'] == 'raw')
    aligned = next(row for row in result['feasible_candidates'] if row['variant'] == 'asset_contact_aligned')
    assert aligned['tcp_pose_world_xyzw'][2] - raw['tcp_pose_world_xyzw'][2] == pytest.approx(.006)
    assert config['graspgen_base_to_tcp'][2][3] == .097
    assert np.array_equal(original['poses_world_base'], saved)


def test_missing_full_arm_geometry_fails_admission():
    class Sim:
        def multi_ik(self, *args, **kwargs):
            value = solution()
            del value['robot_collision_points_world']
            return {'solutions': [value], 'attempts': []}

    result = filter_candidates(generated(), [0, 0, 0], [[0, 0, 0]], calibration(), Sim(), uniform_field())
    assert not result['accepted']
    assert all(row['solution_rejections'][0]['reason'] == 'ik_missing_robot_collision_geometry'
               for row in result['rejected_candidates'])


def test_coarse_fine_comparison_preserves_five_mm_margin():
    class Sim:
        def multi_ik(self, *args, **kwargs):
            raise AssertionError('Unknown fine tool region must reject before IK')

    result = filter_candidates(generated(), [0, 0, 0], [[0, 0, 0]], calibration(), Sim(),
                               uniform_field(), uniform_field(False, (-.15, .15), .03))
    assert not result['accepted']
    assert all(row['coarse_tool_collision']['accepted'] for row in result['rejected_candidates'])
    assert all(row['coarse_fine_tool_admission_change'] for row in result['rejected_candidates'])
    assert result['thresholds']['collision_margin_m'] == .005
    with pytest.raises(ValueError, match='5mm'):
        filter_candidates(generated(), [0, 0, 0], [[0, 0, 0]], calibration(), Sim(), uniform_field(),
                          collision_margin=.001)


def test_dense_path_rejects_ik_branch_jump():
    class Sim:
        def multi_ik(self, *args, **kwargs):
            return {'solutions': [solution(joints=[1.] * 7)], 'attempts': []}

    result = validate_pose_path([[0, 0, 0, 0, 0, 0, 1]], calibration(), Sim(),
                               uniform_field(), initial_joints=np.zeros(7))
    assert not result['accepted']
    assert result['solution_rejections'][0]['reason'] == 'ik_branch_discontinuity'


def test_dense_path_can_keep_continuous_collision_free_branch():
    class Sim:
        def multi_ik(self, tcp, seed_joints=None, **kwargs):
            return {'solutions': [solution(joints=[1.] * 7), solution(joints=[.05] * 7)], 'attempts': []}

    result = validate_pose_path([[0, 0, 0, 0, 0, 0, 1], [.01, 0, 0, 0, 0, 0, 1]],
        calibration(), Sim(), uniform_field(), initial_joints=np.zeros(7))
    assert result['accepted'] and len(result['checks']) == 2
    assert result['solution_joints'] == [[.05] * 7] * 2


class PalmEdgeField:
    """Nominal palm is clear; a 2mm FK adjustment enters a seen surface."""

    bounds_min = np.full(3, -.3)
    bounds_max = np.full(3, .3)
    spacing = np.full(3, .003)

    def states(self, points):
        points = np.asarray(points)
        occupied = (points[:, 0] > .001) & (points[:, 2] < -.05)
        return {'occupied': occupied, 'free': ~occupied, 'unknown': np.zeros(len(points), bool)}

    def query(self, points):
        return np.where(self.states(points)['occupied'], .003, -.1)

    def collision_values(self, points, margin, finger_flags, target_points, contact_radius=.012):
        return self.query(points) + margin, np.zeros(len(points), bool)


def test_candidate_ik_fk_tool_check_rejects_collision_hidden_by_nominal_pose():
    class Sim:
        def multi_ik(self, tcp, **kwargs):
            value = solution()
            value['eef_pose_world_xyzw'] = [tcp[0] + .002, *tcp[1:]]
            return {'solutions': [value], 'attempts': []}

    result = filter_candidates(generated(), [0, 0, 0], [[0, 0, 0]], calibration(), Sim(), PalmEdgeField())
    raw = next(row for row in result['rejected_candidates'] if row['variant'] == 'raw')
    assert raw['collision']['accepted']
    check = raw['solution_rejections'][0]
    assert check['reason'] == 'fk_tool_collision_or_unknown'
    assert not check['fk_tool_collision']['accepted']
    assert check['fk_position_adjustment_m'] == pytest.approx(.002)
    assert check['fk_pose_source'] == 'isolated_robot_forward_kinematics'


def test_accepted_candidates_preserve_raw_pose_and_record_checked_fk_pose():
    class Sim:
        def multi_ik(self, tcp, **kwargs):
            value = solution()
            value['eef_pose_world_xyzw'] = [tcp[0] + .002, *tcp[1:]]
            return {'solutions': [value], 'attempts': []}

    result = filter_candidates(generated(), [0, 0, 0], [[0, 0, 0]], calibration(), Sim(), uniform_field())
    assert result['accepted']
    assert result['selected_tcp_pose'][0] == 0
    assert result['selected_fk_tcp_pose'][0] == .002
    assert result['feasible_candidates'][0]['fk_tool_collision']['accepted']


def test_dense_path_returns_checked_fk_command_poses_without_replacing_nominal_input():
    class Sim:
        def multi_ik(self, tcp, **kwargs):
            value = solution()
            value['eef_pose_world_xyzw'] = [tcp[0] + .002, *tcp[1:]]
            return {'solutions': [value], 'attempts': []}

    poses = np.asarray([[0, 0, 0, 0, 0, 0, 1], [.01, 0, 0, 0, 0, 0, 1]])
    saved = poses.copy()
    result = validate_pose_path(poses, calibration(), Sim(), uniform_field(), initial_joints=np.zeros(7))
    assert result['accepted']
    assert np.array_equal(poses, saved)
    assert np.allclose(np.asarray(result['fk_poses_world_xyzw'])[:, 0], [.002, .012])
    assert all(row['fk_tool_collision']['accepted'] for row in result['checks'])


def test_dense_path_rejects_fk_tool_collision_before_admitting_arm():
    class Sim:
        def multi_ik(self, tcp, **kwargs):
            value = solution()
            value['eef_pose_world_xyzw'] = [tcp[0] + .002, *tcp[1:]]
            return {'solutions': [value], 'attempts': []}

    result = validate_pose_path([[0, 0, 0, 0, 0, 0, 1]], calibration(), Sim(), PalmEdgeField(),
                               initial_joints=np.zeros(7))
    assert not result['accepted']
    assert result['solution_rejections'][0]['reason'] == 'fk_tool_collision_or_unknown'


def test_fake_ik_transport_missing_fk_uses_declared_nominal_fallback():
    class Sim:
        def multi_ik(self, tcp, **kwargs):
            return {'solutions': [solution()], 'attempts': []}

    result = validate_pose_path([[0, 0, 0, 0, 0, 0, 1]], calibration(), Sim(), uniform_field(),
                               initial_joints=np.zeros(7))
    assert result['accepted']
    assert result['checks'][0]['fk_pose_source'] == 'nominal_fallback_missing_transport_fk'
    assert result['fk_poses_world_xyzw'] == [[0, 0, 0, 0, 0, 0, 1]]
