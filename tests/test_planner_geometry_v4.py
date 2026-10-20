"""Independent finite-difference and observed-motion checks for planner adapters."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vlm_tamp.rekep_libero.geometry import matrix_pose, pose_matrix, transform_points
from vlm_tamp.rekep_libero.object_tracking_v4 import register_closed_surface
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain, fixed_orientation_residual
from vlm_tamp.rekep_libero.task_geometry_v4 import bind_bowl_goal, joint_residual, pose_residual


@pytest.fixture
def chain():
    return RobotChain(json.loads((Path(__file__).parent / 'data/panda_chain_v4.json').read_text()))


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_seven_joint_pose_jacobian_against_independent_difference(chain, seed):
    q = np.random.default_rng(seed).uniform(chain.limits[:, 0] + .1, chain.limits[:, 1] - .1)
    _, jacobian = chain.forward(q, jacobian=True)
    expected = np.empty((6, 7))
    for column in range(7):
        perturb = np.eye(7)[column] * 1e-6
        positive, negative = pose_matrix(chain.forward(q + perturb)), pose_matrix(chain.forward(q - perturb))
        expected[:3, column] = (positive[:3, 3] - negative[:3, 3]) / 2e-6
        expected[3:, column] = Rotation.from_matrix(negative[:3, :3].T @ positive[:3, :3]).as_rotvec() / 2e-6
    np.testing.assert_allclose(jacobian, expected, atol=2e-8)


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_full_drawer_joint_constraint_jacobian(chain, seed):
    reference = pose_matrix(chain.forward(chain.reference_joints))[:3, :3]
    q = chain.reference_joints + np.random.default_rng(seed).normal(0, .25, 7)
    spec = {'type': 'point_on_line', 'point_local_m': [.01, -.02, .03],
            'line_point_world_m': [.02, -.13, 1.02], 'axis_world': [.03, 1., 0.],
            'minimum_travel_m': 0., 'maximum_travel_m': .15}

    def result(joints):
        pose, jacobian = chain.forward(joints, jacobian=True)
        line = joint_residual(spec, pose, jacobian)
        angle, angle_jac = fixed_orientation_residual(reference, pose_matrix(pose)[:3, :3], jacobian[3:])
        return np.r_[line['equality'], angle], np.vstack([line['equality_joint_jacobian'], angle_jac])

    _, jacobian = result(q)
    numeric = np.column_stack([(result(q + step)[0] - result(q - step)[0]) / 2e-6 for step in np.eye(7) * 1e-6])
    np.testing.assert_allclose(jacobian, numeric, atol=2e-8)


def test_bowl_binding_accounts_for_observed_grasp_offset():
    initial_support = np.array([-.1, .02, .905])
    target = np.array([.05, -.03, .915])
    spec = {'object_support_point_initial_world_m': initial_support.tolist(),
            'goal': {'type': 'point_target', 'point_local_m': [0., 0., 0.], 'target_world_m': target.tolist()}}
    motion = np.eye(4)
    motion[:3, :3] = Rotation.from_euler('z', .1).as_matrix()
    motion[:3, 3] = [.005, -.003, .05]
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler('xy', [.4, -.2]).as_matrix()
    tcp[:3, 3] = [-.12, .04, 1.0]
    goal, requested = bind_bowl_goal(spec, motion, matrix_pose(tcp))
    support = transform_points([initial_support], motion)[0]
    independent_local = tcp[:3, :3].T @ (support - tcp[:3, 3])
    np.testing.assert_allclose(goal['point_local_m'], independent_local)
    np.testing.assert_allclose(pose_residual(goal, requested)['equality'], 0., atol=1e-12)
    assert goal['attachment_binding_status'] == 'bound_after_observed_retained_lift'


def test_closure_registration_handles_object_displacement_without_claiming_attachment():
    generator = np.random.default_rng(0)
    cloud = generator.normal(0, [.03, .025, .02], (1500, 3)) + [-.1, .01, .94]
    center = cloud.mean(axis=0)
    rotation = Rotation.from_euler('zy', [.05, .025]).as_matrix()
    movement = np.eye(4)
    movement[:3, :3] = rotation
    movement[:3, 3] = center + [.010, -.003, .004] - rotation @ center
    observed = transform_points(cloud, movement) + generator.normal(0, .0001, cloud.shape)
    fit = register_closed_surface(cloud, observed)
    assert fit['verified']
    assert not fit['attachment_claimed']
    np.testing.assert_allclose(transform_points(cloud, fit['transform_initial_to_closed_world']),
                               transform_points(cloud, movement), atol=.0001)


def test_missing_surface_does_not_become_a_grasp_attachment():
    with pytest.raises(ValueError, match='Insufficient observed surface'):
        register_closed_surface(np.zeros((100, 3)), np.zeros((20, 3)))
