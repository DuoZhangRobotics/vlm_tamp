"""Numerical checks for the robot-door manifold, independent of simulator state."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from vlm_tamp.coupled_manifold import CoupledHinge, CoupledHandlePoint, projected_path, ompl_plan
from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain


def fixture():
    axes = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0],
                       [1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    anchors = np.asarray([[0, 0, .1], [0, 0, .2], [.1, 0, .3], [.2, 0, .35],
                          [.25, .1, .4], [.35, .1, .45], [.4, .05, .5]])
    reference = np.eye(4)
    reference[:3, :3] = Rotation.from_euler('xyz', [.2, -.3, .4]).as_matrix()
    reference[:3, 3] = [.5, .1, .45]
    chain = RobotChain({'reference_joints': [0.] * 7, 'joint_axes_world': axes,
        'joint_anchors_world_m': anchors, 'reference_tcp_world': reference,
        'joint_limits': [[-2., 2.]] * 7})
    problem = CoupledHinge(chain, [.2, .1, .45], [0., 0., 1.], reference,
                           progress_bounds=(-.2, .4), position_tolerance=1e-6,
                           orientation_tolerance=1e-5)
    return chain, problem


class CoupledManifoldTests(unittest.TestCase):
    def test_reference_is_on_two_dimensional_manifold(self):
        _, problem = fixture()
        residual, jacobian = problem.evaluate(np.zeros(8))
        np.testing.assert_allclose(residual, 0., atol=1e-14)
        self.assertEqual(jacobian.shape, (6, 8))
        self.assertEqual(np.linalg.matrix_rank(jacobian), 6)
        self.assertTrue(problem.satisfied(np.zeros(8)))

    def test_hinge_pose_is_exact_circle_and_coupled_rotation(self):
        _, problem = fixture()
        angle = .3
        pose = problem.pose(angle)
        expected = problem.origin + Rotation.from_rotvec(angle * problem.axis).apply(
            problem.reference_tcp[:3, 3] - problem.origin)
        np.testing.assert_allclose(pose[:3, 3], expected, atol=1e-14)
        delta = pose[:3, :3] @ problem.reference_tcp[:3, :3].T
        np.testing.assert_allclose(Rotation.from_matrix(delta).as_rotvec(), angle * problem.axis, atol=1e-14)

    def test_full_eight_coordinate_jacobian_against_finite_difference(self):
        _, problem = fixture()
        rng = np.random.default_rng(13)
        for _ in range(8):
            point = rng.uniform(-.15, .15, 8)
            _, analytic = problem.evaluate(point)
            finite = np.column_stack([(problem.evaluate(point + np.eye(8)[column] * 1e-6)[0]
                                       - problem.evaluate(point - np.eye(8)[column] * 1e-6)[0]) / 2e-6
                                      for column in range(8)])
            np.testing.assert_allclose(analytic, finite, atol=3e-8, rtol=3e-7)

    def test_bounded_projection_can_hold_requested_angle(self):
        _, problem = fixture()
        result = problem.project(np.r_[np.zeros(7), .04], fixed_progress=.04)
        self.assertTrue(result['accepted'], result)
        self.assertEqual(result['state'][7], .04)
        self.assertLess(result['position_error_m'], 1e-6)
        self.assertLess(result['orientation_error_rad'], 1e-5)
        self.assertTrue(problem.in_bounds(result['state']))

    def test_projected_edge_checks_interior_collision(self):
        _, problem = fixture()
        start = np.zeros(8)
        goal = problem.project(np.r_[np.zeros(7), .10], fixed_progress=.10)['state']
        path, checked = projected_path(problem, [start, goal], maximum_joint_step=.01,
            maximum_progress_step=.01, valid=lambda x: not (.035 < x[7] < .065))
        self.assertFalse(checked['accepted'])
        self.assertTrue(checked['rejected'])
        self.assertTrue(all(problem.satisfied(x) for x in path))
        self.assertFalse(checked['continuous_clearance_certified'])

    def test_projection_rejects_unreachable_fixed_progress(self):
        chain, problem = fixture()
        chain.limits[:] = [-1e-5, 1e-5]
        blocked = CoupledHinge(chain, problem.origin, problem.axis, problem.reference_tcp,
            progress_bounds=(0., .4), position_tolerance=1e-6, orientation_tolerance=1e-5)
        result = blocked.project(np.r_[np.zeros(7), .3], fixed_progress=.3)
        self.assertFalse(result['accepted'])
        self.assertGreater(result['position_error_m'], .05)

    def test_progress_bounds_and_invalid_axis_are_explicit(self):
        chain, problem = fixture()
        self.assertFalse(problem.in_bounds(np.r_[np.zeros(7), .5]))
        with self.assertRaises(ValueError):
            problem.project(np.zeros(8), fixed_progress=.5)
        with self.assertRaises(ValueError):
            CoupledHinge(chain, problem.origin, [0., 0., 0.], problem.reference_tcp)

    def test_planner_rejects_endpoint_outside_its_stricter_manifold_tolerance(self):
        _, problem = fixture()
        problem.position_tolerance, problem.orientation_tolerance = .01, .1
        off_manifold = np.zeros(8)
        off_manifold[0] = .001
        self.assertTrue(problem.satisfied(off_manifold))
        path, result = ompl_plan(problem, off_manifold, np.zeros(8), lambda x: True)
        self.assertEqual(path.shape, (0, 8))
        self.assertEqual(result['status'], 'invalid_start_constraint')


class HandlePointTests(unittest.TestCase):
    def test_four_equality_jaw_axis_jacobian_against_finite_difference(self):
        chain, rigid = fixture()
        problem = CoupledHandlePoint(chain, rigid.origin, rigid.axis, rigid.reference_tcp,
            pad_local=[.01, -.02, .06], jaw_axis_local=rigid.reference_tcp[:3, :3].T @ [1., 0., 0.],
            progress_bounds=(-.2, .4))
        self.assertEqual(problem.codimension, 4)
        rng = np.random.default_rng(29)
        for _ in range(8):
            point = rng.uniform(-.1, .1, 8)
            residual, analytic = problem.evaluate(point)
            finite = np.column_stack([(problem.evaluate(point + np.eye(8)[column] * 1e-6)[0]
                                       - problem.evaluate(point - np.eye(8)[column] * 1e-6)[0]) / 2e-6
                                      for column in range(8)])
            self.assertEqual(residual.shape, (4,))
            self.assertEqual(np.linalg.matrix_rank(analytic), 4)
            np.testing.assert_allclose(analytic, finite, atol=3e-8, rtol=3e-7)
            self.assertEqual(analytic[3, 7], 0.)

    def test_rotation_about_jaw_allows_seventy_three_degree_up_tilt(self):
        class RollChain:
            limits = np.tile([-2., 2.], (7, 1))

            def forward(self, q, jacobian=False):
                rotation = Rotation.from_euler('xyz', [np.pi / 2 + q[0], q[1], 0.])
                pose = np.r_[[.5, 0., .4], rotation.as_quat()]
                jac = np.zeros((6, 7))
                jac[3:, 0] = [1., 0., 0.]
                jac[3:, 1] = rotation.as_matrix().T @ [0., 1., 0.]
                return (pose, jac) if jacobian else pose

        chain = RollChain()
        reference = pose_matrix(chain.forward(np.zeros(7)))
        problem = CoupledHandlePoint(chain, [.2, 0., .4], [0., 0., 1.], reference,
            pad_local=[0., 0., 0.], jaw_axis_local=[1., 0., 0.])
        rolled = np.zeros(8)
        rolled[0] = np.deg2rad(73.)
        self.assertTrue(problem.satisfied(rolled))
        self.assertAlmostEqual(problem.errors(rolled)['orientation_cone_angle_rad'], np.deg2rad(73.))
        self.assertAlmostEqual(problem.errors(rolled)['jaw_axis_error_rad'], 0.)
        np.testing.assert_allclose(problem.evaluate(rolled)[0], 0., atol=1e-14)
        rolled[1] = .2
        self.assertFalse(problem.satisfied(rolled))
        self.assertGreater(problem.errors(rolled)['jaw_axis_error_rad'], .1)

    def test_five_equality_axis_aligned_jacobian_against_finite_difference(self):
        chain, rigid = fixture()
        problem = CoupledHandlePoint(chain, rigid.origin, rigid.axis, rigid.reference_tcp,
            pad_local=[.01, -.02, .06], up_local=rigid.reference_tcp[:3, :3].T @ rigid.axis,
            progress_bounds=(-.2, .4), align_axis=True)
        self.assertEqual(problem.codimension, 5)
        rng = np.random.default_rng(23)
        for _ in range(8):
            point = rng.uniform(-.1, .1, 8)
            residual, analytic = problem.evaluate(point)
            finite = np.column_stack([(problem.evaluate(point + np.eye(8)[column] * 1e-6)[0]
                                       - problem.evaluate(point - np.eye(8)[column] * 1e-6)[0]) / 2e-6
                                      for column in range(8)])
            self.assertEqual(residual.shape, (5,))
            self.assertEqual(np.linalg.matrix_rank(analytic), 5)
            np.testing.assert_allclose(analytic, finite, atol=3e-8, rtol=3e-7)
            np.testing.assert_allclose(analytic[3:, 7], 0., atol=1e-14)

    def test_axis_aligned_projection_preserves_point_and_bar_direction(self):
        chain, rigid = fixture()
        problem = CoupledHandlePoint(chain, rigid.origin, rigid.axis, rigid.reference_tcp,
            pad_local=[.01, -.02, .06], up_local=rigid.reference_tcp[:3, :3].T @ rigid.axis,
            progress_bounds=(-.2, .4), align_axis=True, position_tolerance=1e-6,
            orientation_tolerance=1e-5)
        result = problem.project(np.r_[np.zeros(7), .10], fixed_progress=.10)
        self.assertTrue(result['accepted'], result)
        self.assertLess(result['position_error_m'], 1e-6)
        self.assertLess(result['orientation_cone_angle_rad'], 1e-5)
        self.assertLess(result['weighted_residual_norm'], 1e-6)

    def test_point_constraint_jacobian_includes_local_pad_offset(self):
        chain, rigid = fixture()
        problem = CoupledHandlePoint(chain, rigid.origin, rigid.axis, rigid.reference_tcp,
            pad_local=[.01, -.02, .06], up_local=rigid.reference_tcp[:3, :3].T @ rigid.axis,
            progress_bounds=(-.2, .4))
        rng = np.random.default_rng(19)
        for _ in range(8):
            point = rng.uniform(-.1, .1, 8)
            residual, analytic = problem.evaluate(point)
            finite = np.column_stack([(problem.evaluate(point + np.eye(8)[column] * 1e-6)[0]
                                       - problem.evaluate(point - np.eye(8)[column] * 1e-6)[0]) / 2e-6
                                      for column in range(8)])
            self.assertEqual(residual.shape, (3,))
            self.assertEqual(np.linalg.matrix_rank(analytic), 3)
            np.testing.assert_allclose(analytic, finite, atol=3e-8, rtol=3e-7)

    def test_fixed_progress_point_projection(self):
        chain, rigid = fixture()
        problem = CoupledHandlePoint(chain, rigid.origin, rigid.axis, rigid.reference_tcp,
            pad_local=[.01, -.02, .06], up_local=rigid.reference_tcp[:3, :3].T @ rigid.axis,
            progress_bounds=(-.2, .4), position_tolerance=1e-6)
        result = problem.project(np.r_[np.zeros(7), .10], fixed_progress=.10)
        self.assertTrue(result['accepted'], result)
        self.assertLess(result['position_error_m'], 1e-6)
        self.assertEqual(result['state'][7], .10)

    def test_bar_symmetry_is_free_but_orientation_cone_remains_required(self):
        class SpinChain:
            limits = np.tile([-2., 2.], (7, 1))

            def forward(self, q, jacobian=False):
                pose = np.r_[[.5, 0., .4], Rotation.from_euler('xyz', q[:3]).as_quat()]
                return (pose, np.zeros((6, 7))) if jacobian else pose

        chain = SpinChain()
        reference = pose_matrix(chain.forward(np.zeros(7)))
        problem = CoupledHandlePoint(chain, [.2, 0., .4], [0., 0., 1.], reference,
            pad_local=[0., 0., 0.], up_local=[0., 0., 1.])
        axial_spin = np.zeros(8)
        axial_spin[2] = 1.2
        np.testing.assert_allclose(problem.evaluate(axial_spin)[0], 0., atol=1e-14)
        self.assertTrue(problem.satisfied(axial_spin))
        excessive_tilt = np.zeros(8)
        excessive_tilt[0] = .5
        np.testing.assert_allclose(problem.evaluate(excessive_tilt)[0], 0., atol=1e-14)
        self.assertFalse(problem.satisfied(excessive_tilt))
        self.assertGreater(problem.errors(excessive_tilt)['orientation_cone_violation_rad'], 0.)


if __name__ == '__main__':
    unittest.main()
