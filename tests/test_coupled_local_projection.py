"""Local contact binding must preserve IK branch and reject local infeasibility."""

import unittest
import numpy as np

from vlm_tamp.coupled_local_projection import project_nearby
from vlm_tamp.coupled_manifold import CoupledHandlePoint
from vlm_tamp.coupled_pipeline import project_ik_candidate
from vlm_tamp.rekep_libero.geometry import pose_matrix
from test_coupled_manifold import fixture


class LinearPoint:
    def __init__(self, target):
        self.target = np.asarray(target, float)
        self.bounds = np.tile([-2., 2.], (8, 1))
        self.bounds[7] = [0., 1.]

    def evaluate(self, x):
        jacobian = np.zeros((3, 8))
        jacobian[:, :3] = np.eye(3)
        return np.asarray(x[:3]) - self.target, jacobian

    def errors(self, x):
        error = float(np.linalg.norm(self.evaluate(x)[0]))
        return {'position_error_m': error, 'orientation_error_rad': 0.,
                'weighted_residual_norm': error}

    def satisfied(self, x):
        return bool(np.all(x >= self.bounds[:, 0]) and np.all(x <= self.bounds[:, 1])
                    and self.errors(x)['position_error_m'] < 1e-6)


class NearbyProjectionTests(unittest.TestCase):
    def test_accurate_hypothetical_ik_keeps_its_joint_posture(self):
        start=np.array([.1,.2,.3,.7,-.4,.3,.9,.2])
        result=project_ik_candidate(LinearPoint(start[:3]),start[:7],start[7])
        self.assertTrue(result['accepted'])
        np.testing.assert_array_equal(result['state'],start)

    def test_only_hypothetical_ik_seed_may_be_clipped_to_margin(self):
        problem=LinearPoint([.019,0.,0.]);problem.bounds[0]=[-.02,.02]
        joints=np.array([.1,0.,0.,.7,-.4,.3,.9])
        self.assertFalse(project_nearby(problem,np.r_[joints,.2])['accepted'])
        result=project_ik_candidate(problem,joints,.2)
        self.assertTrue(result['accepted'],result)
        self.assertAlmostEqual(result['hypothetical_seed_clipping_rad'],.08,places=8)
        self.assertLess(result['position_error_m'],1e-6)
        np.testing.assert_array_equal(result['state'][3:7],joints[3:])

    def test_underdetermined_nullspace_is_unchanged(self):
        start = np.array([.1, .2, .3, .444, -.5, 1.1, -.8, .2])
        original = start.copy()
        problem = LinearPoint([.1002, .1997, .3001])
        result = project_nearby(problem, start, residual_tolerance=1e-8)
        self.assertTrue(result['accepted'], result)
        self.assertLess(result['position_error_m'], 1e-8)
        np.testing.assert_array_equal(result['state'][3:], original[3:])
        np.testing.assert_array_equal(start, original)
        self.assertLess(result['maximum_joint_correction_rad'], .001)

    def test_infeasible_local_goal_fails_inside_neighbourhood(self):
        start = np.zeros(8)
        problem = LinearPoint([.2, 0., 0.])
        result = project_nearby(problem, start)
        self.assertFalse(result['accepted'])
        self.assertLessEqual(result['maximum_joint_correction_rad'], .04 + 1e-12)
        self.assertGreater(result['position_error_m'], .15)
        self.assertIn('local_projection_', result['reason'])

    def test_global_bounds_are_enforced_as_well_as_locality(self):
        problem = LinearPoint([.05, 0., 0.])
        problem.bounds[0] = [-.02, .02]
        result = project_nearby(problem, np.zeros(8))
        self.assertFalse(result['accepted'])
        self.assertLessEqual(result['state'][0], .02)
        self.assertGreater(result['position_error_m'], .029)

    def test_invalid_measured_joint_bounds_do_not_silently_clamp(self):
        problem = LinearPoint([0., 0., 0.])
        problem.bounds[0] = [-.02, .02]
        start = np.zeros(8)
        start[0] = .1
        result = project_nearby(problem, start)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'measured_start_outside_command_bounds')
        np.testing.assert_array_equal(result['state'], start)

    def test_robot_fk_jacobian_local_binding_avoids_wrist_nullspace_drift(self):
        chain, rigid = fixture()
        start = np.r_[np.zeros(7), .1]
        reference = pose_matrix(chain.forward(start[:7]))
        pad = np.array([0., 0., -.0036])
        anchor = reference[:3, 3] + reference[:3, :3] @ pad + [.00012, -.00002, 0.]
        problem = CoupledHandlePoint(chain, rigid.origin, rigid.axis, reference,
            reference_anchor=anchor, pad_local=pad, orientation_cone=np.pi / 2,
            progress_bounds=(-.1, .3))
        result = project_nearby(problem, start, fixed_progress=0., residual_tolerance=1e-7)
        self.assertTrue(result['accepted'], result)
        self.assertLess(result['position_error_m'], 1e-7)
        self.assertLess(result['maximum_joint_correction_rad'], .002)
        self.assertEqual(result['state'][7], 0.)

    def test_equality_fit_does_not_override_other_state_inequalities(self):
        problem = LinearPoint([0., 0., 0.])
        problem.satisfied = lambda x: False
        result = project_nearby(problem, np.zeros(8))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'equality_satisfied_but_state_inequality_failed')


if __name__ == '__main__':
    unittest.main()
