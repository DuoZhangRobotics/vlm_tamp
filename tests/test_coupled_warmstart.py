"""Warm path retargeting keeps the requested goal and rejects invalid branches."""
import unittest
import numpy as np

from vlm_tamp.coupled_warmstart import retarget_path


class ShiftedLine:
    codimension = 3

    def __init__(self, bias):
        self.bias = bias
        self.bounds = np.tile([-2., 2.], (8, 1))
        self.bounds[7] = [0., 1.]

    def evaluate(self, x):
        jacobian = np.zeros((3, 8))
        jacobian[:, :3] = np.eye(3)
        jacobian[0, 7] = -1.
        return x[:3] - [x[7] + self.bias, 0., 0.], jacobian

    def errors(self, x):
        norm = float(np.linalg.norm(self.evaluate(x)[0]))
        return {'position_error_m': norm, 'orientation_error_rad': 0., 'weighted_residual_norm': norm}

    def satisfied(self, x):
        return bool(np.all(x >= self.bounds[:, 0]) and np.all(x <= self.bounds[:, 1])
                    and self.errors(x)['weighted_residual_norm'] <= 1e-5)


class WarmStartTests(unittest.TestCase):
    def path(self):
        path = np.zeros((3, 8))
        path[:, 0], path[:, 7] = [0., .05, .1], [0., .0505, .101]
        path[:, 6] = .4
        return path

    def test_retargets_changed_model_and_goal_without_losing_nullspace_branch(self):
        problem = ShiftedLine(.0002)
        start = self.path()[0].copy()
        start[0] = .0002
        path, result = retarget_path(problem, start, self.path(), lambda x: True, native_validation=False)
        self.assertTrue(result['retargeted'], result)
        self.assertFalse(result['exact_solution'])  # Native edges are still pending.
        self.assertTrue(result['native_validation_pending'])
        np.testing.assert_allclose(path[:, 6], .4, atol=0)
        self.assertAlmostEqual(path[-1, 7], .101)
        self.assertLess(result['maximum_joint_projection_correction_rad'], .002)
        self.assertTrue(all(problem.satisfied(x) for x in path))

    def test_infeasible_local_retarget_cannot_return_a_partial_success(self):
        problem = ShiftedLine(.2)
        path, result = retarget_path(problem, self.path()[0], self.path(), lambda x: True, native_validation=False)
        self.assertEqual(path.shape, (0, 8))
        self.assertFalse(result['exact_solution'])
        self.assertTrue(result['failure_is_start'])

    def test_wider_hypothetical_retarget_preserves_already_projected_start(self):
        problem = ShiftedLine(.08)
        start = self.path()[0].copy()
        start[0] = .08  # The executor has already checked this measured binding.
        narrow, failure = retarget_path(problem, start, self.path(), lambda x: True,
            native_validation=False)
        self.assertEqual(narrow.shape, (0, 8))
        self.assertFalse(failure['failure_is_start'])
        wide, result = retarget_path(problem, start, self.path(), lambda x: True,
            maximum_joint_correction_rad=.2, native_validation=False)
        np.testing.assert_array_equal(wide[0], start)
        self.assertTrue(all(problem.satisfied(x) for x in wide))
        self.assertGreater(result['maximum_joint_projection_correction_rad'], .04)
        self.assertFalse(result['exact_solution'])  # Full native edge checks remain mandatory.
        self.assertTrue(result['native_validation_pending'])

    def test_new_interior_obstacle_rejects_prior_path(self):
        problem = ShiftedLine(0.)
        path, result = retarget_path(problem, self.path()[0], self.path(),
            lambda x: not (.04 < x[7] < .07), native_validation=False)
        self.assertEqual(path.shape, (0, 8))
        self.assertFalse(result['exact_solution'])
        self.assertEqual(result['failed_state'], 1)


if __name__ == '__main__':
    unittest.main()
