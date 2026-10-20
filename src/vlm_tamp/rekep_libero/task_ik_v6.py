"""Point-goal placement with soft orientation and measured surface admission."""

import numpy as np
from scipy.optimize import least_squares

from vlm_tamp.pose_constraints import skew
from .geometry import pose_matrix
from .robot_chain_v4 import fixed_orientation_residual
from .task_ik_v5 import TaskIK as PoseTaskIK


class TaskIK(PoseTaskIK):
    def solve_point(self, target_world, point_local, seed, orientation_reference=None, seed_count=2):
        seed = np.asarray(seed, float)
        target_world, point_local = np.asarray(target_world, float), np.asarray(point_local, float)
        reference = (pose_matrix(self.chain.forward(seed))[:3, :3] if orientation_reference is None
                     else np.asarray(orientation_reference))
        low, high = self.chain.limits.T
        low, high = low + .0001, high - .0001
        generator = np.random.default_rng(0)
        for index in range(seed_count):
            initial = np.clip(seed if index == 0 else seed + generator.normal(0, .3, 7), low, high)
            cached_q, value, derivative = None, None, None

            def both(q):
                nonlocal cached_q, value, derivative
                if cached_q is None or not np.array_equal(q, cached_q):
                    pose, jac = self.chain.forward(q, jacobian=True)
                    matrix = pose_matrix(pose)
                    rotation = matrix[:3, :3]
                    point = matrix[:3, 3] + rotation @ point_local
                    jp = jac[:3] - rotation @ skew(point_local) @ jac[3:]
                    angle, ja = fixed_orientation_residual(reference, rotation, jac[3:])
                    value = np.r_[point - target_world, .015 * angle, .00005 * (q - seed)]
                    derivative = np.vstack([jp, .015 * ja, .00005 * np.eye(7)])
                    cached_q = q.copy()
                return value, derivative

            solved = least_squares(lambda q: both(q)[0], initial, jac=lambda q: both(q)[1], bounds=(low, high),
                                   max_nfev=100, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            pose = self.chain.forward(solved.x)
            matrix = pose_matrix(pose)
            error = float(np.linalg.norm(matrix[:3, 3] + matrix[:3, :3] @ point_local - target_world))
            if error <= self.config['ik_position_tolerance_m'] and self.self_valid(solved.x):
                angle, _ = fixed_orientation_residual(reference, matrix[:3, :3], np.eye(3))
                return {'joints': solved.x.tolist(), 'pose_world_xyzw': pose.tolist(),
                        'position_error_m': error, 'orientation_error_rad': float(np.linalg.norm(angle)),
                        'orientation_is_soft_preference': True, 'evaluations': solved.nfev}
        return None

    def point_path(self, targets_world, point_local, seed):
        reference = pose_matrix(self.chain.forward(seed))[:3, :3]
        solutions, previous = [], np.asarray(seed, float)
        for index, target in enumerate(targets_world):
            solution = self.solve_point(target, point_local, previous, reference)
            if solution is None:
                return {'accepted': False, 'reason': 'manipulation_point_unreachable', 'sample': index}
            q = np.asarray(solution['joints'])
            if np.max(np.abs(q - previous)) > .5:
                return {'accepted': False, 'reason': 'manipulation_joint_discontinuity', 'sample': index}
            solutions.append(solution)
            previous = q
        return {'accepted': True, 'solutions': solutions, 'orientation_is_soft_preference': True,
                'maximum_position_error_m': max(item['position_error_m'] for item in solutions),
                'scene_collision_checked': False}
