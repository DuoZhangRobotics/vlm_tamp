"""Fast calibrated seven-joint IK and complete manipulation admission."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import interpolate_path, matrix_pose, pose_matrix
from .robot_chain_v4 import RobotChain, fixed_orientation_residual


class TaskIK:
    def __init__(self, robot_record, config, self_valid=None):
        self.chain = RobotChain(robot_record)
        self.config = config
        self.self_valid = self_valid or (lambda _q: True)

    def solve(self, target_pose, seed, seed_count=None):
        target = pose_matrix(target_pose)
        seed = np.asarray(seed, float)
        low, high = self.chain.limits.T
        low, high = low + .0001, high - .0001
        count = self.config['ik_seed_count'] if seed_count is None else seed_count
        generator = np.random.default_rng(0)
        starts = [np.clip(seed, low, high)]
        for index in range(1, count):
            value = (self.chain.reference_joints if index == count - 1 else seed + generator.normal(0, .4, 7))
            starts.append(np.clip(value, low, high))
        for initial in starts:
            cached_q, cached_value, cached_jacobian = None, None, None

            def both(q):
                nonlocal cached_q, cached_value, cached_jacobian
                if cached_q is None or not np.array_equal(cached_q, q):
                    pose, jacobian = self.chain.forward(q, jacobian=True)
                    matrix = pose_matrix(pose)
                    angle, angular_jacobian = fixed_orientation_residual(target[:3, :3], matrix[:3, :3], jacobian[3:])
                    cached_value = np.r_[matrix[:3, 3] - target[:3, 3], .15 * angle]
                    cached_jacobian = np.vstack([jacobian[:3], .15 * angular_jacobian])
                    cached_q = q.copy()
                return cached_value, cached_jacobian

            solved = least_squares(lambda q: both(q)[0], initial, jac=lambda q: both(q)[1],
                                   bounds=(low, high), max_nfev=self.config['ik_maximum_evaluations'],
                                   ftol=1e-8, xtol=1e-8, gtol=1e-8)
            actual_pose = self.chain.forward(solved.x)
            actual = pose_matrix(actual_pose)
            position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            angle_error = float(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude())
            if (position_error <= self.config['ik_position_tolerance_m']
                    and angle_error <= self.config['ik_orientation_tolerance_rad']
                    and self.self_valid(solved.x)):
                return {'joints': solved.x.tolist(), 'pose_world_xyzw': actual_pose.tolist(),
                        'position_error_m': position_error, 'orientation_error_rad': angle_error,
                        'evaluations': solved.nfev}
        return None

    def path(self, poses, seed):
        solutions = []
        previous = np.asarray(seed, float)
        for index, pose in enumerate(poses):
            solution = self.solve(pose, previous, seed_count=1)
            if solution is None:
                return {'accepted': False, 'reason': 'manipulation_pose_unreachable', 'sample': index}
            following = np.asarray(solution['joints'])
            if np.max(np.abs(following - previous)) > .5:
                return {'accepted': False, 'reason': 'manipulation_joint_discontinuity', 'sample': index}
            solutions.append(solution)
            previous = following
        return {'accepted': True, 'solutions': solutions,
                'maximum_position_error_m': max((item['position_error_m'] for item in solutions), default=0.),
                'maximum_orientation_error_rad': max((item['orientation_error_rad'] for item in solutions), default=0.),
                'scene_collision_checked': False}

    def approach(self, initial_joints, grasp_solution):
        target = np.asarray(grasp_solution['pose_world_xyzw'])
        target_joints = np.asarray(grasp_solution['joints'])
        standoff = target.copy()
        standoff[:3] -= .07 * pose_matrix(target)[:3, 2]
        pre = self.solve(standoff, target_joints)
        if pre is None:
            return {'accepted': False, 'reason': 'pregrasp_unreachable'}
        pre_joints = np.asarray(pre['joints'])
        count = max(1, int(np.ceil(np.max(np.abs(pre_joints - initial_joints)) / .12)))
        joint_path = [np.asarray(initial_joints) + weight * (pre_joints - initial_joints)
                      for weight in np.linspace(0., 1., count + 1)]
        if not all(self.self_valid(q) for q in joint_path):
            return {'accepted': False, 'reason': 'approach_self_collision'}
        poses = [self.chain.forward(q).tolist() for q in joint_path]
        final_path = self.path(interpolate_path([pre['pose_world_xyzw'], target], position_step=.01)[1:], pre_joints)
        if not final_path['accepted']:
            return final_path
        joint_path.extend(item['joints'] for item in final_path['solutions'])
        poses.extend(item['pose_world_xyzw'] for item in final_path['solutions'])
        return {'accepted': True, 'poses_world_xyzw': poses, 'joint_postures': np.asarray(joint_path).tolist(),
                'all_waypoints_fk_bound': True, 'environment_collision_checked': False,
                'self_collision_checked': self.config['self_collision_checks']}


def manipulation_poses(spec, grasp_pose, config):
    """Nominal attachment before closure; actual observed binding is rechecked."""
    grasp_pose = np.asarray(grasp_pose, float)
    if spec['task_type'] == 'pick_place':
        lift = grasp_pose.copy()
        lift[2] += config['lift_distance_m']
        release = grasp_pose.copy()
        release[:3] += np.asarray(spec['goal']['target_world_m']) - spec['object_support_point_initial_world_m']
        over = release.copy()
        over[2] = max(lift[2], release[2] + config['lift_distance_m'])
        retreat = release.copy()
        retreat[2] += .05
        knots = [grasp_pose, lift, over, release, retreat]
    else:
        end = grasp_pose.copy()
        end[:3] += config['drawer_maximum_command_travel_m'] * np.asarray(spec['pull_axis_world'])
        knots = [grasp_pose, end]
    return interpolate_path(knots, position_step=config['preflight_sampling_m'])


def projected_bowl_pose(spec, tcp):
    """Return the point-target TCP pose for the already observed binding."""
    transform = pose_matrix(tcp)
    transform[:3, 3] = np.asarray(spec['target_world_m']) - transform[:3, :3] @ spec['point_local_m']
    return matrix_pose(transform)
