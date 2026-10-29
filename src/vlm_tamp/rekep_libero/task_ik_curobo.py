"""Multiple GPU IK proposals, robot-model validation, and joint-limit reserve.

cuRobo generates poses. VAMP validates self collision; the grasp selector and
executor validate observed scene geometry and intended contacts. Point-goal
placement retains a bounded CPU refinement because orientation is free there.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from vlm_tamp.pose_constraints import skew
from .curobo_client import CuroboClient
from .curobo_model import export_model
from .geometry import interpolate_path, pose_matrix
from .robot_chain_v4 import fixed_orientation_residual
from .task_ik_v6 import TaskIK as BaseIK


def joint_reserve(q, limits):
    return float(np.minimum(np.asarray(q)-limits[:, 0], limits[:, 1]-q).min())


def posture_cost(q, seed, limits):
    gaps = np.minimum(q-limits[:, 0], limits[:, 1]-q)
    return float(np.linalg.norm(q-seed) + .015*np.sum(1/(gaps+.05)))


class TaskIK(BaseIK):
    def __init__(self, robot_record, config, self_valid=None, *, output, sim=None):
        super().__init__(robot_record, config, self_valid)
        self.output = Path(output)
        self.reserve = float(config.get('command_joint_reserve_rad',0.))
        model = export_model(robot_record, self.output/'curobo-model', self.reserve)
        self.client = CuroboClient(model, config, self.output/'curobo.log')
        self.events = []
        self._save('curobo-runtime.json', self.client.metadata)
        try:
            self.validate_frames(sim)
        except BaseException:
            self.client.close()
            raise

    def _save(self, name, value):
        (self.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    def validate_frames(self, sim=None):
        low, high = self.chain.limits.T
        rng = np.random.default_rng(0)
        qs = np.vstack([self.chain.reference_joints,
                        rng.uniform(low+self.reserve, high-self.reserve, (24, 7))])
        actual = self.client.forward(qs.tolist())
        errors = []
        for index, (q, actual_pose) in enumerate(zip(qs, actual)):
            reference = pose_matrix(self.chain.forward(q))
            computed = pose_matrix(actual_pose)
            position = float(np.linalg.norm(reference[:3, 3]-computed[:3, 3]))
            angle = float(Rotation.from_matrix(reference[:3, :3].T@computed[:3, :3]).magnitude())
            row = {'sample': index, 'position_error_m': position, 'orientation_error_rad': angle}
            if sim is not None and index < 4:
                measured = pose_matrix(sim.fk(q)['eef_pose_world_xyzw'])
                row['isolated_sim_position_error_m'] = float(np.linalg.norm(measured[:3, 3]-computed[:3, 3]))
                row['isolated_sim_orientation_error_rad'] = float(Rotation.from_matrix(measured[:3, :3].T@computed[:3, :3]).magnitude())
            errors.append(row)
        accepted = (len(actual) == len(qs) and all(
            row['position_error_m'] < 2e-5 and row['orientation_error_rad'] < 2e-4
            and row.get('isolated_sim_position_error_m', 0) < 2e-5
            and row.get('isolated_sim_orientation_error_rad', 0) < 2e-4 for row in errors))
        self._save('curobo-frame-validation.json', {'accepted': accepted, 'samples': errors,
            'robot_only_calibration': True, 'quaternion_conversion': 'LIBERO xyzw to cuRobo wxyz',
            'tcp': 'controller grip_site, not stock panda_hand'})
        if not accepted:
            raise ValueError('cuRobo calibrated FK/TCP does not match LIBERO')

    def solve_many(self, target_pose, seed, maximum=None):
        seed = np.asarray(seed, float)
        query_started = time.perf_counter()
        target = pose_matrix(target_pose)
        rpc_started = time.perf_counter()
        result = self.client.solve(np.asarray(target_pose).tolist(), seed.tolist())
        rpc_seconds = time.perf_counter()-rpc_started
        postprocess_started = time.perf_counter()
        solutions, rejected = [], {'solver': 0, 'bounds': 0, 'residual': 0, 'duplicate': 0, 'self_collision': 0}
        proposals = list(zip(result['solutions'], result['success']));optimizer_failed_flags=0
        if self.config.get('curobo_query_reproducible', False):
            proposals.sort(key=lambda item: (posture_cost(np.asarray(item[0]), seed, self.chain.limits)
                if np.isfinite(item[0]).all() else float('inf'), tuple(item[0])))
        for q, success in proposals:
            q = np.asarray(q, float)
            if not success:
                # Optimizer convergence is diagnostic. Independent robot FK,
                # requested residuals, real bounds and self geometry decide.
                optimizer_failed_flags += 1
            if not np.isfinite(q).all():
                rejected['bounds'] += 1
                continue
            # Proposals may round beyond a float32 boundary or need a box
            # projection. The projected candidate must still pass actual FK
            # residual and self checks; no out-of-range command is admitted.
            proposed=q.copy()
            q=np.clip(q,self.chain.limits[:,0]+self.reserve,self.chain.limits[:,1]-self.reserve)
            pose = self.chain.forward(q)
            matrix = pose_matrix(pose)
            position = float(np.linalg.norm(matrix[:3, 3]-target[:3, 3]))
            angle = float(Rotation.from_matrix(target[:3, :3].T@matrix[:3, :3]).magnitude())
            if position > self.config['ik_position_tolerance_m'] or angle > self.config['ik_orientation_tolerance_rad']:
                rejected['residual'] += 1
                continue
            if any(np.max(np.abs(q-np.asarray(other['joints']))) < .025 for other in solutions):
                rejected['duplicate'] += 1
                continue
            if not self.self_valid(q):
                rejected['self_collision'] += 1
                continue
            solutions.append({'joints': q.tolist(), 'pose_world_xyzw': pose.tolist(),
                'position_error_m': position, 'orientation_error_rad': angle,
                'joint_reserve_rad': joint_reserve(q, self.chain.limits),
                'posture_cost': posture_cost(q, seed, self.chain.limits), 'backend': 'curobo',
                'scene_collision_checked': False,'solver_reported_success':bool(success),
                'joint_bound_projection_rad':float(np.max(np.abs(q-proposed)))})
        solutions.sort(key=lambda row: row['posture_cost'])
        for rank, solution in enumerate(solutions):
            solution['posture_rank'] = rank
        self.events.append({'query': len(self.events), 'pose': np.asarray(target_pose).tolist(),
            'returned': len(result['solutions']), 'distinct_robot_valid': len(solutions),
            'rejected': rejected,'optimizer_failed_flags':optimizer_failed_flags,
            'admitted_failed_flag_candidates':sum(not s['solver_reported_success'] for s in solutions), 'elapsed_seconds': result['elapsed_seconds'],
            'worker_rpc_seconds': rpc_seconds,
            'cpu_postprocessing_seconds': time.perf_counter()-postprocess_started,
            'wall_seconds': time.perf_counter()-query_started,
            'peak_gpu_allocated_bytes': result['peak_gpu_allocated_bytes'],
            **{k: result[k] for k in ('seed_pool_sha256', 'query_sha256', 'explicit_seed_count') if k in result}})
        return solutions if maximum is None else solutions[:maximum]

    def solve(self, target_pose, seed, seed_count=None):
        solutions = self.solve_many(target_pose, seed)
        # Joint reserve is already a hard bound. During local continuation,
        # preserve the selected branch instead of trading motion for reserve.
        return min(solutions, key=lambda row: np.linalg.norm(np.asarray(row['joints'])-seed)) if solutions else None

    def approach(self, initial_joints, grasp_solution):
        target = np.asarray(grasp_solution['pose_world_xyzw'])
        target_q = np.asarray(grasp_solution['joints'])
        standoff = target.copy()
        standoff[:3] -= .07*pose_matrix(target)[:3, 2]
        reverse_poses = interpolate_path([target, standoff], position_step=.01)
        reverse_q, reverse_fk = [target_q], [target]
        previous = target_q
        for pose in reverse_poses[1:]:
            solution = self.solve(pose, previous, seed_count=1)
            if solution is None:
                return {'accepted': False, 'reason': 'pregrasp_unreachable'}
            q = np.asarray(solution['joints'])
            if np.max(np.abs(q-previous)) > .35:
                return {'accepted': False, 'reason': 'approach_joint_discontinuity'}
            reverse_q.append(q)
            reverse_fk.append(np.asarray(solution['pose_world_xyzw']))
            previous = q
        initial = np.asarray(initial_joints)
        count = max(1, int(np.ceil(np.max(np.abs(previous-initial))/.12)))
        joint_path = [initial+w*(previous-initial) for w in np.linspace(0, 1, count+1)]
        if not all(self.self_valid(q) for q in joint_path):
            return {'accepted': False, 'reason': 'approach_self_collision'}
        poses = [self.chain.forward(q) for q in joint_path]
        joint_path.extend(reverse_q[-2::-1])
        poses.extend(reverse_fk[-2::-1])
        if not np.array_equal(np.asarray(joint_path[-1]), target_q):
            raise ValueError('Approach must terminate at the admitted grasp posture')
        return {'accepted': True, 'poses_world_xyzw': np.asarray(poses).tolist(),
                'joint_postures': np.asarray(joint_path).tolist(),
                'all_waypoints_fk_bound': True, 'environment_collision_checked': False,
                'self_collision_checked': True, 'endpoint_matches_selected_grasp': True,
                'method': 'backward Cartesian continuation anchored at selected grasp; joint interpolation to standoff'}

    def solve_point(self, target_world, point_local, seed, orientation_reference=None, seed_count=2):
        seed = np.asarray(seed, float)
        target_world, point_local = np.asarray(target_world, float), np.asarray(point_local, float)
        reference = (pose_matrix(self.chain.forward(seed))[:3, :3] if orientation_reference is None
                     else np.asarray(orientation_reference))
        low, high = self.chain.limits.T
        low, high = low+self.reserve, high-self.reserve
        rng, solutions = np.random.default_rng(0), []
        for index in range(seed_count):
            initial = np.clip(seed if index == 0 else seed+rng.normal(0, .3, 7), low, high)
            cached, residual, derivative = None, None, None

            def both(q):
                nonlocal cached, residual, derivative
                if cached is None or not np.array_equal(q, cached):
                    pose, jac = self.chain.forward(q, jacobian=True)
                    matrix = pose_matrix(pose)
                    rotation = matrix[:3, :3]
                    point = matrix[:3, 3]+rotation@point_local
                    jp = jac[:3]-rotation@skew(point_local)@jac[3:]
                    angle, ja = fixed_orientation_residual(reference, rotation, jac[3:])
                    residual = np.r_[point-target_world, .015*angle, .00005*(q-seed)]
                    derivative = np.vstack([jp, .015*ja, .00005*np.eye(7)])
                    cached = q.copy()
                return residual, derivative

            result = least_squares(lambda q: both(q)[0], initial, jac=lambda q: both(q)[1],
                bounds=(low, high), max_nfev=100, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            pose = self.chain.forward(result.x)
            matrix = pose_matrix(pose)
            error = float(np.linalg.norm(matrix[:3, 3]+matrix[:3, :3]@point_local-target_world))
            if error <= self.config['ik_position_tolerance_m'] and self.self_valid(result.x):
                solutions.append({'joints': result.x.tolist(), 'pose_world_xyzw': pose.tolist(),
                    'position_error_m': error,
                    'orientation_error_rad': float(Rotation.from_matrix(reference.T@matrix[:3, :3]).magnitude()),
                    'orientation_is_soft_preference': True,
                    'joint_reserve_rad': joint_reserve(result.x, self.chain.limits),
                    'posture_cost': posture_cost(result.x, seed, self.chain.limits),
                    'backend': 'bounded CPU point-goal refinement', 'evaluations': result.nfev})
        return min(solutions, key=lambda row: row['posture_cost']) if solutions else None

    def close(self):
        self._save('curobo-queries.json', self.events)
        self.client.close()
