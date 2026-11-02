"""cuRobo IK results, unchanged, for the native motion planner.

FK errors are reported, not used for a second IK acceptance decision. cuRobo
optimizes with shared observed points and native VAMP robot spheres/self pairs.
The native planner owns complete path search. Point continuation uses cuRobo.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from .curobo_client import CuroboClient
from .curobo_model import export_model
from .geometry import interpolate_path, pose_matrix
from .task_ik_v6 import TaskIK as BaseIK


def joint_reserve(q, limits):
    return float(np.minimum(np.asarray(q)-limits[:, 0], limits[:, 1]-q).min())


def posture_cost(q, seed, limits):
    return float(np.linalg.norm(q-seed))


class TaskIK(BaseIK):
    def __init__(self, robot_record, config, self_valid=None, *, output, sim=None):
        super().__init__(robot_record, config, self_valid)
        self.output = Path(output)
        self.reserve = float(config.get('command_joint_reserve_rad',0.))
        model = export_model(robot_record, self.output/'curobo-model', self.reserve)
        from .curobo_collision_model import calibrated_spheres, native_self_collision_pairs
        world_robot = np.linalg.inv(np.asarray(robot_record['calibration']['world_to_robot']))
        self.world_robot = world_robot
        spheres, provenance = calibrated_spheres(self.chain, None, world_robot, include_self=True)
        model['kinematics'].update(collision_link_names=list(spheres), collision_spheres=spheres,
            collision_sphere_buffer=0., self_collision_ignore={}, self_collision_buffer={})
        model['native_sphere_indices'] = [i for group in provenance['groups'] for i in group['native_indices']]
        model['native_self_collision_pairs'] = native_self_collision_pairs()
        self.client = CuroboClient(model, config, self.output/'curobo.log')
        self.events = []
        self._save('curobo-runtime.json', self.client.metadata)
        try:
            self.validate_frames(sim)
        except BaseException:
            self.client.close()
            raise

    def configure_scene(self, context):
        """Give IK the exact observed point partitions already prepared for VAMP."""
        return self._configure_clouds(context.ik_clouds_world, bool(context.record.get('palm_contact_allowed')))

    def configure_observed_scene(self, scene):
        """Prepare the shared points for a direct IK stroke without building a path planner."""
        from vlm_tamp.collision_scene_points import planner_points, world_points
        clouds, _ = planner_points(scene, self.world_robot)
        return self._configure_clouds(world_points(clouds, self.world_robot),
            getattr(scene, 'skill_contact_surface', None) == 'palm')

    def _configure_clouds(self, clouds, palm_contact):
        import hashlib
        path = self.output/'curobo-observed-scene.npz'
        np.savez(path, **clouds)
        result = self.client.request('configure_scene', arrays_path=str(path.resolve()),
            arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            palm_contact=palm_contact)
        self.scene_setup = result
        return result

    def move_scene_target(self, transform):
        """Move the observed target for this goal; fixed geometry remains fixed."""
        return self.client.request('move_scene_target', transform=np.asarray(transform).tolist())

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
        solutions, rejected = [], {'solver': 0, 'invalid_output': 0, 'duplicate': 0}
        proposals = list(zip(result['solutions'], result['success']));optimizer_failed_flags=0
        if self.config.get('curobo_query_reproducible', False):
            proposals.sort(key=lambda item: (posture_cost(np.asarray(item[0]), seed, self.chain.limits)
                if np.isfinite(item[0]).all() else float('inf'), tuple(item[0])))
        for q, success in proposals:
            q = np.asarray(q, float)
            if not success:
                optimizer_failed_flags += 1
                rejected['solver'] += 1
                continue
            if q.shape != (7,) or not np.isfinite(q).all():
                rejected['invalid_output'] += 1
                continue
            if any(np.array_equal(q, other['joints']) for other in solutions):
                rejected['duplicate'] += 1
                continue
            pose = self.chain.forward(q)
            matrix = pose_matrix(pose)
            position = float(np.linalg.norm(matrix[:3, 3]-target[:3, 3]))
            angle = float(Rotation.from_matrix(target[:3, :3].T@matrix[:3, :3]).magnitude())
            solutions.append({'joints': q.tolist(), 'pose_world_xyzw': pose.tolist(),
                'position_error_m': position, 'orientation_error_rad': angle,
                'joint_reserve_rad': joint_reserve(q, self.chain.limits),
                'posture_cost': posture_cost(q, seed, self.chain.limits), 'backend': 'curobo',
                'scene_collision_checked': result.get('scene_collision_checked', False),'solver_reported_success':bool(success),
                'candidate_modified': False, 'joint_bound_projection_rad': 0.,
                'collision_admission': 'cuRobo IK feasibility, then native path planning'})
        solutions.sort(key=lambda row: row['posture_cost'])
        for rank, solution in enumerate(solutions):
            solution['posture_rank'] = rank
        self.events.append({'query': len(self.events), 'pose': np.asarray(target_pose).tolist(),
            'returned': len(result['solutions']), 'distinct_solver_successes': len(solutions),
            'acceptance_authority': 'cuRobo success flag; no CPU rescue or pose correction',
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
        return min(solutions, key=lambda row: np.linalg.norm(np.asarray(row['joints'])-seed)) if solutions else None

    def contact_solutions(self, target_world: ArrayLike, contact_local: ArrayLike, seed: ArrayLike,
                          orientation_reference: ArrayLike, *, local: bool,
                          contact_normal_local: ArrayLike | None = None,
                          surface_normal_world: ArrayLike | None = None) -> list[dict[str, Any]]:
        """Request native point/normal IK; report FK errors without repairing solutions."""
        began = time.perf_counter()
        seed, point, offset = [np.asarray(x, float) for x in (seed, target_world, contact_local)]
        oriented = contact_normal_local is not None or surface_normal_world is not None
        if oriented and (contact_normal_local is None or surface_normal_world is None):
            raise ValueError('Supply both robot contact normal and signed surface normal')
        normal_arguments = {}
        if oriented:
            from vlm_tamp.contact_orientation import normal_vector
            normal_arguments = {
                'contact_normal_local': normal_vector(contact_normal_local).tolist(),
                'surface_normal_world': normal_vector(surface_normal_world).tolist()}
        result = self.client.request('solve_contact', point=point.tolist(),
            contact_local=offset.tolist(), seed=seed.tolist(),
            reference_quaternion_xyzw=Rotation.from_matrix(orientation_reference).as_quat().tolist(), local=local,
            **normal_arguments)
        candidates = []
        for values, success in zip(result['solutions'], result['success']):
            q = np.asarray(values, float)
            if not success or q.shape != (7,) or not np.isfinite(q).all():
                continue
            pose = self.chain.forward(q); transform = pose_matrix(pose)
            error = float(np.linalg.norm(transform[:3, 3]+transform[:3, :3]@offset-point))
            candidates.append({'joints': q.tolist(), 'pose_world_xyzw': pose.tolist(),
                'position_error_m': error, 'solver_reported_success': bool(success),
                'orientation_is_soft_preference': False, 'candidate_modified': False,
                'backend': result['backend'],
                'scene_collision_checked': result.get('scene_collision_checked', False),
                'self_collision_checked': result.get('self_collision_checked', False)})
            if oriented:
                from vlm_tamp.contact_orientation import contact_orientation_report
                candidates[-1]['contact_orientation'] = contact_orientation_report(
                    transform[:3, :3], contact_normal_local, surface_normal_world)
        self.events.append({'query': len(self.events), 'backend': result['backend'],
            'contact_constraint': 'surface_normal' if oriented else 'point',
            'seed_count': result.get('seed_count'), 'returned': len(result['solutions']),
            'distinct_solver_successes': len(candidates), 'elapsed_seconds': result['elapsed_seconds'],
            'scene_collision_checked': result.get('scene_collision_checked', False),
            'self_collision_checked': result.get('self_collision_checked', False),
            'constraint_feasible_seeds': sum(result.get('feasible', [])),
            'minimum_position_error_m': min(result.get('position_error_m', []), default=None),
            'closest_posture_distance_rad': min((float(np.linalg.norm(np.asarray(c['joints'])-seed))
                for c in candidates), default=None),
            **{k:result[k] for k in ('solver_created','first_solve_for_contact_solver',
                'solver_setup_seconds','solve_call_seconds','solver_reported_seconds') if k in result},
            'wall_seconds': time.perf_counter()-began})
        return sorted(candidates, key=lambda row: np.linalg.norm(np.asarray(row['joints'])-seed))

    def solve_contact(self, target_world: ArrayLike, contact_local: ArrayLike, seed: ArrayLike,
                      orientation_reference: ArrayLike, *, contact_normal_local: ArrayLike | None = None,
                      surface_normal_world: ArrayLike | None = None) -> dict[str, Any] | None:
        """Return the closest native continuation solution for the selected model."""
        candidates = self.contact_solutions(target_world, contact_local, seed, orientation_reference, local=True,
            contact_normal_local=contact_normal_local, surface_normal_world=surface_normal_world)
        return candidates[0] if candidates else None

    def solve_contact_many(self, target_world: ArrayLike, contact_local: ArrayLike, seed: ArrayLike,
                           orientation_reference: ArrayLike, *, contact_normal_local: ArrayLike | None = None,
                           surface_normal_world: ArrayLike | None = None) -> list[dict[str, Any]]:
        """Return the existing acquisition batch without model substitution."""
        return self.contact_solutions(target_world, contact_local, seed, orientation_reference, local=False,
            contact_normal_local=contact_normal_local, surface_normal_world=surface_normal_world)

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
        """Compatibility entry point; no SciPy point solve or CPU refinement."""
        reference = (pose_matrix(self.chain.forward(seed))[:3, :3]
                     if orientation_reference is None else np.asarray(orientation_reference))
        return self.solve_contact(target_world, point_local, seed, reference)

    def close(self):
        self._save('curobo-queries.json', self.events)
        self.client.close()
