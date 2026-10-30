"""cuRobo v2 contact-point goals followed by exact native admission.

The GPU sees only observed scene surfaces and calibrated robot geometry. Its
arm/palm proxy guides optimization; it never replaces the finger contact rules,
partial-pose equality, wrist limit or complete constrained-path checks.
"""
from pathlib import Path
from collections import Counter
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .coupled_storage import read_json, write_json
from .partial_pose_arc import PartialPoseArc, short_wrist_limit, wrist_excursion, path_wrist_admitted
from .rekep_libero.curobo_collision_model import calibrated_spheres, expected_spheres
from .rekep_libero.geometry import pose_matrix


class CuroboV2ArcProposer:
    def __init__(self, ik, context, folder, goal):
        if getattr(ik.client, 'backend', None) != 'v2':
            raise ValueError('cuRobo v2 arc proposals require the v2 worker')
        self.ik, self.context, self.folder = ik, context, Path(folder)
        self.goal = float(goal)
        options = getattr(context, 'motion_options', context.scene.config)
        self.solver_mode = options.get('curobo_v2_arc_solver_mode', 'local_continuation')
        self.global_fallback_enabled = bool(options.get('curobo_v2_global_fallback', True))
        self.step_degrees = float(options.get('arc_continuation_step_deg', 3.))
        self.speed_degrees_per_second = float(options.get('arc_continuation_speed_deg_s', 15.))
        if (not np.isfinite(self.goal) or self.goal <= 0 or
                not np.isfinite(self.step_degrees) or self.step_degrees <= 0 or
                not np.isfinite(self.speed_degrees_per_second) or self.speed_degrees_per_second <= 0):
            raise ValueError('Arc goal, continuation step and speed must be finite and positive')
        self.steps = max(1, int(np.ceil(np.degrees(self.goal)/self.step_degrees)))
        self.local_dt = np.degrees(self.goal)/self.steps/self.speed_degrees_per_second
        self.closed = False
        self.model = PartialPoseArc(context.chain, context.origin, context.axis,
            context.reference_q, context.pad_local, grasp_constraint=context.grasp_constraint,
            position_tolerance_m=context.scene.config.get('arc_position_tolerance_m', .003),
            orientation_tolerance_rad=context.scene.config.get('arc_orientation_tolerance_rad',0.))
        self.queries = []
        before = time.perf_counter()
        aperture=tuple(np.asarray(getattr(context,'metadata',{}).get('gripper_qpos',[]),float).round(6))
        if not hasattr(ik, '_v2_arc_spheres') or getattr(ik,'_v2_arc_aperture',None)!=aperture:
            ik._v2_arc_spheres = calibrated_spheres(context.chain, context.scene.robot, context.world_robot)
            ik._v2_arc_aperture=aperture
        spheres, provenance = ik._v2_arc_spheres
        reference = read_json(self.folder/'scene-reference.json')
        self.configure_request = dict(collision_spheres=spheres,
            position_tolerance=self.model.position_tolerance_m,
            pad_local=context.pad_local.tolist(), scene_arrays_path=reference['arrays_path'],
            scene_arrays_sha256=reference['arrays_sha256'], origin=context.origin.tolist(),
            axis=context.axis.tolist(), mesh_pitch=float(context.scene.config.get('curobo_v2_mesh_pitch_m', .005)),
            solver_mode=self.solver_mode, local_dt=float(self.local_dt),
            moving_mesh_pitch=float(options.get('curobo_v2_moving_mesh_pitch_m', .002)),
            local_seed_std=float(options.get('curobo_v2_local_seed_std_rad', .02)),
            local_velocity_weight=float(options.get('curobo_v2_local_velocity_weight', .001)))
        self.setup = ik.client.request('configure_arc', **self.configure_request)
        self.arc_id = self.setup['arc_id']
        try:
            self._validate_frames(provenance, before)
        except BaseException:
            self.close()
            raise

    def _validate_frames(self, provenance, before):
        context = self.context
        probes = [context.chain.reference_joints, context.reference_q]
        gpu = self.ik.client.request('arc_forward', joints=np.asarray(probes).tolist(), arc_id=self.arc_id)
        checks = []
        for q, actual_spheres, actual_point in zip(probes, gpu['spheres_world'], gpu['contact_points_world']):
            expected = expected_spheres(context.scene.robot, provenance['groups'], q)
            actual = np.asarray(actual_spheres)
            # cuRobo can pad the collision sphere array with disabled entries.
            actual = actual[actual[:, 3] > 0]
            if expected.shape != actual.shape:
                raise ValueError(f'GPU sphere shape mismatch: {actual.shape} vs {expected.shape}')
            tcp = pose_matrix(context.chain.forward(q))
            point = tcp[:3, 3]+tcp[:3, :3]@context.pad_local
            checks.append({'sphere_maximum_error_m': float(np.abs(actual-expected).max()),
                           'contact_point_error_m': float(np.linalg.norm(point-actual_point))})
        self.setup.update(frame_checks=checks, sphere_count=provenance['sphere_count'],
                          total_setup_seconds=time.perf_counter()-before)
        write_json(self.folder/'curobo-v2-scene.json', self.setup)
        if not checks or any(max(row.values()) > 2e-5 for row in checks):
            raise ValueError('cuRobo v2 contact-frame/robot-sphere calibration failed')

    def close(self):
        if not self.closed:
            result = self.ik.client.request('end_arc', arc_id=self.arc_id)
            self.closed = True
            return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def candidates(self, seed, angle, *, maximum=8, deadline=None, gpu_result=None,
                   excluded_seeds=(), previous_goals=()):
        if self.closed:
            raise ValueError('Arc planning scope is closed')
        before = time.perf_counter()
        rotation = Rotation.from_rotvec(float(angle)*self.context.axis).as_matrix() @ self.model.reference_rotation
        request = {'point': self.model.point(angle).tolist(),
            'reference_quaternion_xyzw': Rotation.from_matrix(rotation).as_quat().tolist(),
            'seed': np.asarray(seed).tolist(), 'angle': float(angle),
            'mode': self.model.grasp_constraint, 'arc_id': self.arc_id}
        reused=gpu_result is not None
        gpu = gpu_result if reused else self.ik.client.request('solve_arc', **request)
        query = {'angle_rad': float(angle), 'gpu_seconds': 0. if reused else gpu['elapsed_seconds'],
                 'solver_seconds': 0. if reused else gpu['solver_seconds'], 'returned_seeds': len(gpu['solutions']),
                 'reused_gpu_batch':reused,'previously_examined_seeds':len(excluded_seeds),
                 'gpu_accepted_seeds': sum(gpu['success']), 'checks': [], 'gpu_result': gpu,
                 'input': request}
        # Failed GPU flags may reflect proxy or convergence error. They are
        # proposals only; every candidate still requires independent admission.
        order = sorted((i for i in range(len(gpu['solutions'])) if i not in excluded_seeds), key=lambda i:
            (not gpu['success'][i], np.linalg.norm(np.asarray(gpu['solutions'][i])-seed)))
        valid, records = [], []
        for index in order:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            initial = np.asarray(gpu['solutions'][index], float)
            if initial.shape != (7,) or not np.isfinite(initial).all():
                records.append({'seed': index, 'accepted': False, 'reason': 'invalid_joint_proposal',
                                'equality_polish_seconds': 0.})
                continue
            position_error = float(np.linalg.norm(self.model.residual(initial, angle)[:3]))
            # An optimizer's failed flag or initial residual is not an endpoint
            # validity test. Refine every finite proposal that the budget permits.
            t0 = time.perf_counter()
            q, projection = self.model.project(initial, float(angle), evaluations=50)
            row = {'seed': index, 'source': 'curobo_v2', 'gpu_accepted': bool(gpu['success'][index]),
                   'gpu_input_position_error_m': position_error,
                   'projection': projection, 'equality_polish_seconds': time.perf_counter()-t0,
                   'accepted': False}
            records.append(row)
            if not projection['accepted']:
                row['reason'] = 'endpoint_constraint_projection_failed'
                continue
            q, native = self.context.project(q)
            row['native_projection'] = native
            if not native['accepted']:
                row['reason'] = 'native_arc_constraint_failed'
                continue
            check = self.context.validate(q)
            excursion = wrist_excursion(self.context, q)
            error = float(self.context.progress(q)-angle)
            row.update(scene=check, joints=q.tolist(), progress_error_rad=error,
                       wrist_excursion_deg=float(np.degrees(excursion)),
                       wrist_limit_deg=float(np.degrees(short_wrist_limit(self.context,self.goal))),
                       progress_tolerance_rad=.002,
                       position_tolerance_m=self.model.position_tolerance_m)
            row['accepted'] = bool(check['accepted'] and abs(error) < .002 and
                excursion <= short_wrist_limit(self.context, self.goal))
            row['rejection_reasons'] = (["joint_bounds" if check.get('joint_bound_violations') else "scene_collision"] if not check['accepted'] else []) + (
                ["goal_progress_error"] if abs(error) >= .002 else []) + (
                ["wrist_excursion_limit"] if excursion > short_wrist_limit(self.context, self.goal) else [])
            if row['accepted'] and not any(np.max(np.abs(q-other)) < .12 for other in [*previous_goals,*valid]):
                valid.append(q)
            if len(valid) >= maximum:
                break
        valid.sort(key=lambda q: np.linalg.norm(q-seed))
        query.update(checks=records, admitted=len(valid), total_seconds=time.perf_counter()-before,
            unexamined_seeds=len(order)-len(records),
            stopped_because=('admitted_candidate_budget' if len(valid)>=maximum else
                'deadline' if len(records)<len(order) else 'all_seeds_examined'),
            rejected_by={reason:sum(reason in row.get('rejection_reasons',[row.get('reason')])
                for row in records) for reason in sorted({reason for row in records for reason in
                row.get('rejection_reasons',[row.get('reason')]) if reason})})
        self.queries.append(query)
        write_json(self.folder/'curobo-v2-queries.json', self.queries)
        return valid, records

    def plan(self, goal, *, seconds):
        """Plan the whole fixed arc through local IK; never return a partial path."""
        if not np.isclose(goal, self.goal, atol=1e-12, rtol=0):
            raise ValueError('Continuation goal differs from its configured arc')
        before = time.perf_counter()
        deadline = before+float(seconds)
        context = self.context
        previous = context.reference_q.copy()
        path = np.empty((0, 7))
        report = {'goal_backend': 'curobo_v2', 'goal_angle_rad': float(goal),
            'planner': 'cuRobo local IK continuation with native MC-VAMP connections',
            'solver_mode': self.solver_mode, 'arc_id': self.arc_id,
            'grasp_constraint': context.grasp_constraint,
            'complete_arc_required': True, 'execution_during_search': False,
            'waypoints_are_planning_internal_only': True, 'arc_anchor_reset_between_steps': False,
            'step_degrees': self.step_degrees, 'planning_sample_duration_seconds': float(self.local_dt),
            'desired_arc_speed_degrees_per_second': self.speed_degrees_per_second,
            'global_queries': [], 'continuation_layers': [], 'goal_proposals': [],
            'goal_generation_seconds': 0., 'native_connection_seconds': 0.,
            'planning_wrist_limit_deg': float(np.degrees(short_wrist_limit(context, goal)))}
        if hasattr(context, 'record'):
            context.record.update(planning_wrist_limit_deg=report['planning_wrist_limit_deg'],
                independent_wrist_excursion_limit_used=bool(short_wrist_limit(context, goal)<np.pi-1e-8))

        def finish(status, phase=None, summary=None):
            report.update(status=status, exact_solution=status == 'solved',
                total_seconds=time.perf_counter()-before,
                terminal_progress_rad=float(context.progress(previous)))
            if status != 'solved':
                report['failure'] = {'phase': phase, 'summary': summary,
                    'reached_angle_deg': float(np.degrees(context.progress(previous))),
                    'attempted_angle_deg': report.get('attempted_angle_deg'),
                    'motion_executed': False}
            write_json(self.folder/'continuation-progress.json', report)
            return (path if status == 'solved' else np.empty((0, 7))), report

        report['start'] = context.validate(previous)
        if not report['start']['accepted']:
            from .contact_collision_evidence import current_collision_witness
            start=report['start']
            start['overlap_evidence']=current_collision_witness(context.scene,context.metadata,start)
            detail=(f"{start['blocking_robot_part']} overlaps {start['blocking_surface']}"
                if start.get('blocking_robot_part') and start.get('blocking_surface') else
                'joint bounds' if start.get('joint_bound_violations') else 'robot self-collision or scene admission')
            return finish('invalid_start', 'start_admission',
                f'The measured start is rejected: {detail}. No goal IK or path search was attempted.')
        for index, angle in enumerate(np.linspace(0., goal, self.steps+1)[1:]):
            report['attempted_angle_deg'] = float(np.degrees(angle))
            if time.perf_counter() >= deadline:
                return finish('search_budget_exhausted', 'continuation', 'The deadline arrived before the next local IK query.')
            checks=[];seen=set();tried_goals=[];batch=None
            layer_begin=time.perf_counter()
            layer = {'index': index, 'angle_rad': float(angle), 'admitted_goals': 0,
                'goal_generation_seconds': 0., 'rejected_candidates': {},
                'connections': [], 'connected': False}
            report['continuation_layers'].append(layer)
            # Connect the first admitted endpoint before refining alternatives.
            # Reuse the same GPU batch after a failed connection, progressively
            # exploring remaining seeds without repeating IK or tried goals.
            while time.perf_counter()<deadline:
                layer_started=time.perf_counter()
                goals,new_checks=self.candidates(previous,float(angle),maximum=1,deadline=deadline,
                    gpu_result=batch,excluded_seeds=seen,previous_goals=tried_goals)
                generation_seconds=time.perf_counter()-layer_started
                report['goal_generation_seconds']+=generation_seconds
                layer['goal_generation_seconds']+=generation_seconds
                checks.extend(new_checks)
                seen.update(c['seed'] for c in new_checks if 'seed' in c)
                layer['admitted_goals']+=len(goals)
                report['goal_proposals'],report['admissible_goals']=checks,layer['admitted_goals']
                layer['rejected_candidates']=dict(Counter(reason for check in checks for reason in
                    check.get('rejection_reasons',[check['reason']] if check.get('reason') else [])))
                if not goals:break
                q=goals[0]
                if any(np.max(np.abs(q-other))<.12 for other in tried_goals):break
                tried_goals.append(q)
                if getattr(self,'queries',None):batch=self.queries[-1]['gpu_result']
                remaining = deadline-time.perf_counter()
                if remaining <= 0:
                    break
                step = float(np.max(np.abs(q-previous)))
                maximum_step=getattr(context,'motion_options',{}).get('maximum_local_joint_step_rad')
                if maximum_step is not None and step > float(maximum_step):
                    layer['connections'].append({'status': 'explicit_joint_step_limit',
                        'maximum_step_rad': step,'allowed_step_rad':float(maximum_step)})
                    continue
                edge, connection = context.plan(previous, q, seconds=min(1., remaining), iterations=1500)
                report['native_connection_seconds'] += connection.get('total_seconds', 0.)
                layer['connections'].append(connection)
                if not connection['exact_solution']:
                    continue
                wrist_ok, wrist = path_wrist_admitted(context, edge, goal)
                collision_ok = all(context.validate(sample)['accepted'] for sample in edge)
                constraint_ok = all(context.constraint_errors(sample)['accepted'] for sample in edge)
                connection.update(wrist, wrist_admitted=bool(wrist_ok),
                    all_samples_admitted=bool(collision_ok and constraint_ok))
                if not (wrist_ok and collision_ok and constraint_ok):
                    connection.update(exact_solution=False, status='complete_edge_admission_failed')
                    continue
                previous = q.copy()
                if len(path):
                    path = np.vstack([path, edge[1:]])
                else:
                    # Native planning may correct a slightly off-manifold
                    # measured start under its existing .01-radian endpoint
                    # gate. Keep that validated first state, not the raw one.
                    path = edge.copy()
                    report['start_projection_max_joint_delta_rad'] = float(
                        np.max(np.abs(path[0]-context.reference_q)))
                layer['connected'] = True
                break
            layer['seconds'] = time.perf_counter()-layer_begin
            report['terminal_progress_rad'] = float(context.progress(previous))
            write_json(self.folder/'continuation-progress.json', report)
            if not layer['connected']:
                if time.perf_counter() >= deadline:
                    return finish('search_budget_exhausted', 'continuation',
                        f'The deadline expired at {np.degrees(angle):.1f} degrees; no partial trajectory is returned.')
                if not layer['admitted_goals']:
                    return finish('continuation_no_admissible_goal', 'goal_admission',
                        f'No local IK candidate passed admission at {np.degrees(angle):.1f} degrees: {layer["rejected_candidates"]}.')
                statuses = dict(Counter(row['status'] for row in layer['connections']))
                collisions = {name: int(sum(row.get(name, 0) for row in layer['connections']))
                    for name in ('moving_rejections', 'static_rejections', 'self_rejections', 'bounds_rejections')}
                return finish('continuation_connection_failed', 'native_connection',
                    f'{layer["admitted_goals"]} admitted goals could not connect at {np.degrees(angle):.1f} degrees: '
                    f'{statuses}; rejected state checks: {collisions}.')
        return finish('solved')

    def global_fallback(self, local_result, *, deadline, started):
        """Try the retained global endpoint branch within the same arc budget."""
        context = self.context
        report = dict(local_result)
        report['local_failure'] = report.pop('failure', None)
        report['global_fallback'] = {'attempted': True, 'strategy': 'retained_global_ik',
            'reason': 'Local continuation could not complete the arc; search an alternative endpoint posture.'}
        self.close()
        before_setup = time.perf_counter()
        setup = self.ik.client.request('configure_arc', **{**self.configure_request, 'solver_mode': 'global'})
        self.arc_id, self.closed = setup['arc_id'], False
        setup['total_setup_seconds'] = time.perf_counter()-before_setup
        report['global_fallback']['setup'] = setup
        write_json(self.folder/'curobo-v2-fallback-scene.json', setup)
        before_goals = time.perf_counter()
        if before_goals < deadline:
            # Leave time to connect the admitted goals rather than consume the
            # whole remaining budget on endpoint alternatives.
            goals, checks = self.candidates(context.reference_q, self.goal,
                deadline=before_goals+.5*(deadline-before_goals))
        else:
            goals, checks = [], []
        report['goal_generation_seconds'] += time.perf_counter()-before_goals
        report['goal_proposals'], report['admissible_goals'] = checks, len(goals)
        report['global_fallback']['admitted_goals'] = len(goals)
        for q in goals:
            remaining = deadline-time.perf_counter()
            if remaining <= 0:
                break
            path, result = context.plan(context.reference_q, q, seconds=min(1.5, remaining), iterations=3000)
            report['native_connection_seconds'] += result.get('total_seconds', 0.)
            report['global_queries'].append(result)
            if not result['exact_solution']:
                continue
            wrist_ok, wrist = path_wrist_admitted(context, path, self.goal)
            collision_ok = all(context.validate(sample)['accepted'] for sample in path)
            constraint_ok = all(context.constraint_errors(sample)['accepted'] for sample in path)
            result.update(wrist, wrist_admitted=bool(wrist_ok),
                all_samples_admitted=bool(collision_ok and constraint_ok))
            if not (wrist_ok and collision_ok and constraint_ok):
                result.update(exact_solution=False, status='complete_path_admission_failed')
                continue
            report.update(status='solved', exact_solution=True,
                planner='cuRobo local continuation with retained global IK fallback and native admission',
                total_seconds=time.perf_counter()-started,
                terminal_progress_rad=float(context.progress(path[-1])),
                start_projection_max_joint_delta_rad=float(np.max(np.abs(path[0]-context.reference_q))))
            report['global_fallback']['solved'] = True
            write_json(self.folder/'continuation-progress.json', report)
            return path, report
        report.update(status='search_budget_exhausted' if time.perf_counter() >= deadline else 'continuation_and_global_failed',
            exact_solution=False, total_seconds=time.perf_counter()-started)
        reasons = Counter(reason for check in checks for reason in
            check.get('rejection_reasons', [check['reason']] if check.get('reason') else []))
        statuses = dict(Counter(row['status'] for row in report['global_queries']))
        report['global_fallback']['solved'] = False
        report['failure'] = {'phase': 'global_fallback', 'motion_executed': False,
            'local_failure': report['local_failure'], 'goal_rejections': dict(reasons),
            'connection_outcomes': statuses,
            'summary': f'Local continuation failed; global fallback admitted {len(goals)} goals. '
                       f'Goal rejections: {dict(reasons)}; connection outcomes: {statuses}.'}
        write_json(self.folder/'continuation-progress.json', report)
        return np.empty((0, 7)), report

    def generate(self, goal, *, deadline=None):
        goals, records = self.candidates(self.context.reference_q, goal, deadline=deadline)
        return self.model, goals, records

    def project(self, seed, angle):
        goals, records = self.candidates(seed, angle, maximum=1)
        return ((goals[0], {'accepted': True, 'backend': 'curobo_v2'}) if goals else
                (np.asarray(seed), {'accepted': False, 'backend': 'curobo_v2', 'tested': len(records)}))
