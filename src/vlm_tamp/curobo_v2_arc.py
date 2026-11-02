"""Full-angle cuRobo IK endpoints for upstream MC-VAMP constrained planning.

IK and MC-VAMP use shared observed points and robot spheres. Returned planner
paths are not rechecked. Local continuation is a reported
fallback after the full-angle attempt fails.
"""
from pathlib import Path
from collections import Counter
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .coupled_storage import read_json, write_json
from .partial_pose_arc import PartialPoseArc, short_wrist_limit, wrist_excursion
from .rekep_libero.curobo_collision_model import calibrated_spheres, expected_spheres
from .rekep_libero.geometry import pose_matrix


class CuroboV2ArcProposer:
    def __init__(self, ik, context, folder, goal):
        if getattr(ik.client, 'backend', None) != 'v2':
            raise ValueError('cuRobo v2 arc proposals require the v2 worker')
        self.ik, self.context, self.folder = ik, context, Path(folder)
        self.goal = float(goal)
        options = getattr(context, 'motion_options', context.scene.config)
        self.solver_mode = 'global'
        self.continuation_fallback_enabled = bool(options.get('curobo_v2_continuation_fallback', True))
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
        # IK accuracy and the path's geometric tolerance have different roles.
        # Ask the GPU for the desired accuracy; never re-solve its output on CPU.
        self.ik_position_tolerance_m = float(options.get('curobo_v2_arc_ik_position_tolerance_m', .001))
        self.queries = []
        before = time.perf_counter()
        if not hasattr(ik, '_v2_arc_spheres'):
            ik._v2_arc_spheres = calibrated_spheres(context.chain, context.scene.robot, context.world_robot, include_self=True)
        spheres, provenance = ik._v2_arc_spheres
        ik.configure_scene(context)
        reference = read_json(self.folder/'scene-reference.json')
        self.configure_request = dict(collision_spheres=spheres,
            position_tolerance=self.ik_position_tolerance_m,
            orientation_tolerance=self.model.orientation_tolerance_rad,
            pad_local=context.pad_local.tolist(), scene_arrays_path=reference['arrays_path'],
            scene_arrays_sha256=reference['arrays_sha256'], origin=context.origin.tolist(),
            axis=context.axis.tolist(), mesh_pitch=float(context.scene.config.get('curobo_v2_mesh_pitch_m', .005)),
            solver_mode=self.solver_mode, local_dt=float(self.local_dt),
            moving_mesh_pitch=float(options.get('curobo_v2_moving_mesh_pitch_m', .002)),
            local_seed_std=float(options.get('curobo_v2_local_seed_std_rad', .02)))
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

    def candidates(self, seed, angle, *, maximum=None, deadline=None, gpu_result=None,
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
        # Preserve the GPU result, including failure. There is no CPU rescue,
        # joint clipping or endpoint projection in this handoff.
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
            q = initial.copy()
            errors = gpu.get('position_error_m')
            position_error = float(errors[index]) if errors is not None else None
            row = {'seed': index, 'source': 'curobo_v2', 'gpu_accepted': bool(gpu['success'][index]),
                   'gpu_input_position_error_m': position_error,
                   'joints': q.tolist(), 'candidate_modified': False,
                   'equality_polish_seconds': 0.,
                   'ik_position_tolerance_m': self.ik_position_tolerance_m,
                   'accepted': False}
            records.append(row)
            if not row['gpu_accepted']:
                row['reason'] = 'curobo_ik_failed'
                continue
            excursion = wrist_excursion(self.context, q)
            error = float(self.context.progress(q)-angle)
            row.update(scene_admission='owned_by_path_planner', joints=q.tolist(), progress_error_rad=error,
                       wrist_excursion_deg=float(np.degrees(excursion)),
                       wrist_limit_deg=float(np.degrees(short_wrist_limit(self.context,self.goal))),
                       position_tolerance_m=self.model.position_tolerance_m)
            row['accepted'] = bool(excursion <= short_wrist_limit(self.context, self.goal))
            row['rejection_reasons'] = ["wrist_excursion_limit"] if not row['accepted'] else []
            if row['accepted'] and not any(np.array_equal(q, other) for other in [*previous_goals,*valid]):
                valid.append(q)
            if maximum is not None and len(valid) >= maximum:
                break
        valid.sort(key=lambda q: np.linalg.norm(q-seed))
        query.update(checks=records, admitted=len(valid), total_seconds=time.perf_counter()-before,
            unexamined_seeds=len(order)-len(records),
            stopped_because=('admitted_candidate_budget' if maximum is not None and len(valid)>=maximum else
                'deadline' if len(records)<len(order) else 'all_seeds_examined'),
            rejected_by={reason:sum(reason in row.get('rejection_reasons',[row.get('reason')])
                for row in records) for reason in sorted({reason for row in records for reason in
                row.get('rejection_reasons',[row.get('reason')]) if reason})})
        self.queries.append(query)
        write_json(self.folder/'curobo-v2-queries.json', self.queries)
        return valid, records

    def continuation_plan(self, goal, *, seconds):
        """Fallback only: build a complete arc through local continuation."""
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
                if any(np.array_equal(q, other) for other in tried_goals):break
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
                connection.update(path_validation='upstream planner checks only',additional_path_checks=False)
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

    def plan(self, goal, *, seconds):
        """One full-angle goal query and native constrained plan before any fallback."""
        if not np.isclose(goal,self.goal,atol=1e-12,rtol=0):
            raise ValueError('Requested arc differs from configured arc')
        began=time.perf_counter();deadline=began+float(seconds);context=self.context
        report={'goal_backend':'curobo_v2','goal_angle_rad':float(goal),
            'planner':'MC-VAMP CRRTC full arc first','solver_mode':'global',
            'grasp_constraint':context.grasp_constraint,'complete_arc_required':True,
            'execution_during_search':False,'search_order':['full_arc','continuation_fallback'],
            'global_queries':[],'continuation_layers':[],'goal_proposals':[],
            'full_arc_ik_queries':0,'goal_admission_scope':'IK targets; collision admission belongs to MC-VAMP',
            'goal_generation_seconds':0.,'native_connection_seconds':0.,
            'continuation_fallback':{'attempted':False},'additional_path_checks':False}
        goals=[];phase=time.perf_counter()
        if phase<deadline:
            report['full_arc_ik_queries']+=1
            goals,checks=self.candidates(context.reference_q,float(goal),deadline=deadline)
            report['goal_proposals']=checks
        report['goal_generation_seconds']=time.perf_counter()-phase
        report['admissible_goals']=len(goals)
        for target in goals:
            remaining=deadline-time.perf_counter()
            if remaining<=0:break
            path,connection=context.plan(context.reference_q,target,seconds=min(5.,remaining),iterations=5000)
            report['native_connection_seconds']+=connection.get('total_seconds',0.)
            report['global_queries'].append(connection)
            if connection['exact_solution']:
                report.update(status='solved',exact_solution=True,total_seconds=time.perf_counter()-began,
                    solved_by='full_arc',terminal_progress_rad=float(context.progress(path[-1])))
                write_json(self.folder/'continuation-progress.json',report)
                return path,report
        primary_status=('search_budget_exhausted' if time.perf_counter()>=deadline else
            'full_arc_no_ik_goal' if not goals else 'full_arc_connection_failed')
        report['full_arc_attempt']={'status':primary_status,'goal_queries':report['full_arc_ik_queries'],
            'goal_generation_seconds':report['goal_generation_seconds'],
            'native_connection_seconds':report['native_connection_seconds'],
            'admitted_goals':len(goals),'connections':report['global_queries']}
        remaining=deadline-time.perf_counter()
        if self.continuation_fallback_enabled and remaining>0:
            self.close();phase=time.perf_counter()
            setup=self.ik.client.request('configure_arc',**{**self.configure_request,'solver_mode':'local_continuation'})
            self.arc_id,self.closed,self.solver_mode=setup['arc_id'],False,'local_continuation'
            setup['total_setup_seconds']=time.perf_counter()-phase
            path,fallback=self.continuation_plan(goal,seconds=max(0.,deadline-time.perf_counter()))
            report['continuation_fallback']={'attempted':True,'reason':primary_status,
                'setup':setup,'status':fallback['status'],'solved':fallback['exact_solution']}
            report['continuation_layers']=fallback['continuation_layers']
            report['goal_generation_seconds']+=fallback['goal_generation_seconds']
            report['native_connection_seconds']+=fallback['native_connection_seconds']
            report.update(status=fallback['status'],exact_solution=fallback['exact_solution'],
                terminal_progress_rad=fallback['terminal_progress_rad'],
                continuation_result=fallback,total_seconds=time.perf_counter()-began,
                solved_by='continuation_fallback' if fallback['exact_solution'] else None)
            if not fallback['exact_solution']:report['failure']=fallback.get('failure')
            write_json(self.folder/'continuation-progress.json',report)
            return path,report
        report.update(status=primary_status,exact_solution=False,total_seconds=time.perf_counter()-began,
            failure={'phase':'full_arc','motion_executed':False,
                'summary':f'Full-arc search ended with {primary_status}; {len(goals)} goal postures, '
                    f'{len(report["global_queries"])} native connections. No partial motion was executed.'})
        write_json(self.folder/'continuation-progress.json',report)
        return np.empty((0,7)),report

    def generate(self, goal, *, deadline=None):
        goals, records = self.candidates(self.context.reference_q, goal, deadline=deadline)
        return self.model, goals, records

    def project(self, seed, angle):
        goals, records = self.candidates(seed, angle, maximum=1)
        return ((goals[0], {'accepted': True, 'backend': 'curobo_v2'}) if goals else
                (np.asarray(seed), {'accepted': False, 'backend': 'curobo_v2', 'tested': len(records)}))
