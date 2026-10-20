"""Full-arm collision admission, local contact policies and moving-source geometry."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.runtime/mc-vamp-deps')]

from try_rekep_bowl_v3 import read_observation_v3, save, snapshot_protocol
from vlm_tamp.rekep_libero.drawer_feedback_v5 import closed_reference, measure_progress, retained_progress
from vlm_tamp.rekep_libero.geometry import interpolate_path, pose_matrix
from vlm_tamp.rekep_libero.object_motion_v3 import observed_object_cloud, verify_observed_transport
from vlm_tamp.rekep_libero.object_tracking_v4 import register_closed_surface
from vlm_tamp.rekep_libero.physical_tasks_v4 import original_residuals
from vlm_tamp.rekep_libero.cache_evidence_v7 import cached_scene, grasp_proposal_agreement
from vlm_tamp.rekep_libero.physical_tasks_v7 import choose_grasp
from vlm_tamp.rekep_libero.sim_client_v7 import SimClientV7
from vlm_tamp.rekep_libero.task_geometry_v4 import bind_bowl_goal, bind_drawer_line, pose_residual
from vlm_tamp.rekep_libero.task_geometry_v5 import compile_scene
from vlm_tamp.rekep_libero.task_ik_v6 import TaskIK
from vlm_tamp.rekep_libero.scene_state_v7 import ObservedScene
from vlm_tamp.rekep_libero.collision_paths_v7 import bowl_transform, check_joint_path, drawer_transform
from vlm_tamp.rekep_libero.geometry import transform_points


class EpisodeFailure(RuntimeError):
    """A recorded physical or observation failure without automatic retries."""


def run_episode(task, state_index, output, config, detector):
    output.mkdir(parents=True)
    source = ROOT / f'results/rekep-libero-v2/libero_goal-{task:02d}-state{state_index:03d}-seed0'
    sim = SimClientV7(log_path=output / 'simulator.log')
    summary = {'version': config['version'], 'task_index': task, 'initial_state': state_index,
               'seed': 0, 'status': 'running', 'failure_stage': None, 'official_task_success': False,
               'grasp_retention_verified': False, 'new_api_calls': 0, 'full_task_grasp_admission': False,
               'full_arm_environment_admission': True, 'moving_scene_geometry_enabled': True,
               'full_hidden_geometry_validated': False, 'collision_search_planner_used': False,
               'native_rekep_optimizer_used': False, 'ompl_used_for_execution': False,
               'software_scene_collision_checks': 'full-arm observed surfaces, local contact policy, sampled joint edges and moving-payload envelope',
               'unknown_space_validated': False, 'physical_contacts_enabled': True,
               'assisted_grasping': False, 'observed_drawer_feedback': task == 0}
    stage, start = 'grounding', time.monotonic()
    commands, feedback = [], []
    state = None
    try:
        state = sim.reset(task, state_index, output / 'initial', seed=config['seed'])
        state = sim.action(state['metadata']['eef_pose_world_xyzw'], gripper=-1,
                           max_steps=config['open_steps'], output_dir=output / 'opened')
        bundle, program, source_cloud, source_mask = cached_scene(source, output / 'opened', config)
        save(output / 'cache-evidence.json', bundle['cache_evidence'])
        spec, grasp_cloud, target_cloud, target_mask, masks = compile_scene(bundle, program, source_cloud, source_mask, config)
        save(output / 'task-spec.json', spec)
        for name, mask in masks.items():
            np.save(output / f'{name}-mask.npy', mask)
            rgb = bundle['cameras'][bundle['main_camera']]['rgb'].copy()
            rgb[mask] = .5 * rgb[mask] + .5 * np.array([30, 190, 80])
            Image.fromarray(rgb).save(output / f'{name}-mask-overlay.png')
        np.save(output / 'source-cloud.npy', source_cloud)
        np.save(output / 'grasp-cloud.npy', grasp_cloud)
        if target_cloud is not None:
            np.save(output / 'target-cloud.npy', target_cloud)
        shutil.copytree(source / 'program', output / 'cached-program')
        shutil.copyfile(source / 'perception/numbered.png', output / 'numbered.png')
        summary['automatic_ordinal_role_correction'] = spec.get('automatic_ordinal_role_correction', False)
        stage = 'grasp_cache_validation'
        cached = ROOT / f"results/physical-tasks-v6-eval/{'bowl' if task == 8 else 'drawer'}/goal{task:02d}-state{state_index:03d}"
        grasp_evidence = grasp_proposal_agreement(grasp_cloud, np.load(cached / 'grasp-cloud.npy', allow_pickle=False), config)
        if not grasp_evidence['accepted']:
            raise EpisodeFailure('Cached GraspGen proposals fail current selected RGB-D geometry agreement')
        shutil.copytree(cached / 'grasps', output / 'grasps')
        with np.load(output / 'grasps/grasp_candidates.npz', allow_pickle=False) as data:
            generated = {key: data[key].copy() for key in ('poses_world_base', 'scores', 'retained_indices')}
        save(output / 'grasp-cache-evidence.json', {'source': cached / 'grasps', **grasp_evidence,
             'source_cloud_sha256': hashlib.sha256(grasp_cloud.tobytes()).hexdigest(),
             'checkpoint': 'pinned GraspGen Panda, identical to v6', 'new_graspgen_inference_calls': 0})
        stage = 'task_aware_grasp_selection'
        robot_record = sim.robot_chain()
        save(output / 'robot-chain.json', robot_record)
        if config['self_collision_checks']:
            import vamp
            environment = vamp.Environment()
            self_valid = lambda joints: vamp.panda.validate(np.asarray(joints).tolist(), environment, True)
        else:
            self_valid = None
        ik = TaskIK(robot_record, config, self_valid)
        cameras, _ = read_observation_v3(output / 'opened', state['metadata'])
        stage = 'scene_reconstruction'
        scene = ObservedScene(cameras, state['metadata'], spec, grasp_cloud, masks['source'], bundle['main_camera'],
                              masks.get('parent'), target_cloud, config)
        scene.save(output / 'initial-scene.npz')
        save(output / 'observed-clearance.json', scene.record())
        save(output / 'contact-policy.json', scene.policy.record(spec['task_type']))
        initial_clearance = scene.check(state['metadata']['robot_collision_volumes_world'], state['metadata']['eef_pose_world_xyzw'])
        save(output / 'initial-actual-clearance.json', initial_clearance)
        if not initial_clearance['accepted']:
            raise EpisodeFailure('Initial robot posture intersects observed scene geometry after self filtering')
        stage = 'task_aware_grasp_selection'
        selected = choose_grasp(generated, grasp_cloud, spec, state['metadata']['robot_calibration'], ik,
                                state['metadata']['joint_positions'], output, sim, scene,
                                state['metadata']['gripper_qpos'], save)
        save(output / 'chosen-grasp.json', selected)
        summary.update(selected_grasp_index=selected['index'], selected_grasp_variant=selected['variant'],
                       full_task_grasp_admission=True, ranking=selected['ranking'])
        path = np.asarray(selected['approach']['poses_world_xyzw'])
        joint_path = selected['approach']['joint_postures']
        save(output / 'approach-command-poses.json', {'poses_world_xyzw': path, 'joint_postures': joint_path})

        collision_context = {'phase': 'approach', 'payload': False,
                             'transform': lambda pose: scene.observed_transform,
                             'stage': lambda pose: 'grasp' if np.linalg.norm(np.asarray(pose)[:3] - path[-1, :3]) <= .012 else 'approach'}
        collision_checks, scene_updates = [], []

        def actual_clearance(label):
            pose = state['metadata']['eef_pose_world_xyzw']
            result = scene.check(state['metadata']['robot_collision_volumes_world'], pose,
                                 stage=collision_context['stage'](pose),
                                 moving_transform=collision_context['transform'](pose),
                                 check_payload=collision_context['payload'])
            collision_checks.append({'kind': 'actual_action_boundary', 'label': label,
                                     'control_steps': state['control_steps'], 'result': result})
            save(output / 'collision-checks.json', collision_checks)
            if not result['accepted']:
                raise EpisodeFailure('Actual robot/payload state fails full-arm observed-scene admission')
            return result

        def refresh_scene(folder, transform):
            current_cameras, _ = read_observation_v3(output / folder, state['metadata'])
            update = scene.refresh(current_cameras, state['metadata'], transform)
            scene_updates.append({'observation': folder, 'control_steps': state['control_steps'], **update})
            scene.save(output / f'scene-revision-{scene.revision:03d}.npz')
            save(output / 'scene-updates.json', scene_updates)
            return current_cameras

        def move(pose, steps, gripper=0, joints=None, name=None, reserve=0):
            nonlocal state
            remaining = config['maximum_control_steps'] - state['control_steps'] - reserve
            if remaining <= 0:
                raise EpisodeFailure('600-control-step budget exhausted or reserved for release')
            if joints is None:
                solved = ik.solve(pose, state['metadata']['joint_positions'], seed_count=2)
                if solved is None:
                    raise EpisodeFailure('Action pose has no bounded robot-only IK continuation')
                joints = solved['joints']
            checked = check_joint_path(scene, sim, ik, [state['metadata']['joint_positions'], joints],
                gripper_qpos=state['metadata']['gripper_qpos'],
                transform_at_pose=collision_context['transform'], stage_at_pose=collision_context['stage'],
                check_payload=collision_context['payload'], config=config)
            collision_checks.append({'kind': 'command_preflight', 'label': name or stage,
                                     'control_steps': state['control_steps'], 'result': checked})
            save(output / 'collision-checks.json', collision_checks)
            if not checked['accepted']:
                raise EpisodeFailure('Requested motion fails full-arm/contact/moving-payload collision admission')
            state = sim.action(pose, gripper=gripper, max_steps=min(steps, remaining),
                               nullspace_joint_target=joints, nullspace_kp=config['nullspace_kp'],
                               output_dir=output / name if name else None)
            commands.append({'stage': stage, 'requested_pose_world_xyzw': np.asarray(pose).tolist(),
                             'requested_joint_posture': joints, 'actual_pose_world_xyzw': state['metadata']['eef_pose_world_xyzw'],
                             'actual_joints': state['metadata']['joint_positions'], 'control_steps': state['control_steps'],
                             'pose_reached': state['reached_pose'], 'executed_steps': state['executed_steps']})
            save(output / 'progress.json', {'stage': stage, 'control_steps': state['control_steps']})
            # Closure and prismatic pulls get an observed moving-part update
            # before the actual-state check. Neither uses commanded hand travel
            # as evidence that the manipulated object actually moved.
            if not (collision_context['phase'] == 'pull' or (collision_context['phase'] == 'grasp' and gripper == 1)):
                actual_clearance(name or stage)
            return state

        def checked_move(pose, steps, gripper=0, name=None, reserve=0):
            solved = ik.solve(pose, state['metadata']['joint_positions'], seed_count=2)
            if solved is None:
                raise EpisodeFailure('Observed execution state has no admitted IK continuation')
            return move(solved['pose_world_xyzw'], steps, gripper, solved['joints'], name, reserve)

        stage = 'approach_execution'
        for index, waypoint in enumerate(path[1:], 1):
            move(waypoint, config['approach_waypoint_steps'], joints=joint_path[index])
        target = path[-1]
        collision_context.update(phase='grasp', stage=lambda pose: 'grasp')
        move(target, config['grasp_settle_steps'], gripper=-1, joints=joint_path[-1], name='at-grasp')
        summary['at_grasp_position_error_m'] = float(np.linalg.norm(np.asarray(state['metadata']['eef_pose_world_xyzw'])[:3] - target[:3]))
        stage = 'grasp_closure'
        move(target, config['closure_steps'], gripper=1, joints=joint_path[-1], name='closed')
        closed_pose = np.asarray(state['metadata']['eef_pose_world_xyzw'])
        summary['closed_gripper_qpos'] = state['metadata']['gripper_qpos']
        save(output / 'original-stage1-costs.json', {'diagnostic_only': True,
             'cached_role_overridden': spec.get('automatic_ordinal_role_correction', False),
             'costs': original_residuals(program, 0, closed_pose, bundle['points_world'], False)})
        stage = 'closure_tracking'
        closed_cameras, _ = read_observation_v3(output / 'closed', state['metadata'])
        closed_cloud = observed_object_cloud(closed_cameras, grasp_cloud, np.eye(4), state['metadata']['robot_collision_volumes_world'])
        closure = register_closed_surface(grasp_cloud, closed_cloud)
        save(output / 'closure-surface-registration.json', closure)
        if not closure['verified']:
            raise EpisodeFailure('Observed surface cannot be tracked through physical closure')
        initial_closed = np.asarray(closure['transform_initial_to_closed_world'])
        if task == 8:
            refresh_scene('closed', initial_closed)
        else:
            centre = grasp_cloud.mean(0)
            closed_centre = transform_points([centre], initial_closed)[0]
            closure_displacement = float((closed_centre - centre) @ spec['pull_axis_world'])
            refresh_scene('closed', scene.drawer_transform(closure_displacement))
            summary['observed_closure_drawer_displacement_m'] = closure_displacement
        collision_context['transform'] = lambda pose: scene.observed_transform
        actual_clearance('closure_after_observed_registration')

        if task == 8:
            stage = 'grasp_retention'
            collision_context.update(phase='carry', payload=True, stage=lambda pose: 'carry',
                                     transform=bowl_transform(closed_pose, initial_closed))
            support_closed = (initial_closed @ np.r_[spec['object_support_point_initial_world_m'], 1.])[:3]
            local_lift = (np.linalg.inv(pose_matrix(closed_pose)) @ np.r_[support_closed, 1.])[:3]
            target_support = support_closed + [0., 0., config['lift_distance_m']]
            count = int(np.ceil(config['lift_distance_m'] / .015))
            for weight in np.linspace(0., 1., count + 1)[1:]:
                lifted = ik.solve_point(support_closed + weight * (target_support - support_closed),
                                        local_lift, state['metadata']['joint_positions'])
                if lifted is None:
                    raise EpisodeFailure('Observed bowl attachment has no admitted lift continuation')
                move(lifted['pose_world_xyzw'], config['transport_waypoint_steps'], joints=lifted['joints'])
            move(lifted['pose_world_xyzw'], config['hold_steps'], gripper=1, joints=lifted['joints'], name='lifted-held')
            cameras, _ = read_observation_v3(output / 'lifted-held', state['metadata'])
            motion = pose_matrix(state['metadata']['eef_pose_world_xyzw']) @ np.linalg.inv(pose_matrix(closed_pose)) @ initial_closed
            observed = observed_object_cloud(cameras, grasp_cloud, motion, state['metadata']['robot_collision_volumes_world'])
            retention = verify_observed_transport(grasp_cloud, observed, motion, state['metadata']['gripper_qpos'])
            save(output / 'retention.json', retention)
            np.save(output / 'retention-observed-cloud.npy', observed)
            summary['grasp_retention_verified'] = retention['verified']
            if not retention['verified']:
                raise EpisodeFailure('RGB-D does not establish retained bowl transport')
            refresh_scene('lifted-held', retention['transform_world'])
            collision_context['transform'] = bowl_transform(state['metadata']['eef_pose_world_xyzw'], retention['transform_world'])
            actual_clearance('retention_after_observed_binding')
            goal, release_pose = bind_bowl_goal(spec, retention['transform_world'], state['metadata']['eef_pose_world_xyzw'])
            save(output / 'bound-goal.json', goal)
            held = np.asarray(state['metadata']['eef_pose_world_xyzw'])
            over = release_pose.copy()
            over[2] = max(held[2], release_pose[2] + config['lift_distance_m'])
            manipulation = interpolate_path([held, over, release_pose], position_step=config['transport_sampling_m'])
            stage = 'observed_attachment_task_preflight'
            point_local = np.asarray(goal['point_local_m'])
            targets = np.asarray([pose_matrix(pose)[:3, 3] + pose_matrix(pose)[:3, :3] @ point_local for pose in manipulation])
            checked = ik.point_path(targets, point_local, state['metadata']['joint_positions'])
            save(output / 'observed-attachment-preflight.json', checked)
            if not checked['accepted']:
                raise EpisodeFailure('Measured bowl attachment does not admit the complete placement')
            collision_context.update(phase='place', stage=lambda pose: 'place' if np.linalg.norm(
                transform_points([point_local], pose_matrix(pose))[0] - goal['target_world_m']) < .025 else 'carry')
            collision_preflight = check_joint_path(scene, sim, ik, [item['joints'] for item in checked['solutions']],
                gripper_qpos=state['metadata']['gripper_qpos'], transform_at_pose=collision_context['transform'],
                stage_at_pose=collision_context['stage'], check_payload=True, config=config)
            save(output / 'observed-attachment-collision-preflight.json', collision_preflight)
            if not collision_preflight['accepted']:
                raise EpisodeFailure('Measured bowl attachment path fails full-arm or payload clearance')
            save(output / 'manipulation-command-poses.json', {'poses_world_xyzw': manipulation})
            stage = 'placement_execution'
            reserve = config['release_steps'] + 10
            for solution in checked['solutions'][1:]:
                move(solution['pose_world_xyzw'], config['transport_waypoint_steps'], joints=solution['joints'], reserve=reserve)
            corrections = []
            for _ in range(4):
                residual = float(np.linalg.norm(pose_residual(goal, state['metadata']['eef_pose_world_xyzw'])['equality']))
                if residual <= .006 or config['maximum_control_steps'] - state['control_steps'] <= reserve + 12:
                    break
                correction = ik.solve_point(goal['target_world_m'], point_local, state['metadata']['joint_positions'])
                if correction is None:
                    break
                clearance = scene.joints(sim, correction['joints'], gripper_qpos=state['metadata']['gripper_qpos'],
                    stage='place', moving_transform=collision_context['transform'](correction['pose_world_xyzw']), check_payload=True)
                if not clearance['accepted']:
                    break
                move(correction['pose_world_xyzw'], 12, joints=correction['joints'], reserve=reserve)
                corrections.append({'error_before_m': residual, 'control_steps': state['control_steps']})
            save(output / 'placement-feedback.json', corrections)
            summary['before_release_goal_error_m'] = float(np.linalg.norm(pose_residual(goal, state['metadata']['eef_pose_world_xyzw'])['equality']))
            sim.observe(output / 'before-release')
            stage = 'release_execution'
            released_transform = collision_context['transform'](state['metadata']['eef_pose_world_xyzw']).copy()
            move(state['metadata']['eef_pose_world_xyzw'], config['release_steps'], gripper=-1, name='released')
            release_cameras, _ = read_observation_v3(output / 'released', state['metadata'])
            release_observed = observed_object_cloud(release_cameras, grasp_cloud, released_transform,
                                                     state['metadata']['robot_collision_volumes_world'])
            release_registration = register_closed_surface(transform_points(grasp_cloud, released_transform), release_observed)
            save(output / 'release-surface-registration.json', release_registration)
            if not release_registration['verified']:
                raise EpisodeFailure('Released object cannot be localized before retreat')
            released_transform = np.asarray(release_registration['transform_initial_to_closed_world']) @ released_transform
            refresh_scene('released', released_transform)
            collision_context.update(phase='released', payload=False, stage=lambda pose: 'released',
                                     transform=lambda pose: scene.observed_transform)
            actual_clearance('release_after_observed_registration')
            retreat = np.asarray(state['metadata']['eef_pose_world_xyzw']).copy()
            retreat[2] += .05
            checked_move(retreat, 10, name='final')
        else:
            axis = np.asarray(spec['pull_axis_world'])
            reference = closed_reference(grasp_cloud, initial_closed)
            line = bind_drawer_line(spec, closed_pose)
            save(output / 'bound-path-constraint.json', line)
            collision_context.update(phase='pull', payload=True, stage=lambda pose: 'pull',
                                     transform=drawer_transform(scene, closed_pose, closure_displacement))
            stage = 'drawer_probe'
            probe = closed_pose.copy()
            probe[:3] += config['drawer_probe_m'] * axis
            checked_move(probe, 24, gripper=1, name='probe-held')
            cameras, _ = read_observation_v3(output / 'probe-held', state['metadata'])
            hand_s = float((np.asarray(state['metadata']['eef_pose_world_xyzw'])[:3] - closed_pose[:3]) @ axis)
            measured = measure_progress(reference, cameras, state['metadata']['robot_collision_volumes_world'], axis, hand_s)
            measured.update(control_steps=state['control_steps'], phase='probe', retained=retained_progress(measured, state['metadata']['gripper_qpos']))
            feedback.append(measured)
            save(output / 'retention.json', {'verified': measured['retained'], 'source': measured['source'], 'measurement': measured})
            summary['grasp_retention_verified'] = measured['retained']
            if not measured['retained'] or measured['observed_progress_m'] < .012:
                raise EpisodeFailure('Observed handle does not follow the probe pull')
            refresh_scene('probe-held', scene.drawer_transform(closure_displacement + measured['observed_progress_m']))
            # Bind subsequent predictions to the observed displacement at the
            # current hand state; feedback drift is not accumulated as geometry.
            collision_context['transform'] = drawer_transform(scene, state['metadata']['eef_pose_world_xyzw'],
                closure_displacement + measured['observed_progress_m'])
            actual_clearance('probe_after_observed_drawer_update')
            stage = 'drawer_pull_execution'
            previous_s = measured['observed_progress_m']
            stalled = 0
            for iteration in range(config['drawer_maximum_feedback_iterations']):
                remaining = config['drawer_travel_m'] - previous_s
                if remaining <= config['drawer_progress_tolerance_m']:
                    break
                requested_step = min(config['drawer_chunk_m'], remaining + .003)
                requested_s = min(config['drawer_maximum_command_travel_m'], hand_s + requested_step)
                request = closed_pose.copy()
                request[:3] += requested_s * axis
                folder = f'pull-observation-{iteration:02d}'
                checked_move(request, config['drawer_feedback_control_steps'], gripper=1, name=folder)
                cameras, _ = read_observation_v3(output / folder, state['metadata'])
                hand_s = float((np.asarray(state['metadata']['eef_pose_world_xyzw'])[:3] - closed_pose[:3]) @ axis)
                measured = measure_progress(reference, cameras, state['metadata']['robot_collision_volumes_world'], axis,
                                             hand_s, previous_s, maximum_progress=.18)
                measured.update(control_steps=state['control_steps'], phase='pull', requested_hand_progress_m=requested_s,
                                retained=retained_progress(measured, state['metadata']['gripper_qpos']),
                                line_residual=pose_residual(line, state['metadata']['eef_pose_world_xyzw']))
                feedback.append(measured)
                save(output / 'drawer-feedback.json', feedback)
                if not measured['retained']:
                    raise EpisodeFailure('Drawer feedback detected tracking loss or hand/handle separation')
                refresh_scene(folder, scene.drawer_transform(closure_displacement + measured['observed_progress_m']))
                collision_context['transform'] = drawer_transform(scene, state['metadata']['eef_pose_world_xyzw'],
                    closure_displacement + measured['observed_progress_m'])
                actual_clearance('pull_after_observed_drawer_update')
                stalled = stalled + 1 if measured['observed_increment_m'] < config['drawer_feedback_minimum_progress_m'] else 0
                if stalled >= 2:
                    raise EpisodeFailure('Drawer feedback detected two pulls without sufficient observed progress')
                previous_s = measured['observed_progress_m']
            summary.update(observed_drawer_progress_m=previous_s, actual_hand_progress_m=hand_s,
                           drawer_observed_goal_reached=previous_s >= config['drawer_travel_m'] - config['drawer_progress_tolerance_m'])
            sim.observe(output / 'final')
            if not summary['drawer_observed_goal_reached']:
                raise EpisodeFailure('Observed drawer travel remains below the configured goal')
        summary['status'] = 'completed_commands'
    except Exception as error:
        # Episode boundary: record unexpected implementation errors and keep all
        # remaining planned episodes in the experiment, with full traceback.
        summary.update(status='failed', failure_stage=stage, failure=str(error), exception_type=type(error).__name__)
        (output / 'error-traceback.txt').write_text(traceback.format_exc())
    finally:
        closed = sim.close()
        summary.update(control_steps=closed.get('control_steps', 0), video=closed.get('video_path'),
                       elapsed_seconds=time.monotonic() - start)
        if (output / 'initial/actions.json').is_file():
            actions = json.loads((output / 'initial/actions.json').read_text())
            summary['official_task_success'] = any(action['evaluation_task_success'] for action in actions)
            summary['final_task_success'] = bool(actions and actions[-1]['evaluation_task_success'])
            summary['closing_control_steps'] = sum(action['control'][-1] > 0 for action in actions)
        save(output / 'command-log.json', commands)
        save(output / 'drawer-feedback.json', feedback)
        if 'scene' in locals():
            save(output / 'final-scene-scope.json', scene.record())
        if 'collision_checks' in locals():
            summary.update(collision_check_records=len(collision_checks), scene_updates=len(scene_updates))
        save(output / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tasks', type=int, nargs='+', default=[8, 0])
    parser.add_argument('--states', type=int, nargs='+', default=[1, 11, 21, 31, 41])
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve existing results; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'configs/physical_tasks_v7.json').read_text())
    save(output / 'configuration.json', config)
    snapshot_protocol(output, args)
    protocol = json.loads((output / 'protocol.json').read_text())
    sources = [Path(__file__), ROOT / 'configs/physical_tasks_v7.json', ROOT / 'src/vlm_tamp/drawer_local_plane.py',
               ROOT / 'src/vlm_tamp/drawer_constraints.py', ROOT / 'src/vlm_tamp/pose_constraints.py',
               ROOT / 'scripts/report_physical_tasks_v7.py', ROOT / 'scripts/replay_moving_drawer_v7.py',
               ROOT / 'scripts/validate_synchronized_geometry_v7.py',
               ROOT / 'tests/test_collision_scene_v7.py', ROOT / 'tests/test_cache_evidence_v7.py',
               ROOT / 'docs/physical_tasks_v7.md']
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = output / 'source-snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        protocol['sources_sha256'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol.update(development_only=args.development, all_official_states_previously_inspected=True,
                    physical_contacts_enabled=True,
                    software_scene_collision_checks='full-arm observed surfaces and moving-payload envelope at sampled states',
                    unknown_space_validated=False, held_object_collision_geometry_validated="observed convex envelope only; hidden body unvalidated",
                    collision_joint_sampling_rad=config['collision_joint_sampling_rad'],
                    observed_clearance_margin_m=config['observed_clearance_margin_m'],
                    robot_self_depth_tolerance_m=config['robot_self_depth_tolerance_m'],
                    new_graspgen_inference_calls=0, all_configured_moving_robot_parts_checked=True,
                    runtime_thread_limits={name: os.environ.get(name) for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']},
                    self_collision_checks=config['self_collision_checks'], native_rekep_optimizer_used=False,
                    benchmark_type=config['benchmark_type'], no_held_out_generalization_claim=True,
                    ordinal_grounding_adaptation='middle of three RGB-mask-supported measured bars',
                    original_results_preserved=True, mc_vamp_revision='24e0f8adef0649a957fec77bf33f8c2f04515a18')
    save(output / 'protocol.json', protocol)
    ledger = ROOT / 'results/rekep-libero-api/ledger.json'
    ledger_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
    planned = [(task, index) for task in args.tasks for index in args.states]
    save(output / 'manifest.json', {'planned_episodes': planned, 'all_in_denominator': True, 'seed': config['seed'],
                                  'maximum_control_steps': config['maximum_control_steps']})
    detector = None  # Reuse input-verified pinned GraspGen requests to isolate collision changes.
    summaries = []
    for task, index in planned:
        summaries.append(run_episode(task, index, output / f'goal{task:02d}-state{index:03d}', config, detector))
        save(output / 'summary.json', {'planned_episodes': len(planned), 'completed_episodes': len(summaries),
                                     'official_task_successes': sum(row['official_task_success'] for row in summaries),
                                     'api_ledger_unchanged': hashlib.sha256(ledger.read_bytes()).hexdigest() == ledger_hash,
                                     'episodes': summaries})


if __name__ == '__main__':
    main()
