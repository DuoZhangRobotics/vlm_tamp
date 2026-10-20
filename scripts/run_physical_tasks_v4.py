"""Complete RGB-D bowl placement and drawer pull with ordinary LIBERO contacts.

This is a versioned constraint-guided IK/OSC baseline, not the frozen strict
ReKep benchmark. Every requested episode remains in the reported denominator.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from try_rekep_bowl_v3 import read_observation_v3, save, snapshot_protocol
from vlm_tamp.rekep_libero.geometry import interpolate_path, pose_matrix
from vlm_tamp.rekep_libero.object_motion_v3 import observed_object_cloud, verify_observed_transport
from vlm_tamp.rekep_libero.object_tracking_v4 import register_closed_surface
from vlm_tamp.rekep_libero.physical_tasks_v4 import (
    cached_scene, choose_grasp, compile_scene, original_residuals,
)
from vlm_tamp.rekep_libero.sim_client_v3 import SimClientV3
from vlm_tamp.rekep_libero.task_geometry_v4 import bind_bowl_goal, bind_drawer_line, pose_residual


class EpisodeFailure(RuntimeError):
    """A recorded pipeline failure, including failures before any movement."""


def run_episode(task, state_index, output, config, detector, development_plan=None):
    output.mkdir(parents=True)
    source = ROOT / f'results/rekep-libero-v2/libero_goal-{task:02d}-state{state_index:03d}-seed0'
    sim = SimClientV3(log_path=output / 'simulator.log')
    summary = {'version': config['version'], 'task_index': task, 'initial_state': state_index,
               'seed': 0, 'status': 'running', 'failure_stage': None, 'official_task_success': False,
               'grasp_retention_verified': False, 'new_api_calls': 0,
               'native_rekep_optimizer_used': False, 'software_collision_checks': False,
               'physical_contacts_enabled': True, 'assisted_grasping': False,
               'development_replay': development_plan is not None}
    stage, start = 'grounding', time.monotonic()
    commands = []
    try:
        state = sim.reset(task, state_index, output / 'initial', seed=config['seed'])
        state = sim.action(state['metadata']['eef_pose_world_xyzw'], gripper=-1,
                           max_steps=config['open_steps'], output_dir=output / 'opened')
        bundle, program, source_cloud, source_mask = cached_scene(source, output / 'opened')
        spec, grasp_cloud, target_cloud, target_mask = compile_scene(bundle, program, source_cloud, config)
        save(output / 'cache-evidence.json', bundle['cache_evidence'])
        save(output / 'task-spec.json', spec)
        np.save(output / 'source-mask.npy', source_mask)
        np.save(output / 'source-cloud.npy', source_cloud)
        np.save(output / 'grasp-cloud.npy', grasp_cloud)
        if target_cloud is not None:
            np.save(output / 'target-cloud.npy', target_cloud)
            np.save(output / 'target-mask.npy', target_mask)
        shutil.copytree(source / 'program', output / 'cached-program')
        shutil.copyfile(source / 'perception/numbered.png', output / 'numbered.png')
        stage = 'grasp_generation'
        if task == 0:
            import open3d as o3d
            o3d.utility.random.seed(config['seed'])
            generated = detector.generate(grasp_cloud, output / 'grasps')
        else:
            shutil.copytree(source / 'stage1-grasps', output / 'grasps')
            with np.load(output / 'grasps/grasp_candidates.npz', allow_pickle=False) as data:
                generated = {key: data[key].copy() for key in ('poses_world_base', 'scores', 'retained_indices')}
        stage = 'grasp_selection'
        if development_plan:
            plan = ROOT / development_plan
            selected = json.loads((plan / 'chosen-grasp.json').read_text())
            path = np.asarray(json.loads((plan / 'approach-command-poses.json').read_text())['poses_world_xyzw'])
            checked = json.loads((plan / f'approach-{selected["index"]}-{selected["variant"]}.json').read_text())
            joint_path = checked['solution_joints']
            grasp_cloud = np.load(plan / 'selected_cloud.npy', allow_pickle=False)
            summary['ranking'] = 'previous successful development grasp and paired approach'
        else:
            selected = choose_grasp(generated, bundle['points_world'][spec['source_keypoint']], grasp_cloud,
                                    spec, state['metadata']['robot_calibration'], sim,
                                    np.asarray(state['metadata']['joint_positions']))
            path = np.asarray(selected['approach']['poses_world_xyzw'])
            joint_path = selected['approach']['joint_postures']
            summary['ranking'] = selected['ranking']
        save(output / 'chosen-grasp.json', selected)
        save(output / 'approach-command-poses.json', {'poses_world_xyzw': path})
        summary.update(selected_grasp_index=selected['index'], selected_grasp_variant=selected['variant'])

        def move(pose, steps, gripper=0, joints=None, name=None):
            nonlocal state
            if state.get('step_limit_reached'):
                raise EpisodeFailure('600-control-step budget exhausted')
            state = sim.action(pose, gripper=gripper, max_steps=steps,
                               nullspace_joint_target=joints, nullspace_kp=config['nullspace_kp'],
                               output_dir=output / name if name else None)
            commands.append({'stage': stage, 'requested_pose_world_xyzw': np.asarray(pose).tolist(),
                             'requested_joint_posture': joints, 'actual_pose_world_xyzw': state['metadata']['eef_pose_world_xyzw'],
                             'actual_joints': state['metadata']['joint_positions'],
                             'control_steps': state['control_steps'], 'pose_reached': state['reached_pose'],
                             'executed_steps': state['executed_steps']})
            return state

        stage = 'approach_execution'
        for index, waypoint in enumerate(path[1:], 1):
            if joint_path is not None:
                joints = joint_path[index]
            else:
                solved = sim.multi_ik(waypoint, seed_joints=state['metadata']['joint_positions'],
                                      max_iterations=180, max_starts=3)
                if not solved['solutions']:
                    raise EpisodeFailure(f'Approach IK failed at waypoint {index}')
                solution = min(solved['solutions'], key=lambda item: np.linalg.norm(
                    np.asarray(item['cspace_position']) - state['metadata']['joint_positions']))
                joints = solution['cspace_position']
                waypoint = np.asarray(solution['eef_pose_world_xyzw'])
            move(waypoint, config['approach_waypoint_steps'], joints=joints)
        target = np.asarray(commands[-1]['requested_pose_world_xyzw'])
        move(target, config['grasp_settle_steps'], gripper=-1, joints=joints, name='at-grasp')
        summary['at_grasp_position_error_m'] = float(np.linalg.norm(
            np.asarray(state['metadata']['eef_pose_world_xyzw'])[:3] - target[:3]))
        stage = 'grasp_closure'
        move(target, config['closure_steps'], gripper=1, joints=joints, name='closed')
        closed_pose = np.asarray(state['metadata']['eef_pose_world_xyzw'])
        summary['closed_gripper_qpos'] = state['metadata']['gripper_qpos']
        save(output / 'original-stage1-costs.json', original_residuals(
            program, 0, closed_pose, bundle['points_world'], False))
        stage = 'closure_tracking'
        closed_cameras, _ = read_observation_v3(output / 'closed', state['metadata'])
        closed_cloud = observed_object_cloud(closed_cameras, grasp_cloud, np.eye(4),
                                            state['metadata']['robot_collision_volumes_world'])
        closure = register_closed_surface(grasp_cloud, closed_cloud)
        save(output / 'closure-surface-registration.json', closure)
        if not closure['verified']:
            raise EpisodeFailure('Observed surface cannot be tracked through gripper closure')
        initial_closed = np.asarray(closure['transform_initial_to_closed_world'])

        if task == 8:
            stage = 'grasp_retention'
            probe = closed_pose.copy()
            probe[2] += config['lift_distance_m']
            for waypoint in interpolate_path([closed_pose, probe], position_step=.01)[1:]:
                move(waypoint, config['transport_waypoint_steps'])
            move(probe, config['hold_steps'], gripper=1, name='lifted-held')
            cameras, _ = read_observation_v3(output / 'lifted-held', state['metadata'])
            motion = (pose_matrix(state['metadata']['eef_pose_world_xyzw']) @ np.linalg.inv(pose_matrix(closed_pose))
                      @ initial_closed)
            observed = observed_object_cloud(cameras, grasp_cloud, motion, state['metadata']['robot_collision_volumes_world'])
            retention = verify_observed_transport(grasp_cloud, observed, motion, state['metadata']['gripper_qpos'])
            np.save(output / 'retention-observed-cloud.npy', observed)
            save(output / 'retention.json', retention)
            summary['grasp_retention_verified'] = retention['verified']
            if not retention['verified']:
                raise EpisodeFailure('RGB-D does not establish that the bowl follows the closed gripper')
            goal, release_pose = bind_bowl_goal(spec, retention['transform_world'], state['metadata']['eef_pose_world_xyzw'])
            save(output / 'bound-goal.json', goal)
            stage = 'placement_execution'
            held_pose = np.asarray(state['metadata']['eef_pose_world_xyzw'])
            over_plate = release_pose.copy()
            over_plate[2] = max(held_pose[2], release_pose[2] + config['lift_distance_m'])
            path = interpolate_path([held_pose, over_plate, release_pose], position_step=config['transport_sampling_m'])
            save(output / 'manipulation-command-poses.json', {'poses_world_xyzw': path})
            for waypoint in path[1:]:
                move(waypoint, config['transport_waypoint_steps'])
            move(release_pose, 15, gripper=1, name='before-release')
            summary['before_release_goal_error_m'] = float(np.linalg.norm(pose_residual(
                goal, state['metadata']['eef_pose_world_xyzw'])['equality']))
            stage = 'release_execution'
            move(release_pose, config['release_steps'], gripper=-1, name='released')
            retreat = np.asarray(state['metadata']['eef_pose_world_xyzw']).copy()
            retreat[2] += .05
            move(retreat, 20, name='final')
        else:
            stage = 'drawer_probe'
            line = bind_drawer_line(spec, closed_pose)
            save(output / 'bound-path-constraint.json', line)
            axis = np.asarray(spec['pull_axis_world'])
            probe = closed_pose.copy()
            probe[:3] += config['drawer_probe_m'] * axis
            for waypoint in interpolate_path([closed_pose, probe], position_step=.01)[1:]:
                move(waypoint, config['transport_waypoint_steps'])
            move(probe, config['hold_steps'], gripper=1, name='probe-held')
            cameras, _ = read_observation_v3(output / 'probe-held', state['metadata'])
            motion = (pose_matrix(state['metadata']['eef_pose_world_xyzw']) @ np.linalg.inv(pose_matrix(closed_pose))
                      @ initial_closed)
            observed = observed_object_cloud(cameras, grasp_cloud, motion, state['metadata']['robot_collision_volumes_world'])
            retention = verify_observed_transport(grasp_cloud, observed, motion, state['metadata']['gripper_qpos'],
                                                   minimum_motion=.015)
            save(output / 'retention.json', retention)
            np.save(output / 'retention-observed-cloud.npy', observed)
            summary['grasp_retention_verified'] = retention['verified']
            if not retention['verified']:
                raise EpisodeFailure('RGB-D does not establish that the handle follows the closed gripper')
            stage = 'drawer_pull_execution'
            pull = closed_pose.copy()
            pull[:3] += config['drawer_travel_m'] * axis
            path = interpolate_path([state['metadata']['eef_pose_world_xyzw'], pull], position_step=.01)
            save(output / 'manipulation-command-poses.json', {'poses_world_xyzw': path})
            residuals = []
            for waypoint in path[1:]:
                move(waypoint, config['transport_waypoint_steps'])
                residuals.append({'control_steps': state['control_steps'], **pose_residual(
                    line, state['metadata']['eef_pose_world_xyzw'])})
            move(pull, config['hold_steps'], gripper=1, name='final')
            save(output / 'path-residuals.json', residuals)
            summary['maximum_line_error_m'] = max(float(np.linalg.norm(row['equality'])) for row in residuals)
        summary['status'] = 'completed_commands'
    except (EpisodeFailure, ValueError, RuntimeError, OSError) as error:
        summary.update(status='failed', failure_stage=stage, failure=str(error), exception_type=type(error).__name__)
        (output / 'error-traceback.txt').write_text(traceback.format_exc())
    finally:
        closed = sim.close()
        summary.update(control_steps=closed.get('control_steps', 0), video=closed.get('video_path'),
                       elapsed_seconds=time.monotonic() - start)
        actions_path = output / 'initial/actions.json'
        if actions_path.is_file():
            actions = json.loads(actions_path.read_text())
            summary['official_task_success'] = any(action['evaluation_task_success'] for action in actions)
            summary['final_task_success'] = bool(actions and actions[-1]['evaluation_task_success'])
            summary['closing_control_steps'] = sum(action['control'][-1] > 0 for action in actions)
        save(output / 'command-log.json', commands)
        save(output / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tasks', type=int, nargs='+', default=[8, 0])
    parser.add_argument('--states', type=int, nargs='+', default=[1, 11, 21, 31, 41])
    parser.add_argument('--development-plan', default=None,
                        help='Replay previous successful bowl approach on state 1; development only')
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Existing results must be preserved; choose a new output directory')
    if args.development_plan and (args.tasks != [8] or args.states != [1]):
        raise ValueError('A development replay only supports the known bowl state 1')
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'configs/physical_tasks_v4.json').read_text())
    save(output / 'configuration.json', config)
    snapshot_protocol(output, args)
    protocol = json.loads((output / 'protocol.json').read_text())
    for source in [Path(__file__), ROOT / 'configs/physical_tasks_v4.json']:
        relative = source.relative_to(ROOT)
        destination = output / 'source-snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        protocol['sources_sha256'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol.update(development_only=args.development_plan is not None,
                    physical_contacts_enabled=True, software_collision_checks=False,
                    native_rekep_optimizer_used=False,
                    benchmark_type=config['benchmark_type'],
                    state_1_previously_used_for_development=True,
                    original_strict_results_preserved=True)
    save(output / 'protocol.json', protocol)
    ledger = ROOT / 'results/rekep-libero-api/ledger.json'
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    planned = [(task, index) for task in args.tasks for index in args.states]
    save(output / 'manifest.json', {'planned_episodes': planned, 'all_in_denominator': True})
    detector = None
    if 0 in args.tasks:
        from vlm_tamp.rekep_libero.perception import GraspGen
        detector = GraspGen()
    summaries = []
    for task, index in planned:
        summaries.append(run_episode(task, index, output / f'goal{task:02d}-state{index:03d}',
                                     config, detector, args.development_plan))
        save(output / 'summary.json', {'planned_episodes': len(planned), 'completed_episodes': len(summaries),
                                     'official_task_successes': sum(row['official_task_success'] for row in summaries),
                                     'api_ledger_unchanged': hashlib.sha256(ledger.read_bytes()).hexdigest() == before,
                                     'episodes': summaries})


if __name__ == '__main__':
    main()
