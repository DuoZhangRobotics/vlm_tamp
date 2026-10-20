"""Move, close and lift with simulator contacts, bypassing software admission.

This is a physical grasp diagnostic, not the strict ReKep planner benchmark.
All commands are ordinary robot actions. No object state or attachment is set.
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

from try_rekep_bowl_v3 import cached_bundle, read_observation_v3, save, snapshot_protocol
from vlm_tamp.rekep_libero.geometry import interpolate_path, pose_matrix
from vlm_tamp.rekep_libero.object_motion_v3 import observed_object_cloud, verify_observed_transport
from vlm_tamp.rekep_libero.sim_client_v3 import SimClientV3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/rekep-bowl-physics-only-v1/attempt-01')
    parser.add_argument('--source-plan', default='results/rekep-bowl-pilot-v3/physical-10')
    parser.add_argument('--approach-steps', type=int, default=12)
    parser.add_argument('--closure-steps', type=int, default=60)
    parser.add_argument('--nullspace-kp', type=float, default=80.)
    args = parser.parse_args()
    output, plan_folder = ROOT / args.output, ROOT / args.source_plan
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve existing results; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    snapshot_protocol(output, args)
    script = Path(__file__)
    destination = output / 'source-snapshot/scripts' / script.name
    shutil.copyfile(script, destination)
    protocol = json.loads((output / 'protocol.json').read_text())
    protocol['sources_sha256'][str(script.relative_to(ROOT))] = hashlib.sha256(script.read_bytes()).hexdigest()
    protocol.update(software_collision_checks=False, abort_on_tracking_error=False,
                    simulator_contacts='unchanged; physical contact remains enabled',
                    assisted_grasping=False, constraint_benchmark=False)
    save(output / 'protocol.json', protocol)
    ledger_path = ROOT / 'results/rekep-libero-api/ledger.json'
    ledger_hash = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    sim = SimClientV3(log_path=output / 'simulator.log')
    summary = {'version': 'bowl-physics-only-1.0', 'task_index': 8, 'initial_state': 1,
               'seed': 0, 'new_api_calls': 0, 'software_collision_checks': False,
               'abort_on_tracking_error': False, 'physical_contacts_enabled': True,
               'assisted_grasping': False, 'provided_inventory': False,
               'source_plan': str(plan_folder), 'official_task_success': False,
               'physical_grasp_attempts': 0, 'grasp_lift_success': False, 'status': 'running'}
    before = time.monotonic()
    try:
        state = sim.reset(8, 1, output / 'initial', seed=0)
        state = sim.action(state['metadata']['eef_pose_world_xyzw'], gripper=-1,
                           max_steps=60, output_dir=output / 'opened')
        original = ROOT / 'results/rekep-libero-v2/libero_goal-08-state001-seed0'
        _, evidence = cached_bundle(original, output / 'opened')
        save(output / 'cached-evidence.json', evidence)
        chosen = json.loads((plan_folder / 'chosen-grasp.json').read_text())
        path = np.asarray(json.loads((plan_folder / 'approach-command-poses.json').read_text())['poses_world_xyzw'])
        checked = json.loads((plan_folder / f'approach-{chosen["index"]}-{chosen["variant"]}.json').read_text())
        joints = checked['solution_joints']
        if len(path) != len(joints):
            raise ValueError('Cached robot pose and joint commands must remain paired')
        cloud = np.load(plan_folder / 'selected_cloud.npy', allow_pickle=False)
        np.save(output / 'reference-object-cloud.npy', cloud)
        save(output / 'chosen-grasp.json', chosen)
        summary.update(selected_index=chosen['index'], selected_variant=chosen['variant'])
        execution = []
        for index, waypoint in enumerate(path[1:], start=1):
            state = sim.action(waypoint, gripper=0, max_steps=args.approach_steps,
                               nullspace_joint_target=joints[index], nullspace_kp=args.nullspace_kp)
            execution.append({'waypoint_index': index, 'requested_pose': waypoint,
                              'actual_pose': state['metadata']['eef_pose_world_xyzw'],
                              'reached_pose': state['reached_pose'],
                              'posture_error_rad': state.get('posture_error_rad'),
                              'control_steps': state['control_steps']})
            save(output / 'approach-execution.json', execution)
            if state['step_limit_reached']:
                raise RuntimeError('600-control budget ended during approach')
        target, final_joints = path[-1], joints[-1]
        # Hold the requested grasp while remaining open, allowing physical
        # settling regardless of the software pose-completion threshold.
        state = sim.action(target, gripper=-1, max_steps=20,
                           nullspace_joint_target=final_joints, nullspace_kp=args.nullspace_kp,
                           output_dir=output / 'at-grasp')
        requested = pose_matrix(target)
        actual = pose_matrix(state['metadata']['eef_pose_world_xyzw'])
        summary['at_grasp_position_error_m'] = float(np.linalg.norm(actual[:3, 3] - requested[:3, 3]))
        summary['at_grasp_pose_reached'] = bool(state['reached_pose'])
        state = sim.action(target, gripper=1, max_steps=args.closure_steps,
                           nullspace_joint_target=final_joints, nullspace_kp=args.nullspace_kp,
                           output_dir=output / 'closed')
        summary.update(physical_grasp_attempts=int(state['executed_steps'] > 0),
                       closed_gripper_qpos=state['metadata']['gripper_qpos'], status='closed_gripper')
        closed_pose = np.asarray(state['metadata']['eef_pose_world_xyzw'])
        lift_goal = closed_pose.copy()
        lift_goal[2] += .05
        lift_path = interpolate_path(np.asarray([closed_pose, lift_goal]))
        for waypoint in lift_path[1:]:
            state = sim.action(waypoint, gripper=0, max_steps=12,
                               nullspace_kp=args.nullspace_kp)
            if state['step_limit_reached']:
                raise RuntimeError('600-control budget ended during lift')
        state = sim.action(lift_goal, gripper=1, max_steps=20,
                           nullspace_kp=args.nullspace_kp, output_dir=output / 'lifted-held')
        cameras, _ = read_observation_v3(output / 'lifted-held', state['metadata'])
        motion = pose_matrix(state['metadata']['eef_pose_world_xyzw']) @ np.linalg.inv(pose_matrix(closed_pose))
        observed = observed_object_cloud(cameras, cloud, motion, state['metadata']['robot_collision_volumes_world'])
        np.save(output / 'observed-object-cloud.npy', observed)
        retention = verify_observed_transport(cloud, observed, motion, state['metadata']['gripper_qpos'])
        save(output / 'lift-retention.json', retention)
        summary.update(grasp_lift_success=retention['verified'],
                       actual_hand_lift_m=float(state['metadata']['eef_pose_world_xyzw'][2] - closed_pose[2]),
                       status='grasp_lift_success' if retention['verified'] else 'grasp_not_retained')
    except Exception as error:
        summary.update(status='error', failure=str(error), exception_type=type(error).__name__)
        (output / 'error-traceback.txt').write_text(traceback.format_exc())
    finally:
        closed = sim.close()
        summary.update(control_steps=closed.get('control_steps', 0), video=closed.get('video_path'),
                       elapsed_seconds=time.monotonic() - before)
        actions_path = output / 'initial/actions.json'
        if actions_path.is_file():
            actions = json.loads(actions_path.read_text())
            summary['physical_grasp_attempts'] = int(any(action['control'][-1] > 0 for action in actions))
            summary['closing_controls'] = sum(action['control'][-1] > 0 for action in actions)
            summary['official_task_success'] = any(action['evaluation_task_success'] for action in actions)
        summary['api_ledger_unchanged'] = hashlib.sha256(ledger_path.read_bytes()).hexdigest() == ledger_hash
        save(output / 'summary.json', summary)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
