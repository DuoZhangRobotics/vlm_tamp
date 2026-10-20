"""Validate physical joint tracking and synchronized robot-only geometry.

This is a small controller check, not a task-success episode or collision
certificate. It spends no API budget and never reads scene object identities.
"""

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from vlm_tamp.coupled_client import CoupledClient
from vlm_tamp.coupled_scene import RobotVolumes
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain


TASKS = {'open_microwave': 'KITCHEN_SCENE7_open_the_microwave',
         'close_microwave': 'KITCHEN_SCENE6_close_the_microwave'}
GEOMETRY_TOLERANCE_M = .0001
PLANNER_MARGIN_M = .002


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def geometry_category(name):
    if 'finger' in name:
        return 'finger'
    if 'gripper' in name and 'hand' in name:
        return 'hand'
    return 'arm'


def compare_volumes(predicted, expected):
    predicted = {volume['robot_geom_name']: volume for volume in predicted if not volume.get('fixed')}
    expected = {volume['robot_geom_name']: volume for volume in expected if not volume.get('fixed')}
    if set(predicted) != set(expected):
        raise ValueError('Robot collision geometry name sets do not match')
    rows = []
    for name in sorted(expected):
        first, second = predicted[name], expected[name]
        if first['type'] != second['type']:
            raise ValueError('Robot collision geometry types disagree: ' + name)
        if first['type'] == 'convex':
            a, b = np.asarray(first['vertices_world']), np.asarray(second['vertices_world'])
            if a.shape != b.shape or not len(a):
                raise ValueError('Robot collision geometry vertex shapes disagree: ' + name)
            error = float(np.linalg.norm(a - b, axis=1).max())
        elif first['type'] == 'sphere':
            error = float(np.linalg.norm(np.asarray(first['center_world']) - second['center_world'])
                          + abs(float(first['radius']) - float(second['radius'])))
        else:
            raise ValueError('Unsupported robot collision geometry: ' + first['type'])
        rows.append({'robot_geom_name': name, 'category': geometry_category(name), 'max_error_m': error})
    if not rows:
        raise ValueError('No moving robot collision geometry was compared')
    categories = {category: {'count': sum(row['category'] == category for row in rows),
        'max_error_m': max((row['max_error_m'] for row in rows if row['category'] == category), default=0.)}
        for category in ('arm', 'hand', 'finger')}
    maximum = max(row['max_error_m'] for row in rows)
    return {'accepted': maximum <= GEOMETRY_TOLERANCE_M, 'max_error_m': maximum,
            'categories': categories, 'volumes': rows, 'tolerance_m': GEOMETRY_TOLERANCE_M}


def geometry_check(sim, chain, metadata, output):
    """Compare at identical q and aperture after a fresh synchronized observe.

    A physical gripper may drift slightly during a hold. Consequently, using
    reset-time finger volumes would confound geometry and aperture changes.
    Both isolated reference and comparison below use this snapshot's aperture.
    """
    q = np.asarray(metadata['joint_positions'], float)
    aperture = np.asarray(metadata['gripper_qpos'], float)
    reference = sim.fk(chain.reference_joints, include_volumes=True, gripper_qpos=aperture)
    isolated = sim.fk(q, include_volumes=True, gripper_qpos=aperture)
    if (not np.allclose(reference['gripper_qpos'], aperture, rtol=0, atol=1e-12)
            or not np.allclose(isolated['gripper_qpos'], aperture, rtol=0, atol=1e-12)):
        raise ValueError('Isolated robot FK did not retain the explicitly captured aperture')
    robot = RobotVolumes(chain, reference['robot_collision_volumes_world'], chain.reference_joints)
    predicted = robot.forward(q)
    result = {
        'reference_joints': chain.reference_joints.tolist(), 'test_joints': q.tolist(),
        'captured_gripper_qpos': aperture.tolist(),
        'fixed_aperture_poe_vs_isolated_fk': compare_volumes(predicted, isolated['robot_collision_volumes_world']),
        'synchronized_snapshot_vs_isolated_fk': compare_volumes(
            metadata['robot_collision_volumes_world'], isolated['robot_collision_volumes_world']),
        'same_explicit_aperture_in_both_isolated_fk_calls': True,
        'scene_geometry_used': False,
    }
    result['accepted'] = (result['fixed_aperture_poe_vs_isolated_fk']['accepted']
                          and result['synchronized_snapshot_vs_isolated_fk']['accepted'])
    save(output / 'reference-isolated-fk.json', reference)
    save(output / 'test-isolated-fk.json', isolated)
    save(output / 'geometry-check.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', choices=TASKS, default='open_microwave')
    parser.add_argument('--state-index', type=int, default=1)
    parser.add_argument('--joint-kp', type=float, default=150.)
    args = parser.parse_args()
    if not np.isfinite(args.joint_kp) or not 0 < args.joint_kp <= 300:
        parser.error('--joint-kp must be finite and in (0,300]')
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'task': TASKS[args.task], 'state_index': args.state_index, 'seed': 0,
        'settling_steps': 5, 'hold_steps': 8, 'small_move_steps': 10,
        'small_move_joint_index': 0, 'small_move_delta_rad': .01,
        'hold_max_joint_error_tolerance_rad': .008,
        'small_move_max_joint_error_tolerance_rad': .005,
        'small_move_min_signed_progress_rad': .005,
        'geometry_tolerance_m': GEOMETRY_TOLERANCE_M,
        'comparison_planner_clearance_margin_m': PLANNER_MARGIN_M,
        'geometry_tolerance_fraction_of_margin': GEOMETRY_TOLERANCE_M / PLANNER_MARGIN_M,
        'controller': 'official robosuite JOINT_POSITION, damping ratio1',
        'joint_kp': args.joint_kp,
        'new_api_calls': 0, 'physical_task_test': False,
        'tracking_tolerances_are_not_a_collision_clearance_certificate': True,
    }
    save(output / 'protocol.json', protocol)
    summary = {'accepted': False, 'no_api_calls': True, 'physical_task_test': False,
               'new_api_calls': 0, 'task': args.task, 'state_index': args.state_index}
    sim = None
    try:
        sim = CoupledClient(log_path=output / 'simulator.log')
        state = sim.reset(TASKS[args.task], args.state_index, output / 'initial', seed=0)
        record = sim.robot_chain()
        save(output / 'chain-reference.json', record)
        chain = RobotChain(record)
        switch = sim.enable_joint_tracking(kp=args.joint_kp)
        save(output / 'switch.json', switch)
        q_hold = np.asarray(switch['metadata']['joint_positions'])
        hold = sim.joint_action(q_hold, gripper=0, max_steps=8)
        save(output / 'hold-command.json', {'target_joints': q_hold.tolist(), 'gripper': 0,
                                         'max_steps': 8, 'response': hold})
        held = sim.observe(output / 'after-hold')
        hold_error = float(np.max(np.abs(np.asarray(held['metadata']['joint_positions']) - q_hold)))
        hold_geometry = geometry_check(sim, chain, held['metadata'], output / 'after-hold')

        move_start = np.asarray(held['metadata']['joint_positions'])
        target = move_start.copy()
        target[0] += .01
        if np.any(target <= chain.limits[:, 0]) or np.any(target >= chain.limits[:, 1]):
            raise ValueError('Small validation motion would exceed configured robot limits')
        moved = sim.joint_action(target, gripper=0, max_steps=10)
        save(output / 'small-move-command.json', {'target_joints': target.tolist(), 'gripper': 0,
                                                'max_steps': 10, 'response': moved})
        final = sim.observe(output / 'after-small-move')
        final_q = np.asarray(final['metadata']['joint_positions'])
        move_error = float(np.max(np.abs(final_q - target)))
        progress = float(final_q[0] - move_start[0])
        move_geometry = geometry_check(sim, chain, final['metadata'], output / 'after-small-move')
        maximum = max(check[key]['max_error_m'] for check in (hold_geometry, move_geometry)
                      for key in ('fixed_aperture_poe_vs_isolated_fk', 'synchronized_snapshot_vs_isolated_fk'))
        summary.update(
            hold_max_joint_error_rad=hold_error, small_motion_max_joint_error_rad=move_error,
            small_motion_signed_joint0_progress_rad=progress,
            robot_geometry_fk_max_error_m=maximum,
            geometry_tolerance_m=GEOMETRY_TOLERANCE_M,
            comparison_planner_clearance_margin_m=PLANNER_MARGIN_M,
            controls=final['control_steps'],
            hold_geometry=hold_geometry, small_motion_geometry=move_geometry,
            accepted=bool(hold_error <= .008 and move_error <= .005 and progress >= .005
                          and hold_geometry['accepted'] and move_geometry['accepted']),
        )
    except Exception as error:
        summary.update(error=str(error), traceback=traceback.format_exc())
    finally:
        if sim is not None:
            try:
                save(output / 'close.json', sim.close())
            except Exception as error:
                summary.update(accepted=False, close_error=str(error))
        save(output / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {'hold_geometry', 'small_motion_geometry', 'traceback'}}), flush=True)
    return 0 if summary['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
