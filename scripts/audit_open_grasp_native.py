"""Evaluation-only native collision audit; never used by the grounding policy."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maximum', type=int, default=107)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Existing environment/configuration are reused without installation or
    # editing; service credentials have no role in this offline audit.
    for name in list(os.environ):
        if name.endswith(('_API_KEY', '_TOKEN', '_SECRET')):
            os.environ.pop(name, None)
    os.environ['LIBERO_CONFIG_PATH'] = str(ROOT/'.runtime/libero')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, str(ROOT/'third_party/LIBERO'))
    import mujoco
    from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
    from vlm_tamp.rekep_libero.geometry import pose_matrix
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import ControlEnv

    case = args.run/'open_microwave-state001'
    inputs = []
    for path in sorted(case.glob('start-validation-*.json')):
        pose = read(path)
        for start in pose['starts']:
            inputs.append({'pose': pose['pose_index'], 'rank': start['rank'],
                'state': start['projected_state'], 'model_collisions': start['closed_scene']['collisions']})
    inputs = inputs[:args.maximum]
    random.seed(0)
    np.random.seed(0)
    suite = benchmark.get_benchmark_dict()['libero_90'](task_order_index=0)
    task_name = 'KITCHEN_SCENE7_open_the_microwave'
    matches = [i for i in range(suite.n_tasks)
               if Path(suite.get_task_bddl_file_path(i)).stem == task_name]
    if len(matches) != 1:
        raise ValueError('Expected exactly one official opening task')
    task_index = matches[0]
    official = np.asarray(suite.get_task_init_states(task_index)[1])
    initial_state_sha = hashlib.sha256(official.tobytes()).hexdigest()
    begin = time.monotonic()
    env = ControlEnv(bddl_file_name=suite.get_task_bddl_file_path(task_index),
        use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False,
        control_freq=20, horizon=1000, ignore_done=True)
    try:
        env.seed(0)
        env.reset()
        env.set_init_state(official)
        model, live = env.sim.model._model, env.sim.data._data
        robot = env.robots[0]
        arm_indices = np.asarray(robot._ref_joint_pos_indexes, int)
        finger_indices = np.asarray(robot._ref_gripper_joint_pos_indexes, int)
        robot_indices = np.r_[arm_indices, finger_indices]
        nonrobot = np.ones(model.nq, bool)
        nonrobot[robot_indices] = False
        baseline_qpos, baseline_qvel = live.qpos.copy(), live.qvel.copy()
        baseline_time = float(live.time)
        clone = mujoco.MjData(model)
        clone.qpos[:] = live.qpos
        clone.qvel[:] = live.qvel
        clone.act[:] = live.act
        clone.ctrl[:] = live.ctrl
        clone.mocap_pos[:] = live.mocap_pos
        clone.mocap_quat[:] = live.mocap_quat
        clone.time = live.time
        cloud = np.load(case/'handle-cloud.npy')
        handle_tree = cKDTree(cloud)
        axis_xy = np.asarray(read(case/'task-spec.json')['source_anchor_world_m'])[:2]
        radius = float(np.quantile(np.linalg.norm(cloud[:, :2]-axis_xy, axis=1), .98)+.004)
        bar_low, bar_high = float(cloud[:, 2].min()-.015), float(cloud[:, 2].max()+.015)
        aperture = float(read(case/'geometry.json')['cylinder_radius_m'])+.001
        chain = RobotChain(read(case/'robot-chain.json'))
        saved_steps = read(case/'initial/evaluation_contact_diagnostics.json')['steps']
        saved_state = next(row for row in saved_steps if row['step'] == 60)
        saved_joints = {row['joint_name']: row for row in saved_state['evaluation_only_articulation_state']}
        scene_joint_comparison = []
        for index in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            if name not in saved_joints or name.startswith(('robot0_', 'gripper0_')):
                continue
            position = float(live.qpos[model.jnt_qposadr[index]])
            saved = saved_joints[name]
            scene_joint_comparison.append({'joint_name': name, 'official_initial_position': position,
                'saved_step60_position': saved['position'], 'position_difference': position-saved['position'],
                'anchor_difference_m': float(np.linalg.norm(live.xanchor[index]-saved['anchor_world_m'])),
                'axis_difference': float(np.linalg.norm(live.xaxis[index]-saved['axis_world']))})
        rows = []
        maximum_tcp_position_error = 0.
        maximum_tcp_rotation_matrix_error = 0.
        for entry in inputs:
            clone.qpos[:] = baseline_qpos
            clone.qpos[arm_indices] = np.asarray(entry['state'])[:7]
            clone.qpos[finger_indices] = [aperture, -aperture]
            mujoco.mj_forward(model, clone)
            known_tcp = pose_matrix(chain.forward(np.asarray(entry['state'])[:7]))
            position_error = float(np.linalg.norm(clone.site_xpos[int(robot.eef_site_id)]-known_tcp[:3, 3]))
            rotation_error = float(np.linalg.norm(clone.site_xmat[int(robot.eef_site_id)].reshape(3, 3)-known_tcp[:3, :3]))
            maximum_tcp_position_error = max(maximum_tcp_position_error, position_error)
            maximum_tcp_rotation_matrix_error = max(maximum_tcp_rotation_matrix_error, rotation_error)
            if position_error > 2e-5 or rotation_error > 2e-4:
                raise AssertionError('Native robot/TCP differs from the frozen calibrated robot model')
            if not np.array_equal(clone.qpos[nonrobot], baseline_qpos[nonrobot]):
                raise AssertionError('Native scene state changed in a cloned configuration check')
            contacts = []
            for contact in clone.contact[:clone.ncon]:
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(index))
                         for index in [contact.geom1, contact.geom2]]
                flags = [bool(name and name.startswith(('robot0_', 'gripper0_'))) for name in names]
                if not any(flags):
                    continue
                position = np.asarray(contact.pos).copy()
                robot_names = [name for name, flag in zip(names, flags) if flag]
                finger_pair = len(robot_names) == 1 and 'finger' in robot_names[0]
                observed_distance = float(handle_tree.query(position)[0])
                cylinder = (np.linalg.norm(position[:2]-axis_xy) <= radius
                            and bar_low <= position[2] <= bar_high)
                contacts.append({'geom_names': names, 'robot_flags': flags,
                    'distance_m': float(contact.dist), 'position_world_m': position.tolist(),
                    'selected_observed_handle_distance_m': observed_distance,
                    'finger_selected_observed_surface_pair': bool(finger_pair and observed_distance <= .006),
                    'finger_observed_cylinder_region_pair': bool(finger_pair and cylinder),
                    'evaluation_only': True})
            penetration = [c for c in contacts if c['distance_m'] < 0]
            robust = [c for c in contacts if c['distance_m'] < -.0001]
            unpermitted_observed = [c for c in robust if not c['finger_selected_observed_surface_pair']]
            unpermitted_cylinder = [c for c in robust if not c['finger_observed_cylinder_region_pair']]
            model_only_inferred = all(c['blocked_inferred_panel_points'] == c['blocked_points']
                                      for c in entry['model_collisions'])
            rows.append({'pose': entry['pose'], 'rank': entry['rank'],
                'model_rejection_inferred_only': model_only_inferred,
                'native_robot_contacts': contacts, 'negative_distance_contacts': len(penetration),
                'robust_penetration_contacts_over_0p1mm': len(robust),
                'robust_nonselected_observed_handle_contacts': len(unpermitted_observed),
                'robust_nonselected_cylinder_contacts': len(unpermitted_cylinder)})
        live_unchanged = (np.array_equal(live.qpos, baseline_qpos)
                          and np.array_equal(live.qvel, baseline_qvel)
                          and live.time == baseline_time)
        if not live_unchanged:
            raise AssertionError('Reference environment changed during cloned native evaluation')
        summary = {'evaluation_only': True, 'policy_uses_native_scene_geometry': False,
            'task': task_name, 'suite': 'libero_90', 'official_state_index': 1, 'seed': 0,
            'official_state_sha256': initial_state_sha, 'hypothetical_starts_checked': len(rows),
            'renderer_used': False, 'environment_actions_executed': 0,
            'physical_episode_or_success_evaluation': False, 'live_reference_state_unchanged': live_unchanged,
            'clone_nonrobot_qpos_unchanged': True, 'gripper_aperture_qpos': [aperture, -aperture],
            'negative_distance_threshold_m': 0., 'robust_penetration_threshold_m': -.0001,
            'contact_role_association': 'finger contact within 6mm of observed selected surface; observed vertical cylinder alternate association reported separately',
            'elapsed_seconds': time.monotonic()-begin,
            'native_robot_vs_frozen_chain_maximum_tcp_position_error_m': maximum_tcp_position_error,
            'native_robot_vs_frozen_chain_maximum_rotation_matrix_error': maximum_tcp_rotation_matrix_error,
            'official_initial_vs_saved_step60_scene_joint_comparison': scene_joint_comparison,
            'limitations': ['Exact official initial state; five settling and sixty gripper-opening actions were not replayed.',
                'Contacts are hypothetical native configuration checks, not executed grasps or physical success.',
                'Geometric observed-handle association labels intended contacts; native geometry names remain evaluation-only.',
                'No native collision results are supplied to the frozen V14 policy or used to change this run.'],
            'groups': {}}
        for name, group in [('all', rows), ('model_inferred_only', [r for r in rows if r['model_rejection_inferred_only']]),
                            ('model_mixed_observed_inferred', [r for r in rows if not r['model_rejection_inferred_only']])]:
            summary['groups'][name] = {'starts': len(group),
                'with_negative_distance_contacts': sum(r['negative_distance_contacts'] > 0 for r in group),
                'with_robust_penetration_contacts': sum(r['robust_penetration_contacts_over_0p1mm'] > 0 for r in group),
                'with_robust_nonselected_observed_handle_contacts': sum(r['robust_nonselected_observed_handle_contacts'] > 0 for r in group),
                'with_robust_nonselected_cylinder_contacts': sum(r['robust_nonselected_cylinder_contacts'] > 0 for r in group)}
        (args.output/'native-contact-configurations.json').write_text(json.dumps(rows, indent=2))
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
