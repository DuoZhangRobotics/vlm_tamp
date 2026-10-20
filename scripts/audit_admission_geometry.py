"""Evaluation-only native contact checks for rejected grasp configurations."""
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vlm_tamp.coupled_storage import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    episode, output = ROOT/args.episode, ROOT/args.output
    output.mkdir(parents=True, exist_ok=False)
    for name in list(os.environ):
        if name.endswith(('_API_KEY', '_TOKEN', '_SECRET')):
            os.environ.pop(name, None)
    from vlm_tamp.libero import configure_libero
    configure_libero(gl='egl')
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import ControlEnv
    from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
    from vlm_tamp.rekep_libero.geometry import pose_matrix
    import mujoco
    row = read_json(episode/'summary.json')
    metadata = read_json(episode/'opened/metadata.json')
    actions = read_json(episode/'initial/actions.json')[:metadata['control_steps']]
    task = next(t for t in read_json(ROOT/'configs/articulated_libero.json')['tasks'] if t['label'] == row['task'])
    random.seed(0)
    np.random.seed(0)
    suite = benchmark.get_benchmark_dict()['libero_90'](task_order_index=0)
    index = next(i for i in range(suite.n_tasks) if Path(suite.get_task_bddl_file_path(i)).stem == task['name'])
    env = ControlEnv(bddl_file_name=suite.get_task_bddl_file_path(index), use_camera_obs=False,
        has_renderer=False, has_offscreen_renderer=False, control_freq=20, horizon=1000, ignore_done=True)
    try:
        env.seed(0)
        env.reset()
        env.set_init_state(np.asarray(suite.get_task_init_states(index)[row['initial_state']]))
        robot = env.robots[0]
        arm, grip = np.asarray(robot._ref_joint_pos_indexes), np.asarray(robot._ref_gripper_joint_pos_indexes)
        for _ in range(5):
            env.step(np.zeros(7))
        for action in actions:
            env.step(np.asarray(action['control']))
        model, live = env.sim.model._model, env.sim.data._data
        prepared_error = float(np.max(np.abs(live.qpos[arm]-metadata['joint_positions'])))
        if prepared_error > 1e-7:
            raise ValueError(f'Setup replay differs from recorded robot state: {prepared_error} rad')
        saved = {k: np.asarray(getattr(live, k)).copy() for k in ['qpos', 'qvel', 'act', 'ctrl']}
        saved_time = float(live.time)
        nonrobot = np.ones(model.nq, bool)
        nonrobot[np.r_[arm, grip]] = False
        clone = mujoco.MjData(model)
        clone.qpos[:] = live.qpos
        clone.qvel[:] = live.qvel
        clone.act[:] = live.act
        aperture = float(read_json(episode/'geometry.json')['cylinder_radius_m'])+.001
        chain = RobotChain(read_json(episode/'robot-chain.json'))
        names = [env.sim.model.geom_id2name(i) or '' for i in range(model.ngeom)]
        flags = [name.startswith(('robot0_', 'gripper0_')) for name in names]
        # Asset identifiers classify contacts in evaluation only. The policy
        # continues to use its observed surface tags and never reads this file.
        handle_geoms = {'microwave_1_g14', 'microwave_1_g15', 'microwave_1_g16'}
        configurations, counts = [], Counter()
        maximum_fk_error = 0.
        for path in sorted(episode.glob('selection-*/start-validation-*.json*')):
            record = read_json(path)
            for start in record.get('starts', []):
                if start.get('terminal_gate') != 'start_scene_collision':
                    continue
                q = np.asarray(start['projected_state'])[:7]
                clone.qpos[arm] = q
                clone.qpos[grip] = [aperture, -aperture]
                mujoco.mj_forward(model, clone)
                if np.max(np.abs(clone.qpos[nonrobot]-saved['qpos'][nonrobot]), initial=0.) > 1e-12:
                    raise ValueError('A cloned diagnostic changed nonrobot joint positions')
                tcp = pose_matrix(chain.forward(q))
                error = float(np.linalg.norm(clone.site_xpos[int(robot.eef_site_id)]-tcp[:3, 3]))
                maximum_fk_error = max(maximum_fk_error, error)
                if error > 2e-5:
                    raise ValueError('Native robot differs from recorded calibrated kinematics')
                contacts = []
                for contact in clone.contact[:clone.ncon]:
                    ids = [int(contact.geom1), int(contact.geom2)]
                    if not any(flags[i] for i in ids) or contact.dist >= -.0001:
                        continue
                    pair = [names[i] for i in ids]
                    robot_names = [names[i] for i in ids if flags[i]]
                    external = [names[i] for i in ids if not flags[i]]
                    intended = (len(robot_names) == 1 and 'finger' in robot_names[0]
                                and len(external) == 1 and external[0] in handle_geoms)
                    contacts.append({'geoms': pair, 'penetration_m': -float(contact.dist),
                                     'intended_handle_contact': intended})
                collisions = start['closed_scene']['collisions']
                inferred_only = all(c['scene_part'] == 'moving' and
                    c['blocked_points'] == c.get('blocked_inferred_panel_points', 0) for c in collisions)
                unpermitted = [c for c in contacts if not c['intended_handle_contact']]
                configurations.append({'source': str(path.relative_to(episode)), 'pose': record['pose_index'],
                    'rank': start['rank'], 'model_rejection_inferred_only': inferred_only,
                    'native_robust_contacts': contacts, 'native_unpermitted_contacts': len(unpermitted)})
                counts['queried'] += 1
                counts['native_clear_including_intended_contact'] += not unpermitted
                counts['native_no_robust_contacts'] += not contacts
                if inferred_only:
                    counts['inferred_only_queried'] += 1
                    counts['inferred_only_native_clear'] += not unpermitted
        live_unchanged = (float(live.time) == saved_time and all(np.array_equal(getattr(live, k), v) for k, v in saved.items()))
        if not live_unchanged:
            raise ValueError('Live simulator changed during isolated configuration queries')
        result = {'evaluation_only': True, 'policy_inputs_include_native_geometry': False,
            'physical_grasps_executed': 0, 'native_contacts_predict_grasp_success': False,
            'initial_state': row['initial_state'], 'seed': 0, 'setup_actions_replayed': len(actions),
            'settling_actions_replayed': 5, 'prepared_robot_max_joint_error_rad': prepared_error,
            'live_simulator_unchanged_during_queries': live_unchanged,
            'maximum_native_vs_calibrated_tcp_error_m': maximum_fk_error,
            'robust_penetration_threshold_m': .0001, 'counts': dict(counts),
            'source_episode': str(episode.relative_to(ROOT)),
            'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        write_json(output/'configurations.json', configurations)
        write_json(output/'summary.json', result)
        print(result, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
