"""Evaluator-only replay of retained commands when historical RGB-D is absent.

This does not rerun perception, planning, or agent decisions. Native geometry is
read only after the simulator has closed, for comparison with historical logs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from replay_subscription_episode import compress_observations, source_commands
from vlm_tamp.coupled_storage import compact_episode, read_json, write_json
from vlm_tamp.profiled_sim_client import ProfiledClient, execute_batch


def run(source, output, attempt=1):
    source, output = Path(source).resolve(), Path(output).resolve()
    events = [json.loads(line) for line in (source / 'tool-events.jsonl').read_text().splitlines()]
    if any(e['request']['operation'] == 'execute_sequence' and e['response']['ok'] for e in events):
        raise ValueError('Sequence artifacts need an explicit stage mapping; refusing an incomplete replay')
    commands = source_commands(source, attempt)
    original = source / f'attempt{attempt:02d}'
    actions = read_json(original / 'initial/actions.json')
    historical = np.asarray([r['target_joints'] for r in actions if r.get('execution_controller') == 'JOINT_POSITION'])
    references = np.vstack([c['references'] for c in commands])
    if not np.array_equal(references, historical):
        raise ValueError('Retained commands do not exactly cover the historical joint references')
    protocol = read_json(source / 'protocol.json')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', {
        'scope': 'development evaluator-only saved-command replay; not a fresh agent trial',
        'source': str(source), 'attempt': attempt, 'state_index': protocol['case']['state_index'],
        'seed': 0, 'maximum_control_steps': 600, 'stop_on_native_success': True,
        'historical_raw_rgbd_available': (original / 'opened/obs.npz').exists(),
        'exact_sensor_reconstruction_claimed': False,
        'perception_or_planning_reexecuted': False, 'execution_veto_checks': 0,
        'commands': [{'event': c['event'], 'gripper': c['gripper'], 'steps': len(c['references']),
                      'sha256': hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()} for c in commands],
    })
    sim, error, tracking, executions = None, None, [], []
    started = time.monotonic()
    try:
        sim = ProfiledClient(output / 'simulator.log', timeout=300)
        state = sim.request('reset', task_name='KITCHEN_SCENE7_open_the_microwave',
                            state_index=protocol['case']['state_index'], output_dir=str(output / 'initial'),
                            seed=0, suite_name='libero_90', scene_variant=protocol.get('scene_variant', 'nominal'),
                            stop_on_success=True)
        state = sim.action(state['metadata']['eef_pose_world_xyzw'], gripper=-1, max_steps=60,
                           output_dir=output / 'opened')
        write_json(output / 'robot-chain.json', sim.robot_chain())
        state = sim.enable_joint_tracking(kp=300.)
        for command in commands:
            state, result = execute_batch(sim, state, command['references'], gripper=command['gripper'],
                                          event_sink=tracking.append, stop_on_success=True)
            executions.append({'event': command['event'], **{k: v for k, v in result.items() if k != 'events'}})
            if result['terminated_on_simulator_success']:
                break
        sim.observe(output / 'final')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            sim.close()
    actual = read_json(output / 'initial/actions.json') if (output / 'initial/actions.json').exists() else []
    n = min(len(actual), len(actions))
    comparisons = {}
    for key in ('joint_positions', 'eef_pose_world_xyzw', 'gripper_qpos'):
        if n:
            delta = np.asarray([a[key] for a in actual[:n]]) - np.asarray([a[key] for a in actions[:n]])
            comparisons[key + '_maximum_absolute_difference'] = float(np.abs(delta).max())
    first_success = next((a['step'] for a in actual if a['evaluation_task_success']), None)
    issued = np.asarray([t['reference_joints'] for t in tracking]).reshape(-1, 7)
    result = {
        'error': error, 'control_steps': len(actual), 'historical_control_steps': len(actions),
        'native_success': first_success is not None, 'first_native_success_step': first_success,
        'steps_after_first_native_success': len(actual) - first_success if first_success else 0,
        'issued_references_match_source_prefix': bool(np.array_equal(issued, references[:len(issued)])),
        'all_saved_commands_issued': len(issued) == len(references),
        'historical_robot_state_comparison': comparisons,
        'historical_raw_sensor_comparison': 'unavailable in compact handoff; not claimed',
        'wall_seconds': time.monotonic() - started, 'execution_veto_checks': 0,
    }
    write_json(output / 'tracking.json', tracking)
    write_json(output / 'executions.json', executions)
    write_json(output / 'summary.json', result)
    result['storage'] = compact_episode(output)
    result['lossless_sensor_compression'] = compress_observations(output)
    write_json(output / 'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('storage', 'lossless_sensor_compression')}), flush=True)
    if error:
        raise RuntimeError(error)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    run(args.source, args.output, args.attempt)
