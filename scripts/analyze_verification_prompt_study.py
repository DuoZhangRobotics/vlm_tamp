"""Read completed prompt-study evidence; never send evaluator data to the actor."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_tamp.checkpoint_diagnostic import read_json, sha256, write_json


def read(path, default=None):
    return read_json(path) if Path(path).exists() else default


def paired_initial_inputs(folders):
    paths = [p / 'attempt01/opened/obs.npz' for p in folders]
    if not all(p.exists() for p in paths):
        return {'available': False}
    same, differences = {}, {}
    with np.load(paths[0]) as a, np.load(paths[1]) as b:
        for key in sorted(set(a.files) | set(b.files)):
            equal = key in a and key in b and np.array_equal(a[key], b[key], equal_nan=True)
            same[key] = bool(equal)
            if not equal and key in a and key in b and a[key].shape == b[key].shape:
                delta = b[key].astype(float) - a[key].astype(float)
                finite = np.isfinite(delta)
                changed = finite & (delta != 0)
                differences[key] = {'different_values': int(changed.sum()),
                    'maximum_absolute_difference': float(np.abs(delta[finite]).max()) if finite.any() else None,
                    'mean_absolute_difference': float(np.abs(delta[finite]).mean()) if finite.any() else None}
                if key.endswith('_rgb'):
                    differences[key]['different_pixels'] = int(np.any(changed, axis=2).sum())
    metadata = [read(p.parent / 'metadata.json', {}) for p in paths]
    robot = {key: float(np.abs(np.asarray(metadata[0][key]) - np.asarray(metadata[1][key])).max())
             for key in ['joint_positions', 'eef_pose_world_xyzw', 'gripper_qpos']}
    return {'available': True, 'arrays_identical': same, 'all_arrays_identical': all(same.values()),
            'all_depth_world_robot_prior_arrays_identical': all(v for k, v in same.items() if not k.endswith('_rgb')),
            'differences': differences, 'maximum_robot_state_differences': robot}


def native_diagnostics(folder, events):
    actions = read(folder / 'attempt01/initial/actions.json', [])
    contacts = read(folder / 'attempt01/initial/evaluation_contact_diagnostics.json', {}).get('steps', [])
    if not actions or not contacts:
        return {'available': False}
    angles = [-np.degrees(next(j['position'] for j in row['evaluation_only_articulation_state']
                              if j['joint_name'] == 'microwave_1_microjoint')) for row in contacts]
    touching = [row['step'] for row in contacts if any(
        'microwave_1' in ' '.join(contact['geom_names']) for contact in row['robot_involved_contacts'])]
    step, boundaries = 60, {}
    episode = folder / 'attempt01'
    for path in sorted(episode.glob('*-issued-*.npy'), key=lambda p: int(p.name.split('-')[0])):
        count = len(np.load(path))
        boundaries[int(path.name.split('-')[0])] = (step, step + count)
        step += count
    if step != len(actions):
        raise ValueError('Unmapped physical references in ' + str(folder))
    sources, predictions = Counter(), []
    for path in sorted(episode.glob('*-scene-pose-update.json')):
        update = read_json(path)
        sources[update['source']] += 1
        if update['source'] == 'hand_motion_prediction':
            start, end = boundaries[int(path.name.split('-')[0])]
            predictions.append({'artifact': path.name, 'start_step': start, 'end_step': end,
                'hand_predicted_degrees': float(np.degrees(update['hand_predicted_angle_rad'])),
                'native_door_change_degrees': float(angles[end - 1] - angles[start - 1])})
    endpoint = None
    for event in events:
        if event['request']['operation'] not in {'execute', 'execute_sequence'} or not event['response'].get('ok'):
            continue
        result = event['response']['result']
        if event['request']['operation'] == 'execute_sequence':
            result = result['stages'][0]
        index = result['observation']['control_steps'] - 1
        action = actions[index]
        endpoint = {'step': index + 1, 'tcp_target_error_m': action.get('position_error_m'),
                    'maximum_joint_target_error_rad': float(np.abs(np.asarray(action['joint_positions']) -
                                                                 np.asarray(action['target_joints'])).max())}
        break
    first_success = next((i for i, a in enumerate(actions) if a['evaluation_task_success']), None)
    return {'available': True, 'evaluator_only': True,
        'native_success': first_success is not None,
        'control_steps_after_first_success': len(actions) - first_success - 1 if first_success is not None else 0,
        'peak_native_opening_degrees': float(max(angles)), 'final_native_opening_degrees': float(angles[-1]),
        'last_robot_microwave_contact_step': touching[-1] if touching else None,
        'contact_interpretation': 'Any robot/microwave contact, not a retained-grasp certificate.',
        'scene_update_sources': dict(sources), 'hand_prediction_events': predictions,
        'prediction_over_one_degree_with_native_change_below_005_degrees': sum(
            abs(p['hand_predicted_degrees']) > 1 and abs(p['native_door_change_degrees']) < .05 for p in predictions),
        'first_executed_path_endpoint': endpoint}


def analyze(out):
    protocol = read_json(out / 'protocol.json')
    cases = {c['case_id']: c for c in read_json(out / 'cases-private.json')}
    offline = read(out / 'results.json', {'responses': []})['responses']
    scored = read(out / 'scored-results-private.json', {}).get('responses', [])
    offline_groups = {}
    for condition in protocol['conditions']:
        selected = [r for r in offline if r['condition'] == condition]
        group = {'responses': len(selected), 'valid_decisions': sum(r['valid_decision'] for r in selected),
                 'mean_response_seconds': statistics.mean(r['wall_seconds'] for r in selected) if selected else None,
                 'scores_by_subset': {}}
        for subset in ['calibration', 'additional_development']:
            rows = [r for r in scored if r['condition'] == condition and cases[r['case_id']]['subset'] == subset]
            group['scores_by_subset'][subset] = {'reviewed_responses': len(rows),
                'total_score': sum(r['total'] for r in rows), 'maximum_score': 8 * len(rows),
                'mean_score': statistics.mean(r['total'] for r in rows) if rows else None}
        offline_groups[condition] = group
    physical = []
    for job in read_json(out / 'physical-schedule-private.json'):
        result_path = out / f'{job["case_id"]}-result.json'
        if not result_path.exists():
            physical.append({**job, 'completed': False})
            continue
        result = read_json(result_path)
        folder = out / 'physical' / job['case_id']
        events = [json.loads(l) for l in (folder / 'tool-events.jsonl').read_text().splitlines()]
        audit = read(folder / 'audit.json', {})
        episodes = result['attempts']
        types = Counter(e['request']['operation'] for e in events)
        memory = folder / 'public-inputs/experience/guidelines.json'
        physical.append({**job, 'completed': True, 'native_success': result['native_success'],
            'control_steps': episodes[0]['control_steps'] if episodes else None,
            'termination_reason': result['agent_exit'].get('termination_reason'),
            'normal_terminal': result['normal_terminal'], 'launcher_returncode': result['returncode'],
            'wall_seconds': result['wall_seconds'], 'agent_wall_seconds': result['agent_exit'].get('wall_seconds'),
            'thread_id': result['agent_exit'].get('thread_id'), 'physical_attempts': len(episodes),
            'tool_call_counts': dict(types), 'robot_tool_calls': len(events),
            'failed_planning_calls': sum(e['request']['operation'].startswith('plan_') and
                (not e['response'].get('ok') or e['response'].get('result', {}).get('planned') is False) for e in events),
            'successful_model_updates': sum(e['request']['operation'] in {'ground', 'set_model', 'apply_tracking'}
                                          and e['response'].get('ok') for e in events),
            'isolation_passed': audit.get('filesystem_isolation', {}).get('passed'),
            'forbidden_feedback_fields': audit.get('forbidden_evaluation_fields_in_feedback'),
            'api_ledgers_unchanged': audit.get('api_ledgers_unchanged_since_checkpoint'),
            'issued_references_match': all(x.get('references_match') for x in audit.get('trajectory_audit', [])),
            'execution_veto_checks': sum(e.get('execution_veto_checks', 0) for e in episodes),
            'premature_end': any(e.get('premature_end', False) for e in episodes),
            'prompt_correct': sha256(folder / 'agent-prompt.txt') == sha256(out / 'inputs' / f'physical-{job["condition"]}.txt'),
            'memory_correct': memory.exists() and sha256(memory) == protocol['memory_sha256'],
            'native_diagnostics': native_diagnostics(folder, events)})
    pairs = [{'state': state, **paired_initial_inputs([out / 'physical' / f'state{state:03d}-{condition}'
              for condition in protocol['conditions']])} for state in [45, 0]]
    maintenance = read(out / 'post-run-maintenance.json', {}).get('files', {})
    compatible_sources = all(sha256(ROOT / n) == (
        maintenance[n]['after_sha256'] if n in maintenance and maintenance[n]['before_sha256'] == h else h)
        for n, h in protocol['source_sha256'].items())
    result = {'offline': offline_groups, 'physical': physical, 'initial_pairs': pairs,
        'registered_source_hashes_match': all(sha256(ROOT / n) == h for n, h in protocol['source_sha256'].items()),
        'source_hashes_match_with_documented_postrun_setup_maintenance': compatible_sources,
        'numerical_source_hashes_match': all(sha256(ROOT / n) == h for n, h in protocol['numerical_source_sha256'].items()),
        'measured_sources_verified_at_collection_end': read(out / 'source-verification-at-collection-end.json', {}),
        'registered_input_hashes_match': all(sha256(out / n) == h for n, h in protocol['registered_files_sha256'].items()),
        'scope': 'Development prompt comparison; offline rubric scores and physical native success are separate outcomes.',
        'native_diagnostics_never_supplied_to_actor': True}
    write_json(out / 'analysis-private.json', result)
    print(json.dumps({'offline': offline_groups,
        'physical': [{k: r.get(k) for k in ['case_id', 'completed', 'native_success', 'control_steps',
                     'termination_reason']} for r in physical]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze((ROOT / args.output).resolve())
