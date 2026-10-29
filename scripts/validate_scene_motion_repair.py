"""Replay a development prefix, then physically test one newly planned arc.

This is a declared evaluator diagnostic, excluded from agent success rates.
Native geometry is read only after the service closes, never by the planner.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

from diagnose_contact_prefix import compress_diagnostic_json
from replay_subscription_episode import compress_observations, source_commands
from serve_subscription_robot import RobotService
from vlm_tamp.coupled_storage import read_json, write_json


def run(source, output, through, degrees, mode):
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / 'public'
    workspace.mkdir()
    events = [json.loads(line) for line in (source / 'tool-events.jsonl').read_text().splitlines()]
    by_sequence = {e['sequence']: e for e in events}
    commands = {c['event']: c for c in source_commands(source, 1) if c['event'] <= through}
    references = np.vstack([c['references'] for c in commands.values()])
    expected = read_json(source / 'attempt01/initial/actions.json')
    joint_references = np.asarray([a['target_joints'] for a in expected if a.get('execution_controller') == 'JOINT_POSITION'])
    assert np.array_equal(references, joint_references[:len(references)])
    state_index = read_json(source / 'protocol.json')['case']['state_index']
    write_json(output / 'protocol.json', {'scope': 'development physical continuation test; no agent calls',
        'source': str(source), 'state_index': state_index, 'through_event': through,
        'continuation': {'degrees': degrees, 'grasp_constraint': mode},
        'native_geometry_used_for_planning': False, 'execution_veto_checks': 0})
    service = RobotService(output, workspace, state_index, 1, stop_on_success=True)
    report = {'events': [], 'error': None, 'continuation_planned': False, 'continuation_executed': False}
    started = time.monotonic()
    try:
        for e in events:
            seq, request = e['sequence'], e['request']
            if seq > through:
                break
            if not e['response']['ok']:
                continue
            op, args = request['operation'], request.get('arguments', {})
            service.calls = seq
            if seq in commands:
                c = commands[seq]
                label = args['plan_id'] if op == 'execute' else ('close' if args.get('close', True) else 'open')
                arc = label.startswith('p') and by_sequence[int(label[1:])]['request']['operation'] == 'plan_arc'
                result = service.issue(c['references'], c['gripper'], label, articulated=arc)
            elif op.startswith('plan_'):
                continue
            else:
                service.calls -= 1
                result = service.dispatch(op, args)
            report['events'].append({'sequence': seq, 'request': request, 'result': result})
            write_json(output / 'test-events.json', report['events'])
            print(json.dumps({'sequence': seq, 'operation': op, 'control_steps': service.state['control_steps']}), flush=True)
        report['prefix_control_steps'] = service.state['control_steps']
        report['prefix_references_match'] = bool(np.array_equal(
            np.asarray([r['reference_joints'] for r in service.events]), references))
        assert report['prefix_references_match']
        service.calls = through
        request = {'degrees': degrees, 'grasp_constraint': mode}
        plan = service.dispatch('plan_arc', request)
        report['continuation_planned'] = True
        report['plan'] = plan
        result = service.dispatch('execute', {'plan_id': plan['plan_id'], 'gripper': 0, 'speed': .35, 'hold': 6})
        report.update(continuation_executed=True, execution=result)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        (output / 'error.txt').write_text(traceback.format_exc())
    finally:
        service.finish_attempt('declared_physical_continuation_test_complete')
        report['lossless_sensor_compression'] = compress_observations(output)
        report['lossless_diagnostic_compression'] = compress_diagnostic_json(output)
    actual = read_json(output / 'attempt01/initial/actions.json')
    contacts = read_json(output / 'attempt01/initial/evaluation_contact_diagnostics.json')['steps']
    n = report.get('prefix_control_steps', min(len(actual), len(expected)))
    report['maximum_prefix_joint_difference_rad'] = float(np.abs(
        np.asarray([a['joint_positions'] for a in actual[:n]]) - np.asarray([a['joint_positions'] for a in expected[:n]])).max())
    def opening(row):
        joint = next(j for j in row['evaluation_only_articulation_state'] if j['joint_name'] == 'microwave_1_microjoint')
        return float(-np.degrees(joint['position']))
    report['evaluation_only'] = {'prefix_opening_degrees': opening(contacts[n - 1]),
        'final_opening_degrees': opening(contacts[-1]),
        'native_progress_degrees': opening(contacts[-1]) - opening(contacts[n - 1]),
        'final_control_steps': len(actual), 'native_success': any(a['evaluation_task_success'] for a in actual),
        'steps_with_robot_object_contact_after_prefix': sum(any(
            'microwave_1' in ' '.join(c['geom_names']) for c in row['robot_involved_contacts']) for row in contacts[n:])}
    report['wall_seconds'] = time.monotonic() - started
    write_json(output / 'summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in {
        'events', 'plan', 'execution', 'lossless_sensor_compression', 'lossless_diagnostic_compression'}}), flush=True)
    if report['error']:
        raise RuntimeError(report['error'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through', type=int, required=True)
    parser.add_argument('--degrees', type=float, required=True)
    parser.add_argument('--mode', choices=['point', 'rigid', 'jaw_axis'], default='point')
    args = parser.parse_args()
    run(args.source, args.output, args.through, args.degrees, args.mode)
