"""Replay a recorded public prefix, then serve the unchanged numerical tools.

Replay is an explicit experimental initialization, not an agent policy. No
native object pose, contact record, or evaluator diagnosis initializes a model.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np

from replay_subscription_episode import compress_observations, source_commands
from serve_subscription_robot import RobotService, handle_request
from vlm_tamp.checkpoint_diagnostic import build_packet, latest_observation, prefix_events
from vlm_tamp.coupled_storage import write_json


def compare_arrays(actual, expected):
    """Report equality without confusing rendering noise with geometry drift."""
    with np.load(actual) as a, np.load(expected) as b:
        result = {}
        if set(a.files) != set(b.files):
            raise ValueError('Replay array inventory differs')
        for key in a.files:
            x, y = a[key], b[key]
            shape_ok = x.shape == y.shape
            equal = shape_ok and np.array_equal(x, y, equal_nan=True)
            delta = float(np.nanmax(np.abs(x.astype(float) - y.astype(float)))) if shape_ok else None
            # A one-byte RGB rendering difference is disclosed; geometric arrays
            # must agree to 1e-8. This is an initialization check, not an action veto.
            tolerance = 1 if key.endswith('_rgb') else 1e-8
            result[key] = {'exact': bool(equal), 'maximum_difference': delta,
                           'within_tolerance': bool(shape_ok and np.allclose(
                               x, y, atol=tolerance, rtol=0, equal_nan=True))}
        return result


def replay_checkpoint(service, source, cutoff):
    source = Path(source).resolve()
    rows = prefix_events(source / 'tool-events.jsonl', cutoff)
    by_sequence = {row['sequence']: row for row in rows}
    if any(row['request']['operation'] == 'execute_sequence' for row in rows):
        raise ValueError('This registered replay supports execute/gripper prefixes only')
    commands = {c['event']: c for c in source_commands(source, 1) if c['event'] <= cutoff}
    packet = service.output / 'source-prefix-public'
    manifest = build_packet(source, cutoff, packet)
    write_json(service.output / 'source-prefix-provenance.json', manifest)
    replayed = []
    for old in rows:
        seq, request = old['sequence'], old['request']
        op, arguments = request['operation'], request.get('arguments', {})
        started = time.monotonic()
        service.calls = seq
        if op.startswith('plan_'):
            # Planning makes no motion. Its original issued references are
            # replayed below; re-running a stochastic planner would change input.
            response = old['response']
        else:
            try:
                if seq in commands:
                    command = commands[seq]
                    label = arguments['plan_id'] if op == 'execute' else ('close' if arguments.get('close', True) else 'open')
                    articulated = label.startswith('p') and by_sequence[int(label[1:])]['request']['operation'] == 'plan_arc'
                    result = service.issue(command['references'], command['gripper'], label,
                                           articulated=articulated)
                else:
                    service.calls = seq - 1
                    result = service.dispatch(op, arguments)
                response = {'ok': True, 'result': result}
            except Exception as error:
                if old['response']['ok']:
                    raise
                response = {'ok': False, 'error_type': type(error).__name__, 'error': str(error)}
                if response != old['response']:
                    raise ValueError('Replay failed differently from recorded prefix') from error
        if response['ok'] != old['response']['ok']:
            raise ValueError('Replay request outcome changed')
        record = {'sequence': seq, 'request': request, 'response': response,
                  'wall_seconds': time.monotonic() - started}
        replayed.append(record)
        with (service.output / 'tool-events.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps({'replayed_event': seq, 'operation': op,
                          'control_steps': service.state['control_steps']}), flush=True)
    # Preserve original public plan artifacts. Existing live observations/model
    # files take precedence; no private source data is copied into the workspace.
    for relative in manifest['asset_provenance']:
        if not relative.startswith('robot_data/'):
            continue
        target = service.workspace / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(packet / relative, target)
    observation = latest_observation(replayed)
    status = service.episode_status()
    prefix = service.workspace / 'checkpoint-prefix'
    prefix.mkdir()
    (prefix / 'history.jsonl').write_text(''.join(json.dumps({k: r[k] for k in
        ('sequence', 'request', 'response')}) + '\n' for r in replayed))
    write_json(prefix / 'checkpoint.json', {
        'last_completed_event': cutoff, 'current_observation': observation,
        'current_model': service.spec, 'geometry_epoch': service.epoch, **status,
        'scope': 'Live episode initialized by replaying this recorded prefix. Paths resolve from workspace root.',
        'historical_plan_tokens': 'Historical plans are stale. Obtain a fresh plan before further execution.',
    })
    references = np.vstack([command['references'] for command in commands.values()])
    audit = {'issued_references_match': bool(np.array_equal(references,
        np.asarray([event['reference_joints'] for event in service.events]))),
        'control_steps': service.state['control_steps'], 'geometry_epoch': service.epoch,
        'native_success': service.episode_success, 'native_geometry_used': False,
        'current_model_exact': service.spec == rows[-1]['response']['result']['model'],
        'model_arrays': compare_arrays(service.public / f'attempt01/geometry{cutoff:03d}/points.npz',
                                      packet / f'robot_data/attempt01/geometry{cutoff:03d}/points.npz'),
        'sensor_arrays': compare_arrays(service.workspace / observation['sensor_arrays'],
                                       packet / observation['sensor_arrays']),
        'robot_state_exact': all(observation[key] == latest_observation(rows)[key] for key in
                                 ('joint_positions', 'eef_pose_world_xyzw', 'gripper_qpos')),
    }
    audit['passed'] = (audit['issued_references_match'] and audit['control_steps'] == 277
        and audit['geometry_epoch'] == rows[-1]['response']['result']['geometry_epoch']
        and audit['current_model_exact'] and audit['robot_state_exact'] and not audit['native_success']
        and all(v['within_tolerance'] for group in ('sensor_arrays', 'model_arrays') for v in audit[group].values()))
    write_json(service.output / 'checkpoint-replay-audit.json', audit)
    if not audit['passed']:
        raise ValueError('Checkpoint replay differs; no acting model may start')
    shutil.rmtree(packet)
    write_json(service.public / 'episode-status.json', status)
    write_json(service.output / 'live-episode-status.json', status)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--through', type=int, default=12)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.workspace.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((args.source / 'protocol.json').read_text())
    service = RobotService(args.output, args.workspace, source_protocol['case']['state_index'], 1,
        fit_diagnostics=True, stop_on_success=True, persistent_episode=True, surface_contact_tools=True)
    started = time.monotonic()
    deadline = {'deadline_monotonic': started + 1800, 'wall_budget_seconds': 1800,
                'scope': 'Prefix replay plus subsequent acting; one physical episode'}
    write_json(args.output / 'episode-deadline.json', deadline)
    write_json(service.public / 'episode-deadline.json', deadline)
    reason = 'checkpoint_replay_error'
    try:
        replay_checkpoint(service, args.source, args.through)
        write_json(args.output / 'checkpoint-ready.json', {
            'started_monotonic': started, **deadline, 'replay_seconds': time.monotonic() - started})
        if args.preflight_only:
            reason = 'declared_replay_preflight_complete'
            return
        requests, replies = args.workspace / 'requests', args.workspace / 'replies'
        requests.mkdir(exist_ok=True)
        replies.mkdir(exist_ok=True)
        seen = set()
        while not (args.output / 'stop-service').exists():
            for path in sorted(requests.glob('*.json'), key=lambda p: p.stat().st_mtime):
                if path.name in seen or not re.fullmatch(r'[0-9a-f]{32}\.json', path.name):
                    continue
                if path.is_symlink() or not path.resolve().is_relative_to(requests.resolve()) or path.stat().st_size > 2_000_000:
                    continue
                seen.add(path.name)
                handle_request(service, path, replies)
            time.sleep(.1)
        terminal = args.output / 'stop-service-reason.json'
        reason = json.loads(terminal.read_text())['reason'] if terminal.exists() else 'service_shutdown'
    finally:
        service.finish_attempt(reason)
        write_json(args.output / 'observation-compression.json', compress_observations(args.output))


if __name__ == '__main__':
    main()
