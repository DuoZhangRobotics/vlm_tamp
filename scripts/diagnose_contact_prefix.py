"""Evaluator-only prefix replay with fresh perception and frozen planner probes.

Physical commands are retained historical references. Perception operations are
repeated on new observations; their original raw RGB-D is unavailable. No agent
is invoked, and none of the diagnostic planner proposals is executed.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from replay_subscription_episode import compress_observations, source_commands
from serve_subscription_robot import RobotService
from vlm_tamp.coupled_storage import preserve_scene, read_json, write_json
from vlm_tamp.mcvamp_contact_numerical import NativeScene, clear_native_cache
from vlm_tamp.partial_pose_arc import full_arc_plan

ROOT = Path(__file__).resolve().parents[1]
SENSOR_OPERATIONS = {'reset', 'observe', 'ground', 'start_tracking',
                     'measure_tracking', 'apply_tracking', 'inspect_scene', 'set_model'}


def compress_diagnostic_json(output):
    """Keep large diagnostic records losslessly, using read_json's gzip support."""
    records = []
    for folder in (output / 'snapshots', output / 'probes'):
        for path in sorted(folder.rglob('*.json')):
            if path.stat().st_size <= 1024 * 1024:
                continue
            original = path.read_bytes()
            destination = path.with_suffix('.json.gz')
            if destination.exists():
                raise FileExistsError(destination)
            with gzip.open(destination, 'wb', compresslevel=1) as stream:
                stream.write(original)
            with gzip.open(destination, 'rb') as stream:
                if stream.read() != original:
                    raise ValueError(f'Lossless diagnostic compression failed: {path}')
            records.append({'path': str(path), 'sha256': hashlib.sha256(original).hexdigest(),
                            'before_bytes': len(original), 'after_bytes': destination.stat().st_size})
            path.unlink()
    return records


def snapshot(service, output):
    folder = output / 'snapshots' / f'event{service.calls:03d}'
    folder.mkdir(parents=True)
    preserve_scene(service.scene, folder)
    write_json(folder / 'metadata.json', service.state['metadata'])
    write_json(folder / 'observation.json', {
        'control_steps': service.state['control_steps'], 'geometry_epoch': service.epoch,
        'fresh_replay_observation': True, 'historical_sensor_equivalence_claimed': False,
        'observation': service.latest_observation,
        'public_workspace': str(service.workspace), 'spec': service.spec,
        'choices': service.choices, 'pose_update': getattr(service.scene, 'pose_update', None),
    })
    return str(folder)


def probe(scene, record, metadata, config, arguments, folder):
    """Use the same full_arc_plan entry point and settings as contact_arc."""
    folder.mkdir(parents=True)
    goal = float(np.radians(arguments['degrees']))
    mode = arguments.get('grasp_constraint', config.get('grasp_constraint', 'point'))
    options = {**config, 'hinge_refinement_accepted': bool(arguments.get('refined', False)),
               'grasp_constraint': mode}
    start = np.asarray(metadata['joint_positions'])
    write_json(folder / 'planning-input.json', {
        'arguments': arguments, 'configuration': options, 'metadata': metadata,
        'task_spec': scene.spec, 'reference_joints': start, 'goal_angle_rad': goal,
        'planning_only': True, 'instrumentation': 'Record Python validate inputs and results in memory; native search internals are unchanged.',
    })
    checks = []
    with NativeScene(scene, record, metadata, start, goal, aligned=False,
                     grasp_constraint=mode, output=folder) as context:
        context.motion_options = options
        validate = context.validate

        def traced_validate(q):
            result = validate(q)
            checks.append({'joints': np.asarray(q).copy(), 'result': result,
                           'progress_rad': float(context.progress(q))})
            return result

        context.validate = traced_validate
        path, result = full_arc_plan(context, goal,
                                    seconds=float(options.get('arc_search_seconds', 60.)),
                                    proposals=int(options.get('arc_goal_proposals', 128)),
                                    seed=int(options.get('arc_seed', 0)))
        result['exact_solution'] = bool(len(path) > 1 and result['status'] == 'solved')
        write_json(folder / 'result.json', result)
        write_json(folder / 'native-scene.json', context.record)
        write_json(folder / 'validation-trace.json', checks)
        np.save(folder / 'path.npy', path)
        np.save(folder / 'door-progress.npy', [context.progress(q) for q in path])
    brief = {k: result[k] for k in ('status', 'exact_solution', 'admissible_goals',
                                   'last_progress_rad', 'continuation_layers', 'total_seconds') if k in result}
    brief.update(arguments=arguments, folder=str(folder), python_validation_checks=len(checks),
                 generated_path_executed=False)
    return brief


def run(source, output, through, probe_events, extra_point_probe):
    source, output = source.resolve(), output.resolve()
    events = [json.loads(line) for line in (source / 'tool-events.jsonl').read_text().splitlines()]
    by_sequence = {e['sequence']: e for e in events}
    commands = {c['event']: c for c in source_commands(source, 1) if c['event'] <= through}
    if not commands:
        raise ValueError('No physical commands in requested prefix')
    actions = read_json(source / 'attempt01/initial/actions.json')
    references = np.vstack([c['references'] for c in commands.values()])
    historical = np.asarray([a['target_joints'] for a in actions if a.get('execution_controller') == 'JOINT_POSITION'])
    if not np.array_equal(references, historical[:len(references)]):
        raise ValueError('Retained command prefix does not match historical joint references')
    protocol = read_json(source / 'protocol.json')
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / 'public'
    workspace.mkdir()
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [ROOT / 'scripts/serve_subscription_robot.py',
                               *sorted((ROOT / 'src/vlm_tamp').glob('*.py')),
                               ROOT / 'src/vlm_tamp/native/mcvamp_contact_numerical.cc']}
    write_json(output / 'protocol.json', {
        'scope': 'development evaluator-only saved-command prefix replay and fresh perception; excluded from agent success rates',
        'source': str(source), 'state_index': protocol['case']['state_index'], 'through_event': through,
        'fresh_planner_probe_events': probe_events, 'extra_point_probe_degrees': extra_point_probe,
        'agent_model_calls': 0, 'historical_rgbd_available': False,
        'exact_historical_sensor_reconstruction_claimed': False,
        'native_scene_geometry_is_evaluator_only': True, 'planner_source_hashes': source_hashes,
        'commands': [{'event': c['event'], 'gripper': c['gripper'], 'steps': len(c['references']),
                      'sha256': hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()} for c in commands.values()],
        'all_generated_planner_paths_remain_unexecuted': True, 'execution_veto_checks': 0,
    })
    service = RobotService(output, workspace, state_index=protocol['case']['state_index'],
                           attempts=1, stop_on_success=True, scene_variant=protocol.get('scene_variant', 'nominal'))
    inherited = read_json(output / 'inherited-tools.json')
    inherited.update(prior_successful_paths_replayed=True, evaluator_only_prefix_diagnostic=True)
    write_json(output / 'inherited-tools.json', inherited)
    report = {'error': None, 'events': [], 'probes': []}
    started = time.monotonic()
    try:
        for event in events:
            seq, request = event['sequence'], event['request']
            if seq > through:
                break
            operation, arguments = request['operation'], request.get('arguments', {})
            if not event['response']['ok']:
                continue
            before = time.monotonic()
            if seq in commands:
                command = commands[seq]
                label = arguments['plan_id'] if operation == 'execute' else ('close' if arguments.get('close', True) else 'open')
                plan_event = by_sequence[int(label[1:])] if label.startswith('p') else None
                articulated = bool(plan_event and plan_event['request']['operation'] == 'plan_arc')
                service.calls = seq
                response = service.issue(command['references'], command['gripper'], label, articulated=articulated)
            elif operation in SENSOR_OPERATIONS:
                service.calls = seq - 1
                response = service.dispatch(operation, arguments)
            elif operation.startswith('plan_'):
                # Planning only creates proposals. Replay uses their retained issued references.
                continue
            else:
                raise ValueError(f'Unhandled successful prefix operation: {seq} {operation}')
            row = {'sequence': seq, 'request': request, 'response': response,
                   'control_steps': service.state['control_steps'], 'wall_seconds': time.monotonic() - before}
            if service.scene is not None:
                row['snapshot'] = snapshot(service, output)
            report['events'].append(row)
            write_json(output / 'replay-events.json', report['events'])
            print(json.dumps({'event': seq, 'operation': operation, 'steps': row['control_steps'], 'seconds': row['wall_seconds']}), flush=True)
            if service.episode_success:
                break
        scene, record, metadata, config = service.scene, service.record, service.state['metadata'], service.config
        write_json(output / 'robot-chain.json', record)
        service.finish_attempt('declared_evaluator_prefix_complete')
        actual = read_json(output / 'attempt01/initial/actions.json')
        issued = np.asarray([e['reference_joints'] for e in service.events])
        report['replay'] = {'control_steps': len(actual),
                            'issued_references_match_source_prefix': bool(np.array_equal(issued, references)),
                            'native_success': any(a['evaluation_task_success'] for a in actual),
                            'historical_raw_sensor_comparison': 'Unavailable; observations are new'}
        for key in ('joint_positions', 'eef_pose_world_xyzw', 'gripper_qpos'):
            difference = np.asarray([a[key] for a in actual]) - np.asarray([a[key] for a in actions[:len(actual)]])
            report['replay'][key + '_maximum_absolute_difference'] = float(np.abs(difference).max())
        if not report['replay']['issued_references_match_source_prefix']:
            raise ValueError('Executed prefix differs from retained references')
        if not report['replay']['native_success']:
            queries = [(f'event{seq:03d}', by_sequence[seq]['request']['arguments']) for seq in probe_events]
            if any(by_sequence[seq]['request']['operation'] != 'plan_arc' for seq in probe_events):
                raise ValueError('Only archived arc requests can be diagnostic probes')
            if extra_point_probe is not None:
                queries.append(('matched-point-probe', {'degrees': extra_point_probe, 'grasp_constraint': 'point'}))
            for name, arguments in queries:
                answer = probe(scene, record, metadata, config, arguments, output / 'probes' / name)
                report['probes'].append(answer)
                write_json(output / 'summary.json', report)
                print(json.dumps(answer), flush=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        (output / 'error.txt').write_text(traceback.format_exc())
    finally:
        service.finish_attempt('declared_evaluator_prefix_complete')
        clear_native_cache()
        report['lossless_sensor_compression'] = compress_observations(output)
        report['lossless_diagnostic_compression'] = compress_diagnostic_json(output)
        report['wall_seconds'] = time.monotonic() - started
        write_json(output / 'summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in {
        'events', 'lossless_sensor_compression', 'lossless_diagnostic_compression'}}), flush=True)
    if report['error']:
        raise RuntimeError(report['error'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through', type=int, required=True)
    parser.add_argument('--probe-events', type=int, nargs='*', default=[])
    parser.add_argument('--extra-point-probe', type=float)
    args = parser.parse_args()
    run(args.source, args.output, args.through, args.probe_events, args.extra_point_probe)
