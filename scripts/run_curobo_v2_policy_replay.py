"""Replan a retained Luna decision sequence with a selected numerical backend.

This is a development engineering replay, not a new learner episode. It uses
fresh physics and RGB-D, reruns grasp selection and planning, and stops if the
fixed decision sequence requests a plan that the new backend could not produce.
It never executes the old saved joint trajectories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'.runtime/mc-vamp-deps')]
from serve_subscription_robot import RobotService, handle_request
from vlm_tamp.coupled_storage import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/compact-luna-practice-20261022/practice-v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--numerical-backend', choices=['legacy','curobo_v2'], default='curobo_v2')
    args = parser.parse_args()
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=False)
    workspace = output/'public-workspace';workspace.mkdir()
    requests, replies = workspace/'requests', workspace/'replies'
    requests.mkdir();replies.mkdir()
    events = [json.loads(s) for s in (args.source/'tool-events.jsonl').read_text().splitlines()]
    selected, resets = [], 0
    for event in events:
        if event['request']['operation'] == 'reset':
            resets += 1
        if resets > 1:
            break
        selected.append(event)
    protocol = read_json(args.source/'protocol.json')
    state = protocol['case']['state_index']
    hashes = {}
    for path in [*ROOT.glob('src/vlm_tamp/**/*.py'), *ROOT.glob('scripts/*.py')]:
        name = path.relative_to(ROOT)
        target = output/'source'/name;target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(name)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output/'source-hashes.json', hashes)
    write_json(output/'protocol.json', {'scope': __doc__, 'source': str(args.source.resolve()),
        'state_index': state, 'numerical_backend': args.numerical_backend,
        'historical_trajectories_executed': False, 'separate_reflection_reviewer': False,
        'maximum_control_steps': 600, 'stop_on_native_success': True,
        'source_tool_events_sha256': hashlib.sha256((args.source/'tool-events.jsonl').read_bytes()).hexdigest()})
    service = RobotService(output, workspace, state, 1, fit_diagnostics=True,
        stop_on_success=True, numerical_backend=args.numerical_backend)
    started = time.perf_counter();reason = 'fixed_sequence_exhausted';error = None
    try:
        for event in selected:
            request = event['request'];op = request['operation']
            if op == 'practice_status':
                continue
            if op == 'execute' and request['arguments']['plan_id'] not in service.plans:
                reason = 'fixed_sequence_requires_unavailable_plan'
                break
            service.calls = int(event['sequence'])-1
            path = requests/(uuid.uuid4().hex+'.json')
            write_json(path, request)
            t0 = time.perf_counter();handle_request(service, path, replies)
            response = read_json(replies/path.name)
            result = response.get('result', {})
            print(json.dumps({'sequence': event['sequence'], 'operation': op,
                'ok': response['ok'], 'seconds': time.perf_counter()-t0,
                'task_success': result.get('task_success'), 'error': response.get('error')}), flush=True)
            if service.episode_success:
                reason = 'native_success';break
            if op in {'reset','ground','plan_grasp','execute','gripper'} and not response['ok']:
                reason = 'fixed_sequence_operation_failed';error = response.get('error');break
    finally:
        service.finish_attempt(reason)
        changed = [name for name, digest in hashes.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest]
        summary = {'native_success': bool(service.results and service.results[-1].get('official_task_success')),
            'termination_reason': reason, 'error': error, 'attempts': service.results,
            'wall_seconds': time.perf_counter()-started, 'numerical_backend': args.numerical_backend,
            'source_integrity': {'passed': not changed, 'changed': changed},
            'scope': 'Fixed decision replay with fresh plans and physics; not an adaptive learner or held-out test.'}
        write_json(output/'summary.json', summary)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
