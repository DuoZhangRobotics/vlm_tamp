"""One bounded Qwen robotics pilot using the existing public robot service.

This is an integration/feasibility trial. Its local function-calling harness
differs from Codex, so its outcomes are not a controlled Luna/model comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from vlm_tamp.local_agent import (LocalChatClient, PublicTools, run_actor, write_json,
    DEFAULT_MAX_OUTPUT_TOKENS,DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TEMPERATURE,DEFAULT_PRESENCE_PENALTY)
from vlm_tamp.subscription_agent import clean_environment, verify_isolation
from report_subscription_pilot import archive_workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state', type=int, required=True)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8127/v1')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--max-output-tokens',type=int,default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument('--model-call-seconds',type=float,default=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument('--temperature',type=float,default=DEFAULT_TEMPERATURE,
                        help='Qwen general-thinking sampling temperature; historical coding-style probes used 0.6.')
    parser.add_argument('--presence-penalty',type=float,default=DEFAULT_PRESENCE_PENALTY,
                        help='Checkpoint-recommended general-thinking presence penalty.')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'results') or not 0 <= args.state < 50 or not 1 <= args.seconds <= 1800:
        raise ValueError('Use a new results directory, state 0..49, and at most 1800 seconds')
    client = LocalChatClient(args.endpoint,'qwen9b',out/'model-calls',seed=args.seed,
        max_tokens=args.max_output_tokens,request_timeout_seconds=args.model_call_seconds,
        temperature=args.temperature,presence_penalty=args.presence_penalty)
    model_info=client.server_info()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'server-model-info.json',model_info)
    work = Path(tempfile.mkdtemp(prefix='vlm-qwen-public-'))
    env = clean_environment()
    backend, summary, error = None, None, None
    guide = ROOT / 'data/agent_memory/luna-scene-motion-v1/guidelines.json'
    subprocess.run(['/usr/bin/python3', str(ROOT / 'scripts/prepare_subscription_pilot.py'),
        '--workspace', str(work), '--output', str(out), '--state', str(args.state),
        '--attempts', '1', '--stop-on-success', '--persistent-episode', '--surface-contact-tools',
        '--compact-interface', '--abstract-experience', '--guidelines-source', str(guide)],
        cwd=ROOT, env=env, check=True)
    readonly = [work / 'experience', work / 'example', work / 'tools', work / 'tools.txt',
                work / 'robot.py', work / 'robot_interface.py']
    isolation = verify_isolation(work, [ROOT / 'src/vlm_tamp/rigid_hinge_tracking.py',
        ROOT / 'results/astra-luna-comparison-20261023/failure-diagnoses.json', out / 'protocol.json'], readonly_paths=readonly)
    write_json(out / 'isolation.json', isolation)
    if not isolation['passed']:
        raise RuntimeError('Public-workspace isolation preflight failed')
    protocol = json.loads((out / 'protocol.json').read_text())
    protocol.update(model='Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        model_runtime='vllm-0.32.0', authentication='local unauthenticated loopback; no external inference',
        actor_harness='local_agent-v2-named-contracts', numerical_backend='curobo_v2',
        scope='Development integration pilot on a reused diagnostic state; not a matched Codex/Luna comparison.',
        one_physical_episode=True, actor_seconds=args.seconds, output_tokens_per_call=client.max_tokens,
        model_request_seconds=client.request_timeout_seconds,context_tokens=model_info['max_model_len'],
        temperature=client.temperature, presence_penalty=client.presence_penalty,
        top_p=.95, top_k=20, model_sampling_seed=args.seed,
        private_geometry_supplied=False, memory_updates=False, reflection_reviewer=False,
        different_harness_from_codex=True)
    write_json(out / 'protocol.json', protocol)
    sources = [*sorted((ROOT / 'src/vlm_tamp').rglob('*.py')), *sorted((ROOT / 'src/vlm_tamp/native').glob('*.cc')),
               *sorted((ROOT / 'scripts').glob('*.py')), ROOT / 'configs/articulated_libero.json']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(out / 'source-hashes.json', hashes)
    prompt = (ROOT / 'results/astra-luna-comparison-20261023/inputs/prompt.txt').read_text()
    prompt += '\nLocal interface: use read_file for documentation and artifacts, view_image to inspect original images, run_python for numerical analysis, and write_file for notes. reset, observe and episode_status are initially enabled named functions. Use robot_help to read another operation contract and enable its named function; reading tools/OPERATION.txt also enables it. Call that function with its exact argument schema. These wrappers expose the existing public workspace and do not choose robot actions.\n'
    (out / 'prompt.txt').write_text(prompt)
    write_json(out / 'input-hashes.json', {str(p.relative_to(work)): hashlib.sha256(p.read_bytes()).hexdigest()
        for base in [work / 'experience', work / 'example', work / 'tools'] for p in base.rglob('*') if p.is_file()})
    try:
        with (out / 'service.log').open('w') as log:
            backend = subprocess.Popen([str(ROOT / '.runtime/envs/rekep-libero/bin/python'),
                str(ROOT / 'scripts/serve_subscription_robot.py'), '--output', str(out), '--workspace', str(work),
                '--state', str(args.state), '--attempts', '1', '--scene-variant', 'nominal',
                '--numerical-backend', 'curobo_v2', '--fit-diagnostics', '--stop-on-success',
                '--persistent-episode', '--surface-contact-tools'], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT)
            write_json(out / 'processes.json', {'service_pid': backend.pid, 'public_workspace': str(work)})
            deadline = time.monotonic() + 90
            while not (work / 'requests').exists():
                if backend.poll() is not None:
                    raise RuntimeError('Robot service exited during startup; see service.log')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Robot service startup timed out')
                time.sleep(.1)
            deadline = time.monotonic() + args.seconds
            record = {'deadline_monotonic': deadline, 'wall_budget_seconds': args.seconds}
            write_json(out / 'episode-deadline.json', record)
            write_json(work / 'robot_data/episode-deadline.json', record)
            summary = run_actor(client, PublicTools(work, extra_readonly=readonly), prompt, out, seconds=args.seconds)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        write_json(out / 'runtime-error.json', {'error': error})
    finally:
        write_json(out / 'stop-service-reason.json', {'reason': (summary or {}).get('termination_reason', 'actor_runtime_error')})
        if backend is not None and backend.poll() is None:
            (out / 'stop-service').touch()
            backend.wait(timeout=900)
        evaluation = json.loads((out / 'evaluation-private.json').read_text()) if (out / 'evaluation-private.json').exists() else {}
        episodes = evaluation.get('attempts', [])
        events = [json.loads(x) for x in (out / 'tool-events.jsonl').read_text().splitlines()] if (out / 'tool-events.jsonl').exists() else []
        tool_times = {}
        for event in events:
            key = event['request']['operation']
            tool_times[key] = tool_times.get(key, 0.) + event['wall_seconds']
        original_inputs = json.loads((out / 'input-hashes.json').read_text())
        input_ok = all((work / p).is_file() and hashlib.sha256((work / p).read_bytes()).hexdigest() == h for p, h in original_inputs.items())
        source_ok = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
        result = {'state': args.state, 'error': error, 'actor': summary, 'episodes': episodes,
            'native_success': any(x.get('official_task_success') for x in episodes),
            'control_steps': sum(x.get('control_steps', 0) for x in episodes),
            'robot_service_seconds_by_operation': tool_times,
            'model_request_seconds': sum(x['wall_seconds'] for x in client.calls),
            'model_calls': len(client.calls), 'supplied_inputs_unchanged': input_ok,
            'sources_unchanged': source_ok, 'isolation_passed': isolation['passed'],
            'controls_after_success': sum(x.get('control_steps_after_first_success', 0) for x in episodes),
            'comparability': 'Local harness integration trial; no controlled model-quality comparison with prior Codex runs.'}
        write_json(out / 'summary.json', result)
        archive_workspace(work, out / 'agent-workspace')
        shutil.rmtree(work)
        print(json.dumps(result), flush=True)
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
