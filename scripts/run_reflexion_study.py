"""Registered three-condition Luna learning pilot on a replayed failed prefix."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_tamp.checkpoint_diagnostic import build_packet, read_json, sha256, verify_packet, write_json
from vlm_tamp.episode_persistence import run_persistent_agent
from vlm_tamp.reflexion_experiment import (
    ACTOR_PROMPT, REVIEW_PROMPT, prepare_review_packet, remaining_actor_seconds, tree_hashes, validate_reflection,
)
from vlm_tamp.storage_budget import require_storage_headroom
from vlm_tamp.subscription_agent import clean_environment, codex_arguments, verify_isolation
from vlm_tamp.public_analysis import DEFAULT_BUNDLE, load_manifest, readonly_paths as analysis_readonly
from run_subscription_agent_pilot import ledger_hashes

SOURCE = ROOT / 'results/tool-repair-learning-20261021/study-v1/transfer/state045-with_learned_memory'
MEMORY = ROOT / 'data/agent_memory/luna-scene-motion-v1/guidelines.json'
MEMORY_SHA = '65f9612d3f66380234d78f3e05c6342f784b3f137ac4df8032f7fb04ba0c13c2'
PYTHON = ROOT / '.runtime/envs/rekep-libero/bin/python'
CONDITIONS = ['no_new_memory', 'episode_record', 'reflection']


def utc():
    return datetime.now(timezone.utc).isoformat()


def freeze_sources(out):
    files = [*sorted((ROOT / 'src/vlm_tamp').rglob('*.py')),
             *sorted((ROOT / 'src/vlm_tamp/native').glob('*.cc')),
             *sorted((ROOT / 'scripts').glob('*.py')), ROOT / 'configs/articulated_libero.json',
             ROOT / 'tests/test_reflexion_experiment.py', ROOT / 'tests/test_public_analysis.py',
             ROOT / 'tests/test_reflexion_restart.py',
             DEFAULT_BUNDLE / 'manifest.json']
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        hashes[str(relative)] = sha256(path)
        target = out / 'source-snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return hashes


def verify_freeze(out):
    protocol = read_json(out / 'protocol.json')
    for name, digest in protocol['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Registered implementation changed: ' + name)
    for name, digest in protocol['input_sha256'].items():
        if sha256(out / name) != digest:
            raise ValueError('Registered input changed: ' + name)
    if sha256(MEMORY) != MEMORY_SHA:
        raise ValueError('Frozen common memory changed')
    return {'passed': True, 'files': len(protocol['source_sha256'])}


def completed_record(case, destination):
    rows = [json.loads(line) for line in (case / 'tool-events.jsonl').read_text().splitlines()]
    manifest = build_packet(case, rows[-1]['sequence'], destination,
        shared_sources={'experience/guidelines.json': MEMORY})
    # Only the permitted official boolean and episode budget outcome leave the
    # evaluator. No geometry, contact, angle or explanatory labels are exposed.
    episodes = read_json(case / 'evaluation-private.json')['attempts']
    if len(episodes) != 1:
        raise ValueError('Expected one completed source episode')
    ep = episodes[0]
    write_json(destination / 'outcome.json', {
        'task_success': ep['official_task_success'], 'control_steps': ep['control_steps'],
        'termination_reason': ep['reason'],
        'scope': 'Outcome of this previous completed attempt; not the live retry.',
    })
    manifest['packet_sha256'] = tree_hashes(destination)
    manifest['outcome_fields'] = ['official task-success boolean', 'control steps', 'termination reason']
    return manifest


def prepare(out, *, conditions=None):
    conditions = list(CONDITIONS if conditions is None else conditions)
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(CONDITIONS):
        raise ValueError('Choose one or more distinct declared conditions')
    out.mkdir(parents=True, exist_ok=False)
    analysis_manifest = load_manifest(DEFAULT_BUNDLE)
    preflight = out.parent / 'preflight002/checkpoint-replay-audit.json'
    if not read_json(preflight)['passed']:
        raise ValueError('Faithful replay preflight must pass before registration')
    if sha256(MEMORY) != MEMORY_SHA:
        raise ValueError('Unexpected common memory')
    schedule = []
    for trial, ordering in [(1, conditions), (2, list(reversed(conditions)))]:
        schedule.extend({'trial': trial, 'condition': condition,
                         'case_id': f'trial{trial:02d}-{condition}'} for condition in ordering)
    write_json(out / 'schedule.json', schedule)
    (out / 'actor-prompt.txt').write_text(ACTOR_PROMPT)
    (out / 'review-prompt.txt').write_text(REVIEW_PROMPT)
    manifest = completed_record(SOURCE, out / 'records/seed')
    write_json(out / 'records/seed-manifest-private.json', manifest)
    hashes = freeze_sources(out)
    write_json(out / 'protocol.json', {
        'created_utc': utc(), 'scope': 'Counterfactual learning pilot seeded by one archived Luna failure; development only.',
        'source_episode': str(SOURCE), 'source_prefix_last_event': 12, 'state_index': 45, 'seed': 0,
        'checkpoint_control_steps': 277, 'remaining_control_steps': 323,
        'replay': 'Exact original issued commands; fresh public sensors and same public grounding/model requests. No native pose restoration.',
        'preflight_audit': str(preflight), 'preflight_sha256': sha256(preflight),
        'model': 'gpt-6-luna', 'reasoning_effort': 'high', 'authentication': 'ChatGPT subscription',
        'public_analysis_environment': {'packages': analysis_manifest['packages'],
            'python_major_minor': analysis_manifest['python_major_minor'],
            'bundle_manifest_sha256': sha256(DEFAULT_BUNDLE / 'manifest.json'),
            'startup_capability_and_isolation_check_required': True},
        'separately_billed_api_calls': False, 'model_weights_updated': False,
        'conditions': conditions, 'physical_attempts': len(schedule), 'attempts_per_condition': 2,
        'review_calls': len(schedule), 'seconds_per_review': 300, 'review_retries': 0,
        'review_restart_context_explicit': True,
        'review_restart_context_source': 'Same fixed public prefix already supplied to the actor; no corrected estimate or private diagnosis.',
        'actor_record_changed_by_review_context_fix': False,
        'comparison_scope': ('Exploratory single-condition follow-up; earlier controls are nonconcurrent.'
                             if len(conditions) == 1 else 'Registered within-block condition comparison.'),
        'compute_control': 'Every condition gets one bounded isolated review before every attempt. Controls discard its text; only reflection exposes it.',
        'matched_caps_not_guaranteed_equal_usage': True,
        'episode_wall_budget_seconds': 1800, 'prefix_replay_time_included': True,
        'control_steps_per_episode': 600, 'stop_on_native_success': True,
        'no_voluntary_early_failure': True, 'max_concurrent_simulators': 1,
        'numerical_tools_frozen': True, 'execution_time_vetoes': False,
        'same_prompt_all_conditions': True, 'memory_sha256': MEMORY_SHA,
        'memory_window': 'Only the preceding completed public episode and, in reflection, one candidate review. Trial 1 uses the shared archived failure. Trial 2 uses its own condition trial 1.',
        'reviewer_inputs': 'Public tool requests/responses, available public artifacts, common frozen guidance, permitted outcome and explicit public retry-start state. No actor-authored hindsight notes, corrected geometry or evaluator diagnosis.',
        'review_validation': 'Format and exact public citation values only. No semantic correction or expert hints. Invalid reflection is retained privately and withheld; no retry; assigned condition stays in denominator.',
        'fresh_contexts': True, 'cross_condition_memory': False,
        'primary_outcome': f'Native task success in all {len(schedule)} scheduled episodes, with infrastructure failures reported separately.',
        'secondary_outcomes': ['control steps', 'elapsed replay/actor/review time', 'model usage', 'citation fidelity', 'subsequent policy changes'],
        'transfer_claim': False, 'memory_promoted_to_default': False,
        'source_sha256': hashes,
        'input_sha256': {name: sha256(out / name) for name in
                         ['schedule.json', 'actor-prompt.txt', 'review-prompt.txt', 'records/seed-manifest-private.json']},
    })
    write_json(out / 'status.json', {'active': False, 'phase': 'registered', 'completed_episodes': 0})


def run_review(out, job, packet, manifest):
    folder = out / 'reviews' / job['case_id']
    folder.mkdir(parents=True, exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix='vlm-reflexion-review-'))
    record = {**job, 'started_utc': utc(), 'valid_reflection': False, 'completed': False}
    started = time.monotonic()
    try:
        shutil.copytree(packet, work / 'inputs')
        readonly = [work / 'inputs', *analysis_readonly(work)]
        write_json(folder / 'isolation.json', verify_isolation(work,
            [out / 'protocol.json', SOURCE / 'evaluation-private.json'], readonly_paths=readonly))
        record['inputs_before'] = verify_packet(work / 'inputs', manifest)
        command = codex_arguments(work, folder / 'answer.txt', readonly_paths=readonly)
        write_json(folder / 'command.json', command)
        with (folder / 'events.jsonl').open('w') as log, (folder / 'stderr.log').open('w') as err:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True,
                stdout=log, stderr=err, cwd=work, env=clean_environment(), start_new_session=True)
            try:
                process.communicate((out / 'review-prompt.txt').read_text(), timeout=300)
            except subprocess.TimeoutExpired:
                record['timed_out'] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
        record['returncode'] = process.returncode
        record['inputs_after'] = verify_packet(work / 'inputs', manifest)
        record['completed'] = process.returncode == 0
        if (work / 'reflection.json').exists():
            shutil.copyfile(work / 'reflection.json', folder / 'reflection.json')
            try:
                rows = [json.loads(line) for line in (packet / 'history.jsonl').read_text().splitlines()]
                record['validation'] = validate_reflection(read_json(folder / 'reflection.json'), rows)
                record['valid_reflection'] = True
            except (ValueError, TypeError) as error:
                record['validation_error'] = str(error)
        else:
            record['validation_error'] = 'No reflection.json produced'
        record['usage'] = []
        for line in (folder / 'events.jsonl').read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') == 'thread.started':
                record['thread_id'] = event['thread_id']
            if event.get('type') == 'turn.completed':
                record['usage'].append(event.get('usage', {}))
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        (folder / 'error.txt').write_text(traceback.format_exc())
    finally:
        # Preserve model-authored work, never duplicate immutable record inputs.
        for path in work.iterdir():
            if path.name in {'inputs', 'allowed-canary.txt', '.analysis_site'} or path.is_symlink():
                continue
            target = folder / 'workspace-output' / path.name
            target.parent.mkdir(exist_ok=True)
            if path.is_dir():
                shutil.copytree(path, target, symlinks=True)
            else:
                shutil.copyfile(path, target)
        shutil.rmtree(work)
        record['wall_seconds'] = time.monotonic() - started
        write_json(folder / 'result.json', record)
    return record


def run_episode(out, job, packet, manifest, review):
    folder = out / 'episodes' / job['case_id']
    folder.mkdir(parents=True, exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix='vlm-reflexion-actor-'))
    env = clean_environment()
    result = {**job, 'started_utc': utc(), 'completed': False, 'native_success': False}
    backend = None
    started = time.monotonic()
    readonly = []
    try:
        write_json(folder / 'storage-preflight.json', require_storage_headroom(ROOT) + require_storage_headroom(work))
        subprocess.run(['/usr/bin/python3', str(ROOT / 'scripts/prepare_subscription_pilot.py'),
            '--workspace', str(work), '--output', str(folder), '--state', '45', '--attempts', '1',
            '--abstract-experience', '--stop-on-success', '--persistent-episode', '--surface-contact-tools',
            '--guidelines-source', str(MEMORY)], cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
        readonly = [work / 'experience/guidelines.json', work / 'checkpoint-prefix', *analysis_readonly(work)]
        if job['condition'] != 'no_new_memory':
            recent = work / 'experience/recent'
            shutil.copytree(packet, recent / 'record')
            verify_packet(recent / 'record', manifest)
            if job['condition'] == 'reflection' and review['valid_reflection']:
                shutil.copyfile(out / 'reviews' / job['case_id'] / 'reflection.json', recent / 'reflection.json')
                result['reflection_exposed'] = True
            readonly.append(recent)
        result.setdefault('reflection_exposed', False)
        protocol = read_json(folder / 'protocol.json')
        protocol.update(counterfactual_checkpoint=True, inherited_prefix_event=12,
            condition=job['condition'], trial=job['trial'], episode_wall_budget_seconds=1800,
            prefix_replay_time_included=True, held_out_initial_state_evaluation=False,
            unconditioned_hinge_fit_diagnostics=True, scene_variant='nominal')
        write_json(folder / 'protocol.json', protocol)
        write_json(folder / 'api-ledger-checkpoint.json', ledger_hashes())
        write_json(folder / 'authentication-check.json', read_json(out / 'authentication-check.json'))
        write_json(folder / 'source-hashes.json', read_json(out / 'protocol.json')['source_sha256'])
        with (folder / 'service.log').open('w') as log:
            command = [str(PYTHON), str(ROOT / 'scripts/serve_reflexion_checkpoint.py'),
                       '--source', str(SOURCE), '--output', str(folder), '--workspace', str(work)]
            write_json(folder / 'service-command.json', command)
            backend = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 600
            while not (folder / 'checkpoint-ready.json').exists() or not (work / 'requests').exists():
                if backend.poll() is not None:
                    raise RuntimeError('Checkpoint service failed before actor launch; inspect service.log')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Checkpoint initialization exceeded 600 seconds')
                time.sleep(.2)
            write_json(folder / 'isolation.json', verify_isolation(work,
                [out / 'protocol.json', SOURCE / 'evaluation-private.json', ROOT / 'src/vlm_tamp/hinged_contact_tools.py'],
                readonly_paths=readonly))
            input_hashes = {str(path.relative_to(work)): (tree_hashes(path) if path.is_dir() else sha256(path)) for path in readonly}
            write_json(folder / 'learner-input-hashes.json', input_hashes)
            ready = read_json(folder / 'checkpoint-ready.json')
            result['replay_seconds'] = ready['replay_seconds']
            seconds = remaining_actor_seconds(ready, time.monotonic())
            result['actor_seconds_granted'] = seconds
            code = run_persistent_agent(work, folder, env, readonly,
                (out / 'actor-prompt.txt').read_text(), seconds)
            result['agent_returncode'] = code
            actual_deadline = read_json(folder / 'episode-deadline.json')['deadline_monotonic']
            result['deadline_extension_seconds'] = actual_deadline - ready['deadline_monotonic']
            result['readonly_inputs_unchanged'] = input_hashes == {
                str(path.relative_to(work)): (tree_hashes(path) if path.is_dir() else sha256(path)) for path in readonly}
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        (folder / 'runner-error.txt').write_text(traceback.format_exc())
    finally:
        if backend is not None and backend.poll() is None:
            (folder / 'stop-service').touch()
            backend.wait(timeout=900)  # Finish any already-issued trajectory.
        if (folder / 'agent-exit.json').exists():
            with (folder / 'report.log').open('w') as log:
                completed = subprocess.run([str(PYTHON), str(ROOT / 'scripts/report_subscription_pilot.py'),
                    '--run', str(folder), '--workspace', str(work)], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT)
            result['report_returncode'] = completed.returncode
        if (folder / 'evaluation-private.json').exists():
            result['attempts'] = read_json(folder / 'evaluation-private.json')['attempts']
            result['native_success'] = any(ep['official_task_success'] for ep in result['attempts'])
        if (folder / 'agent-exit.json').exists():
            result['agent_exit'] = read_json(folder / 'agent-exit.json')
        if (folder / 'audit.json').exists():
            result['completed'] = bool(result.get('agent_exit', {}).get('normal_terminal') and not result.get('error'))
            shutil.rmtree(work)
        else:
            result['unarchived_workspace'] = str(work)
        result['wall_seconds'] = time.monotonic() - started
        result['source_integrity_after'] = verify_freeze(out)
        write_json(folder / 'result.json', result)
    return result


def report(out):
    jobs = read_json(out / 'schedule.json')
    rows, reviews = [], []
    for job in jobs:
        path = out / 'episodes' / job['case_id'] / 'result.json'
        rows.append(read_json(path) if path.exists() else {**job, 'not_completed': True})
        path = out / 'reviews' / job['case_id'] / 'result.json'
        if path.exists():
            reviews.append(read_json(path))
    summary = {'episodes': rows, 'reviews': reviews, 'conditions': {},
               'planned_episodes': len(jobs), 'completed_episodes': sum(r.get('completed', False) for r in rows),
               'scope': 'One practiced failure checkpoint; not held out and not a test of transfer.'}
    for condition in CONDITIONS:
        selected = [r for r in rows if r['condition'] == condition]
        summary['conditions'][condition] = {
            'successes': sum(r.get('native_success', False) for r in selected),
            'scheduled_attempts': len(selected), 'completed_attempts': sum(r.get('completed', False) for r in selected)}
    write_json(out / 'summary.json', summary)
    return summary


def run(out):
    verify_freeze(out)
    status = subprocess.run(['codex', 'login', 'status'], capture_output=True, text=True,
                            env=clean_environment(), check=True)
    if 'Logged in using ChatGPT' not in status.stdout + status.stderr:
        raise RuntimeError('ChatGPT subscription required; API fallback forbidden')
    write_json(out / 'authentication-check.json', {'logged_in_using_chatgpt': True,
        'api_key_authentication': False, 'client_version': subprocess.check_output(['codex', '--version'], text=True).strip(),
        'requested_model': 'gpt-6-luna', 'requested_reasoning': 'high'})
    error = None
    try:
        for job in read_json(out / 'schedule.json'):
            result_path = out / 'episodes' / job['case_id'] / 'result.json'
            if result_path.exists():
                if not read_json(result_path)['completed']:
                    raise RuntimeError('Existing incomplete episode requires diagnosis; no automatic retry')
                continue
            verify_freeze(out)
            if job['trial'] == 1:
                packet = out / 'records/seed'
                manifest = read_json(out / 'records/seed-manifest-private.json')
            else:
                previous = f'trial01-{job["condition"]}'
                packet = out / 'records' / previous
                manifest_path = out / 'records' / f'{previous}-manifest-private.json'
                if not packet.exists():
                    manifest = completed_record(out / 'episodes' / previous, packet)
                    write_json(manifest_path, manifest)
                manifest = read_json(manifest_path)
            verify_packet(packet, manifest)
            review_packet = out / 'review-inputs' / job['case_id']
            review_manifest_path = out / 'review-inputs' / f'{job["case_id"]}-manifest-private.json'
            if not review_packet.exists():
                review_manifest = prepare_review_packet(packet, manifest, review_packet, 12)
                write_json(review_manifest_path, review_manifest)
            review_manifest = read_json(review_manifest_path)
            verify_packet(review_packet, review_manifest)
            write_json(out / 'status.json', {'active': True, 'job': job, 'phase': 'review', 'updated_utc': utc()})
            review_path = out / 'reviews' / job['case_id'] / 'result.json'
            review = read_json(review_path) if review_path.exists() else run_review(out, job, review_packet, review_manifest)
            if review.get('error') or not review.get('completed'):
                raise RuntimeError('Review process failed; retained evidence requires diagnosis before more calls')
            write_json(out / 'status.json', {'active': True, 'job': job, 'phase': 'physical_attempt', 'updated_utc': utc()})
            print(json.dumps({'starting': job}), flush=True)
            result = run_episode(out, job, packet, manifest, review)
            summary = report(out)
            print(json.dumps({'finished': job, 'completed': result['completed'],
                              'native_success': result['native_success'], 'error': result.get('error')}), flush=True)
            if not result['completed']:
                raise RuntimeError('Incomplete episode retained; investigate before continuing scheduled jobs')
        summary = report(out)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        (out / 'driver-error.txt').write_text(traceback.format_exc())
        raise
    finally:
        summary = report(out)
        write_json(out / 'status.json', {'active': False, 'phase': 'error' if error else 'complete',
            'error': error, 'completed_episodes': summary['completed_episodes'], 'updated_utc': utc()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, help='Registration only; defaults to all three conditions')
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT / 'results') or destination == ROOT / 'results':
        raise ValueError('Use a new results/<study> directory')
    if args.action == 'prepare':
        prepare(destination, conditions=args.conditions)
    else:
        if args.conditions is not None:
            raise ValueError('Conditions are frozen at registration')
        {'run': run, 'report': report}[args.action](destination)
