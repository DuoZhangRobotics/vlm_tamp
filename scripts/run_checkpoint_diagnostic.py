"""Prepare/run isolated subscription comparisons on saved public robot prefixes.

No simulator is launched. No answer or evaluator rubric enters practice memory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_tamp.checkpoint_diagnostic import (
    PROMPT, build_packet, parse_decision, read_json, sha256, verify_packet, write_json,
)
from vlm_tamp.subscription_agent import clean_environment, codex_arguments, verify_isolation

SOURCE = ROOT / 'results/tool-repair-learning-20261021/study-v1/transfer'
MODELS = ['gpt-6-luna', 'gpt-6.1-sol']
CASES = [
    ('state000-with_learned_memory', 10, 'accepted_visual_motion',
     'Pose acceptance and hinge-fit acceptance are different. Recognize accepted measured motion, '
     'without treating it as native task success or guaranteed future retention. Continued fresh '
     'planning or a justified informative check can both be reasonable.'),
    ('state029-with_learned_memory', 10, 'finite_planning_failure',
     'Distinguish finite planning failure from physical execution failure and global infeasibility. '
     'Read the latest failure status and existing measured scene support before choosing a next query.'),
    ('state041-with_learned_memory', 10, 'prediction_with_unresolved_support',
     'Hand prediction is not measured object motion. Sparse visual support does not prove lost '
     'contact or no motion. Do not use the later native success to require continuation or penalize '
     'reasonable investigation. A specific check can be preferable to an unsupported conclusion.'),
    ('state045-with_learned_memory', 11, 'model_identity_change',
     'The latest grounding changes contact radius and axis sign relative to earlier estimates. '
     'Notice model inconsistency when relevant; do not copy contact-anchor components into the '
     'hinge field. No ground-truth hinge or mandatory recovery operation is supplied.'),
    ('state045-with_learned_memory', 12, 'incorrect_model_edit',
     'The preceding edit uses the earlier contact-anchor x value as hinge x. Attribute that '
     'change to the edit, not solely the refit. Correcting the edit alone does not validate the '
     'newly grounded contact geometry. Alternative evidence-seeking next actions are allowed.'),
    ('state045-with_learned_memory', 46, 'strong_depth_disagreement',
     'The scene update is a contact-retention prediction with rejected visual fitting and '
     '100 percent contact-cloud free-space conflict. Do not assert observed 20-degree object '
     'motion or definite lost contact. Choose a concrete investigation or correction addressing '
     'the mismatch; proposing another arc solely because the previous plan executed is insufficient.'),
]
RUBRIC = {
    'visibility': 'Evaluator only; never copied to an acting-model workspace.',
    'grading': 'Model identities masked for qualitative review; reviewer supplies evidence and uncertainty.',
    'dimensions': {
        'tool_contract': '0: invalid/stale request; 1: plausible but incomplete or unsupported arguments; '
                         '2: documented operation with suitable arguments. Syntactic validity is not plan feasibility.',
        'evidence_fidelity': '0: material unsupported or contradicted claim; 1: largely calibrated but '
                             'misses an important distinction; 2: accurate, scoped use of accessible evidence.',
        'next_test_value': '0: unjustified repetition or unrelated action; 1: relevant but generic or '
                           'redundant; 2: concrete next query/action addresses current uncertainty or supported progress.',
        'prediction_and_limits': '0: fabricated certainty/outcome; 1: vague expectation or limitations; '
                                 '2: checkable expected observation and material remaining uncertainty.',
    },
    'grading_rules': [
        'Do not reward agreement with one preferred action sequence.',
        'Use only evidence accessible before the cutoff to judge the response.',
        'Evaluator knowledge of later motion or eventual success cannot count as actor knowledge.',
        'Missing observations can make the decision unresolved; report them rather than inventing an answer.',
        'Tool-semantic and numerical claims can be checked mechanically; action utility needs qualified review.',
        'Two samples at one checkpoint are repeated responses, not independent physical episodes.',
        'Offline decision quality cannot establish physical recovery or learning transfer.',
    ],
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    randomizer = random.Random(20261021)
    identifiers = randomizer.sample(range(100000, 999999), len(CASES))
    records = []
    for ident, (source, cutoff, category, expectation) in zip(identifiers, CASES):
        key = f'c{ident}'
        manifest = build_packet(SOURCE / source, cutoff, out / 'packets' / key)
        manifest.update(case_id=key, private_category=category, private_expectation=expectation)
        manifest['packet_verification'] = verify_packet(out / 'packets' / key, manifest)
        records.append(manifest)
        print(json.dumps({'prepared': key, 'files': len(manifest['packet_sha256']),
                          'missing_historical_assets': len(manifest['missing_assets'])}), flush=True)
    write_json(out / 'cases-private.json', records)
    write_json(out / 'rubric-private.json', RUBRIC)
    (out / 'prompt.txt').write_text(PROMPT)
    jobs = [{'case_id': row['case_id'], 'model': model, 'replicate': repeat}
            for row in records for model in MODELS for repeat in [1, 2]]
    randomizer.shuffle(jobs)
    for number, row in enumerate(jobs, 1):
        row['run_id'] = f'run{number:03d}'
    write_json(out / 'schedule-private.json', jobs)
    files = [Path(__file__), ROOT / 'src/vlm_tamp/checkpoint_diagnostic.py',
             ROOT / 'src/vlm_tamp/subscription_agent.py', ROOT / 'tests/test_checkpoint_diagnostic.py']
    source_hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        source_hashes[str(relative)] = sha256(path)
        dest = out / 'source-snapshot' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    write_json(out / 'protocol.json', {
        'created_utc': utc(), 'scope': 'Development decision diagnostic on recorded Luna-generated prefixes.',
        'models': MODELS, 'reasoning_effort': 'high', 'authentication': 'ChatGPT subscription',
        'separately_billed_api_calls': False, 'checkpoints': len(records),
        'replicates_per_model_checkpoint': 2, 'planned_responses': len(jobs),
        'seconds_per_response': 600, 'maximum_concurrent_responses': 2,
        'automatic_retries': 0, 'physical_robot_operations': 0,
        'memory': 'Exact previously evaluated memory in every packet; no evaluation-derived lesson added.',
        'images': 'Four current original views attached in identical order; retained earlier views accessible.',
        'history': 'Original public requests and responses through cutoff; fresh context, no original private reasoning.',
        'missing_inputs': 'Unavailable historical arrays listed per packet. All within-pair inputs identical.',
        'selection': 'Purposeful development selection includes successful and unsuccessful source runs; not held out.',
        'shared_source_states': 4, 'three_checkpoints_share_state45_episode': True,
        'source_sha256': source_hashes,
        'preregistered_files_sha256': {name: sha256(out / name) for name in
            ['cases-private.json', 'rubric-private.json', 'schedule-private.json', 'prompt.txt']},
    })
    return records


def run_one(out, job, case, forbidden, *, prompt_path=None):
    folder = out / 'responses' / job['run_id']
    folder.mkdir(parents=True, exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix='vlm-checkpoint-'))
    record = {**job, 'started_utc': utc(), 'completed': False, 'valid_decision': False}
    prompt_path = Path(prompt_path) if prompt_path is not None else out / 'prompt.txt'
    prompt = prompt_path.read_text()
    record['prompt_sha256'] = sha256(prompt_path)
    start = time.monotonic()
    try:
        shutil.copytree(out / 'packets' / case['case_id'], work / 'inputs')
        readonly = [work / 'inputs']
        link = work / 'private-canary-link'
        link.symlink_to(out / 'rubric-private.json')
        isolation = verify_isolation(work, [*forbidden, link], readonly_paths=readonly)
        link.unlink()
        write_json(folder / 'isolation.json', isolation)
        record['isolation_passed'] = isolation['passed']
        record['input_verification_before'] = verify_packet(work / 'inputs', case)
        args = codex_arguments(work, folder / 'answer.txt', model=job['model'], readonly_paths=readonly)
        images = []
        for relative in case['current_images']:
            images += ['--image', str(work / 'inputs' / relative)]
        index = args.index('-o')
        args[index:index] = images
        write_json(folder / 'command.json', args)
        with (folder / 'events.jsonl').open('w') as log, (folder / 'stderr.log').open('w') as err:
            process = subprocess.Popen(args, stdin=subprocess.PIPE, text=True, stdout=log,
                stderr=err, cwd=work, env=clean_environment(), start_new_session=True)
            try:
                process.communicate(prompt, timeout=600)
            except subprocess.TimeoutExpired:
                record['timed_out'] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
        record['returncode'] = process.returncode
        record['input_verification_after'] = verify_packet(work / 'inputs', case)
        if (work / 'decision.json').exists():
            shutil.copyfile(work / 'decision.json', folder / 'decision.json')
            try:
                decision = parse_decision(folder / 'decision.json')
                record.update(valid_decision=True, next_operation=decision['next_request']['operation'])
            except (ValueError, TypeError) as error:
                record['decision_error'] = str(error)
        else:
            record['decision_error'] = 'Requested decision.json was not produced'
        for path in work.iterdir():
            if path.name not in {'inputs', 'decision.json', 'allowed-canary.txt', '.analysis_site'}:
                dest = folder / 'workspace-output' / path.name
                dest.parent.mkdir(exist_ok=True)
                if path.is_dir():
                    shutil.copytree(path, dest, symlinks=True)
                else:
                    shutil.copy2(path, dest, follow_symlinks=False)
        # CLI stdout contains usage and thread identity, not just answer text.
        for line in (folder / 'events.jsonl').read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') == 'thread.started':
                record['thread_id'] = event.get('thread_id')
            if event.get('type') == 'turn.completed':
                record['usage'] = event.get('usage')
        record['completed'] = process.returncode == 0
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    finally:
        record['wall_seconds'] = time.monotonic() - start
        # Preserve any abnormal input changes; normal copies are redundant.
        if (work / 'inputs').exists() and not record.get('input_verification_after', {}).get('passed'):
            try:
                verify_packet(work / 'inputs', case)
            except ValueError:
                shutil.copytree(work / 'inputs', folder / 'changed-inputs')
        shutil.rmtree(work)
        record['temporary_workspace_removed'] = not work.exists()
        write_json(folder / 'result.json', record)
    return record


def run(out):
    protocol = read_json(out / 'protocol.json')
    for name, digest in protocol['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Registered source changed: ' + name)
    for name, digest in protocol['preregistered_files_sha256'].items():
        if sha256(out / name) != digest:
            raise ValueError('Registered inputs changed: ' + name)
    cases = {c['case_id']: c for c in read_json(out / 'cases-private.json')}
    for key, case in cases.items():
        verify_packet(out / 'packets' / key, case)
    status = subprocess.run(['codex', 'login', 'status'], capture_output=True, text=True,
                            env=clean_environment(), check=True)
    if 'Logged in using ChatGPT' not in status.stdout + status.stderr:
        raise RuntimeError('Subscription authentication required; no API fallback')
    write_json(out / 'authentication-check.json', {
        'chatgpt_subscription': True, 'api_key_fallback': False,
        'client_version': subprocess.check_output(['codex', '--version'], text=True).strip(),
    })
    jobs = read_json(out / 'schedule-private.json')
    # Existing responses are never overwritten or retried. Resuming only runs
    # registered jobs that have not started; partial outputs remain visible.
    existing, pending = [], []
    for job in jobs:
        folder = out / 'responses' / job['run_id']
        if (folder / 'result.json').exists():
            existing.append(read_json(folder / 'result.json'))
        elif folder.exists():
            raise RuntimeError('Partial run needs diagnosis; refusing automatic retry: ' + job['run_id'])
        else:
            pending.append(job)
    forbidden = [out / 'rubric-private.json', out / 'cases-private.json',
                 SOURCE / 'state045-with_learned_memory/agent-workspace/findings.txt',
                 SOURCE / 'state045-with_learned_memory/attempt01/initial/actions.json',
                 ROOT / 'docs/recovery_pipeline_review_20261021.md',
                 Path.home() / '.codex/auth.json']
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_one, out, job, cases[job['case_id']], forbidden) for job in pending]
        for future in as_completed(futures):
            record = future.result()
            existing.append(record)
            write_json(out / 'results.json', {'updated_utc': utc(), 'planned': len(jobs),
                'completed_records': len(existing), 'responses': sorted(existing, key=lambda x: x['run_id'])})
            print(json.dumps({k: record.get(k) for k in
                ['run_id', 'model', 'case_id', 'completed', 'valid_decision', 'wall_seconds', 'error']}), flush=True)
    print(json.dumps({'finished': len(existing), 'planned': len(jobs)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    if not out.is_relative_to(ROOT / 'results') or out == ROOT / 'results':
        raise ValueError('Use a dedicated results subdirectory')
    if not args.prepare and not args.run:
        parser.error('Choose --prepare and/or --run')
    if args.prepare:
        prepare(out)
    if args.run:
        run(out)


if __name__ == '__main__':
    main()
