"""Luna-only generic-verification ablation with separately registered live trials."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from vlm_tamp.checkpoint_diagnostic import PROMPT, build_packet, read_json, sha256, verify_packet, write_json
from vlm_tamp.subscription_agent import clean_environment
from vlm_tamp.storage_budget import require_storage_headroom
from run_checkpoint_diagnostic import RUBRIC, run_one, utc
from run_subscription_dino_study import retained_inputs

OLD = ROOT / 'results/tool-repair-learning-20261021'
SOURCE = OLD / 'study-v1/transfer'
MEMORY = ROOT / 'data/agent_memory/luna-scene-motion-v1/guidelines.json'
MEMORY_SHA = '65f9612d3f66380234d78f3e05c6342f784b3f137ac4df8032f7fb04ba0c13c2'
CONDITIONS = ('baseline', 'verification')
CASES = [
    ('state045-with_learned_memory', 11, 'calibration', 'changed_contact_identity',
     'Same known calibration case: automatic recovery selected another object part. Check '
     'contact identity, radius and axis consistency without requiring one specific next action.'),
    ('state045-with_learned_memory', 12, 'calibration', 'incorrect_model_edit',
     'Same known calibration case: the edit used an earlier contact-anchor component as hinge '
     'x. Trace the edit accurately; correcting one field does not validate the new contact.'),
    ('state000-without_learned_memory', 19, 'additional_development', 'recovery_refit',
     'A new view yields a fit after release, surface selection and failed grounding. Compare '
     'its identity with the current image and history; do not assume a retained grasp merely '
     'because grounding succeeded. Do not declare the fit wrong solely because its coordinates changed.'),
    ('state029-without_learned_memory', 8, 'additional_development', 'accepted_pose_unstable_pivot',
     'Pose motion is accepted but the pivot fit is not. Contact-cloud support is relatively '
     'strong. Recognize what was measured, keep grasp/hinge uncertainty, and avoid treating '
     'every rejected hinge fit as rejection of the measured rigid motion.'),
    ('state041-without_learned_memory', 25, 'additional_development', 'rejected_pivot_used_as_hypothesis',
     'The edit uses XY from event 23, whose pose was accepted but hinge estimate rejected as '
     'unstable or inconsistent. It remains a model hypothesis, not an accepted measured pivot. '
     'The retained Z does not by itself change an upright hinge line. Use current evidence '
     'without assuming either the original or edited model must be correct.'),
    ('state045-without_learned_memory', 19, 'additional_development', 'pose_transport_without_pivot_update',
     'apply_tracking transports the accepted measured pose while leaving the pivot unchanged. '
     'Do not describe it as successful hinge refinement or treat the rejected pivot fit as '
     'proof that pose transport failed. Evaluate contact and current geometry separately.'),
]
PHYSICAL = [
    {'case_id': 'state045-baseline', 'state': 45, 'condition': 'baseline'},
    {'case_id': 'state045-verification', 'state': 45, 'condition': 'verification'},
    {'case_id': 'state000-verification', 'state': 0, 'condition': 'verification'},
    {'case_id': 'state000-baseline', 'state': 0, 'condition': 'baseline'},
]


def augment(base, verification):
    return base.rstrip() + '\n\n' + verification.strip() + '\n'


def auth_check():
    result = subprocess.run(['codex', 'login', 'status'], env=clean_environment(),
                            capture_output=True, text=True, check=True)
    if 'Logged in using ChatGPT' not in result.stdout + result.stderr:
        raise RuntimeError('ChatGPT subscription required; no API fallback')
    return {'chatgpt_subscription': True, 'api_key_fallback': False,
            'client_version': subprocess.check_output(['codex', '--version'], text=True).strip()}


def prepare(out):
    require_storage_headroom(ROOT)
    if sha256(MEMORY) != MEMORY_SHA:
        raise ValueError('Evaluated memory changed')
    # Preserve the numerical baseline, including its documented report-only fix.
    old_freeze = read_json(OLD / 'tool-freeze.json')['source_sha256']
    exception = read_json(OLD / 'reporting-fix/compatibility.json')['files']
    numerical = {}
    for name, digest in old_freeze.items():
        expected = exception[name]['after_sha256'] if name in exception else digest
        if sha256(ROOT / name) != expected:
            raise ValueError('Numerical baseline drift: ' + name)
        numerical[name] = expected
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs'
    inputs.mkdir()
    shutil.copyfile(MEMORY, inputs / 'guidelines.json')
    verification = (ROOT / 'prompts/generic_result_verification.txt').read_text()
    physical_base = (ROOT / 'prompts/recovery_learning_transfer.txt').read_text()
    (inputs / 'offline-baseline.txt').write_text(PROMPT)
    (inputs / 'offline-verification.txt').write_text(augment(PROMPT, verification))
    (inputs / 'physical-baseline.txt').write_text(physical_base)
    (inputs / 'physical-verification.txt').write_text(augment(physical_base, verification))
    (inputs / 'verification-only.txt').write_text(verification)
    randomizer = random.Random(20261022)
    identifiers = randomizer.sample(range(100000, 999999), len(CASES))
    cases = []
    for ident, (name, cutoff, subset, category, expectation) in zip(identifiers, CASES):
        key = f'c{ident}'
        manifest = build_packet(SOURCE / name, cutoff, out / 'packets' / key,
            shared_sources={'experience/guidelines.json': inputs / 'guidelines.json'})
        manifest.update(case_id=key, subset=subset, private_category=category,
                        private_expectation=expectation)
        manifest['packet_verification'] = verify_packet(out / 'packets' / key, manifest)
        cases.append(manifest)
        print(json.dumps({'prepared': key, 'subset': subset,
                          'missing_assets': len(manifest['missing_assets'])}), flush=True)
    write_json(out / 'cases-private.json', cases)
    write_json(out / 'rubric-private.json', {**RUBRIC,
        'additional_rules': [
            'Condition labels are masked during qualitative grading; same rubric in both conditions.',
            'Report the two reused calibration checkpoints separately from four additional development checkpoints.',
            'Do not score verbosity, repetition of the added prompt, or blanket re-grounding as competence.',
            'The added scaffold is human-authored process guidance, not an agent-learned memory.',
        ]})
    jobs = [{'case_id': c['case_id'], 'model': 'gpt-6-luna', 'condition': condition,
             'replicate': repeat, 'subset': c['subset']}
            for c in cases for condition in CONDITIONS for repeat in [1, 2]]
    randomizer.shuffle(jobs)
    for i, job in enumerate(jobs, 1):
        job['run_id'] = f'run{i:03d}'
    write_json(out / 'schedule-private.json', jobs)
    write_json(out / 'physical-schedule-private.json', PHYSICAL)
    sources = [Path(__file__), ROOT / 'scripts/run_checkpoint_diagnostic.py',
        ROOT / 'scripts/report_checkpoint_diagnostic.py', ROOT / 'src/vlm_tamp/checkpoint_diagnostic.py',
        ROOT / 'tests/test_checkpoint_diagnostic.py', ROOT / 'tests/test_verification_prompt.py',
        ROOT / 'prompts/generic_result_verification.txt', OLD / 'retain_failure_inputs.py']
    hashes = dict(numerical)
    for path in sources:
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    for name in hashes:
        target = out / 'source-snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    registered = ['cases-private.json', 'rubric-private.json', 'schedule-private.json',
                  'physical-schedule-private.json', *[str(p.relative_to(out)) for p in inputs.iterdir()]]
    write_json(out / 'protocol.json', {
        'created_utc': utc(), 'model': 'gpt-6-luna', 'reasoning_effort': 'high',
        'authentication': 'ChatGPT subscription', 'separately_billed_api_calls': False,
        'conditions': list(CONDITIONS), 'treatment': 'Only the appended generic verification paragraph.',
        'human_authored_scaffold': True, 'new_agent_memory_promoted': False,
        'memory_sha256': MEMORY_SHA,
        'memory_scope': 'Identical previously evaluated memory in both conditions, including counterfactual '
                        'continuations of prefixes originally generated without that memory.',
        'offline': {'checkpoints': 6, 'calibration_checkpoints': 2,
            'additional_development_checkpoints': 4, 'responses': 24,
            'replicates_per_condition_checkpoint': 2, 'seconds_per_response': 600,
            'max_concurrent': 2, 'fresh_contexts': True, 'automatic_retries': 0,
            'no_physical_operations': True, 'purposefully_selected_development_not_held_out': True},
        'physical': {'episodes': 4, 'states': [45, 0], 'development_states_previously_seen': True,
            'episodes_per_condition_state': 1, 'seconds_per_episode': 1800, 'control_step_cap': 600,
            'seed': 0, 'fresh_context_and_reset_each_episode': True,
            'stop_immediately_on_native_success': True, 'no_post_success_actions': True,
            'execution_time_vetoes': False, 'optional_dino_tools': False,
            'numerical_tools_fixed': True, 'order': PHYSICAL,
            'selection_frozen_before_offline_responses': True,
            'primary_outcome': 'Native task success; all four scheduled episodes stay in the denominator.',
            'secondary_outcomes': ['steps and wall time', 'recovery and model-change trace',
                                   'planning failures', 'observation/model-revision call counts']},
        'source_sha256': hashes, 'numerical_source_sha256': numerical,
        'registered_files_sha256': {name: sha256(out / name) for name in registered},
    })
    write_json(out / 'status.json', {'phase': 'prepared', 'active': False,
                                   'offline_completed': 0, 'physical_completed': 0})


def check_freeze(out):
    protocol = read_json(out / 'protocol.json')
    for name, digest in protocol['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Registered source changed: ' + name)
    for name, digest in protocol['registered_files_sha256'].items():
        if sha256(out / name) != digest:
            raise ValueError('Registered input changed: ' + name)
    for case in read_json(out / 'cases-private.json'):
        verify_packet(out / 'packets' / case['case_id'], case)


def run_offline(out):
    check_freeze(out)
    write_json(out / 'authentication-check.json', auth_check())
    cases = {c['case_id']: c for c in read_json(out / 'cases-private.json')}
    pending, rows = [], []
    for job in read_json(out / 'schedule-private.json'):
        folder = out / 'responses' / job['run_id']
        if (folder / 'result.json').exists():
            rows.append(read_json(folder / 'result.json'))
        elif folder.exists():
            raise RuntimeError('Partial request retained; no automatic retry: ' + job['run_id'])
        else:
            pending.append(job)
    forbidden = [out / 'rubric-private.json', out / 'cases-private.json',
        ROOT / 'docs/checkpoint_diagnostic_20261021.md', ROOT / 'docs/recovery_pipeline_review_20261021.md',
        ROOT / 'results/checkpoint-diagnostic-20261021/scored-results-private.json',
        SOURCE / 'state045-with_learned_memory/attempt01/initial/actions.json',
        Path.home() / '.codex/auth.json']
    write_json(out / 'status.json', {'phase': 'offline', 'active': True,
                                   'offline_completed': len(rows), 'physical_completed': 0})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_one, out, job, cases[job['case_id']], forbidden,
                    prompt_path=out / 'inputs' / f'offline-{job["condition"]}.txt') for job in pending]
        for future in as_completed(futures):
            record = future.result()
            rows.append(record)
            write_json(out / 'results.json', {'planned': 24, 'completed_records': len(rows),
                'responses': sorted(rows, key=lambda row: row['run_id'])})
            write_json(out / 'status.json', {'phase': 'offline', 'active': True,
                'offline_completed': len(rows), 'physical_completed': 0})
            print(json.dumps({k: record.get(k) for k in ['run_id', 'condition', 'case_id',
                'completed', 'valid_decision', 'wall_seconds', 'error']}), flush=True)
    write_json(out / 'status.json', {'phase': 'offline_complete', 'active': False,
                                   'offline_completed': len(rows), 'physical_completed': 0})


def load_retainer():
    path = OLD / 'retain_failure_inputs.py'
    spec = importlib.util.spec_from_file_location('failure_retainer_for_verification', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.retain


def run_physical(out):
    check_freeze(out)
    auth_check()
    offline = read_json(out / 'results.json')
    if offline.get('completed_records') != 24:
        raise ValueError('Complete the registered offline block first')
    (out / 'physical').mkdir(exist_ok=True)
    retain = load_retainer()
    rows = []
    for job in read_json(out / 'physical-schedule-private.json'):
        folder = out / 'physical' / job['case_id']
        result_path = out / f'{job["case_id"]}-result.json'
        if result_path.exists():
            rows.append(read_json(result_path))
            continue
        if folder.exists():
            raise RuntimeError('Partial episode retained; diagnose without automatic reset: ' + job['case_id'])
        check_freeze(out)
        require_storage_headroom(ROOT)
        prompt = out / 'inputs' / f'physical-{job["condition"]}.txt'
        command = [sys.executable, str(ROOT / 'scripts/run_subscription_agent_pilot.py'),
            '--output', str(folder), '--state', str(job['state']), '--scene-variant', 'nominal',
            '--prompt-file', str(prompt), '--agent-seconds', '1800', '--fit-diagnostics',
            '--abstract-experience', '--stop-on-success', '--persistent-episode',
            '--surface-contact-tools', '--attempts', '1',
            '--guidelines-source', str(out / 'inputs/guidelines.json')]
        write_json(out / f'{job["case_id"]}-command.json', command)
        write_json(out / 'status.json', {'phase': 'physical', 'active': True,
            'current_case': job['case_id'], 'offline_completed': 24, 'physical_completed': len(rows)})
        start = time.monotonic()
        with (out / f'{job["case_id"]}-launcher.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            write_json(out / 'physical-process.json', {'pid': process.pid, 'case': job['case_id']})
            while process.poll() is None:
                retain(folder)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            retain(folder)
        episodes = read_json(folder / 'evaluation-private.json').get('attempts', []) if (
            folder / 'evaluation-private.json').exists() else []
        exit_record = read_json(folder / 'agent-exit.json') if (folder / 'agent-exit.json').exists() else {}
        record = {**job, 'returncode': process.returncode, 'wall_seconds': time.monotonic() - start,
            'attempts': episodes, 'agent_exit': exit_record, 'prompt_sha256': sha256(prompt),
            'memory_sha256': sha256(out / 'inputs/guidelines.json'),
            'native_success': bool(episodes and episodes[0]['official_task_success']),
            'normal_terminal': bool(exit_record.get('normal_terminal')),
            'source_freeze_matches_after': True}
        check_freeze(out)
        if (folder / 'agent-workspace').exists():
            record['retention'] = retained_inputs(folder)
        write_json(result_path, record)
        rows.append(record)
        write_json(out / 'physical-results.json', {'planned': 4, 'completed_records': len(rows),
                                                  'episodes': rows})
        print(json.dumps({k: record.get(k) for k in ['case_id', 'condition', 'native_success',
              'normal_terminal', 'returncode', 'wall_seconds']}), flush=True)
    write_json(out / 'status.json', {'phase': 'collection_complete', 'active': False,
                                   'offline_completed': 24, 'physical_completed': len(rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--physical', action='store_true')
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    if not out.is_relative_to(ROOT / 'results') or out == ROOT / 'results':
        raise ValueError('Use a dedicated results directory')
    if not any([args.prepare, args.offline, args.physical]):
        parser.error('Choose --prepare, --offline and/or --physical')
    if args.prepare:
        prepare(out)
    if args.offline:
        run_offline(out)
    if args.physical:
        run_physical(out)
