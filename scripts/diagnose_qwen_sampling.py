"""Bounded, registered sampling diagnostic; no robot motion or learning updates.

Primary requests replay the three retained 32k failures exactly except for
temperature and presence penalty. A predeclared gate controls confirmation
requests or a small two-factor follow-up on the first failed case.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_tamp.local_agent import LocalChatClient, parse_call, write_json
from vlm_tamp.robot_contracts import validate_operation

PRIMARY = ('vision1-r1', 'vision2-r2', 'vision3-r2')
CONFIRM = ('vision1-r2', 'vision2-r1', 'vision3-r1')
OLD_IO = ROOT / 'results/robot-io-contract-20261024/qwen-retest'
OLD_32K = ROOT / 'results/qwen32k-budget-20261025/study'
SOURCE_NAMES = ('scripts/diagnose_qwen_sampling.py', 'src/vlm_tamp/local_agent.py',
                'src/vlm_tamp/local_context_budget.py', 'src/vlm_tamp/robot_contracts.py',
                'scripts/serve_local_qwen.py')


def read(path):
    return json.loads(Path(path).read_text())


def repetition(text):
    paragraphs = [re.sub(r'\s+', ' ', x).strip().lower() for x in re.split(r'\n\s*\n', text)]
    paragraphs = [p for p in paragraphs if len(p) >= 80]
    seen, repeated, total, occurrences = set(), 0, 0, 0
    for paragraph in paragraphs:
        total += len(paragraph)
        if paragraph in seen:
            repeated += len(paragraph)
            occurrences += 1
        seen.add(paragraph)
    return {'long_paragraphs': len(paragraphs), 'repeated_paragraphs': occurrences,
            'repeated_character_fraction': repeated / total if total else 0.,
            'definition': 'Exact repeated occurrences of normalized paragraphs >=80 characters; lowercase and collapse whitespace. Not a measure of hidden cognition.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8127/v1')
    args = parser.parse_args()
    root = args.run.resolve()
    out = root / 'study'
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(OLD_IO / 'public', out / 'public')
    info = LocalChatClient(args.endpoint, 'qwen9b', out / 'preflight').server_info()
    old_rows = {r['id']: r for r in read(OLD_IO / 'results.json')}
    protocol = {
        'model': 'Qwen/Qwen3.5-9B', 'checkpoint_revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        'dtype': 'bfloat16', 'server_info': info, 'output_tokens': 32768, 'request_seconds': 900,
        'recommended_sampling': {'temperature': 1.0, 'top_p': .95, 'top_k': 20, 'presence_penalty': 1.5},
        'unchanged_effective_defaults': {'min_p': 0., 'repetition_penalty': 1.},
        'recommendation_source': 'https://huggingface.co/Qwen/Qwen3.5-9B#best-practices',
        'pinned_readme_sha256': hashlib.sha256((ROOT / '.runtime/models/Qwen3.5-9B/README.md').read_bytes()).hexdigest(),
        'primary_cases': list(PRIMARY), 'primary_baseline': str(OLD_32K.relative_to(ROOT)),
        'primary_changed_request_fields': ['presence_penalty', 'temperature'],
        'primary_unchanged': 'Saved messages, tools, image bytes, seeds, thinking=true, output allowance, model checkpoint, and server settings.',
        'baseline_is_retained_not_concurrent': True,
        'latency_scope': 'Descriptive wall time on a shared GPU, not an isolated throughput comparison.',
        'confirmation_cases': list(CONFIRM),
        'confirmation_gate': 'All 3 primary generations finish without truncation/error and at least 2 yield valid ground calls.',
        'confirmation_baseline': 'Historical 4k completed calls; confirmation changes sampling AND output allowance. Not included in the 32k paired comparison.',
        'factor_followup_if_primary_gate_fails': 'First failed primary case in fixed order, two extra requests: temperature=.6/presence=1.5 and temperature=1/presence=0. Saved .6/0 and new 1/1.5 provide other corners. No retry or best-of selection.',
        'tool_assisted_gate': 'All 6 primary+confirmation generations finish, and at least 5 have valid ground calls. This is a small-sample feasibility gate, not a reliability estimate.',
        'tool_assisted_plan': {'fresh_to_qwen': ['state020-r1-luna', 'state035-r1-luna'],
            'scope': 'Previously recorded same-asset observations, not exposed in prior Qwen probes; not new assets or a held-out transfer benchmark.',
            'cases': 2, 'maximum_model_turns_each': 8, 'maximum_ground_calls_each': 3,
            'seconds_each': 1200, 'output_tokens_per_request': 32768,
            'tools': 'Sandboxed Python, original images/crops, public files, and truthful numerical ground feedback. Native geometry/evaluator answers withheld.'},
        'geometry_evaluation': 'Observer-only saved RGB-D fit acceptance and native hinge-line distance, never fed to the one-step model requests.',
        'physical_motion': False, 'memory_updates': False, 'reflection_reviewer': False,
        'automatic_retries': 0, 'sources_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_NAMES},
        'image_sha256': {str(p.relative_to(out / 'public')): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out / 'public').rglob('*.png'))},
    }
    write_json(root / 'protocol.json', protocol)
    for name in SOURCE_NAMES:
        dest = out / 'source-snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    rows = []

    def attempt(ident, phase, temperature=1., penalty=1.5, suffix=''):
        source = OLD_32K if ident in PRIMARY else OLD_IO
        old_call = source / 'responses' / ident / 'call001'
        before, old_timing = read(old_call / 'request.json'), read(old_call / 'timing.json')
        sample_id = ident + suffix
        folder = out / 'responses' / sample_id
        client = LocalChatClient(args.endpoint, 'qwen9b', folder, seed=before['seed'],
            temperature=temperature, presence_penalty=penalty)
        row = {'id': sample_id, 'case_id': ident, 'phase': phase,
            'source_private': old_rows[ident]['source_private'], 'generation_completed': False,
            'contract_valid': False, 'baseline_call': str(old_call.relative_to(ROOT)),
            'baseline_timing': old_timing, 'sampling': {'temperature': temperature, 'presence_penalty': penalty}}
        write_json(out / 'status.json', {'phase': phase, 'current_case': sample_id,
            'completed': len(rows), 'current_call': f'study/responses/{sample_id}/call001'})
        try:
            response, timing = client.complete(before['messages'], before['tools'], out / 'public')
            row['timing'] = timing
            row['generation_completed'] = timing['finish_reason'] in {'stop', 'tool_calls'}
            row['repetition'] = repetition(response.get('reasoning', response.get('reasoning_content', '')))
            calls = response.get('tool_calls', [])
            if timing['finish_reason'] == 'length':
                row['failure'] = 'output_token_limit'
            elif not row['generation_completed']:
                row['failure'] = 'unexpected_finish_reason'
            elif len(calls) != 1:
                row['failure'] = 'missing_or_multiple_ground_calls'
            else:
                name, payload = parse_call(calls[0], before['tools'])
                if name != 'ground':
                    raise ValueError('Expected one ground call')
                validate_operation(name, payload, {'frontview': {'width': 512, 'height': 512},
                                                  'agentview': {'width': 512, 'height': 512}})
                row.update(contract_valid=True, operation=name, arguments=payload)
        except Exception as error:
            row.update(error=f'{type(error).__name__}: {error}', timing=client.calls[-1] if client.calls else None)
            if hasattr(error, 'request_diagnostic'):
                row['request_diagnostic'] = error.request_diagnostic
        actual = read(folder / 'call001/request.json')
        expected = {**before, 'max_tokens': 32768, 'temperature': temperature, 'presence_penalty': penalty}
        changes = sorted(k for k in before.keys() | actual.keys() if before.get(k) != actual.get(k))
        row['input_equivalence'] = {'changed_fields': changes, 'matches_registered_request': actual == expected}
        prompt_count = (row.get('timing') or {}).get('context_budget', {}).get('prompt_tokens')
        row['input_equivalence']['prompt_tokens'] = prompt_count
        row['input_equivalence']['baseline_prompt_tokens'] = (old_timing.get('usage') or {}).get('prompt_tokens')
        if actual != expected:
            write_json(folder / 'result.json', row)
            raise RuntimeError('Unexpected request mutation; experiment stopped')
        rows.append(row)
        write_json(folder / 'result.json', row)
        write_json(out / 'results.json', rows)
        print(json.dumps({k: row.get(k) for k in ['id', 'phase', 'generation_completed', 'contract_valid', 'failure', 'error']} |
                         {'seconds': (row.get('timing') or {}).get('wall_seconds'),
                          'usage': (row.get('timing') or {}).get('usage')}), flush=True)
        return row

    primary = [attempt(ident, 'primary') for ident in PRIMARY]
    primary_gate = all(r['generation_completed'] for r in primary) and sum(r['contract_valid'] for r in primary) >= 2
    if primary_gate:
        confirm = [attempt(ident, 'confirmation') for ident in CONFIRM]
        gate = all(r['generation_completed'] for r in primary + confirm) and sum(r['contract_valid'] for r in primary + confirm) >= 5
    else:
        ident = next((r['case_id'] for r in primary if not r['generation_completed'] or not r['contract_valid']), PRIMARY[0])
        attempt(ident, 'factor_followup', .6, 1.5, '-temperature06-penalty15')
        attempt(ident, 'factor_followup', 1., 0., '-temperature10-penalty0')
        gate = False
    source_ok = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in protocol['sources_sha256'].items())
    image_ok = all(hashlib.sha256((out / 'public' / p).read_bytes()).hexdigest() == h for p, h in protocol['image_sha256'].items())
    summary = {'requests': len(rows), 'primary_completed': sum(r['generation_completed'] for r in primary),
        'primary_contract_valid': sum(r['contract_valid'] for r in primary), 'primary_gate_passed': primary_gate,
        'tool_assisted_gate_passed': gate, 'output_limit_hits': sum(r.get('failure') == 'output_token_limit' for r in rows),
        'model_seconds': sum((r.get('timing') or {}).get('wall_seconds', 0) for r in rows),
        'output_tokens': sum(((r.get('timing') or {}).get('usage') or {}).get('completion_tokens', 0) for r in rows),
        'source_hashes_match': source_ok, 'image_hashes_match': image_ok,
        'scope': 'Offline one-step diagnostics; no physical success, practice learning, or matched Luna comparison established.'}
    write_json(out / 'summary.json', summary)
    write_json(out / 'status.json', {'phase': 'complete', 'completed': len(rows), 'tool_assisted_gate_passed': gate})
    print(json.dumps(summary), flush=True)
    if not source_ok or not image_ok:
        raise SystemExit('Input/source integrity check failed')


if __name__ == '__main__':
    main()
