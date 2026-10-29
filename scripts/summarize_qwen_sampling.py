"""Collect exact request/tool timings and outcomes without further generation."""
import argparse
import csv
import json
from pathlib import Path


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def summarize(root):
    rows = read(root / 'study/results.json', [])
    primary = [r for r in rows if r['phase'] == 'primary']
    confirmation = [r for r in rows if r['phase'] == 'confirmation']
    geometry = read(root / 'geometry-private.json', [])
    requests, tool_rows, context_counts = [], [], []
    for phase in ['study', 'tool-assisted', 'tool-assisted-retest']:
        for path in sorted((root / phase).glob('**/call*/timing.json')):
            timing = read(path, {}); usage = timing.get('usage') or {}
            response = read(path.with_name('response.json'), {})
            request = read(path.with_name('request.json'), {})
            budget = timing.get('context_budget') or {}
            parts = path.relative_to(root).parts
            case = parts[2] if phase == 'study' else parts[1]
            context_counts.append({'file': str(path.relative_to(root)),
                'measured': budget.get('prompt_tokens'), 'actual': usage.get('prompt_tokens'),
                'match': budget.get('prompt_tokens') == usage.get('prompt_tokens') if usage.get('prompt_tokens') is not None else None})
            requests.append({'phase': phase, 'case': case, 'call': path.parent.name,
                'finish_reason': timing.get('finish_reason'), 'wall_seconds': timing.get('wall_seconds'),
                'context_admission_seconds': budget.get('tokenization_seconds'),
                'prompt_tokens': usage.get('prompt_tokens'), 'output_tokens': usage.get('completion_tokens'),
                'reasoning_tokens': (usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                'requested_output_tokens': request.get('max_tokens'),
                'temperature': request.get('temperature'), 'presence_penalty': request.get('presence_penalty'),
                'tools': ', '.join(c['function']['name'] for c in response.get('tool_calls', [])),
                'error': timing.get('error'), 'request': str(path.with_name('request.json').relative_to(root)),
                'response': str(path.with_name('response.json').relative_to(root))})
        for path in sorted((root / phase).glob('fresh*/tool-events.jsonl')):
            for line in path.read_text().splitlines():
                event = json.loads(line); result = event['result']
                tool_rows.append({'phase': phase, 'case': path.parent.name, 'turn': event['turn'],
                    'operation': event['call']['function']['name'], 'seconds': event['seconds'],
                    'ok': result.get('ok', True),
                    'error': result.get('error') or (result.get('stderr') if result.get('ok') is False else '')})
    for name, records in [('timing.csv', requests), ('tool-timing.csv', tool_rows)]:
        if records:
            with (root / name).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
    initial = read(root / 'tool-assisted/summary.json', {}).get('cases', [])
    retest = read(root / 'tool-assisted-retest/summary.json', {}).get('cases', [])
    summary = {
        'sampling': {
            'requests': len(rows), 'complete_generations': sum(r['generation_completed'] for r in rows),
            'valid_calls': sum(r['contract_valid'] for r in rows),
            'primary_cases': len(primary),
            'primary_baseline_valid_calls': 0,
            'primary_valid_calls': sum(r['contract_valid'] for r in primary),
            'primary_baseline_seconds': sum(r['baseline_timing']['wall_seconds'] for r in primary),
            'primary_new_seconds': sum(r['timing']['wall_seconds'] for r in primary),
            'primary_baseline_output_tokens': sum(r['baseline_timing']['usage']['completion_tokens'] for r in primary),
            'primary_new_output_tokens': sum(r['timing']['usage']['completion_tokens'] for r in primary),
            'confirmation_old_4k_seconds': sum(r['baseline_timing']['wall_seconds'] for r in confirmation),
            'confirmation_new_32k_seconds': sum(r['timing']['wall_seconds'] for r in confirmation),
            'one_step_model_seconds': sum(r['timing']['wall_seconds'] for r in rows),
            'scope': 'Primary changes only temperature and presence penalty. Confirmation also changes the historical output allowance. Shared-GPU times are descriptive.'},
        'one_step_geometry': {
            'fit_accepted': sum(g['geometry_accepted'] for g in geometry), 'proposals': len(geometry),
            'hinge_errors_mm_for_accepted_fits': [g['native_hinge_line_error_mm'] for g in geometry if g['geometry_accepted']],
            'admitted_fits_using_depth_handle_recovery': sum(bool((g.get('fit_diagnostics') or {}).get('handle_recovery')) for g in geometry),
            'observer_seconds': sum(g.get('geometry_seconds', 0) for g in geometry),
            'interpretation': 'Admitted depth fit does not verify the native hinge or physical success. Recovery is numerical-tool assistance, not model-localization accuracy.'},
        'initial_tool_assisted': initial, 'documented_tool_assisted_retest': retest,
        'total_new_model_calls': len(requests),
        'total_model_seconds': sum(r['wall_seconds'] or 0 for r in requests),
        'total_output_tokens': sum(r['output_tokens'] or 0 for r in requests),
        'output_limit_hits': sum(r['finish_reason'] == 'length' for r in requests),
        'model_request_errors': sum(bool(r['error']) for r in requests),
        'all_requests_allow_at_least_32k': all(r['requested_output_tokens'] >= 32768 for r in requests),
        'context_counts_match': all(r['match'] is True for r in context_counts),
        'tool_seconds': sum(r['seconds'] for r in tool_rows),
        'grasp_planning_seconds': 0, 'ik_seconds': 0, 'arc_planning_seconds': 0, 'physical_execution_seconds': 0,
        'scope': 'Development diagnostics on one asset; no physical episode, self-practice transfer, teacher or learner-memory promotion. Luna remains primary.'}
    write(root / 'summary.json', summary)
    write(root / 'context-count-audit.json', context_counts)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    result = summarize(parser.parse_args().run.resolve())
    print(json.dumps({k: result[k] for k in ['total_new_model_calls', 'total_model_seconds', 'total_output_tokens', 'output_limit_hits', 'context_counts_match']}))
