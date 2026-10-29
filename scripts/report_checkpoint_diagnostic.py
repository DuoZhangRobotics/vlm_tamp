"""Mask model identities and check request contracts for an offline diagnostic."""
import argparse
import ast
import inspect
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_tamp.checkpoint_diagnostic import read_json, write_json, sha256

METHODS = {
    'observe': 'capture', 'episode_status': 'episode_status', 'ground': 'ground',
    'inspect_scene': 'inspect_scene', 'start_tracking': 'start_tracking',
    'measure_tracking': 'measure_tracking', 'apply_tracking': 'apply_tracking',
    'set_model': 'set_model', 'plan_grasp': 'plan_grasp', 'plan_arc': 'plan_arc',
    'plan_joint': 'plan_joint', 'plan_sequence': 'plan_sequence',
    'execute_sequence': 'execute_sequence', 'ik': 'ik_solutions',
    'execute': 'execute_plan', 'gripper': 'gripper',
    'select_surface_contact': 'select_surface_contact', 'plan_pose': 'plan_pose',
}


def signatures():
    tree = ast.parse((ROOT / 'scripts/serve_subscription_robot.py').read_text())
    service = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RobotService')
    result = {}
    for function in service.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        params = []
        args = function.args
        regular = args.posonlyargs + args.args
        defaults = [None] * (len(regular) - len(args.defaults)) + list(args.defaults)
        for arg, default in zip(regular, defaults):
            if arg.arg != 'self':
                params.append(inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=inspect.Parameter.empty if default is None else '<default>'))
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            params.append(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if default is None else '<default>'))
        result[function.name] = inspect.Signature(params)
    return result


def contract_check(decision, history, method_signatures):
    request = decision['next_request']
    operation, arguments = request['operation'], request['arguments']
    reasons = []
    if operation not in METHODS:
        reasons.append('Operation is unavailable or cannot end/reset this active episode')
    else:
        try:
            method_signatures[METHODS[operation]].bind(**arguments)
        except TypeError as error:
            reasons.append(str(error))
    if operation in {'execute', 'execute_sequence'}:
        plan = arguments.get('plan_id')
        created = next((row['sequence'] for row in history
                        if row['response'].get('result', {}).get('plan_id') == plan), None)
        if created is None:
            reasons.append('Plan token was not produced in this prefix')
        else:
            changes = {'execute', 'execute_sequence', 'gripper', 'ground', 'set_model',
                       'apply_tracking', 'select_surface_contact'}
            if any(row['sequence'] > created and row['response'].get('ok')
                   and row['request']['operation'] in changes for row in history):
                reasons.append('Plan predates successful motion or model replacement')
    return {'signature_and_prefix_checks_pass': not reasons, 'issues': reasons,
            'scope': 'Method signature and obvious plan staleness only; not numerical feasibility or physical efficacy.'}


def build_blind(out):
    jobs = read_json(out / 'schedule-private.json')
    random.Random(630527).shuffle(jobs)
    mapping = {f'b{i:03d}': job['run_id'] for i, job in enumerate(jobs, 1)}
    mapping_file = out / 'blind-mapping-private.json'
    if mapping_file.exists() and read_json(mapping_file) != mapping:
        raise ValueError('Existing blind mapping differs')
    write_json(mapping_file, mapping)
    cases = {row['case_id']: row for row in read_json(out / 'cases-private.json')}
    jobs_by_id = {job['run_id']: job for job in jobs}
    method_signatures = signatures()
    views = []
    for blind_id, run_id in mapping.items():
        folder = out / 'responses' / run_id
        if not (folder / 'result.json').exists():
            continue
        result = read_json(folder / 'result.json')
        case = cases[jobs_by_id[run_id]['case_id']]
        row = {'blind_id': blind_id, 'case_id': case['case_id'],
               'case_expectation': case['private_expectation'],
               'valid_decision': result.get('valid_decision', False)}
        if row['valid_decision']:
            decision = read_json(folder / 'decision.json')
            history = [json.loads(line) for line in
                       (out / 'packets' / case['case_id'] / 'history.jsonl').read_text().splitlines()]
            row.update(decision=decision, contract=contract_check(decision, history, method_signatures))
        else:
            row['error'] = result.get('decision_error', result.get('error', 'No valid response'))
        views.append(row)
    write_json(out / 'blind-review.json', views)
    write_json(out / 'blind-review-provenance.json', {
        'created_before_manual_scores': not (out / 'scores-blind.json').exists(),
        'script_sha256': sha256(Path(__file__)), 'responses_available': len(views),
        'grading_rubric_sha256': sha256(out / 'rubric-private.json'),
        'model_name_and_latency_excluded_from_review': True,
    })
    print(json.dumps({'blind_responses': len(views), 'planned': len(jobs)}))


def summarize(out, group_by='model'):
    mapping = read_json(out / 'blind-mapping-private.json')
    scores = read_json(out / 'scores-blind.json')
    if {s['blind_id'] for s in scores} != set(mapping) or len(scores) != len(mapping):
        raise ValueError('Every registered response needs exactly one review entry')
    jobs = {j['run_id']: j for j in read_json(out / 'schedule-private.json')}
    blind = {r['blind_id']: r for r in read_json(out / 'blind-review.json')}
    dimensions = list(read_json(out / 'rubric-private.json')['dimensions'])
    rows, models = [], {}
    for score in scores:
        job = jobs[mapping[score['blind_id']]]
        row = {**job, **score}
        if set(row['scores']) != set(dimensions) or any(
                type(n) is not int or not 0 <= n <= 2 for n in row['scores'].values()):
            raise ValueError('Invalid rubric score')
        row['total'] = sum(row['scores'].values())
        row['contract'] = blind[row['blind_id']].get('contract')
        rows.append(row)
        bucket = models.setdefault(job[group_by], {'responses': 0, 'total_score': 0,
            'dimensions': {d: 0 for d in dimensions}, 'case_totals': {}})
        bucket['responses'] += 1
        bucket['total_score'] += row['total']
        for key, value in row['scores'].items():
            bucket['dimensions'][key] += value
        bucket['case_totals'].setdefault(job['case_id'], []).append(row['total'])
    summary = {'models' if group_by == 'model' else 'conditions': models, 'group_by': group_by, 'responses': rows,
        'review': 'Qualitative assistant review with model or condition labels masked; not an independent human panel.',
        'interpretation': 'Purposeful correlated development checkpoints; rubric totals are exploratory, '
                          'not recovery success rates or statistical model/prompt rankings.',
        'scores_sha256': sha256(out / 'scores-blind.json')}
    write_json(out / 'scored-results-private.json', summary)
    print(json.dumps(models, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--group-by', choices=['model', 'condition'], default='model')
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    if not destination.is_relative_to(ROOT / 'results'):
        raise ValueError('Use a results directory')
    if args.summarize:
        summarize(destination, args.group_by)
    else:
        build_blind(destination)
