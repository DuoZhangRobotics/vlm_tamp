"""Replay matched high-reasoning drawer replies, costs, geometry and report assets."""

import argparse
import base64
import hashlib
import json
from collections import Counter

import numpy as np
from PIL import Image

from report_drawer_models_high import (
    DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_TRIAL, HISTORICAL_REPORT,
    aggregate_model, audit_metrics, historical_comparison, model_key,
)
from report_task_grounding_scope import local_path, read_rows, response_context, response_text
from verify_task_grounding_scope import AssetReferences, _without_result_link, line_formula_check
from vlm_tamp.drawer_constraints import compile_drawer
from vlm_tamp.experiment import input_price_ceiling
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_landmark_trial import file_sha
from vlm_tamp.rgbd_numeric_evaluation import capture_checks
from vlm_tamp.rgbd_numericals import json_text


def _usage_totals():
    return dict.fromkeys(('api_attempts', 'estimated_usage_cost_usd', 'unknown_usage_reserved_usd',
                          'input_tokens', 'output_tokens', 'reasoning_tokens',
                          'cached_input_tokens', 'cache_write_tokens'), 0)


def verify(config_path=DEFAULT_CONFIG, trial_path=DEFAULT_TRIAL, report_path=DEFAULT_OUTPUT):
    from vlm_tamp.drawer_model_comparison import (
        check_protocol, grade_drawer, inputs as trial_inputs, model_config,
        request_arguments, request_cost_reservation, resolve_choices,
    )

    config, protocol, records = check_protocol(local_path(config_path))
    trial, report = local_path(trial_path), local_path(report_path)
    manifest = json.loads((trial / 'manifest.json').read_text())
    summary = json.loads((report / 'summary.json').read_text())
    rows = read_rows(trial / 'predictions.jsonl')
    assert manifest['config'] == summary['config'] == config
    assert config['reasoning_effort'] == summary['reasoning_effort'] == 'high'
    assert config['max_retries'] == 0 and config['max_requests'] == 8
    assert config['models'] == {'luna': 'gpt-6-luna', 'astra': 'gpt-6-astra'}
    assert config['overall_budget_usd'] == 2
    assert abs(config['overall_budget_usd'] - config['prior_usage_cost_usd'] - config['max_budget_usd']) < 1e-12
    assert summary['planned_drawer_cases'] == len(config['cases']) == 8
    assert Counter(model_key(case) for case in config['cases']) == {'luna': 4, 'astra': 4}
    assert {case['condition'] for case in config['cases']} == {'direct_rgb'}
    assert set(rows) == {case['id'] for case in config['cases']}
    tasks = {task['case_id']: task for task in summary['drawer_tasks']}
    assert set(tasks) == set(rows)
    assert summary['prompt_text'] == manifest['prompt_text'] == (ROOT / config['prompt']).read_text()
    assert summary['frozen_protocol_sha256'] == file_sha(ROOT / config['frozen_protocol'])
    assert manifest['provider'] == 'openai' and not manifest['dry_run'] and not manifest['offline_mock']
    assert not manifest['model_input_inventory'] and not manifest['reference_answers_sent']
    assert not manifest['depth_sent_to_vlm']
    assert not summary['robot_execution_tested'] and not summary['joint_space_compiled']
    assert not summary['model_ranking_established']
    observed = {key: _usage_totals() for key in config['models']}
    frozen_requests = {item['case_id']: item for item in protocol['requests']}
    captures, formula_checks, paired_requests = {}, [], {}
    responses = completed = truncated = failures = 0
    for case in config['cases']:
        identifier, key = case['id'], model_key(case)
        row, task, folder = rows[identifier], tasks[identifier], trial / identifier
        record = records[case['example_id']]
        per_model = model_config(config, case)
        assert row['case_id'] == identifier and row['example_id'] == case['example_id']
        assert task['raw_row'] == row and task['status'] == row['status']
        assert task['model'] == per_model['model'] and task['model_key'] == key
        assert task['metrics'] == grade_drawer(record, ROOT, case, row)
        assert task['response_context'] == response_context(folder, row)
        assert task['binding_audit'] == audit_metrics(task['metrics'], task['response_context']['selection_scored'])
        saved = json.loads((report / 'assets' / (identifier + '-result.json')).read_text())
        assert saved == _without_result_link(task)
        candidates, original, numbered, shape, payload = trial_inputs(config, case, record)
        assert candidates is None and numbered is None and shape == (512, 512)
        assert set(payload) == {'instruction', 'task_type', 'blanks'}
        assert json.loads((folder / 'payload.json').read_text()) == payload
        request = json.loads((folder / 'request.json').read_text())
        assert request == request_arguments(config, case, record)
        assert frozen_requests[identifier]['request_sha256'] == hashlib.sha256(json_text(request).encode()).hexdigest()
        assert request['model'] == per_model['model'] and request['reasoning'] == {'effort': 'high'}
        assert request['max_output_tokens'] == config['max_output_tokens'] and request['store'] is False
        assert request['service_tier'] == config['service_tier'] == 'default'
        assert request['instructions'] == summary['prompt_text']
        content = request['input'][0]['content']
        assert len(content) == 2 and content[0]['type'] == 'input_text' and content[1]['type'] == 'input_image'
        assert json.loads(content[0]['text']) == payload
        assert content[1]['detail'] == config['image_detail'] == 'high'
        assert base64.b64decode(content[1]['image_url'].split(',', 1)[1], validate=True) == original
        assert (folder / 'original.png').read_bytes() == original
        assert not (folder / 'numbered.png').exists()
        pair = dict(request)
        pair.pop('model')
        previous = paired_requests.setdefault(case['example_id'], pair)
        assert previous == pair
        capture = ROOT / config['dataset'] / case['example_id']
        assert original == (capture / 'rgb.png').read_bytes()
        depth = np.load(capture / 'depth_meters.npy', allow_pickle=False)
        cloud = np.load(capture / 'world_points.npy', allow_pickle=False)
        calibration = json.loads((capture / 'calibration.json').read_text())
        captures[case['example_id']] = capture_checks(depth, cloud, calibration)
        assert captures[case['example_id']]['passed']
        reservation = request_cost_reservation(config, request)
        assert reservation > 0
        aggregate_before = sum(group['estimated_usage_cost_usd'] + group['unknown_usage_reserved_usd']
                               for group in observed.values())
        response_file = folder / 'response.json'
        if response_file.exists():
            assert row['attempts'] == 1
            assert aggregate_before + reservation <= config['max_budget_usd'] + 1e-12
            raw = json.loads(response_file.read_text())
            responses += 1
            observed[key]['api_attempts'] += 1
            assert raw['model'] == per_model['model'] or raw['model'].startswith(per_model['model'] + '-')
            assert row['response_model'] == raw['model'] and row['response_status'] == raw['status']
            usage = raw.get('usage') or {}
            inputs, outputs = usage.get('input_tokens') or 0, usage.get('output_tokens') or 0
            details_in, details_out = usage.get('input_tokens_details') or {}, usage.get('output_tokens_details') or {}
            cost = (inputs * input_price_ceiling(per_model) + outputs * per_model['pricing']['output_per_million']) / 1e6
            assert abs(row['cost_usd'] - cost) < 1e-12
            observed[key]['estimated_usage_cost_usd'] += cost
            observed[key]['input_tokens'] += inputs
            observed[key]['output_tokens'] += outputs
            observed[key]['reasoning_tokens'] += details_out.get('reasoning_tokens') or 0
            observed[key]['cached_input_tokens'] += details_in.get('cached_tokens') or 0
            observed[key]['cache_write_tokens'] += details_in.get('cache_write_tokens') or 0
            assert outputs <= request['max_output_tokens']
            if inputs + outputs == 0:
                observed[key]['unknown_usage_reserved_usd'] += reservation
            if raw['status'] == 'completed':
                completed += 1
                try:
                    choices = json.loads(response_text(raw))
                    blanks = resolve_choices(choices, 'direct_rgb', None, shape)
                    numeric = compile_drawer(blanks, payload, depth, calibration, config['geometry'])
                except (ValueError, TypeError, KeyError):
                    assert row['status'] == 'invalid_output' and not task['response_context']['selection_scored']
                else:
                    assert row['status'] == 'valid' and task['response_context']['selection_scored']
                    assert row['choices'] == choices and row['blanks'] == blanks and row['numericals'] == numeric
                    assert json.loads((folder / 'numericals.json').read_text()) == numeric
                    assert numeric['planner_ready'] is False and numeric['robot_execution_tested'] is False
                    assert numeric['opening_goal']['status'] == 'unresolved'
                    assert numeric['grasp']['status'] == 'unresolved'
                    assert numeric['relations'] == numeric['path_constraints']
                    for relation in numeric['relations']:
                        assert relation['type'] == 'point_on_line' and relation['scope'] == 'path' and relation['stage'] == 'pull'
                        checked = line_formula_check(relation)
                        assert checked['passed']
                        formula_checks.append(checked)
            elif raw['status'] == 'incomplete' and (raw.get('incomplete_details') or {}).get('reason') == 'max_output_tokens':
                truncated += 1
                assert row['status'] == 'response_truncated'
                assert row.get('choices') is None and row.get('blanks') is None and row.get('numericals') is None
                assert not task['response_context']['selection_scored']
            else:
                assert row['status'] == 'invalid_output' and not task['response_context']['selection_scored']
        elif row['attempts']:
            failures += 1
            assert row['attempts'] == 1 and row['status'] == 'api_failure'
            assert aggregate_before + reservation <= config['max_budget_usd'] + 1e-12
            observed[key]['api_attempts'] += 1
            observed[key]['unknown_usage_reserved_usd'] += reservation
            assert row['cost_usd'] == 0 and not task['response_context']['selection_scored']
        else:
            assert row['status'] == 'not_run' and row['cost_usd'] == 0
            assert not task['response_context']['selection_scored']
        for artifact in task['artifacts'].values():
            path = report / artifact['asset']
            assert path.exists() and path.resolve().is_relative_to(report.resolve())
            if 'source' in artifact:
                assert file_sha(local_path(artifact['source'])) == artifact['sha256'] == file_sha(path)
            if path.suffix == '.png':
                with Image.open(path) as image:
                    image.verify()
    assert len(paired_requests) == len(captures) == 4
    assert set(paired_requests) == {f'libero_goal-00-state{state:03d}-seed0' for state in (0, 10, 20, 30)}
    ledger = manifest['ledger']
    assert summary['ledger'] == ledger
    for key, values in observed.items():
        actual = ledger['per_model'][key]
        for name, value in values.items():
            assert abs(actual[name] - value) < 1e-12, (key, name, actual[name], value)
        assert actual['accounting_input_rate_per_million'] == input_price_ceiling(model_config(
            config, next(case for case in config['cases'] if model_key(case) == key)))
        assert summary['model_summaries'][key] == aggregate_model([
            task for task in summary['drawer_tasks'] if task['model_key'] == key])
        assert summary['model_summaries'][key]['planned_cases'] == 4
    for name in _usage_totals():
        assert abs(ledger[name] - sum(values[name] for values in observed.values())) < 1e-12
    cost, reserved = ledger['estimated_usage_cost_usd'], ledger['unknown_usage_reserved_usd']
    assert ledger['api_attempts'] == responses + failures <= 8
    assert cost + reserved <= config['max_budget_usd'] + 1e-12
    assert config['prior_usage_cost_usd'] + cost + reserved <= 2 + 1e-12
    assert summary['historical_luna_none'] == historical_comparison()
    assert (report / 'assets/historical-frontview-summary.json').read_bytes() == (ROOT / HISTORICAL_REPORT).read_bytes()
    for name, source in (('config.json', local_path(config_path)), ('protocol.json', ROOT / config['frozen_protocol']),
                         ('manifest.json', trial / 'manifest.json'), ('prompt.txt', ROOT / config['prompt'])):
        assert (report / 'assets' / name).read_bytes() == source.read_bytes()
    page = (report / 'index.html').read_text()
    parser = AssetReferences()
    parser.feed(page)
    assert parser.drawer_articles == 8 and parser.canvases == 8
    for path in parser.paths:
        if path == 'verification.json':
            continue
        assert not path.startswith(('http:', 'https:', 'file:', 'javascript:', '/'))
        assert (report / path).exists() and (report / path).resolve().is_relative_to(report.resolve())
    assert 'Both models use <strong>high reasoning</strong>' in page
    assert 'Strict numerical binding correct' in page and 'not a general model ranking' in page
    assert 'unrequested requests' in page.lower() and 'truncated responses' in page
    assert 'not a model-only ablation' in page
    verification = {
        'passed': True, 'matched_high_reasoning_models': config['models'], 'reasoning_effort': 'high',
        'common_output_cap': config['max_output_tokens'], 'planned_cases_retained': 8,
        'real_api_replies_replayed': responses, 'completed_replies': completed,
        'truncated_replies_unscored': truncated, 'api_failures_reserved': failures,
        'same_frontal_rgb_images_checked': 4, 'same_prompt_schema_and_geometry_checked': True,
        'inventory_depth_reference_or_crops_sent_to_vlm': False,
        'base_grader_unchanged': True, 'strict_binding_audit_verified': True,
        'line_formula_checks': len(formula_checks),
        'maximum_line_jacobian_error': max((value['maximum_jacobian_error'] for value in formula_checks), default=None),
        'depth_capture_inverse_consistency_checks': len(captures),
        'estimated_new_usage_cost_usd': cost, 'unknown_usage_reserved_usd': reserved,
        'prior_usage_cost_usd': config['prior_usage_cost_usd'],
        'cumulative_usage_estimate_usd': config['prior_usage_cost_usd'] + cost,
        'overall_budget_usd': 2, 'per_model_usage_replayed': observed,
        'preserved_artifacts': len(protocol.get('preserved_artifacts', {})),
        'report_linked_assets_checked': len(set(parser.paths)),
        'model_ranking_or_planner_execution_established': False, 'browser_rendering_tested': False,
    }
    (report / 'verification.json').write_text(json_text(verification) + '\n')
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--trial', default=DEFAULT_TRIAL)
    parser.add_argument('--report', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json_text(verify(args.config, args.trial, args.report)))


if __name__ == '__main__':
    main()
