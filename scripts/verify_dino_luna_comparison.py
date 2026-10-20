"""Replay the Luna trial using frozen feature/compiler checks and new billing controls."""

import argparse
import base64
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from report_dino_landmark_comparison import aggregate, local_path, read_rows
from report_dino_luna_comparison import (
    DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_TRIAL, MINI_REPORT, comparison_data, read_response_context,
)
from verify_dino_landmark_comparison import AssetReferences, response_text, verify_features
from vlm_tamp.dataset import read_records
from vlm_tamp.experiment import input_price_ceiling
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_axis_constraints import compile_independent_axis
from vlm_tamp.rgbd_candidate_evaluation import grade_choices, proposal_coverage
from vlm_tamp.rgbd_landmark_choices import resolve_blanks, validate_blanks
from vlm_tamp.rgbd_landmark_trial import candidate_folder, file_sha, inputs
from vlm_tamp.rgbd_luna_landmark_trial import check_protocol, request_cost_reservation, sdk_arguments
from vlm_tamp.rgbd_numeric_evaluation import capture_checks
from vlm_tamp.rgbd_numericals import POINT_FIELDS, json_text, pixel_index, template


def _response_model_matches(actual, requested):
    """Allow a dated provider resolution without accepting another model family."""
    return actual == requested or (
        requested == "gpt-6-luna" and isinstance(actual, str)
        and re.fullmatch(r"gpt-6-luna-\d{4}-\d{2}-\d{2}", actual) is not None
    )


def verify(config_path=DEFAULT_CONFIG, trial_path=DEFAULT_TRIAL, report_path=DEFAULT_OUTPUT):
    config, protocol, records = check_protocol(local_path(config_path))
    _, root = read_records(ROOT / config['dataset'])
    trial, report = local_path(trial_path), local_path(report_path)
    manifest = json.loads((trial / 'manifest.json').read_text())
    summary = json.loads((report / 'summary.json').read_text())
    rows = read_rows(trial)
    assert manifest['config'] == config and manifest['provider'] == 'openai'
    assert not manifest['offline_mock'] and not manifest['dry_run']
    assert not manifest['model_input_inventory'] and not manifest['reference_answers_sent'] and not manifest['depth_sent_to_vlm']
    assert manifest['input_images'] == 2 and manifest['segmentation_used'] and not manifest['crops_used']
    assert manifest['prompt_text'] == (ROOT / config['prompt']).read_text()
    assert manifest['frozen_protocol_sha256'] == file_sha(ROOT / config['frozen_protocol']) == summary['frozen_protocol_sha256']
    assert summary['config'] == config and summary['prompt_text'] == manifest['prompt_text']
    assert len(config['cases']) == summary['cases'] == summary['planned_cases'] == 12
    assert Counter(case['condition'] for case in config['cases']) == {'dinov2': 6, 'dinov3': 6}
    assert set(rows) <= {case['id'] for case in config['cases']}
    summary_tasks = {task['id']: task for task in summary['tasks']}
    assert len(summary_tasks) == 12
    checked_features, checked_checkpoints, checks, captures, paired = {}, set(), [], {}, {}
    inputs_total, outputs_total, reasoning, cached, cache_written, raw_responses = 0, 0, 0, 0, 0, 0
    failed_attempts, unknown_reserved = 0, 0.
    for case, frozen in zip(config['cases'], protocol['requests']):
        identifier = case['id']
        row = rows.get(identifier, {'case_id': identifier, 'status': 'not_run', 'choices': None,
                                   'blanks': None, 'numericals': None, 'attempts': 0, 'cost_usd': 0.,
                                   'reason': 'No saved result; retained in configured denominator'})
        task = summary_tasks[identifier]
        assert task['raw_row'] == row and task['status'] == row['status']
        context = read_response_context(trial, identifier, row)
        assert task['response_context'] == context
        assert task['choices'] == row.get('choices') and task['numericals'] == row.get('numericals')
        candidates, original, numbered, shape, payload = inputs(config, case, records[case['example_id']])
        assert frozen['case_id'] == identifier and frozen['condition'] == case['condition'] and frozen['payload'] == payload
        folder = trial / identifier
        args = sdk_arguments(config, payload, original, numbered, manifest['prompt_text'])
        assert json.loads((folder / 'request.json').read_text()) == args
        assert args['model'] == 'gpt-6-luna' and args['reasoning'] == {'effort': 'none'}
        assert args['max_output_tokens'] == 1000 and args['store'] is False
        assert json.loads((folder / 'payload.json').read_text()) == payload
        images = [item for item in args['input'][0]['content'] if item['type'] == 'input_image']
        assert len(images) == 2
        for image, expected in zip(images, (original, numbered)):
            assert base64.b64decode(image['image_url'].split(',', 1)[1], validate=True) == expected
        assert (folder / 'original.png').read_bytes() == original and (folder / 'numbered.png').read_bytes() == numbered
        coverage_key = case['condition']+'/'+case['example_id']
        coverage = proposal_coverage(records[case['example_id']], root, candidates)
        assert coverage == protocol['offline_proposal_coverage'][coverage_key] == summary['proposal_coverage_before_selection'][coverage_key]
        candidate_dir = candidate_folder(config, case)
        metadata = json.loads((candidate_dir / 'metadata.json').read_text())
        assert task['candidate_metadata'] == metadata == summary['candidate_metadata'][coverage_key]
        if coverage_key not in checked_features:
            checked_features[coverage_key] = verify_features(config, case, candidates, metadata, checked_checkpoints)
            paired.setdefault(case['example_id'], {})[case['condition']] = {
                'original': file_sha(candidate_dir / 'original.png'),
                'masks': file_sha(candidate_dir / 'predicted_masks.npy'),
                'segmentation': metadata['segmentation'], 'proposal': metadata['proposal_settings'],
                'depth': metadata['depth_sha256'], 'cloud': metadata['world_points_sha256'],
            }
        capture = ROOT / config['depth_dataset'] / case['example_id']
        depth = np.load(capture / 'depth_meters.npy', allow_pickle=False)
        cloud = np.load(capture / 'world_points.npy', allow_pickle=False)
        calibration = json.loads((capture / 'calibration.json').read_text())
        captures[case['example_id']] = capture_checks(depth, cloud, calibration)
        assert captures[case['example_id']]['passed']
        raw_path = folder / 'response.json'
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
            assert raw['status'] in {'completed', 'incomplete'} and _response_model_matches(raw['model'], config['model']) and raw.get('id')
            assert row['response_model'] == raw['model'] and row['response_status'] == raw['status']
            if raw['status'] == 'incomplete':
                assert raw['incomplete_details']['reason'] == 'max_output_tokens'
                assert row['status'] == 'invalid_output' and not context['selection_diagnostic_applicable']
            usage = raw['usage']
            assert usage['input_tokens'] > 0 and usage['output_tokens'] > 0
            inputs_total += usage['input_tokens']
            outputs_total += usage['output_tokens']
            reasoning += (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0) or 0
            cached += (usage.get('input_tokens_details') or {}).get('cached_tokens', 0) or 0
            cache_written += (usage.get('input_tokens_details') or {}).get('cache_write_tokens', 0) or 0
            raw_responses += 1
            assert row['attempts'] == 1
            try:
                if raw['status'] != 'completed':
                    raise ValueError('A complete structured answer is required')
                choices = validate_blanks(json.loads(response_text(raw)), candidates)
            except (ValueError, TypeError) as error:
                assert row['status'] == 'invalid_output' and row.get('choices') is None
                assert row['error_type'] == type(error).__name__
                assert row.get('blanks') is None and row.get('numericals') is None
            else:
                assert raw['status'] == 'completed'
                assert choices == row['choices']
                resolved = resolve_blanks(choices, candidates, shape)
                assert resolved == row['blanks'] == task['resolved_blanks']
                by_id = {candidate['id']: candidate for candidate in candidates}
                for field in POINT_FIELDS:
                    selected = choices[field.removesuffix('_uv')+'_id']
                    assert resolved[field] is None if selected is None else (
                        pixel_index(resolved[field], shape) == tuple(by_id[selected]['pixel_xy']))
                declared = template(records[case['example_id']]['model_inputs']['instruction'], case['upright'])
                numeric = compile_independent_axis(resolved, declared, depth, calibration, protocol['geometry'])
                assert numeric == row['numericals'] == json.loads((folder / 'numericals.json').read_text())
                assert row['status'] == 'valid'
            expected_cost = (usage['input_tokens']*input_price_ceiling(config)
                             +usage['output_tokens']*config['pricing']['output_per_million'])/1e6
            assert abs(expected_cost-row['cost_usd']) < 1e-12
        else:
            assert row['status'] in {'not_run', 'api_failure'}
            if row['status'] == 'api_failure':
                assert row['attempts'] == 1 and row.get('error_type')
                assert row['cost_usd'] == 0
                failed_attempts += 1
                unknown_reserved += request_cost_reservation(config, manifest['prompt_text'], payload, candidates)
            else:
                assert row.get('attempts', 0) == 0
            assert row.get('choices') is None and row.get('numericals') is None
        metric = grade_choices(records[case['example_id']], root, case, row.get('choices'), candidates, row.get('numericals'))
        assert metric == task['metrics']
        assert task['visual_selection_diagnostics'] == (
            metric['selection_diagnostics'] if context['selection_diagnostic_applicable'] else None)
        checks.extend(metric['relation_checks'])
        assert json.loads((report / 'assets' / (identifier+'-result.json')).read_text()) == task
        for name in ('original.png', 'numbered.png', 'candidates.json', 'metadata.json'):
            assert (report / 'assets' / (identifier+'-'+name)).read_bytes() == (candidate_dir / name).read_bytes()
        for name in ('request.json', 'payload.json', 'response.json'):
            if (folder / name).exists():
                assert (report / 'assets' / (identifier+'-'+name)).read_bytes() == (folder / name).read_bytes()
    assert len(checked_features) == 8 and len(captures) == 4
    assert all(pair['dinov2'] == pair['dinov3'] for pair in paired.values())
    assert captures == summary['capture_checks']
    for condition in config['conditions']:
        assert aggregate([task for task in summary['tasks'] if task['condition'] == condition]) == summary['condition_summaries'][condition]
        assert summary['condition_summaries'][condition]['cases'] == 6
    baseline = json.loads(Path(summary['baseline_source']).read_text())
    assert baseline['semantic_complete_goals'] == summary['baseline_direct_xy']['semantic_complete_goals'] == 2
    assert baseline['cases'] == summary['baseline_direct_xy']['cases'] == 6
    assert all(baseline[key] == value for key, value in summary['baseline_direct_xy'].items())
    ledger = manifest['ledger']
    assert ledger == summary['ledger'] and ledger['api_attempts'] == raw_responses + failed_attempts <= 12
    assert ledger['input_tokens'] == inputs_total and ledger['output_tokens'] == outputs_total
    assert ledger['reasoning_tokens'] == reasoning == 0 and ledger['cached_input_tokens'] == cached
    assert ledger['cache_write_tokens'] == cache_written
    assert ledger['accounting_input_rate_per_million'] == input_price_ceiling(config) == .125
    assert summary['stopped'] == manifest.get('stopped')
    assert summary['incomplete_comparison'] == any(task['response_context']['status'] != 'completed'
                                                  or task['status'] != 'valid' for task in summary['tasks'])
    assert summary['encoder_ranking_established'] is False
    cost = (inputs_total*input_price_ceiling(config)+outputs_total*config['pricing']['output_per_million'])/1e6
    assert abs(cost-ledger['estimated_usage_cost_usd']) < 1e-12
    assert cost + unknown_reserved <= config['max_budget_usd'] <= 2.00
    assert abs(ledger['unknown_usage_reserved_usd'] - unknown_reserved) < 1e-12
    assert all(check['passed'] for check in checks)
    for key in ('part_coverage_verified', 'contact_coverage_verified', 'axis_coverage_verified', 'grasp_stability_verified',
                'ik_tested', 'robot_execution_tested', 'joint_space_compiled', 'reference_answers_in_inference'):
        assert summary[key] is False
    parser = AssetReferences()
    page = (report / 'index.html').read_text()
    parser.feed(page)
    assert parser.articles == parser.canvases == 12
    for value in parser.paths:
        path = (report / value).resolve()
        assert path.is_relative_to(report.resolve()) and path.is_file()
        if path.suffix == '.png':
            with Image.open(path) as image:
                image.verify()
    for path, name in ((local_path(config_path), 'config.json'),
                       (ROOT / config['frozen_protocol'], 'protocol.json'), (trial / 'manifest.json', 'manifest.json')):
        assert (report / 'assets' / name).read_bytes() == path.read_bytes()
    assert (report / 'assets/prompt.txt').read_text() == manifest['prompt_text']
    assert 'does not isolate the feature backbone' in page
    history = json.loads(MINI_REPORT.read_text())
    comparison = comparison_data(summary, history)
    assert comparison == summary['historical_mini_comparison']
    assert summary['reasoning_effort'] == 'none' and summary['model_ranking_established'] is False
    assert summary['actual_response_models'] == dict(Counter(
        json.loads(path.read_text()).get('model') for path in trial.glob('*/response.json')))
    assert (report / 'assets/historical-mini-summary.json').read_bytes() == MINI_REPORT.read_bytes()
    assert json.loads((report / 'assets/mini-luna-comparison.json').read_text()) == comparison
    assert 'GPT-6 Luna' in page and 'reasoning disabled' in page
    assert 'Original RGB sent to Mini' not in page and 'Numbered RGB sent to Mini' not in page
    verification = {
        'passed': True, 'requested_model': config['model'], 'actual_response_models': summary['actual_response_models'],
        'reasoning_tokens': reasoning, 'cache_write_tokens': cache_written,
        'accounting_input_rate_per_million': input_price_ceiling(config),
        'historical_mini_inputs_and_statuses_checked': True, 'model_ranking_established': False,
        'planned_cases_retained': 12, 'real_api_replies_replayed': raw_responses,
        'response_completion_counts': {condition: summary['condition_summaries'][condition]['response_completion_counts']
                                       for condition in config['conditions']},
        'incomplete_comparison': summary['incomplete_comparison'], 'stopped': manifest.get('stopped'),
        'actual_feature_scene_pairs_checked': len(checked_features), 'matched_scene_inputs_checked': len(paired),
        'checkpoint_hashes_checked': len(checked_checkpoints), 'depth_captures_checked': len(captures),
        'numerical_relation_self_checks': len(checks),
        'maximum_jacobian_error': max((check['maximum_jacobian_error'] for check in checks), default=None),
        'preserved_artifacts': len(protocol['preserved_artifacts']), 'estimated_usage_cost_usd': cost,
        'report_linked_assets_checked': len(set(parser.paths)),
        'physical_part_contact_axis_or_execution_verified': False, 'browser_rendering_tested': False,
    }
    (report / 'verification.json').write_text(json_text(verification)+'\n')
    return verification



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--trial", default=DEFAULT_TRIAL)
    parser.add_argument("--report", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json_text(verify(args.config, args.trial, args.report)))


if __name__ == "__main__":
    main()
