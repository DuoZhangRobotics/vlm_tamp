"""Replay actual two-point RGB replies, frozen geometry, costs and stress assets."""

import argparse
import base64
import hashlib
import json

import numpy as np
from PIL import Image

from report_drawer_geometry_stress import (
    BASELINE_REPORT, DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_TRIAL, aggregate,
)
from report_drawer_plane_ablation import support_cloud_data
from report_task_grounding_scope import local_path, read_rows, response_context, response_text
from verify_task_grounding_scope import AssetReferences, _without_result_link, line_formula_check
from vlm_tamp.drawer_local_plane import compile_local_drawer
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
    from vlm_tamp.drawer_geometry_stress import (
        FIELDS, check_protocol, compile_response, inputs, request_arguments,
        request_cost_reservation, settings,
    )

    cfg, protocol, records = check_protocol(local_path(config_path))
    trial, report = local_path(trial_path), local_path(report_path)
    manifest = json.loads((trial / 'manifest.json').read_text())
    summary = json.loads((report / 'summary.json').read_text())
    rows = read_rows(trial / 'predictions.jsonl')
    assert cfg == manifest['config'] == summary['config']
    assert cfg['model'] == 'gpt-6-luna' and cfg['reasoning_effort'] == 'high'
    assert cfg['max_output_tokens'] == 6000 and cfg['max_requests'] == 4 and cfg['max_retries'] == 0
    assert cfg['overall_budget_usd'] == 2
    assert abs(cfg['prior_usage_cost_usd'] + cfg['max_budget_usd'] - 2) < 1e-12
    assert len(cfg['cases']) == len(rows) == len(summary['tasks']) == 4
    assert summary['frozen_protocol_sha256'] == file_sha(ROOT / cfg['frozen_protocol'])
    assert manifest['frozen_protocol_sha256'] == summary['frozen_protocol_sha256']
    assert summary['prompt_text'] == manifest['prompt_text'] == (ROOT / cfg['prompt']).read_text()
    assert manifest['provider'] == 'openai' and not manifest['dry_run'] and not manifest['offline_mock']
    assert not manifest['model_input_inventory'] and not manifest['reference_answers_sent']
    assert not manifest['depth_sent_to_vlm'] and not manifest['crops_used']
    assert not manifest['segmentation_used_in_inference'] and manifest['robot_actions'] == 0
    assert not summary['planner_ready'] and not summary['official_benchmark']
    assert not summary['robot_execution_tested']
    frozen_requests = {request['case_id']: request for request in protocol['requests']}
    tasks = {task['case_id']: task for task in summary['tasks']}
    assert set(tasks) == set(rows) == {case['id'] for case in cfg['cases']}
    observed = _usage_totals()
    captures, formula_checks, actual_axes = {}, [], {}
    responses = completed = truncated = failures = selected_support = 0
    for case in cfg['cases']:
        identifier = case['id']
        row, task, folder = rows[identifier], tasks[identifier], trial / identifier
        record = records[case['example_id']]
        assert task['raw_row'] == row and task['status'] == row['status']
        assert task['metrics'] == (row.get('metrics') or {})
        assert task['support_audit'] == (row.get('support_audit') or {})
        assert task['response_context'] == response_context(folder, row)
        original, payload = inputs(cfg, case, record)
        assert set(payload) == {'instruction', 'task_type', 'blanks'}
        assert set(payload['blanks']) == set(FIELDS) == {
            'source_uv', 'handle_uv', 'joint_type', 'motion', 'path_constraint'}
        assert json.loads((folder / 'payload.json').read_text()) == payload
        request = json.loads((folder / 'request.json').read_text())
        assert request == request_arguments(cfg, case, record)
        assert hashlib.sha256(json_text(request).encode()).hexdigest() == frozen_requests[identifier]['request_sha256']
        assert request['reasoning'] == {'effort': 'high'} and request['model'] == 'gpt-6-luna'
        assert request['max_output_tokens'] == cfg['max_output_tokens'] and request['store'] is False
        assert request['service_tier'] == cfg['service_tier'] == 'default'
        content = request['input'][0]['content']
        assert len(content) == 2 and content[0]['type'] == 'input_text' and content[1]['type'] == 'input_image'
        assert json.loads(content[0]['text']) == payload and content[1]['detail'] == 'high'
        assert base64.b64decode(content[1]['image_url'].split(',', 1)[1], validate=True) == original
        assert (folder / 'original.png').read_bytes() == original
        capture = ROOT / cfg['dataset'] / case['example_id']
        assert (capture / 'rgb.png').read_bytes() == original
        depth = np.load(capture / 'depth_meters.npy', allow_pickle=False)
        cloud = np.load(capture / 'world_points.npy', allow_pickle=False)
        calibration = json.loads((capture / 'calibration.json').read_text())
        captures[identifier] = capture_checks(depth, cloud, calibration)
        assert captures[identifier]['passed']
        intervention = json.loads((capture / 'intervention.json').read_text())
        assert task['capture_intervention'] == intervention
        assert intervention['robot_actions'] == intervention['paid_api_calls'] == 0
        assert not intervention['official_initial_state_after_intervention']
        assert not intervention['model_inventory']
        assert not intervention['camera_selected_using_object_pose_or_axis_or_mask']
        reference = json.loads((capture / 'evaluation/reference.json').read_text())
        actual_axes[identifier] = reference['middle_drawer']['axisopeningworld']
        reservation = request_cost_reservation(cfg, request)
        assert reservation > 0
        before = observed['estimated_usage_cost_usd'] + observed['unknown_usage_reserved_usd']
        response_file = folder / 'response.json'
        numeric = row.get('numericals')
        if response_file.exists():
            assert row['attempts'] == 1 and before + reservation <= cfg['max_budget_usd'] + 1e-12
            raw = json.loads(response_file.read_text())
            assert raw['model'] == 'gpt-6-luna' or raw['model'].startswith('gpt-6-luna-')
            assert row['response_model'] == raw['model'] and row['response_status'] == raw['status']
            responses += 1
            observed['api_attempts'] += 1
            usage = raw.get('usage') or {}
            inputs_used, outputs_used = usage.get('input_tokens') or 0, usage.get('output_tokens') or 0
            details_in, details_out = usage.get('input_tokens_details') or {}, usage.get('output_tokens_details') or {}
            cost = (inputs_used * input_price_ceiling(cfg) + outputs_used * cfg['pricing']['output_per_million']) / 1e6
            assert abs(cost - row['cost_usd']) < 1e-12 and outputs_used <= cfg['max_output_tokens']
            observed['estimated_usage_cost_usd'] += cost
            observed['input_tokens'] += inputs_used
            observed['output_tokens'] += outputs_used
            observed['reasoning_tokens'] += details_out.get('reasoning_tokens') or 0
            observed['cached_input_tokens'] += details_in.get('cached_tokens') or 0
            observed['cache_write_tokens'] += details_in.get('cache_write_tokens') or 0
            if inputs_used + outputs_used == 0:
                observed['unknown_usage_reserved_usd'] += reservation
            replay = compile_response(cfg, case, record, raw)
            for key, value in replay.items():
                assert row[key] == value
            if raw['status'] == 'completed':
                completed += 1
            elif raw['status'] == 'incomplete' and (raw.get('incomplete_details') or {}).get('reason') == 'max_output_tokens':
                truncated += 1
                assert row['status'] == 'response_truncated' and not task['response_context']['selection_scored']
            if row['status'] == 'valid':
                assert task['response_context']['selection_scored']
                assert json.loads(response_text(raw)) == row['choices'] == row['blanks']
                assert set(row['blanks']) == set(FIELDS)
                recomputed = compile_local_drawer(row['blanks'], {'instruction': record['model_inputs']['instruction']},
                                                 depth, calibration, settings(cfg))
                assert recomputed == numeric
                assert json.loads((folder / 'numericals.json').read_text()) == numeric
                assert set(numeric['landmarks']) == {'source_uv', 'handle_uv'}
                assert not numeric['planner_ready'] and not numeric['robot_execution_tested']
                assert numeric['grasp']['status'] == numeric['opening_goal']['status'] == 'unresolved'
                support = numeric['local_plane_support']
                candidates = {tuple(point) for point in support['candidate_pixel_xy']}
                inliers = {tuple(point) for point in support['inlier_pixel_xy']}
                rejected = {tuple(point) for point in support['rejected_candidate_pixel_xy']}
                assert inliers <= candidates and rejected <= candidates and not inliers & rejected
                if numeric['drawer_front_plane'].get('sample_count') is not None:
                    assert inliers | rejected == candidates
                for x, y in candidates:
                    assert 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]
                    assert np.isfinite(depth[y, x]) and depth[y, x] > 0 and np.isfinite(cloud[y, x]).all()
                selected_support += len(inliers)
                assert numeric['relations'] == numeric['path_constraints']
                for relation in numeric['relations']:
                    assert relation['type'] == 'point_on_line' and relation['scope'] == 'path' and relation['stage'] == 'pull'
                    checked = line_formula_check(relation)
                    assert checked['passed']
                    formula_checks.append(checked)
            else:
                assert not task['response_context']['selection_scored'] and numeric is None
        elif row['attempts']:
            failures += 1
            assert row['status'] == 'api_failure' and row['attempts'] == 1
            assert before + reservation <= cfg['max_budget_usd'] + 1e-12
            observed['api_attempts'] += 1
            observed['unknown_usage_reserved_usd'] += reservation
            assert row['cost_usd'] == 0 and not task['response_context']['selection_scored']
        else:
            assert row['status'] == 'not_run' and row['cost_usd'] == 0
            assert not task['response_context']['selection_scored']
        with Image.open(capture / 'rgb.png') as image:
            rgb = np.asarray(image.convert('RGB'))
        assert task['cloud_data'] == support_cloud_data(rgb, cloud, numeric or {}, {})
        saved = json.loads((report / 'assets' / (identifier + '-result.json')).read_text())
        assert saved == _without_result_link(task)
        for artifact in task['artifacts'].values():
            path = report / artifact['asset']
            assert path.exists() and path.resolve().is_relative_to(report.resolve())
            if 'source' in artifact:
                assert file_sha(local_path(artifact['source'])) == artifact['sha256'] == file_sha(path)
            if path.suffix == '.png':
                with Image.open(path) as image:
                    image.verify()
    ledger = manifest['ledger']
    assert summary['ledger'] == ledger
    for key, value in observed.items():
        assert abs(ledger[key] - value) < 1e-12
    assert ledger['accounting_input_rate_per_million'] == input_price_ceiling(cfg)
    assert observed['api_attempts'] == responses + failures <= 4
    cumulative = cfg['prior_usage_cost_usd'] + observed['estimated_usage_cost_usd']
    assert abs(cumulative - manifest['cumulative_estimated_usage_cost_usd']) < 1e-12
    assert cumulative + observed['unknown_usage_reserved_usd'] <= 2 + 1e-12
    assert summary['aggregate'] == aggregate(summary['tasks'])
    assert summary['baseline_summary'] == json.loads((ROOT / BASELINE_REPORT).read_text())
    for name, source in (('config.json', local_path(config_path)), ('protocol.json', ROOT / cfg['frozen_protocol']),
                         ('manifest.json', trial / 'manifest.json'), ('prompt.txt', ROOT / cfg['prompt']),
                         ('baseline-summary.json', ROOT / BASELINE_REPORT)):
        assert (report / 'assets' / name).read_bytes() == source.read_bytes()
    assert len(captures) == 4
    assert len({tuple(np.round(axis, 8)) for axis in actual_axes.values()}) == 3
    parser = AssetReferences()
    page = (report / 'index.html').read_text()
    parser.feed(page)
    assert parser.drawer_articles == parser.canvases == 4
    for path in parser.paths:
        if path == 'verification.json':
            continue
        assert not path.startswith(('http:', 'https:', 'file:', 'javascript:', '/'))
        assert (report / path).exists() and (report / path).resolve().is_relative_to(report.resolve())
    assert 'valid JSON answer can still leave the geometry unresolved' in page
    assert 'not an official LIBERO benchmark' in page and 'not a matched geometry ablation' in page
    assert 'five blanks' in page and 'same fixed local-depth plane policy' in page
    verification = {
        'passed': True, 'real_luna_high_responses_replayed': responses,
        'completed_responses': completed, 'truncated_responses_unscored': truncated,
        'api_failure_attempts_reserved': failures, 'planned_cases_retained': 4,
        'rgb_only_two_point_payloads_checked': 4, 'reference_inventory_or_depth_sent_to_vlm': False,
        'same_frozen_local_geometry_policy_checked': True,
        'new_capture_interventions_checked': 4, 'distinct_held_back_joint_axis_directions': 3,
        'official_benchmark_initial_states': False, 'robot_actions': 0,
        'observed_inlier_support_pixels_checked': selected_support,
        'held_back_part_scoring_replayed_after_geometry': True,
        'depth_capture_inverse_consistency_checks': len(captures),
        'line_formula_checks': len(formula_checks),
        'maximum_line_jacobian_error': max((value['maximum_jacobian_error'] for value in formula_checks), default=None),
        'estimated_new_usage_cost_usd': observed['estimated_usage_cost_usd'],
        'unknown_usage_reserved_usd': observed['unknown_usage_reserved_usd'],
        'prior_usage_cost_usd': cfg['prior_usage_cost_usd'], 'cumulative_usage_estimate_usd': cumulative,
        'overall_budget_usd': 2, 'preserved_artifacts': len(protocol.get('preserved_artifacts', {})),
        'report_linked_assets_checked': len(set(parser.paths)),
        'robot_execution_tested': False, 'browser_rendering_tested': False,
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
