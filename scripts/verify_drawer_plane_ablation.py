"""Replay observed-only drawer plane ablation and audit saved report provenance."""

import argparse
import base64
import json

import numpy as np
from PIL import Image

from report_drawer_plane_ablation import (
    DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_TRIAL, aggregate, support_cloud_data,
)
from report_task_grounding_scope import local_path, read_rows, response_text
from verify_task_grounding_scope import AssetReferences, _without_result_link, line_formula_check
from vlm_tamp.drawer_constraints import compile_drawer
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_landmark_trial import file_sha
from vlm_tamp.rgbd_numeric_evaluation import capture_checks
from vlm_tamp.rgbd_numericals import json_text


def verify(config_path=DEFAULT_CONFIG, trial_path=DEFAULT_TRIAL, report_path=DEFAULT_OUTPUT):
    from vlm_tamp.drawer_local_plane import compile_local_drawer
    from vlm_tamp.drawer_plane_trial import check_protocol, replay_case, settings

    config, protocol, records = check_protocol(local_path(config_path))
    trial, report = local_path(trial_path), local_path(report_path)
    rows = read_rows(trial / 'predictions.jsonl')
    sources = read_rows(ROOT / config['source_trial'] / 'predictions.jsonl')
    manifest = json.loads((trial / 'manifest.json').read_text())
    summary = json.loads((report / 'summary.json').read_text())
    assert len(config['cases']) == len(rows) == len(summary['tasks']) == 4
    assert summary['config'] == manifest['config'] == config
    assert summary['manifest'] == manifest
    assert summary['frozen_protocol_sha256'] == file_sha(ROOT / config['frozen_protocol'])
    assert summary['new_api_calls'] == summary['new_usage_cost_usd'] == 0
    assert manifest['api_calls'] == manifest['new_usage_cost_usd'] == 0
    assert manifest['offline_replay'] and not manifest['offline_mock']
    assert not summary['planner_ready'] and not summary['robot_execution_tested']
    assert not summary['new_views_tested']
    assert config['overall_budget_usd'] == 2 and config['prior_usage_cost_usd'] <= 2
    assert not summary['provenance']['held_back_labels_used_in_fit']
    assert not summary['provenance']['reference_normal_or_joint_axis_used_in_fit']
    tasks = {task['case_id']: task for task in summary['tasks']}
    assert set(tasks) == set(rows) == {case['id'] for case in config['cases']}
    captures, formula_checks = {}, []
    raw_responses_checked = supports_checked = 0
    for case in config['cases']:
        identifier = case['id']
        row, task = rows[identifier], tasks[identifier]
        source = sources[row['source_case_id']]
        assert source['status'] == 'valid' and source['attempts'] == 1
        assert source.get('response_model', 'gpt-6-luna').startswith('gpt-6-luna')
        assert row['source_blanks'] == source['blanks']
        assert row['new_api_calls'] == row['cost_usd'] == 0 and not row['semantic_pixels_changed']
        assert row['old_numericals'] == source['numericals']
        assert row['blanks'] == {key: source['blanks'][key] for key in (
            'source_uv', 'handle_uv', 'joint_type', 'motion', 'path_constraint')}
        assert row == replay_case(config, case, records[case['example_id']], source)
        assert task['raw_row'] == row and task['metrics'] == row['metrics']
        assert task['old_metrics'] == row['old_metrics'] and task['support_audit'] == row['support_audit']
        assert task['numericals'] == row['numericals'] and task['old_numericals'] == row['old_numericals']
        assert task['blanks'] == row['blanks']
        source_folder = ROOT / config['source_trial'] / row['source_case_id']
        request = json.loads((source_folder / 'request.json').read_text())
        raw = json.loads((source_folder / 'response.json').read_text())
        assert request['model'] == 'gpt-6-luna' and request['reasoning'] == {'effort': 'high'}
        assert raw['status'] == 'completed'
        assert json.loads(response_text(raw)) == source['choices']
        assert not (trial / identifier / 'request.json').exists()
        assert not (trial / identifier / 'response.json').exists()
        raw_responses_checked += 1
        capture = ROOT / config['dataset'] / case['example_id']
        original = (capture / 'rgb.png').read_bytes()
        content = request['input'][0]['content']
        assert len(content) == 2 and content[1]['type'] == 'input_image'
        assert base64.b64decode(content[1]['image_url'].split(',', 1)[1], validate=True) == original
        assert (source_folder / 'original.png').read_bytes() == original
        depth = np.load(capture / 'depth_meters.npy', allow_pickle=False)
        cloud = np.load(capture / 'world_points.npy', allow_pickle=False)
        calibration = json.loads((capture / 'calibration.json').read_text())
        captures[case['example_id']] = capture_checks(depth, cloud, calibration)
        assert captures[case['example_id']]['passed']
        payload = {'instruction': records[case['example_id']]['model_inputs']['instruction'],
                   'task_type': 'open_middle_drawer', 'requested_extension_m': None}
        assert compile_drawer(source['blanks'], payload, depth, calibration, config['geometry']) == row['old_numericals']
        numeric = compile_local_drawer(row['blanks'], payload, depth, calibration, settings(config))
        assert numeric == row['numericals']
        assert set(numeric['landmarks']) == {'source_uv', 'handle_uv'}
        for key in numeric['landmarks']:
            old_landmark, new_landmark = source['numericals']['landmarks'][key], numeric['landmarks'][key]
            assert new_landmark['status'] == old_landmark['status']
            for field in ('uv', 'pixel_xy', 'depth_m', 'position_m'):
                assert new_landmark.get(field) == old_landmark.get(field)
        assert numeric['planner_ready'] is False and numeric['robot_execution_tested'] is False
        assert numeric['grasp']['status'] == numeric['opening_goal']['status'] == 'unresolved'
        support = numeric['local_plane_support']
        candidates = {tuple(map(int, point)) for point in support.get('candidate_pixel_xy', [])}
        inliers = {tuple(map(int, point)) for point in support.get('inlier_pixel_xy', [])}
        rejected = {tuple(map(int, point)) for point in support.get('rejected_candidate_pixel_xy', [])}
        assert inliers <= candidates and rejected <= candidates and not inliers & rejected
        if numeric['drawer_front_plane'].get('sample_count') is not None:
            assert inliers | rejected == candidates
        for x, y in candidates:
            assert 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]
            assert np.isfinite(depth[y, x]) and depth[y, x] > 0 and np.isfinite(cloud[y, x]).all()
        supports_checked += len(inliers)
        for relation in numeric['relations']:
            assert relation['type'] == 'point_on_line' and relation['scope'] == 'path' and relation['stage'] == 'pull'
            checked = line_formula_check(relation)
            assert checked['passed']
            formula_checks.append(checked)
        assert numeric['relations'] == numeric['path_constraints']
        with Image.open(capture / 'rgb.png') as image:
            rgb = np.asarray(image.convert('RGB'))
        assert task['cloud_data'] == support_cloud_data(rgb, cloud, numeric, row['old_numericals'])
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
    assert len(captures) == raw_responses_checked == 4
    assert summary['aggregate'] == aggregate(summary['tasks'])
    for name, source in (('config.json', local_path(config_path)),
                         ('protocol.json', ROOT / config['frozen_protocol']),
                         ('manifest.json', trial / 'manifest.json')):
        assert (report / 'assets' / name).read_bytes() == source.read_bytes()
    parser = AssetReferences()
    page = (report / 'index.html').read_text()
    parser.feed(page)
    assert parser.drawer_articles == parser.canvases == 4
    for path in parser.paths:
        if path == 'verification.json':
            continue
        assert not path.startswith(('http:', 'https:', 'file:', 'javascript:', '/'))
        assert (report / path).exists() and (report / path).resolve().is_relative_to(report.resolve())
    assert 'No new API calls' in page and 'Two different geometry contracts' in page
    assert 'middle-panel support purity' in page and 'new views and drawer configurations have not been tested' in page
    verification = {
        'passed': True, 'saved_luna_high_responses_checked': raw_responses_checked,
        'new_api_calls': 0, 'new_usage_cost_usd': 0,
        'same_four_selected_source_and_handle_pixels_checked': True,
        'old_and_new_geometries_replayed': 4,
        'old_five_point_results_preserved': True, 'new_two_point_contract_reported_separately': True,
        'observed_fit_support_pixels_checked': supports_checked,
        'held_back_part_scoring_replayed_only_after_fitting': True,
        'depth_capture_inverse_consistency_checks': len(captures),
        'line_formula_checks': len(formula_checks),
        'maximum_line_jacobian_error': max((item['maximum_jacobian_error'] for item in formula_checks), default=None),
        'report_linked_assets_checked': len(set(parser.paths)),
        'preserved_artifacts': len(protocol.get('preserved_artifacts', {})),
        'cumulative_usage_estimate_usd': config['prior_usage_cost_usd'], 'overall_budget_usd': 2,
        'new_views_tested': False, 'robot_execution_tested': False, 'browser_rendering_tested': False,
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
