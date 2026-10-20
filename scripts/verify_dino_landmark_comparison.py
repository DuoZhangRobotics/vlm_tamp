"""Replay saved real replies, numerical compilation, matched inputs and report."""

import argparse
import base64
import hashlib
import json
import subprocess
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
from PIL import Image

from report_dino_landmark_comparison import (
    DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_TRIAL, aggregate, local_path, read_response_context, read_rows,
)
from vlm_tamp.dataset import read_records
from vlm_tamp.dinov2_features import file_sha as streamed_sha
from vlm_tamp.dinov3_features import MODEL_ID, MODEL_REVISION
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_axis_constraints import compile_independent_axis
from vlm_tamp.rgbd_candidate_evaluation import grade_choices, proposal_coverage
from vlm_tamp.rgbd_landmark_choices import resolve_blanks, sdk_arguments, validate_blanks, validate_candidates
from vlm_tamp.rgbd_landmark_trial import candidate_folder, check_protocol, file_sha, inputs
from vlm_tamp.rgbd_numeric_evaluation import capture_checks
from vlm_tamp.rgbd_numericals import POINT_FIELDS, json_text, pixel_index, template


class AssetReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths, self.articles, self.canvases = [], 0, 0

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'article':
            self.articles += 1
        if tag == 'canvas' and values.get('class') == 'cloud':
            self.canvases += 1
        for key in ('src', 'href'):
            value = values.get(key)
            if value and not value.startswith(('http:', 'https:', '#', 'data:')):
                self.paths.append(value)


def response_text(raw):
    return ''.join(item['text'] for block in raw.get('output', []) if block['type'] == 'message'
                   for item in block['content'] if item['type'] == 'output_text')


def verify_features(config, case, candidates, metadata, checked_checkpoints):
    folder = candidate_folder(config, case)
    capture = ROOT / config['depth_dataset'] / case['example_id']
    with Image.open(folder / 'original.png') as image:
        rgb = np.asarray(image.convert('RGB'))
    height, width = rgb.shape[:2]
    cloud = np.load(capture / 'world_points.npy', allow_pickle=False)
    depth = np.load(capture / 'depth_meters.npy', allow_pickle=False)
    masks = np.load(folder / 'predicted_masks.npy', allow_pickle=False)
    patch = np.load(folder / 'patch_features.npy', allow_pickle=False)
    assert metadata['model_inference_executed'] and metadata['cached_real_features_used']
    assert not metadata['offline_mock'] and not metadata['provided_inventory'] and not metadata['reference_masks_used']
    assert metadata['mask_zero_excluded'] and not metadata['cross_region_merge_used']
    assert metadata['segmentation'] == config['segmentation']
    assert metadata['proposal_settings'] == config['proposal']
    assert file_sha(folder / 'original.png') == file_sha(capture / 'rgb.png') == metadata['source_image_sha256']
    assert file_sha(capture / 'depth_meters.npy') == metadata['depth_sha256']
    assert file_sha(capture / 'world_points.npy') == metadata['world_points_sha256']
    assert file_sha(folder / 'predicted_masks.npy') == metadata['shared_predicted_masks_sha256']
    assert streamed_sha(ROOT / 'third_party/MobileSAM/weights/mobile_sam.pt') == metadata['segmentation_checkpoint_sha256']
    source = Path(metadata['feature_source_path'])
    assert file_sha(source / 'metadata.json') == metadata['feature_source_metadata_sha256']
    assert file_sha(folder / 'patch_features.npy') == file_sha(source / 'patch_features.npy') == metadata['source_patch_sha256']
    source_metadata = json.loads((source / 'metadata.json').read_text())
    assert source_metadata['model_inference_executed'] and not source_metadata.get('offline_mock', False)
    assert patch.ndim == 3 and patch.shape[2] == 384 and np.isfinite(patch).all() and float(np.std(patch)) > 0
    encoder = metadata['encoder']
    if case['condition'] == 'dinov2':
        assert encoder == source_metadata['encoder']
        assert encoder['model'] == 'dinov2_vits14' and encoder['model_input'] == 'RGB only'
        assert not encoder['depth_enters_encoder'] and encoder['model_inference_executed']
        assert encoder['original_shape'] == [height, width]
        assert list(patch.shape[:2]) == encoder['patch_grid_shape']
        assert encoder['patch_size'] == 14 and encoder['channels'] == 384
        checkpoint = Path(encoder['checkpoint_path'])
        expected = encoder['checkpoint_sha256']
        commit = subprocess.check_output(['git', '-C', str(ROOT / 'third_party/dinov2'), 'rev-parse', 'HEAD'], text=True).strip()
        assert commit == encoder['repository_commit']
    else:
        assert encoder == source_metadata
        assert encoder['model_id'] == MODEL_ID and encoder['requested_revision'] == MODEL_REVISION
        assert encoder['checkpoint_source'] == 'official_huggingface_snapshot' and encoder['revision_pinned_download']
        assert encoder['rgb_only'] and not encoder['depth_input'] and not encoder['training'] and not encoder['trust_remote_code']
        assert encoder['image_shape_hw'] == [height, width] and encoder['feature_channels'] == 384
        assert list(patch.shape[:2]) == encoder['grid_shape_hw'] and encoder['excluded_class_and_register_tokens'] == 5
        assert encoder['rgb_sha256'] == hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
        assert encoder['rgb_file_sha256'] == file_sha(capture / 'rgb.png')
        assert encoder['implementation_sha256'] == file_sha(ROOT / 'src/vlm_tamp/dinov3_features.py')
        checkpoint = Path(encoder['checkpoint_directory']) / 'model.safetensors'
        expected = encoder['weights_sha256']
        assert file_sha(checkpoint.parent / 'config.json') == encoder['config_sha256']
    if str(checkpoint) not in checked_checkpoints:
        assert streamed_sha(checkpoint) == expected
        checked_checkpoints.add(str(checkpoint))
    validate_candidates(candidates, (height, width))
    assert metadata['candidate_count'] == len(candidates)
    assert masks.ndim == 3 and masks.shape[1:] == (height, width) and masks.dtype == bool
    proposal = config['proposal']
    for candidate in candidates:
        x, y = candidate['pixel_xy']
        assert candidate['uv'] == [(x+.5)/width, (y+.5)/height]
        assert np.array_equal(np.asarray(candidate['position_world_m']), cloud[y, x])
        region = candidate['predicted_region_id']
        assert 1 <= region <= len(masks) and masks[region-1, y, x]
        assert proposal['minimum_depth_m'] <= depth[y, x] <= proposal['maximum_depth_m']
        assert np.all(np.asarray(candidate['position_world_m']) >= proposal['bounds_min'])
        assert np.all(np.asarray(candidate['position_world_m']) <= proposal['bounds_max'])
    return {'candidate_points': len(candidates), 'feature_grid_shape': list(patch.shape),
            'model': encoder.get('model', encoder.get('model_id')), 'checkpoint_hash_checked': True}


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
    inputs_total, outputs_total, reasoning, cached, raw_responses = 0, 0, 0, 0, 0
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
            assert raw['status'] in {'completed', 'incomplete'} and raw['model'] == config['model'] and raw.get('id')
            if raw['status'] == 'incomplete':
                assert raw['incomplete_details']['reason'] == 'max_output_tokens'
                assert row['status'] == 'invalid_output' and not context['selection_diagnostic_applicable']
            usage = raw['usage']
            assert usage['input_tokens'] > 0 and usage['output_tokens'] > 0
            inputs_total += usage['input_tokens']
            outputs_total += usage['output_tokens']
            reasoning += (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0) or 0
            cached += (usage.get('input_tokens_details') or {}).get('cached_tokens', 0) or 0
            raw_responses += 1
            assert row['attempts'] == 1
            try:
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
            expected_cost = (usage['input_tokens']*config['pricing']['input_per_million']
                             +usage['output_tokens']*config['pricing']['output_per_million'])/1e6
            assert abs(expected_cost-row['cost_usd']) < 1e-12
        else:
            assert row['status'] == 'not_run' and row.get('attempts', 0) == 0
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
    assert ledger == summary['ledger'] and ledger['api_attempts'] == raw_responses <= 12
    assert ledger['input_tokens'] == inputs_total and ledger['output_tokens'] == outputs_total
    assert ledger['reasoning_tokens'] == reasoning and ledger['cached_input_tokens'] == cached
    assert summary['stopped'] == manifest.get('stopped')
    assert summary['incomplete_comparison'] == any(task['response_context']['status'] != 'completed'
                                                  or task['status'] != 'valid' for task in summary['tasks'])
    assert summary['encoder_ranking_established'] is False
    cost = (inputs_total*config['pricing']['input_per_million']+outputs_total*config['pricing']['output_per_million'])/1e6
    assert abs(cost-ledger['estimated_usage_cost_usd']) < 1e-12
    assert cost <= config['max_budget_usd'] <= .15 and ledger['unknown_usage_reserved_usd'] == 0
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
    verification = {
        'passed': True, 'planned_cases_retained': 12, 'real_api_replies_replayed': raw_responses,
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
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--trial', default=DEFAULT_TRIAL)
    parser.add_argument('--report', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json_text(verify(args.config, args.trial, args.report)))


if __name__ == '__main__':
    main()
