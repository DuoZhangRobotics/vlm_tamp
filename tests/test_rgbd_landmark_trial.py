import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vlm_tamp.libero import ROOT
from vlm_tamp import rgbd_landmark_trial as trial
from vlm_tamp.rgbd_landmark_choices import blank_schema, sdk_arguments
from vlm_tamp.rgbd_numericals import json_text


def config(paired=True):
    result = json.loads((ROOT / 'configs/rgbd_dino_comparison.json').read_text())
    if not paired:
        result = json.loads((ROOT / 'configs/rgbd_dinov2_choices_trial.json').read_text())
    return result


def write_config(tmp_path, value):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(value))
    return path


def bundles(tmp_path, monkeypatch, *, paired=True):
    declared = config(paired)
    declared['dataset'] = 'dataset'
    declared['depth_dataset'] = 'depth'
    declared['prompt'] = 'prompt.txt'
    declared['frozen_protocol'] = 'frozen.json'
    declared['geometry_source_config'] = 'geometry.json'
    if paired:
        declared['candidate_datasets'] = {'dinov2': 'candidates/dinov2', 'dinov3': 'candidates/dinov3'}
    else:
        declared['candidate_dataset'] = 'candidates/dinov2'
    monkeypatch.setattr(trial, 'ROOT', tmp_path)
    (tmp_path / 'dataset').mkdir()
    (tmp_path / 'prompt.txt').write_text('Fill numbered landmark IDs.')
    (tmp_path / 'geometry.json').write_text(json_text({'geometry': {'fixture': 'unchanged policy'}}))
    identifiers = sorted({case['example_id'] for case in declared['cases']})
    records = [{'example_id': identifier, 'model_inputs': {'instruction': 'put the bowl on the plate'},
                'reference': {'oracle_fixture': 'never enter the model request'}} for identifier in identifiers]
    (tmp_path / 'dataset/records.jsonl').write_text(''.join(json_text(record)+'\n' for record in records))
    by_id = {record['example_id']: record for record in records}
    for identifier in identifiers:
        depth = tmp_path / declared['depth_dataset'] / identifier
        depth.mkdir(parents=True)
        np.save(depth / 'depth_meters.npy', np.ones((2, 2)))
        np.save(depth / 'world_points.npy', np.ones((2, 2, 3)))
        (depth / 'calibration.json').write_text('{}')
        for condition in trial.declared_conditions(declared):
            case = next(case for case in declared['cases'] if case['example_id'] == identifier
                        and trial.case_condition(declared, case) == condition)
            folder = trial.candidate_folder(declared, case)
            folder.mkdir(parents=True)
            pixel = [7, 8] if condition == 'dinov2' else [17, 18]
            (folder / 'candidates.json').write_text(json_text([
                {'id': 0, 'pixel_xy': pixel, 'predicted_region_id': 1,
                 'position_world_m': [1., 2., 3.], 'offline_owner': 'never enter the model request'},
            ]))
            Image.new('RGB', (512, 512), color='gray').save(folder / 'original.png')
            Image.new('RGB', (1536, 1536), color='white').save(folder / 'numbered.png')
            (folder / 'metadata.json').write_text('{}')
            np.save(folder / 'patch_features.npy', np.ones((1, 1, 3)))
            np.save(folder / 'predicted_masks.npy', np.ones((1, 2, 2), dtype=bool))
    # Implementation-file placeholders exercise the actual freeze hash loop in
    # a temporary tree. No project source or previous result files are changed.
    for relative in (
        'src/vlm_tamp/dinov2_features.py', 'src/vlm_tamp/dinov3_features.py',
        'src/vlm_tamp/rgbd_keypoint_candidates.py', 'src/vlm_tamp/rgbd_landmark_choices.py',
        'src/vlm_tamp/rgbd_landmark_trial.py', 'src/vlm_tamp/rgbd_candidate_evaluation.py',
        'src/vlm_tamp/rgbd_numericals.py', 'src/vlm_tamp/rgbd_axis_constraints.py',
        'src/vlm_tamp/rgbd_numeric_evaluation.py', 'src/vlm_tamp/pose_constraints.py',
        'src/vlm_tamp/rgb_role_points.py', 'src/vlm_tamp/rgb_role_choices.py',
        'src/vlm_tamp/experiment.py', 'scripts/prepare_dinov2_landmarks.py',
        'scripts/prepare_dino_landmark_comparison.py',
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('temporary implementation hash fixture\n')
    previous = tmp_path / 'results/old/summary.json'
    previous.parent.mkdir(parents=True)
    previous.write_text('{"preserve": true}\n')
    monkeypatch.setattr(trial, 'proposal_coverage', lambda record, root, candidates: {
        'fixture_pixel': candidates[0]['pixel_xy'], 'scope': 'temporary offline coverage fixture',
    })
    return declared, write_config(tmp_path, declared), by_id


def test_legacy_six_and_paired_twelve_configs_remain_bounded(tmp_path):
    for paired in (False, True):
        declared = config(paired)
        assert trial.read_config(write_config(tmp_path, declared)) == declared
    changed = config()
    changed['max_budget_usd'] = .001
    assert trial.read_config(write_config(tmp_path, changed))['max_budget_usd'] == .001


@pytest.mark.parametrize('change', [
    {'max_requests': 13}, {'max_requests': True}, {'max_budget_usd': .151},
    {'max_budget_usd': float('nan')}, {'max_budget_usd': False}, {'max_retries': 1},
    {'max_retries': True}, {'service_tier': 'priority'}, {'model': 'other-model'},
    {'conditions': ['dinov2', 'dinov2']}, {'conditions': ['dinov2', 'dinov4']},
    {'candidate_datasets': {'dinov2': 'one-directory'}},
])
def test_invalid_admission_and_encoder_settings_are_rejected(tmp_path, change):
    declared = config()
    declared.update(change)
    with pytest.raises(ValueError):
        trial.read_config(write_config(tmp_path, declared))


def test_duplicate_cases_unbalanced_conditions_and_unpaired_instructions_are_rejected(tmp_path):
    original = config()
    changed = copy.deepcopy(original)
    changed['cases'][1]['id'] = changed['cases'][0]['id']
    with pytest.raises(ValueError, match='unique'):
        trial.read_config(write_config(tmp_path, changed))
    changed = copy.deepcopy(original)
    changed['cases'][0]['condition'] = 'dinov3'
    with pytest.raises(ValueError, match='exactly six'):
        trial.read_config(write_config(tmp_path, changed))
    changed = copy.deepcopy(original)
    changed['cases'][6]['upright'] = True
    with pytest.raises(ValueError, match='same scenes'):
        trial.read_config(write_config(tmp_path, changed))


def test_condition_selects_its_own_bundle_without_exposing_reference_or_metric_metadata(tmp_path, monkeypatch):
    declared, _, records = bundles(tmp_path, monkeypatch)
    for case, expected in ((declared['cases'][0], [7, 8]), (declared['cases'][6], [17, 18])):
        candidates, original, numbered, shape, payload = trial.inputs(declared, case, records[case['example_id']])
        assert candidates[0]['pixel_xy'] == expected and shape == (512, 512)
        assert 'offline_owner' not in json_text(payload) and 'position_world_m' not in json_text(payload)
        assert 'oracle_fixture' not in json_text(payload)
        args = sdk_arguments(declared, payload, original, numbered, 'Choose IDs')
        assert len(args['input'][0]['content']) == 3
        assert args['text']['format']['schema'] == blank_schema(candidates)
        assert args['store'] is False and 'previous_response_id' not in args


def test_original_scene_shape_is_enforced_before_any_api_call(tmp_path, monkeypatch):
    declared, _, records = bundles(tmp_path, monkeypatch)
    case = declared['cases'][0]
    Image.new('RGB', (512, 511)).save(trial.candidate_folder(declared, case) / 'original.png')
    with pytest.raises(ValueError, match='512x512'):
        trial.inputs(declared, case, records[case['example_id']])


def test_freeze_preserves_encoder_specific_coverage_hashes_and_prior_results(tmp_path, monkeypatch):
    declared, path, _ = bundles(tmp_path, monkeypatch)
    previous = tmp_path / 'results/old/summary.json'
    before = previous.read_bytes()
    protocol = trial.freeze(path)
    assert len(protocol['requests']) == 12
    assert [request['condition'] for request in protocol['requests']] == ['dinov2']*6+['dinov3']*6
    key = declared['cases'][0]['example_id']
    coverage = protocol['offline_proposal_coverage']
    assert coverage['dinov2/'+key]['fixture_pixel'] == [7, 8]
    assert coverage['dinov3/'+key]['fixture_pixel'] == [17, 18]
    assert len(coverage) == 8
    assert 'src/vlm_tamp/dinov3_features.py' in protocol['implementation_hashes']
    assert 'scripts/prepare_dino_landmark_comparison.py' in protocol['implementation_hashes']
    assert protocol['preserved_artifacts']['results/old/summary.json'] == trial.file_sha(previous)
    assert previous.read_bytes() == before
    with pytest.raises(FileExistsError):
        trial.freeze(path)


def test_dry_run_builds_twelve_requests_and_correct_encoder_manifest_without_client(tmp_path, monkeypatch):
    declared, path, _ = bundles(tmp_path, monkeypatch)
    trial.freeze(path)
    output = tmp_path / 'dry-run'
    manifest = trial.run(path, output, dry_run=True)
    assert manifest['local_feature_model'] == {
        'dinov2': 'dinov2_vits14', 'dinov3': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    }
    assert manifest['conditions'] == ['dinov2', 'dinov3']
    assert manifest['planned_case_count'] == 12 and manifest['ledger']['api_attempts'] == 0
    assert len(list(output.glob('*/request.json'))) == 12
    assert not (output / 'predictions.jsonl').exists()
    for case in declared['cases']:
        request = json.loads((output / case['id'] / 'request.json').read_text())
        assert request['model'] == declared['model'] and request['store'] is False


def test_budget_refusal_preserves_all_twelve_not_run_denominators_without_model_outputs(tmp_path, monkeypatch):
    declared, path, _ = bundles(tmp_path, monkeypatch)
    declared['max_budget_usd'] = .000001
    path.write_text(json_text(declared))
    trial.freeze(path)
    calls = []

    def refuse_call(**kwargs):
        calls.append(kwargs)
        raise AssertionError('Budget refusal must happen before any provider request')

    constructor = lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(create=refuse_call))
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=constructor))
    output = tmp_path / 'zero-call-budget-stop'
    manifest = trial.run(path, output, allow_paid=True)
    rows = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
    assert not calls
    assert len(rows) == 12 and {row['status'] for row in rows} == {'not_run'}
    assert all(row['attempts'] == 0 and row['choices'] is None for row in rows)
    assert {row['condition'] for row in rows} == {'dinov2', 'dinov3'}
    assert manifest['stopped']['reason'] == 'budget_admission'
    assert manifest['ledger']['api_attempts'] == 0
    assert all('exceeds' in row['reason'] for row in rows)
    assert not list(output.glob('*/response.json'))
