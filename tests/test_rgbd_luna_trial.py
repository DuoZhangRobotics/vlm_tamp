"""Safeguards for the separately frozen, low-cost Luna comparison.

These tests never call the API or change the previous Mini protocol/results.
"""

import copy
import json

import pytest

from vlm_tamp import rgbd_luna_landmark_trial as trial
from vlm_tamp.experiment import BudgetExceeded, Ledger
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_landmark_choices import candidate_payload
from vlm_tamp.rgbd_numericals import json_text


def declared_config():
    return json.loads((ROOT / 'configs/rgbd_dino_luna_comparison.json').read_text())


def write_config(tmp_path, value):
    path = tmp_path / 'luna-config.json'
    path.write_text(json.dumps(value))
    return path


def test_authorized_budget_accepts_the_same_twelve_cases_and_lower_limits(tmp_path):
    config = declared_config()
    assert config['max_budget_usd'] == 2
    assert trial.read_config(write_config(tmp_path, config)) == config
    original = json.loads((ROOT / 'configs/rgbd_dino_comparison.json').read_text())
    assert config['cases'] == original['cases']
    assert config['candidate_datasets'] == original['candidate_datasets']
    config['max_budget_usd'] = .05
    assert trial.read_config(write_config(tmp_path, config))['max_budget_usd'] == .05


@pytest.mark.parametrize('change', [
    {'model': 'gpt-5.4-mini-2026-03-17'},
    {'model': 'gpt-6'},
    {'reasoning_effort': 'medium'},
    {'service_tier': 'priority'},
    {'max_retries': 1},
    {'max_retries': True},
    {'max_retries': False},
    {'max_requests': 13},
    {'max_requests': True},
    {'max_budget_usd': 2.001},
    {'max_budget_usd': 0},
    {'max_budget_usd': -1},
    {'max_budget_usd': True},
    {'max_budget_usd': float('nan')},
    {'max_budget_usd': float('inf')},
    {'max_output_tokens': 3000},
    {'image_token_allowance': 1},
    {'prompt': 'a-different-prompt.txt'},
    {'dataset': 'different-scenes'},
    {'depth_dataset': 'different-depth'},
    {'geometry_source_config': 'different-geometry.json'},
    {'conditions': ['dinov3', 'dinov2']},
    {'candidate_datasets': {'dinov2': 'different-candidates', 'dinov3': 'different-candidates'}},
])
def test_unapproved_model_budget_and_frozen_input_changes_are_rejected(tmp_path, change):
    config = declared_config()
    config.update(change)
    with pytest.raises(ValueError):
        trial.read_config(write_config(tmp_path, config))


@pytest.mark.parametrize('field,value', [
    ('input_per_million', 0),
    ('cache_write_per_million', .1),
    ('output_per_million', .1),
])
def test_declared_rates_cannot_be_reduced_to_bypass_admission(tmp_path, field, value):
    config = declared_config()
    config['pricing'][field] = value
    with pytest.raises(ValueError):
        trial.read_config(write_config(tmp_path, config))


def test_pairwise_scene_and_instruction_edits_do_not_create_an_undeclared_trial(tmp_path):
    config = declared_config()
    # Changing both encoders still passes the old paired-case balance check.
    # The Luna run must instead preserve the exact parent requests.
    for index in (0, 6):
        config['cases'][index]['example_id'] = 'unapproved-scene'
        config['cases'][index]['upright'] = True
    with pytest.raises(ValueError):
        trial.read_config(write_config(tmp_path, config))
    config = declared_config()
    config['cases'] = config['cases'][6:] + config['cases'][:6]
    with pytest.raises(ValueError):
        trial.read_config(write_config(tmp_path, config))


def test_numeric_zero_cannot_replace_a_frozen_boolean_case_flag(tmp_path):
    config = declared_config()
    config['cases'][0]['upright'] = 0
    with pytest.raises(ValueError):
        trial.read_config(write_config(tmp_path, config))


def test_cache_write_cost_admission_blocks_a_request_the_input_rate_would_admit():
    config = declared_config()
    candidates = [{'id': 0, 'pixel_xy': [7, 8], 'predicted_region_id': 1}]
    payload = candidate_payload('put the bowl on the plate', candidates, (512, 512))
    reservation = trial.request_cost_reservation(config, 'Select visible landmark IDs.', payload, candidates)
    assert isinstance(reservation, float) and reservation > 0
    output_cost = config['max_output_tokens'] * config['pricing']['output_per_million'] / 1e6
    # Remove only the 25% cache-write premium to locate a budget boundary.
    lower_input_estimate = output_cost + (reservation-output_cost) * .1 / .125
    assert lower_input_estimate < reservation
    boundary_config = copy.deepcopy(config)
    boundary_config['max_budget_usd'] = (reservation + lower_input_estimate) / 2
    ledger = Ledger(boundary_config)
    with pytest.raises(BudgetExceeded):
        ledger.admit(reservation)
    assert ledger.attempts == 0
    ledger.admit(lower_input_estimate)
    assert ledger.attempts == 1


def test_unknown_usage_reservation_stays_inside_the_two_dollar_limit():
    config = declared_config()
    config['max_requests'] = 12
    ledger = Ledger(config)
    ledger.actual_cost = 1.99
    ledger.unknown_cost_reservation = .009
    with pytest.raises(BudgetExceeded):
        ledger.admit(.002)
    assert ledger.attempts == 0
    ledger.admit(.001)
    assert ledger.attempts == 1


@pytest.fixture
def parent_tree(tmp_path, monkeypatch):
    """Small, already-frozen parent for testing new hashes without altering data."""
    config = declared_config()
    parent_config = json.loads((ROOT / config['parent_config']).read_text())

    def put(relative, text):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    config_path = put('luna-config.json', json_text(config))
    put(config['parent_config'], json_text(parent_config))
    put(config['prompt'], 'Unchanged landmark selection prompt.\n')
    geometry = {'fixture': 'same geometry policy'}
    put(config['geometry_source_config'], json_text({'geometry': geometry}))
    records = [{'example_id': identifier, 'model_inputs': {'instruction': 'put the bowl on the plate'}}
               for identifier in sorted({case['example_id'] for case in config['cases']})]
    dataset = put(config['dataset'] + '/records.jsonl', ''.join(json_text(record)+'\n' for record in records))
    candidate = put(config['candidate_datasets']['dinov2'] + '/candidate-input.fixture', 'frozen candidates\n')
    source = put('src/vlm_tamp/rgbd_landmark_trial.py', 'historical source\n')
    previous = put('results/old/summary.json', '{"unchanged": true}\n')
    mini_response = put('results/rgbd-dino-comparison-mini/case/response.json', '{"old answer": true}\n')
    mini_image = put('results/rgbd-dino-comparison-report/case/selected.fixture', 'old report pixels\n')
    for relative in (
        'src/vlm_tamp/rgbd_luna_landmark_trial.py',
        'scripts/run_dino_luna_comparison.py',
        'scripts/report_dino_luna_comparison.py',
        'scripts/verify_dino_luna_comparison.py',
    ):
        put(relative, 'new transport fixture\n')
    parent = {
        'version': 'historical-test-fixture', 'config': parent_config,
        'geometry': geometry, 'dataset_sha256': trial.file_sha(dataset),
        'prompt_sha256': trial.file_sha(tmp_path / config['prompt']),
        'requests': [{'case_id': case['id'], 'payload': {'same request': case['id']}}
                     for case in config['cases']],
        'input_hashes': {str(candidate.relative_to(tmp_path)): trial.file_sha(candidate)},
        'implementation_hashes': {str(source.relative_to(tmp_path)): trial.file_sha(source)},
        'preserved_artifacts': {str(previous.relative_to(tmp_path)): trial.file_sha(previous)},
    }
    parent_path = put(parent_config['frozen_protocol'], json_text(parent))
    monkeypatch.setattr(trial, 'ROOT', tmp_path)
    # Only the inherited checker is replaced; freeze/check_protocol themselves
    # execute real file hashing and validation in this isolated temporary tree.
    monkeypatch.setattr(trial.historical, 'check_protocol', lambda path: (
        parent_config, copy.deepcopy(parent), {record['example_id']: record for record in records},
    ))
    return config_path, config, parent, parent_path, candidate, source, previous, mini_response, mini_image


def test_freeze_retains_parent_requests_and_inputs_and_preserves_every_prior_file(parent_tree):
    config_path, config, parent, parent_path, candidate, source, previous, mini_response, mini_image = parent_tree
    before = {path: path.read_bytes() for path in (parent_path, candidate, source, previous, mini_response, mini_image)}
    protocol = trial.freeze(config_path)
    assert protocol['requests'] == parent['requests']
    assert protocol['input_hashes'] == parent['input_hashes']
    assert protocol['parent_protocol_sha256'] == trial.file_sha(parent_path)
    assert protocol['budget_authorized_usd'] == 2
    for path in (previous, mini_response, mini_image):
        relative = str(path.relative_to(trial.ROOT))
        assert protocol['preserved_artifacts'][relative] == trial.file_sha(path)
    assert {path: path.read_bytes() for path in before} == before
    assert trial.check_protocol(config_path)[0] == config
    with pytest.raises(FileExistsError):
        trial.freeze(config_path)


@pytest.mark.parametrize('changed_file', [
    'parent_protocol', 'candidate_input', 'historical_source', 'previous_summary',
    'mini_response', 'mini_report_asset',
])
def test_preflight_refuses_changed_parent_input_or_previous_results_before_inference(parent_tree, changed_file):
    config_path, _, _, parent_path, candidate, source, previous, mini_response, mini_image = parent_tree
    trial.freeze(config_path)
    paths = dict(parent_protocol=parent_path, candidate_input=candidate, historical_source=source,
                 previous_summary=previous, mini_response=mini_response, mini_report_asset=mini_image)
    changed = paths[changed_file]
    changed.write_text(changed.read_text() + 'changed after freeze\n')
    with pytest.raises(ValueError):
        trial.check_protocol(config_path)
