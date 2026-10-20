"""Frozen local-candidate selection trial; geometry and old results are unchanged."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import read_records
from .experiment import BudgetExceeded, Ledger
from .libero import ROOT
from .providers import ProviderResponse
from .rgbd_axis_constraints import compile_independent_axis
from .rgbd_candidate_evaluation import proposal_coverage
from .rgbd_landmark_choices import (
    blank_schema, candidate_payload, resolve_blanks, sdk_arguments, validate_blanks,
)
from .rgbd_numericals import json_text, template


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config['model'] != 'gpt-5.4-mini-2026-03-17' or config.get('service_tier', 'default') != 'default':
        raise ValueError('Use the declared Mini snapshot with standard service')
    if (isinstance(config['max_retries'], bool) or not isinstance(config['max_retries'], int)
            or config['max_retries'] != 0):
        raise ValueError('Landmark trials never retry API requests')
    conditions = declared_conditions(config)
    count = 6 * len(conditions)
    budget = config['max_budget_usd']
    ceiling = .15 if len(conditions) == 2 else .1
    if (isinstance(config['max_requests'], bool) or not isinstance(config['max_requests'], int)
            or config['max_requests'] != count or count > 12
            or isinstance(budget, bool) or not isinstance(budget, (int, float))
            or not math.isfinite(budget) or not 0 < budget <= ceiling):
        raise ValueError('Declare six calls per encoder, at most twelve calls and fifteen cents total')
    cases = config['cases']
    if not isinstance(cases, list) or len(cases) != count or any(not isinstance(case, dict) for case in cases):
        raise ValueError('Declare exactly six cases per encoder condition')
    identifiers = [case.get('id') for case in cases]
    if (any(not isinstance(identifier, str) or not identifier or identifier in {'.', '..'}
            or Path(identifier).name != identifier for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError('Case IDs must be unique nonempty directory names')
    observed = Counter()
    signatures = {}
    for case in cases:
        condition = case_condition(config, case)
        if condition not in conditions or not isinstance(case.get('upright'), bool):
            raise ValueError('Every case needs a declared encoder condition and boolean upright flag')
        if not isinstance(case.get('example_id'), str) or not case['example_id']:
            raise ValueError('Every case needs a source example ID')
        observed[condition] += 1
        signatures.setdefault(condition, Counter())[(case['example_id'], case['upright'])] += 1
    if any(observed[condition] != 6 for condition in conditions):
        raise ValueError('Each declared encoder condition must contain exactly six cases')
    if len(conditions) == 2 and signatures[conditions[0]] != signatures[conditions[1]]:
        raise ValueError('Paired encoders must use the same scenes and upright instructions')
    return config


def declared_conditions(config):
    if 'candidate_datasets' not in config:
        return ['dinov2']
    mapping = config['candidate_datasets']
    conditions = config.get('conditions')
    if (not isinstance(mapping, dict) or not isinstance(conditions, list) or not 1 <= len(conditions) <= 2
            or any(not isinstance(condition, str) or condition not in {'dinov2', 'dinov3'}
                   for condition in conditions)
            or len(set(conditions)) != len(conditions) or set(mapping) != set(conditions)
            or any(not isinstance(value, str) or not value for value in mapping.values())):
        raise ValueError('Declare matching candidate datasets for one or two DINO encoder conditions')
    return conditions


def case_condition(config, case):
    return case.get('condition', 'dinov2') if 'candidate_datasets' not in config else case.get('condition')


def candidate_folder(config, case):
    dataset = (config['candidate_datasets'][case_condition(config, case)]
               if 'candidate_datasets' in config else config['candidate_dataset'])
    return ROOT / dataset / case['example_id']


def inputs(config, case, record):
    folder = candidate_folder(config, case)
    candidates = json.loads((folder / 'candidates.json').read_text())
    original, numbered = (folder / 'original.png').read_bytes(), (folder / 'numbered.png').read_bytes()
    with Image.open(folder / 'original.png') as image:
        shape = (image.height, image.width)
    if shape != (512, 512):
        raise ValueError('Use original frozen 512x512 scenes')
    payload = candidate_payload(record['model_inputs']['instruction'], candidates, shape, case['upright'])
    return candidates, original, numbered, shape, payload


def freeze(config_path):
    config = read_config(config_path)
    destination = ROOT / config['frozen_protocol']
    if destination.exists():
        raise FileExistsError('Preserve the existing frozen landmark protocol')
    records, root = read_records(ROOT / config['dataset'])
    records = {record['example_id']: record for record in records}
    geometry = json.loads((ROOT / config['geometry_source_config']).read_text())['geometry']
    requests, coverage, files = [], {}, {}
    for case in config['cases']:
        candidates, _, _, shape, payload = inputs(config, case, records[case['example_id']])
        condition = case_condition(config, case)
        requests.append({'case_id': case['id'], 'condition': condition, 'payload': payload,
                         'schema_sha256': hashlib.sha256(json_text(blank_schema(candidates)).encode()).hexdigest()})
        folder = candidate_folder(config, case)
        for name in ('candidates.json', 'original.png', 'numbered.png', 'metadata.json', 'patch_features.npy',
                     'predicted_masks.npy'):
            files[str((folder / name).relative_to(ROOT))] = file_sha(folder / name)
        for name in ('depth_meters.npy', 'world_points.npy', 'calibration.json'):
            path = ROOT / config['depth_dataset'] / case['example_id'] / name
            files[str(path.relative_to(ROOT))] = file_sha(path)
        coverage_key = condition + '/' + case['example_id']
        if coverage_key not in coverage:
            coverage[coverage_key] = proposal_coverage(records[case['example_id']], root, candidates)
        if shape != (512, 512):
            raise ValueError('Use original frozen 512x512 scenes')
    implementation = [
        'src/vlm_tamp/dinov2_features.py', 'src/vlm_tamp/rgbd_keypoint_candidates.py',
        'src/vlm_tamp/rgbd_landmark_choices.py', 'src/vlm_tamp/rgbd_landmark_trial.py',
        'src/vlm_tamp/rgbd_candidate_evaluation.py', 'src/vlm_tamp/rgbd_numericals.py',
        'src/vlm_tamp/rgbd_axis_constraints.py', 'src/vlm_tamp/rgbd_numeric_evaluation.py',
        'src/vlm_tamp/pose_constraints.py', 'src/vlm_tamp/rgb_role_points.py',
        'src/vlm_tamp/rgb_role_choices.py', 'src/vlm_tamp/experiment.py',
        'scripts/prepare_dinov2_landmarks.py',
    ]
    if 'candidate_datasets' in config:
        implementation.extend(['src/vlm_tamp/dinov3_features.py', 'scripts/prepare_dino_landmark_comparison.py'])
    preserved = {}
    for pattern in ('**/predictions.jsonl', '**/summary.json'):
        for path in sorted((ROOT / 'results').glob(pattern)):
            preserved[str(path.relative_to(ROOT))] = file_sha(path)
    protocol = {'version': 'rgbd-landmark-choices-1.0', 'config': config, 'geometry': geometry,
                'dataset_sha256': file_sha(root / 'records.jsonl'),
                'prompt_sha256': file_sha(ROOT / config['prompt']),
                'implementation_hashes': {path: file_sha(ROOT / path) for path in implementation},
                'input_hashes': files, 'requests': requests, 'offline_proposal_coverage': coverage,
                'preserved_artifacts': preserved,
                'scope': 'development scenes, six cases per declared encoder; no population model ranking',
                'conditions': declared_conditions(config),
                'comparison_changes': ['candidate proposer', 'RGB point numbering', 'ID selection format'],
                'geometry_policy_unchanged': True, 'semantic_grader_unchanged': True,
                'reference_answers_sent': False, 'provided_inventory': False}
    destination.write_text(json_text(protocol)+'\n')
    return protocol


def check_protocol(config_path):
    config = read_config(config_path)
    protocol = json.loads((ROOT / config['frozen_protocol']).read_text())
    if config != protocol['config']:
        raise ValueError('Frozen landmark settings changed')
    if file_sha(ROOT / config['prompt']) != protocol['prompt_sha256']:
        raise ValueError('Frozen landmark prompt changed')
    geometry = json.loads((ROOT / config['geometry_source_config']).read_text())['geometry']
    if geometry != protocol['geometry']:
        raise ValueError('The shared geometry policy changed')
    for path, sha in {**protocol['implementation_hashes'], **protocol['input_hashes'],
                      **protocol['preserved_artifacts']}.items():
        if file_sha(ROOT / path) != sha:
            raise ValueError('Frozen or previous artifact changed: '+path)
    records, root = read_records(ROOT / config['dataset'])
    if file_sha(root / 'records.jsonl') != protocol['dataset_sha256']:
        raise ValueError('The source scenes changed')
    return config, protocol, {record['example_id']: record for record in records}


def run(config_path, output, *, allow_paid=False, dry_run=False):
    config, protocol, records = check_protocol(config_path)
    prompt = (ROOT / config['prompt']).read_text()
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new landmark selection output directory')
    if not dry_run and not allow_paid:
        raise PermissionError('Actual Mini calls require --allow-paid')
    output.mkdir(parents=True)
    ledger, arguments = Ledger(config), []
    conditions = declared_conditions(config)
    feature_models = {'dinov2': 'dinov2_vits14', 'dinov3': 'facebook/dinov3-vits16-pretrain-lvd1689m'}
    manifest = {'config': config, 'provider': 'openai', 'model': config['model'], 'offline_mock': False,
                'dry_run': dry_run, 'dataset': str(ROOT / config['dataset'] / 'records.jsonl'),
                'model_input_inventory': False, 'reference_answers_sent': False, 'depth_sent_to_vlm': False,
                'local_feature_model': ({condition: feature_models[condition] for condition in conditions}
                                        if 'candidate_datasets' in config else feature_models[conditions[0]]),
                'conditions': conditions, 'cases': config['cases'], 'planned_case_count': len(config['cases']),
                'segmentation_used': True, 'crops_used': False,
                'input_images': 2, 'localization_source': 'Mini-selected IDs from frozen RGB-D proposals',
                'frozen_protocol_sha256': file_sha(ROOT / config['frozen_protocol']), 'prompt_text': prompt,
                'ledger': ledger.as_dict()}
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json_text(manifest)+'\n')
    if len(protocol['requests']) != len(config['cases']):
        raise ValueError('Frozen requests must cover every declared case')
    for case, frozen in zip(config['cases'], protocol['requests']):
        candidates, original, numbered, shape, payload = inputs(config, case, records[case['example_id']])
        if (payload != frozen['payload'] or frozen['case_id'] != case['id']
                or frozen.get('condition', 'dinov2') != case_condition(config, case)
                or frozen['schema_sha256'] != hashlib.sha256(json_text(blank_schema(candidates)).encode()).hexdigest()):
            raise ValueError('Declared landmark request changed')
        folder = output / case['id']
        folder.mkdir()
        args = sdk_arguments(config, payload, original, numbered, prompt)
        (folder / 'request.json').write_text(json_text(args)+'\n')
        (folder / 'payload.json').write_text(json_text(payload)+'\n')
        (folder / 'original.png').write_bytes(original)
        (folder / 'numbered.png').write_bytes(numbered)
        arguments.append((case, folder, args, candidates, shape, payload))
    if dry_run:
        return manifest
    from openai import OpenAI
    client = OpenAI(base_url='https://api.openai.com/v1', timeout=config['timeout_seconds'], max_retries=0)
    rows = [{'case_id': case['id'], 'example_id': case['example_id'],
             'condition': case_condition(config, case), 'base_case_id': case.get('base_case_id', case['id']),
             'choices': None, 'blanks': None, 'numericals': None, 'status': 'not_run', 'attempts': 0,
             'cost_usd': 0., 'reason': 'Inference has not started'} for case in config['cases']]

    def save():
        manifest['ledger'] = ledger.as_dict()
        manifest['completed_case_count'] = sum(row['status'] != 'not_run' for row in rows)
        manifest_path.write_text(json_text(manifest)+'\n')
        (output / 'predictions.jsonl').write_text(''.join(json_text(row)+'\n' for row in rows))

    def stop(index, reason, detail):
        manifest['stopped'] = {'reason': reason, 'detail': detail, 'next_case_id': rows[index]['case_id']}
        for pending in rows[index:]:
            if pending['status'] == 'not_run':
                pending['reason'] = detail
        save()

    save()
    for index, (case, folder, args, candidates, shape, payload) in enumerate(arguments):
        input_bound = len((prompt+json_text(payload)+json_text(blank_schema(candidates))).encode())
        reservation = ((input_bound+config['image_token_allowance']+512)*config['pricing']['input_per_million']
                       +config['max_output_tokens']*config['pricing']['output_per_million'])/1e6
        row = rows[index]
        try:
            ledger.admit(reservation)
        except BudgetExceeded as error:
            stop(index, 'budget_admission', str(error))
            break
        row.pop('reason', None)
        row['status'] = 'api_failure'
        row['attempts'] = 1
        try:
            response = client.responses.create(**args)
        except Exception as error:
            ledger.unknown_cost_reservation += reservation
            row['error_type'] = type(error).__name__
            stop(index, 'api_failure', 'Stopped after API failure; no retry or substitute inference')
            break
        raw = response.model_dump(mode='json')
        (folder / 'response.json').write_text(json_text(raw)+'\n')
        row['cost_usd'] = ledger.account(ProviderResponse(response.output_text, raw,
                                                        response.usage.input_tokens if response.usage else 0,
                                                        response.usage.output_tokens if response.usage else 0), reservation)
        manifest['ledger'] = ledger.as_dict()
        manifest_path.write_text(json_text(manifest)+'\n')
        try:
            row['choices'] = validate_blanks(json.loads(response.output_text), candidates)
            row['blanks'] = resolve_blanks(row['choices'], candidates, shape)
            capture = ROOT / config['depth_dataset'] / case['example_id']
            depth = np.load(capture / 'depth_meters.npy', allow_pickle=False)
            calibration = json.loads((capture / 'calibration.json').read_text())
            declared = template(records[case['example_id']]['model_inputs']['instruction'], case['upright'])
            row['numericals'] = compile_independent_axis(row['blanks'], declared, depth, calibration, protocol['geometry'])
            row['status'] = 'valid'
            (folder / 'numericals.json').write_text(json_text(row['numericals'])+'\n')
        except (ValueError, TypeError, KeyError) as error:
            row['status'], row['error_type'] = 'invalid_output', type(error).__name__
        save()
        print(case['id'], row['status'], 'estimated $'+format(row['cost_usd'], '.5f'), flush=True)
    return manifest
