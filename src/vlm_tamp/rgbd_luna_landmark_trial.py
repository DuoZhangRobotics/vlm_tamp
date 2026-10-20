"""Versioned Luna transport over frozen DINO landmark inputs and numerical grading."""

import hashlib
import json
import math
import copy
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import read_records
from .experiment import BudgetExceeded, Ledger, input_price_ceiling
from . import rgbd_landmark_trial as historical
from .libero import ROOT
from .providers import ProviderResponse
from .rgbd_axis_constraints import compile_independent_axis
from .rgbd_landmark_choices import (
    blank_schema, candidate_payload, resolve_blanks, sdk_arguments, validate_blanks,
)
from .rgbd_numericals import json_text, template


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path):
    """Allow only the approved Luna controls over the fixed historical inputs."""
    config = json.loads(Path(path).read_text())
    if config.get('parent_config') != 'configs/rgbd_dino_comparison.json':
        raise ValueError('Use the frozen matched Mini inputs as the parent')
    parent = historical.read_config(ROOT / config['parent_config'])
    changes = {'model', 'reasoning_effort', 'max_output_tokens', 'max_budget_usd',
               'frozen_protocol', 'parent_config', 'pricing', 'pricing_source', 'pricing_checked'}
    if set(config) != set(parent) | {'parent_config'}:
        raise ValueError('Declare exactly the parent fields and parent_config')
    for key in set(parent) - changes:
        if json_text(config[key]) != json_text(parent[key]):
            raise ValueError('Frozen matched input or request control changed: ' + key)
    if (config['model'] != 'gpt-6-luna' or config['reasoning_effort'] != 'none'
            or type(config['max_output_tokens']) is not int or config['max_output_tokens'] != 1000
            or config['frozen_protocol'] != 'configs/frozen_rgbd_dino_luna_comparison.json'):
        raise ValueError('Use the approved Luna model, no reasoning and 1000 output-token limit')
    budget = config['max_budget_usd']
    if (isinstance(budget, bool) or not isinstance(budget, (int, float))
            or not math.isfinite(budget) or not 0 < budget <= 2):
        raise ValueError('The user authorized at most two dollars for this experiment')
    prices = {'input_per_million': .10, 'cached_input_per_million': .01,
              'cache_write_per_million': .125, 'output_per_million': .50}
    if config['pricing'] != prices:
        raise ValueError('Use the verified standard Luna prices including cache writes')
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
        raise FileExistsError('Preserve the existing frozen Luna protocol')
    _, parent, _ = historical.check_protocol(ROOT / config['parent_config'])
    protocol = copy.deepcopy(parent)
    protocol.update(version='rgbd-luna-landmark-choices-1.0', config=config,
                    parent_config=config['parent_config'],
                    parent_config_sha256=file_sha(ROOT / config['parent_config']),
                    parent_protocol=parent['config']['frozen_protocol'],
                    parent_protocol_sha256=file_sha(ROOT / parent['config']['frozen_protocol']),
                    comparison_changes=['API model', 'reasoning effort', 'output-token cap', 'budget and pricing'],
                    geometry_policy_unchanged=True, semantic_grader_unchanged=True,
                    candidate_inputs_unchanged=True,
                    budget_authorized_usd=2.0)
    for name in ('src/vlm_tamp/rgbd_luna_landmark_trial.py',
                 'scripts/run_dino_luna_comparison.py',
                 'scripts/report_dino_luna_comparison.py',
                 'scripts/verify_dino_luna_comparison.py'):
        protocol['implementation_hashes'][name] = file_sha(ROOT / name)
    for folder in ('results/rgbd-dino-comparison-mini', 'results/rgbd-dino-comparison-report'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file():
                protocol['preserved_artifacts'][str(path.relative_to(ROOT))] = file_sha(path)
    destination.write_text(json_text(protocol) + '\n')
    return protocol


def check_protocol(config_path):
    config = read_config(config_path)
    _, parent, _ = historical.check_protocol(ROOT / config['parent_config'])
    protocol = json.loads((ROOT / config['frozen_protocol']).read_text())
    if json_text(config) != json_text(protocol['config']):
        raise ValueError('Frozen Luna settings changed')
    if (file_sha(ROOT / config['parent_config']) != protocol['parent_config_sha256']
            or file_sha(ROOT / protocol['parent_protocol']) != protocol['parent_protocol_sha256']):
        raise ValueError('The historical parent protocol changed')
    if protocol['requests'] != parent['requests'] or protocol['input_hashes'] != parent['input_hashes']:
        raise ValueError('Luna must use exactly the frozen candidate inputs and requests')
    if file_sha(ROOT / config['prompt']) != protocol['prompt_sha256']:
        raise ValueError('Frozen landmark prompt changed')
    geometry = json.loads((ROOT / config['geometry_source_config']).read_text())['geometry']
    if geometry != protocol['geometry'] or geometry != parent['geometry']:
        raise ValueError('The shared geometry policy changed')
    for path, sha in {**protocol['implementation_hashes'], **protocol['input_hashes'],
                      **protocol['preserved_artifacts']}.items():
        if file_sha(ROOT / path) != sha:
            raise ValueError('Frozen or previous artifact changed: ' + path)
    records, root = read_records(ROOT / config['dataset'])
    if file_sha(root / 'records.jsonl') != protocol['dataset_sha256']:
        raise ValueError('The source scenes changed')
    return config, protocol, {record['example_id']: record for record in records}


def request_cost_reservation(config, prompt, payload, candidates):
    # UTF-8 bytes and a fixed two-image allowance overestimate the usual request.
    input_bound = len((prompt + json_text(payload) + json_text(blank_schema(candidates))).encode())
    return ((input_bound + config['image_token_allowance'] + 512) * input_price_ceiling(config)
            + config['max_output_tokens'] * config['pricing']['output_per_million']) / 1e6


def run(config_path, output, *, allow_paid=False, dry_run=False):
    config, protocol, records = check_protocol(config_path)
    prompt = (ROOT / config['prompt']).read_text()
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new landmark selection output directory')
    if not dry_run and not allow_paid:
        raise PermissionError('Actual Luna calls require --allow-paid')
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
                'input_images': 2, 'localization_source': 'Luna-selected IDs from frozen RGB-D proposals',
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
        reservation = request_cost_reservation(config, prompt, payload, candidates)
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
        row['response_model'] = raw.get('model')
        row['response_status'] = raw.get('status')
        (folder / 'response.json').write_text(json_text(raw)+'\n')
        row['cost_usd'] = ledger.account(ProviderResponse(response.output_text, raw,
                                                        response.usage.input_tokens if response.usage else 0,
                                                        response.usage.output_tokens if response.usage else 0), reservation)
        manifest['ledger'] = ledger.as_dict()
        manifest_path.write_text(json_text(manifest)+'\n')
        try:
            if raw.get('status') != 'completed':
                raise ValueError('A complete structured answer is required')
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
