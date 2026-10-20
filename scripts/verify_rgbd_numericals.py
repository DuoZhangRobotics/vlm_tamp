"""Replay actual saved replies, depth compilation, metrics and preserved-file checks."""

import json
import numpy as np

from vlm_tamp.dataset import read_records
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_numeric_evaluation import capture_checks, grade_case
from vlm_tamp.rgbd_numeric_trial import check_protocol, file_sha, sdk_arguments
from vlm_tamp.rgbd_numericals import compile_numericals, json_text, template, validate_blanks
from vlm_tamp.rgbd_axis_constraints import compile_independent_axis


config, protocol, records = check_protocol(ROOT / 'configs/rgbd_numeric_independent_axis.json')
trial, report = ROOT / 'results/rgbd-numerical-independent-axis', ROOT / 'results/rgbd-numerical-report'
manifest = json.loads((trial / 'manifest.json').read_text())
usage = manifest.get('source_ledger', manifest['ledger'])
assert manifest['config'] == config
assert manifest['frozen_protocol_sha256'] == file_sha(ROOT / config['frozen_protocol'])
assert file_sha(manifest['source_manifest']) == manifest['source_manifest_sha256']
summary = json.loads((report / 'summary.json').read_text())
rows = list(map(json.loads, (trial / 'predictions.jsonl').read_text().splitlines()))
_, root = read_records(ROOT / config['dataset'])
assert len(rows) == 6 and not manifest['offline_mock'] and not manifest['dry_run']
assert not manifest['model_input_inventory'] and not manifest['depth_sent_to_vlm']
input_tokens, output_tokens, checks = 0, 0, []
for index, (case, row) in enumerate(zip(config['cases'], rows)):
    folder = trial / case['id']
    capture = ROOT / config['depth_dataset'] / case['example_id']
    image = (capture / 'rgb.png').read_bytes()
    payload = template(records[case['example_id']]['model_inputs']['instruction'], case['upright'])
    assert json.loads((folder / 'request.json').read_text()) == sdk_arguments(config, payload, image, manifest['prompt_text'])
    raw = json.loads((folder / 'response.json').read_text())
    assert raw['status'] == 'completed'
    assert raw['model'] == config['model']
    text = ''.join(item['text'] for block in raw['output'] if block['type'] == 'message'
                   for item in block['content'] if item['type'] == 'output_text')
    blanks = validate_blanks(json.loads(text))
    assert blanks == row['blanks']
    depth = np.load(capture / 'depth_meters.npy', allow_pickle=False)
    cloud = np.load(capture / 'world_points.npy', allow_pickle=False)
    calibration = json.loads((capture / 'calibration.json').read_text())
    compiler = compile_independent_axis if manifest.get('derived') else compile_numericals
    numeric = compiler(blanks, payload, depth, calibration, config['geometry'])
    assert numeric == row['numericals'] == json.loads((folder / 'numericals.json').read_text())
    metric = grade_case(records[case['example_id']], root, case, row)
    assert metric == summary['tasks'][index]['metrics']
    assert capture_checks(depth, cloud, calibration)['passed']
    checks.extend(metric['relation_checks'])
    input_tokens += raw['usage']['input_tokens']
    output_tokens += raw['usage']['output_tokens']
assert input_tokens == usage['input_tokens']
assert output_tokens == usage['output_tokens']
cost = (input_tokens*.75+output_tokens*4.5)/1e6
assert abs(cost-usage['estimated_usage_cost_usd']) < 1e-12
assert usage['api_attempts'] == 6 and cost <= .1
assert usage['unknown_usage_reserved_usd'] == 0
assert manifest['ledger']['api_attempts'] == 0 and manifest['derived']
assert checks and all(check['passed'] for check in checks)
text = (report / 'index.html').read_text()
assert text.count('<article ') == 6 and text.count('class="cloud"') == 6
for case in config['cases']:
    for suffix in ('-points.png', '-depth.png', '-numericals.json'):
        assert (report / 'assets' / (case['id']+suffix)).is_file()
verification = {'saved_real_api_replies_replayed': 6, 'depth_captures_checked': 4,
                'numerical_relation_self_checks': len(checks),
                'maximum_jacobian_error': max(check['maximum_jacobian_error'] for check in checks),
                'preserved_artifacts': len(protocol['preserved_artifacts']),
                'estimated_new_usage_usd': cost, 'report_assets_checked': 18,
                'joint_space_or_execution_verified': False, 'passed': True}
(report / 'verification.json').write_text(json_text(verification)+'\n')
print(verification)
