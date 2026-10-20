"""Compile independent object-axis relations without another model call."""

import json
import shutil
import numpy as np

from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_axis_constraints import compile_independent_axis
from vlm_tamp.rgbd_numeric_trial import check_protocol, file_sha
from vlm_tamp.rgbd_numericals import json_text


config, protocol, records = check_protocol(ROOT / 'configs/rgbd_numeric_evaluation_fixed.json')
source = ROOT / 'results/rgbd-numerical-mini'
output = ROOT / 'results/rgbd-numerical-independent-axis'
if output.exists():
    raise FileExistsError('Preserve existing derived numerical results')
output.mkdir()
rows = list(map(json.loads, (source / 'predictions.jsonl').read_text().splitlines()))
for case, row in zip(config['cases'], rows):
    original = source / case['id']
    folder = output / case['id']
    folder.mkdir()
    for name in ('request.json', 'response.json', 'payload.json', 'original.png'):
        shutil.copyfile(original / name, folder / name)
    capture = ROOT / config['depth_dataset'] / case['example_id']
    payload = json.loads((original / 'payload.json').read_text())
    calibration = json.loads((capture / 'calibration.json').read_text())
    depth = np.load(capture / 'depth_meters.npy', allow_pickle=False)
    row['numericals'] = compile_independent_axis(row['blanks'], payload, depth, calibration, config['geometry'])
    row['additional_api_calls'] = 0
    row['source_prediction_sha256'] = file_sha(source / 'predictions.jsonl')
    (folder / 'numericals.json').write_text(json_text(row['numericals'])+'\n')
(output / 'predictions.jsonl').write_text(''.join(json_text(row)+'\n' for row in rows))
parent = ROOT / config['frozen_protocol']
config['frozen_protocol'] = 'configs/frozen_rgbd_numeric_independent_axis.json'
protocol['config'] = config
protocol['parent_protocol'] = {'path': str(parent.relative_to(ROOT)), 'sha256': file_sha(parent)}
protocol['derived_processing'] = {'mode': 'independent_axis_v1_1', 'extra_api_calls': 0,
                                  'geometry_policy_unchanged': True, 'source_blanks_unchanged': True,
                                  'reason': 'object-axis constraints do not require a grasp pose'}
protocol['implementation_hashes']['src/vlm_tamp/rgbd_axis_constraints.py'] = file_sha(ROOT / 'src/vlm_tamp/rgbd_axis_constraints.py')
protocol['preserved_artifacts']['results/rgbd-numerical-report-v1/summary.json'] = file_sha(ROOT / 'results/rgbd-numerical-report-v1/summary.json')
(ROOT / 'configs/rgbd_numeric_independent_axis.json').write_text(json_text(config)+'\n')
(ROOT / config['frozen_protocol']).write_text(json_text(protocol)+'\n')
manifest = json.loads((source / 'manifest.json').read_text())
manifest.update(config=config, derived=True, provider='derived', numerical_compiler='independent_axis_v1_1',
                source_ledger=manifest['ledger'], source_manifest=str(source / 'manifest.json'),
                source_manifest_sha256=file_sha(source / 'manifest.json'),
                ledger={'api_attempts': 0, 'estimated_usage_cost_usd': 0., 'unknown_usage_reserved_usd': 0.},
                frozen_protocol_sha256=file_sha(ROOT / config['frozen_protocol']))
(output / 'manifest.json').write_text(json_text(manifest)+'\n')
print({'derived_cases': len(rows), 'additional_api_calls': 0})
