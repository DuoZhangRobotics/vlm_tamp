"""Check real GPU IK independence from query order and worker lifetime."""
import json
import argparse
from pathlib import Path
import numpy as np
from vlm_tamp.rekep_libero.curobo_model import export_model
from vlm_tamp.rekep_libero.curobo_client import CuroboClient

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/vlm-strategy-frozen-20261018'
CASE = SOURCE/'open_bottom_drawer-state015-heuristic'
OUT = ROOT/'results/vlm-recovery-ik-audit-20261018'

def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(OUT))
    args = parser.parse_args(); OUT = Path(args.output).resolve()
    OUT.mkdir(exist_ok=False)
    record = json.loads((CASE/'robot-chain.json').read_text())
    config = json.loads((SOURCE/'protocol.json').read_text())['configuration']['open_bottom_drawer']
    config['curobo_query_reproducible'] = True
    model = export_model(record, OUT/'model', config['command_joint_reserve_rad'])
    q = json.loads((CASE/'opened/metadata.json').read_text())['joint_positions']
    poses = [r['pose_world_xyzw'] for r in json.loads((CASE/'decisions/round-00/contact-menu.json').read_text())['candidates']]
    runs = []
    for number, order in enumerate(([1, 0, 1, 2, 1], [0, 1, 2, 1])):
        client = CuroboClient(model, config, OUT/f'worker-{number}.log')
        try:
            rows = [{'pose_index': i, **client.solve(poses[i], q)} for i in order]
            runs.append(rows)
        finally:
            client.close()
    baseline = runs[0][0]
    comparisons = []
    for worker, rows in enumerate(runs):
        for i, row in enumerate(rows):
            if row['pose_index'] != 1: continue
            comparisons.append({'worker':worker, 'query':i,
                'seed_pool_equal':row['seed_pool_sha256']==baseline['seed_pool_sha256'],
                'success_equal':row['success']==baseline['success'],
                'max_joint_difference_rad':float(np.max(np.abs(np.array(row['solutions'])-baseline['solutions'])))})
    report = {'comparisons':comparisons, 'runs':runs,
              'passed':all(r['seed_pool_equal'] and r['success_equal'] and r['max_joint_difference_rad']<1e-6 for r in comparisons),
              'physical_outcomes_used_to_select_seeds':False}
    (OUT/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed':report['passed'], 'comparisons':comparisons}), flush=True)
    if not report['passed']: raise SystemExit(1)

if __name__ == '__main__': main()
