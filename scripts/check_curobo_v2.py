"""GPU integration checks for the calibrated cuRobo v2 adapter, without motion."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'.runtime/mc-vamp-deps')]
import numpy as np
from vlm_tamp.coupled_storage import read_json, write_json
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/compact-luna-practice-20261022/practice-v1/attempt01')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = read_json(args.source/'007-arc/scene-reference.json')['configuration']
    config['curobo_backend'] = 'v2'
    record = read_json(args.source/'robot-chain.json')
    started = time.perf_counter()
    ik = TaskIK(record, config, lambda q: True, output=args.output)
    rows = []
    try:
        setup = time.perf_counter()-started
        q = ik.chain.reference_joints.copy()
        rng = np.random.default_rng(14)
        targets = [q, q, q, *[np.clip(q+rng.normal(0, .15, 7),
            ik.chain.limits[:, 0]+.03, ik.chain.limits[:, 1]-.03) for _ in range(5)]]
        for target in targets:
            before = time.perf_counter()
            solutions = ik.solve_many(ik.chain.forward(target), q)
            row = {'query': len(rows), 'solutions': len(solutions),
                   'wall_seconds': time.perf_counter()-before,
                   'ik_seconds': ik.events[-1]['elapsed_seconds']}
            rows.append(row)
            print(json.dumps(row), flush=True)
        result = {'accepted': all(row['solutions'] for row in rows),
                  'initialization_seconds': setup, 'runtime': ik.client.metadata, 'queries': rows,
                  'robot_motion_issued': False, 'self_collision_check_in_this_kinematic_test': False}
        write_json(args.output/'result.json', result)
        if not result['accepted']:
            raise RuntimeError('At least one reachable target failed v2 IK')
    finally:
        ik.close()


if __name__ == '__main__':
    main()
