"""Run real cuRobo GPU FK and multi-IK on a preserved drawer grasp."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'.runtime/mc-vamp-deps')]
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/curobo-development/ik-smoke')
    args = parser.parse_args()
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=False)
    folder = ROOT/'results/physical-tasks-v11-eval/goal00-state011'
    config = json.loads((ROOT/'configs/physical_tasks_curobo.json').read_text())
    robot = json.loads((folder/'robot-chain.json').read_text())
    metadata = json.loads((folder/'opened/metadata.json').read_text())
    row = next(row for row in json.loads((folder/'grasp-admission.json').read_text())['rejected_candidates']
        if row['index'] == 75 and row['variant'] == 'asset_contact_aligned_half_turn')
    import vamp
    world = vamp.Environment()
    ik = TaskIK(robot, config, lambda q: vamp.panda.validate(np.asarray(q).tolist(), world, True), output=output)
    try:
        solutions = ik.solve_many(row['tcp_pose_world_xyzw'], metadata['joint_positions'])
        result = {'solutions': solutions, 'count': len(solutions), 'curobo_gpu_inference_used': True,
                  'physical_execution_measured': False, 'environment_collision_checked': False,
                  'runtime': ik.client.metadata}
        (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'distinct_robot_valid_ik': len(solutions), 'runtime': ik.client.metadata}), flush=True)
        if not solutions:
            raise ValueError('cuRobo smoke test returned no robot-valid solution')
    finally:
        ik.close()


if __name__ == '__main__':
    main()
