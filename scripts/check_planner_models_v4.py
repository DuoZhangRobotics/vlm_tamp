"""Validate planner FK/Jacobian against the isolated, calibrated Panda model."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.runtime/mc-vamp-deps')]

from try_rekep_bowl_v3 import save
from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.rekep_libero.sim_client_v4 import SimClientV4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/planner-feasibility-v4/models-01')
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve previous model checks')
    output.mkdir(parents=True, exist_ok=True)
    import vamp
    with SimClientV4(log_path=output / 'simulator.log') as sim:
        sim.reset(0, 1, output / 'initial', seed=0)
        record = sim.robot_chain()
        save(output / 'robot-chain.json', record)
        chain = RobotChain(record)
        reference = chain.reference_joints
        world_robot = np.linalg.inv(np.asarray(record['calibration']['world_to_robot']))
        mc_eef = np.asarray(vamp.panda.eefk(reference), float)[0]
        eef_tcp = np.linalg.inv(mc_eef) @ np.linalg.inv(world_robot) @ pose_matrix(chain.forward(reference))
        save(output / 'mc-vamp-frames.json', {'robot_base_to_world': world_robot,
                                             'mc_eef_to_controller_tcp': eef_tcp,
                                             'quaternion_order_mc_tsr': 'wxyz followed by xyz'})
        generator = np.random.default_rng(0)
        low, high = chain.limits.T
        samples = np.vstack([reference, generator.uniform(low + .10, high - .10, size=(24, 7))])
        checks = []
        for index, q in enumerate(samples):
            isolated = sim.robot_jacobian(q)
            expected = pose_matrix(isolated['eef_pose_world_xyzw'])
            pose, jacobian = chain.forward(q, jacobian=True)
            predicted = pose_matrix(pose)
            mc_pose = world_robot @ np.asarray(vamp.panda.eefk(q), float)[0] @ eef_tcp
            finite = np.empty((6, 7))
            for column in range(7):
                step = np.eye(7)[column] * 1e-6
                a, b = pose_matrix(chain.forward(q + step)), pose_matrix(chain.forward(q - step))
                finite[:3, column] = (a[:3, 3] - b[:3, 3]) / 2e-6
                finite[3:, column] = Rotation.from_matrix(b[:3, :3].T @ a[:3, :3]).as_rotvec() / 2e-6
            checks.append({'index': index, 'joints': q,
                           'reference_pose_world_xyzw': isolated['eef_pose_world_xyzw'],
                           'poe_position_error_m': np.linalg.norm(predicted[:3, 3] - expected[:3, 3]),
                           'poe_orientation_error_rad': Rotation.from_matrix(expected[:3, :3].T @ predicted[:3, :3]).magnitude(),
                           'poe_jacobian_absolute_error': np.max(np.abs(jacobian - isolated['pose_jacobian'])),
                           'poe_finite_difference_error': np.max(np.abs(jacobian - finite)),
                           'mc_vamp_position_error_m': np.linalg.norm(mc_pose[:3, 3] - expected[:3, 3]),
                           'mc_vamp_orientation_error_rad': Rotation.from_matrix(expected[:3, :3].T @ mc_pose[:3, :3]).magnitude()})
            print(f'Checked robot configuration {index + 1}/25', flush=True)
        metrics = {key: float(max(row[key] for row in checks)) for key in checks[0]
                   if key.endswith(('_m', '_rad', '_error'))}
        result = {'configurations': len(checks), 'seed': 0, 'metrics': metrics,
                  'poe_calibration_passed': metrics['poe_position_error_m'] < 1e-8
                                             and metrics['poe_jacobian_absolute_error'] < 1e-8
                                             and metrics['poe_finite_difference_error'] < 1e-7,
                  'mc_vamp_frame_check_passed': metrics['mc_vamp_position_error_m'] < .001
                                                  and metrics['mc_vamp_orientation_error_rad'] < .005,
                  'collision_geometry_validated': False, 'physical_execution_attempted': False,
                  'simulator_object_state_used': False}
        save(output / 'configurations.json', checks)
        save(output / 'summary.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
