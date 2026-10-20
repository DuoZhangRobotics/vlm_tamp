"""Physical-state smoke test for measured gripper overshoot and isolated FK."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from try_rekep_bowl_v3 import save
from vlm_tamp.rekep_libero.sim_client_v8 import SimClientV8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/contact-policy-v8-development/gripper-geometry-smoke')
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError('Preserve validation evidence')
    output.mkdir(parents=True)
    sim = SimClientV8(log_path=output / 'simulator.log')
    try:
        state = sim.reset(8, 11, output / 'initial', seed=0)
        state = sim.action(state['metadata']['eef_pose_world_xyzw'], gripper=-1, max_steps=60)
        reference = ROOT / 'results/contact-policy-v8-development/pilot-01/goal08-state011/command-log.json'
        commands = json.loads(reference.read_text())
        for command in commands:
            state = sim.action(command['requested_pose_world_xyzw'], gripper=0,
                max_steps=command['executed_steps'], nullspace_joint_target=command['requested_joint_posture'],
                nullspace_kp=80)
        before = sim.observe(output / 'before')
        measured = before['metadata']['gripper_qpos']
        q = np.asarray(before['metadata']['joint_positions'])
        q[0] += .001
        geometry = sim.fk(q.tolist(), include_volumes=True, gripper_qpos=measured)
        invalid_rejected = False
        try:
            sim.fk(q.tolist(), gripper_qpos=[.06, -.06])
        except RuntimeError as error:
            invalid_rejected = 'exceed configured joint limits' in str(error)
        after = sim.observe(output / 'after')
        checks = {
            'current_measured_gripper_geometry_accepted': geometry['gripper_geometry_source'] == 'exact current proprioception',
            'measured_geometry_not_clamped': not geometry['measured_gripper_qpos_clamped'] and np.array_equal(geometry['gripper_qpos'], measured),
            'unmeasured_out_of_limit_hypothesis_rejected': invalid_rejected,
            'live_arm_state_unchanged': np.array_equal(before['metadata']['joint_positions'], after['metadata']['joint_positions']),
            'live_gripper_state_unchanged': np.array_equal(measured, after['metadata']['gripper_qpos']),
            'live_tcp_unchanged': np.allclose(before['metadata']['eef_pose_world_xyzw'], after['metadata']['eef_pose_world_xyzw'], atol=1e-12, rtol=0),
            'no_control_steps_in_kinematic_queries': before['control_steps'] == after['control_steps'],
            'full_moving_robot_geometry_returned': len(geometry['robot_collision_volumes_world']) == 11}
        result = {'checks': checks, 'all_checks_passed': all(checks.values()), 'measured_gripper_qpos': measured,
                  'overshoot_beyond_nominal_m': max(0., measured[0] - .04, -measured[1] - .04),
                  'scope': 'development smoke test, not a task-success episode', 'new_api_calls': 0}
        save(output / 'result.json', result)
        print(json.dumps(result))
        if not result['all_checks_passed']:
            raise ValueError('Measured robot geometry validation failed')
    finally:
        sim.close()


if __name__ == '__main__':
    main()
