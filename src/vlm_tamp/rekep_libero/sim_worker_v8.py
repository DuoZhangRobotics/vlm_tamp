"""Synchronized sensors with exact measured gripper geometry for isolated FK."""

import contextlib
import json
import sys
import traceback

import numpy as np

from .sim_worker_v7 import SimulatorWorkerV7


class SimulatorWorkerV8(SimulatorWorkerV7):
    def fk(self, joints=None, include_volumes=False, gripper_qpos=None):
        self._require_env()
        measured = np.asarray(self.env.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes])
        requested = np.asarray(gripper_qpos, float) if gripper_qpos is not None else None
        is_measured = (requested is None or (requested.shape == measured.shape
            and np.isfinite(requested).all() and np.max(np.abs(requested - measured)) <= 1e-12))
        # MuJoCo's soft joint stops can have micrometre-scale overshoot. That is
        # a measured physical state, not an invalid hypothetical command. Do
        # not clamp it or silently move the collision geometry. V7 copies this
        # exact current state into its isolated buffer when qpos is omitted.
        result = super().fk(joints, include_volumes, None if is_measured else gripper_qpos)
        result['gripper_geometry_source'] = 'exact current proprioception' if is_measured else 'bounded hypothetical gripper posture'
        result['measured_gripper_qpos_clamped'] = False
        return result


def main():
    output, worker = sys.stdout, SimulatorWorkerV8()
    with contextlib.redirect_stdout(sys.stderr):
        try:
            for line in sys.stdin:
                request = {}
                try:
                    request = json.loads(line)
                    response = {'ok': True, 'id': request.get('id'), **worker.dispatch(request)}
                except Exception as error:
                    traceback.print_exc(file=sys.stderr)
                    response = {'ok': False, 'id': request.get('id'), 'error': str(error), 'error_type': type(error).__name__}
                output.write(json.dumps(response, allow_nan=False) + '\n')
                output.flush()
                if request.get('op') == 'close':
                    break
        finally:
            worker.close()


if __name__ == '__main__':
    main()
