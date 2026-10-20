"""Robot-only chain/Jacobian export layered on the unchanged v3 worker."""

import contextlib
import json
import sys
import traceback

import numpy as np

from .geometry import pose_matrix
from .sim_worker_v3 import SimulatorWorkerV3


class SimulatorWorkerV4(SimulatorWorkerV3):
    def robot_chain(self, joints=None):
        self._require_env()
        solver = self.ik_solver
        pose = solver.fk(joints)
        robot_joints = [self.env.sim.model.joint_name2id(name) for name in self.robot.robot_model.joints]
        data = solver.data
        return {'reference_joints': data.qpos[solver.qpos_indices].tolist(),
                'joint_axes_world': data.xaxis[robot_joints].tolist(),
                'joint_anchors_world_m': data.xanchor[robot_joints].tolist(),
                'reference_tcp_world': pose_matrix(pose).tolist(),
                'joint_limits': solver.limits.tolist(),
                'calibration': self._robot_calibration(),
                'scene_objects_used': False, 'live_state_mutated': False,
                'source': 'known robot revolute axes, anchors and controller TCP in isolated kinematic state'}

    def robot_jacobian(self, joints):
        self._require_env()
        solver = self.ik_solver
        pose = solver.fk(joints)
        jp, jr = np.zeros((3, solver.model.nv)), np.zeros((3, solver.model.nv))
        solver.mujoco.mj_jacSite(solver.model, solver.data, jp, jr, solver.site_id)
        rotation = pose_matrix(pose)[:3, :3]
        jacobian = np.vstack([jp[:, solver.qvel_indices], rotation.T @ jr[:, solver.qvel_indices]])
        return {'eef_pose_world_xyzw': pose.tolist(), 'pose_jacobian': jacobian.tolist(),
                'convention': 'world translation; right/body rotation', 'live_state_mutated': False}

    def dispatch(self, request):
        if request['op'] in {'robot_chain', 'robot_jacobian'}:
            return getattr(self, request['op'])(**{key: value for key, value in request.items() if key not in {'op', 'id'}})
        return super().dispatch(request)


def main():
    output = sys.stdout
    worker = SimulatorWorkerV4()
    with contextlib.redirect_stdout(sys.stderr):
        try:
            for line in sys.stdin:
                request = {}
                try:
                    request = json.loads(line)
                    response = {'ok': True, 'id': request.get('id'), **worker.dispatch(request)}
                except Exception as error:
                    traceback.print_exc(file=sys.stderr)
                    response = {'ok': False, 'id': request.get('id'), 'error': str(error),
                                'error_type': type(error).__name__}
                output.write(json.dumps(response, allow_nan=False) + '\n')
                output.flush()
                if request.get('op') == 'close':
                    break
        finally:
            worker.close()


if __name__ == '__main__':
    main()
