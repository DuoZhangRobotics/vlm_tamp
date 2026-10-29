"""Joint servo for coupled planning, layered on the frozen sensor worker.

The official OSC remains active for reset and initial perception.
An explicit operation changes only the controller and its action bookkeeping;
all subsequent arm motion is produced by normal ``env.step`` dynamics.
"""

import contextlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from .rekep_libero.kinematics import pose_matrix, rotation_error
from .rekep_libero.sim_worker import MAX_CONTROL_STEPS
from .rekep_libero.sim_worker_articulated import ArticulatedWorker


JOINT_DELTA_SCALE_RAD = .05
REFERENCE_MODES = {'measured_delta','slew_limited'}


def slew_limited_reference(target, previous, limits, maximum_speed=1., control_hz=20.):
    target=checked_joint_target(target,limits);previous=np.asarray(previous,float)
    if previous.shape!=(7,) or not np.isfinite(previous).all():raise ValueError('Finite previous joint reference required')
    if not np.isfinite(maximum_speed) or not 0<maximum_speed<=1:raise ValueError('Reference speed must be in (0,1] rad/s')
    step=maximum_speed/control_hz
    return np.clip(previous+np.clip(target-previous,-step,step),np.asarray(limits)[:,0],np.asarray(limits)[:,1])


def checked_joint_target(target, limits):
    target = np.asarray(target, dtype=float)
    bounds = np.asarray(limits, dtype=float)
    if target.shape != (7,) or bounds.shape != (7, 2) or not np.isfinite(target).all():
        raise ValueError('Expected seven finite arm joints and seven joint-limit pairs')
    if np.any(target < bounds[:, 0]) or np.any(target > bounds[:, 1]):
        raise ValueError('Joint target exceeds configured Panda limits')
    return target


def normalized_joint_command(target, measured, gripper):
    measured = np.asarray(measured, dtype=float)
    if measured.shape != (7,) or not np.isfinite(measured).all():
        raise ValueError('Expected seven finite measured arm joints')
    if gripper not in (-1, 0, 1):
        raise ValueError('Gripper must be -1 open, 0 hold, or +1 close')
    delta = np.asarray(target, dtype=float) - measured
    clipped = np.clip(delta, -JOINT_DELTA_SCALE_RAD, JOINT_DELTA_SCALE_RAD)
    return np.r_[clipped / JOINT_DELTA_SCALE_RAD, float(gripper)], measured + clipped


class CoupledWorker(ArticulatedWorker):
    def __init__(self):
        super().__init__()
        self.joint_tracking_enabled = False
        self.joint_reference_mode='measured_delta'
        self._pending_joint_reference=None
        self._joint_reference=None
        self.controller_changes = []

    def reset(self, *args, **kwargs):
        self.joint_tracking_enabled = False
        self.controller_changes = []
        return super().reset(*args, **kwargs)

    def _robot_calibration(self):
        result = super()._robot_calibration()
        result['execution_controller'] = 'JOINT_POSITION' if self.joint_tracking_enabled else 'OSC_POSE'
        if self.joint_tracking_enabled:
            result.pop('osc_position_scale_m', None)
            result.pop('osc_rotation_scale_rad', None)
            result['joint_delta_scale_rad'] = JOINT_DELTA_SCALE_RAD
        return result

    def _state_metadata(self):
        result = super()._state_metadata()
        result['joint_velocities'] = np.asarray(
            self.env.sim.data.qvel[self.robot._ref_joint_vel_indexes]).tolist()
        result['execution_controller'] = 'JOINT_POSITION' if self.joint_tracking_enabled else 'OSC_POSE'
        if self.joint_tracking_enabled:
            result['joint_servo'] = {
                'action_convention': 'official absolute joint goal with reference slew limiting' if self.joint_reference_mode=='slew_limited' else 'legacy measured-relative normalized delta',
                'reference_mode':self.joint_reference_mode,'maximum_reference_speed_rad_s':1.,
                'joint_delta_scale_rad': JOINT_DELTA_SCALE_RAD,
                'kp': np.asarray(self.robot.controller.kp).tolist(),
                'kd': np.asarray(self.robot.controller.kd).tolist(),
                'absolute_reference_is_not_state_assignment': True,
            }
        return result

    def enable_joint_tracking(self, kp=150.,reference_mode='measured_delta'):
        self._require_env()
        if self.joint_tracking_enabled:
            raise ValueError('Joint tracking is already enabled')
        if not np.isfinite(kp) or not 0<float(kp)<=300:
            raise ValueError('Joint tracking stiffness must be in (0,300]')
        if reference_mode not in REFERENCE_MODES:raise ValueError('Unknown joint reference mode')
        from robosuite.controllers import load_controller_config
        from robosuite.utils.buffers import DeltaBuffer

        # SingleArm._load_controller only constructs the official controller;
        # Robot.reset would assign qpos and is deliberately never called here.
        data = self.env.sim.data
        fields = ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat')
        before = {name: np.asarray(getattr(data, name)).copy() for name in fields}
        previous = self.robot.controller
        previous_config = self.robot.controller_config
        previous_actions = self.robot.recent_actions
        previous_dim = int(self.env.env._action_dim)
        config = load_controller_config(default_controller='JOINT_POSITION')
        config.update(input_min=-1, input_max=1, output_min=-JOINT_DELTA_SCALE_RAD,
                      output_max=JOINT_DELTA_SCALE_RAD, interpolation=None,kp=float(kp))
        try:
            self.robot.controller_config = config
            self.robot._load_controller()
            self.robot.controller.update_base_pose(self.robot.base_pos, self.robot.base_ori)
            self.robot.controller.reset_goal()
            self.env.env._action_dim = sum(robot.action_dim for robot in self.env.robots)
            self.robot.recent_actions = DeltaBuffer(dim=self.robot.action_dim)
            if self.robot.controller.control_dim != 7 or self.env.env.action_dim != 8:
                raise RuntimeError('Official joint controller did not expose Panda7 plus gripper')
            changes = {name: (float(np.max(np.abs(np.asarray(getattr(data, name)) - values)))
                             if values.size else 0.) for name, values in before.items()}
            # mj_forward may normalize a free quaternion at floating-point
            # roundoff. No field is assigned or restored by this switch.
            if any(value > 1e-12 for value in changes.values()):
                raise RuntimeError(f'Controller construction unexpectedly changed integration state: {changes}')
        except Exception:
            self.robot.controller = previous
            self.robot.controller_config = previous_config
            self.robot.recent_actions = previous_actions
            self.env.env._action_dim = previous_dim
            raise
        self.joint_tracking_enabled = True
        self.joint_reference_mode=reference_mode
        self._joint_reference=np.asarray(self.robot.controller.goal_qpos).copy()
        ordinary_set_goal=self.robot.controller.set_goal
        def set_command_reference(action,set_qpos=None):
            reference=self._pending_joint_reference
            return ordinary_set_goal(action,set_qpos=reference if reference is not None else set_qpos)
        # Use the official controller's absolute goal API. This assigns a
        # desired position only; qpos/qvel and physical contacts are untouched.
        self.robot.controller.set_goal=set_command_reference
        record = {'control_step': self.control_steps, 'from': 'OSC_POSE', 'to': 'JOINT_POSITION',
                  'reference_mode':reference_mode,'maximum_reference_speed_rad_s':1.,
                  'robot_action_dim': self.robot.action_dim, 'environment_action_dim': self.env.env.action_dim,
                  'joint_delta_scale_rad': JOINT_DELTA_SCALE_RAD,
                  'kp': np.asarray(self.robot.controller.kp).tolist(),
                  'kd': np.asarray(self.robot.controller.kd).tolist(),
                  'integration_state_max_changes': changes, 'physical_steps_during_switch': 0,
                  'state_assignment': False, 'assisted_attachment': False,
                  'source': 'official robosuite JointPositionController, same simulation and robot'}
        self.controller_changes.append(record)
        if self.episode_dir:
            (Path(self.episode_dir) / 'controller-changes.json').write_text(
                json.dumps(self.controller_changes, indent=2) + '\n')
        return {**self._response_state(), 'controller_change': record}

    def set_joint_reference_mode(self,mode):
        self._require_env()
        if not self.joint_tracking_enabled or mode not in REFERENCE_MODES:
            raise ValueError('Enable joint tracking and select a supported reference mode')
        previous=self.joint_reference_mode
        data=self.env.sim.data
        before={name:np.asarray(getattr(data,name)).copy() for name in ('qpos','qvel','act','ctrl')}
        self.joint_reference_mode=mode
        self._joint_reference=np.clip(np.asarray(self.robot.controller.goal_qpos).copy(),self.ik_solver.limits[:,0],self.ik_solver.limits[:,1])
        changes={name:float(np.max(np.abs(np.asarray(getattr(data,name))-value))) if value.size else 0. for name,value in before.items()}
        if any(changes.values()):raise RuntimeError('Reference mode change altered integration state')
        record={'control_step':self.control_steps,'from_reference_mode':previous,'reference_mode':mode,
            'state_assignment':False,'physical_steps_during_switch':0,'integration_state_max_changes':changes,
            'controller_gains_changed':False,'actuator_limits_changed':False,'physical_contacts_changed':False}
        self.controller_changes.append(record)
        if self.episode_dir:(Path(self.episode_dir)/'controller-changes.json').write_text(json.dumps(self.controller_changes,indent=2)+'\n')
        return {**self._response_state(),'reference_mode_change':record}

    def action(self, *args, **kwargs):
        if self.joint_tracking_enabled:
            raise ValueError('OSC action is disabled after controller switch; use joint_action')
        return super().action(*args, **kwargs)

    def joint_action(self, target_joints, gripper=1, max_steps=1, output_dir=None):
        self._require_env()
        if not self.joint_tracking_enabled:
            raise ValueError('Call enable_joint_tracking before joint_action')
        target = checked_joint_target(target_joints, self.ik_solver.limits)
        if isinstance(max_steps, bool) or int(max_steps) != max_steps or not 1 <= int(max_steps) <= self.maximum_control_steps:
            raise ValueError(f'Joint action max_steps must be an integer in [1,{self.maximum_control_steps}]')
        measured = np.asarray(self.env.sim.data.qpos[self.robot._ref_joint_pos_indexes])
        normalized_joint_command(target, measured, gripper)
        target_pose = self.ik_solver.fk(target)
        target_transform = pose_matrix(target_pose)
        executed, posture_error = 0, float(np.max(np.abs(target - measured)))
        for _ in range(min(int(max_steps), self.maximum_control_steps - self.control_steps)):
            before = np.asarray(self.env.sim.data.qpos[self.robot._ref_joint_pos_indexes]).copy()
            previous_reference=np.asarray(self._joint_reference if self._joint_reference is not None else before).copy()
            if self.joint_reference_mode=='slew_limited':
                bounded_goal=slew_limited_reference(target,previous_reference,self.ik_solver.limits)
                control=np.r_[np.zeros(7),float(gripper)]
                self._pending_joint_reference=bounded_goal.copy()
            else:control,bounded_goal=normalized_joint_command(target,before,gripper)
            try:self.latest_obs, _, _, _ = self.env.step(control)
            finally:self._pending_joint_reference=None
            self._joint_reference=bounded_goal.copy()
            self.control_steps += 1
            executed += 1
            actual_joints = np.asarray(self.env.sim.data.qpos[self.robot._ref_joint_pos_indexes]).copy()
            actual_pose = self._eef_pose()
            actual_transform = pose_matrix(actual_pose)
            posture_error = float(np.max(np.abs(actual_joints - target)))
            controller_goal = np.asarray(self.robot.controller.goal_qpos).copy()
            if controller_goal.shape != (7,) or not np.allclose(controller_goal, bounded_goal, rtol=0, atol=1e-10):
                raise RuntimeError('Official joint controller goal disagrees with commanded bounded reference')
            self.action_log.append({
                'step': self.control_steps, 'execution_controller': 'JOINT_POSITION',
                'target_joints': target.tolist(), 'joint_reference_before_step': previous_reference.tolist(),
                'measured_joints_before_step':before.tolist(),'joint_reference_mode':self.joint_reference_mode,
                'controller_goal_joints': controller_goal.tolist(),
                'reference_clipped': bool(np.max(np.abs(target-bounded_goal))>1e-10),
                'target_pose_world_xyzw': np.asarray(target_pose).tolist(),
                'control': control.tolist(), 'eef_pose_world_xyzw': np.asarray(actual_pose).tolist(),
                'joint_positions': actual_joints.tolist(),
                'joint_velocities': np.asarray(self.env.sim.data.qvel[self.robot._ref_joint_vel_indexes]).tolist(),
                'gripper_qpos': np.asarray(self.env.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes]).tolist(),
                'posture_error_rad': posture_error,
                'position_error_m': float(np.linalg.norm(target_transform[:3, 3] - actual_transform[:3, 3])),
                'orientation_error_rad': float(np.linalg.norm(rotation_error(target_transform[:3, :3], actual_transform[:3, :3]))),
                'evaluation_task_success': bool(self.env.check_success()),
                'completion_is_not_trajectory_validity': True,
            })
            self.evaluation_contact_log.append(self._evaluation_contacts())
            self._save_frame()
        self._save_frame(force=True)
        self._write_action_log()
        response = self._response_state()
        response.update(executed_steps=executed, posture_error_rad=posture_error,
                        target_joints=target.tolist(), execution_controller='JOINT_POSITION')
        if output_dir is not None:
            response.update(self.observe(output_dir))
        return response

    def dispatch(self, request):
        if request['op'] in {'enable_joint_tracking', 'set_joint_reference_mode', 'joint_action'}:
            arguments = {key: value for key, value in request.items() if key not in {'op', 'id'}}
            return getattr(self, request['op'])(**arguments)
        return super().dispatch(request)


def main():
    protocol, worker = sys.stdout, CoupledWorker()
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
                protocol.write(json.dumps(response, allow_nan=False) + '\n')
                protocol.flush()
                if request.get('op') == 'close':
                    break
        finally:
            worker.close()


if __name__ == '__main__':
    main()
