"""Isolated hypothetical arm/gripper geometry, with unchanged physical dynamics."""

import contextlib
import json
import sys
import traceback

import numpy as np

from .kinematics import matrix_pose
from .sim_worker import IMAGE_SIZE
from .sim_worker_v3 import ARM_VIEW_POLICIES
from .sim_worker_v4 import SimulatorWorkerV4
from vlm_tamp.rgbd_numericals import metric_depth, world_points


class SimulatorWorkerV7(SimulatorWorkerV4):
    def _sensor_arrays(self):
        """Capture depth, calibration and robot prior at one physical state.

        mj_step updates qpos before all derived camera/geometry transforms have
        been refreshed. The original first two images could precede the fixed
        views' mj_forward, causing isolated robot pixels to look like obstacles.
        Direct calibrated renders after forward remove that frame mismatch.
        """
        sim = self.env.sim
        sim.forward()
        arrays, cameras = {}, {}
        for logical, actual in [('agentview', 'agentview'), ('wrist', 'robot0_eye_in_hand')]:
            rgb, raw = sim.render(camera_name=actual, width=IMAGE_SIZE, height=IMAGE_SIZE, depth=True)
            calibration = self._camera_calibration(actual)
            depth = metric_depth(raw[::-1], calibration['near_m'], calibration['far_m'])
            arrays[logical + '_rgb'] = rgb[::-1]
            arrays[logical + '_depth'] = depth.astype(np.float32)
            arrays[logical + '_world_points'] = world_points(depth, calibration['intrinsic'], calibration['camera_to_world']).astype(np.float32)
            cameras[logical] = calibration
        for name in ['frontview', *ARM_VIEW_POLICIES]:
            rgb, raw, calibration = self._frontview() if name == 'frontview' else self._fixed_view(ARM_VIEW_POLICIES[name])
            depth = metric_depth(raw, calibration['near_m'], calibration['far_m'])
            arrays[name + '_rgb'] = rgb
            arrays[name + '_depth'] = depth.astype(np.float32)
            arrays[name + '_world_points'] = world_points(depth, calibration['intrinsic'], calibration['camera_to_world']).astype(np.float32)
            cameras[name] = calibration
        arrays.update(self._robot_depth_priors(cameras))
        for calibration in cameras.values():
            calibration['capture_policy'] = 'fresh direct render and calibration after current-state kinematic refresh'
            calibration['capture_control_steps'] = self.control_steps
            calibration['capture_simulation_time_s'] = float(sim.data.time)
        return arrays, cameras

    def _robot_depth_priors(self, cameras):
        """Render every known robot part visible in the normal sensor view.

        Include configured contact-pad boxes when the normal camera enables
        their render group. No unrendered collision proxy is inserted into the
        sensor prior, and scene depth matching is not widened to millimetres.
        """
        sim = self.env.sim
        model, context = sim.model._model, sim._render_context_offscreen
        groups, render_groups = model.geom_group.copy(), context.vopt.geomgroup.copy()
        fields = {name: getattr(sim.data._data, name).copy()
                  for name in ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat')}
        prefixes = (self.robot.robot_model.naming_prefix, self.robot.gripper.naming_prefix)
        visible = [index for index, name in enumerate(sim.model.geom_names)
                   if name and name.startswith(prefixes) and render_groups[int(groups[index])]]
        names = [sim.model.geom_id2name(index) for index in visible]
        geometry = self.ik_solver.visual_volumes(data=sim.data._data) + self.ik_solver.collision_volumes(data=sim.data._data)
        vertices = np.concatenate([v['vertices_world'] for v in geometry if v['type'] == 'convex'])
        arrays = {}
        try:
            model.geom_group[:] = 5
            model.geom_group[visible] = 4
            context.vopt.geomgroup[:] = 0
            context.vopt.geomgroup[4] = 1
            for name, calibration in cameras.items():
                if name == 'frontview':
                    _, raw, _ = self._frontview()
                elif name in ARM_VIEW_POLICIES:
                    _, raw, _ = self._fixed_view(ARM_VIEW_POLICIES[name])
                else:
                    actual = 'robot0_eye_in_hand' if name == 'wrist' else name
                    _, raw = sim.render(camera_name=actual, width=IMAGE_SIZE, height=IMAGE_SIZE, depth=True)
                    raw = raw[::-1]
                depth = metric_depth(raw, calibration['near_m'], calibration['far_m'])
                transform = np.asarray(calibration['camera_to_world'])
                extent = max(0., float(np.max(((vertices - transform[:3, 3]) @ transform[:3, :3])[:, 2]))) + .001
                valid = np.isfinite(depth) & (depth > calibration['near_m']) & (depth <= extent)
                arrays[name + '_robot_depth_prior'] = np.where(valid, depth, np.nan).astype(np.float32)
                calibration['robot_depth_prior'] = {
                    'array': name + '_robot_depth_prior', 'units': 'meters', 'depth_convention': 'optical-axis z',
                    'source': 'configured robot rendered geometry at current proprioception only',
                    'scene_geometry_used': False, 'simulator_instance_segmentation_used': False,
                    'rendered_robot_geometry_names': names, 'matching_tolerance_m': .0001,
                    'purpose': 'same-pixel observed self endpoints only; no free-space clearing',
                    'background': 'NaN'}
        finally:
            model.geom_group[:] = groups
            context.vopt.geomgroup[:] = render_groups
            for name, values in fields.items():
                getattr(sim.data._data, name)[:] = values
        return arrays

    def fk(self, joints=None, include_volumes=False, gripper_qpos=None):
        self._require_env()
        solver = self.ik_solver
        solver.fk(joints)  # Copies live proprioception to an isolated buffer.
        if gripper_qpos is not None:
            values = np.asarray(gripper_qpos, float)
            indices = np.asarray(self.robot._ref_gripper_joint_pos_indexes, int)
            joint_ids = [self.env.sim.model.joint_name2id(name) for name in self.robot.gripper.joints]
            limits = np.asarray(solver.model.jnt_range[joint_ids])
            if values.shape != indices.shape or not np.isfinite(values).all():
                raise ValueError('Expected one finite hypothetical position per configured gripper joint')
            if ((values < limits[:, 0] - 1e-8) | (values > limits[:, 1] + 1e-8)).any():
                raise ValueError('Hypothetical gripper positions exceed configured joint limits')
            solver.data.qpos[indices] = values
            solver.mujoco.mj_forward(solver.model, solver.data)
        transform = np.eye(4)
        transform[:3, :3] = solver.data.site_xmat[solver.site_id].reshape(3, 3)
        transform[:3, 3] = solver.data.site_xpos[solver.site_id]
        result = {'eef_pose_world_xyzw': matrix_pose(transform).tolist(),
                  'gripper_qpos': solver.data.qpos[self.robot._ref_gripper_joint_pos_indexes].tolist(),
                  'state_mutation': 'isolated robot kinematic buffer only', 'live_state_mutated': False,
                  'scene_geometry_used': False}
        if include_volumes:
            result['robot_collision_volumes_world'] = [v for v in solver.collision_volumes() if not v['fixed']]
        return result


def main():
    output, worker = sys.stdout, SimulatorWorkerV7()
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
