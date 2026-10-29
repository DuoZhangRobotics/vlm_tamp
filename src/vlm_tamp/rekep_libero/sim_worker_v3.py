"""Versioned physical LIBERO adapter with additional calibrated depth views."""

import contextlib
from functools import partial
import json
from pathlib import Path
import sys
import traceback
from types import FunctionType, MethodType

import numpy as np
from scipy.spatial.transform import Rotation

from vlm_tamp.rgbd_numericals import metric_depth, world_points
from .ik_v3 import MultiStartPandaIK
from .kinematics import osc_action, pose_matrix, rotation_error
from .sim_worker import IMAGE_SIZE, MAX_CONTROL_STEPS, SimulatorWorker


ARM_VIEW_POLICIES = {
    "arm_side": {"position_world_m": [-0.8, 0.75, 1.65],
                 "look_at_world_m": [-0.35, 0, 1.05]},
    "arm_back": {"position_world_m": [-0.95, -0.55, 1.5],
                 "look_at_world_m": [-0.35, 0, 1.0]},
    "arm_top": {"position_world_m": [-0.25, 0, 2.0],
                "look_at_world_m": [-0.25, 0, 0.9]},
    "arm_base": {"position_world_m": [-0.7, 0, 2.1],
                 "look_at_world_m": [-0.65, 0, 1.15]},
    "arm_lower_left": {"position_world_m": [-1.1, 0.55, 1.0],
                       "look_at_world_m": [-0.5, 0, 1.15]},
    "arm_lower_right": {"position_world_m": [-1.1, -0.55, 1.0],
                        "look_at_world_m": [-0.5, 0, 1.15]},
    "arm_wide_front": {"position_world_m": [-0.55, 1.5, 1.5],
                       "look_at_world_m": [-0.2, 0, 1.35]},
    "arm_wide_back": {"position_world_m": [-0.55, -1.5, 1.5],
                      "look_at_world_m": [-0.2, 0, 1.35]},
    "arm_wide_side": {"position_world_m": [-1.55, 0, 1.55],
                      "look_at_world_m": [-0.2, 0, 1.35]},
}
for _policy in ARM_VIEW_POLICIES.values():
    _policy["source"] = "fixed nominal Panda workspace camera; no object identities or scene poses"


def camera_rotation(position, look_at):
    backward = np.asarray(position, dtype=float) - np.asarray(look_at, dtype=float)
    backward /= np.linalg.norm(backward)
    # A downward vertical camera cannot use world +z as its up reference.
    up = np.array([0., 1., 0.]) if abs(backward[2]) > 0.99 else np.array([0., 0., 1.])
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(backward, right), backward])


def scaled_osc_action(current_pose, target_pose, gripper=0, servo_scale=1):
    scale = float(servo_scale)
    if not np.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("servo_scale must be finite and in (0,1]")
    control = osc_action(current_pose, target_pose, gripper)
    control[:6] *= scale
    return control


def completion_tolerances(position_tolerance=.004, orientation_tolerance=.05, posture_tolerance=.03):
    thresholds = {"position_tolerance": float(position_tolerance),
                  "orientation_tolerance": float(orientation_tolerance),
                  "posture_tolerance": float(posture_tolerance)}
    maxima = {"position_tolerance": .004, "orientation_tolerance": .05, "posture_tolerance": .03}
    for name, value in thresholds.items():
        if not np.isfinite(value) or not 0 < value <= maxima[name]:
            raise ValueError(f"{name} must be finite, positive and no greater than {maxima[name]}")
    return thresholds


def checked_nullspace_kp(value=10):
    value = float(value)
    if not np.isfinite(value) or not 0 < value <= 100:
        raise ValueError("nullspace_kp must be finite and in (0,100]")
    return value


def controller_target_reached(position_error, orientation_error, posture_error=None,
                              position_tolerance=.004, orientation_tolerance=.05, posture_tolerance=.03):
    thresholds = completion_tolerances(position_tolerance, orientation_tolerance, posture_tolerance)
    return (position_error < thresholds["position_tolerance"] and orientation_error < thresholds["orientation_tolerance"]
            and (posture_error is None or posture_error < thresholds["posture_tolerance"]))


class SimulatorWorkerV3(SimulatorWorker):
    def reset(self, task_index, state_index, output_dir, seed=0):
        response = super().reset(task_index, state_index, output_dir, seed)
        self.evaluation_contact_log = []
        return response

    def _robot_calibration(self):
        if not isinstance(self.ik_solver, MultiStartPandaIK):
            self.ik_solver = MultiStartPandaIK(self.env.sim, self.robot)
        result = super()._robot_calibration()
        # During the inherited reset the original solver is constructed before
        # observe. This helper works with that solver's exact robot samples too.
        flags = MultiStartPandaIK.contact_face_flags(
            self.ik_solver, data=self.env.sim.data._data, tool_only=True)
        result["tool_collision_contact_face_flags"] = flags.tolist()
        result["contact_face_policy"] = "inward face of configured Panda pad collision boxes only"
        prefix = self.robot.gripper.naming_prefix
        finger_ids = [self.env.sim.model.body_name2id(prefix + name) for name in ("leftfinger", "rightfinger")]
        attach_z = self.ik_solver.model.body_pos[finger_ids, 2]
        if not np.allclose(attach_z, attach_z[0], atol=1e-9):
            raise ValueError("Configured Panda finger attachments have inconsistent local z")
        result["physical_finger_attach_z_m"] = float(attach_z[0])
        result["physical_finger_attach_source"] = "known configured Panda finger body origin in gripper base"
        indices = self.ik_solver.qvel_indices
        result["arm_joint_damping_nm_s_per_rad"] = self.ik_solver.model.dof_damping[indices].tolist()
        result["arm_joint_friction_loss_nm"] = self.ik_solver.model.dof_frictionloss[indices].tolist()
        result["arm_joint_armature_kg_m2"] = self.ik_solver.model.dof_armature[indices].tolist()
        result["arm_dynamics_source"] = "configured robot arm joint parameters only"
        return result

    def _state_metadata(self):
        result = super()._state_metadata()
        result["robot_collision_point_geom_names"] = self.ik_solver.collision_point_geom_names()
        result["robot_collision_contact_face_flags"] = self.ik_solver.contact_face_flags(
            data=self.env.sim.data._data).tolist()
        result["robot_visual_volumes_world"] = self.ik_solver.visual_volumes(data=self.env.sim.data._data)
        return result

    def _set_nullspace_joints(self, target):
        if target is None:
            return None
        target = self.ik_solver._checked_joints(target)
        # update_initial_joints changes only OSC reference data. Its internal
        # forward call may renormalize an unrelated free joint by one ULP, so
        # preserve the actual integration state exactly as for camera renders.
        state = {name: getattr(self.env.sim.data._data, name).copy()
                 for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat")}
        try:
            self.robot.controller.update_initial_joints(target)
        finally:
            for name, values in state.items():
                getattr(self.env.sim.data._data, name)[:] = values
        return target

    def _set_nullspace_kp(self, value=10):
        """Bind a private official OSC callable with an explicit posture gain.

        The original bytecode, defaults and __class__ closure stay unchanged;
        its copied globals replace one official torque helper lookup. Other
        controllers, module globals and Cartesian gains are never modified.
        """
        gain = checked_nullspace_kp(value)
        controller = self.robot.controller
        original = getattr(controller, "_vlm_original_run_controller", controller.run_controller)
        if gain == 10:
            if hasattr(controller, "_vlm_original_run_controller"):
                controller.run_controller = original
            controller._vlm_nullspace_kp = gain
            return gain
        if not hasattr(controller, "_vlm_original_run_controller"):
            controller._vlm_original_run_controller = original
        function = original.__func__
        official = function.__globals__.get("nullspace_torques")
        if official is None:
            raise RuntimeError("The configured OSC lacks its official nullspace torque helper")
        globals_copy = dict(function.__globals__)
        globals_copy["nullspace_torques"] = partial(official, joint_kp=gain)
        custom = FunctionType(function.__code__, globals_copy, function.__name__,
                              function.__defaults__, function.__closure__)
        custom.__kwdefaults__ = function.__kwdefaults__
        custom.__qualname__ = function.__qualname__
        controller.run_controller = MethodType(custom, controller)
        controller._vlm_nullspace_kp = gain
        return gain

    def _evaluation_contacts(self):
        """Diagnostics only: never returned in an observation/action envelope."""
        data, model = self.env.sim.data._data, self.ik_solver.model
        prefixes = (self.robot.robot_model.naming_prefix, self.robot.gripper.naming_prefix)
        contacts = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            names = [self.env.sim.model.geom_id2name(int(value)) for value in (contact.geom1, contact.geom2)]
            is_robot = [bool(name and name.startswith(prefixes)) for name in names]
            if not any(is_robot):
                continue
            force = np.zeros(6)
            self.ik_solver.mujoco.mj_contactForce(model, data, index, force)
            contacts.append({"geom_names": names, "robot_flags": is_robot,
                             "distance_m": float(contact.dist), "position_world_m": contact.pos.tolist(),
                             "contact_frame_world": contact.frame.reshape(3, 3).tolist(),
                             "force_contact_frame_n_nm": force.tolist(), "normal_force_n": float(force[0])})
        return {"step": self.control_steps, "robot_involved_contacts": contacts,
                "source": "evaluation-only native simulator contacts; excluded from policy input"}

    def action(self, target_pose_world_xyzw, gripper=0, max_steps=20, output_dir=None, servo_scale=1,
               nullspace_joint_target=None, position_tolerance=.004, orientation_tolerance=.05,
               posture_tolerance=.03, nullspace_kp=10):
        self._require_env()
        target = pose_matrix(target_pose_world_xyzw)
        # Validate optional gain before a single physical simulator step.
        scaled_osc_action(self._eef_pose(), target_pose_world_xyzw, gripper, servo_scale)
        if not 1 <= int(max_steps) <= self.maximum_control_steps:
            raise ValueError(f"Action max_steps must be in [1,{self.maximum_control_steps}]")
        thresholds = completion_tolerances(position_tolerance, orientation_tolerance, posture_tolerance)
        gain = checked_nullspace_kp(nullspace_kp)
        nullspace_target = self._set_nullspace_joints(nullspace_joint_target)
        self._set_nullspace_kp(gain)
        executed, reached, posture_error, tcp_reached = 0, False, None, False
        criterion = f"tcp_position_lt_{thresholds['position_tolerance']}m_and_orientation_lt_{thresholds['orientation_tolerance']}rad"
        if nullspace_target is not None:
            criterion += f"_and_max_joint_error_lt_{thresholds['posture_tolerance']}rad"
        for _ in range(min(int(max_steps), self.maximum_control_steps - self.control_steps)):
            control = scaled_osc_action(self._eef_pose(), target_pose_world_xyzw, gripper, servo_scale)
            self.latest_obs, _, _, _ = self.env.step(control)
            self.control_steps += 1
            executed += 1
            actual = pose_matrix(self._eef_pose())
            position_error = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
            orientation_error = float(np.linalg.norm(rotation_error(target[:3, :3], actual[:3, :3])))
            task_success = bool(self.env.check_success())
            actual_joints = self.env.sim.data.qpos[self.robot._ref_joint_pos_indexes]
            posture_error = (None if nullspace_target is None else
                             float(np.max(np.abs(actual_joints - nullspace_target))))
            tcp_reached = controller_target_reached(position_error, orientation_error, **thresholds)
            reached = controller_target_reached(position_error, orientation_error, posture_error, **thresholds)
            self.action_log.append({"step": self.control_steps,
                                    "target_pose_world_xyzw": list(target_pose_world_xyzw),
                                    "control": control.tolist(), "eef_pose_world_xyzw": self._eef_pose().tolist(),
                                    "joint_positions": self.env.sim.data.qpos[self.robot._ref_joint_pos_indexes].tolist(),
                                    "joint_velocities": self.env.sim.data.qvel[self.robot._ref_joint_vel_indexes].tolist(),
                                    "gripper_qpos": self.env.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes].tolist(),
                                    "servo_scale": float(servo_scale),
                                    "nullspace_kp": gain,
                                    "nullspace_target_joints": None if nullspace_target is None else nullspace_target.tolist(),
                                    "controller_nullspace_reference_joints": self.robot.controller.initial_joint.tolist(),
                                    "posture_error_rad": posture_error, "reached_criterion": criterion,
                                    "completion_tolerances": thresholds,
                                    "tcp_pose_reached": tcp_reached, "controller_target_reached": reached,
                                    "position_error_m": position_error, "orientation_error_rad": orientation_error,
                                    "evaluation_task_success": task_success})
            self.evaluation_contact_log.append(self._evaluation_contacts())
            self._save_frame()
            if reached and gripper == 0:
                break
        self._save_frame(force=True)
        self._write_action_log()
        response = self._response_state()
        response.update(executed_steps=executed, reached_pose=reached, tcp_pose_reached=tcp_reached,
                        posture_error_rad=posture_error, reached_criterion=criterion,
                        completion_tolerances=thresholds, nullspace_kp=gain)
        if output_dir is not None:
            response.update(self.observe(output_dir))
        return response

    def _write_action_log(self):
        super()._write_action_log()
        if self.episode_dir:
            (self.episode_dir / "evaluation_contact_diagnostics.json").write_text(
                json.dumps({"evaluation_only": True, "excluded_from_policy_input": True,
                            "steps": getattr(self, "evaluation_contact_log", [])}, indent=2) + "\n")

    def robot_visual_meshes(self, output_dir):
        """Export exact known robot visual triangles at the CURRENT pose only."""
        self._require_env()
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        npz_path, manifest_path = output / "robot_visual_meshes.npz", output / "robot_visual_meshes.json"
        if npz_path.exists() or manifest_path.exists():
            raise FileExistsError("Robot visual mesh artifact already exists; choose a new directory")
        model, data = self.ik_solver.model, self.env.sim.data._data
        arrays, meshes = {}, []
        for index, local in enumerate(self.ik_solver._visual_volumes_local):
            geom_id = local["geom_id"]
            mesh_id = int(model.geom_dataid[geom_id])
            first_vertex, vertex_count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            first_face, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
            vertices = np.array(model.mesh_vert[first_vertex:first_vertex + vertex_count], dtype=np.float64)
            faces = np.array(model.mesh_face[first_face:first_face + face_count], dtype=np.int32)
            if not len(faces) or np.min(faces) < 0 or np.max(faces) >= len(vertices):
                raise ValueError("Configured robot visual mesh has invalid local triangle indices")
            prefix = f"geom_{index:03d}"
            keys = {"vertices_key": prefix + "_vertices_local", "faces_key": prefix + "_faces",
                    "rotation_key": prefix + "_rotation_world", "position_key": prefix + "_position_world"}
            arrays[keys["vertices_key"]] = vertices
            arrays[keys["faces_key"]] = faces
            arrays[keys["rotation_key"]] = data.geom_xmat[geom_id].reshape(3, 3).copy()
            arrays[keys["position_key"]] = data.geom_xpos[geom_id].copy()
            # Audit topology after exact-coordinate welding, without moving a
            # vertex or filling any hole. Consumers must reject open material.
            _, inverse = np.unique(vertices, axis=0, return_inverse=True)
            welded = inverse[faces]
            edges = np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]])
            sorted_edges = np.sort(edges, axis=1)
            _, inverse_edges, counts = np.unique(sorted_edges, axis=0, return_inverse=True, return_counts=True)
            winding = np.bincount(inverse_edges, weights=np.where(edges[:, 0] < edges[:, 1], 1., -1.))
            meshes.append({"index": index, "robot_geom_name": local["robot_geom_name"],
                           "robot_body_name": self.env.sim.model.body_id2name(int(model.geom_bodyid[geom_id])),
                           **keys, "vertex_count": len(vertices), "triangle_count": len(faces),
                           "watertight_topology": bool(np.all(counts == 2)),
                           "winding_consistent": bool(np.all(winding == 0)),
                           "boundary_edge_count": int(np.count_nonzero(counts == 1)),
                           "nonmanifold_edge_count": int(np.count_nonzero(counts > 2)),
                           "source": "exact configured robot visual mesh; no convexification or hole closing"})
        manifest = {"schema_version": "current-robot-visual-meshes-1.0", "meshes": meshes,
                    "frame": "local vertices @ rotation_world.T + position_world", "units": "meters",
                    "current_pose_only": True, "no_scene_geometry": True,
                    "control_steps": self.control_steps,
                    "joint_positions": data.qpos[self.ik_solver.qpos_indices].tolist(),
                    "policy": "current known robot material only; never future unknown-space clearing"}
        np.savez_compressed(npz_path, **arrays)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return {"robot_visual_meshes_artifact": {
            "npz_path": str(npz_path), "manifest_path": str(manifest_path),
            "source": "configured current robot visual meshes only", "current_pose_only": True,
            "no_scene_geometry": True, "mesh_count": len(meshes),
            "watertight_topology_count": sum(mesh["watertight_topology"] for mesh in meshes)}}

    def observe(self, output_dir):
        response = super().observe(output_dir)
        artifact = self.robot_visual_meshes(output_dir)
        with np.load(Path(output_dir) / "obs.npz") as arrays:
            coverage = self._coverage_diagnostic(arrays, response["metadata"]["cameras"])
        coverage_path = Path(output_dir).resolve() / "camera_coverage.json"
        coverage_path.write_text(json.dumps(coverage, indent=2) + "\n")
        artifact["robot_workspace_coverage_artifact"] = str(coverage_path)
        response["metadata"].update(artifact)
        (Path(output_dir).resolve() / "metadata.json").write_text(json.dumps(response["metadata"], indent=2) + "\n")
        return response

    @staticmethod
    def _coverage_diagnostic(arrays, cameras):
        # Known robot workspace and audited candidate robot geometry only.
        points = np.array([[-.7, 0, 1.1], [-.6, 0, 1.05], [-.75, .05, 1.24],
                           [-.44, .04, 1.59], [-.45, .03, 1.58], [-.43, .05, 1.6],
                           [-.45, 0, 1.6], [-.2, 0, 1.35]])
        result = {"source": "nominal configured robot workspace and audited robot samples only",
                  "points_world_m": points.tolist(), "margin_m": .003, "cameras": {}}
        for name, calibration in cameras.items():
            transform = np.asarray(calibration["camera_to_world"])
            optical = (points - transform[:3, 3]) @ transform[:3, :3]
            projected = optical @ np.asarray(calibration["intrinsic"]).T
            pixels = projected[:, :2] / projected[:, 2:]
            samples = []
            for point, camera_point, pixel in zip(points, optical, pixels):
                sample = {"point_world_m": point.tolist(), "pixel_uv": pixel.tolist(),
                          "optical_depth_m": float(camera_point[2]), "classification": "outside_frustum"}
                if (calibration["near_m"] < camera_point[2] < calibration["far_m"]
                        and np.all(pixel >= 0) and pixel[0] < IMAGE_SIZE and pixel[1] < IMAGE_SIZE):
                    u, v = np.clip(np.rint(pixel).astype(int), 0, IMAGE_SIZE - 1)
                    measured = float(arrays[name + "_depth"][v, u])
                    prior = float(arrays[name + "_robot_depth_prior"][v, u])
                    difference = measured - float(camera_point[2])
                    sample.update(scene_depth_m=measured,
                                  scene_endpoint_matches_current_robot=bool(np.isfinite(prior) and abs(prior-measured) <= 1e-5),
                                  classification="observed_free" if difference > .003 else
                                  "occluded" if difference < -.003 else "near_observed_surface")
                samples.append(sample)
            result["cameras"][name] = samples
        result["per_point_observed_free_views"] = [
            [name for name, samples in result["cameras"].items() if samples[index]["classification"] == "observed_free"]
            for index in range(len(points))]
        return result

    def _fixed_view(self, policy):
        sim = self.env.sim
        cam = sim.model.camera_name2id("agentview")
        if int(sim.model.cam_bodyid[cam]) != 0:
            raise ValueError("Fixed arm cameras require a world-parent camera")
        position, quaternion = sim.model.cam_pos[cam].copy(), sim.model.cam_quat[cam].copy()
        try:
            sim.model.cam_pos[cam] = policy["position_world_m"]
            quat = Rotation.from_matrix(camera_rotation(policy["position_world_m"],
                                                        policy["look_at_world_m"])).as_quat()
            sim.model.cam_quat[cam] = quat[[3, 0, 1, 2]]
            sim.forward()
            rgb, raw_depth = sim.render(camera_name="agentview", width=IMAGE_SIZE,
                                        height=IMAGE_SIZE, depth=True)
            calibration = self._camera_calibration("agentview")
            calibration["view_policy"] = policy
            return rgb[::-1], raw_depth[::-1], calibration
        finally:
            sim.model.cam_pos[cam] = position
            sim.model.cam_quat[cam] = quaternion
            sim.forward()

    def _sensor_arrays(self):
        # Preserve the exact existing three sensor arrays before adding views.
        arrays, cameras = super()._sensor_arrays()
        for name, policy in ARM_VIEW_POLICIES.items():
            rgb, raw_depth, calibration = self._fixed_view(policy)
            depth = metric_depth(raw_depth, calibration["near_m"], calibration["far_m"])
            arrays[name + "_rgb"] = rgb
            arrays[name + "_depth"] = depth.astype(np.float32)
            arrays[name + "_world_points"] = world_points(
                depth, calibration["intrinsic"], calibration["camera_to_world"]).astype(np.float32)
            cameras[name] = calibration
        arrays.update(self._robot_depth_priors(cameras))
        return arrays, cameras

    def _robot_depth_priors(self, cameras):
        """Project only configured current robot visual meshes into each camera.

        No environment shape, object identity, mesh or segmentation is returned.
        Actual scene depth remains the observation. These priors only identify
        coincident observed self endpoints, never free space behind the robot.
        """
        sim = self.env.sim
        model = sim.model._model
        context = sim._render_context_offscreen
        groups, render_groups = model.geom_group.copy(), context.vopt.geomgroup.copy()
        # Repeated camera-only mj_forward calls can renormalize a free-joint
        # quaternion by one floating-point unit. Preserve integration state.
        state_fields = {name: getattr(sim.data._data, name).copy()
                        for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat")}
        robot_visual_ids = [item["geom_id"] for item in self.ik_solver._visual_volumes_local]
        robot_vertices = np.concatenate([np.asarray(volume["vertices_world"])
                                         for volume in self.ik_solver.visual_volumes(data=sim.data._data)])
        arrays = {}
        try:
            model.geom_group[:] = 5
            model.geom_group[robot_visual_ids] = 4
            context.vopt.geomgroup[:] = 0
            context.vopt.geomgroup[4] = 1
            for name, calibration in cameras.items():
                if name == "frontview":
                    _, raw_depth, _ = self._frontview()
                elif name in ARM_VIEW_POLICIES:
                    _, raw_depth, _ = self._fixed_view(ARM_VIEW_POLICIES[name])
                else:
                    actual = "robot0_eye_in_hand" if name == "wrist" else name
                    _, raw_depth = sim.render(camera_name=actual, width=IMAGE_SIZE,
                                               height=IMAGE_SIZE, depth=True)
                    raw_depth = raw_depth[::-1]
                depth = metric_depth(raw_depth, calibration["near_m"], calibration["far_m"])
                transform = np.asarray(calibration["camera_to_world"])
                robot_z = ((robot_vertices - transform[:3, 3]) @ transform[:3, :3])[:, 2]
                # The OpenGL far background can quantize slightly below raw 1.
                # Bound finite hits by the exact known robot vertex z extent.
                max_robot_depth = max(0., float(np.max(robot_z))) + .001
                valid = (np.isfinite(depth) & (depth > calibration["near_m"])
                         & (depth <= max_robot_depth))
                arrays[name + "_robot_depth_prior"] = np.where(valid, depth, np.nan).astype(np.float32)
                calibration["robot_depth_prior"] = {
                    "array": name + "_robot_depth_prior", "units": "meters", "depth_convention": "optical-axis z",
                    "source": "configured robot visual meshes projected at current proprioception only",
                    "background": "NaN", "purpose": "matched observed self endpoints only; no free-space clearing",
                    "scene_geometry_used": False, "matching_tolerance_m": 1e-5,
                    "maximum_known_robot_optical_depth_m": max_robot_depth,
                }
        finally:
            model.geom_group[:] = groups
            context.vopt.geomgroup[:] = render_groups
            for name, values in state_fields.items():
                getattr(sim.data._data, name)[:] = values
        return arrays

    def fk(self, joints=None, include_volumes=False):
        self._require_env()
        pose = self.ik_solver.fk(joints)
        points, fingers = self.ik_solver.collision_points(moving_only=True)
        result = {"eef_pose_world_xyzw": pose.tolist(),
                  "robot_collision_points_world": points.tolist(),
                  "robot_collision_finger_flags": fingers.tolist(),
                  "robot_collision_points_include_fixed_base": False,
                  "robot_collision_point_geom_names": self.ik_solver.collision_point_geom_names(moving_only=True),
                  "robot_collision_contact_face_flags": self.ik_solver.contact_face_flags(
                      moving_only=True).tolist(),
                  "state_mutation": "isolated IK buffer only"}
        if include_volumes:
            result["robot_collision_volumes_world"] = [v for v in self.ik_solver.collision_volumes()
                                                        if not v["fixed"]]
            result["robot_visual_volumes_world"] = self.ik_solver.visual_volumes()
        return result

    def multiik(self, target_pose_world_xyzw, seed_joints=None, max_iterations=200,
                max_starts=12, include_volumes=False):
        self._require_env()
        best, feasible, attempts = self.ik_solver.solve_all(
            target_pose_world_xyzw, initial_joint_pos=seed_joints,
            max_iterations=max_iterations, max_starts=max_starts)
        solutions = []
        for result in feasible:
            geometry = self.fk(result.cspace_position, include_volumes=include_volumes)
            solutions.append({**result.as_dict(), **geometry})
        geometry = self.fk(best.cspace_position, include_volumes=include_volumes)
        return {**best.as_dict(), **geometry, "solutions": solutions,
                "attempts": [{"seed_index": index, **result.as_dict()} for index, result in attempts],
                "num_feasible_solutions": len(solutions),
                "num_starts": len(attempts), "method": "bounded deterministic Panda DLS multi-start"}

    def ik(self, target_pose_world_xyzw, seed_joints=None, max_iterations=120):
        return self.multiik(target_pose_world_xyzw, seed_joints, max_iterations)

    multi_ik = multiik

    def dispatch(self, request):
        if request["op"] in {"fk", "multiik", "multi_ik", "robot_visual_meshes"}:
            arguments = {key: value for key, value in request.items() if key not in {"op", "id"}}
            return getattr(self, request["op"])(**arguments)
        return super().dispatch(request)


def main():
    protocol_output = sys.stdout
    worker = SimulatorWorkerV3()
    with contextlib.redirect_stdout(sys.stderr):
        try:
            for line in sys.stdin:
                request = {}
                try:
                    request = json.loads(line)
                    response = {"ok": True, "id": request.get("id"), **worker.dispatch(request)}
                except Exception as exc:
                    traceback.print_exc(file=sys.stderr)
                    response = {"ok": False, "id": request.get("id"), "error": str(exc),
                                "error_type": type(exc).__name__}
                protocol_output.write(json.dumps(response, allow_nan=False) + "\n")
                protocol_output.flush()
                if request.get("op") == "close":
                    break
        finally:
            worker.close()


if __name__ == "__main__":
    main()
