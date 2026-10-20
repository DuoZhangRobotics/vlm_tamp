"""Robot-only transforms and isolated seven-joint Panda inverse kinematics.

All poses use world xyz + quaternion xyzw.  No environment object, joint or
segmentation identity is used by this module.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Expected finite xyz + quaternion xyzw pose")
    norm = np.linalg.norm(pose[3:])
    if norm < 1e-9:
        raise ValueError("Pose quaternion must be nonzero")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(pose[3:] / norm).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def matrix_pose(transform):
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Expected finite homogeneous transform")
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()]


def rotation_error(target_rotation, current_rotation):
    """World-frame rotation increment, matching robosuite OSC left update."""
    return Rotation.from_matrix(np.asarray(target_rotation) @ np.asarray(current_rotation).T).as_rotvec()


def osc_action(current_pose, target_pose, gripper=0.0):
    current, target = pose_matrix(current_pose), pose_matrix(target_pose)
    if gripper not in (-1, 0, 1):
        raise ValueError("Gripper must be -1 open, 0 hold, or +1 close")
    position = (target[:3, 3] - current[:3, 3]) / 0.05
    rotation = rotation_error(target[:3, :3], current[:3, :3]) / 0.5
    return np.r_[np.clip(position, -1, 1), np.clip(rotation, -1, 1), gripper]


def tcp_to_rekep():
    """ReKep +x approach and +y closing correspond to Panda +z and +x."""
    transform = np.eye(4)
    transform[:3, :3] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    return transform


@dataclass
class IKResult:
    success: bool
    num_descents: int
    position_error: float
    orientation_error: float
    cspace_position: np.ndarray

    def as_dict(self):
        return {
            "success": bool(self.success), "num_descents": int(self.num_descents),
            "position_error": float(self.position_error),
            "orientation_error": float(self.orientation_error),
            "cspace_position": self.cspace_position.tolist(),
        }


class PandaIK:
    """Damped least squares using a separate MuJoCo data buffer.

    The model is read-only and the live simulation state is never changed.
    The only Jacobian and pose queried belong to the robot's known TCP.
    """

    def __init__(self, sim, robot):
        import mujoco

        self.mujoco = mujoco
        self.sim = sim
        self.model = sim.model._model
        self.data = mujoco.MjData(self.model)
        self.qpos_indices = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        self.qvel_indices = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
        if len(self.qpos_indices) != 7 or len(self.qvel_indices) != 7:
            raise ValueError("Panda IK requires exactly seven arm joints")
        self.site_id = int(robot.eef_site_id)
        joint_ids = [sim.model.joint_name2id(name) for name in robot.robot_model.joints]
        self.arm_joint_body_ids = set(int(value) for value in self.model.jnt_bodyid[joint_ids])
        self.limits = np.asarray(sim.model.jnt_range[joint_ids], dtype=float).copy()
        self.jac_position = np.zeros((3, self.model.nv))
        self.jac_rotation = np.zeros((3, self.model.nv))
        self.robot = robot
        self.collision_samples = self._robot_collision_samples()

    def _robot_collision_samples(self):
        """Sample only configured robot collision geoms, never scene geoms."""
        samples = []
        self._collision_volumes_local = []
        robot_prefix = self.robot.robot_model.naming_prefix
        gripper_prefix = self.robot.gripper.naming_prefix
        for geom_id, name in enumerate(self.sim.model.geom_names):
            if not name or not name.startswith((robot_prefix, gripper_prefix)):
                continue
            if int(self.model.geom_group[geom_id]) != 0:
                continue
            kind = int(self.model.geom_type[geom_id])
            if kind == int(self.mujoco.mjtGeom.mjGEOM_MESH):
                mesh_id = int(self.model.geom_dataid[geom_id])
                first = int(self.model.mesh_vertadr[mesh_id])
                count = int(self.model.mesh_vertnum[mesh_id])
                vertices = np.asarray(self.model.mesh_vert[first:first + count], dtype=float)
                # Deterministic vertices include extremes as well as broad surface coverage.
                indices = np.unique(np.r_[np.linspace(0, count - 1, min(18, count)).astype(int),
                                           np.argmin(vertices, axis=0), np.argmax(vertices, axis=0)])
                points = vertices[indices]
            elif kind == int(self.mujoco.mjtGeom.mjGEOM_BOX):
                points = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * self.model.geom_size[geom_id]
                vertices = points
            elif kind == int(self.mujoco.mjtGeom.mjGEOM_SPHERE):
                points = np.r_[np.eye(3), -np.eye(3)] * self.model.geom_size[geom_id, 0]
            else:
                # Current Panda collision geometry is mesh/box; reject unsupported
                # robot shapes rather than silently omit their geometry.
                raise ValueError(f"Unsupported robot collision geom type {kind}: {name}")
            body_id = int(self.model.geom_bodyid[geom_id])
            moving = False
            while body_id:
                moving |= body_id in self.arm_joint_body_ids
                body_id = int(self.model.body_parentid[body_id])
            samples.append((geom_id, name, points, name.startswith(gripper_prefix), "finger" in name, not moving))
            if kind == int(self.mujoco.mjtGeom.mjGEOM_SPHERE):
                self._collision_volumes_local.append({"geom_id": geom_id, "type": "sphere", "radius": float(self.model.geom_size[geom_id, 0]),
                                                      "fixed": not moving, "robot_geom_name": name})
            else:
                # MuJoCo's configured mesh collision geometry is its convex hull;
                # use every mesh vertex rather than a bounding approximation.
                hull = ConvexHull(vertices)
                equations = hull.equations
                _, distinct = np.unique(np.round(equations, decimals=10), axis=0, return_index=True)
                self._collision_volumes_local.append({"geom_id": geom_id, "type": "convex", "vertices": vertices[hull.vertices],
                                                      "planes": equations[np.sort(distinct)], "fixed": not moving,
                                                      "robot_geom_name": name})
        if not samples:
            raise ValueError("Robot collision geometry was not available")
        return samples

    def collision_points(self, data=None, tool_only=False, moving_only=False):
        data = data or self.data
        clouds, flags = [], []
        for geom_id, _, local, is_tool, is_finger, fixed in self.collision_samples:
            if tool_only and not is_tool:
                continue
            if moving_only and fixed:
                continue
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            clouds.append(local @ rotation.T + data.geom_xpos[geom_id])
            flags.extend([is_finger] * len(local))
        return np.vstack(clouds), np.asarray(flags, dtype=bool)

    def collision_volumes(self, data=None):
        """Exact current known-robot collision interiors in world coordinates.

        These volumes describe the robot prior, not environmental occupancy.
        Convex interiors satisfy normal dot point + offset <= 0.
        """
        data = data or self.data
        volumes = []
        for local in self._collision_volumes_local:
            geom_id = local["geom_id"]
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            position = data.geom_xpos[geom_id]
            common = {"type": local["type"], "fixed": local["fixed"], "robot_geom_name": local["robot_geom_name"],
                      "source": "configured robot collision geometry"}
            if local["type"] == "sphere":
                volumes.append({**common, "center_world": position.tolist(), "radius_m": local["radius"]})
            else:
                normals = local["planes"][:, :3] @ rotation.T
                offsets = local["planes"][:, 3] - normals @ position
                vertices = local["vertices"] @ rotation.T + position
                volumes.append({**common, "planes_world": np.c_[normals, offsets].tolist(),
                                "vertices_world": vertices.tolist()})
        return volumes

    def collision_points_at_joints(self, joints):
        self.data.qpos[self.qpos_indices] = joints
        self.mujoco.mj_forward(self.model, self.data)
        return self.collision_points(moving_only=True)

    def solve(self, target_pose, initial_joint_pos=None, max_iterations=120,
              position_tolerance=0.003, orientation_tolerance=0.05):
        target = pose_matrix(target_pose)
        iterations = int(max_iterations)
        if not 1 <= iterations <= 500:
            raise ValueError("IK max_iterations must be in [1,500]")
        live_data = self.sim.data._data
        # MuJoCo 2.3.7 does not expose mj_copyData in its Python bindings.
        # Copy the state needed by forward kinematics into our own data buffer.
        for field in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
            getattr(self.data, field)[:] = getattr(live_data, field)
        self.data.time = live_data.time
        seed = (np.asarray(live_data.qpos[self.qpos_indices]).copy()
                if initial_joint_pos is None else np.asarray(initial_joint_pos, dtype=float))
        if seed.shape != (7,) or not np.isfinite(seed).all():
            raise ValueError("IK seed must contain seven finite arm joints")
        joints = np.clip(seed, self.limits[:, 0] + 1e-5, self.limits[:, 1] - 1e-5)
        best = None
        for step in range(iterations + 1):
            self.data.qpos[self.qpos_indices] = joints
            self.mujoco.mj_forward(self.model, self.data)
            position = self.data.site_xpos[self.site_id]
            orientation = self.data.site_xmat[self.site_id].reshape(3, 3)
            dp = target[:3, 3] - position
            dr = rotation_error(target[:3, :3], orientation)
            position_error, orientation_error = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
            result = IKResult(position_error <= position_tolerance and orientation_error <= orientation_tolerance,
                              step, position_error, orientation_error, joints.copy())
            cost = position_error + 0.12 * orientation_error
            if best is None or cost < best[0]:
                best = (cost, result)
            if result.success or step == iterations:
                break
            self.mujoco.mj_jacSite(self.model, self.data, self.jac_position, self.jac_rotation, self.site_id)
            jacobian = np.vstack([self.jac_position[:, self.qvel_indices],
                                  0.12 * self.jac_rotation[:, self.qvel_indices]])
            error = np.r_[dp, 0.12 * dr]
            damping = 0.025
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(6), error)
            # Keep individual descents bounded near singularities and joint limits.
            delta *= min(1.0, 0.12 / max(1e-12, float(np.max(np.abs(delta)))))
            joints = np.clip(joints + delta, self.limits[:, 0] + 1e-5, self.limits[:, 1] - 1e-5)
        return best[1]
