"""Deterministic robot-only IK restarts without changing live simulation state."""

import numpy as np
from scipy.spatial import ConvexHull

from .kinematics import PandaIK, matrix_pose


class _RobotKinematicsApi:
    """Avoid scene contact/dynamics work for the isolated IK data buffer."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def mj_forward(self, model, data):
        # Kinematics updates robot body/site/geom frames. comPos supplies cdof
        # and subtree COMs consumed by MuJoCo's standard site Jacobian routine.
        self._module.mj_kinematics(model, data)
        self._module.mj_comPos(model, data)


class MultiStartPandaIK(PandaIK):
    """Keep the released DLS descent and try several known Panda configurations.

    Every seed uses only arm joint limits and robot proprioception. Successful
    configurations remain available to the caller for independent collision
    checks; the lowest pose residual is not automatically a safe configuration.
    """

    def __init__(self, sim, robot):
        super().__init__(sim, robot)
        self.mujoco = _RobotKinematicsApi(self.mujoco)
        self._visual_volumes_local = []
        prefixes = (robot.robot_model.naming_prefix, robot.gripper.naming_prefix)
        for geom_id, name in enumerate(sim.model.geom_names):
            if not name or not name.startswith(prefixes) or int(self.model.geom_group[geom_id]) != 1:
                continue
            if int(self.model.geom_type[geom_id]) != int(self.mujoco.mjtGeom.mjGEOM_MESH):
                raise ValueError(f"Unsupported known robot visual geom: {name}")
            mesh_id = int(self.model.geom_dataid[geom_id])
            first, count = int(self.model.mesh_vertadr[mesh_id]), int(self.model.mesh_vertnum[mesh_id])
            vertices = np.asarray(self.model.mesh_vert[first:first + count], dtype=float)
            hull = ConvexHull(vertices)
            _, distinct = np.unique(np.round(hull.equations, 10), axis=0, return_index=True)
            body_id = int(self.model.geom_bodyid[geom_id])
            moving = False
            while body_id:
                moving |= body_id in self.arm_joint_body_ids
                body_id = int(self.model.body_parentid[body_id])
            self._visual_volumes_local.append({"geom_id": geom_id, "robot_geom_name": name,
                                               "vertices": vertices[hull.vertices],
                                               "planes": hull.equations[np.sort(distinct)],
                                               "fixed": not moving})

    def collision_point_geom_names(self, tool_only=False, moving_only=False):
        names = []
        for _, name, points, is_tool, _, fixed in self.collision_samples:
            if tool_only and not is_tool:
                continue
            if moving_only and fixed:
                continue
            names.extend([name] * len(points))
        return names

    def visual_volumes(self, data=None):
        """Known rendered robot hulls for observed self-surface identification.

        These are visual geometry overapproximations, not free-space evidence.
        In particular a future robot configuration cannot clear an unseen path.
        """
        data = data or self.data
        result = []
        for local in self._visual_volumes_local:
            geom_id = local["geom_id"]
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            position = data.geom_xpos[geom_id]
            normals = local["planes"][:, :3] @ rotation.T
            offsets = local["planes"][:, 3] - normals @ position
            result.append({"type": "convex", "fixed": local["fixed"],
                           "robot_geom_name": local["robot_geom_name"], "render_group": 1,
                           "source": "configured robot visual geometry, not scene geometry",
                           "volume_policy": "convex visual hull; no free-space clearing",
                           "planes_world": np.c_[normals, offsets].tolist(),
                           "vertices_world": (local["vertices"] @ rotation.T + position).tolist()})
        return result

    def _copy_live_state(self):
        live = self.sim.data._data
        for field in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
            getattr(self.data, field)[:] = getattr(live, field)
        self.data.time = live.time

    def _checked_joints(self, joints):
        value = np.asarray(joints, dtype=float)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("Expected seven finite Panda arm joints")
        if np.any(value < self.limits[:, 0]) or np.any(value > self.limits[:, 1]):
            raise ValueError("Arm joints exceed configured robot limits")
        return value.copy()

    def fk(self, joints=None):
        """Return the known robot TCP pose using the isolated data buffer."""
        self._copy_live_state()
        if joints is not None:
            self.data.qpos[self.qpos_indices] = self._checked_joints(joints)
        self.mujoco.mj_forward(self.model, self.data)
        transform = np.eye(4)
        transform[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        transform[:3, 3] = self.data.site_xpos[self.site_id]
        return matrix_pose(transform)

    def contact_face_flags(self, data=None, tool_only=False, moving_only=False):
        """Label only inward faces of the configured Panda pad boxes.

        Broad finger meshes and the outer pad faces remain ordinary collision
        geometry. The mask follows exactly ``collision_points`` sample order.
        """
        data = data or self.data
        tcp_rotation = data.site_xmat[self.site_id].reshape(3, 3)
        tcp_position = data.site_xpos[self.site_id]
        flags = []
        for geom_id, name, local, is_tool, _, fixed in self.collision_samples:
            if tool_only and not is_tool:
                continue
            if moving_only and fixed:
                continue
            selected = np.zeros(len(local), dtype=bool)
            if name.endswith("_pad_collision"):
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                world = local @ rotation.T + data.geom_xpos[geom_id]
                tool = (world - tcp_position) @ tcp_rotation
                selected = np.abs(np.abs(tool[:, 0]) - np.min(np.abs(tool[:, 0]))) < 1e-7
            flags.extend(selected.tolist())
        return np.asarray(flags, dtype=bool)

    def seed_configurations(self, initial_joint_pos=None, max_starts=12):
        if not 1 <= int(max_starts) <= 12:
            raise ValueError("max_starts must be in [1,12]")
        current = (np.asarray(self.sim.data._data.qpos[self.qpos_indices]).copy()
                   if initial_joint_pos is None else np.asarray(initial_joint_pos, dtype=float))
        if current.shape != (7,) or not np.isfinite(current).all():
            raise ValueError("IK seed must contain seven finite arm joints")
        neutral = np.array([0, -0.5, 0, -2.1, 0, 1.6, np.pi / 4])
        centered = self.limits.mean(axis=1)
        seeds = [current, neutral, centered]
        # Known robot postures, not object-dependent grasp guesses.
        for axis, offset in ((0, 0.8), (0, -0.8), (2, 0.8), (2, -0.8),
                             (6, 1.4), (6, -1.4)):
            value = neutral.copy()
            value[axis] += offset
            seeds.append(value)
        for offset in (-0.6, 0.6):
            value = current.copy()
            value[2] += offset
            value[4] -= offset
            seeds.append(value)
        seeds.append(np.array([0, 0.3, 0, -1.4, 0, 1.8, np.pi / 4]))
        distinct = []
        for value in seeds:
            value = np.clip(value, self.limits[:, 0] + 1e-5, self.limits[:, 1] - 1e-5)
            if not any(np.allclose(value, prior, atol=1e-6) for prior in distinct):
                distinct.append(value)
        return distinct[:int(max_starts)]

    def solve_all(self, target_pose, initial_joint_pos=None, max_iterations=200,
                  position_tolerance=0.003, orientation_tolerance=0.05, max_starts=12):
        seeds = self.seed_configurations(initial_joint_pos, max_starts)
        attempts = []
        feasible = []
        for seed_index, seed in enumerate(seeds):
            result = super().solve(target_pose, initial_joint_pos=seed,
                                   max_iterations=max_iterations,
                                   position_tolerance=position_tolerance,
                                   orientation_tolerance=orientation_tolerance)
            attempts.append((seed_index, result))
            if result.success and not any(np.linalg.norm(result.cspace_position - prior.cspace_position) < 0.01
                                          for prior in feasible):
                feasible.append(result)
        reference = seeds[0]
        def rank(result):
            return (not result.success, result.position_error + 0.12 * result.orientation_error,
                    float(np.linalg.norm(result.cspace_position - reference)))
        feasible.sort(key=rank)
        best = min((result for _, result in attempts), key=rank)
        # Leave our own buffer at the selected solution, never the live data.
        self.fk(best.cspace_position)
        return best, feasible, attempts

    def solve(self, target_pose, initial_joint_pos=None, max_iterations=120,
              position_tolerance=0.003, orientation_tolerance=0.05):
        best, _, _ = self.solve_all(target_pose, initial_joint_pos, max_iterations,
                                    position_tolerance, orientation_tolerance)
        return best
