"""Isolated cuRobo v2 numerical worker with the existing JSON-lines protocol.

Only calibrated robot geometry and explicitly supplied observed scene geometry
cross this boundary. LIBERO's private object state is never queried here.
"""
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch


def v2_model(robot_config):
    """Translate our calibrated v1 export, keeping the URDF and joint frames."""
    result = copy.deepcopy(robot_config)
    kin = result['kinematics']
    kin['tool_frames'] = [kin.pop('ee_link', 'tcp')]
    kin['format_version'] = 2.0
    cspace = kin['cspace']
    cspace['default_joint_position'] = cspace.pop('retract_config')
    return result


def normalize_ik_optimizer_config(config):
    """Translate the pinned v0.8 particle YAML's legacy cost names locally.

    RobotCostManagerCfg silently ignores bound_cfg/multi_link_pose_cfg. With
    collision disabled this leaves MPPI with no objective; its first fallback
    crashes while summing an empty cost list. Preserve weights and constraints.
    """
    result = copy.deepcopy(config)
    for group in ('cost_cfg', 'constraint_cfg', 'hybrid_cost_constraint_cfg'):
        costs = result.get('rollout', {}).get(group, {})
        for old, new in [('bound_cfg', 'cspace_cfg'), ('multi_link_pose_cfg', 'tool_pose_cfg')]:
            if old in costs:
                if new in costs:
                    raise ValueError(f'Ambiguous cuRobo cost configuration: {old} and {new}')
                costs[new] = costs.pop(old)
        cspace = costs.get('cspace_cfg', {})
        if cspace.get('cost_type') == 'TELEPORT':
            cspace['cost_type'] = 'POSITION'
    return result


def ik_optimizer_configs():
    from curobo.content import get_task_configs_path
    from curobo._src.util.config_io import join_path, load_yaml
    return [normalize_ik_optimizer_config(load_yaml(join_path(get_task_configs_path(), name)))
            for name in ('ik/particle_ik.yml', 'ik/lbfgs_ik.yml')]


class Worker:
    def initialize(self, robot_config, config):
        import warp as wp
        from curobo import runtime
        from curobo.types import DeviceCfg
        wp.config.quiet = True
        wp.config.kernel_cache_dir = os.path.join(os.environ['XDG_CACHE_HOME'], 'warp')
        # v0.8.0's mesh adapter references this namespace, removed in Warp
        # 1.18. The public function is unchanged; keep compatibility local to
        # this worker without patching either installed dependency.
        if not hasattr(wp, 'torch'):
            wp.torch = SimpleNamespace(device_from_torch=wp.device_from_torch)
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError('Warp cannot initialize the CUDA device')
        runtime.cache_dir = os.path.join(os.environ['XDG_CACHE_HOME'], 'curobo')
        runtime.cuda_graph_reset = True
        self.config = dict(config)
        self.device = DeviceCfg()
        self.count = int(config.get('curobo_num_seeds', 64))
        self.return_count = min(self.count, int(config.get('curobo_return_seeds', 32)))
        self.robot_config = v2_model(robot_config)
        self.seed = int(config.get('seed', 0))
        self.solver = self.create_solver(self.robot_config, collision=False)
        return {'backend': 'curobo_v2', 'version': importlib.metadata.version('nvidia-curobo'),
                'upstream_revision': '4ea77366ca48ee453e7df139e39fa6532af49f3b',
                'torch': torch.__version__, 'cuda': torch.version.cuda, 'warp': wp.__version__,
                'gpu': torch.cuda.get_device_name(), 'joint_names': self.solver.joint_names,
                'num_seeds': self.count, 'return_seeds': self.return_count,
                'cuda_graph': True, 'grasp_ik_environment_collision': False,
                'compatibility': ['Warp public device_from_torch alias for v0.8.0 mesh adapter',
                                  'v0.8.0 particle IK legacy cost names normalized'],
                'full_pose_error_cleanup': 'release optimizer graphs after a numerical exception; preserve the error',
                'arc_optimizer_lifetime': 'one configured arc in local_continuation mode; one query in global mode',
                'native_admission_required': True}

    def create_solver(self, robot, *, collision=False, **kwargs):
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        torch.manual_seed(self.seed)
        cfg = InverseKinematicsCfg.create(robot=copy.deepcopy(robot), device_cfg=self.device,
            optimizer_configs=ik_optimizer_configs(),
            num_seeds=self.count, seed_solver_num_seeds=self.count,
            self_collision_check=False, load_collision_spheres=collision,
            position_tolerance=kwargs.pop('position_tolerance', float(self.config.get('ik_position_tolerance_m', .003))),
            orientation_tolerance=kwargs.pop('orientation_tolerance', float(self.config.get('ik_orientation_tolerance_rad', .01))),
            use_cuda_graph=True, random_seed=self.seed, **kwargs)
        # The v0.8.0 factory applies self_collision_check=False to optimization
        # rollouts only. Match it in metrics too; native VAMP remains the
        # authoritative robot self-collision check for our calibrated export.
        for costs in cfg.metrics_rollout_config.get_cost_manager_configs():
            costs.disable_self_collision()
        for rollout in cfg.optimizer_rollout_configs:
            if rollout.cost_cfg is None or rollout.cost_cfg.tool_pose_cfg is None:
                raise ValueError('Every cuRobo IK optimizer stage requires a tool-pose objective')
        return InverseKinematics(cfg)

    def configure_arc(self, collision_spheres, pad_local, scene_arrays_path,
                      scene_arrays_sha256, origin, axis, mesh_pitch=.005,
                      solver_mode='global', local_dt=.2, moving_mesh_pitch=.002,
                      local_seed_std=.02, local_velocity_weight=.001,position_tolerance=None):
        """Load sensor-derived scene surfaces, preserving moving-part ownership."""
        from pathlib import Path
        started = time.perf_counter()
        if solver_mode not in {'global', 'local_continuation'}:
            raise ValueError('Unsupported arc solver mode')
        for name, value in {'mesh_pitch': mesh_pitch, 'moving_mesh_pitch': moving_mesh_pitch,
                            'local_dt': local_dt}.items():
            if not np.isfinite(value) or value <= 0.:
                raise ValueError(f'{name} must be finite and positive')
        for name, value in {'local_seed_std': local_seed_std,
                            'local_velocity_weight': local_velocity_weight}.items():
            if not np.isfinite(value) or value < 0.:
                raise ValueError(f'{name} must be finite and nonnegative')
        origin, axis, pad_local = [np.asarray(value, float) for value in (origin, axis, pad_local)]
        if any(value.shape != (3,) or not np.isfinite(value).all()
               for value in (origin, axis, pad_local)) or np.linalg.norm(axis) < 1e-12:
            raise ValueError('Expected finite origin, nonzero axis, and contact offset vectors')
        path = Path(scene_arrays_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != scene_arrays_sha256:
            raise ValueError('Observed scene array hash mismatch')
        with np.load(path, allow_pickle=False) as arrays:
            # contact_reference tracks the association with the original
            # contact patch. It is not an additional measured obstacle in
            # either solver mode: moving_reference already holds current
            # moving surfaces, matching native obstacle ownership.
            moving = arrays['moving_reference']
            clouds = {'stationary': arrays['static_points'], 'moving': moving}
        for role, points in clouds.items():
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError(f'{role} observed surfaces must contain finite xyz points')
        from curobo.scene import Mesh, Scene
        self._destroy_arc_solver()
        self.arc_origin, self.arc_axis = origin, axis / np.linalg.norm(axis)
        self.arc_solver_mode = solver_mode
        self.arc_local_dt = float(local_dt)
        self.arc_local_seed_std = float(local_seed_std)
        self.arc_local_velocity_weight = float(local_velocity_weight)
        self.arc_position_tolerance=float(self.config.get('arc_position_tolerance_m',.006) if position_tolerance is None else position_tolerance)
        if not np.isfinite(self.arc_position_tolerance) or self.arc_position_tolerance<=0:
            raise ValueError('Arc position tolerance must be positive')
        self.arc_rng = np.random.default_rng(self.seed)
        meshes = [];self.moving_mesh_name=None
        if not hasattr(self, 'mesh_cache'):
            self.mesh_cache = {}
        mesh_pitches = {'stationary': float(mesh_pitch),
                       'moving': float(moving_mesh_pitch if solver_mode == 'local_continuation' else mesh_pitch)}
        for role, points in clouds.items():
            if not len(points):continue
            pitch = mesh_pitches[role]
            digest = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
            key = (role, digest, pitch)
            if key not in self.mesh_cache:
                self.mesh_cache[key] = Mesh.from_pointcloud(points, pitch=pitch,
                    name=role+'_'+digest[:16])
            mesh = self.mesh_cache[key]
            mesh.pose = [0., 0., 0., 1., 0., 0., 0.]
            meshes.append(mesh)
            if role == 'moving':
                self.moving_mesh_name = mesh.name
        scene = Scene(mesh=meshes)
        model = copy.deepcopy(self.robot_config)
        model['kinematics'].update(tool_frames=['contact_point'],
            collision_link_names=list(collision_spheres), collision_spheres=collision_spheres,
            collision_sphere_buffer=0., self_collision_ignore={}, self_collision_buffer={},
            extra_links={'contact_point': {'parent_link_name': 'tcp', 'link_name': 'contact_point',
                'joint_name': 'contact_point_fixed', 'joint_type': 'FIXED',
                'fixed_transform': [*np.round(pad_local, 10).tolist(), 1., 0., 0., 0.]}})
        signature = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
        self.arc_robot_config, self.arc_scene = model, scene
        # Reusing one optimizer is allowed only within this configured arc.
        # Even identical geometry on the next request gets fresh optimizer
        # and local sampler state; the CPU surface mesh cache can survive.
        self.arc_solver = self.create_arc_solver()
        self.arc_solver_used = False
        self.arc_model_signature = signature
        self.arc_counter = getattr(self, 'arc_counter', 0) + 1
        self.active_arc_id = self.arc_counter
        # Limit CPU mesh retention; the active GPU scene owns its mesh handles.
        if len(self.mesh_cache) > 4:
            self.mesh_cache = {key: value for key, value in self.mesh_cache.items() if value in meshes}
        torch.cuda.synchronize()
        return {'backend': 'curobo_v2', 'scene_setup_seconds': time.perf_counter()-started,
                'arc_id': self.active_arc_id, 'solver_mode': solver_mode,
                'optimizer_generation': self.arc_optimizer_generation,
                'optimizer_lifetime': 'one_arc' if solver_mode == 'local_continuation' else 'one_query',
                'solver_recreated': True, 'mesh_pitch_m': mesh_pitch,
                'mesh_pitch_by_role_m': mesh_pitches,
                'current_moving_surface_only': True,
                'local_optimization_dt_seconds': self.arc_local_dt if solver_mode == 'local_continuation' else None,
                'local_seed_std_rad': self.arc_local_seed_std if solver_mode == 'local_continuation' else None,
                'velocity_regularization_weight': (self.arc_local_velocity_weight if solver_mode == 'local_continuation'
                    else float(self.config.get('curobo_v2_arc_velocity_weight', 1.0))),
                'seed_velocity_weight': 0. if solver_mode == 'local_continuation' else .01,
                'seed_penalty_reference': 'disabled' if solver_mode == 'local_continuation' else 'zero_joint_prior_retained_from_global_baseline',
                'all_seeds_local': solver_mode == 'local_continuation',
                'observed_points': {key: len(value) for key, value in clouds.items()},
                'mesh_triangles': {mesh.name: len(mesh.faces)//3 for mesh in meshes},
                'scene_arrays_sha256': scene_arrays_sha256, 'native_admission_required': True,
                'map_method': 'upstream Mesh.from_pointcloud; GPU mesh SDF',
                'hidden_object_geometry_used': False}

    def create_arc_solver(self):
        local = self.arc_solver_mode == 'local_continuation'
        solver = self.create_solver(self.arc_robot_config, collision=True, scene_model=self.arc_scene,
            position_tolerance=float(getattr(self,'arc_position_tolerance',self.config.get('arc_position_tolerance_m', .006))),
            collision_cache={'mesh': 2}, orientation_tolerance=3.2,
            velocity_regularization_weight=(self.arc_local_velocity_weight if local
                else float(self.config.get('curobo_v2_arc_velocity_weight', 1.0))),
            seed_velocity_weight=0. if local else .01, optimizer_collision_activation_distance=.002,
            **({'optimization_dt': self.arc_local_dt} if local else {}))
        if local:
            # The pinned IK implementation reads these config fields. Local
            # seeds must bypass global LM reseeding and early seed-only exits.
            solver.config.use_lm_seed = False
            solver.config.exit_early = False
        self.arc_optimizer_generation = getattr(self, 'arc_optimizer_generation', 0) + 1
        return solver

    def _destroy_arc_solver(self):
        solver = getattr(self, 'arc_solver', None)
        self.arc_solver, self.active_arc_id = None, None
        if solver is not None:
            solver.destroy()

    def _check_arc(self, arc_id=None, *, required=False):
        if getattr(self, 'arc_solver', None) is None or getattr(self, 'active_arc_id', None) is None:
            raise ValueError('No configured arc is active')
        if required and arc_id is None:
            raise ValueError('arc_id is required for local continuation')
        if arc_id is not None and (isinstance(arc_id, bool) or not isinstance(arc_id, int)
                                   or arc_id != self.active_arc_id):
            raise ValueError('Stale or invalid arc_id')

    def end_arc(self, arc_id):
        self._check_arc(arc_id, required=True)
        self._destroy_arc_solver()
        return {'arc_id': arc_id, 'arc_ended': True, 'backend': 'curobo_v2'}

    def arc_forward(self, joints, arc_id=None):
        self._check_arc(arc_id)
        joints = np.asarray(joints, float)
        if joints.ndim != 2 or joints.shape[1] != 7 or not len(joints) or not np.isfinite(joints).all():
            raise ValueError('Expected a nonempty batch of finite seven-joint configurations')
        state = self.arc_solver.compute_kinematics(self.joint_state(joints))
        return {'spheres_world': state.robot_spheres.detach().cpu().reshape(len(joints), -1, 4).tolist(),
                'contact_points_world': state.tool_poses.position.detach().cpu().reshape(-1, 3).tolist()}

    def solve_arc(self, point, reference_quaternion_xyzw, seed, angle, mode='point', arc_id=None):
        local = getattr(self, 'arc_solver_mode', None) == 'local_continuation'
        self._check_arc(arc_id, required=local)
        point, q, initial = [np.asarray(value, float) for value in (point, reference_quaternion_xyzw, seed)]
        if (point.shape != (3,) or q.shape != (4,) or initial.shape != (7,)
                or not np.isfinite(np.r_[point, q, initial, angle]).all()
                or np.linalg.norm(q) < 1e-12):
            raise ValueError('Expected a finite point, nonzero quaternion, seven joints, and angle')
        q = q / np.linalg.norm(q)
        if mode not in {'point', 'rigid', 'jaw_axis'}:
            raise ValueError('Unsupported contact constraint')
        from scipy.spatial.transform import Rotation
        from curobo.motion_retargeter import ToolPoseCriteria
        from curobo.types import GoalToolPose, Pose
        started = time.perf_counter()
        # Global mode retains the diagnostic baseline's query reset. Local
        # continuation reuses state only inside this explicitly identified arc;
        # configure_arc always starts the next arc with a fresh optimizer.
        solver_setup_seconds = 0.
        optimizer_reused = bool(self.arc_solver_used and local)
        solver_recreated = bool(self.arc_solver_used and not local)
        if self.arc_solver_used and not local:
            setup_started = time.perf_counter()
            self.arc_solver.destroy()
            self.arc_solver = self.create_arc_solver()
            solver_setup_seconds = time.perf_counter()-setup_started
        self.arc_solver_used = True
        rotation = Rotation.from_rotvec(float(angle)*self.arc_axis)
        translation = self.arc_origin-rotation.apply(self.arc_origin)
        quat = rotation.as_quat()
        if self.moving_mesh_name is not None:
            self.arc_solver.scene_collision_checker.data.update_obstacle_pose(self.moving_mesh_name,
                Pose.from_list([*translation, quat[3], *quat[:3]], device_cfg=self.device))
        # Orientation is a soft proposal preference in point mode. The native
        # model remains responsible for exact jaw-axis and wrist constraints.
        orient_weights = {'point': [.002]*3, 'rigid': [1.]*3, 'jaw_axis': [0., 1., 1.]}[mode]
        self.arc_solver.update_tool_pose_criteria({'contact_point': ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[1., 1., 1., *orient_weights],
            project_distance_to_goal=mode == 'jaw_axis', device_cfg=self.device)})
        goal = GoalToolPose.from_poses({'contact_point': Pose(
            position=self.device.to_device(point[None]),
            quaternion=self.device.to_device(q[[3, 0, 1, 2]][None]))})
        if local:
            perturbations = self.arc_rng.normal(0., self.arc_local_seed_std, (self.count, 7))
            perturbations[0] = 0.
            lower = self.arc_solver.seed_ik_solver.action_min.detach().cpu().numpy().reshape(-1)
            upper = self.arc_solver.seed_ik_solver.action_max.detach().cpu().numpy().reshape(-1)
            seeds = np.clip(initial[None] + perturbations, lower, upper)[None]
        else:
            seeds = initial.reshape(1, 1, 7)
        result = self.arc_solver.solve_pose(goal, current_state=self.joint_state(initial[None]),
            seed_config=self.device.to_device(seeds), return_seeds=self.count)
        torch.cuda.synchronize()
        return {'solutions': result.solution.detach().cpu().reshape(-1, 7).tolist(),
                'success': result.success.detach().cpu().reshape(-1).tolist(),
                'position_error_m': result.position_error.detach().cpu().reshape(-1).tolist(),
                'orientation_error_rad': result.rotation_error.detach().cpu().reshape(-1).tolist(),
                'feasible': result.feasible.detach().cpu().reshape(-1).tolist(),
                'elapsed_seconds': time.perf_counter()-started,
                'solver_seconds': float(result.solve_time),
                'solver_setup_seconds': solver_setup_seconds,
                'arc_id': self.active_arc_id, 'solver_mode': self.arc_solver_mode,
                'optimizer_generation': self.arc_optimizer_generation,
                'solver_recreated': solver_recreated,
                'optimizer_reused_within_arc': optimizer_reused,
                'query_scoped_optimizer': not local,
                'optimizer_lifetime': 'one_arc' if local else 'one_query',
                'all_seeds_local': local,
                'backend': 'curobo_v2', 'native_admission_required': True}

    def joint_state(self, values):
        from curobo.types import JointState
        return JointState.from_position(self.device.to_device(values), joint_names=self.solver.joint_names)

    def fk(self, joints):
        state = self.solver.compute_kinematics(self.joint_state(joints))
        pose = state.tool_poses.get_link_pose('tcp')
        p = pose.position.detach().cpu().numpy().reshape(-1, 3)
        q = pose.quaternion.detach().cpu().numpy().reshape(-1, 4)
        return {'poses_world_xyzw': np.c_[p, q[:, 1:], q[:, :1]].tolist()}

    def solve(self, pose, seed):
        from curobo.types import GoalToolPose, Pose
        started = time.perf_counter()
        pose = np.asarray(pose, float)
        seed = np.asarray(seed, float)
        if pose.shape != (7,) or seed.shape != (7,) or not np.isfinite(np.r_[pose, seed]).all():
            raise ValueError('Expected a finite xyz/xyzw pose and seven joints')
        if self.config.get('curobo_query_reproducible', False):
            self.solver.reset_seed()
        goal = GoalToolPose.from_poses({'tcp': Pose(
            position=self.device.to_device(pose[:3][None]),
            quaternion=self.device.to_device(pose[[6, 3, 4, 5]][None]))})
        try:
            result = self.solver.solve_pose(goal, current_state=self.joint_state(seed[None]),
                seed_config=self.device.to_device(seed.reshape(1, 1, 7)), return_seeds=self.return_count)
        except Exception:
            # Upstream warmup can populate graph inputs before graph capture
            # succeeds. Its unconditional cleanup clears partial graph state,
            # so a later request can initialize normally. Do not hide/retry
            # this failed request or turn it into an infeasibility result.
            self.solver.destroy()
            raise
        torch.cuda.synchronize()
        return {'solutions': result.solution.detach().cpu().reshape(-1, 7).tolist(),
                'success': result.success.detach().cpu().reshape(-1).tolist(),
                'elapsed_seconds': time.perf_counter()-started,
                'solver_seconds': float(result.solve_time),
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(), 'backend': 'curobo_v2'}


def main():
    protocol, worker = sys.stdout, Worker()
    with contextlib.redirect_stdout(sys.stderr):
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                op = request.pop('op')
                request_id = request.pop('id', None)
                if op == 'close':
                    result = {}
                elif op in {'initialize', 'fk', 'solve', 'configure_arc', 'arc_forward', 'solve_arc', 'end_arc'}:
                    result = getattr(worker, op)(**request)
                else:
                    raise ValueError(f'Unknown cuRobo v2 operation: {op}')
                response = {'ok': True, 'id': request_id, **result}
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                response = {'ok': False, 'id': locals().get('request_id'), 'error': str(error)}
            protocol.write(json.dumps(response, allow_nan=False)+'\n')
            protocol.flush()
            if locals().get('op') == 'close':
                break


if __name__ == '__main__':
    main()
