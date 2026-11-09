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
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
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
        self.native_indices = self.robot_config.pop('native_sphere_indices', [])
        self.native_self_pairs = self.robot_config.pop('native_self_collision_pairs', [])
        self.point_scene = None
        self.seed = int(config.get('seed', 0))
        self.solver = self.create_solver(self.robot_config)
        return {'backend': 'curobo_v2', 'version': importlib.metadata.version('nvidia-curobo'),
                'upstream_revision': '4ea77366ca48ee453e7df139e39fa6532af49f3b',
                'torch': torch.__version__, 'cuda': torch.version.cuda, 'warp': wp.__version__,
                'gpu': torch.cuda.get_device_name(), 'joint_names': self.solver.joint_names,
                'num_seeds': self.count, 'return_seeds': self.return_count,
                'cuda_graph': True, 'grasp_ik_environment_collision': 'current observed scene supplied before planning',
                'ik_self_collision': True, 'robot_collision_model': 'native VAMP spheres and self pairs',
                'contact_continuation_method': 'single native LM seed; main sampled optimizer disabled',
                'compatibility': ['Warp public device_from_torch alias for v0.8.0 mesh adapter',
                                  'v0.8.0 particle IK legacy cost names normalized'],
                'full_pose_error_cleanup': 'release optimizer graphs after a numerical exception; preserve the error',
                'arc_optimizer_lifetime': 'one configured arc in local_continuation mode; one query in global mode',
                'native_admission_required': True}

    def create_solver(self, robot, *, collision=True, environment_collision=True, **kwargs):
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        torch.manual_seed(self.seed)
        count = int(kwargs.pop('num_seeds', self.count))
        cfg = InverseKinematicsCfg.create(robot=copy.deepcopy(robot), device_cfg=self.device,
            optimizer_configs=ik_optimizer_configs(),
            num_seeds=count, seed_solver_num_seeds=count,
            self_collision_check=collision, load_collision_spheres=collision,
            optimizer_collision_activation_distance=0.,
            position_tolerance=kwargs.pop('position_tolerance', float(self.config.get('ik_position_tolerance_m', .003))),
            orientation_tolerance=kwargs.pop('orientation_tolerance', float(self.config.get('ik_orientation_tolerance_rad', .01))),
            use_cuda_graph=True, random_seed=self.seed, **kwargs)
        if collision and self.native_self_pairs:
            indices = {native: index for index, native in enumerate(self.native_indices)}
            pairs = [[indices[a], indices[b]] for a, b in self.native_self_pairs]
            cfg.robot_config.kinematics.self_collision_config.collision_pairs = torch.tensor(
                pairs, dtype=torch.int16, device=self.device.device)
            cfg.robot_config.kinematics.self_collision_config.sphere_padding.zero_()
        for rollout in cfg.optimizer_rollout_configs:
            if rollout.cost_cfg is None or rollout.cost_cfg.tool_pose_cfg is None:
                raise ValueError('Every cuRobo IK optimizer stage requires a tool-pose objective')
        return InverseKinematics(cfg,
            scene_collision_checker=self.point_scene if environment_collision else None)

    def configure_scene(self, arrays_path: str, arrays_sha256: str,
                        palm_contact: bool = False, finger_model: dict | None = None,
                        fingertip_contact: bool = False) -> dict[str, Any]:
        """Update the shared collision world without changing observed geometry."""
        from pathlib import Path
        from .curobo_scene_points import ObservedPointScene
        started = time.perf_counter()
        path = Path(arrays_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != arrays_sha256:
            raise ValueError('Observed point scene hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            clouds = {name: data[name].astype(np.float32) for name in ('stationary', 'moving', 'contact')}
        if any(x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() for x in clouds.values()):
            raise ValueError('Observed point partitions must contain finite xyz values')
        needed = max(1, len(clouds['stationary'])+len(clouds['moving']), len(clouds['contact']))
        recreated = self.point_scene is None or needed > self.point_scene.capacity
        if recreated:
            self.solver.destroy()
            retained = {}
            for key, cached in getattr(self, 'contact_solvers', {}).items():
                if cached['environment_collision']:
                    cached['solver'].destroy()
                else:
                    retained[key] = cached
            self.contact_solvers = retained
            self.point_scene = ObservedPointScene(self.device, self.native_indices, 1 << (needed-1).bit_length())
            self.solver = self.create_solver(self.robot_config)
        self.point_scene.update(clouds, palm_contact, finger_model, fingertip_contact)
        return {'scene_collision_checked': True, 'self_collision_checked': True,
            'observed_points': {k: len(v) for k, v in clouds.items()},
            'scene_sha256': arrays_sha256, 'solver_recreated': recreated,
            'scene_setup_seconds': time.perf_counter()-started,
            'geometry': 'native cuRobo zero-size cuboids at VAMP observed points',
            'point_radius_m': 0., 'collision_activation_distance_m': 0.,
            'palm_contact_allowed': bool(palm_contact), 'fingertip_contact_allowed': bool(fingertip_contact),
            'finger_environment_collision_checked': finger_model is not None}

    def move_scene_target(self, transform: ArrayLike) -> dict[str, Any]:
        matrix = np.asarray(transform, float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError('Expected a finite target transform')
        if self.point_scene is None or not self.point_scene.loaded:
            raise ValueError('Configure the observed scene before moving target geometry')
        self.point_scene.move_target(matrix)
        return {'target_geometry_updated': True}

    def collision_diagnostics(self, joints: ArrayLike) -> dict[str, Any]:
        """Read native collision costs for offline model-parity tests; no motion."""
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer
        state=self.solver.compute_kinematics(self.joint_state(joints))
        spheres=state.robot_spheres
        cost=self.solver.core.metrics_rollout.constraint_manager.get_cost('self_collision')
        cost.setup_batch_tensors(spheres.shape[0],spheres.shape[1])
        self_cost=cost.forward(spheres)
        buffer=CollisionBuffer.from_shape(spheres.shape,self.device)
        environment=self.point_scene.get_sphere_distance(state,buffer,
            self.device.to_device([1.]),self.device.to_device([0.]))
        return {'spheres_world':spheres.detach().cpu().tolist(),
            'self_collision_cost':self_cost.detach().cpu().tolist(),
            'environment_cost_per_sphere':environment.detach().cpu().tolist()}

    def solve_contact(self, point: ArrayLike, contact_local: ArrayLike,
                      reference_quaternion_xyzw: ArrayLike, seed: ArrayLike, local: bool = True,
                      contact_normal_local: ArrayLike | None = None,
                      surface_normal_world: ArrayLike | None = None,
                      environment_collision: bool = True) -> dict[str, Any]:
        """Solve a point or point-plus-normal goal with native cuRobo criteria.

        Surface-contact goals align a calibrated robot face against the explicitly
        selected outward normal. Goal-frame yaw is free. Continuation uses one
        current-state seed in native LM, followed by cuRobo's result metrics with
        its main optimizer disabled. Acquisition uses the existing batch. No failed
        continuation is retried by sampled optimization or a different model.
        """
        from curobo.motion_retargeter import ToolPoseCriteria
        from curobo.types import GoalToolPose, Pose
        from scipy.spatial.transform import Rotation
        from vlm_tamp.contact_orientation import frame_with_z, normal_vector
        point, offset, quat, initial = [np.asarray(x, float) for x in
            (point, contact_local, reference_quaternion_xyzw, seed)]
        started = time.perf_counter()
        oriented = contact_normal_local is not None or surface_normal_world is not None
        if oriented and (contact_normal_local is None or surface_normal_world is None):
            raise ValueError('Surface-contact IK requires both the robot face normal and signed surface normal')
        constraint = 'surface_normal' if oriented else 'point'
        local_normal = normal_vector(contact_normal_local) if oriented else None
        outward_normal = normal_vector(surface_normal_world) if oriented else None
        signature = (tuple(offset), tuple(local_normal) if oriented else None)
        frame_quat = (Rotation.from_matrix(frame_with_z(local_normal)).as_quat()[[3, 0, 1, 2]]
            if oriented else np.array([1., 0., 0., 0.]))
        if oriented:
            quat = Rotation.from_matrix(frame_with_z(-outward_normal)).as_quat()
        count = 1 if local else self.count
        # Keep acquisition and continuation alive independently. Switching
        # between them must not repeatedly evict and warm the other optimizer.
        if not hasattr(self, 'contact_solvers'):self.contact_solvers={}
        cache_key=(constraint, bool(local), bool(environment_collision))
        cached=self.contact_solvers.get(cache_key)
        created=cached is None or cached['offset']!=signature
        setup_seconds=0.
        if created:
            if cached is not None:cached['solver'].destroy()
            setup_started=time.perf_counter()
            model = copy.deepcopy(self.robot_config)
            model['kinematics'].update(tool_frames=['contact_point'], extra_links={
                'contact_point': {'parent_link_name': 'tcp', 'link_name': 'contact_point',
                    'joint_name': 'contact_point_fixed', 'joint_type': 'FIXED',
                    'fixed_transform': [*offset.tolist(), *frame_quat.tolist()]}})
            solver = self.create_solver(model, num_seeds=count, environment_collision=environment_collision,
                **({} if oriented else {'orientation_tolerance': 3.2}))
            if local:
                solver.seed_ik_solver.config.position_tolerance = solver.config.position_tolerance
                solver.seed_ik_solver.config.orientation_tolerance = solver.config.orientation_tolerance
            solver.update_tool_pose_criteria({'contact_point': ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1., 1., 1., 1., 1., 0.] if oriented else [1., 1., 1., 0., 0., 0.],
                project_distance_to_goal=oriented, device_cfg=self.device)})
            cached={'solver':solver,'offset':signature,'used':False,'environment_collision':environment_collision}
            self.contact_solvers[cache_key]=cached
            setup_seconds=time.perf_counter()-setup_started
        solver=cached['solver'];first_solve=not cached['used']
        goal = GoalToolPose.from_poses({'contact_point': Pose(
            position=self.device.to_device(point[None]),
            quaternion=self.device.to_device(quat[[3, 0, 1, 2]][None]))})
        solve_started=time.perf_counter()
        current_state=self.joint_state(initial[None])
        seed_config=self.device.to_device(initial.reshape(1, 1, 7))
        local_seconds=0.
        if local:
            # A single supplied seed exactly fills the native LM solver's seed
            # count. No additional random/particle seeds are generated.
            local_result=solver.seed_ik_solver.solve_single(goal_tool_poses=goal,
                current_state=current_state,seed_config=seed_config,return_seeds=1)
            local_seconds=float(local_result.solve_time)
            # The SDK's pose/bounds/self-collision result evaluation, without
            # MPPI/L-BFGS or any modification of LM's joint solution.
            result=solver.solve_pose(goal,current_state=current_state,
                seed_config=local_result.solution,return_seeds=1,run_optimizer=False)
            result.success=result.success & local_result.success
        else:
            result=solver.solve_pose(goal,current_state=current_state,
                seed_config=seed_config,return_seeds=count)
        torch.cuda.synchronize()
        solve_call_seconds=time.perf_counter()-solve_started
        cached['used']=True
        return {'solutions': result.solution.detach().cpu().reshape(-1, 7).tolist(),
                'success': result.success.detach().cpu().reshape(-1).tolist(),
                'feasible': result.feasible.detach().cpu().reshape(-1).tolist(),
                'position_error_m': result.position_error.detach().cpu().reshape(-1).tolist(),
                'scene_collision_checked': bool(environment_collision and self.point_scene is not None and self.point_scene.loaded),
                'self_collision_checked': True,
                'elapsed_seconds': time.perf_counter()-started,
                'solver_created':created,'first_solve_for_contact_solver':first_solve,
                'solver_setup_seconds':setup_seconds,'solve_call_seconds':solve_call_seconds,
                'solver_reported_seconds':float(result.solve_time)+local_seconds,
                'native_local_solve_seconds':local_seconds if local else None,
                'native_result_metrics_seconds':float(result.solve_time) if local else None,
                'local_solver_success':local_result.success.detach().cpu().reshape(-1).tolist() if local else None,
                'solution_method':'native_curobo_lm_from_previous_posture' if local else 'native_curobo_acquisition_batch',
                'sampled_optimizer_used':not local,
                'continuation_fallback_used':False,
                'backend': 'curobo_v2_contact_continuation' if local else 'curobo_v2_contact_ik', 'seed_count': count,
                'contact_constraint': constraint, 'wrist_twist_constrained': False,
                'orientation_is_soft_preference': False}

    def configure_arc(self, collision_spheres, pad_local, scene_arrays_path,
                      scene_arrays_sha256, origin, axis, mesh_pitch=.005,
                      solver_mode='global', local_dt=.2, moving_mesh_pitch=.002,
                      local_seed_std=.02,position_tolerance=None,
                      orientation_tolerance=None):
        """Load sensor-derived scene surfaces, preserving moving-part ownership."""
        from pathlib import Path
        started = time.perf_counter()
        if solver_mode not in {'global', 'local_continuation'}:
            raise ValueError('Unsupported arc solver mode')
        if self.point_scene is None or not self.point_scene.loaded:
            raise ValueError('Configure the shared observed collision world before arc IK')
        for name, value in {'mesh_pitch': mesh_pitch, 'moving_mesh_pitch': moving_mesh_pitch,
                            'local_dt': local_dt}.items():
            if not np.isfinite(value) or value <= 0.:
                raise ValueError(f'{name} must be finite and positive')
        for name, value in {'local_seed_std': local_seed_std}.items():
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
        from curobo.scene import Scene
        self._destroy_arc_solver()
        self.arc_origin, self.arc_axis = origin, axis / np.linalg.norm(axis)
        self.arc_solver_mode = solver_mode
        self.arc_local_dt = float(local_dt)
        self.arc_local_seed_std = float(local_seed_std)
        self.arc_orientation_tolerance = (float(self.config.get('arc_orientation_tolerance_rad', np.radians(3.)))
            if orientation_tolerance is None else float(orientation_tolerance))
        self.arc_position_tolerance=float(self.config.get('arc_position_tolerance_m',.006) if position_tolerance is None else position_tolerance)
        if not np.isfinite(self.arc_position_tolerance) or self.arc_position_tolerance<=0:
            raise ValueError('Arc position tolerance must be positive')
        self.arc_rng = np.random.default_rng(self.seed)
        # The caller already supplied exactly the native planner's point world.
        # Only its moving partitions are transported to each proposed arc goal.
        self.moving_mesh_name=None
        scene = Scene(mesh=[])
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
        torch.cuda.synchronize()
        return {'backend': 'curobo_v2', 'scene_setup_seconds': time.perf_counter()-started,
                'arc_id': self.active_arc_id, 'solver_mode': solver_mode,
                'optimizer_generation': self.arc_optimizer_generation,
                'optimizer_lifetime': 'one_arc' if solver_mode == 'local_continuation' else 'one_query',
                'solver_recreated': True, 'gpu_environment_collision_checked': True,
                'collision_owner': 'cuRobo IK and native MC-VAMP paths, shared observed points',
                'current_moving_surface_only': True,
                'local_optimization_dt_seconds': None,
                'local_seed_std_rad': self.arc_local_seed_std if solver_mode == 'local_continuation' else None,
                'velocity_regularization_weight': None,
                'velocity_regularization_source': 'upstream_optimizer_default',
                'seed_velocity_weight': 0.,
                'seed_penalty_reference': 'disabled; no zero-joint posture prior',
                'all_seeds_local': solver_mode == 'local_continuation',
                'observed_points': {key: len(value) for key, value in clouds.items()},
                'mesh_triangles': {},
                'scene_arrays_sha256': scene_arrays_sha256, 'native_admission_required': True,
                'map_method': 'same VAMP observed points through native cuRobo point distances',
                'hidden_object_geometry_used': False}

    def create_arc_solver(self):
        local = self.arc_solver_mode == 'local_continuation'
        # Load the shared robot spheres for FK calibration, with an empty GPU world.
        solver = self.create_solver(self.arc_robot_config, collision=True, scene_model=self.arc_scene,
            position_tolerance=float(getattr(self,'arc_position_tolerance',self.config.get('arc_position_tolerance_m', .006))),
            collision_cache={'mesh': 2},
            orientation_tolerance=self.arc_orientation_tolerance or (1e-6/.15))
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
        transform=np.eye(4);transform[:3,:3]=rotation.as_matrix();transform[:3,3]=translation
        if self.point_scene is None:
            raise ValueError('Configure the observed IK collision scene before an arc query')
        self.point_scene.move_target(transform)
        # Express the caller-selected attachment directly in cuRobo.
        # There is no soft orientation objective for a point-only target.
        orient_weights = {'point': [0.]*3, 'rigid': [1.]*3, 'jaw_axis': [0., 1., 1.]}[mode]
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
                'scene_collision_checked': self.point_scene is not None and self.point_scene.loaded,
                'self_collision_checked': True,
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
                elif op in {'initialize', 'fk', 'solve', 'solve_contact', 'configure_scene', 'move_scene_target', 'collision_diagnostics',
                            'configure_arc', 'arc_forward', 'solve_arc', 'end_arc'}:
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
