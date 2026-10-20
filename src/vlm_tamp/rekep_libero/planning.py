"""Native ReKep optimizers with a documented, narrow Panda compatibility layer.

Vendor files stay unchanged. Adaptations are seven-joint regularization,
5 mm collision margins, conservative RGB-D SDF queries, and a calibrated tool
frame. Detector grasps are fixed; no top-down grasp prior is evaluated.
"""

import ast
import hashlib
import types
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from vlm_tamp.libero import ROOT

from .geometry import (
    evaluate_collision, graspgen_to_tcp, interpolate_path, matrix_pose, pose_matrix,
    rekep_to_tcp, tcp_to_rekep, transform_points,
)
from .perception import _temporary_modules


@dataclass
class IKResult:
    success: bool
    num_descents: int
    position_error: float
    cspace_position: np.ndarray
    orientation_error: float = np.inf
    robot_collision_points_world: object = None
    robot_collision_finger_flags: object = None


def _ik_result(result):
    if isinstance(result, IKResult):
        return result
    if not isinstance(result, dict):
        result = vars(result)
    joints = np.asarray(result['cspace_position'], float)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError('Panda IK must return seven finite arm joints')
    return IKResult(bool(result['success']), int(result['num_descents']), float(result['position_error']), joints,
                    float(result.get('orientation_error', np.inf)), result.get('robot_collision_points_world'),
                    result.get('robot_collision_finger_flags'))


def _call_ik(ik, tcp_pose, joints=None, iterations=120):
    if hasattr(ik, 'ik'):
        result = ik.ik(np.asarray(tcp_pose).tolist(), seed_joints=None if joints is None else np.asarray(joints).tolist(),
                       max_iterations=iterations)
    else:
        result = ik(np.asarray(tcp_pose), initial_joint_pos=joints, max_iterations=iterations)
    return _ik_result(result)


class ClientIK:
    def __init__(self, client, calibration):
        self.client, self.calibration = client, calibration

    def solve(self, target_pose, max_iterations=20, initial_joint_pos=None):
        tcp = rekep_to_tcp(matrix_pose(target_pose), self.calibration)
        return _call_ik(self.client, tcp, initial_joint_pos, max_iterations)


def select_grasp(generated, keypoint_world, target_points, calibration, ik, sdf,
                 *, proximity_limit=.10, membership_limit=.035, collision_margin=.005):
    """Rank feasible detector poses by keypoint distance, then confidence.

    Membership refers to the TCP/contact region, rather than the detector's base
    origin. Width is not guessed: the checkpoint and physical Panda opening are
    fixed, and complete physical collision geometry comes from calibration.
    """
    poses, scores = generated['poses_world_base'], generated['scores']
    cloud = np.asarray(target_points, float)
    target_tree = cKDTree(cloud)
    points = np.asarray(calibration['tool_collision_points_tcp'], float)
    fingers = np.asarray(calibration['tool_collision_finger_flags'], bool)
    rejected, feasible = [], []
    for index in generated['retained_indices']:
        index = int(index)
        record = {'index': index, 'score': float(scores[index])}
        try:
            tcp = graspgen_to_tcp(poses[index], calibration)
            distance = float(np.linalg.norm(tcp[:3] - keypoint_world))
            membership = float(target_tree.query(tcp[:3])[0])
            record.update(keypoint_distance_m=distance, region_distance_m=membership)
            if distance > proximity_limit or membership > membership_limit:
                record['reason'] = 'target_membership_or_keypoint_proximity'
            else:
                result = _call_ik(ik, tcp)
                if not result.success:
                    record['reason'] = 'ik_unreachable'
                    record['ik_position_error_m'] = result.position_error
                else:
                    checks = evaluate_collision(sdf, transform_points(points, pose_matrix(tcp)), fingers,
                                                cloud, margin=collision_margin)
                    record['collision'] = checks
                    if not checks['accepted']:
                        record['reason'] = 'tool_collision_or_unknown'
                    elif result.robot_collision_points_world is not None:
                        arm = np.asarray(result.robot_collision_points_world, float)
                        arm_flags = np.asarray(result.robot_collision_finger_flags, bool)
                        arm_checks = evaluate_collision(sdf, arm, arm_flags, cloud, margin=collision_margin)
                        record['arm_collision'] = arm_checks
                        if not arm_checks['accepted']:
                            record['reason'] = 'arm_collision_or_unknown'
            if 'reason' in record:
                rejected.append(record)
            else:
                record['tcp_pose_world_xyzw'] = tcp.tolist()
                feasible.append(record)
        except (ValueError, FloatingPointError) as error:
            record['reason'] = 'invalid_candidate'
            record['detail'] = str(error)
            rejected.append(record)
    feasible.sort(key=lambda item: (item['keypoint_distance_m'], -item['score'], item['index']))
    chosen = feasible[0] if feasible else None
    return {'accepted': chosen is not None, 'selected_index': None if chosen is None else chosen['index'],
            'selected_tcp_pose': None if chosen is None else chosen['tcp_pose_world_xyzw'],
            'feasible_candidates': feasible, 'rejected_candidates': rejected,
            'failure': None if chosen is not None else 'grasp_no_feasible_detector_candidates',
            'thresholds': {'proximity_m': proximity_limit, 'membership_m': membership_limit,
                           'collision_margin_m': collision_margin}, 'width_prediction': 'not provided by checkpoint'}


def _numpy_fps(points, count):
    points = np.asarray(points)
    if len(points) <= count:
        return points.copy()
    indices = [0]
    distances = np.full(len(points), np.inf)
    for _ in range(1, count):
        distances = np.minimum(distances, np.sum((points - points[indices[-1]]) ** 2, axis=1))
        indices.append(int(np.argmax(distances)))
    return points[indices]


def _native_modules(collision_function):
    source = ROOT / 'third_party/ReKep'
    # Cached numba environments from dynamically imported vendor helpers can
    # refer to an unimportable '<dynamic>' module on a subsequent episode. Use
    # their unchanged pure NumPy bodies instead of creating vendor JIT caches.
    transform_tree = ast.parse((source / 'transform_utils.py').read_text())
    for node in ast.walk(transform_tree):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    transform = types.ModuleType('rekep_libero_native_transform')
    exec(compile(transform_tree, str(source / 'transform_utils.py'), 'exec'), transform.__dict__)
    names = {
        'normalize_vars', 'unnormalize_vars', 'consistency', 'transform_keypoints', 'batch_transform_points',
        'get_samples_jitted', 'path_length', 'angle_between_rotmat', 'angle_between_quats',
        'linear_interpolate_poses', 'get_linear_interpolation_steps',
    }
    tree = ast.parse((source / 'utils.py').read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        # These pure native bodies are retained, but executing as ordinary NumPy
        # avoids heavyweight unused scene libraries and persistent vendor caches.
        node.decorator_list = []
    utilities = types.ModuleType('utils')
    utilities.__dict__.update(np=np, T=transform, R=Rotation, Slerp=Slerp,
                              farthest_point_sampling=_numpy_fps, calculate_collision_cost=collision_function)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source / 'utils.py'), 'exec'), utilities.__dict__)

    class PandaJointRegularization(ast.NodeTransformer):
        def visit_Subscript(self, node):
            self.generic_visit(node)
            if isinstance(node.slice, ast.Slice) and node.slice.lower is None and node.slice.step is None:
                upper = node.slice.upper
                last = isinstance(upper, ast.UnaryOp) and isinstance(upper.op, ast.USub)
                last = last and isinstance(upper.operand, ast.Constant) and upper.operand.value == 1
                value = node.value
                joint_name = isinstance(value, ast.Attribute) and value.attr == 'cspace_position'
                joint_name |= isinstance(value, ast.Name) and value.id == 'reset_joint_pos'
                if last and joint_name:
                    return ast.copy_location(value, node)
            return node

    modules = []
    with _temporary_modules({'utils': utilities, 'transform_utils': transform}):
        for name in ['subgoal_solver', 'path_solver']:
            path = source / f'{name}.py'
            adapted = PandaJointRegularization().visit(ast.parse(path.read_text()))
            ast.fix_missing_locations(adapted)
            module = types.ModuleType(f'rekep_libero_{name}')
            exec(compile(adapted, str(path), 'exec'), module.__dict__)
            modules.append(module)
    return modules, {
        'subgoal_source_sha256': hashlib.sha256((source / 'subgoal_solver.py').read_bytes()).hexdigest(),
        'path_source_sha256': hashlib.sha256((source / 'path_solver.py').read_bytes()).hexdigest(),
        'adaptations': ['all seven Panda joints regularized', '5mm RGB-D collision margin',
                        'pure native NumPy helpers; deterministic NumPy farthest-point sampling',
                        'no synthetic out-of-workspace warmup', 'detector-fixed grasp; top-down prior disabled'],
    }


class PlanningAdapter:
    def __init__(self, ik, robot_calibration, bounds_min=(-.65, -.9, .65), bounds_max=(.65, .9, 1.8),
                 *, sampling_maxfun=5000, maxiter=200, collision_margin=.005):
        self.client, self.calibration = ik, robot_calibration
        self.ik = ClientIK(ik, robot_calibration)
        self.current_sdf = None
        self.contact_points = None
        self.margin = collision_margin
        self._tool_centered = None
        self._finger_flags = None

        def collision_cost(poses, _sdf_func, centered, _native_threshold):
            if len(poses) == 0:
                return 0.
            world = np.concatenate([transform_points(centered, pose) for pose in poses])
            flags = np.zeros(len(centered), bool)
            if self._tool_centered is not None:
                distance, index = cKDTree(self._tool_centered).query(centered)
                flags = self._finger_flags[index] & (distance < .002)
            flags = np.tile(flags, len(poses))
            values, _ = self.current_sdf.collision_values(world, self.margin, flags, self.contact_points)
            return float(np.maximum(values, 0).sum())

        (subgoal, path), self.metadata = _native_modules(collision_cost)
        common = {'bounds_min': list(bounds_min), 'bounds_max': list(bounds_max), 'sampling_maxfun': sampling_maxfun,
                  'max_collision_points': 60, 'constraint_tolerance': .0001,
                  'minimizer_options': {'maxiter': maxiter}}
        reset = np.asarray(robot_calibration.get('reset_joint_positions', [0, -.4, 0, -2.2, 0, 1.8, .8]), float)
        if reset.shape != (7,):
            raise ValueError('Reset regularization must contain all seven Panda arm joints')
        # Constructor warmups ask IK about artificial scenes; suppress that side
        # effect while retaining the native solve/objective implementations.
        subgoal.SubgoalSolver._warmup = lambda _self: None
        path.PathSolver._warmup = lambda _self: None
        self.subgoal = subgoal.SubgoalSolver(dict(common), self.ik, reset)
        self.path = path.PathSolver({**common, 'opt_pos_step_size': .20, 'opt_rot_step_size': .78,
                                    'opt_interpolate_pos_step_size': .02, 'opt_interpolate_rot_step_size': .10},
                                   self.ik, reset)
        self.subgoal._setup_sdf = lambda _voxels: self.current_sdf.query
        self.path._setup_sdf = lambda _voxels: self.current_sdf.query
        for solver in [self.subgoal, self.path]:
            original = solver._check_opt_result

            def checked(result, *args, _original=original):
                returned = _original(result, *args)
                debug = next(item for item in args if isinstance(item, dict))
                debug['accepted'] = bool(returned.success)
                return returned

            solver._check_opt_result = checked

    def solve_stage(self, stage, current_tcp_pose, joints, keypoints, movable_mask, sdf,
                    collisionpoints=None, *, fixed_grasp_tcp_pose=None, target_points=None, from_scratch=True):
        self.current_sdf, self.contact_points = sdf, target_points
        start = tcp_to_rekep(current_tcp_pose, self.calibration)
        start_matrix = pose_matrix(start)
        points_tcp = np.asarray(self.calibration['tool_collision_points_tcp'], float)
        flags = np.asarray(self.calibration['tool_collision_finger_flags'], bool)
        current_tool_world = transform_points(points_tcp, pose_matrix(current_tcp_pose))
        self._tool_centered = transform_points(current_tool_world, np.linalg.inv(start_matrix))
        self._finger_flags = flags
        if collisionpoints is None:
            collisionpoints = current_tool_world
        all_keypoints = np.r_[start[:3][None], np.asarray(keypoints, float)]
        movable_mask = np.asarray(movable_mask, bool)
        if movable_mask.shape == (len(keypoints),):
            movable_mask = np.r_[True, movable_mask]
        if movable_mask.shape != (len(all_keypoints),):
            raise ValueError('Movable-keypoint mask dimensions disagree')
        goal_functions = stage.subgoal_constraints if hasattr(stage, 'subgoal_constraints') else stage['subgoal_constraints']
        path_functions = stage.path_constraints if hasattr(stage, 'path_constraints') else stage['path_constraints']
        if fixed_grasp_tcp_pose is not None:
            goal = tcp_to_rekep(fixed_grasp_tcp_pose, self.calibration)
            subgoal_debug = {'accepted': True, 'type': 'detector_fixed_grasp', 'top_down_prior': False}
        else:
            goal, subgoal_debug = self.subgoal.solve(start, all_keypoints, movable_mask, goal_functions,
                path_functions, sdf.sdf_voxels, np.asarray(collisionpoints), False, np.asarray(joints), from_scratch)
        if not subgoal_debug.get('accepted', False):
            return {'accepted': False, 'failure': 'subgoal_optimizer_rejected', 'subgoal': subgoal_debug,
                    'path': None, 'poses_world_xyzw': []}
        path_poses, path_debug = self.path.solve(start, goal, all_keypoints, movable_mask, path_functions,
            sdf.sdf_voxels, np.asarray(collisionpoints), np.asarray(joints), from_scratch)
        if not path_debug.get('accepted', False):
            return {'accepted': False, 'failure': 'path_optimizer_rejected', 'subgoal': subgoal_debug,
                    'path': path_debug, 'poses_world_xyzw': []}
        dense_rekep = interpolate_path(np.r_[start[None], path_poses])
        dense_tcp = np.asarray([rekep_to_tcp(pose, self.calibration) for pose in dense_rekep])
        validation = self.validate_path(dense_tcp, all_keypoints, movable_mask, path_functions, sdf,
                                        joints, target_points, goal_functions if fixed_grasp_tcp_pose is None else [])
        return {'accepted': validation['accepted'], 'failure': validation['failure'], 'subgoal': subgoal_debug,
                'path': path_debug, 'validation': validation, 'poses_world_xyzw': dense_tcp.tolist(),
                'goal_tcp_pose': rekep_to_tcp(goal, self.calibration).tolist()}

    def validate_path(self, poses_tcp, keypoints, movable_mask, path_functions, sdf, joints,
                      target_points=None, goal_functions=()):
        start_rekep = pose_matrix(tcp_to_rekep(poses_tcp[0], self.calibration))
        centered_keypoints = np.asarray(keypoints, float).copy()
        centered_keypoints[movable_mask] = transform_points(centered_keypoints[movable_mask], np.linalg.inv(start_rekep))
        tool = np.asarray(self.calibration['tool_collision_points_tcp'], float)
        flags = np.asarray(self.calibration['tool_collision_finger_flags'], bool)
        limits = np.asarray(self.calibration['joint_limits'], float)
        checks, seed = [], np.asarray(joints)
        for index, pose_tcp in enumerate(poses_tcp):
            ik = _call_ik(self.client, pose_tcp, seed)
            if not ik.success or (ik.cspace_position < limits[:, 0]).any() or (ik.cspace_position > limits[:, 1]).any():
                return {'accepted': False, 'failure': 'path_ik_or_joint_limits', 'sample_index': index, 'checks': checks}
            seed = ik.cspace_position
            collision = evaluate_collision(sdf, transform_points(tool, pose_matrix(pose_tcp)), flags, target_points,
                                            margin=self.margin)
            if not collision['accepted']:
                return {'accepted': False, 'failure': 'path_tool_collision_or_unknown', 'sample_index': index,
                        'collision': collision, 'checks': checks}
            if ik.robot_collision_points_world is not None:
                arm = evaluate_collision(sdf, ik.robot_collision_points_world, ik.robot_collision_finger_flags,
                                         target_points, margin=self.margin)
                if not arm['accepted']:
                    return {'accepted': False, 'failure': 'path_arm_collision_or_unknown', 'sample_index': index,
                            'collision': arm, 'checks': checks}
            transformed = centered_keypoints.copy()
            transformed[movable_mask] = transform_points(centered_keypoints[movable_mask],
                pose_matrix(tcp_to_rekep(pose_tcp, self.calibration)))
            functions = list(path_functions) + (list(goal_functions) if index == len(poses_tcp) - 1 else [])
            residuals = [float(function(transformed[0], transformed[1:])) for function in functions]
            if any(not np.isfinite(value) or value > .0001 for value in residuals):
                return {'accepted': False, 'failure': 'path_constraint_residual', 'sample_index': index,
                        'residuals': residuals, 'checks': checks}
            checks.append({'sample_index': index, 'ik_position_error_m': ik.position_error,
                           'collision': collision, 'constraint_residuals': residuals})
        return {'accepted': True, 'failure': None, 'checks': checks}
