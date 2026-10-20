"""Joint-space OMPL/MC-VAMP feasibility on an RGB-D estimated drawer line.

This smoke test uses calibrated robot FK and native VAMP self-collision checks.
It does not execute a drawer task or establish scene clearance or grasping.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.runtime/ompl-deps'), str(ROOT / '.runtime/mc-vamp-deps')]

from try_rekep_bowl_v3 import save
from vlm_tamp.pose_constraints import perpendicular_basis
from vlm_tamp.rekep_libero.geometry import pose_matrix, transform_points
from vlm_tamp.rekep_libero.object_motion_v3 import remove_robot_points
from vlm_tamp.rekep_libero.perception import _read_observation
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain, fixed_orientation_residual
from vlm_tamp.rekep_libero.task_geometry_v4 import joint_residual


def pose_ik(chain, target, seed, tries=8):
    target = pose_matrix(target)
    generator = np.random.default_rng(0)
    best = None
    for index in range(tries):
        initial = seed if index == 0 else np.clip(seed + generator.normal(0, .7, 7),
                                                  chain.limits[:, 0] + 1e-5, chain.limits[:, 1] - 1e-5)

        def residual(q):
            pose = pose_matrix(chain.forward(q))
            return np.r_[pose[:3, 3] - target[:3, 3],
                         .15 * Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec()]

        solved = least_squares(residual, initial, bounds=tuple(chain.limits.T), max_nfev=200,
                               xtol=1e-10, gtol=1e-10, ftol=1e-10)
        error = float(np.linalg.norm(residual(solved.x)))
        if best is None or error < best['weighted_error']:
            actual = pose_matrix(chain.forward(solved.x))
            best = {'joints': solved.x, 'weighted_error': error,
                    'position_error_m': float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
                    'orientation_error_rad': float(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude()),
                    'seeds_attempted': index + 1}
        if error < 1e-7:
            break
    return best


def make_problem(source, models, output):
    chain_record = json.loads((models / 'robot-chain.json').read_text())
    chain = RobotChain(chain_record)
    frames = json.loads((models / 'mc-vamp-frames.json').read_text())
    selected = json.loads((source / 'chosen-grasp.json').read_text())
    spec = json.loads((source / 'task-spec.json').read_text())
    start = np.asarray(selected['solution_joints'])
    start_pose = pose_matrix(chain.forward(start))
    axis = np.asarray(spec['pull_axis_world'])
    line = {**spec['path_constraint'], 'line_point_world_m': start_pose[:3, 3].tolist()}
    requested_goal = start_pose.copy()
    requested_goal[:3, 3] += spec['requested_travel_m'] * axis
    full = pose_ik(chain, requested_goal, start)
    # A shorter fixture isolates planner API compatibility from the selected
    # grasp orientation's workspace limit. It never counts as a task success.
    tested_travel = .10
    goal_pose = start_pose.copy()
    goal_pose[:3, 3] += tested_travel * axis
    goal = pose_ik(chain, goal_pose, start)
    if goal['weighted_error'] > 1e-7:
        raise ValueError('The shortened planner fixture has no accurately solved IK goal')
    save(output / 'problem.json', {'line': line, 'fixed_rotation_world': start_pose[:3, :3],
                                   'start_joints': start, 'goal_joints': goal['joints'],
                                   'requested_travel_m': spec['requested_travel_m'], 'tested_travel_m': tested_travel,
                                   'requested_goal_ik': full, 'tested_goal_ik': goal,
                                   'source_scene': source, 'attachment_binding_status': 'kinematic fixture; physical attachment unverified',
                                   'planner_fixture_is_task_success_test': False})
    return chain, frames, line, start_pose[:3, :3], start, np.asarray(goal['joints']), full


def evaluate(chain, line, fixed_rotation, joints):
    pose, jacobian = chain.forward(joints, jacobian=True)
    result = joint_residual(line, pose, jacobian)
    angular, angular_jacobian = fixed_orientation_residual(fixed_rotation, pose_matrix(pose)[:3, :3], jacobian[3:])
    return (np.r_[result['equality'], angular],
            np.vstack([result['equality_joint_jacobian'], angular_jacobian]), result['inequality'])


def path_metrics(path, chain, line, fixed_rotation, environment, vamp):
    results = [evaluate(chain, line, fixed_rotation, q) for q in path]
    return {'states': len(path),
            'maximum_line_error_m': float(max(np.linalg.norm(item[0][:2]) for item in results)),
            'maximum_orientation_error_rad': float(max(np.linalg.norm(item[0][2:]) for item in results)),
            'maximum_travel_bound_violation_m': float(max(np.max(item[2]) for item in results)),
            'all_states_self_collision_valid': all(vamp.panda.validate(q.tolist(), environment, True) for q in path),
            'all_states_within_calibrated_joint_limits': bool(np.all(path >= chain.limits[:, 0])
                                                             and np.all(path <= chain.limits[:, 1])),
            'observed_scene_collision_validated': False}


def ompl_plan(kind, chain, line, rotation, start, goal, environment, vamp):
    from ompl import base as ob, geometric as og, util as ou

    class Constraint(ob.Constraint):
        def __init__(self):
            super().__init__(7, 5, 1e-5)

        def function(self, values, out):
            out[:] = evaluate(chain, line, rotation, values)[0]

        def jacobian(self, values, out):
            out[:] = evaluate(chain, line, rotation, values)[1]

    # OMPL forbids RNG seed zero. The experiment's simulation seed remains 0.
    ou.RNG.setSeed(1)
    constraint = Constraint()
    constraint.setMaxIterations(100)
    ambient = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    for index, (low, high) in enumerate(chain.limits):
        bounds.setLow(index, float(low))
        bounds.setHigh(index, float(high))
    ambient.setBounds(bounds)
    space = (ob.ProjectedStateSpace if kind == 'projected' else ob.AtlasStateSpace)(ambient, constraint)
    information = ob.ConstrainedSpaceInformation(space)
    # Required explicitly by the 2.0.1 nanobind wheel before calling setup.
    space.setSpaceInformation(information)
    space.setDelta(.04)
    if kind == 'atlas':
        space.setRho(.15)
        space.setEpsilon(.01)
    space.setup()
    setup = og.SimpleSetup(information)

    def valid(state):
        joints = np.asarray([state[index] for index in range(7)])
        inequality = evaluate(chain, line, rotation, joints)[2]
        return bool(np.all(inequality <= 1e-6) and vamp.panda.validate(joints.tolist(), environment, True))

    setup.setStateValidityChecker(valid)
    initial, target = space.allocState(), space.allocState()
    initial.copy(start)
    target.copy(goal)
    if kind == 'atlas':
        space.anchorChart(initial)
        space.anchorChart(target)
    setup.setStartAndGoalStates(initial, target, 1e-6)
    planner = og.RRTConnect(information)
    planner.setRange(.15)
    setup.setPlanner(planner)
    before = time.perf_counter()
    status = setup.solve(10.)
    elapsed = time.perf_counter() - before
    exact = bool(setup.haveExactSolutionPath())
    if exact:
        solved = setup.getSolutionPath()
        solved.interpolate(150)
        path = np.asarray([[state[index] for index in range(7)] for state in solved.getStates()])
        checked = bool(solved.check())
    else:
        path, checked = np.empty((0, 7)), False
    metadata = {'planner': f'OMPL {kind} + RRTConnect', 'status': str(status),
                'exact_solution': exact, 'solve_seconds': elapsed,
                'ompl_path_check_passed': checked, 'maximum_solve_seconds': 10.,
                'constraint_tolerance': 1e-5, 'delta': .04, 'range': .15}
    # Do not manually free the nanobind-owned state objects. Each constrained
    # run is isolated in a process; the 2.0.1 wheel emits reference diagnostics
    # on shutdown, so a persistent production binding needs separate validation.
    metadata['ompl_rng_seed'] = 1
    return path, metadata


def mc_plan(chain, frames, line, rotation, start, goal, environment, vamp):
    axis = np.asarray(line['axis_world'])
    basis = perpendicular_basis(axis)
    world_line = np.eye(4)
    world_line[:3, :3] = np.column_stack([axis, basis])
    world_line[:3, 3] = line['line_point_world_m']
    world_robot = np.asarray(frames['robot_base_to_world'])
    robot_line = np.linalg.inv(world_robot) @ world_line
    line_eef = np.linalg.inv(robot_line) @ np.asarray(vamp.panda.eefk(start), float)[0]

    def packed(matrix):
        quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return np.r_[quat[3], quat[:3], matrix[:3, 3]].tolist()

    lower = [-.000001, -1e-5, -1e-5, -1e-5, -1e-5, -1e-5]
    upper = [line['maximum_travel_m'], 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
    task = vamp.panda.TaskSpaceConstraint([packed(line_eef)], [packed(robot_line)], lower, upper)
    constraints = vamp.panda.Composable_TaskSpaceConstraint(task)
    settings = vamp.CRRTCSettings()
    settings.rrtc_settings.max_iterations = 10000
    settings.rrtc_settings.max_samples = 100000
    settings.rrtc_settings.range = .15
    settings.constraint_settings.num_projection_iterations = 100
    settings.constraint_settings.std_dev_scaling_factor = .05
    before = time.perf_counter()
    result = vamp.panda.crrtc(start.tolist(), goal.tolist(), environment, settings, constraints, vamp.panda.halton())
    elapsed = time.perf_counter() - before
    sparse = np.asarray(result.path.numpy(), float)
    projected, raw_errors = [], []
    before_interpolation = time.perf_counter()
    for first, second in zip(sparse[:-1], sparse[1:]):
        samples = max(2, int(np.ceil(np.linalg.norm(second - first) / .003)))
        for weight in np.linspace(0., 1., samples, endpoint=False):
            raw = first + weight * (second - first)
            raw_errors.append(evaluate(chain, line, rotation, raw)[0])
            # The binding discards its C++ projection success flag. Check the
            # resulting residuals explicitly instead of assuming convergence.
            point = constraints.projectConfiguration(raw.astype(np.float32), 0, .5, 1., 100, False)
            projected.append(np.asarray(point, float))
    if len(sparse):
        projected.append(sparse[-1])
    path = np.asarray(projected, float).reshape(-1, 7)
    interpolation_seconds = time.perf_counter() - before_interpolation
    maximum_native = max((float(constraints.distanceToConstraint(q.astype(np.float32))) for q in path), default=0.)
    return path, {'planner': 'MC-VAMP constrained branch crrtc', 'exact_solution': bool(result.solved),
                  'solve_seconds': elapsed, 'native_planner_seconds': result.nanoseconds / 1e9,
                  'iterations': result.iterations, 'maximum_iterations': 10000,
                  'tsr_start_distance': float(task.distanceToConstraint(start.astype(np.float32))),
                  'tsr_goal_distance': float(task.distanceToConstraint(goal.astype(np.float32))),
                  'tsr_transverse_bounds_m': 1e-5, 'tsr_orientation_bounds_rad': 1e-5,
                  'raw_path_state_count': len(sparse), 'constraint_aware_resampling_used': True,
                  'resampling_seconds': interpolation_seconds,
                  'maximum_native_projection_residual': maximum_native,
                  'all_dense_projections_converged': maximum_native <= 1e-6,
                  'maximum_raw_linear_line_error_m': float(max((np.linalg.norm(e[:2]) for e in raw_errors), default=0.)),
                  'maximum_raw_linear_orientation_error_rad': float(max((np.linalg.norm(e[2:]) for e in raw_errors), default=0.)),
                  'maximum_dense_joint_step_rad': float(np.max(np.abs(np.diff(path, axis=0)))) if len(path) > 1 else 0.}


def depth_collision_diagnostic(source, frames, start, goal, vamp):
    cameras, metadata = _read_observation(source / 'opened/obs.npz', source / 'opened/metadata.json')
    world = np.concatenate([item['world_points'].reshape(-1, 3) for item in cameras.values()])
    world = world[np.isfinite(world).all(axis=1)]
    world = world[(world[:, 2] > .85) & (world[:, 2] < 1.7)
                  & (world[:, 0] > -1.05) & (world[:, 0] < .6)
                  & (np.abs(world[:, 1]) < .8)]
    world = remove_robot_points(world, metadata['robot_collision_volumes_world'])
    cells = np.floor(world / .006).astype(int)
    world = world[np.unique(cells, axis=0, return_index=True)[1]]
    points = transform_points(world, np.linalg.inv(np.asarray(frames['robot_base_to_world'])))
    # Native sphere radii from the compiled model set CAPT's pruning bounds.
    spheres = vamp.panda.fk(start.tolist())
    radii = [sphere.r for sphere in spheres]
    environment = vamp.Environment()
    environment.add_pointcloud(points.tolist(), min(radii), max(radii), .003)
    return {'observed_point_count': len(points), 'voxel_size_m': .006, 'point_radius_m': .003,
            'native_capt_accepts_rgbd_points': True,
            'start_admitted': bool(vamp.panda.validate(start.tolist(), environment, True)),
            'goal_admitted': bool(vamp.panda.validate(goal.tolist(), environment, True)),
            'selected_handle_contact_exemptions': False,
            'unknown_space_policy': 'not represented by surface point-cloud collision queries',
            'robot_sphere_model_calibrated_for_collision': False,
            'suitable_for_execution': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner', choices=['projected', 'atlas', 'mc-vamp'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', default='results/physical-tasks-v4-dev/automatic-01/goal00-state001')
    parser.add_argument('--models', default='results/planner-feasibility-v4/models-01')
    args = parser.parse_args()
    output, source = ROOT / args.output, ROOT / args.source
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve previous planner tests')
    output.mkdir(parents=True, exist_ok=True)
    import vamp
    chain, frames, line, rotation, start, goal, full = make_problem(source, ROOT / args.models, output)
    environment = vamp.Environment()
    native = mc_plan if args.planner == 'mc-vamp' else ompl_plan
    if args.planner == 'mc-vamp':
        path, result = native(chain, frames, line, rotation, start, goal, environment, vamp)
    else:
        path, result = native(args.planner, chain, line, rotation, start, goal, environment, vamp)
    if len(path):
        save(output / 'joint-path.json', {'joints': path})
        result['path_metrics'] = path_metrics(path, chain, line, rotation, environment, vamp)
    result.update(ompl_version=importlib.metadata.version('ompl'),
                  mc_vamp_revision='24e0f8adef0649a957fec77bf33f8c2f04515a18',
                  tested_travel_m=.10, requested_travel_m=.14,
                  full_requested_goal_ik=full, physical_execution_attempted=False,
                  collision_scope='native Panda self collision and joint limits; environment empty',
                  speed_comparison_fair=False,
                  timing_note='single compatibility run; Python FK callbacks versus compiled SIMD; not a performance benchmark')
    save(output / 'summary.json', result)
    save(output / 'observed-depth-collision.json', depth_collision_diagnostic(source, frames, start, goal, vamp))
    print(json.dumps({key: value for key, value in result.items() if key != 'full_requested_goal_ik'}), flush=True)


if __name__ == '__main__':
    main()
