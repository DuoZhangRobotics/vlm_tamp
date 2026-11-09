"""Bounded whole-sequence selection using the existing cuRobo and MC-VAMP solvers."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .coupled_storage import write_json
from .mcvamp_contact_numerical import NativeScene
from .rekep_libero.geometry import pose_matrix, transform_points
from .timed_execution import schedule


def scheduled_controls(paths: list[np.ndarray], speed: float, hold: int, release_steps: int) -> int:
    """Price the joined path with the exact executor schedule, including holds and release."""
    joined = np.vstack([path if index == 0 else path[1:] for index, path in enumerate(paths)])
    return len(schedule(joined, maximum_joint_speed=speed, endpoint_hold_steps=hold))+release_steps


def plan_object_transfer(
    ik: Any, scene: Any, record: dict, metadata: dict,
    metadata_at: Callable[[np.ndarray], dict], local_point: np.ndarray,
    local_axis: np.ndarray, axis_world: np.ndarray, plane_normal: np.ndarray,
    targets: list[tuple[str, np.ndarray, np.ndarray]], folder: Path,
    *, speed: float, hold: int, release_steps: int, available_controls: int,
    budget_seconds: float, width: int = 5,
) -> tuple[np.ndarray, list[dict], dict]:
    """Keep a bounded set of low-cost prefixes until the complete transfer is selected.

    All goals come from one existing native IK batch per retained prefix. Paths,
    joint limits, collision geometry and requested constraints are unchanged.
    Runtime errors propagate; empty solver results and budget rejection are
    ordinary search outcomes, not evidence of global geometric infeasibility.
    """
    started = time.perf_counter()
    deadline = started+budget_seconds
    seed = np.asarray(metadata['joint_positions'])
    frontier = [{'q': seed, 'paths': [], 'reports': [], 'controls': hold+release_steps,
                 'branch_indices': []}]
    trace = {'selection': 'lowest executor control cost among bounded complete native sequences',
             'retained_prefix_limit': width, 'available_control_steps': available_controls,
             'speed_rad_s': speed, 'hold_steps': hold, 'release_steps': release_steps,
             'budget_seconds': budget_seconds, 'global_optimality_claimed': False,
             'path_shortcut_used': False, 'goals_modified': False, 'levels': []}
    query_count = path_count = rejected_budget = 0
    timed_out = False
    for stage_index, (name, target, plane_point) in enumerate(targets):
        children, queries = [], []
        for parent_index, parent in enumerate(frontier):
            if time.perf_counter() >= deadline:
                timed_out = True
                break
            current = metadata if stage_index == 0 else metadata_at(parent['q'])
            hand = pose_matrix(current['eef_pose_world_xyzw'])
            with NativeScene(scene, record, current, parent['q'], 0., articulated=False) as context:
                context.set_object_carry(local_point, local_axis, axis_world, plane_point, plane_normal)
                ik.configure_scene(context)
                solutions = ik.solve_contact_many(
                    target, local_point, parent['q'], hand[:3, :3],
                    contact_normal_local=-local_axis, surface_normal_world=axis_world)
                query_count += 1
                attempts = []
                query = {'parent_branches': parent['branch_indices'], 'ik': copy.deepcopy(ik.events[-1]),
                         'attempts': attempts}
                queries.append(query)
                seen = set()
                for rank, solution in enumerate(solutions):
                    remaining = deadline-time.perf_counter()
                    if remaining <= 0:
                        timed_out = True
                        break
                    goal = np.asarray(solution['joints'])
                    key = tuple(goal)
                    if key in seen:
                        continue
                    seen.add(key)
                    path, report = context.plan(parent['q'], goal, constrained=True,
                                                seconds=min(8., remaining))
                    path_count += 1
                    attempt = {'ik_rank': rank, 'goal_joints': goal.tolist(), 'path_result': report}
                    attempts.append(attempt)
                    if report['status'] in {'native_worker_failed', 'native_worker_timeout', 'native_exception'}:
                        write_json(folder/'sequence-search.json', {**trace, 'failed_query': query})
                        raise RuntimeError('Native object transfer planner failed: '+report['status'])
                    if not report['exact_solution']:
                        continue
                    paths = [*parent['paths'], path]
                    controls = scheduled_controls(paths, speed, hold, release_steps)
                    attempt['prefix_controls_including_hold_and_release'] = controls
                    if controls > available_controls:
                        rejected_budget += 1
                        attempt['selection_outcome'] = 'prefix_exceeds_remaining_controls'
                        continue
                    binding = {
                        'source_point_world_m': transform_points([local_point], hand)[0].tolist(),
                        'target_point_world_m': target.tolist(), 'contact_local_m': local_point.tolist(),
                        'reference_frame': 'grasp-time_TCP; fixed through the prepared rigid transfer',
                        'source': 'observed object pose plus rigid prediction within prepared transfer',
                        'attachment_model': 'rigid', 'attachment_verified': False,
                        'contact_constraint': 'point', 'object_orientation_constrained': True,
                        'payload_collision_checked': False, 'task_success_predicted': False,
                        'additional_clearance_m': 0., 'tcp_pose_world_xyzw': ik.chain.forward(path[-1]).tolist(),
                        'carry_constraints': {
                            'object_axis_world': (hand[:3, :3]@local_axis).tolist(),
                            'object_axis_local': local_axis.tolist(), 'target_axis_world': axis_world.tolist(),
                            'support_plane_point_world_m': plane_point.tolist(),
                            'support_plane_normal_world': plane_normal.tolist(),
                            'scope': 'MC-VAMP whole-path axis and support half-space constraints',
                            'additional_clearance_m': 0.}}
                    result = {**report, 'object_point_goal': binding,
                              'requested_point_world_m': target.tolist(), 'grasp_required': True,
                              'physical_attachment_verified': False,
                              'ik_rank': rank, 'sequence_parent_index': parent_index}
                    children.append({'q': path[-1], 'paths': paths,
                                     'reports': [*parent['reports'], {'stage': name, 'result': result}],
                                     'controls': controls,
                                     'branch_indices': [*parent['branch_indices'], rank]})
        # Equal-cost paths retain their native proposal order; no pose perturbation or tie-breaking menu.
        children.sort(key=lambda item: item['controls'])
        frontier = children[:width]
        trace['levels'].append({'stage': name, 'queries': queries,
                               'retained': [{'controls': item['controls'], 'branches': item['branch_indices']}
                                            for item in frontier]})
        trace.update(ik_queries=query_count, path_queries=path_count, budget_rejected_prefixes=rejected_budget,
                     search_deadline_reached=timed_out, elapsed_seconds=time.perf_counter()-started)
        write_json(folder/'sequence-search.json', trace)
        if not frontier:
            break
    complete = bool(frontier and len(frontier[0]['paths']) == len(targets))
    if not complete:
        diagnostic = {**trace, 'status': 'object_transfer_search_deadline' if timed_out else
                      'object_transfer_candidates_exhausted', 'physical_infeasibility_proved': False,
                      'physical_motion_issued': False}
        error = ValueError('No complete transfer fitting the remaining control budget was found: '
                           +diagnostic['status'])
        error.planning_diagnostic = diagnostic
        raise error
    chosen = frontier[0]
    trace.update(status='solved', selected_branches=chosen['branch_indices'],
                 complete_controls_including_release=chosen['controls'])
    write_json(folder/'sequence-search.json', trace)
    for stage, path in zip(chosen['reports'], chosen['paths']):
        target_folder = folder/stage['stage']
        target_folder.mkdir()
        np.save(target_folder/'path.npy', path)
        write_json(target_folder/'result.json', stage['result'])
    combined = np.vstack([path if index == 0 else path[1:] for index, path in enumerate(chosen['paths'])])
    return combined, chosen['reports'], trace
