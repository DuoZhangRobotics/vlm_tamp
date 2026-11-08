"""Object-point goals bound to a caller-declared rigid hand attachment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .rekep_libero.geometry import pose_matrix, transform_points
from .surface_contact_tools import plan_point_goal


def object_point_binding(
    metadata: dict, source_point_world_m: ArrayLike, target_point_world_m: ArrayLike,
    attachment_model: str,
) -> dict:
    """Express the current object point in the measured TCP frame without choosing a point."""
    if attachment_model != 'rigid':
        raise ValueError('Declare attachment_model=rigid for an object assumed held by the hand')
    source = np.asarray(source_point_world_m, float)
    target = np.asarray(target_point_world_m, float)
    if source.shape != (3,) or target.shape != (3,) or not (
        np.isfinite(source).all() and np.isfinite(target).all()
    ):
        raise ValueError('Supply finite current object and destination world points')
    tcp = pose_matrix(metadata['eef_pose_world_xyzw'])
    local = transform_points([source], np.linalg.inv(tcp))[0]
    return {
        'source_point_world_m': source.tolist(), 'target_point_world_m': target.tolist(),
        'contact_local_m': local.tolist(), 'reference_frame': 'measured_current_TCP',
        'attachment_model': attachment_model, 'attachment_verified': False,
        'source': 'caller-selected current object point and measured robot proprioception',
        'contact_constraint': 'point', 'object_orientation_constrained': False,
        'payload_collision_checked': False, 'task_success_predicted': False,
        'additional_clearance_m': 0.,
    }


def plan_object_point(
    ik: Any, scene: Any, record: dict, metadata: dict, source_point_world_m: ArrayLike,
    target_point_world_m: ArrayLike, attachment_model: str, folder: Path | str,
    *, budget_seconds: float = 30.,
) -> tuple[np.ndarray, dict]:
    """Plan the supplied material point's destination, leaving wrist orientation free.

    The caller supplies the current point and rigid-attachment assumption. This
    operation does not estimate object support, certify retention, or release it.
    """
    binding = object_point_binding(
        metadata, source_point_world_m, target_point_world_m, attachment_model)

    def solve_candidate() -> tuple[dict, list]:
        """Use native partial-pose IK on the object offset, without robot-landmark substitution."""
        solutions = ik.solve_contact_many(
            binding['target_point_world_m'], binding['contact_local_m'],
            np.asarray(metadata['joint_positions']),
            pose_matrix(metadata['eef_pose_world_xyzw'])[:3, :3])
        if ik.events:
            binding['ik_diagnostic'] = {key: ik.events[-1][key] for key in (
                'scene_collision_checked', 'self_collision_checked', 'minimum_position_error_m',
                'constraint_feasible_seeds', 'returned', 'elapsed_seconds') if key in ik.events[-1]}
        return binding, solutions

    path, report = plan_point_goal(
        ik, scene, record, metadata, target_point_world_m, folder, solve_candidate,
        budget_seconds=budget_seconds, goal_key='object_point_goal')
    report.update(grasp_required=True, physical_attachment_verified=False)
    return path, report
