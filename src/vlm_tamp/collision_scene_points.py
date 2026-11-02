"""The observed point partitions shared by cuRobo IK and native VAMP paths."""
from __future__ import annotations

from typing import Any

import numpy as np

from .contact_collision_evidence import selected_contact_points
from .rekep_libero.geometry import transform_points
from .rekep_libero.scene_state_v7 import voxel_points


def planner_points(scene: Any, world_robot: np.ndarray,
                   policy: str = 'veto') -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Prepare the existing role-specific samples; add no radius or solid fill."""
    if policy == 'advisory':
        return {name: np.empty((0, 3), np.float32) for name in ('stationary', 'moving', 'contact')}, {
            'stationary': len(scene.static_points), 'moving': len(scene.moving_reference),
            'contact_reference': len(scene.contact_reference)}
    selected = selected_contact_points(scene)
    inverse = np.linalg.inv(world_robot)
    clouds = {name: np.ascontiguousarray(transform_points(voxel_points(points, spacing), inverse), np.float32)
        for name, points, spacing in (
            ('stationary', scene.static_points, .0075),
            ('moving', scene.moving_reference[~selected], .005),
            ('contact', scene.moving_reference[selected], .003))}
    return clouds, {name: len(points) for name, points in clouds.items()}


def world_points(clouds: dict[str, np.ndarray], world_robot: np.ndarray) -> dict[str, np.ndarray]:
    """Express exactly the native planner's samples in cuRobo's world frame."""
    return {name: transform_points(points, world_robot).astype(np.float32) for name, points in clouds.items()}
