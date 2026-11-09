"""Observed point geometry through cuRobo's native sphere/box distance kernels.

A zero-size box has distance ||x - point||. This preserves VAMP's zero-radius
observed points without meshing, voxel inflation, or a new collision kernel.
Only the environment query applies phase-specific finger placement/permissions
and permitted palm contact; the original sphere array remains for self collision.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from curobo._src.geom.collision import SceneCollision, SceneCollisionCfg, create_collision_checker
from curobo._src.geom.collision.buffer_collision import CollisionBuffer


class ObservedPointScene(SceneCollision):
    """Keep native GPU obstacle storage alive as measured point clouds change."""

    def __init__(self, device: Any, native_indices: list[int], capacity: int) -> None:
        cfg = SceneCollisionCfg(device_cfg=device, cache={'primitive': capacity})
        ordinary = create_collision_checker(cfg)
        super().__init__(data=ordinary.data, checker=ordinary.checker, device_cfg=device)
        self.contact = create_collision_checker(cfg)
        # cuRobo initializes a cuboid cache to 1 cm boxes. Explicitly remove
        # those dimensions: our geometry consists of points, not occupied cubes.
        self.data.cuboids.dims.zero_()
        self.contact.data.cuboids.dims.zero_()
        self.capacity = capacity
        self.native_indices = native_indices
        self.batch_buffers: dict[tuple[int, bool, int, int], CollisionBuffer] = {}
        self.reference_clouds: dict[str, np.ndarray] = {}
        self.palm_contact = False
        self.fingertip_contact = False
        self.loaded = False
        self._masks: dict[tuple[int, bool], torch.Tensor] = {}
        self.check_fingers = False
        self.finger_coefficients = device.to_device(np.zeros((4, 4), np.float32))

    def _enabled(self, count: int, contact: bool) -> list[bool]:
        """Apply phase permissions without changing self-collision spheres."""
        indices = self.native_indices + [-1]*(count-len(self.native_indices))
        return [1 <= i < (59 if self.check_fingers else 55)
                and not (contact and ((self.palm_contact and 37 <= i < 55)
                         or (self.fingertip_contact and i in (56, 58)))) for i in indices]

    def update(self, clouds: dict[str, np.ndarray], palm_contact: bool,
               finger_model: dict | None = None, fingertip_contact: bool = False) -> None:
        """Load the same role-separated points used by native path planning."""
        self.reference_clouds = clouds
        self.palm_contact = bool(palm_contact)
        self.fingertip_contact = bool(fingertip_contact)
        self.check_fingers = finger_model is not None
        if finger_model is not None:
            self.finger_coefficients.copy_(self.device_cfg.to_device(finger_model['coefficients']))
        for (count, contact), mask in self._masks.items():
            allowed = self._enabled(count, contact)
            mask.copy_(torch.tensor(allowed, device=mask.device).reshape_as(mask))
        self._upload(self.data.cuboids, np.vstack([clouds['stationary'], clouds['moving']]))
        self._upload(self.contact.data.cuboids, clouds['contact'])
        self.loaded = True

    def _upload(self, storage: Any, points: np.ndarray) -> None:
        count = len(points)
        if count > self.capacity:
            raise ValueError(f'Observed scene has {count} points, GPU cache capacity is {self.capacity}')
        # Identity inverse rotation; a world point p has inverse translation -p.
        storage.inv_pose[0, :count, :3].copy_(self.device_cfg.to_device(-points))
        storage.enable[0].zero_()
        storage.enable[0, :count] = 1
        storage.count[0] = count

    def move_target(self, transform: np.ndarray) -> None:
        """Transport only the observed moving and contact partitions for a goal."""
        rotation, translation = transform[:3, :3], transform[:3, 3]
        clouds = self.reference_clouds
        moving = clouds['moving']@rotation.T+translation
        contact = clouds['contact']@rotation.T+translation
        self._upload(self.data.cuboids, np.vstack([clouds['stationary'], moving]))
        self._upload(self.contact.data.cuboids, contact)

    def _environment_spheres(self, spheres: torch.Tensor, contact: bool) -> torch.Tensor:
        count = spheres.shape[-2]
        key = (count, contact)
        if key not in self._masks:
            allowed = self._enabled(count, contact)
            self._masks[key] = torch.tensor(allowed, device=spheres.device).reshape(1, 1, count, 1)
        # These operations are captured once; coefficients and permission masks
        # update in place between free approach and contact, including warm graphs.
        if all(i in self.native_indices for i in (37,40,45,55,56,57,58)):
            base,first,second = [spheres[..., self.native_indices.index(i), :3] for i in (37,40,45)]
            u,v = first-base,second-base
            f = self.finger_coefficients
            centers = (base[..., None, :]+u[..., None, :]*f[:,0:1]
                       +v[..., None, :]*f[:,1:2]+torch.linalg.cross(u,v)[...,None,:]*f[:,2:3])
            spheres = spheres.clone()
            for j,i in enumerate(range(55,59)):
                index=self.native_indices.index(i)
                spheres[..., index, :3] = centers[...,j,:]
                spheres[..., index, 3] = f[j,3]
        radii = torch.where(self._masks[key], spheres[..., 3:4], -1.)
        return torch.cat([spheres[..., :3], radii], dim=-1)

    def get_sphere_distance(self, state: Any, collision_buffer: CollisionBuffer,
                            weight: torch.Tensor, activation_distance: torch.Tensor,
                            env_query_idx: torch.Tensor | None = None,
                            return_loss: bool = False) -> torch.Tensor:
        """Delegate distances and their gradients to cuRobo's native kernels."""
        spheres = state.robot_spheres
        # The installed Warp kernel flattens pairs into a signed 32-bit index.
        # Partition only the query batch; every sphere sees every observed point.
        size = max(1, (2**31-1)//(spheres.shape[1]*spheres.shape[2]*self.capacity))
        outputs = []
        for first in range(0, len(spheres), size):
            query = spheres[first:first+size]
            idx = None if env_query_idx is None else env_query_idx[first:first+size]
            costs = []
            for contact, checker in ((False, self), (True, self.contact)):
                key = (id(collision_buffer), contact, first, len(query))
                buffer = self.batch_buffers.get(key)
                if buffer is None or buffer.shape != query.shape[:-1]:
                    buffer = CollisionBuffer.from_shape(query.shape, self.device_cfg)
                    self.batch_buffers[key] = buffer
                costs.append(checker.checker.get_sphere_distance(checker.data,
                    self._environment_spheres(query, contact), buffer,
                    weight, activation_distance, idx, return_loss))
            outputs.append(costs[0]+costs[1])
        return torch.cat(outputs, dim=0)

    get_sphere_collision = get_sphere_distance
