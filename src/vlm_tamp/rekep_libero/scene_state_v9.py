"""Extend a selected support's RGB-D seed over its connected observed surface."""

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from scipy.spatial import cKDTree

from .scene_state_v8 import ObservedScene as SceneV8
from .scene_state_v7 import observed_points, voxel_points, hull_volume
from .geometry import pose_matrix, transform_points


def connected_source_mask(points, seed, low, high, *, spacing=.003):
    """Own a current observed component, seeded by the tracked selected body."""
    points, seed = np.asarray(points), np.asarray(seed)
    indices = np.flatnonzero((points >= low).all(1) & (points <= high).all(1))
    candidates = points[indices]
    if not len(candidates):
        raise ValueError('Tracked source has no bounded observed candidates')
    cells = np.floor((candidates - low) / spacing).astype(int) + 1
    shape = np.ceil((high - low) / spacing).astype(int) + 3
    occupancy = np.zeros(shape, bool)
    occupancy[tuple(cells.T)] = True
    components, _ = label(binary_dilation(occupancy), generate_binary_structure(3, 3))
    ids = components[tuple(cells.T)]
    seed_hits = cKDTree(seed).query(candidates)[0] <= .006
    if seed_hits.sum() < 40:
        raise ValueError('Tracked source lacks observed component seed support')
    chosen = int(np.argmax(np.bincount(ids[seed_hits])))
    selected = np.zeros(len(points), bool)
    selected[indices[ids == chosen]] = True
    return selected, {'source': 'current measured component seeded by observed tracked source',
                      'seed_supported_points': int(seed_hits.sum()), 'selected_observed_points': int(selected.sum()),
                      'bounded_world_low_m': np.asarray(low).tolist(), 'bounded_world_high_m': np.asarray(high).tolist(),
                      'current_component_ownership': True, 'simulator_scene_geometry_used': False}


def expand_selected_support(points, seed, support_height, *, padding=.025, spacing=.003):
    points, seed = np.asarray(points, float), np.asarray(seed, float)
    low, high = seed.min(0) - padding, seed.max(0) + padding
    # The selected plate's seed is above the tabletop. Retain its raised rim
    # and side, without growing through the table into unrelated objects.
    low[2] = max(float(support_height) + .002, float(np.quantile(seed[:, 2], .01)) - .003)
    high[2] = float(np.quantile(seed[:, 2], .99)) + .025
    indices = np.flatnonzero((points >= low).all(1) & (points <= high).all(1))
    candidates = points[indices]
    if not len(candidates):
        raise ValueError('Selected support has no observed surface candidates')
    cells = np.floor((candidates - low) / spacing).astype(int) + 1
    shape = np.ceil((high - low) / spacing).astype(int) + 3
    occupancy = np.zeros(shape, bool)
    occupancy[tuple(cells.T)] = True
    components, _ = label(binary_dilation(occupancy), generate_binary_structure(3, 3))
    ids = components[tuple(cells.T)]
    seed_hits = cKDTree(seed).query(candidates)[0] <= .0035
    if seed_hits.sum() < 40:
        raise ValueError('Selected support lacks connected RGB-D seed evidence')
    selected_id = int(np.argmax(np.bincount(ids[seed_hits])))
    selected = candidates[ids == selected_id]
    return selected, {'source': 'RGB selected support seed expanded over connected measured surfaces',
        'original_seed_points': len(seed), 'expanded_observed_support_points': len(selected),
        'seed_supported_candidates': int(seed_hits.sum()), 'bounded_lateral_expansion_m': padding,
        'connectivity_voxel_m': spacing, 'minimum_selected_height_world_m': float(low[2]),
        'stationary_tabletop_height_world_m': float(support_height),
        'simulator_instance_ids_or_meshes_used': False}


class ObservedScene(SceneV8):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracking_reference = self.moving_reference.copy()
        self.support_expansion = {'applied': False}
        if self.region.task_type == 'pick_place':
            self.target_points, evidence = expand_selected_support(self.static_points, self.target_points,
                self.region.support_height, padding=float(self.config.get('support_region_growth_m', .025)))
            self.support_expansion = {'applied': True, **evidence}

    def refresh(self, cameras, metadata, transform):
        if self.region.task_type != 'pick_place':
            return super().refresh(cameras, metadata, transform)
        transform = pose_matrix(transform)
        if not np.isfinite(transform).all():
            raise ValueError('Observed moving-part transform must be finite')
        points, report = observed_points(cameras, metadata, spacing=self.spacing,
                                         robot_depth_tolerance=self.depth_tolerance)
        seed = transform_points(self.tracking_reference, transform)
        low, high = seed.min(0) - [.02, .02, .015], seed.max(0) + [.02, .02, .015]
        low[2] = max(low[2], self.region.support_height + .0015)
        # Do not absorb the selected stationary plate into the carried bowl.
        candidate = cKDTree(self.target_points).query(points)[0] > self.policy.support_tolerance
        masked, association = connected_source_mask(points[candidate], seed, low, high)
        selected = np.zeros(len(points), bool)
        selected[np.flatnonzero(candidate)[masked]] = True
        if selected.sum() < 75:
            raise ValueError('Observed bowl update lacks sufficient selected-source support')
        canonical = transform_points(points[selected], np.linalg.inv(transform))
        self.moving_reference = voxel_points(np.vstack([self.tracking_reference, canonical]), self.spacing)
        self.region.contact_reference = self.moving_reference.copy()
        self.payload_volume = hull_volume(self.moving_reference, 'predicted_moving_source')
        self.static_points, self.static_tree = points[~selected], cKDTree(points[~selected])
        self.observed_transform, self.self_filter = transform, report
        self.revision += 1
        return {'revision': self.revision, 'current_moving_supported_points': int(selected.sum()),
                'stationary_observed_points': len(self.static_points), 'support_plane_frame': 'stationary world',
                'source_association': association, 'initial_reference_expansion_recursive': False,
                'source': 'fresh RGB-D component ownership and observed source transform',
                'simulator_object_state_used': False}

    def record(self):
        result = super().record()
        result.update(schema_version='observed-scene-9.0', selected_support_region=self.support_expansion)
        return result

    def save(self, path):
        np.savez_compressed(Path(path), static_points=self.static_points.astype(np.float32),
                            moving_reference=self.moving_reference.astype(np.float32),
                            contact_reference=self.region.contact_reference.astype(np.float32),
                            observed_transform=self.observed_transform,
                            selected_support_reference=self.target_points.astype(np.float32))
