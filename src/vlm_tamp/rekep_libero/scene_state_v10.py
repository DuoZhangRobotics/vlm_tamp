"""Keep measured tabletop points out of the selected moving-body model."""

import numpy as np
from scipy.spatial import cKDTree

from .scene_state_v7 import hull_volume
from .scene_state_v9 import ObservedScene as SceneV9, connected_source_mask


def separate_source_from_support(points, seed, coarse_floor):
    points, seed = np.asarray(points, float), np.asarray(seed, float)
    low, high = np.quantile(seed, .01, axis=0), np.quantile(seed, .99, axis=0)
    low -= [.018, .018, .003]
    high += [.018, .018, .003]
    centre = np.median(seed[:, :2], axis=0)
    radius = np.linalg.norm(points[:, :2] - centre, axis=1)
    support = ((radius > .075) & (radius < .16)
               & (points[:, 2] >= coarse_floor - .004) & (points[:, 2] < low[2]))
    if support.sum() < 75:
        raise ValueError('Source/support separation lacks observed local support-plane evidence')
    upper_floor = float(np.quantile(points[support, 2], .99))
    low[2] = max(low[2], upper_floor + .0015)
    if low[2] >= high[2]:
        raise ValueError('Selected source lacks height separation from observed support')
    selected, evidence = connected_source_mask(points, seed, low, high)
    if selected.sum() < 75:
        raise ValueError('Source/support separation leaves insufficient selected surface')
    return selected, upper_floor, {'source': 'RGB source quantile bounds and observed local table-height upper band',
        'original_coarse_floor_m': float(coarse_floor), 'observed_table_upper_band_m': upper_floor,
        'source_minimum_selected_height_m': float(low[2]), 'table_band_observations': int(support.sum()),
        'source_observations': int(selected.sum()), 'source_component': evidence,
        'table_removed_from_world_collision_map': False, 'simulator_scene_geometry_used': False}


class ObservedScene(SceneV9):
    def __init__(self, cameras, metadata, spec, source_cloud, source_mask, main_camera,
                 parent_mask=None, target_cloud=None, config=None):
        super().__init__(cameras, metadata, spec, source_cloud, source_mask, main_camera,
                         parent_mask, target_cloud, config)
        self.source_support_separation = {'applied': False}
        if self.region.task_type == 'pick_place':
            points = np.vstack([self.static_points, self.moving_reference])
            selected, floor, evidence = separate_source_from_support(points, source_cloud, self.region.support_height)
            self.source_support_separation = {'applied': True, **evidence}
            self.moving_reference = points[selected]
            self.tracking_reference = self.moving_reference.copy()
            self.region.body_reference = self.moving_reference.copy()
            self.region.body_tree = cKDTree(self.moving_reference)
            self.region.contact_reference = self.moving_reference.copy()
            self.region.support_height = floor
            self.payload_volume = hull_volume(self.moving_reference, 'predicted_moving_source')
            self.static_points, self.static_tree = points[~selected], cKDTree(points[~selected])
            centre = self.moving_reference[:, :2].mean(0)
            radius = np.linalg.norm(self.moving_reference[:, :2] - centre, axis=1).max() + .01
            contact = ((self.static_points[:, 2] >= floor - .004)
                       & (self.static_points[:, 2] <= floor + .001)
                       & (np.linalg.norm(self.static_points[:, :2] - centre, axis=1) <= radius))
            self.region.initial_support = self.static_points[contact]

    def record(self):
        result = super().record()
        result.update(schema_version='observed-scene-10.0', source_support_separation=self.source_support_separation)
        return result
