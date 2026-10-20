"""Phase-specific touch-link pairs, independent of simulator scene identities.

Fingers may touch the selected grasped object over their functional geometry,
not only a small rubber-pad patch. Physics still establishes the grasp. Arms,
palm and unrelated objects do not inherit these permissions.
"""

import numpy as np
from scipy.spatial import cKDTree

from .geometry import pose_matrix, transform_points


CONTACT_STAGES = frozenset({'grasp', 'carry', 'pull', 'place', 'release', 'retreat_contact'})


class ContactPolicy:
    def __init__(self, robot_volumes, spec, config):
        self.finger_names = frozenset(v['robot_geom_name'] for v in robot_volumes
            if any(v['robot_geom_name'].endswith(suffix) for suffix in
                   ['finger1_collision', 'finger2_collision', 'finger1_pad_collision', 'finger2_pad_collision']))
        self.spec, self.config = spec, config
        self.region_tolerance = float(config.get('target_contact_association_m', .004))
        self.support_tolerance = float(config.get('support_contact_association_m', .004))
        if not self.finger_names or min(self.region_tolerance, self.support_tolerance) <= 0:
            raise ValueError('Known finger geometry and positive contact tolerances are required')

    def robot_allowed(self, points, volume, tcp_pose, contact_points, stage, *args, **kwargs):
        points = np.asarray(points, float).reshape(-1, 3)
        if stage not in CONTACT_STAGES or volume['robot_geom_name'] not in self.finger_names:
            return np.zeros(len(points), bool)
        contact_points = np.asarray(contact_points, float).reshape(-1, 3)
        if not len(points) or not len(contact_points):
            return np.zeros(len(points), bool)
        # Pair-specific contact permission, not a robot-wide collision bypass.
        return cKDTree(contact_points).query(points)[0] <= self.region_tolerance

    def local_panel_allowed(self, points, volume, contact_points, stage, transform):
        points = np.asarray(points, float).reshape(-1, 3)
        result = np.zeros(len(points), bool)
        if (self.spec['task_type'] != 'open_middle_drawer' or stage not in {'grasp', 'pull', 'release', 'retreat_contact'}
                or volume['robot_geom_name'] not in self.finger_names or not len(points)):
            return result
        contact_points = np.asarray(contact_points, float).reshape(-1, 3)
        if not len(contact_points):
            return result
        origin = np.asarray(self.spec['front_plane']['point_world_m'])
        normal = np.asarray(self.spec['pull_axis_world'])
        transform = pose_matrix(transform)
        moved_origin = transform_points([origin], transform)[0]
        moved_normal = transform[:3, :3] @ normal
        band = float(self.config.get('panel_contact_band_m', .004))
        on_local_panel = ((np.abs((points - origin) @ normal) <= band)
                          | (np.abs((points - moved_origin) @ moved_normal) <= band))
        radius = float(self.config.get('panel_contact_handle_radius_m', .05))
        near_selected_handle = cKDTree(contact_points).query(points)[0] <= radius
        # Neighboring protruding handles do not lie in this flat-panel band.
        return on_local_panel & near_selected_handle

    def support_allowed(self, points, moved_payload, support_points, stage, *, initial=False):
        points = np.asarray(points, float).reshape(-1, 3)
        support_points = np.asarray(support_points, float).reshape(-1, 3)
        phases = {'grasp', 'carry'} if initial else {'place', 'release', 'released'}
        if stage not in phases or not len(points) or not len(support_points):
            return np.zeros(len(points), bool)
        # Contact with the identified support is expected, even if an inferred
        # convex envelope touches above an arbitrary lowest-2-mm slice.
        return cKDTree(support_points).query(points)[0] <= self.support_tolerance

    def record(self, task_type):
        return {'schema_version': 'contact-policy-8.0', 'task_type': task_type,
                'compiler': 'known finger touch links + RGB-D task regions + stage',
                'allowed_robot_parts_for_selected_target': sorted(self.finger_names),
                'contact_stages': sorted(CONTACT_STAGES),
                'target_contact_association_m': self.region_tolerance,
                'local_panel_contact_band_m': self.config.get('panel_contact_band_m', .004),
                'local_panel_handle_radius_m': self.config.get('panel_contact_handle_radius_m', .05),
                'support_contact_association_m': self.support_tolerance,
                'palm_and_arm_contacts_allowed': False,
                'unrelated_handles_and_objects_allowed': False,
                'simulator_scene_ids_used': False, 'language_model_ignore_list_used': False,
                'simulator_contact_settings_changed': False}
