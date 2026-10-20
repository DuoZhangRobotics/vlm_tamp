"""Selected support contacts can also occur while leaving the pickup region."""

import numpy as np
from scipy.spatial import cKDTree

from .contact_policy_v8 import ContactPolicy as PolicyV8


class ContactPolicy(PolicyV8):
    def support_allowed(self, points, moved_payload, support_points, stage, *, initial=False):
        allowed = super().support_allowed(points, moved_payload, support_points, stage, initial=initial)
        if initial or stage != 'carry' or self.spec['task_type'] != 'pick_place':
            return allowed
        moving = np.asarray(moved_payload, float).reshape(-1, 3)
        points, support_points = np.asarray(points, float).reshape(-1, 3), np.asarray(support_points, float).reshape(-1, 3)
        if not len(moving) or not len(points) or not len(support_points):
            return allowed
        surface = self.spec['object_surface']
        local_pickup = (np.linalg.norm(moving[:, :2].mean(0) - surface['center_xy_m']) <= .025
                        and moving[:, 2].min() <= surface['low_z_m'] + .025)
        if local_pickup:
            # The selected destination can already touch a source at pickup.
            # Permit leaving that support; do not grant carry-wide exemptions.
            allowed |= cKDTree(support_points).query(points)[0] <= self.support_tolerance
        return allowed

    def record(self, task_type):
        result = super().record(task_type)
        result.update(schema_version='contact-policy-11.0',
            selected_destination_contact_during_local_pickup=task_type == 'pick_place',
            local_pickup_horizontal_radius_m=.025, local_pickup_height_above_initial_support_m=.025)
        return result
