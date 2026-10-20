"""Compile task roles into local, phase-specific contact permissions.

The language model never supplies robot geometry names or an unchecked ignore
matrix. These rules change planning admission only; simulator contact physics
and physical grasp attachment are unchanged.
"""

import numpy as np
from scipy.spatial import cKDTree

from .geometry import pose_matrix, transform_points


CONTACT_STAGES = frozenset({'grasp', 'carry', 'pull', 'place'})


class ContactPolicy:
    def __init__(self, robot_volumes, pad_tolerance=.003, region_tolerance=.003,
                 support_tolerance=.002):
        self.pad_names = frozenset(v['robot_geom_name'] for v in robot_volumes
                                   if v['robot_geom_name'].endswith('_pad_collision'))
        self.pad_tolerance = float(pad_tolerance)
        self.region_tolerance = float(region_tolerance)
        self.support_tolerance = float(support_tolerance)
        if not self.pad_names or min(self.pad_tolerance, self.region_tolerance,
                                     self.support_tolerance) <= 0:
            raise ValueError('Known robot pad geometry and positive contact tolerances are required')

    def robot_allowed(self, points, volume, tcp_pose, contact_points, stage, pad_volumes=(),
                      collision_values=None, clearance_margin=0.):
        points = np.asarray(points, float).reshape(-1, 3)
        allowed = np.zeros(len(points), bool)
        if stage not in CONTACT_STAGES:
            return allowed
        if volume['robot_geom_name'] not in self.pad_names:
            # A configured convex finger mesh can overlap its own pad box.
            # Exempt only points lying in that same physical inward-pad patch;
            # this never grants permission to the remaining finger body.
            name = volume['robot_geom_name']
            if 'finger' not in name or not name.endswith('_collision'):
                return allowed
            pad_name = name.removesuffix('_collision') + '_pad_collision'
            for pad in pad_volumes:
                if pad['robot_geom_name'] != pad_name:
                    continue
                inverse = np.linalg.inv(pose_matrix(tcp_pose))
                local = transform_points(points, inverse)
                vertices = transform_points(pad['vertices_world'], inverse)
                bounded = ((local[:, 1:] >= vertices.min(0)[1:] - .0005).all(1)
                           & (local[:, 1:] <= vertices.max(0)[1:] + .0005).all(1))
                if collision_values is not None:
                    # Exempt the near-pad clearance shell only for points that
                    # do not penetrate the actual finger mesh. Outer shoulder
                    # penetration never receives this extended tolerance.
                    buffer = .0005 + float(clearance_margin)
                    shell = ((local[:, 1:] >= vertices.min(0)[1:] - buffer).all(1)
                             & (local[:, 1:] <= vertices.max(0)[1:] + buffer).all(1)
                             & (np.asarray(collision_values) >= 0))
                    bounded |= shell
                allowed |= bounded & self.robot_allowed(points, pad, tcp_pose, contact_points, stage)
            return allowed
        contact_points = np.asarray(contact_points, float).reshape(-1, 3)
        if not len(points) or not len(contact_points):
            return allowed
        inverse = np.linalg.inv(pose_matrix(tcp_pose))
        local = transform_points(points, inverse)
        vertices = transform_points(volume['vertices_world'], inverse)
        # Panda closing axis is TCP x. Only the pad face directed towards the
        # gripper centre may contact the selected surface, not the finger mesh.
        face = vertices[:, 0].min() if vertices[:, 0].mean() > 0 else vertices[:, 0].max()
        return ((np.abs(local[:, 0] - face) <= self.pad_tolerance)
                & (cKDTree(contact_points).query(points)[0] <= self.region_tolerance))

    def support_allowed(self, points, moved_payload, support_points, stage, *, initial=False):
        points = np.asarray(points, float).reshape(-1, 3)
        allowed = np.zeros(len(points), bool)
        phases = {'grasp', 'carry'} if initial else {'place', 'released'}
        support_points = np.asarray(support_points, float).reshape(-1, 3)
        if stage not in phases or not len(points) or not len(support_points):
            return allowed
        bottom = float(np.min(np.asarray(moved_payload)[:, 2]))
        # Permission is for the lowest support band of the object and a
        # specific observed support region. It cannot clear a whole plate/table.
        return ((np.abs(points[:, 2] - bottom) <= self.support_tolerance)
                & (cKDTree(support_points).query(points)[0] <= self.region_tolerance))

    def record(self, task_type):
        return {'schema_version': 'contact-policy-7.0', 'compiler': 'known Panda pads + grounded task roles',
                'task_type': task_type, 'allowed_robot_parts': sorted(self.pad_names),
                'target_region': 'predicted selected source surface',
                'robot_contact_stages': sorted(CONTACT_STAGES),
                'pad_face_tolerance_m': self.pad_tolerance,
                'target_region_tolerance_m': self.region_tolerance,
                'support_band_tolerance_m': self.support_tolerance,
                'palm_arm_and_outer_finger_contacts_allowed': False,
                'finger_mesh_pad_overlap_policy': 'only the same inward pad patch, with a 0.5 mm tangential boundary tolerance',
                'nonpenetrating_pad_clearance_shell_policy': 'near-pad clearance shell may be exempted; penetrating outer finger points remain blocked',
                'language_model_ignore_list_used': False,
                'simulator_contact_settings_changed': False}
