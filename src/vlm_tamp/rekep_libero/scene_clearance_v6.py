"""Observed occupied-point admission using configured robot convex geometry.

This checks measured obstacle surfaces and intended inward pad contacts. It
does not certify unobserved space, held-object geometry, or continuous edges.
"""

import numpy as np
from scipy.spatial import cKDTree

from .geometry import pose_matrix, transform_points
from .sensor_filter_v3 import robot_endpoint_cameras


class ObservedClearance:
    def __init__(self, cameras, metadata, target_cloud, margin=.001, palm_only=False):
        filtered, self_report = robot_endpoint_cameras(list(cameras.values()), metadata.get('robot_visual_volumes_world', []))
        clouds = []
        for camera in filtered:
            points = np.asarray(camera['world_points']).reshape(-1, 3)
            valid = np.isfinite(points).all(axis=1) & ~np.asarray(camera['robot_endpoint_mask']).reshape(-1)
            clouds.append(points[valid])
        points = np.vstack(clouds)
        keep = ((points >= [-1.05, -.95, .65]).all(axis=1) & (points <= [.8, .95, 1.9]).all(axis=1))
        points = points[keep]
        cells = np.floor(points / .002).astype(int)
        self.points = points[np.unique(cells, axis=0, return_index=True)[1]]
        self.tree = cKDTree(self.points)
        self.target = np.asarray(target_cloud, float)
        self.target_tree = cKDTree(self.target)
        self.margin = margin
        self.palm_only = bool(palm_only)
        # The predicted source's observed bounded extent transfers to other
        # cameras. The original support-plane band is retained as an obstacle.
        low, high = self.target.min(axis=0) - .002, self.target.max(axis=0) + .002
        low[2] = self.target.min(axis=0)[2] + .0015
        self.source_points = ((self.points >= low).all(axis=1) & (self.points <= high).all(axis=1))
        self.report = {'observed_points': len(self.points), 'spacing_m': .002,
                       'clearance_margin_m': margin, 'current_robot_filter': self_report,
                       'unknown_space_policy': 'unobserved space is not validated by this occupied-surface check',
                       'collision_geometry': 'configured Panda convex collision volumes; no scene CAD',
                       'held_object_geometry_checked': False,
                       'admission_scope': 'palm only; other robot intersections diagnostic' if self.palm_only else 'all configured robot parts'}

    def check(self, volumes, tcp_pose, carrying=False):
        inverse_tcp = np.linalg.inv(pose_matrix(tcp_pose))
        collisions, exemptions = [], 0
        for volume in volumes:
            if volume.get('fixed') or volume['type'] != 'convex':
                continue
            vertices = np.asarray(volume['vertices_world'])
            low, high = vertices.min(axis=0) - self.margin, vertices.max(axis=0) + self.margin
            center, radius = (low + high) / 2, np.linalg.norm(high - low) / 2
            indices = np.asarray(self.tree.query_ball_point(center, radius), int)
            if carrying:
                indices = indices[~self.source_points[indices]]
            if not len(indices):
                continue
            points = self.points[indices]
            bounded = (points >= low).all(axis=1) & (points <= high).all(axis=1)
            points = points[bounded]
            if not len(points):
                continue
            planes = np.asarray(volume['planes_world'])
            inside = np.all(points @ planes[:, :3].T + planes[:, 3] <= self.margin, axis=1)
            collided = points[inside]
            if not len(collided):
                continue
            allowed = np.zeros(len(collided), bool)
            name = volume['robot_geom_name']
            if not carrying and 'finger' in name:
                local = transform_points(collided, inverse_tcp)
                local_vertices = transform_points(vertices, inverse_tcp)
                positive = local_vertices[:, 0].mean() > 0
                face = local_vertices[:, 0].min() if positive else local_vertices[:, 0].max()
                allowed = ((np.abs(local[:, 0] - face) <= .003)
                           & (self.target_tree.query(collided)[0] <= .003))
                exemptions += int(allowed.sum())
            if (~allowed).any():
                collisions.append({'robot_geom_name': name, 'blocked_observed_points': int((~allowed).sum())})
        admitted_collisions = ([row for row in collisions if row['robot_geom_name'] == 'gripper0_hand_collision']
                               if self.palm_only else collisions)
        return {'accepted': not admitted_collisions, 'collisions': admitted_collisions,
                'all_robot_intersections_diagnostic': collisions,
                'intended_inward_pad_contact_exemptions': exemptions,
                'carrying_source_removed': carrying, 'unknown_space_validated': False}

    def joints(self, sim, joints, carrying=False):
        geometry = sim.fk(joints, include_volumes=True)
        return self.check(geometry['robot_collision_volumes_world'], geometry['eef_pose_world_xyzw'], carrying)
