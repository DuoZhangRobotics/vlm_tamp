"""Full-arm observed collision checks with corrected contact-pair semantics."""

import numpy as np
from scipy.spatial import cKDTree

from .contact_policy_v8 import ContactPolicy
from .geometry import pose_matrix, transform_points
from .scene_state_v7 import ObservedScene as SceneV7, bounded_points, transformed_volume, observed_points, voxel_points, hull_volume


class ObservedScene(SceneV7):
    def __init__(self, cameras, metadata, spec, source_cloud, source_mask, main_camera,
                 parent_mask=None, target_cloud=None, config=None):
        super().__init__(cameras, metadata, spec, source_cloud, source_mask, main_camera,
                         parent_mask, target_cloud, config)
        self.policy = ContactPolicy(metadata['robot_collision_volumes_world'], spec, self.config)
        if spec['task_type'] == 'pick_place':
            floor = self.region.support_height
            centre = self.moving_reference.mean(0)
            radius = np.max(np.linalg.norm(self.moving_reference[:, :2] - centre[:2], axis=1)) + .01
            selected = ((np.abs(self.static_points[:, 2] - floor) <= .002)
                        & (np.linalg.norm(self.static_points[:, :2] - centre[:2], axis=1) <= radius))
            self.region.initial_support = self.static_points[selected]

    def record(self):
        result = super().record()
        result.update(schema_version='observed-scene-8.0', contact_policy_version='8.0',
                      selected_contact_policy=self.policy.record(self.region.task_type),
                      collision_exemptions='finger/selected-target, local drawer-panel brace, object/task-support')
        result['finger_noncontact_clearance_m'] = self.config.get('finger_noncontact_clearance_m', 0.)
        result['moving_payload_clearance_m'] = self.config.get('moving_payload_clearance_m', 0.)
        result['support_plane_frame'] = 'stationary world'
        return result

    def refresh(self, cameras, metadata, transform):
        if self.region.task_type != 'pick_place':
            return super().refresh(cameras, metadata, transform)
        transform = pose_matrix(transform)
        if not np.isfinite(transform).all():
            raise ValueError('Observed moving-part transform must be finite')
        points, report = observed_points(cameras, metadata, spacing=self.spacing,
                                         robot_depth_tolerance=self.depth_tolerance)
        canonical = transform_points(points, np.linalg.inv(transform))
        association = float(self.config.get('tracked_surface_association_m', .006))
        selected = ((self.region.body_tree.query(canonical)[0] <= association)
                    & (points[:, 2] > self.region.support_height + .0015))
        # The support plane stays in world coordinates when the bowl tilts.
        # Applying its cutoff in the rotated object frame left observed bowl
        # base points in the stationary map and made it collide with itself.
        if len(self.target_points):
            selected &= cKDTree(self.target_points).query(points)[0] > self.policy.support_tolerance
        if selected.sum() < 75:
            raise ValueError('Observed bowl update lacks sufficient selected-source support')
        self.moving_reference = voxel_points(np.vstack([self.moving_reference, canonical[selected]]), self.spacing)
        self.region.contact_reference = self.moving_reference.copy()
        self.payload_volume = hull_volume(self.moving_reference, 'predicted_moving_source')
        self.static_points, self.static_tree = points[~selected], cKDTree(points[~selected])
        self.observed_transform, self.self_filter = transform, report
        self.revision += 1
        return {'revision': self.revision, 'current_moving_supported_points': int(selected.sum()),
                'stationary_observed_points': len(self.static_points), 'support_plane_frame': 'stationary world',
                'source_association_m': association,
                'source': 'fresh RGB-D, bounded initial-source surface association and observed rigid transform',
                'newly_visible_selected_surfaces_fused': True, 'simulator_object_state_used': False}

    def check(self, volumes, tcp_pose, *, stage='approach', moving_transform=None, check_payload=False):
        volumes = list(volumes)
        if not self.required_robot_parts.issubset({v['robot_geom_name'] for v in volumes}):
            raise ValueError('Full-arm collision check is missing configured robot parts')
        transform = self.observed_transform if moving_transform is None else pose_matrix(moving_transform)
        moving = transform_points(self.moving_reference, transform)
        contacts = transform_points(self.region.contact_reference, transform)
        moving_tree = cKDTree(moving)
        collisions, permissions, tested, near_contacts = [], [], [], []
        for volume in volumes:
            if volume.get('fixed'):
                continue
            name = volume['robot_geom_name']
            tested.append(name)
            # Fingers must enter narrow apertures. A positive clearance is
            # logged as near-contact, not mislabeled as geometric collision.
            # Unrelated surfaces inside a finger remain blocking.
            margin = (float(self.config.get('finger_noncontact_clearance_m', 0.))
                      if name in self.policy.finger_names else self.margin)
            for part, points, tree in [('stationary_scene', self.static_points, self.static_tree),
                                        ('moving_source', moving, moving_tree)]:
                indices, values = bounded_points(points, tree, volume, max(margin, self.margin))
                if not len(indices):
                    continue
                hit = points[indices]
                target_allowed = (self.policy.robot_allowed(hit, volume, tcp_pose, contacts, stage)
                                  if part == 'moving_source' else np.zeros(len(hit), bool))
                panel_allowed = self.policy.local_panel_allowed(hit, volume, contacts, stage, transform)
                allowed = target_allowed | panel_allowed
                if target_allowed.any():
                    permissions.append({'robot_geom_name': name, 'scene_part': part, 'rule': 'finger_selected_target',
                                        'points': int(target_allowed.sum())})
                if (panel_allowed & ~target_allowed).any():
                    permissions.append({'robot_geom_name': name, 'scene_part': part, 'rule': 'finger_local_panel_support',
                                        'points': int((panel_allowed & ~target_allowed).sum())})
                proximity = (values > margin) & ~allowed
                if proximity.any():
                    near_contacts.append({'robot_geom_name': name, 'scene_part': part,
                                          'points': int(proximity.sum()),
                                          'minimum_positive_clearance_m': float(values[proximity].min())})
                blocked = (values <= margin) & ~allowed
                if blocked.any():
                    collisions.append({'robot_geom_name': name, 'scene_part': part,
                        'blocked_observed_points': int(blocked.sum()),
                        'minimum_signed_point_distance_m': float(values[blocked].min()),
                        'required_clearance_m': margin,
                        'maximum_point_penetration_m': max(0., float(-values[blocked].min())),
                        'example_points_world_m': hit[blocked][:3].tolist()})
        if check_payload:
            envelope = transformed_volume(self.payload_volume, transform)
            payload_margin = float(self.config.get('moving_payload_clearance_m', 0.))
            indices, values = bounded_points(self.static_points, self.static_tree, envelope, max(payload_margin, self.margin))
            hit = self.static_points[indices]
            placement = self.policy.support_allowed(hit, moving, self.target_points, stage)
            initial = (self.policy.support_allowed(hit, moving, self.region.initial_support, stage, initial=True)
                       if self.region.task_type == 'pick_place' else np.zeros(len(hit), bool))
            mechanism = np.zeros(len(hit), bool)
            if self.region.task_type != 'pick_place' and stage in {'grasp', 'pull'}:
                canonical = transform_points(hit, np.linalg.inv(transform))
                side = (canonical - self.region.origin) @ self.region.side
                tolerance = float(self.config.get('mechanism_seam_contact_band_m', .004))
                edge = ((np.abs(canonical[:, 2] - self.region.z_low) <= tolerance)
                        | (np.abs(canonical[:, 2] - self.region.z_high) <= tolerance)
                        | (np.abs(side - self.region.side_low) <= tolerance)
                        | (np.abs(side - self.region.side_high) <= tolerance))
                plane = np.abs((canonical - self.region.origin) @ self.region.axis) <= .008
                mechanism = edge & plane & (values >= -tolerance)
            allowed = placement | initial | mechanism
            for rule, mask in [('payload_target_support', placement), ('payload_initial_support', initial),
                               ('prismatic_boundary_contact', mechanism)]:
                if mask.any():
                    permissions.append({'robot_geom_name': 'predicted_moving_source_envelope',
                                        'scene_part': 'stationary_scene', 'rule': rule, 'points': int(mask.sum())})
            proximity = (values > payload_margin) & ~allowed
            if proximity.any():
                near_contacts.append({'robot_geom_name': 'predicted_moving_source_envelope',
                                      'scene_part': 'stationary_scene', 'points': int(proximity.sum()),
                                      'minimum_positive_clearance_m': float(values[proximity].min())})
            blocked = (values <= payload_margin) & ~allowed
            if blocked.any():
                collisions.append({'robot_geom_name': 'predicted_moving_source_envelope', 'scene_part': 'stationary_scene',
                    'blocked_observed_points': int(blocked.sum()),
                    'minimum_signed_point_distance_m': float(values[blocked].min()),
                    'required_clearance_m': payload_margin,
                    'maximum_point_penetration_m': max(0., float(-values[blocked].min())),
                    'example_points_world_m': hit[blocked][:3].tolist()})
        return {'accepted': not collisions, 'collisions': collisions, 'stage': stage,
                'contact_permissions_applied': permissions, 'local_contact_exemptions': sum(p['points'] for p in permissions),
                'nonblocking_positive_clearance': near_contacts,
                'checked_robot_parts': tested, 'payload_envelope_checked': bool(check_payload),
                'scene_revision': self.revision, 'moving_transform_from_reference': transform.tolist(),
                'unknown_space_validated': False, 'full_hidden_geometry_validated': False}
