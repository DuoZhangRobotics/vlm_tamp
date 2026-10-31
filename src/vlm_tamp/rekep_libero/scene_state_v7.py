"""Full-arm observed-surface collision checking with explicit movable geometry.

This is an inspectable geometric baseline. It checks all moving Panda collision
volumes, not only the palm. It does not certify unseen object interiors, unknown
space or continuous edges. The moving part and contact region come from RGB
masks and measured depth, never simulator instance identities or scene meshes.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from scipy.spatial import ConvexHull, cKDTree

from .contact_policy_v7 import ContactPolicy
from .geometry import pose_matrix, transform_points
from .sensor_filter_v3 import DEPTH_PRIOR_SOURCE, robot_endpoint_cameras


RENDER_PRIOR_SOURCE = 'configured robot rendered geometry at current proprioception only'


def voxel_points(points, spacing=.002):
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return points
    cells = np.floor(points / spacing).astype(np.int64)
    # Stable integer sorting preserves the same first sample per voxel and
    # lexicographic output order without structured-array flatten/sort copies.
    order=np.lexsort((cells[:,2],cells[:,1],cells[:,0]))
    ordered=cells[order]
    first=np.r_[True,np.any(ordered[1:]!=ordered[:-1],axis=1)]
    return points[order[first]]


def observed_points(cameras, metadata, *, spacing=.002, robot_depth_tolerance=.0001):
    clouds, extra, rows = [], 0, []
    for name, camera in cameras.items():
        points = np.asarray(camera['world_points']).reshape(-1, 3)
        depth = np.asarray(camera['depth']).squeeze()
        matched = np.zeros(depth.shape, bool)
        if camera.get('known_robot_depth_prior') is not None:
            provenance = camera.get('robot_depth_prior', {})
            if (provenance.get('source') not in {DEPTH_PRIOR_SOURCE, RENDER_PRIOR_SOURCE}
                    or provenance.get('scene_geometry_used') is not False
                    or provenance.get('units') != 'meters'
                    or provenance.get('depth_convention') != 'optical-axis z'):
                raise ValueError('Robot prior must come only from configured current robot geometry')
            prior = np.asarray(camera['known_robot_depth_prior']).squeeze()
            if prior.shape != depth.shape:
                raise ValueError('Robot-only depth prior and observation shapes differ')
            # 0.1 mm accommodates measured render/depth conversion differences;
            # no future robot volume clears scene observations or unknown space.
            adjusted = (np.isfinite(depth) & (depth > 0) & np.isfinite(prior) & (prior > 0)
                        & (np.abs(depth - prior) <= robot_depth_tolerance))
            extra += int((adjusted & (np.abs(depth - prior) > 1e-5)).sum())
            matched = adjusted
        else:
            filtered, _ = robot_endpoint_cameras([camera], metadata.get('robot_visual_volumes_world', []))
            matched = filtered[0]['robot_endpoint_mask']
        valid = (np.isfinite(points).all(1) & np.isfinite(depth.reshape(-1))
                 & (depth.reshape(-1) > 0) & ~matched.reshape(-1))
        clouds.append(points[valid])
        rows.append({'camera': name, 'matched_robot_endpoints': int(matched.sum()),
                     'depth_prior_source': camera.get('robot_depth_prior', {}).get('source')})
    points = np.vstack(clouds)
    keep = (points >= [-1.05, -.95, .65]).all(1) & (points <= [.8, .95, 1.9]).all(1)
    return voxel_points(points[keep], spacing), {'camera_counts': rows,
        'v7_robot_depth_tolerance_m': robot_depth_tolerance,
        'additional_numerically_matched_robot_endpoints': extra,
        'future_robot_sweep_used_for_self_filter': False}


def hull_volume(points, name):
    points = np.asarray(points, float).reshape(-1, 3)
    if len(points) < 4:
        raise ValueError('Insufficient observed geometry for a payload envelope')
    hull = ConvexHull(points)
    _, distinct = np.unique(np.round(hull.equations, 9), axis=0, return_index=True)
    return {'type': 'convex', 'fixed': False, 'robot_geom_name': name,
            'vertices_world': points[hull.vertices], 'planes_world': hull.equations[np.sort(distinct)],
            'source': 'convex envelope of predicted-mask-selected observed points'}


def transformed_volume(volume, transform):
    transform = pose_matrix(transform)
    planes = np.asarray(volume['planes_world'])
    normals = planes[:, :3] @ transform[:3, :3].T
    return {**volume, 'vertices_world': transform_points(volume['vertices_world'], transform),
            'planes_world': np.c_[normals, planes[:, 3] - normals @ transform[:3, 3]]}


def bounded_points(points, tree, volume, margin):
    if volume['type'] == 'sphere':
        center, radius = np.asarray(volume['center_world']), float(volume['radius_m'])
        indices = np.asarray(tree.query_ball_point(center, radius + margin), int)
        return indices, np.linalg.norm(points[indices] - center, axis=1) - radius
    if volume['type'] != 'convex':
        raise ValueError('Unsupported known robot collision geometry: ' + volume['type'])
    vertices = np.asarray(volume['vertices_world'])
    low, high = vertices.min(0) - margin, vertices.max(0) + margin
    indices = np.asarray(tree.query_ball_point((low + high) / 2, np.linalg.norm(high - low) / 2), int)
    near = points[indices]
    indices = indices[(near >= low).all(1) & (near <= high).all(1)]
    planes = np.asarray(volume['planes_world'])
    values = np.empty(len(indices))
    for offset in range(0, len(indices), 1024):
        batch = indices[offset:offset + 1024]
        values[offset:offset + 1024] = np.max(points[batch] @ planes[:, :3].T + planes[:, 3], axis=1)
    inside = values <= margin
    return indices[inside], values[inside]


class MovingRegion:
    """A sensor-derived region which moves independently of the stationary map."""

    def __init__(self, spec, reference, source_mask, camera, parent_mask=None):
        self.spec = spec
        self.contact_reference = np.asarray(reference, float)
        self.task_type = spec['task_type']
        self.initial_support = np.empty((0, 3))
        if self.task_type == 'pick_place':
            self.low = self.contact_reference.min(0) - .002
            self.high = self.contact_reference.max(0) + .002
            self.low[2] = np.quantile(self.contact_reference[:, 2], .01) - .0005
            self.source_mask = np.asarray(source_mask, bool)
            self.camera_inverse = np.linalg.inv(pose_matrix(camera['camera_to_world']))
            self.intrinsic = np.asarray(camera['intrinsic'])
            self.description = 'RGB source-mask frustum intersected with measured source bounds; support band excluded'
        else:
            self.origin = np.asarray(spec['front_plane']['point_world_m'])
            self.axis = np.asarray(spec['pull_axis_world'])
            self.up = np.array([0., 0., 1.])
            self.side = np.cross(self.axis, self.up)
            cloud = np.asarray(camera['world_points'])[np.asarray(parent_mask, bool)]
            cloud = cloud[np.isfinite(cloud).all(1)]
            side = (cloud - self.origin) @ self.side
            self.side_low, self.side_high = np.quantile(side, [.01, .99])
            centres = np.asarray(spec['observed_handle_centers_world_m'])
            self.z_low = (centres[0, 2] + centres[1, 2]) / 2 + .001
            self.z_high = (centres[1, 2] + centres[2, 2]) / 2 - .001
            self.normal_low, self.normal_high = -.006, .05
            self.description = 'middle front-panel band between adjacent observed handles, plus selected handle; hidden drawer body unmodelled'

    def expand_observed_source(self, points):
        """Grow the RGB source seed through connected measured surfaces.

        A dominant observed support plane bounds growth below; a small ROI
        bounds growth laterally. No object CAD or simulator segmentation enters.
        """
        centre = self.contact_reference.mean(0)
        radius = np.linalg.norm(points[:, :2] - centre[:2], axis=1)
        floor_candidates = points[(radius > .075) & (radius < .15)
                                  & (points[:, 2] > self.low[2] - .03) & (points[:, 2] < self.low[2] + .002)]
        if len(floor_candidates) < 75:
            raise ValueError('Source expansion lacks observed support-plane evidence')
        edges = np.arange(self.low[2] - .03, self.low[2] + .003, .001)
        counts, _ = np.histogram(floor_candidates[:, 2], edges)
        index = int(np.argmax(counts))
        band = floor_candidates[(floor_candidates[:, 2] >= edges[index])
                                & (floor_candidates[:, 2] < edges[index + 1])]
        self.support_height = float(np.median(band[:, 2]))
        low, high = self.low - [.018, .018, .003], self.high + [.018, .018, .003]
        low[2] = self.support_height + .0015
        candidate_indices = np.flatnonzero((points >= low).all(1) & (points <= high).all(1))
        candidates = points[candidate_indices]
        spacing = .003
        cells = np.floor((candidates - low) / spacing).astype(int) + 1
        shape = np.ceil((high - low) / spacing).astype(int) + 3
        occupancy = np.zeros(shape, bool)
        occupancy[tuple(cells.T)] = True
        components, _ = label(binary_dilation(occupancy), generate_binary_structure(3, 3))
        ids = components[tuple(cells.T)]
        seed = cKDTree(self.contact_reference).query(candidates)[0] <= .0035
        if seed.sum() < 40:
            raise ValueError('RGB-selected source has insufficient connected depth support')
        selected_id = int(np.argmax(np.bincount(ids[seed])))
        selected = ids == selected_id
        self.body_reference = candidates[selected]
        self.body_tree = cKDTree(self.body_reference)
        self.description = 'RGB source seed expanded over connected observed surfaces above an observed support plane; 2 cm bounded lateral growth'
        mask = np.zeros(len(points), bool)
        mask[candidate_indices[selected]] = True
        return mask

    def mask(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        if self.task_type != 'pick_place':
            local = points - self.origin
            side, normal = local @ self.side, local @ self.axis
            return ((side >= self.side_low) & (side <= self.side_high)
                    & (points[:, 2] >= self.z_low) & (points[:, 2] <= self.z_high)
                    & (normal >= self.normal_low) & (normal <= self.normal_high))
        if hasattr(self, 'body_tree'):
            return (self.body_tree.query(points)[0] <= .004) & (points[:, 2] > self.support_height + .001)
        bounded = (points >= self.low).all(1) & (points <= self.high).all(1)
        indices = np.flatnonzero(bounded)
        xyz = transform_points(points[indices], self.camera_inverse)
        projected = xyz @ self.intrinsic.T
        pixels = np.rint(projected[:, :2] / np.maximum(xyz[:, 2, None], 1e-12)).astype(int)
        u, v = pixels.T
        valid = ((xyz[:, 2] > 0) & (u >= 0) & (v >= 0)
                 & (u < self.source_mask.shape[1]) & (v < self.source_mask.shape[0]))
        accepted = np.zeros(len(points), bool)
        chosen = indices[valid]
        accepted[chosen] = self.source_mask[v[valid], u[valid]]
        return accepted


class ObservedScene:
    def __init__(self, cameras, metadata, spec, source_cloud, source_mask,
                 main_camera, parent_mask=None, target_cloud=None, config=None):
        self.config = config or {}
        self.margin = float(self.config.get('observed_clearance_margin_m', .001))
        self.spacing = float(self.config.get('scene_voxel_spacing_m', .002))
        self.depth_tolerance = float(self.config.get('robot_self_depth_tolerance_m', .0001))
        self.region = MovingRegion(spec, source_cloud, source_mask, cameras[main_camera], parent_mask)
        self.policy = ContactPolicy(metadata['robot_collision_volumes_world'])
        self.required_robot_parts = frozenset(v['robot_geom_name'] for v in metadata['robot_collision_volumes_world']
                                               if not v.get('fixed'))
        points, self.self_filter = observed_points(cameras, metadata, spacing=self.spacing,
                                                  robot_depth_tolerance=self.depth_tolerance)
        selected = (self.region.expand_observed_source(points) if spec['task_type'] == 'pick_place'
                    else self.region.mask(points))
        self.moving_reference = points[selected]
        if len(self.moving_reference) < 75:
            raise ValueError('Insufficient predicted moving-part geometry')
        self.payload_volume = hull_volume(self.moving_reference, 'predicted_moving_source')
        self.static_points = points[~selected]
        self.static_tree = cKDTree(self.static_points)
        self.target_points = np.empty((0, 3)) if target_cloud is None else np.asarray(target_cloud, float)
        self.observed_transform = np.eye(4)
        self.revision = 0
        self.guide_points = np.empty((0, 3))
        if spec['task_type'] == 'pick_place':
            self.region.contact_reference = self.moving_reference.copy()
            low = np.min(self.moving_reference[:, 2])
            centre = self.moving_reference.mean(0)
            radius = np.max(np.linalg.norm(self.moving_reference[:, :2] - centre[:2], axis=1)) + .003
            support = ((np.abs(self.static_points[:, 2] - low) <= .003)
                       & (np.linalg.norm(self.static_points[:, :2] - centre[:2], axis=1) <= radius))
            self.region.initial_support = self.static_points[support]
        else:
            # Expand the single-view RGB handle seed with measured protrusion
            # points from the other cameras. Contact on an observed underside
            # is still contact with the selected handle, not with the panel.
            low, high = np.asarray(source_cloud).min(0) - .004, np.asarray(source_cloud).max(0) + .004
            protrusion = (self.moving_reference - self.region.origin) @ self.region.axis
            handle = ((self.moving_reference >= low).all(1) & (self.moving_reference <= high).all(1)
                      & (protrusion >= .012))
            if handle.sum() < 75:
                raise ValueError('Selected handle lacks multi-view observed contact support')
            self.region.contact_reference = self.moving_reference[handle]
            local = self.static_points - self.region.origin
            side, normal = local @ self.region.side, local @ self.region.axis
            edges = ((np.abs(self.static_points[:, 2] - self.region.z_low) <= .002)
                     | (np.abs(self.static_points[:, 2] - self.region.z_high) <= .002)
                     | (np.abs(side - self.region.side_low) <= .002)
                     | (np.abs(side - self.region.side_high) <= .002))
            bounded = ((side >= self.region.side_low - .002) & (side <= self.region.side_high + .002)
                       & (self.static_points[:, 2] >= self.region.z_low - .002)
                       & (self.static_points[:, 2] <= self.region.z_high + .002)
                       & (np.abs(normal) <= .006))
            self.guide_points = self.static_points[edges & bounded]

    def record(self):
        return {'schema_version': 'observed-scene-7.0', 'revision': self.revision,
                'stationary_observed_points': len(self.static_points), 'moving_observed_points': len(self.moving_reference),
                'moving_region_policy': self.region.description, 'self_filter': self.self_filter,
                'clearance_margin_m': self.margin, 'voxel_spacing_m': self.spacing,
                'robot_collision_scope': 'all moving configured Panda links, palm, finger meshes and pads',
                'fixed_mounting_base_checked': False, 'selected_contact_policy': self.policy.record(self.region.task_type),
                'moving_payload_envelope_check_enabled': True, 'complete_hidden_object_shape_checked': False,
                'initial_prismatic_seam_contact_points': len(self.guide_points),
                'selected_contact_surface_points': len(self.region.contact_reference),
                'unknown_space_validated': False, 'continuous_collision_detection': False,
                'simulator_scene_geometry_used': False, 'scene_instance_ids_used': False,
                'moving_transform_from_reference': self.observed_transform.tolist()}

    def save(self, path):
        path = Path(path)
        np.savez_compressed(path, static_points=self.static_points.astype(np.float32),
                            moving_reference=self.moving_reference.astype(np.float32),
                            contact_reference=self.region.contact_reference.astype(np.float32),
                            observed_transform=self.observed_transform)

    def refresh(self, cameras, metadata, transform):
        transform = pose_matrix(transform)
        if not np.isfinite(transform).all():
            raise ValueError('Observed moving-part transform must be finite')
        points, report = observed_points(cameras, metadata, spacing=self.spacing,
                                         robot_depth_tolerance=self.depth_tolerance)
        selected = self.region.mask(transform_points(points, np.linalg.inv(transform)))
        # Rebuild from current depth. Old moving surfaces are never left behind
        # as stationary obstacles, and no whole cabinet/target is erased.
        self.static_points, self.static_tree = points[~selected], cKDTree(points[~selected])
        self.observed_transform, self.self_filter = transform, report
        self.revision += 1
        return {'revision': self.revision, 'current_moving_supported_points': int(selected.sum()),
                'stationary_observed_points': len(self.static_points),
                'source': 'fresh RGB-D and observed selected-surface transform',
                'simulator_object_state_used': False}

    def drawer_transform(self, displacement, closure_transform=None):
        if self.region.task_type == 'pick_place':
            raise ValueError('Drawer displacement is unavailable for pick-and-place')
        if not np.isfinite(displacement) or not -.005 <= displacement <= .185:
            raise ValueError('Drawer displacement exceeds observed-task planning bounds')
        transform = np.eye(4) if closure_transform is None else pose_matrix(closure_transform).copy()
        transform[:3, 3] += float(displacement) * self.region.axis
        return transform

    def check(self, volumes, tcp_pose, *, stage='approach', moving_transform=None, check_payload=False):
        volumes = list(volumes)
        if hasattr(self, 'required_robot_parts'):
            supplied = {v['robot_geom_name'] for v in volumes}
            if not self.required_robot_parts.issubset(supplied):
                raise ValueError('Full-arm collision check is missing configured robot parts')
        transform = self.observed_transform if moving_transform is None else pose_matrix(moving_transform)
        moving = transform_points(self.moving_reference, transform)
        contacts = transform_points(self.region.contact_reference, transform)
        moving_tree = cKDTree(moving)
        collisions, exemptions, tested = [], 0, []
        pads = [volume for volume in volumes if volume['robot_geom_name'] in self.policy.pad_names]
        for volume in volumes:
            if volume.get('fixed'):
                continue
            tested.append(volume['robot_geom_name'])
            for name, points, tree in [('stationary_scene', self.static_points, self.static_tree),
                                        ('moving_source', moving, moving_tree)]:
                indices, values = bounded_points(points, tree, volume, self.margin)
                if not len(indices):
                    continue
                hit = points[indices]
                allowed = (self.policy.robot_allowed(hit, volume, tcp_pose, contacts, stage, pads, values, self.margin)
                           if name == 'moving_source' else np.zeros(len(hit), bool))
                exemptions += int(allowed.sum())
                if (~allowed).any():
                    collisions.append({'robot_geom_name': volume['robot_geom_name'], 'scene_part': name,
                        'blocked_observed_points': int((~allowed).sum()),
                        'maximum_point_penetration_m': max(0., float(-values[~allowed].min())),
                        'example_points_world_m': hit[~allowed][:3].tolist()})
        if check_payload:
            envelope = transformed_volume(self.payload_volume, transform)
            indices, values = bounded_points(self.static_points, self.static_tree, envelope, self.margin)
            hit = self.static_points[indices]
            allowed = self.policy.support_allowed(hit, moving, self.target_points, stage)
            if self.region.task_type == 'pick_place':
                allowed |= self.policy.support_allowed(hit, moving, self.region.initial_support, stage, initial=True)
            elif len(getattr(self, 'guide_points', [])) and stage in {'grasp', 'pull'}:
                displacement = float(transform[:3, 3] @ self.region.axis)
                if abs(displacement) <= .003:
                    # Only initial front seams at a measured mechanism boundary
                    # may touch the panel envelope; deeper cabinet penetration
                    # and robot/cabinet contacts are not exempted.
                    allowed |= ((cKDTree(self.guide_points).query(hit)[0] <= .002)
                                & (values >= -.0002))
            exemptions += int(allowed.sum())
            if (~allowed).any():
                collisions.append({'robot_geom_name': 'predicted_moving_source_envelope', 'scene_part': 'stationary_scene',
                    'blocked_observed_points': int((~allowed).sum()),
                    'maximum_point_penetration_m': max(0., float(-values[~allowed].min())),
                    'example_points_world_m': hit[~allowed][:3].tolist()})
        return {'accepted': not collisions, 'collisions': collisions, 'stage': stage,
                'local_contact_exemptions': exemptions, 'checked_robot_parts': tested,
                'payload_envelope_checked': bool(check_payload), 'scene_revision': self.revision,
                'moving_transform_from_reference': transform.tolist(), 'unknown_space_validated': False}

    def joints(self, sim, joints, *, gripper_qpos=None, **kwargs):
        geometry = sim.fk(joints, include_volumes=True, gripper_qpos=gripper_qpos)
        return self.check(geometry['robot_collision_volumes_world'], geometry['eef_pose_world_xyzw'], **kwargs)

    def planner_state(self, robot_geometry, joints, *, gripper_qpos, stage,
                      drawer_displacement=None, source_transform=None):
        """Geometry adapter for future (arm joints, drawer displacement) search.

        Task manifold/attachment residuals remain a separate planner constraint;
        this method supplies geometry validity, not a complete planner.
        """
        if self.region.task_type == 'open_middle_drawer':
            if drawer_displacement is None or source_transform is not None:
                raise ValueError('Drawer collision state requires an explicit prismatic displacement')
            transform = self.drawer_transform(drawer_displacement)
        else:
            if drawer_displacement is not None:
                raise ValueError('A bowl collision state cannot include drawer displacement')
            transform = self.observed_transform if source_transform is None else pose_matrix(source_transform)
        return self.joints(robot_geometry, joints, gripper_qpos=gripper_qpos, stage=stage,
                           moving_transform=transform, check_payload=stage in {'carry', 'pull', 'place'})
