"""Known robot visual endpoints do not become scene obstacle surfaces.

Rendered robot visual meshes can extend outside their physical collision hulls.
This sensor self filter removes only robot-matched depth endpoints from surface
occupancy evidence. Measured free rays are kept and their occluded tails remain
unknown. The visual hulls never become a known-free workspace prior.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from .geometry import pose_matrix, transform_points


VISUAL_SOURCE = 'configured robot visual geometry, not scene geometry'
DEPTH_PRIOR_SOURCE = 'configured robot visual meshes projected at current proprioception only'


def robot_endpoint_cameras(cameras, visual_volumes):
    """Return copied camera payloads with separate occupancy depth maps."""
    visual_volumes = list(visual_volumes)
    prepared = []
    for volume in visual_volumes:
        if (volume.get('source') != VISUAL_SOURCE or volume.get('type') != 'convex'
                or volume.get('render_group') != 1):
            raise ValueError('Sensor self filtering requires configured current robot visual hulls')
        vertices = np.asarray(volume['vertices_world'], float)
        planes = np.asarray(volume['planes_world'], float)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or planes.ndim != 2 or planes.shape[1] != 4
                or not np.isfinite(vertices).all() or not np.isfinite(planes).all()
                or not len(vertices) or not len(planes)):
            raise ValueError('Invalid configured robot visual hull')
        prepared.append((vertices.min(axis=0), vertices.max(axis=0), planes))
    filtered, rows = [], []
    for index, camera in enumerate(cameras):
        depth = np.asarray(camera.get('depth', camera.get('depth_m')), float).squeeze()
        intrinsic = np.asarray(camera['intrinsic'], float)
        if depth.ndim != 2 or intrinsic.shape != (3, 3):
            raise ValueError('Calibrated depth map shape mismatch')
        if 'world_points' in camera:
            points = np.asarray(camera['world_points'], float)
            if points.shape != depth.shape + (3,):
                raise ValueError('Calibrated depth and world-point shapes must agree')
        else:
            y, x = np.indices(depth.shape)
            directions = np.stack([x, y, np.ones(depth.shape)], axis=-1) @ np.linalg.inv(intrinsic).T
            points = transform_points((directions * depth[..., None]).reshape(-1, 3), camera['camera_to_world'])
        points = points.reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0)
        matched = np.zeros(len(points), bool)
        # This tolerance labels measured endpoint identity only. It does not
        # expand any free-space prior or clear a proposed robot sweep.
        tolerance = 1e-5
        closer_scene_count = None
        if camera.get('known_robot_depth_prior') is not None:
            provenance = camera.get('robot_depth_prior', {})
            if (provenance.get('source') != DEPTH_PRIOR_SOURCE
                    or provenance.get('scene_geometry_used') is not False
                    or provenance.get('units') != 'meters'
                    or provenance.get('depth_convention') != 'optical-axis z'):
                raise ValueError('Robot depth prior must come only from configured current robot visual geometry')
            prior = np.asarray(camera['known_robot_depth_prior'], float).squeeze()
            if prior.shape != depth.shape:
                raise ValueError('Known robot depth prior must match observed optical depth shape')
            prior_valid = np.isfinite(prior) & (prior > 0)
            matched = (np.isfinite(depth) & (depth > 0) & prior_valid
                       & (np.abs(depth - prior) <= tolerance)).reshape(-1)
            closer_scene_count = int((np.isfinite(depth) & prior_valid & (depth < prior - tolerance)).sum())
            method = 'observed depth agrees with same-pixel known-robot-only render'
        else:
            for low, high, planes in prepared:
                near = (valid & ~matched & (points >= low - tolerance).all(axis=1)
                        & (points <= high + tolerance).all(axis=1))
                indices = np.flatnonzero(near)
                for offset in range(0, len(indices), 512):
                    chunk = indices[offset:offset + 512]
                    matched[chunk] = np.all(points[chunk] @ planes[:, :3].T + planes[:, 3] <= tolerance, axis=1)
            method = 'measured endpoint inside configured current robot visual hull'
        occupancy = depth.copy()
        occupancy.reshape(-1)[matched] = np.nan
        # Preserve original depth for carving known free rays before each hit.
        filtered.append({**camera, 'occupancy_depth': occupancy,
                         'robot_endpoint_mask': matched.reshape(depth.shape),
                         'robot_endpoint_free_ray_buffer_m': tolerance})
        rows.append({'camera_index': index, 'matched_robot_endpoints': int(matched.sum()),
                     'valid_depth_endpoints': int(valid.sum()),
                     'self_identification': method, 'closer_scene_endpoints_preserved': closer_scene_count,
                     'input_depth_dtype': str(np.asarray(camera.get('depth', camera.get('depth_m'))).dtype),
                     'input_world_points_dtype': None if 'world_points' not in camera else str(np.asarray(camera['world_points']).dtype),
                     'calibration_source': camera.get('source', 'provided calibrated sensor matrices')})
    return filtered, {'enabled': bool(visual_volumes) or any(camera.get('known_robot_depth_prior') is not None for camera in cameras),
                      'current_visual_hulls': len(visual_volumes),
                      'camera_counts': rows, 'endpoint_numeric_tolerance_m': 1e-5,
                      'matched_robot_free_ray_buffer_m': 1e-5,
                      'robot_buffer_basis': 'simulation calibrated depth/robot-model numerical precision; '
                                            'no hardware depth-noise model claimed',
                      'ordinary_scene_free_ray_buffer': 'unchanged 0.9 * voxel spacing',
                      'policy': 'robot-matched endpoints excluded only from occupied surfaces; '
                                'original free rays retained; occluded space remains unknown',
                      'visual_hulls_clear_unknown': False}


def build_sensor_field(field_type, cameras, bounds_min, bounds_max, voxel_size):
    """Versioned RGB-D fusion with separate free-ray and surface evidence."""
    low, high = np.asarray(bounds_min, float), np.asarray(bounds_max, float)
    if low.shape != (3,) or high.shape != (3,) or np.any(high <= low) or voxel_size <= 0:
        raise ValueError('Invalid observed collision field workspace')
    shape = np.ceil((high - low) / voxel_size).astype(int) + 1
    spacing = (high - low) / (shape - 1)
    axes = [np.linspace(a, b, n) for a, b, n in zip(low, high, shape)]
    centers = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    occupied, free = np.zeros(len(centers), bool), np.zeros(len(centers), bool)
    band = float(np.max(spacing) * .9)
    for camera in cameras:
        depth = np.asarray(camera.get('depth', camera.get('depth_m')), float).squeeze()
        occupancy_depth = np.asarray(camera.get('occupancy_depth', depth), float).squeeze()
        robot_pixels = np.asarray(camera.get('robot_endpoint_mask', np.zeros(depth.shape, bool)), bool)
        robot_buffer = float(camera.get('robot_endpoint_free_ray_buffer_m', band))
        intrinsic = np.asarray(camera['intrinsic'], float)
        inverse = np.linalg.inv(pose_matrix(camera['camera_to_world']))
        if (depth.ndim != 2 or occupancy_depth.shape != depth.shape or intrinsic.shape != (3, 3)
                or robot_pixels.shape != depth.shape or robot_buffer < 0 or not np.isfinite(robot_buffer)):
            raise ValueError('Depth/calibration shape mismatch')
        for offset in range(0, len(centers), 200000):
            xyz = transform_points(centers[offset:offset + 200000], inverse)
            projected, z = xyz @ intrinsic.T, xyz[:, 2]
            pixels = np.rint(projected[:, :2] / np.where(z > 0, z, 1)[:, None]).astype(int)
            u, v = pixels[:, 0], pixels[:, 1]
            visible = (z > 0) & (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
            indices = np.flatnonzero(visible)
            measured = depth[v[indices], u[indices]]
            valid = np.isfinite(measured) & (measured > 0)
            ray_buffer = np.where(robot_pixels[v[indices], u[indices]], robot_buffer, band)
            free[offset + indices[valid & (z[indices] < measured - ray_buffer)]] = True
            surface = occupancy_depth[v[indices], u[indices]]
            surface_valid = np.isfinite(surface) & (surface > 0)
            occupied[offset + indices[surface_valid & (np.abs(z[indices] - surface) <= band)]] = True
    occupied = occupied.reshape(tuple(shape))
    free = free.reshape(tuple(shape)) & ~occupied
    if not free.any():
        raise ValueError('RGB-D does not observe any free workspace')
    signed = (distance_transform_edt(~free, sampling=spacing)
              - distance_transform_edt(free, sampling=spacing)).astype(np.float32)
    return field_type(low, high, occupied, free, signed, spacing)
