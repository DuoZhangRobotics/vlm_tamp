"""Sensor-only consistency checks for reviewed upright-door geometry.

No simulator object properties or task policy enter these functions. Annotated
image edges locate a line; the vertical-axis prior is explicit and inherited
from the existing upright-door tools. Consistency is not semantic certainty.
"""
import numpy as np


def vertical_edge_plane(camera, pixels_xy):
    points = np.asarray(pixels_xy, dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all() or np.linalg.norm(points[1]-points[0]) < 1:
        raise ValueError('Use two distinct finite pixels on the same visible attachment edge')
    k = np.asarray(camera['intrinsic'], float)
    transform = np.asarray(camera['camera_to_world'], float)
    if k.shape != (3, 3) or transform.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(transform).all():
        raise ValueError('Finite camera intrinsics and optical-camera-to-world calibration required')
    up = np.array([0., 0., 1.])
    vanishing = k @ transform[:3, :3].T @ up
    line = np.cross(np.r_[points.mean(0), 1.], vanishing)
    if np.linalg.norm(line[:2]) < 1e-9:
        raise ValueError('Vertical edge is not resolved in this camera')
    line /= np.linalg.norm(line[:2])
    normal = transform[:3, :3] @ k.T @ line
    normal /= np.linalg.norm(normal[:2])
    return {'image_line': line, 'normal_world': normal,
            'offset_world': float(normal @ transform[:3, 3]),
            'annotation_pixel_residuals': np.abs(np.c_[points, np.ones(2)] @ line)}


def triangulate_vertical_hinge(cameras, annotations, *, fit_cameras, reference_z_m):
    if len(set(fit_cameras)) < 2 or not set(fit_cameras) <= set(annotations):
        raise ValueError('Use at least two distinct annotated calibration views')
    if not np.isfinite(reference_z_m):
        raise ValueError('Finite sensor-derived reference height required')
    planes = {name: vertical_edge_plane(cameras[name], points) for name, points in annotations.items()}
    a = np.asarray([planes[name]['normal_world'][:2] for name in fit_cameras])
    b = np.asarray([planes[name]['offset_world']-planes[name]['normal_world'][2]*reference_z_m for name in fit_cameras])
    singular = np.linalg.svd(a, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-15))
    if singular[-1] < 1e-6 or condition > 20:
        raise ValueError('Camera edge planes are insufficiently independent for a stable hinge line')
    xy = np.linalg.lstsq(a, b, rcond=None)[0]
    origin = np.r_[xy, reference_z_m]
    checks = {}
    for name, plane in planes.items():
        camera = cameras[name]
        transform = np.asarray(camera['camera_to_world'], float)
        local = transform[:3, :3].T @ (origin-transform[:3, 3])
        if local[2] <= 0:
            raise ValueError('Triangulated hinge is behind a checking camera')
        image = np.asarray(camera['intrinsic']) @ local
        uv = image[:2] / image[2]
        checks[name] = {'used_for_fit': name in fit_cameras,
            'hinge_line_reprojection_error_px': float(abs(plane['image_line'] @ np.r_[uv, 1.])),
            'annotation_verticality_error_px': float(np.max(plane['annotation_pixel_residuals'])),
            'plane_residual_m': float(abs(plane['normal_world'] @ origin-plane['offset_world']))}
    return {'origin_world_m': origin.tolist(), 'axis_world_unsigned': [0., 0., 1.],
            'condition_number': condition, 'views': checks,
            'axis_prior': 'upright world-Z line; endpoint midpoint locates each annotated image edge',
            'reference_z_is_not_hinge_height_estimation': True,
            'simulator_object_geometry_used': False}


def cross_view_fit_agreement(first, second):
    first_normal = np.asarray(first['panel_normal_world'], float)
    second_normal = np.asarray(second['panel_normal_world'], float)
    first_normal /= np.linalg.norm(first_normal); second_normal /= np.linalg.norm(second_normal)
    a = np.asarray(first['source_anchor_world_m']); b = np.asarray(second['source_anchor_world_m'])
    return {'handle_center_xy_difference_mm': float(np.linalg.norm(a[:2]-b[:2])*1000),
            'signed_panel_normal_difference_deg': float(np.degrees(np.arccos(np.clip(first_normal @ second_normal, -1, 1)))),
            'scope': 'Agreement between separately annotated views sharing one RGB-D capture; not a calibrated accuracy guarantee.'}
