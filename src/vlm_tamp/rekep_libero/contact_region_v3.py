"""Narrow intended pad contacts to observed RGB-predicted object surfaces."""

import numpy as np

from .geometry import pose_matrix, transform_points


class RGBTargetContactField:
    """An SDF wrapper that can remove, never add, contact exemptions.

    A permitted pad contact must be occupied, have an object-mask pixel in at
    least one camera, and agree with that pixel's metric surface in both optical
    depth and world distance. Optical depth alone would confuse a nearby table
    with a bowl in an oblique view.
    """

    def __init__(self, sdf, cameras, predicted_masks, surface_tolerance=.003):
        tolerance = float(surface_tolerance)
        if not np.isfinite(tolerance) or not 0 < tolerance <= .003:
            raise ValueError('RGB object contact surface tolerance must be positive and at most 3mm')
        self.sdf = sdf
        self.tolerance = tolerance
        self.views = []
        for name, mask in predicted_masks.items():
            if name not in cameras:
                raise ValueError('Every predicted target mask needs a calibrated camera')
            camera = cameras[name]
            mask = np.asarray(mask, bool)
            depth = np.asarray(camera['depth'], float).squeeze()
            world = np.asarray(camera['world_points'], float)
            intrinsic = np.asarray(camera['intrinsic'], float)
            if mask.shape != depth.shape or world.shape != depth.shape + (3,) or intrinsic.shape != (3, 3):
                raise ValueError('Target mask and calibrated RGB-D shapes must agree')
            self.views.append({'name': name, 'mask': mask, 'depth': depth, 'world': world,
                               'intrinsic': intrinsic,
                               'inverse': np.linalg.inv(pose_matrix(camera['camera_to_world']))})
        if not self.views:
            raise ValueError('Intended target contacts require RGB-predicted object masks')

    def __getattr__(self, name):
        # Save preserves the numeric field; caller separately saves masks and
        # observation calibration so the contact policy can be reconstructed.
        return getattr(self.sdf, name)

    def surface_support(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        supported = np.zeros(len(points), bool)
        finite = np.isfinite(points).all(axis=1)
        for view in self.views:
            camera_points = transform_points(points, view['inverse'])
            z = camera_points[:, 2]
            projected = camera_points @ view['intrinsic'].T
            safe = np.where((z > 0) & np.isfinite(z), z, 1)
            pixels = np.rint(np.nan_to_num(projected[:, :2] / safe[:, None])).astype(int)
            x, y = pixels[:, 0], pixels[:, 1]
            in_view = (finite & ~supported & (z > 0) & (x >= 0) & (y >= 0)
                       & (x < view['depth'].shape[1]) & (y < view['depth'].shape[0]))
            indices = np.flatnonzero(in_view)
            if not len(indices):
                continue
            x, y = x[indices], y[indices]
            observed = view['depth'][y, x]
            surface = view['world'][y, x]
            agreement = (view['mask'][y, x] & np.isfinite(observed) & (observed > 0)
                         & np.isfinite(surface).all(axis=1)
                         & (np.abs(z[indices] - observed) <= self.tolerance + 1e-9)
                         & (np.linalg.norm(points[indices] - surface, axis=1) <= self.tolerance + 1e-9))
            supported[indices[agreement]] = True
        return supported

    def collision_values(self, points, margin=.005, finger_flags=None, target_points=None, contact_radius=.012):
        points = np.asarray(points, float).reshape(-1, 3)
        values, old_allowed = self.sdf.collision_values(points, margin, finger_flags, target_points, contact_radius)
        states = self.sdf.states(points)
        eligible = old_allowed & states['occupied'] & ~states['unknown']
        allowed = np.zeros(len(points), bool)
        if eligible.any():
            allowed[eligible] = self.surface_support(points[eligible])
        rejected = old_allowed & ~allowed
        if rejected.any():
            values[rejected] = np.maximum(self.sdf.query(points[rejected]) + margin, np.finfo(float).eps)
        # Unknown is always blocked, independently of the underlying old
        # implementation's interpolation or contact behavior.
        unknown = states['unknown']
        values[unknown] = np.maximum(values[unknown], margin + np.finfo(float).eps)
        allowed[unknown] = False
        return values, allowed

    def policy_metadata(self):
        return {'version': 'rgb-contact-region-3.0', 'mask_cameras': [view['name'] for view in self.views],
                'world_surface_tolerance_m': self.tolerance,
                'optical_depth_tolerance_m': self.tolerance,
                'policy': 'narrow existing occupied inward-pad target contacts using RGB mask and measured world surface',
                'unknown_exemptions': False, 'known_robot_prior_unchanged': True}
