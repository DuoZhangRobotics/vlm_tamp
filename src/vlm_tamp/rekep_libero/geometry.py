"""RGB-D occupancy, conservative signed distances, and explicit tool frames.

ReKep uses positive distances in occupied space and negative distances in known
free space. Unknown voxels are blocked; their distance is never an unsigned
nearest-surface distance presented as an SDF. Configured robot geometry may be
excluded as a known kinematic prior; simulator scene geometry never enters.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape == (4, 4):
        result = pose.copy()
    elif pose.shape == (7,):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        result[:3, 3] = pose[:3]
    else:
        raise ValueError("Pose must be xyz + quaternion xyzw or a 4x4 matrix")
    if not np.isfinite(result).all() or not np.allclose(result[3], [0, 0, 0, 1]):
        raise ValueError("Invalid homogeneous pose")
    if not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("Pose rotation must be orthonormal")
    if np.linalg.det(result[:3, :3]) < .999:
        raise ValueError("Pose rotation must be proper")
    return result


def matrix_pose(matrix):
    matrix = pose_matrix(matrix)
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]


def transform_points(points, transform):
    matrix = pose_matrix(transform)
    return np.asarray(points, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]


def graspgen_to_tcp(grasp_world, calibration):
    """The detector predicts a base-link pose, not a grasp-center/TCP pose."""
    return matrix_pose(pose_matrix(grasp_world) @ pose_matrix(calibration['graspgen_base_to_tcp']))


def tcp_to_rekep(pose_tcp, calibration):
    return matrix_pose(pose_matrix(pose_tcp) @ pose_matrix(calibration['tcp_to_rekep']))


def rekep_to_tcp(pose_rekep, calibration):
    return matrix_pose(pose_matrix(pose_rekep) @ np.linalg.inv(pose_matrix(calibration['tcp_to_rekep'])))


def interpolate_path(poses, position_step=.01, rotation_step=.08):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) < 2:
        raise ValueError("A path needs at least two world-frame xyzw poses")
    dense = [poses[0]]
    for start, end in zip(poses[:-1], poses[1:]):
        rotations = Rotation.from_quat([start[3:], end[3:]])
        angle = (rotations[0].inv() * rotations[1]).magnitude()
        count = max(1, int(np.ceil(np.linalg.norm(end[:3] - start[:3]) / position_step)),
                    int(np.ceil(angle / rotation_step)))
        fractions = np.arange(1, count + 1) / count
        quats = Slerp([0, 1], rotations)(fractions).as_quat()
        xyz = start[:3] + fractions[:, None] * (end[:3] - start[:3])
        dense.extend(np.c_[xyz, quats])
    return np.asarray(dense)


@dataclass
class ObservedSDF:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    occupied: np.ndarray
    free: np.ndarray
    sdf_voxels: np.ndarray
    spacing: np.ndarray
    known_robot_prior: object = None

    def __post_init__(self):
        if self.known_robot_prior is None:
            self.known_robot_prior = np.zeros(self.free.shape, bool)
        self.axes = tuple(np.linspace(a, b, n) for a, b, n in
                          zip(self.bounds_min, self.bounds_max, self.sdf_voxels.shape))
        # Out-of-grid space has no observations and is blocked.
        self._interpolator = RegularGridInterpolator(
            self.axes, self.sdf_voxels, bounds_error=False, fill_value=float(np.max(self.spacing)))

    @classmethod
    def from_rgbd(cls, cameras, bounds_min=(-1.05, -.95, .65), bounds_max=(.8, .95, 1.9),
                  voxel_size=.01, surface_band=None):
        """Project voxel centers into calibrated depth maps, carving only seen rays.

        Each camera contains depth (or depth_m), intrinsic, and camera_to_world.
        Surfaces observed by any camera override a free classification. Space
        behind a surface stays unknown unless another view directly sees it free.
        """
        low, high = np.asarray(bounds_min, float), np.asarray(bounds_max, float)
        if low.shape != (3,) or high.shape != (3,) or np.any(high <= low) or voxel_size <= 0:
            raise ValueError("Invalid SDF workspace")
        shape = np.ceil((high - low) / voxel_size).astype(int) + 1
        spacing = (high - low) / (shape - 1)
        axes = [np.linspace(a, b, n) for a, b, n in zip(low, high, shape)]
        centers = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        occupied = np.zeros(len(centers), bool)
        free = np.zeros(len(centers), bool)
        band = float(surface_band if surface_band is not None else np.max(spacing) * .9)
        for camera in cameras:
            depth = np.asarray(camera.get('depth', camera.get('depth_m')), float).squeeze()
            intrinsic = np.asarray(camera['intrinsic'], float)
            inverse = np.linalg.inv(pose_matrix(camera['camera_to_world']))
            if depth.ndim != 2 or intrinsic.shape != (3, 3):
                raise ValueError("Depth/calibration shape mismatch")
            for offset in range(0, len(centers), 200000):
                xyz = transform_points(centers[offset:offset + 200000], inverse)
                projected = xyz @ intrinsic.T
                z = xyz[:, 2]
                safe_z = np.where(z > 0, z, 1)
                pixels = np.rint(projected[:, :2] / safe_z[:, None]).astype(int)
                u, v = pixels[:, 0], pixels[:, 1]
                visible = (z > 0) & (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
                indices = np.flatnonzero(visible)
                d = depth[v[indices], u[indices]]
                valid = np.isfinite(d) & (d > 0)
                indices, d = indices[valid], d[valid]
                occupied[offset + indices[np.abs(z[indices] - d) <= band]] = True
                free[offset + indices[z[indices] < d - band]] = True
        occupied = occupied.reshape(tuple(shape))
        free = free.reshape(tuple(shape)) & ~occupied
        if not free.any():
            raise ValueError("RGB-D does not observe any free workspace")
        blocked = ~free
        inside = distance_transform_edt(blocked, sampling=spacing)
        outside = distance_transform_edt(free, sampling=spacing)
        signed = inside - outside
        return cls(low, high, occupied, free, signed.astype(np.float32), spacing)

    @property
    def unknown(self):
        return ~self.occupied & ~self.free

    def query(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        result = self._interpolator(points)
        result[~np.isfinite(points).all(axis=1)] = float(np.max(self.spacing))
        return result

    def exclude_observed_region(self, points, radius=.012, *, source):
        """Remove a known robot surface or an observed, physically attached part.

        Only occupied cells supported by the supplied geometry are removed.
        Unknown space is retained, so this is not a workspace-clearing shortcut.
        """
        if source not in {'known_robot_geometry', 'observed_attached_region'}:
            raise ValueError('Only known robot or observed attached geometry can be excluded')
        points = np.asarray(points, float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return 0
        occupied_indices = np.argwhere(self.occupied)
        centers = self.bounds_min + occupied_indices * self.spacing
        selected = cKDTree(points).query(centers)[0] <= radius
        indices = tuple(occupied_indices[selected].T)
        self.occupied[indices] = False
        self.free[indices] = True
        blocked = ~self.free
        self.sdf_voxels = (distance_transform_edt(blocked, sampling=self.spacing)
                           - distance_transform_edt(self.free, sampling=self.spacing)).astype(np.float32)
        self.__post_init__()
        return int(selected.sum())

    def exclude_known_robot(self, points, radius=.012):
        return self.exclude_observed_region(points, radius, source='known_robot_geometry')

    def exclude_known_robot_volumes(self, volumes):
        """Exclude exactly configured current robot collision volumes.

        A current robot occupies these cells, so they cannot be environment
        obstacles. This uses the robot's known convex collision geometry, not
        estimated spheres, scene meshes, or a waiver for an unknown arm sweep.
        Unknown cells outside these exact volumes remain blocked.
        """
        removed, prior = 0, np.zeros(self.free.shape, bool)
        for volume in volumes:
            if volume.get('source') != 'configured robot collision geometry' or volume.get('type') != 'convex':
                raise ValueError('Robot exclusion needs configured convex robot collision geometry')
            vertices = np.asarray(volume['vertices_world'], float)
            planes = np.asarray(volume['planes_world'], float)
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or planes.ndim != 2 or planes.shape[1] != 4
                    or not np.isfinite(vertices).all() or not np.isfinite(planes).all()):
                raise ValueError('Invalid robot collision volume')
            if np.any(np.linalg.norm(planes[:, :3], axis=1) < .5):
                raise ValueError('Invalid robot volume halfspace normals')
            low = np.maximum(0, np.ceil((vertices.min(axis=0) - self.bounds_min) / self.spacing).astype(int))
            high = np.minimum(np.asarray(self.free.shape) - 1,
                              np.floor((vertices.max(axis=0) - self.bounds_min) / self.spacing).astype(int))
            if np.any(high < low):
                continue
            indices = np.stack(np.meshgrid(*(np.arange(a, b + 1) for a, b in zip(low, high)), indexing='ij'),
                               axis=-1).reshape(-1, 3)
            centers = self.bounds_min + indices * self.spacing
            # Chunked halfspace evaluation avoids large limb-by-grid matrices.
            for offset in range(0, len(indices), 20000):
                chunk = centers[offset:offset + 20000]
                inside = np.all(chunk @ planes[:, :3].T + planes[:, 3] <= 1e-8, axis=1)
                accepted = tuple(indices[offset:offset + 20000][inside].T)
                prior[accepted] = True
        removed = int((~self.free & prior).sum())
        self.occupied[prior] = False
        self.free[prior] = True
        self.known_robot_prior |= prior
        blocked = ~self.free
        self.sdf_voxels = (distance_transform_edt(blocked, sampling=self.spacing)
                           - distance_transform_edt(self.free, sampling=self.spacing)).astype(np.float32)
        self.__post_init__()
        return removed

    def exclude_attached_region(self, points, radius=.012):
        return self.exclude_observed_region(points, radius, source='observed_attached_region')

    def states(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        valid = (np.isfinite(points).all(axis=1) & (points >= self.bounds_min).all(axis=1)
                 & (points <= self.bounds_max).all(axis=1))
        indices = np.rint((np.nan_to_num(points) - self.bounds_min) / self.spacing).astype(int)
        indices = np.clip(indices, 0, np.asarray(self.free.shape) - 1)
        cells = tuple(indices.T)
        occupied = valid & self.occupied[cells]
        free = valid & self.free[cells]
        return {'occupied': occupied, 'free': free, 'unknown': ~(occupied | free)}

    def collision_values(self, points, margin=.005, finger_flags=None, target_points=None, contact_radius=.012):
        points = np.asarray(points, float).reshape(-1, 3)
        values = self.query(points) + margin
        allowed = np.zeros(len(points), bool)
        if finger_flags is not None and target_points is not None and len(target_points):
            flags = np.asarray(finger_flags, bool)
            if flags.shape != (len(points),):
                raise ValueError("Finger-contact labels must match collision points")
            target = np.asarray(target_points, float).reshape(-1, 3)
            distances = cKDTree(target).query(points)[0]
            # Unknown cells cannot be excused as presumed target contacts.
            allowed = flags & self.states(points)['occupied'] & (distances <= contact_radius)
            values[allowed] = -margin
        return values, allowed

    def save(self, path):
        np.savez_compressed(path, bounds_min=self.bounds_min, bounds_max=self.bounds_max,
                            occupied=self.occupied, free=self.free, sdf_voxels=self.sdf_voxels,
                            spacing=self.spacing, known_robot_prior=self.known_robot_prior)


def evaluate_collision(sdf, tool_points_world, finger_flags=None, target_points=None, margin=.005):
    values, allowed = sdf.collision_values(tool_points_world, margin, finger_flags, target_points)
    return {'accepted': bool(np.all(values <= 0)), 'maximum_violation_m': float(np.max(values, initial=-1)),
            'contact_exemptions': int(allowed.sum()), 'blocked_points': int((values > 0).sum()),
            'unknown_points': int(sdf.states(tool_points_world)['unknown'].sum())}
