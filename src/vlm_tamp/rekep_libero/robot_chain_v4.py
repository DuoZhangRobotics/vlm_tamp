"""Calibrated seven-revolute-joint FK/Jacobian without scene geometry."""

import numpy as np
from scipy.spatial.transform import Rotation

from vlm_tamp.pose_constraints import skew
from .geometry import matrix_pose


class RobotChain:
    """Space product-of-exponentials from robot-only reference axes/anchors."""

    def __init__(self, record):
        self.reference_joints = np.asarray(record['reference_joints'], float)
        self.axes = np.asarray(record['joint_axes_world'], float)
        self.anchors = np.asarray(record['joint_anchors_world_m'], float)
        self.reference_tcp = np.asarray(record['reference_tcp_world'], float)
        self.limits = np.asarray(record['joint_limits'], float)
        if self.reference_joints.shape != (7,) or self.axes.shape != (7, 3) or self.anchors.shape != (7, 3):
            raise ValueError('A calibrated Panda chain requires exactly seven joints')
        if not np.allclose(np.linalg.norm(self.axes, axis=1), 1, atol=1e-8):
            raise ValueError('Revolute joint axes must be normalized')

    def forward(self, joints, jacobian=False):
        q = np.asarray(joints, float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Expected seven finite Panda joint values')
        product = np.eye(4)
        axes, anchors = [], []
        for axis, anchor, delta in zip(self.axes, self.anchors, q - self.reference_joints):
            axes.append(product[:3, :3] @ axis)
            anchors.append(product[:3, :3] @ anchor + product[:3, 3])
            rotation = Rotation.from_rotvec(delta * axis).as_matrix()
            step = np.eye(4)
            step[:3, :3] = rotation
            step[:3, 3] = (np.eye(3) - rotation) @ anchor
            product = product @ step
        tcp = product @ self.reference_tcp
        pose = matrix_pose(tcp)
        if not jacobian:
            return pose
        axes, anchors = np.asarray(axes), np.asarray(anchors)
        position = np.cross(axes, tcp[:3, 3] - anchors).T
        body_rotation = tcp[:3, :3].T @ axes.T
        return pose, np.vstack([position, body_rotation])


def fixed_orientation_residual(reference_rotation, rotation, body_jacobian):
    """SO(3) logarithm and exact right-perturbation Jacobian, away from pi."""
    vector = Rotation.from_matrix(np.asarray(reference_rotation).T @ rotation).as_rotvec()
    angle = np.linalg.norm(vector)
    cross = skew(vector)
    coefficient = (1 / 12 + angle * angle / 720) if angle < 1e-5 else (
        1 / angle**2 - (1 + np.cos(angle)) / (2 * angle * np.sin(angle)))
    right_inverse = np.eye(3) + .5 * cross + coefficient * cross @ cross
    return vector, right_inverse @ body_jacobian
