"""Pinned upstream Panda spheres shared by native planning and GPU IK.

Indices follow third_party/mc-vamp/resources/panda/panda_spherized.urdf.
The fixed base is omitted from environment checks. Free approaches include
fingers at the measured aperture; contact motion exempts fingers. Self collision
stays VAMP's. No sphere radii or observed obstacles are inflated.
"""
import numpy as np
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

ARM_SPHERE_RANGES = ((1, 5), (5, 9), (9, 13), (13, 17), (17, 29), (29, 32), (32, 37))
PALM_SPHERE_RANGE = (37, 55)
FINGER_SPHERE_RANGE = (55, 59)
ENVIRONMENT_SPHERE_INDICES = tuple(range(1, 55))
HAND_BASIS_INDICES = (37, 40, 45)


@lru_cache(maxsize=2)
def finger_mesh_spheres(index: int) -> np.ndarray:
    """Read the upstream finger spheres in their original link frames."""
    path = Path(__file__).resolve().parents[2]/'third_party/mc-vamp/resources/panda/panda_spherized.urdf'
    name={1:'panda_leftfinger',2:'panda_rightfinger'}[index]
    link = ET.parse(path).getroot().find(f"link[@name='{name}']")
    return np.asarray([[*map(float, part.find('origin').get('xyz').split()),
                        float(part.find('geometry/sphere').get('radius'))]
                       for part in link.findall('collision')])


def finger_environment_model(metadata: dict, world_robot: np.ndarray) -> dict:
    """Place upstream spheres at measured robot finger-link poses.

    Three noncollinear palm sphere centers define a rigid frame. Coefficients
    in that frame let GPU IK and SIMD planning transform the same finger centers
    using their existing sphere FK, without another robot FK or a custom shape.
    """
    from .rekep_libero.geometry import pose_matrix
    values = native_spheres(metadata['joint_positions'])
    values[:, :3] = values[:, :3]@world_robot[:3, :3].T+world_robot[:3, 3]
    base, first, second = values[list(HAND_BASIS_INDICES), :3]
    u, v = first-base, second-base
    basis = np.column_stack([u, v, np.cross(u, v)])
    volumes = {item['robot_geom_name']: item for item in metadata['robot_collision_volumes_world']}
    fingers = []
    for index in (1, 2):
        # Link frames avoid MuJoCo's internal mesh-centering transform.
        local=finger_mesh_spheres(index)
        pose = pose_matrix(volumes[f'gripper0_finger{index}_collision']['body_pose_world_xyzw'])
        centers = local[:, :3]@pose[:3, :3].T+pose[:3, 3]
        fingers.extend(np.c_[centers, local[:, 3]])
    fingers = np.asarray(fingers)
    coefficients = np.linalg.solve(basis, (fingers[:, :3]-base).T).T
    tcp = pose_matrix(metadata['eef_pose_world_xyzw'])
    hand = np.vstack([values[slice(*PALM_SPHERE_RANGE)], fingers])
    hand[:, :3] = (hand[:, :3]-tcp[:3, 3])@tcp[:3, :3]
    return {'coefficients': np.c_[coefficients, fingers[:, 3]].tolist(),
            'hand_spheres_tcp': hand.tolist(), 'basis_native_indices': list(HAND_BASIS_INDICES),
            'gripper_qpos': metadata['gripper_qpos'],
            'source': 'upstream finger spheres at measured robot link poses', 'padding_m': 0.}


def native_spheres(joints):
    import vamp
    return np.asarray([[s.x, s.y, s.z, s.r] for s in vamp.panda.fk(np.asarray(joints).tolist())])


def palm_spheres_in_eef(joints):
    import vamp
    values = native_spheres(joints)[slice(*PALM_SPHERE_RANGE)]
    frame = np.asarray(vamp.panda.eefk(np.asarray(joints).tolist()))[0]
    return np.c_[(values[:, :3]-frame[:3, 3])@frame[:3, :3], values[:, 3], np.zeros(len(values))]


def sphere_part(index):
    for link, (first, end) in enumerate(ARM_SPHERE_RANGES, 1):
        if first <= index < end:
            return f'robot0_link{link}_collision'
    if PALM_SPHERE_RANGE[0] <= index < PALM_SPHERE_RANGE[1]:
        return 'gripper0_hand_collision'
    if FINGER_SPHERE_RANGE[0] <= index < FINGER_SPHERE_RANGE[1]:
        return f'gripper0_finger{1+(index-FINGER_SPHERE_RANGE[0])//2}_collision'
    return None
