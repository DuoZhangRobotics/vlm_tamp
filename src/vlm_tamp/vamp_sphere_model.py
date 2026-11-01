"""Pinned upstream Panda spheres shared by native planning and GPU IK.

Indices follow third_party/mc-vamp/resources/panda/panda_spherized.urdf.
The fixed base is omitted from environment checks, as before. All finger
spheres are exempt from every environment partition. Self collision stays VAMP's.
"""
import numpy as np

ARM_SPHERE_RANGES = ((1, 5), (5, 9), (9, 13), (13, 17), (17, 29), (29, 32), (32, 37))
PALM_SPHERE_RANGE = (37, 55)
FINGER_SPHERE_RANGE = (55, 59)
ENVIRONMENT_SPHERE_INDICES = tuple(range(1, 55))


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
    return None
