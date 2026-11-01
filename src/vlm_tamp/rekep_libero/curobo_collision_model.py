"""The same upstream VAMP arm/palm spheres used by native path planning."""
import numpy as np

from ..vamp_sphere_model import ARM_SPHERE_RANGES, PALM_SPHERE_RANGE, native_spheres
from .curobo_model import axis_frame


def calibrated_spheres(chain, robot, world_robot=None):
    if world_robot is None:
        raise ValueError('The VAMP robot-to-world calibration is required')
    world_robot = np.asarray(world_robot)
    values = native_spheres(chain.reference_joints)
    values[:, :3] = values[:, :3]@world_robot[:3, :3].T+world_robot[:3, 3]
    spheres, groups = {}, []
    for index, (first, end) in enumerate((*ARM_SPHERE_RANGES, PALM_SPHERE_RANGE), 1):
        link = f'link{index}' if index <= 7 else 'tcp'
        frame = np.eye(4)
        if index <= 7:
            frame[:3, :3] = axis_frame(chain.axes[index-1])
            frame[:3, 3] = chain.anchors[index-1]
        else:
            frame = chain.reference_tcp
        selected = values[first:end]
        local = (selected[:, :3]-frame[:3, 3])@frame[:3, :3]
        spheres[link] = [{'center': c.tolist(), 'radius': float(r)} for c, r in zip(local, selected[:, 3])]
        groups.append({'link': link, 'native_indices': list(range(first, end)),
            'world_robot': world_robot.tolist(), 'source_robot_volume': 'upstream_vamp_spheres'})
    return spheres, {'groups': groups, 'sphere_count': sum(map(len, spheres.values())),
        'scope': 'Upstream VAMP arm and palm spheres; fingers exempt from all environment surfaces',
        'robot_geometry_shared_with_native_planner': True, 'simulator_object_geometry_used': False}


def expected_spheres(robot, groups, joints):
    values = native_spheres(joints)
    result = []
    for group in groups:
        selected = values[group['native_indices']]
        world = np.asarray(group['world_robot'])
        result.extend(np.c_[selected[:, :3]@world[:3, :3].T+world[:3, 3], selected[:, 3]].tolist())
    return np.asarray(result)
