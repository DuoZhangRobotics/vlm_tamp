"""Export the calibrated robot-only chain, including LIBERO's exact TCP.

cuRobo's URDF parser accepts coordinate-aligned revolute axes. Each generated
joint frame therefore has its z axis along the calibrated axis. This avoids
assuming that the stock Franka hand frame equals the controller grip_site.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def axis_frame(axis):
    z = np.asarray(axis, float)
    z = z / np.linalg.norm(z)
    helper = np.eye(3)[np.argmin(np.abs(z))]
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def export_model(record, directory, margin_rad=0.):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    reference = np.asarray(record['reference_joints'], float)
    axes = np.asarray(record['joint_axes_world'], float)
    anchors = np.asarray(record['joint_anchors_world_m'], float)
    limits = np.asarray(record['joint_limits'], float)
    tcp = np.asarray(record['reference_tcp_world'], float)
    if reference.shape != (7,) or axes.shape != (7, 3) or anchors.shape != (7, 3):
        raise ValueError('Exactly seven calibrated revolute joints are required')
    if not 0 <= margin_rad < .2 or not np.all(limits[:, 0] + margin_rad < limits[:, 1] - margin_rad):
        raise ValueError('Invalid command joint-limit reserve')
    robot = ET.Element('robot', name='libero_calibrated_panda')
    ET.SubElement(robot, 'link', name='world')

    def add_joint(name, parent, child, xyz, rotation, joint_type='fixed', bounds=None):
        ET.SubElement(robot, 'link', name=child)
        joint = ET.SubElement(robot, 'joint', name=name, type=joint_type)
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'origin', xyz=' '.join(map(str, xyz)),
                      rpy=' '.join(map(str, Rotation.from_matrix(rotation).as_euler('xyz'))))
        if bounds is not None:
            ET.SubElement(joint, 'axis', xyz='0 0 1')
            ET.SubElement(joint, 'limit', lower=str(bounds[0]), upper=str(bounds[1]),
                          effort='87', velocity='2.0')

    parent, previous_rotation, previous_anchor = 'world', np.eye(3), np.zeros(3)
    for index in range(7):
        rotation = axis_frame(axes[index])
        fixed_rotation = previous_rotation.T @ rotation @ Rotation.from_rotvec([0, 0, -reference[index]]).as_matrix()
        child = f'link{index+1}'
        add_joint(f'joint{index+1}', parent, child,
                  previous_rotation.T @ (anchors[index] - previous_anchor), fixed_rotation,
                  'revolute', limits[index] + [margin_rad, -margin_rad])
        parent, previous_rotation, previous_anchor = child, rotation, anchors[index]
    add_joint('controller_tcp', parent, 'tcp', previous_rotation.T @ (tcp[:3, 3] - previous_anchor),
              previous_rotation.T @ tcp[:3, :3])
    urdf = directory / 'calibrated-panda.urdf'
    ET.indent(robot)
    ET.ElementTree(robot).write(urdf, encoding='unicode', xml_declaration=True)
    config = {'kinematics': {
        'urdf_path': str(urdf), 'base_link': 'world', 'ee_link': 'tcp',
        'collision_link_names': [], 'collision_spheres': None,
        'use_global_cumul': True,
        'cspace': {'joint_names': [f'joint{i+1}' for i in range(7)],
                   'retract_config': np.clip(reference, limits[:, 0]+margin_rad, limits[:, 1]-margin_rad).tolist(),
                   'null_space_weight': [1.]*7, 'cspace_distance_weight': [1.]*7,
                   'max_acceleration': 15., 'max_jerk': 500.}}}
    (directory/'robot-config.json').write_text(json.dumps(config, indent=2)+'\n')
    (directory/'scope.json').write_text(json.dumps({
        'robot_only_calibration': True, 'simulator_scene_state_used': False,
        'base_frame': 'world', 'end_effector': 'LIBERO controller grip_site',
        'joint_order': [f'joint{i+1}' for i in range(7)], 'command_joint_reserve_rad': margin_rad,
        'curobo_internal_environment_collision_check': False,
        'curobo_internal_self_collision_check': False,
        'self_collision_backend': 'VAMP Panda validation after cuRobo IK',
        'environment_collision_backend': 'full robot mesh versus observed depth, with task contact permissions',
        'collision_free_guarantee': False}, indent=2)+'\n')
    return config
