"""Sampled full-arm/payload path admission, independent of a search planner."""

import numpy as np

from .geometry import pose_matrix, transform_points


def estimated_contact_aperture(sim, grasp_joints, source_cloud, open_qpos):
    geometry = sim.fk(grasp_joints, include_volumes=True, gripper_qpos=open_qpos)
    probe_qpos = [.02, -.02] if np.linalg.norm(np.asarray(open_qpos) - [.02, -.02]) > 1e-5 else [.01, -.01]
    probe = sim.fk(grasp_joints, include_volumes=True, gripper_qpos=probe_qpos)
    probe_volumes = {volume['robot_geom_name']: volume for volume in probe['robot_collision_volumes_world']}
    inverse = np.linalg.inv(pose_matrix(geometry['eef_pose_world_xyzw']))
    points = transform_points(source_cloud, inverse)
    estimates = []
    for volume in geometry['robot_collision_volumes_world']:
        if not volume['robot_geom_name'].endswith('_pad_collision'):
            continue
        vertices = transform_points(volume['vertices_world'], inverse)
        low, high = vertices.min(0), vertices.max(0)
        positive = vertices[:, 0].mean() > 0
        keep = ((points[:, 1:] >= low[1:] - .002).all(1)
                & (points[:, 1:] <= high[1:] + .002).all(1)
                & (np.abs(points[:, 0]) <= .0395))
        cloud = points[keep]
        if len(cloud) < 6:
            return None
        face = low[0] if positive else high[0]
        # The configured Panda's finger 1 is on TCP -x although its prismatic
        # qpos is positive; finger 2 has negative qpos and is on TCP +x.
        # Determine geometric direction from an isolated FK probe rather than
        # treating spatial side as a gripper-joint index.
        index = 0 if volume['robot_geom_name'].endswith('finger1_pad_collision') else 1
        moved = transform_points(probe_volumes[volume['robot_geom_name']]['vertices_world'], inverse)
        moved_face = moved[:, 0].min() if positive else moved[:, 0].max()
        slope = (moved_face - face) / (probe_qpos[index] - open_qpos[index])
        if abs(abs(slope) - 1.) > 1e-5:
            raise ValueError('Panda pad motion does not match the calibrated closing axis')
        endpoint = np.quantile(cloud[:, 0], .95 if positive else .05)
        required_face = endpoint + (.0005 if positive else -.0005)
        q = open_qpos[index] + (required_face - face) / slope
        estimates.append((index, float(np.clip(q, 0 if index == 0 else -.04, .04 if index == 0 else 0))))
    if len(estimates) != 2 or len({item[0] for item in estimates}) != 2:
        raise ValueError('Expected exactly two known Panda contact pads')
    estimates.sort()
    return [item[1] for item in estimates]


def check_joint_path(scene, sim, ik, joint_path, *, gripper_qpos, transform_at_pose,
                     stage_at_pose, check_payload, config):
    """Inspect every waypoint and interpolate joint edges at a frozen resolution.

    This is discrete edge sampling, not a continuous collision certificate.
    Actual controller tracking is checked separately at action boundaries.
    """
    joint_path = np.asarray(joint_path, float)
    if joint_path.ndim != 2 or joint_path.shape[1] != 7 or not len(joint_path):
        raise ValueError('Collision path must contain seven-joint configurations')
    maximum_step = float(config['collision_joint_sampling_rad'])
    tested, exemptions = 0, 0
    for edge in range(max(1, len(joint_path) - 1)):
        a = joint_path[edge]
        b = joint_path[min(edge + 1, len(joint_path) - 1)]
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / maximum_step)))
        weights = np.linspace(0., 1., count + 1) if edge == 0 else np.linspace(0., 1., count + 1)[1:]
        for weight in weights:
            q = a + weight * (b - a)
            if not ik.self_valid(q):
                return {'accepted': False, 'reason': 'path_self_collision', 'edge': edge,
                        'fraction': float(weight), 'tested_states': tested}
            pose = ik.chain.forward(q)
            result = scene.joints(sim, q, gripper_qpos=gripper_qpos,
                                  stage=stage_at_pose(pose), moving_transform=transform_at_pose(pose),
                                  check_payload=check_payload)
            tested += 1
            exemptions += result['local_contact_exemptions']
            if not result['accepted']:
                return {'accepted': False, 'reason': 'full_arm_or_payload_collision', 'edge': edge,
                        'fraction': float(weight), 'tested_states': tested, 'clearance': result}
    return {'accepted': True, 'tested_states': tested, 'local_contact_exemptions': exemptions,
            'joint_sampling_rad': maximum_step, 'all_waypoints_checked': True,
            'continuous_collision_detection': False}


def bowl_transform(grasp_pose, closure_transform=None):
    inverse = np.linalg.inv(pose_matrix(grasp_pose))
    closure = np.eye(4) if closure_transform is None else pose_matrix(closure_transform)
    return lambda pose: pose_matrix(pose) @ inverse @ closure


def drawer_transform(scene, grasp_pose, offset=0.):
    origin = np.asarray(grasp_pose)[:3]
    axis = scene.region.axis
    return lambda pose: scene.drawer_transform(float((np.asarray(pose)[:3] - origin) @ axis) + offset)
