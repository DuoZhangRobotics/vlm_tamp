"""Keep observed moving geometry and each new planning frame synchronized.

Transport is a state estimate, never evidence of attachment or task success.
Stationary observations and contact permissions are preserved.
"""
import copy

import numpy as np
from .rekep_libero.geometry import pose_matrix, transform_points


def hand_motion(scene, chain, before, after):
    tcp = pose_matrix(before['eef_pose_world_xyzw'])
    pads = [np.asarray(v['vertices_world']).mean(0)
            for v in before['robot_collision_volumes_world']
            if v['robot_geom_name'].endswith('_pad_collision')]
    if len(pads) != 2:
        raise ValueError('Exactly two finger pads are needed to locate the grip point')
    local = transform_points([np.mean(pads, axis=0)], np.linalg.inv(tcp))[0]
    first = tcp[:3, 3] + tcp[:3, :3] @ local - scene.origin
    current_tcp = pose_matrix(chain.forward(after['joint_positions']))
    last = current_tcp[:3, 3] + current_tcp[:3, :3] @ local - scene.origin
    first -= scene.axis * (first @ scene.axis)
    last -= scene.axis * (last @ scene.axis)
    if min(np.linalg.norm(first), np.linalg.norm(last)) < 1e-5:
        raise ValueError('Grip point too close to hinge to predict angular motion')
    angle = float(np.arctan2(scene.axis @ np.cross(first, last), first @ last))
    return scene.transform(angle), angle


def transport_scene(scene, transform, *, source, reference_step, fit=None):
    transform = np.asarray(transform, float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.)):
        raise ValueError('A proper finite rigid transform is required')
    if source not in {'rgbd_rigid_motion', 'hand_motion_prediction'}:
        raise ValueError('Pose update source must distinguish observation from prediction')
    r = transform[:3, :3]
    spec = copy.deepcopy(scene.spec)
    # The hinge is fixed in the world. Only geometry belonging to its moving
    # link is transported; updating the hinge is a separate estimation step.
    spec['front_plane']['point_world_m'] = transform_points([scene.panel_origin], transform)[0].tolist()
    spec['panel_normal_world'] = (r @ scene.panel_normal).tolist()
    if 'source_anchor_world_m' in spec:
        spec['source_anchor_world_m'] = transform_points([spec['source_anchor_world_m']], transform)[0].tolist()
    prior = None
    if len(scene.inferred_panel_reference):
        record = copy.deepcopy(scene.panel_completion)
        record['origin_world_m'] = transform_points([record['origin_world_m']], transform)[0].tolist()
        for key in ['side_axis_world', 'vertical_axis_world', 'normal_axis_world']:
            record[key] = (r @ np.asarray(record[key])).tolist()
        prior = {'points_world': transform_points(scene.inferred_panel_reference, transform), 'record': record}
    result = copy.copy(scene)
    result._initialize(scene.static_points, transform_points(scene.moving_reference, transform),
        transform_points(scene.contact_reference, transform), spec, scene.config, scene.robot,
        scene.robot.reference_volumes if scene.robot is not None else [], prior)
    result.pose_update = {'source': source, 'reference_control_step': int(reference_step),
        'transform_world': transform.tolist(), 'stationary_points_changed': False,
        'point_roles_changed': False, 'hinge_refined': False, 'execution_veto': False,
        'assumes_contact_retention': source == 'hand_motion_prediction',
        'fit': fit, 'parent_pose_update': getattr(scene, 'pose_update', None)}
    return result
