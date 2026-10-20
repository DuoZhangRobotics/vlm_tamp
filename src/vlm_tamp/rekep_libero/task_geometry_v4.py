"""Structured task geometry from cached RGB role choices and measured depth.

No scene identities, simulator object poses or joint axes enter this module.
These pose residuals can be bound to a robot backend's FK/Jacobian.
"""

import ast

import numpy as np

from vlm_tamp.drawer_constraints import evaluate_line_constraint
from vlm_tamp.drawer_local_plane import _ransac, local_plane_config
from vlm_tamp.pose_constraints import evaluate_relation, unit
from .geometry import pose_matrix, transform_points


def referenced_points(stage):
    result = set()
    for constraint in stage.subgoal_constraints:
        for node in ast.walk(constraint.definition):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'keypoints':
                first = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) else node.slice
                if not isinstance(first, ast.Slice):
                    indices = ast.literal_eval(first)
                    result.update(indices if isinstance(indices, (list, tuple)) else [indices])
    return sorted(result)


def region_cloud(bundle, indices, exclude_index=None):
    """Select an existing RGB mask covering the requested semantic anchors."""
    camera = bundle['cameras'][bundle['main_camera']]
    pixels = np.asarray(bundle['pixels_yx'])[indices]
    candidates = []
    for mask in bundle['masks']:
        if not 40 <= mask.sum() <= .2 * mask.size:
            continue
        if exclude_index is not None and mask[tuple(bundle['pixels_yx'][exclude_index])]:
            continue
        if all(mask[tuple(pixel)] for pixel in pixels):
            candidates.append(mask)
    if not candidates:
        raise ValueError('No RGB region consistently covers the selected target landmarks')
    mask = min(candidates, key=lambda item: int(item.sum())).copy()
    valid = np.isfinite(camera['world_points']).all(axis=2) & (camera['depth'] > 0)
    mask &= valid
    cloud = camera['world_points'][mask]
    if len(cloud) < 40:
        raise ValueError('Selected RGB target has insufficient measured depth')
    return cloud, mask


def surface_summary(cloud):
    cloud = np.asarray(cloud, float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 30 or not np.isfinite(cloud).all():
        raise ValueError('Measured object surface must contain at least 30 finite XYZ samples')
    low, high = np.quantile(cloud, [.01, .99], axis=0)
    return {'center_xy_m': ((low[:2] + high[:2]) / 2).tolist(),
            'low_z_m': float(low[2]), 'high_z_m': float(high[2]),
            'median_z_m': float(np.median(cloud[:, 2])),
            'extent_m': (high - low).tolist(), 'points': len(cloud),
            'source': '1st/99th-percentile bounds of RGB-selected measured surfaces'}


def bowl_spec(bundle, program, source_cloud, clearance=.006):
    source = program.grasp_keypoints[0]
    camera = bundle['cameras'][bundle['main_camera']]
    source_masks = [mask for mask in bundle['masks'] if mask[tuple(bundle['pixels_yx'][source])]]
    static = [index for index in referenced_points(program.stages[1])
              if index != source and not any(mask[tuple(bundle['pixels_yx'][index])] for mask in source_masks)]
    if not static:
        raise ValueError('Cached bowl program does not identify a separate target region')
    target_cloud, target_mask = region_cloud(bundle, static, exclude_index=source)
    obj, plate = surface_summary(source_cloud), surface_summary(target_cloud)
    target = np.r_[plate['center_xy_m'], plate['median_z_m'] + clearance]
    spec = {'schema_version': 'grounded-task-4.0', 'task_type': 'pick_place',
            'source_keypoint': source, 'target_keypoints': static,
            'object_surface': obj, 'target_surface': plate,
            'object_support_point_initial_world_m': [*obj['center_xy_m'], obj['low_z_m']],
            'release_clearance_m': clearance,
            'goal': {'type': 'point_target', 'point_local_m': [0., 0., 0.],
                     'target_world_m': target.tolist(), 'state_pose': 'estimated_object_support_frame',
                     'frame': 'world', 'equality_tolerance_m': .012},
            'geometric_path_constraints': [], 'attachment_binding_status': 'unresolved_until_observed_lift',
            'geometry_source': 'cached RGB-selected roles plus current metric depth',
            'canonical_object_mesh_used': False, 'simulator_object_state_used': False,
            'calibration_pixels_verified': len(camera['rgb'].shape) == 3}
    return spec, target_cloud, target_mask


def drawer_spec(bundle, program, source_cloud, travel=.14):
    cloud = np.asarray(source_cloud, float)
    model = _ransac(cloud, local_plane_config({}), 0)
    if model is None or model['inliers'].sum() < 100 or model['rms'] > .003:
        raise ValueError('No reliable observed drawer-front plane')
    camera = bundle['cameras'][bundle['main_camera']]
    to_camera = np.asarray(camera['camera_to_world'])[:3, 3] - model['center']
    normal = model['normal'] * (1 if model['normal'] @ to_camera >= 0 else -1)
    if abs(normal[2]) > .2:
        raise ValueError('Observed front normal does not support the horizontal drawer template')
    axis = unit(np.r_[normal[:2], 0.])
    source = program.grasp_keypoints[0]
    landmark = np.asarray(bundle['points_world'][source])
    protrusion = (cloud - model['center']) @ axis
    handle = cloud[(protrusion > .004) & (protrusion < .045)
                   & (np.abs(cloud[:, 2] - landmark[2]) < .027)]
    if len(handle) < 60:
        raise ValueError('Observed protruding handle has insufficient support')
    _, _, vt = np.linalg.svd(handle - handle.mean(axis=0), full_matrices=False)
    handle_axis = unit(vt[0])
    spec = {'schema_version': 'grounded-task-4.0', 'task_type': 'open_middle_drawer',
            'source_keypoint': source, 'joint_type': 'prismatic', 'motion': 'pull',
            'pull_axis_world': axis.tolist(), 'handle_long_axis_world': handle_axis.tolist(),
            'handle_center_world_m': handle.mean(axis=0).tolist(),
            'front_plane': {'point_world_m': model['center'].tolist(), 'normal_world': normal.tolist(),
                            'rms_m': model['rms'], 'inliers': int(model['inliers'].sum()),
                            'inlier_fraction': float(model['inliers'].mean())},
            'requested_travel_m': travel, 'handle_points': len(handle),
            'path_constraint': {'type': 'point_on_line', 'point_local_m': [0., 0., 0.],
                                'line_point_world_m': landmark.tolist(), 'axis_world': axis.tolist(),
                                'minimum_travel_m': 0., 'maximum_travel_m': travel + .015,
                                'equality_tolerance_m': .015, 'state_pose': 'robot_tcp', 'frame': 'world'},
            'orientation_policy': 'preserve the physically closed grasp orientation during pull',
            'attachment_binding_status': 'unresolved_until_observed_pull',
            'geometry_source': 'RGB-selected local region and robust metric-depth plane',
            'axis_assumption': 'horizontal prismatic motion normal to visible cabinet front',
            'simulator_joint_axis_used': False, 'simulator_object_state_used': False}
    return spec, handle


def bind_bowl_goal(spec, retention_transform, held_tcp):
    support = transform_points([spec['object_support_point_initial_world_m']], retention_transform)[0]
    transform = pose_matrix(held_tcp)
    local = transform_points([support], np.linalg.inv(transform))[0]
    goal = {**spec['goal'], 'point_local_m': local.tolist(), 'state_pose': 'robot_tcp',
            'attachment_binding_status': 'bound_after_observed_retained_lift',
            'binding_source': 'RGB-D rigid surface transform and measured controller TCP'}
    tcp_goal = np.asarray(held_tcp, float).copy()
    tcp_goal[:3] = np.asarray(goal['target_world_m']) - transform[:3, :3] @ local
    return goal, tcp_goal


def bind_drawer_line(spec, closed_tcp):
    return {**spec['path_constraint'], 'line_point_world_m': np.asarray(closed_tcp)[:3].tolist(),
            'attachment_binding_status': 'unverified_pending_observed_pull',
            'axis_semantics': spec['axis_assumption']}


def pose_residual(spec, pose):
    transform = pose_matrix(pose)
    evaluate = evaluate_line_constraint if spec['type'] == 'point_on_line' else evaluate_relation
    return evaluate(spec, transform[:3, 3], transform[:3, :3])


def joint_residual(spec, fk_pose, pose_jacobian):
    """Bind analytic pose residuals to an external seven-joint robot Jacobian.

    Jacobian rows must be world translation followed by right/body rotation,
    matching the existing pose-constraint interface. This is an adapter, not
    an implementation or benchmark of the unavailable ST-pRRTC manifold backend.
    """
    jacobian = np.asarray(pose_jacobian, float)
    if jacobian.shape != (6, 7) or not np.isfinite(jacobian).all():
        raise ValueError('Panda pose Jacobian must contain six rows and seven joints')
    result = pose_residual(spec, fk_pose)
    return {**result, 'equality_joint_jacobian': result['equality_jacobian'] @ jacobian,
            'inequality_joint_jacobian': result['inequality_jacobian'] @ jacobian}
