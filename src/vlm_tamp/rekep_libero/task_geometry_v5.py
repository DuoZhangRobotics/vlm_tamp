"""Mask-consistent placement and observed ordinal drawer grounding.

The middle-handle check is an explicit template adaptation: three measured
protruding bars in an RGB parent mask are ordered by height. It is not a claim
that the cached VLM picked the correct landmark. No simulator scene state enters.
"""

import numpy as np

from vlm_tamp.drawer_local_plane import _ransac, local_plane_config
from vlm_tamp.pose_constraints import unit
from .task_geometry_v4 import region_cloud, referenced_points, surface_summary


def bowl_spec(bundle, program, source_cloud, source_mask, config):
    source = program.grasp_keypoints[0]
    mask = np.asarray(source_mask, bool)
    if mask.shape != bundle['cameras'][bundle['main_camera']]['depth'].shape:
        raise ValueError('Admitted source mask must match the observed depth image')
    static = [index for index in referenced_points(program.stages[1])
              if index != source and not mask[tuple(bundle['pixels_yx'][index])]]
    if not static:
        raise ValueError('No separate target references outside the admitted source mask')
    target_cloud, target_mask = region_cloud(bundle, static, exclude_index=source)
    obj, plate = surface_summary(source_cloud), surface_summary(target_cloud)
    clearance = config['release_clearance_m']
    target = [*plate['center_xy_m'], plate['median_z_m'] + clearance]
    spec = {'schema_version': 'grounded-task-5.0', 'task_type': 'pick_place',
            'source_keypoint': source, 'cached_source_keypoint': source, 'target_keypoints': static,
            'source_anchor_world_m': bundle['points_world'][source].tolist(),
            'object_surface': obj, 'target_surface': plate,
            'object_support_point_initial_world_m': [*obj['center_xy_m'], obj['low_z_m']],
            'release_clearance_m': clearance,
            'goal': {'type': 'point_target', 'point_local_m': [0., 0., 0.], 'target_world_m': target,
                     'state_pose': 'estimated_object_support_frame', 'frame': 'world',
                     'equality_tolerance_m': .012}, 'geometric_path_constraints': [],
            'attachment_binding_status': 'unresolved_until_observed_lift',
            'mask_association_policy': 'exclude source landmarks using only the admitted source mask',
            'geometry_source': 'cached RGB role choices and admitted masks plus current metric depth',
            'canonical_object_mesh_used': False, 'simulator_object_state_used': False}
    return spec, source_cloud, target_cloud, target_mask, {'source': mask, 'target': target_mask}


def _front_model(cloud, camera_position, minimum_inliers=200):
    model = _ransac(cloud, local_plane_config({}), 0)
    if model is None or model['inliers'].sum() < minimum_inliers or model['rms'] > .003:
        return None
    direction = unit(np.asarray(camera_position) - model['center'])
    normal = model['normal'] * (1 if model['normal'] @ direction > 0 else -1)
    if abs(normal[2]) > .2 or normal @ direction < .65:
        return None
    support = cloud[model['inliers']]
    low, high = np.quantile(support, [.01, .99], axis=0)
    return {**model, 'normal': normal, 'low': low, 'high': high}


def _bar_clusters(cloud, original_indices):
    """Find separated horizontal protrusions; require the ordinal relation."""
    order = np.argsort(cloud[:, 2], kind='stable')
    splits = np.flatnonzero(np.diff(cloud[order, 2]) > .025) + 1
    groups = np.split(order, splits)
    accepted = []
    for indices in groups:
        points = cloud[indices]
        if len(points) < 60:
            continue
        extent = np.ptp(points, axis=0)
        if extent[0] < .04 or extent[2] > .035 or extent[0] < 3 * extent[2]:
            continue
        accepted.append({'cloud': points, 'indices': original_indices[indices], 'center': points.mean(axis=0)})
    if len(accepted) != 3:
        raise ValueError(f'Ordinal middle-handle check needs three supported bars; observed {len(accepted)}')
    centers = np.asarray([group['center'] for group in accepted])
    gaps = np.diff(centers[:, 2])
    if np.any(gaps < .04) or np.any(gaps > .12) or np.ptp(centers[:, 0]) > .025 or np.ptp(centers[:, 1]) > .015:
        raise ValueError('Observed protrusions do not form a consistent three-handle ordering')
    return accepted


def drawer_spec(bundle, program, config):
    camera = bundle['cameras'][bundle['main_camera']]
    source = program.grasp_keypoints[0]
    source_pixel = tuple(bundle['pixels_yx'][source])
    camera_position = np.asarray(camera['camera_to_world'])[:3, 3]
    world = np.asarray(camera['world_points'])
    valid = np.isfinite(world).all(axis=2) & (camera['depth'] > 0)
    parents, diagnostics = [], []
    for index, proposal in enumerate(bundle['masks']):
        if not proposal[source_pixel] or not 500 <= proposal.sum() <= .2 * proposal.size:
            continue
        mask = proposal & valid
        cloud = world[mask]
        model = _front_model(cloud, camera_position)
        if model is None or model['high'][2] - model['low'][2] < .12 or model['high'][0] - model['low'][0] < .12:
            continue
        axis = unit(np.r_[model['normal'][:2], 0.])
        protrusion = (cloud - model['center']) @ axis
        keep = ((protrusion > .012) & (protrusion < .045)
                & (cloud[:, 0] > model['low'][0] + .008) & (cloud[:, 0] < model['high'][0] - .008)
                & (cloud[:, 2] >= model['low'][2]) & (cloud[:, 2] <= model['high'][2]))
        flat_indices = np.flatnonzero(mask)
        try:
            bars = _bar_clusters(cloud[keep], flat_indices[keep])
        except ValueError as error:
            diagnostics.append({'mask_index': index, 'reason': str(error)})
            continue
        parents.append((int(mask.sum()), index, mask, cloud, model, bars))
    if not parents:
        raise ValueError('No RGB parent region supports a front plane and three ordered handles: ' + str(diagnostics))
    _, parent_index, parent_mask, parent_cloud, model, bars = min(parents, key=lambda item: item[:2])
    middle = bars[1]
    # Fit the panel near the measured middle bar, excluding protruding material.
    axis = unit(np.r_[model['normal'][:2], 0.])
    local = parent_cloud[(np.abs(parent_cloud[:, 2] - middle['center'][2]) < .025)
                         & (np.linalg.norm(parent_cloud[:, :2] - middle['center'][:2], axis=1) < .10)
                         & (np.abs((parent_cloud - model['center']) @ axis) < .006)]
    refined = _front_model(local, camera_position)
    if refined is None:
        raise ValueError('No reliable local panel plane beside the observed middle handle')
    axis = unit(np.r_[refined['normal'][:2], 0.])
    handle = middle['cloud']
    _, _, vt = np.linalg.svd(handle - handle.mean(axis=0), full_matrices=False)
    anchor = handle.mean(axis=0)
    original = np.asarray(bundle['points_world'][source])
    correction = abs(original[2] - anchor[2]) > .025
    handle_mask = np.zeros(parent_mask.shape, bool)
    handle_mask.flat[middle['indices']] = True
    spec = {'schema_version': 'grounded-task-5.0', 'task_type': 'open_middle_drawer',
            'source_keypoint': source, 'cached_source_keypoint': source,
            'cached_source_world_m': original.tolist(), 'source_anchor_world_m': anchor.tolist(),
            'source_anchor_source': 'observed middle of three RGB-mask-supported protruding bars',
            'automatic_ordinal_role_correction': bool(correction), 'manual_role_rebinding': False,
            'cached_program_roles_overridden_for_execution': bool(correction),
            'observed_handle_centers_world_m': [group['center'].tolist() for group in bars],
            'parent_mask_index': parent_index, 'joint_type': 'prismatic', 'motion': 'pull',
            'pull_axis_world': axis.tolist(), 'handle_long_axis_world': unit(vt[0]).tolist(),
            'handle_center_world_m': anchor.tolist(), 'handle_points': len(handle),
            'front_plane': {'point_world_m': refined['center'].tolist(), 'normal_world': refined['normal'].tolist(),
                            'rms_m': refined['rms'], 'inliers': int(refined['inliers'].sum()),
                            'inlier_fraction': float(refined['inliers'].mean()), 'local_points': len(local)},
            'requested_travel_m': config['drawer_travel_m'],
            'preflight_travel_m': config['drawer_maximum_command_travel_m'],
            'path_constraint': {'type': 'point_on_line', 'point_local_m': [0., 0., 0.],
                                'line_point_world_m': anchor.tolist(), 'axis_world': axis.tolist(),
                                'minimum_travel_m': 0., 'maximum_travel_m': config['drawer_maximum_command_travel_m'],
                                'equality_tolerance_m': config['drawer_line_tolerance_m'],
                                'state_pose': 'robot_tcp', 'frame': 'world'},
            'orientation_policy': 'preserve the closed grasp orientation during pull',
            'attachment_binding_status': 'unresolved_until_observed_pull',
            'axis_assumption': 'horizontal prismatic motion normal to the locally measured front panel',
            'geometry_source': 'RGB parent mask, three observed bar protrusions and local metric-depth plane',
            'simulator_joint_axis_used': False, 'simulator_object_state_used': False}
    return spec, handle, None, None, {'source': handle_mask, 'parent': parent_mask}


def compile_scene(bundle, program, source_cloud, source_mask, config):
    if program.task_type == 'pick_place':
        return bowl_spec(bundle, program, source_cloud, source_mask, config)
    return drawer_spec(bundle, program, config)
