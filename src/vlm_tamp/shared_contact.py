"""Robot-geometry contact candidates in an observed part frame.

This module has no simulator object IDs, task-name switches, saved grasp poses,
or evaluation outcomes. It proposes modes; a planner must admit each candidate.
"""
import numpy as np
from scipy.spatial import cKDTree

from .rekep_libero.geometry import pose_matrix, matrix_pose, transform_points


def unit(x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all() or np.linalg.norm(x) < 1e-8:
        raise ValueError('Contact frame requires finite nonzero directions')
    return x / np.linalg.norm(x)


def contact_frame(cloud, spec):
    points = np.asarray(cloud, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20 or not np.isfinite(points).all():
        raise ValueError('Contact candidates require an observed finite part cloud')
    normal = unit(spec['panel_normal_world'])
    if spec['joint_family'] == 'prismatic':
        axis = unit(spec['joint_axis_world'])
        normal = axis if axis @ normal >= 0 else -axis
    # Cartesian-coordinate medians change the covariance under world rotation.
    # Center the PCA with the mean; use robust quantiles in the resulting frame.
    center = points.mean(axis=0)
    _, singular, vt = np.linalg.svd(points-center, full_matrices=False)
    bar = vt[0] - normal * (vt[0] @ normal)
    if spec.get('contact_kind')=='surface' and np.linalg.norm(np.cross(normal,[0.,0.,1.]))>.1:
        # A flat local patch need not have a stable PCA long axis. Its declared
        # surface-contact frame uses gravity; legacy bar proposals are unchanged.
        bar=np.cross(normal,[0.,0.,1.])
    if np.linalg.norm(bar) < .1:
        bar = np.cross(normal, [0., 0., 1.])
    if np.linalg.norm(bar) < .1:
        bar = np.cross(normal, [1., 0., 0.])
    bar = unit(bar)
    across = unit(np.cross(bar, normal))
    # Gravity resolves the sign, but distances and poses remain part-relative.
    if across @ [0, 0, 1] < 0:
        bar = -bar; across = -across
    basis = np.column_stack([bar, normal, across])
    low, high = np.quantile((points-center) @ basis, [.02, .98], axis=0)
    return {'center': center, 'normal': normal, 'bar': bar, 'across': across,
            'basis': basis, 'extent': high-low, 'elongated': bool(singular[0] > 2*singular[1])}


def local_gripper(metadata):
    inverse = np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']))
    return {v['robot_geom_name']: transform_points(v['vertices_world'], inverse)
            for v in metadata['robot_collision_volumes_world'] if 'gripper' in v['robot_geom_name']}


def contact_menu(cloud, spec, metadata, scene_points, *, depth_resolution=.003, mode_order=None):
    """Rank mode families by measured clearance, then emit robot-frame poses."""
    cloud = np.asarray(cloud, float); f = contact_frame(cloud, spec)
    shapes = local_gripper(metadata)
    fingers = np.concatenate([v for k,v in shapes.items() if 'finger' in k and 'pad' not in k])
    pads = np.concatenate([v for k,v in shapes.items() if k.endswith('_pad_collision')])
    palm = np.concatenate([v for k,v in shapes.items() if 'hand_collision' in k])
    pad_center = pads.mean(0)
    points = np.asarray(scene_points, float).reshape(-1, 3)
    rel = (points - f['center']) @ f['basis']
    below = (abs(rel[:,0]) < .10) & (rel[:,1] > -.008) & (rel[:,1] < .10) & (rel[:,2] < -.015) & (rel[:,2] > -.20)
    clearance = float(np.min(-rel[below,2])) if below.any() else .20
    palm_half_span = float(np.ptp(palm[:,0])/2)
    tight = bool(clearance < palm_half_span + .008)
    opening = spec.get('motion') != 'close'
    modes = ['pinch']
    if f['elongated']:
        modes = (['press_drag','pinch'] if tight else ['pinch','press_drag']) if opening else ['push','pinch']
    elif not opening:
        modes = ['push','pinch']
    if mode_order is not None:
        if not mode_order or any(mode not in {'pinch','press_drag','push'} for mode in mode_order):
            raise ValueError('Unknown requested contact mode')
        modes = list(mode_order)
    rows = []
    for mode in modes:
        if mode == 'pinch':
            # Learned proposals remain available; the generic frontal candidate
            # adds an explicit part-relative option without assuming an ID.
            angles = [0.]
        elif mode == 'push': angles = [0., 30., 45.]
        else: angles = [30., 45., 60.]
        for angle in angles:
            theta = np.deg2rad(angle)
            approach = unit(-f['normal']*np.cos(theta)-f['across']*np.sin(theta))
            jaw = f['bar'] if mode != 'pinch' else f['across']
            y = unit(np.cross(approach, jaw)); jaw = unit(np.cross(y, approach))
            rotation = np.column_stack([jaw, y, approach])
            if mode == 'press_drag':
                height = cloud @ f['across']
                top = cloud[height >= np.quantile(height, .75)]
                contact = f['center'] + f['basis'] @ np.median((top-f['center']) @ f['basis'],axis=0)
                # Put the lower finger surface on the measured upper handle.
                transformed = fingers @ rotation.T
                low = transformed @ f['across']
                touch = transformed[low <= np.quantile(low, .05)].mean(0)
                position = contact - touch - 2*depth_resolution*f['across']
                contact_normal = f['across']
                aperture = float(np.clip(f['extent'][0]/4, .012, .032))
            else:
                position = f['center'] - rotation @ pad_center
                if mode == 'push': position -= 2*depth_resolution*f['normal']
                contact_normal = f['normal']
                aperture = .04 if mode == 'pinch' else .012
            transform = np.eye(4); transform[:3,:3] = rotation; transform[:3,3] = position
            rows.append({'mode': mode, 'pose_world_xyzw': matrix_pose(transform).tolist(),
                         'contact_normal_world': contact_normal.tolist(),
                         'bar_axis_world': f['bar'].tolist(), 'half_aperture_m': aperture,
                         'approach_tilt_deg': angle, 'source': 'observed part frame and configured robot geometry',
                         'contact_establishment_depth_m':2*depth_resolution if mode!='pinch' else 0.,
                         'closure_command': 1 if mode == 'pinch' else 0})
    return rows, {'mode_order': modes, 'measured_lower_clearance_m': clearance,
                  'mode_order_source':'caller' if mode_order is not None else 'legacy_clearance_heuristic',
                  'configured_palm_half_span_m': palm_half_span, 'tight_clearance': tight,
                  'observed_extent_m': f['extent'].tolist(), 'part_frame_world': f['basis'].tolist(),
                  'contact_uncertainty_policy':'press through two measured-depth grid cells at the selected finger region',
                  'simulator_object_truth_used': False, 'task_name_used': False}
