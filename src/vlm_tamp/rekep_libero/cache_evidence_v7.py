"""Bounded RGB-D evidence checks for cached role and grasp-pose proposals."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .perception import _read_observation
from .program import parse_program
from .task_geometry_v4 import referenced_points


def region_agreement(old_rgb, new_rgb, old_points, new_points, config):
    old_rgb, new_rgb = np.asarray(old_rgb), np.asarray(new_rgb)
    old_points, new_points = np.asarray(old_points), np.asarray(new_points)
    if (old_rgb.shape != new_rgb.shape or old_points.shape != new_points.shape
            or old_rgb.shape[-1] != 3 or old_points.shape[-1] != 3 or not old_rgb.size
            or not np.isfinite(old_points).all() or not np.isfinite(new_points).all()):
        return {'accepted': False, 'reason': 'missing_or_incompatible_selected_evidence'}
    color = np.abs(old_rgb.astype(float) - new_rgb.astype(float))
    displacement = np.linalg.norm(old_points - new_points, axis=-1)
    rgb_mean, rgb_max, world_max = float(color.mean()), float(color.max()), float(displacement.max())
    return {'accepted': (rgb_mean <= config['cache_rgb_mean_channel_tolerance']
                         and rgb_max <= config['cache_rgb_max_channel_tolerance']
                         and world_max <= config['cache_world_tolerance_m']),
            'rgb_byte_exact': bool(np.array_equal(old_rgb, new_rgb)),
            'world_points_byte_exact': bool(np.array_equal(old_points, new_points)),
            'rgb_mean_absolute_channel_change': rgb_mean, 'rgb_maximum_channel_change': rgb_max,
            'maximum_world_point_change_m': world_max,
            'policy': 'paired selected RGB-D agreement; no simulator object identity or pose'}


def cached_scene(source, observation, config):
    source, observation = Path(source), Path(observation)
    cameras, metadata = _read_observation(observation / 'obs.npz', observation / 'metadata.json')
    main = json.loads((source / 'perception/perception.json').read_text())['main_camera']
    with np.load(source / 'perception/keypoints.npz', allow_pickle=False) as arrays:
        bundle = {key: arrays[key].copy() for key in arrays.files}
    with np.load(source / 'open-gripper/obs.npz', allow_pickle=False) as arrays:
        old_points = arrays[main + '_world_points'].copy()
    old_rgb = np.asarray(Image.open(source / 'perception/rgb.png'))
    rgb, world = cameras[main]['rgb'], cameras[main]['world_points']
    source_mask = np.load(source / 'stage1-grasps/selected_mask.npy', allow_pickle=False)
    task_type = json.loads((source / 'program/metadata.json').read_text())['task_type']
    program = parse_program((source / 'program/output_raw.txt').read_text(), len(bundle['points_world']),
                            task_type, attachment_cost=lambda _index: 1.)
    evidence = {'source_region': region_agreement(old_rgb[source_mask], rgb[source_mask],
                                                old_points[source_mask], world[source_mask], config)}
    indices = sorted(set(program.grasp_keypoints + referenced_points(program.stages[1])) - {-1})
    for index in indices:
        y, x = bundle['pixels_yx'][index]
        patch = np.s_[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        evidence[f'keypoint_{index}'] = region_agreement(old_rgb[patch], rgb[patch],
            old_points[y:y + 1, x:x + 1], world[y:y + 1, x:x + 1], config)
    if not all(check['accepted'] for check in evidence.values()):
        raise ValueError('Cached roles fail bounded selected RGB-D evidence checks; no paid API retry')
    bundle.update(cameras=cameras, meta=metadata, main_camera=main,
                  points_world=world[tuple(bundle['pixels_yx'].T)])
    bundle['cache_evidence'] = {'selected_evidence_accepted': True, 'regions': evidence,
        'full_primary_rgb_identical': bool(np.array_equal(old_rgb, rgb)), 'new_api_calls': 0,
        'world_tolerance_m': config['cache_world_tolerance_m'],
        'rgb_mean_channel_tolerance': config['cache_rgb_mean_channel_tolerance'],
        'rgb_max_channel_tolerance': config['cache_rgb_max_channel_tolerance'],
        'manual_role_rebinding': False}
    cloud = world[source_mask]
    return bundle, program, cloud[np.isfinite(cloud).all(1)], source_mask


def grasp_proposal_agreement(current_cloud, original_cloud, config):
    current, original = np.asarray(current_cloud), np.asarray(original_cloud)
    if current.shape != original.shape or not np.isfinite(current).all() or not np.isfinite(original).all():
        return {'accepted': False, 'reason': 'grasp_source_shape_changed'}
    difference = float(np.linalg.norm(current - original, axis=1).max())
    return {'accepted': difference <= config['cache_world_tolerance_m'],
            'input_cloud_identical': bool(np.array_equal(current, original)),
            'maximum_paired_source_change_m': difference,
            'cached_grasps_are_proposals': True, 'current_geometry_admission_required': True}
