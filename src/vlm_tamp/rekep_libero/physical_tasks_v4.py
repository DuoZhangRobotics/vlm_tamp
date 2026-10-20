"""RGB-grounded task execution with ordinary contact physics and typed goals.

This module does not call the native ReKep optimizers or ST-pRRTC. It exports
their shared numeric task interface and uses a constraint-guided IK baseline.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .geometry import graspgen_to_tcp, interpolate_path, pose_matrix
from .perception import _read_observation
from .program import parse_program
from .task_geometry_v4 import bowl_spec, drawer_spec, referenced_points


def cached_scene(source, observation):
    source, observation = Path(source), Path(observation)
    cameras, metadata = _read_observation(observation / 'obs.npz', observation / 'metadata.json')
    main = json.loads((source / 'perception/perception.json').read_text())['main_camera']
    with np.load(source / 'perception/keypoints.npz', allow_pickle=False) as arrays:
        bundle = {key: arrays[key].copy() for key in arrays.files}
    old_rgb = np.asarray(Image.open(source / 'perception/rgb.png'))
    new_rgb = cameras[main]['rgb']
    bundle.update(cameras=cameras, meta=metadata, main_camera=main)
    bundle['points_world'] = cameras[main]['world_points'][tuple(bundle['pixels_yx'].T)]
    task_type = json.loads((source / 'program/metadata.json').read_text())['task_type']
    program = parse_program((source / 'program/output_raw.txt').read_text(), len(bundle['points_world']),
                            task_type, attachment_cost=lambda _index: 1.)
    source_mask = np.load(source / 'stage1-grasps/selected_mask.npy', allow_pickle=False)
    selected_indices = sorted(set(program.grasp_keypoints + referenced_points(program.stages[1])) - {-1})
    checks = {'source_mask_rgb_identical': bool(np.array_equal(old_rgb[source_mask], new_rgb[source_mask]))}
    for index in selected_indices:
        y, x = bundle['pixels_yx'][index]
        patch = np.s_[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        checks[f'keypoint_{index}_rgb_patch_identical'] = bool(np.array_equal(old_rgb[patch], new_rgb[patch]))
    if not all(checks.values()):
        raise ValueError('Cached selected RGB evidence changed; no paid API retry is permitted')
    bundle['cache_evidence'] = {**checks, 'full_primary_rgb_identical': bool(np.array_equal(old_rgb, new_rgb)),
                                'manually_rebound_landmarks': False, 'new_api_calls': 0}
    source_cloud = cameras[main]['world_points'][source_mask]
    valid = np.isfinite(source_cloud).all(axis=1)
    source_cloud = source_cloud[valid]
    return bundle, program, source_cloud, source_mask


def compile_scene(bundle, program, source_cloud, config):
    if program.task_type == 'pick_place':
        spec, target_cloud, target_mask = bowl_spec(bundle, program, source_cloud, config['release_clearance_m'])
        return spec, source_cloud, target_cloud, target_mask
    spec, handle = drawer_spec(bundle, program, source_cloud, config['drawer_travel_m'])
    return spec, handle, None, None


def candidate_variants(generated, calibration):
    symmetry = np.diag([-1., -1., 1., 1.])
    shift = np.eye(4)
    shift[2, 3] = .0584 - calibration['physical_finger_attach_z_m']
    variants = [('asset_contact_aligned', shift),
                ('asset_contact_aligned_half_turn', shift @ symmetry),
                ('raw', np.eye(4)), ('antipodal_half_turn', symmetry)]
    for index in generated['retained_indices']:
        for name, transform in variants:
            yield int(index), name, graspgen_to_tcp(generated['poses_world_base'][index] @ transform, calibration)


def choose_grasp(generated, keypoint, cloud, spec, calibration, sim, initial_joints):
    tree = cKDTree(cloud)
    feasible, rejected = [], []
    for index, variant, pose in candidate_variants(generated, calibration):
        distance = float(np.linalg.norm(pose[:3] - keypoint))
        membership = float(tree.query(pose[:3])[0])
        row = {'index': index, 'variant': variant, 'score': float(generated['scores'][index]),
               'keypoint_distance_m': distance, 'region_distance_m': membership,
               'tcp_pose_world_xyzw': pose.tolist()}
        if distance > .10 or membership > .025:
            rejected.append({**row, 'reason': 'target_region_or_proximity'})
            continue
        if spec['task_type'] == 'open_middle_drawer':
            rotation = pose_matrix(pose)[:3, :3]
            inward = -float(rotation[:, 2] @ spec['pull_axis_world'])
            bar_alignment = abs(float(rotation[:, 0] @ spec['handle_long_axis_world']))
            row.update(inward_approach_cosine=inward, closing_bar_alignment=bar_alignment)
            if inward < .65 or bar_alignment > .6:
                rejected.append({**row, 'reason': 'handle_approach_or_closing_direction'})
                continue
        feasible.append(row)
    feasible.sort(key=lambda row: (row['keypoint_distance_m'], -row['score'], row['index'], row['variant']))
    for row in feasible:
        solved = sim.multi_ik(row['tcp_pose_world_xyzw'], seed_joints=initial_joints, max_iterations=180)
        candidates = solved['solutions']
        if not candidates:
            rejected.append({**row, 'reason': 'ik_unreachable'})
            continue
        chosen = min(candidates, key=lambda item: np.linalg.norm(np.asarray(item['cspace_position']) - initial_joints))
        checked = checked_approach(sim, initial_joints, chosen)
        if checked is None:
            rejected.append({**row, 'reason': 'no_continuous_approach_ik'})
            continue
        return {**row, 'solution_joints': chosen['cspace_position'],
                'fk_pose_world_xyzw': chosen['eef_pose_world_xyzw'],
                'ik_position_error_m': chosen['position_error'], 'ik_orientation_error_rad': chosen['orientation_error'],
                'rejected_candidates': rejected, 'eligible_pose_variants': len(feasible),
                'approach': checked,
                'software_collision_admission': False, 'ranking': 'closest eligible keypoint distance, confidence tie-break'}
    raise ValueError('No detector candidate satisfies object-region, orientation and IK admission')


def checked_approach(sim, initial_joints, target_solution):
    """Plan robot-only FK transport to standoff, then continuous-seed approach.

    The diagnostic bypasses environment collision admission, but it still
    rejects a candidate if no IK path to its standoff and target can be built.
    """
    target = np.asarray(target_solution['eef_pose_world_xyzw'])
    target_joints = np.asarray(target_solution['cspace_position'])
    standoff = target.copy()
    standoff[:3] -= .07 * pose_matrix(target)[:3, 2]
    solved = sim.multi_ik(standoff, seed_joints=target_joints, max_iterations=180, max_starts=12)
    if not solved['solutions']:
        return None
    pre = min(solved['solutions'], key=lambda item: np.linalg.norm(np.asarray(item['cspace_position']) - target_joints))
    pre_joints = np.asarray(pre['cspace_position'])
    count = max(1, int(np.ceil(np.max(np.abs(pre_joints - initial_joints)) / .12)))
    joint_path = [np.asarray(initial_joints) + value * (pre_joints - initial_joints)
                  for value in np.linspace(0., 1., count + 1)]
    poses = [sim.fk(joints)['eef_pose_world_xyzw'] for joints in joint_path]
    seed = pre_joints
    for waypoint in interpolate_path([pre['eef_pose_world_xyzw'], target], position_step=.01)[1:]:
        solved = sim.multi_ik(waypoint, seed_joints=seed, max_iterations=180, max_starts=12)
        if not solved['solutions']:
            return None
        item = min(solved['solutions'], key=lambda item: np.linalg.norm(np.asarray(item['cspace_position']) - seed))
        following = np.asarray(item['cspace_position'])
        if np.max(np.abs(following - seed)) > .5:
            return None
        joint_path.append(following)
        poses.append(item['eef_pose_world_xyzw'])
        seed = following
    return {'poses_world_xyzw': np.asarray(poses).tolist(),
            'joint_postures': np.asarray(joint_path).tolist(), 'all_waypoints_fk_bound': True,
            'environment_collision_checked': False}


def approach_path(current, selected):
    target = np.asarray(selected['tcp_pose_world_xyzw'])
    standoff = target.copy()
    standoff[:3] -= .07 * pose_matrix(target)[:3, 2]
    return interpolate_path(np.asarray([current, standoff, target]), position_step=.015, rotation_step=.12)


def original_residuals(program, stage_index, pose, points, held):
    source = program.grasp_keypoints[0]
    program.set_attachment_cost(lambda index: 0. if held and index == source else 1.)
    stage = program.stages[stage_index]
    return {constraint.name: constraint(np.asarray(pose)[:3], points)
            for constraint in stage.subgoal_constraints + stage.path_constraints}
