"""Grasp admission with full-arm, selected contacts and moving-payload checks."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .collision_paths_v7 import bowl_transform, check_joint_path, drawer_transform, estimated_contact_aperture
from .geometry import pose_matrix, transform_points
from .physical_tasks_v4 import candidate_variants
from .task_ik_v5 import manipulation_poses


def choose_grasp(generated, cloud, spec, calibration, ik, initial_joints, output, sim, scene, open_qpos, save):
    anchor, tree = np.asarray(spec['source_anchor_world_m']), cKDTree(cloud)
    eligible, rejected, chosen = [], [], None
    for index, variant, pose in candidate_variants(generated, calibration):
        distance, membership = float(np.linalg.norm(pose[:3] - anchor)), float(tree.query(pose[:3])[0])
        row = {'index': index, 'variant': variant, 'score': float(generated['scores'][index]),
               'keypoint_distance_m': distance, 'region_distance_m': membership,
               'tcp_pose_world_xyzw': pose.tolist()}
        if distance > .10 or membership > .025:
            rejected.append({**row, 'reason': 'target_region_or_proximity'})
            continue
        if spec['task_type'] == 'open_middle_drawer':
            rotation = pose_matrix(pose)[:3, :3]
            if -float(rotation[:, 2] @ spec['pull_axis_world']) < .65 or abs(float(rotation[:, 0] @ spec['handle_long_axis_world'])) > .6:
                rejected.append({**row, 'reason': 'handle_approach_or_closing_direction'})
                continue
        eligible.append(row)
    eligible.sort(key=lambda row: (row['keypoint_distance_m'], -row['score'], row['index'], row['variant']))
    for ordinal, row in enumerate(eligible):
        grasp = ik.solve(row['tcp_pose_world_xyzw'], initial_joints)
        if grasp is None:
            rejected.append({**row, 'reason': 'grasp_pose_unreachable_or_self_collision'})
            continue
        clearance = scene.joints(sim, grasp['joints'], gripper_qpos=open_qpos, stage='grasp')
        if not clearance['accepted']:
            rejected.append({**row, 'reason': 'full_arm_grasp_collision', 'clearance': clearance})
            continue
        aperture = estimated_contact_aperture(sim, grasp['joints'], scene.region.contact_reference, open_qpos)
        if aperture is None:
            rejected.append({**row, 'reason': 'no_observed_contact_aperture'})
            continue
        closed_clearance = scene.joints(sim, grasp['joints'], gripper_qpos=aperture, stage='grasp')
        if not closed_clearance['accepted']:
            rejected.append({**row, 'reason': 'predicted_closure_collision', 'predicted_gripper_qpos': aperture,
                             'clearance': closed_clearance})
            continue
        approach = ik.approach(np.asarray(initial_joints), grasp)
        if not approach['accepted']:
            rejected.append({**row, 'reason': approach['reason']})
            continue
        target = np.asarray(grasp['pose_world_xyzw'])
        approach_checked = check_joint_path(scene, sim, ik, approach['joint_postures'], gripper_qpos=open_qpos,
            transform_at_pose=lambda pose: np.eye(4),
            stage_at_pose=lambda pose: 'grasp' if np.linalg.norm(np.asarray(pose)[:3] - target[:3]) <= .012 else 'approach',
            check_payload=False, config=ik.config)
        if not approach_checked['accepted']:
            rejected.append({**row, 'reason': 'full_arm_approach_collision', 'path': approach_checked})
            continue
        future = manipulation_poses(spec, grasp['pose_world_xyzw'], ik.config)
        if spec['task_type'] == 'pick_place':
            local = transform_points([spec['object_support_point_initial_world_m']], np.linalg.inv(pose_matrix(target)))[0]
            targets = np.array([transform_points([local], pose_matrix(pose))[0] for pose in future])
            preflight = ik.point_path(targets, local, grasp['joints'])
            transform = bowl_transform(target)
            stage_for = lambda pose: 'place' if np.linalg.norm(transform_points([local], pose_matrix(pose))[0] - spec['goal']['target_world_m']) < .025 else 'carry'
        else:
            preflight = ik.path(future, grasp['joints'])
            transform, stage_for = drawer_transform(scene, target), lambda pose: 'pull'
        if not preflight['accepted']:
            rejected.append({**row, 'reason': preflight['reason']})
            continue
        manipulation_checked = check_joint_path(scene, sim, ik,
            [item['joints'] for item in preflight['solutions']], gripper_qpos=aperture,
            transform_at_pose=transform, stage_at_pose=stage_for, check_payload=True, config=ik.config)
        if not manipulation_checked['accepted']:
            rejected.append({**row, 'reason': 'moving_payload_task_collision', 'path': manipulation_checked})
            continue
        chosen = {**row, 'solution_joints': grasp['joints'], 'fk_pose_world_xyzw': grasp['pose_world_xyzw'],
                  'approach': approach, 'task_preflight': preflight, 'approach_collision_check': approach_checked,
                  'manipulation_collision_check': manipulation_checked, 'predicted_gripper_qpos': aperture,
                  'selection_ordinal': ordinal, 'eligible_pose_variants': len(eligible),
                  'ranking': 'closest grounded anchor among full-arm/contact/payload/IK feasible candidates; score breaks ties',
                  'attachment_assumption': 'hypothesis for preflight only; physical retention must be observed',
                  'rejected_candidates': rejected, 'full_arm_scene_collision_admission': True,
                  'simulator_object_state_used': False, 'unknown_space_validated': False}
        break
    save(Path(output) / 'grasp-admission.json', {'selected': chosen, 'eligible_pose_variants': len(eligible),
        'generated': len(generated['scores']), 'rejected_candidates': rejected,
        'full_arm_scene_collision_admission': True, 'all_planned_episode_failures_counted': True})
    if chosen is None:
        raise ValueError('No detector grasp passes full-arm, contact-region, IK and moving-payload admission')
    return chosen
