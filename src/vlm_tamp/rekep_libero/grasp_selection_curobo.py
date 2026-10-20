"""Search multiple IK postures before rejecting a GraspGen pose.

The environment/contact checker is deliberately shared with the frozen v11
baseline. cuRobo proposes configurations; it does not certify this scene.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .collision_paths_v7 import bowl_transform, check_joint_path, drawer_transform, estimated_contact_aperture
from .geometry import pose_matrix, transform_points
from .physical_tasks_v4 import candidate_variants
from .task_ik_v5 import manipulation_poses


def admit_posture(grasp, spec, ik, initial_joints, sim, scene, open_qpos):
    clearance = scene.joints(sim, grasp['joints'], gripper_qpos=open_qpos, stage='grasp')
    if not clearance['accepted']:
        return None, {'reason': 'full_arm_grasp_collision', 'clearance': clearance}
    aperture = estimated_contact_aperture(sim, grasp['joints'], scene.region.contact_reference, open_qpos)
    if aperture is None:
        return None, {'reason': 'no_observed_contact_aperture'}
    closure = scene.joints(sim, grasp['joints'], gripper_qpos=aperture, stage='grasp')
    if not closure['accepted']:
        return None, {'reason': 'predicted_closure_collision', 'clearance': closure}
    approach = ik.approach(np.asarray(initial_joints), grasp)
    if not approach['accepted']:
        return None, {'reason': approach['reason']}
    target = np.asarray(grasp['pose_world_xyzw'])
    checked_approach = check_joint_path(scene, sim, ik, approach['joint_postures'], gripper_qpos=open_qpos,
        transform_at_pose=lambda pose: np.eye(4),
        stage_at_pose=lambda pose: 'grasp' if np.linalg.norm(np.asarray(pose)[:3]-target[:3]) <= .012 else 'approach',
        check_payload=False, config=ik.config)
    if not checked_approach['accepted']:
        return None, {'reason': 'full_arm_approach_collision', 'path': checked_approach}
    future = manipulation_poses(spec, target, ik.config)
    if spec['task_type'] == 'pick_place':
        local = transform_points([spec['object_support_point_initial_world_m']], np.linalg.inv(pose_matrix(target)))[0]
        targets = np.array([transform_points([local], pose_matrix(pose))[0] for pose in future])
        preflight = ik.point_path(targets, local, grasp['joints'])
        transform = bowl_transform(target)
        stage_for = lambda pose: 'place' if np.linalg.norm(transform_points([local], pose_matrix(pose))[0]-spec['goal']['target_world_m']) < .025 else 'carry'
    else:
        preflight = ik.path(future, grasp['joints'])
        transform, stage_for = drawer_transform(scene, target), lambda pose: 'pull'
    if not preflight['accepted']:
        return None, {'reason': preflight['reason']}
    checked_task = check_joint_path(scene, sim, ik, [item['joints'] for item in preflight['solutions']],
        gripper_qpos=aperture, transform_at_pose=transform, stage_at_pose=stage_for,
        check_payload=True, config=ik.config)
    if not checked_task['accepted']:
        return None, {'reason': 'moving_payload_task_collision', 'path': checked_task}
    return {'solution_joints': grasp['joints'], 'fk_pose_world_xyzw': grasp['pose_world_xyzw'],
            'minimum_grasp_joint_reserve_rad': grasp['joint_reserve_rad'],
            'approach': approach, 'task_preflight': preflight,
            'approach_collision_check': checked_approach, 'manipulation_collision_check': checked_task,
            'predicted_gripper_qpos': aperture}, None


def choose_grasp(generated, cloud, spec, calibration, ik, initial_joints, output, sim, scene, open_qpos, save):
    anchor, tree = np.asarray(spec['source_anchor_world_m']), cKDTree(cloud)
    eligible, rejected, chosen, tested_postures = [], [], None, 0
    for index, variant, pose in candidate_variants(generated, calibration):
        distance, membership = float(np.linalg.norm(pose[:3]-anchor)), float(tree.query(pose[:3])[0])
        row = {'index': index, 'variant': variant, 'score': float(generated['scores'][index]),
               'keypoint_distance_m': distance, 'region_distance_m': membership,
               'tcp_pose_world_xyzw': pose.tolist()}
        if distance > .10 or membership > .025:
            rejected.append({**row, 'reason': 'target_region_or_proximity'})
            continue
        if spec['task_type'] == 'open_middle_drawer':
            rotation = pose_matrix(pose)[:3, :3]
            if -float(rotation[:, 2]@spec['pull_axis_world']) < .65 or abs(float(rotation[:, 0]@spec['handle_long_axis_world'])) > .6:
                rejected.append({**row, 'reason': 'handle_approach_or_closing_direction'})
                continue
        eligible.append(row)
    eligible.sort(key=lambda row: (row['keypoint_distance_m'], -row['score'], row['index'], row['variant']))
    for ordinal, row in enumerate(eligible):
        solutions = ik.solve_many(row['tcp_pose_world_xyzw'], initial_joints,
                                  maximum=ik.config['maximum_ik_postures_per_grasp'])
        if not solutions:
            rejected.append({**row, 'reason': 'grasp_pose_unreachable_or_self_collision', 'ik_alternatives': 0})
        for alternative, grasp in enumerate(solutions):
            tested_postures += 1
            admitted, failure = admit_posture(grasp, spec, ik, initial_joints, sim, scene, open_qpos)
            if failure:
                rejected.append({**row, 'ik_alternative': alternative, 'joints': grasp['joints'], **failure})
                continue
            chosen = {**row, **admitted, 'ik_alternative': alternative,
                'ik_alternatives_available': len(solutions), 'selection_ordinal': ordinal,
                'eligible_pose_variants': len(eligible), 'tested_ik_postures': tested_postures,
                'ranking': 'closest grounded anchor; multiple IK postures ranked by travel and joint reserve; confidence breaks grasp ties',
                'attachment_assumption': 'hypothesis only; physical retention must be observed',
                'rejected_candidates': rejected, 'full_arm_scene_collision_admission': True,
                'simulator_object_state_used': False, 'unknown_space_validated': False,
                'curobo_used': True, 'curobo_motiongen_used': False, 'collision_free_guarantee': False}
            break
        save(Path(output)/'grasp-admission-progress.json', {'eligible_pose_ordinal': ordinal,
            'eligible_pose_variants': len(eligible), 'tested_ik_postures': tested_postures,
            'selected': chosen is not None})
        if chosen is not None:
            break
    save(Path(output)/'grasp-admission.json', {'selected': chosen, 'eligible_pose_variants': len(eligible),
        'generated': len(generated['scores']), 'tested_ik_postures': tested_postures,
        'rejected_candidates': rejected, 'full_arm_scene_collision_admission': True,
        'all_planned_episode_failures_counted': True, 'curobo_used': True,
        'contact_policy_changed_from_v11': False, 'collision_free_guarantee': False})
    if chosen is None:
        raise ValueError('No cuRobo IK alternative passes full-arm/contact/approach/manipulation admission')
    return chosen
