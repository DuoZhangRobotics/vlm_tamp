"""Select detector grasps that permit approach and subsequent manipulation."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geometry import pose_matrix, transform_points
from .physical_tasks_v4 import candidate_variants
from .task_ik_v5 import manipulation_poses


def choose_grasp(generated, cloud, spec, calibration, ik, initial_joints, outdir, sim, scene):
    anchor = np.asarray(spec['source_anchor_world_m'])
    tree = cKDTree(cloud)
    eligible, rejected = [], []
    for index, variant, pose in candidate_variants(generated, calibration):
        distance = float(np.linalg.norm(pose[:3] - anchor))
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
        eligible.append(row)
    eligible.sort(key=lambda row: (row['keypoint_distance_m'], -row['score'], row['index'], row['variant']))
    chosen = None
    for ordinal, row in enumerate(eligible):
        grasp = ik.solve(row['tcp_pose_world_xyzw'], initial_joints)
        if grasp is None:
            rejected.append({**row, 'reason': 'grasp_pose_unreachable_or_self_collision'})
            continue
        clearance = scene.joints(sim, grasp['joints'])
        if not clearance['accepted']:
            rejected.append({**row, 'reason': 'observed_grasp_collision', 'clearance': clearance})
            continue
        future = manipulation_poses(spec, grasp['pose_world_xyzw'], ik.config)
        # Test the demanding far endpoint first, then the complete sampled path.
        if spec['task_type'] == 'pick_place':
            local = transform_points([spec['object_support_point_initial_world_m']],
                                     np.linalg.inv(pose_matrix(grasp['pose_world_xyzw'])))[0]
            targets = np.array([transform_points([local], pose_matrix(pose))[0] for pose in future])
            endpoint = ik.solve_point(targets[-1], local, grasp['joints'])
        else:
            endpoint = ik.solve(future[-1], grasp['joints'])
        if endpoint is None:
            rejected.append({**row, 'reason': 'complete_task_endpoint_unreachable'})
            continue
        preflight = (ik.point_path(targets, local, grasp['joints']) if spec['task_type'] == 'pick_place'
                     else ik.path(future, grasp['joints']))
        if not preflight['accepted']:
            rejected.append({**row, 'reason': preflight['reason'], 'failed_manipulation_sample': preflight['sample']})
            continue
        approach = ik.approach(np.asarray(initial_joints), grasp)
        if not approach['accepted']:
            rejected.append({**row, 'reason': approach['reason']})
            continue
        rejected_clearance = None
        for sample in list(range(0, len(approach['joint_postures']), 3)) + [len(approach['joint_postures']) - 1]:
            checked = scene.joints(sim, approach['joint_postures'][sample])
            if not checked['accepted']:
                rejected_clearance = {'reason': 'observed_approach_collision', 'sample': sample, 'clearance': checked}
                break
        if rejected_clearance is None:
            solutions = preflight['solutions']
            for sample in list(range(1, len(solutions), 3)) + [len(solutions) - 1]:
                checked = scene.joints(sim, solutions[sample]['joints'], carrying=True)
                if not checked['accepted']:
                    rejected_clearance = {'reason': 'observed_manipulation_collision', 'sample': sample, 'clearance': checked}
                    break
        if rejected_clearance is not None:
            rejected.append({**row, **rejected_clearance})
            continue
        chosen = {**row, 'solution_joints': grasp['joints'], 'fk_pose_world_xyzw': grasp['pose_world_xyzw'],
                  'ik_position_error_m': grasp['position_error_m'], 'ik_orientation_error_rad': grasp['orientation_error_rad'],
                  'task_preflight': preflight, 'approach': approach,
                  'nominal_manipulation_poses_world_xyzw': future.tolist(),
                  'eligible_pose_variants': len(eligible), 'rejected_candidates': rejected,
                  'selection_ordinal': ordinal,
                  'ranking': 'closest observed anchor among full-task/approach feasible detector poses; confidence tie-break',
                  'preflight_attachment_assumption': 'initial selected surface unchanged until closure; recheck after observed binding',
                  'observed_surface_collision_admission': True,
                  'unknown_space_validated': False, 'held_object_collision_geometry_validated': False}
        break
    record = {'selected': chosen, 'rejected_candidates': rejected, 'eligible_pose_variants': len(eligible),
              'generated': len(generated['scores']), 'retained_detector_candidates': len(generated['retained_indices']),
              'full_task_admission': True, 'simulator_object_state_used': False}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'grasp-admission.json').write_text(json.dumps(record, indent=2) + '\n')
    if chosen is None:
        raise ValueError('No detector grasp admits the complete manipulation and its approach')
    print(json.dumps({'grasp_index': chosen['index'], 'variant': chosen['variant'],
                      'full_task_admission': True, 'rejected_before_selection': len(rejected)}), flush=True)
    return chosen
