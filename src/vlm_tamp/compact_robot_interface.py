"""Public presentation layer; never chooses actions or changes numerical replies.

Copied into a learner workspace as robot_interface.py. Standard library only.
Full replies remain available to Python and in explicit JSON artifacts.
"""
import json
from pathlib import Path
import re
import shutil
import sys
import time
import uuid


INDEX = """Robot interface — compact view

Use: python3 robot.py OPERATION '{"argument": value}'
Read a contract before first use: python3 robot.py help OPERATION
Read its exact argument schema: python3 robot.py schema OPERATION
Use python3 robot.py --full OPERATION ... for an unabridged reply.
Full documentation: tools/full.txt. Read relevant sections, not the whole archive.

Python analysis: from robot import call, brief
call(operation, **arguments) returns the ORIGINAL full response for calculations.
brief(operation, **arguments) returns the same compact display as the CLI.
Both issue exactly one request. Do not reissue a motion to get more detail:
read the returned full_response JSON file instead. Print selected calculation
results rather than whole histories, point arrays or every camera diagnostic.

Observe: reset, observe, episode_status
Geometry: ground, refresh_contact, set_model, inspect_scene
Progress measurements when enabled: measure_surface, compare_surface_geometry
Track: start_tracking, measure_tracking, apply_tracking
Apply measured part pose: apply_part_motion (from track_part_motion/track_video_motion)
Plan: plan_grasp, plan_arc, plan_joint, ik
Act: execute, gripper, finish
Settings: motion_settings (effective limits, speeds and collision policy)
Combined planning: plan_sequence, execute_sequence
Grasp planning: eight native GraspGen proposals, ranked by learned score; one
closest valid IK and one approach query per pose. No angle/height/depth sweeps.
The planner's returned path is used without an extra dense validation pass.
Contact skills when enabled: push, hook_pull (execute by default; preview=true has no motion)
push modes: controlled; push_and_release. Scene collisions block plans
(collision_policy="veto"). Only intended pad/target contact is exempt by default;
the arm, palm and proximal fingers remain checked. No extra dense validation.
Other enabled operations are listed below. help is local and makes no robot call.

Each compact reply preserves success/errors and provides full_response for all
omitted fields. Observation previews list at most two RGB images; other cameras,
robot joints, depth arrays and calibration remain in the full response/files.
Open useful images with your image viewer. More views are available when needed.
Depth geometry, IK and collision search run in the numerical tools. Use Python
with NumPy/SciPy/Pillow for your own analysis; versions: analysis-environment.json.
Single-view ground is a bootstrap hypothesis. Small plane/cylinder fit residuals
do not verify its hinge; use independent calibrated views or measured motion.

Plans do not move the robot. Execute explicitly. Use fresh plans after scene or
robot changes. A solved plan does not establish grasp retention or task success.
Stop motion immediately on task_success=true. Read episode_status for actual
control and time budgets. A planning failure is feedback, not episode completion.
task_success=false means incomplete; it does not mean the last action failed
or had the wrong direction. Compare observed state changes with the goal.
Execute requests sequentially; await_request retrieves an already pending reply.
Scene updates label measured RGB-D motion versus prediction assuming contact.
Inspect those labels and uncertainty. Read full diagnostics when they matter.
"""


def pick(value, keys):
    return {key: value[key] for key in keys if key in value} if isinstance(value, dict) else {}


STATE_KEYS = ('instruction', 'attempt', 'control_steps', 'remaining_control_steps',
              'remaining_wall_seconds', 'task_success', 'episode_done',
              'termination_reason', 'needs_initial_reset', 'observation_id',
              'eef_pose_world_xyzw', 'gripper_qpos', 'sensor_arrays', 'camera_calibration',
              'in_flight_requests', 'control_state_is_last_completed_request',
              'image_coordinate_system')
FIT_KEYS = ('accepted', 'pose_accepted', 'source', 'reason', 'correspondences', 'inliers',
            'rotation_rad', 'rms_m', 'landmark_span_m', 'hinge_world_m',
            'bootstrap_spread90_m', 'radius_m', 'simulator_geometry_used', 'landmark_displacements')


def observation(value):
    result = pick(value, STATE_KEYS)
    if 'grasp_evidence' in value:result['grasp_evidence']=value['grasp_evidence']
    if 'camera_overview' in value:result['camera_overview']=value['camera_overview']
    images = value.get('images')
    if isinstance(images, dict):
        names = [n for n in ('frontview', 'agentview') if n in images]
        names += [n for n in images if n not in names]
        result['images'] = {n: images[n] for n in names[:2]}
        result['available_camera_count'] = len(images)
        if isinstance(value.get('image_sizes'), dict):
            result['image_sizes'] = {name: value['image_sizes'][name] for name in names[:2] if name in value['image_sizes']}
    elif isinstance(images, list):
        result['images'] = images[:2]
        result['available_image_count'] = len(images)
    return result


def evidence(value):
    result = pick(value, ('available', 'pose_source', 'geometry_epoch', 'source', 'error',
                         'diagnostic_error', 'control_steps', 'execution_veto'))
    for name in ('moving', 'contact'):
        if isinstance(value.get(name), dict):
            result[name] = pick(value[name], ('sampled_points', 'supported_fraction',
                'multi_view_free_space_conflict_fraction', 'occluded_or_unresolved_fraction',
                'interpretation', 'simulator_object_state_used'))
    return result


def motion(value):
    result = pick(value, ('source', 'reference_control_step', 'geometry_epoch',
        'scene_changed', 'assumes_contact_retention', 'hinge_refined',
        'hand_predicted_angle_rad', 'execution_veto'))
    if isinstance(value.get('fit'), dict):
        result['fit'] = pick(value['fit'], FIT_KEYS)
    if isinstance(value.get('observation_support'), dict):
        result['observation_support'] = evidence(value['observation_support'])
    return result


def practice(value):
    result = pick(value, ('initial_attempts', 'granted_attempts', 'maximum_attempts',
        'attempts_started', 'remaining_granted_attempts', 'possible_additional_attempts',
        'extension_batch_size', 'completed_attempts'))
    if 'learning_phase' in value:
        result['learning_phase'] = pick(value['learning_phase'],
            ('phase', 'verified_baseline_attempts', 'instructions'))
    return result


def compact_response(response):
    """Lossy display with explicit full-file access; never mutates the response."""
    display = pick(response, ('ok', 'error_type', 'error', 'in_flight_requests'))
    if isinstance(response.get('request_diagnostic'), dict):
        display['request_diagnostic'] = pick(response['request_diagnostic'],
            ('status','operation','field','reason','validation_stage','physical_motion_issued',
             'expensive_work_started','interpretation'))
    if isinstance(response.get('planning_diagnostic'), dict):
        diagnostic = response['planning_diagnostic']
        display['planning_diagnostic'] = pick(diagnostic, ('status', 'grasp_constraint',
            'candidate_rejection_counts','blocking_constraint_examples','requested_constraints',
            'selected_region','height_m','radial_residual_m','allowed_radial_residual_m',
            'minimum_height_m','unconfirmed_vertical_features','model_changed',
            'caller_hint_axis','depth_prior_axis','positive_axis_toward_bulk_score_m',
            'fixed_hinge_displacement_mm','allowed_hinge_displacement_mm',
            'module','robot_valid_ik_solutions','endpoint_rejection_counts','endpoint_rejection_examples',
            'budget_timing_assumptions','shortest_rejected_approach_controls',
            'requested_relative_angle_deg', 'start_collision_free', 'goal_proposals_tried',
            'goal_projection_successes', 'collision_free_goals', 'search_seconds',
            'first_blocking_surface_by_rejected_goal', 'first_blocking_robot_part_by_rejected_goal',
            'interpretation', 'full_diagnostic', 'requested_constraints',
            'candidate_status_counts', 'candidates_tried', 'search_stage_counts',
            'attempted_postures', 'requested_probe_degrees', 'physical_motion_issued'))
        if 'scene_observation_support' in diagnostic:
            display['planning_diagnostic']['scene_observation_support'] = evidence(diagnostic['scene_observation_support'])
    value = response.get('result')
    if not isinstance(value, dict):
        if 'result' in response:
            display['result_type'] = type(value).__name__
        return display
    result = observation(value)
    if 'joint_limit_recovery' in value:result['joint_limit_recovery']=value['joint_limit_recovery']
    if 'grasp_stage_contract' in value:result['grasp_stage_contract']=value['grasp_stage_contract']
    if 'skill' in value:
        result.update(pick(value,('skill','mode','preview','status','stop_reason','diagnostic',
            'robot_motion_executed','requested_contact_speed_m_s','requested_distance_m','requested_joint_degrees',
            'observed_progress','progress_units','contact','momentum','report','images','timing_seconds',
            'observed_joint_change_degrees','observed_joint_change_source','progress_measured_at_control_step',
            'estimated_approach_control_steps','estimated_approach_simulation_seconds','remaining_skill_seconds',
            'scene_refresh_required','next_geometry_interfaces','retained_scene_reference_control_step','feedback_cameras',
            'budget_feasibility_scope','nominal_requested_motion_controls','nominal_cost_assumption','preview_artifacts','robot_valid_plan_available','control_budget_feasible','minimum_following_control_steps',
            'collision_policy','environment_collision_predictions','cartesian_path_deviation','simulator_physical_contacts_enabled',
            'perception_recoveries','recovery_options','physical_infeasibility_proved','release_evidence','motion_settings','retained_joint_estimate_id','motion_warnings','contact_warnings','release_motion_observed','release_motion_reference')))
        result['stage_count']=len(value.get('stages',[]))
        result['feedback_count']=sum('object_motion' in s for s in value.get('stages',[]))
        result['stages']=[pick(s,('stage','artifact_id','control_steps','measured_contact_speed',
            'robot_reference_tracking',
            'requested_motion_m','requested_contact_speed_m_s','observed_contact_progress_m','engagement','object_motion',
            'planning_timing_seconds')) for s in value.get('stages',[])[-3:]]
        diagnostic=value.get('diagnostic',{})
        if 'attempts' in diagnostic:
            from collections import Counter
            result['diagnostic']={**pick(diagnostic,('search_complete','global_infeasibility_proved','timing_seconds')),
                'attempted_candidates':len(diagnostic['attempts']),
                'rejection_counts':dict(Counter(a.get('reason','unknown') for a in diagnostic['attempts']))}
        if 'orientation_attempts' in diagnostic:
            attempts=diagnostic['orientation_attempts']
            result['diagnostic']={**pick(diagnostic,('search_complete','global_infeasibility_proved')),
                'orientation_attempts':len(attempts),'reasons':list(dict.fromkeys(a.get('reason') for a in attempts)),
                'blocking_examples':[pick(r,('blocking_robot_part','blocking_surface','joint_bound_violations'))
                    for a in attempts for r in a.get('rejections',[])][:3]}
        if 'point_attempts' in diagnostic:
            result['diagnostic']={**pick(diagnostic,('search_complete','global_infeasibility_proved','timing_seconds')),
                'region_points_tested':len(diagnostic['point_attempts']),
                'point_outcomes':[{'point_world_m':a['point_world_m'],'reason':a['reason'],
                    'postures_tested':len(a.get('attempts',[]))} for a in diagnostic['point_attempts']]}
        result['search']=pick(value.get('search',{}),('candidate_index','contact_name','total_seconds','timing_seconds',
            'reuse_existing_contact','measured_surface_distance_m','region_candidate_index'))
        if 'observation_refreshes' in value:
            result['observation_refreshes']=[{'control_steps':r['control_steps'],
                'recovered':r['after']['accepted'],'reason':r['before'].get('blocking_surface')}
                for r in value['observation_refreshes']]
        if 'observation' in value:result['observation']=observation(value['observation'])
    result.update(pick(value, ('plan_id', 'candidate_id', 'kind', 'joint_path', 'planning_details', 'note',
        'geometry_epoch', 'model_file', 'point_clouds', 'constraints','budget_timing_assumptions','direction_contract',
        'environment_collision_policy','approach_selection','search_summary','search_timing_seconds','expected_executor','opening_required','approach_aperture_qpos',
        'fixed_joint_preserved','historical_fixed_joint_reused','scene_reference_control_step','hinge_evidence',
        'observed_part_pose_applied','hinge_preserved','signed_direction_preserved','retained_observed_scene_reused',
        'joint_model_from_registered_part','handle_model_required',
        'point_world_m', 'normal_toward_camera_world', 'radius_m', 'surface_fit_residual_m',
        'configured_pad_centers_in_tcp', 'contact_permission', 'object_motion_predicted',
        'grasp_or_attachment_required', 'palm_arm_and_other_surface_checks_retained',
        'hinge_update_applied', 'scene_motion_applied', 'reference_control_step',
        'cameras_with_features', 'correspondences')))
    if 'observation' in value:
        result['observation'] = observation(value['observation'])
        if 'grasp_evidence' in value:result['observation'].pop('grasp_evidence',None)
    if 'model' in value:
        result['model'] = pick(value['model'], ('camera', 'joint_family',
            'joint_origin_world_m', 'joint_axis_world', 'source_anchor_world_m',
            'panel_normal_world', 'observed_handle_radius_m', 'constraint','motion'))
    if isinstance(value.get('grounding_review'),dict):
        result['grounding_review']=pick(value['grounding_review'],('accepted','status','failed_checks',
            'measurements','observation_id','reason','calibration_source','semantic_identity_verified',
            'simulator_object_geometry_used','scope','timing_seconds'))
        result.update(pick(value,('accepted','planner_model_updated','robot_motion_executed','review_file','overlay_legend')))
        # Three deliberately requested checking views, not an arbitrary image dump.
        result['images']=value.get('images',[])[:3]
    if any(key in value for key in ['parts','association_id','track_id','estimate_id']):
        result.update(pick(value,('status','part_id','association_id','track_id','estimate_id','observation_ids',
            'reference_observation_id','reference_frame_index',
            'video_reference_frame_index','video_sampling','video_manifest','depth_control_steps',
            'tracking_backend','mask_file','track_file','estimate_file','motion_supported','fit_observations',
            'verify_observations','joint_type_scope','axis_sign_scope','robot_motion_executed','planner_model_updated',
            'semantic_identity_verified','physical_contact_verified','scope')))
        if 'images' in value:result['images']=value['images'][:4]
        if 'parts' in value:
            result['parts']=[pick(p,('part_id','caller_label','observation_id','camera','mask_box_xyxy','raw_pixels',
                'usable_depth_pixels','masked_robot_or_invalid_pixels','sam_mask_score','overlay','mask_file')) for p in value['parts']]
        if 'views' in value:
            result['views']=[pick(v,('camera','projected_in_frustum','supported_pixels','occluded_source_points',
                'free_space_conflict_source_points','invalid_projected_depth','comparison')) for v in value['views']]
        if 'camera_reports' in value:
            result['camera_reports']=[pick(v,('camera','query_points','reference_observation_id','visible_depth_points_per_frame','rgb_frames','depth_anchors')) for v in value['camera_reports']]
        if 'interval_motion' in value:
            result['interval_motion']=[]
            for interval in value['interval_motion']:
                entry=pick(interval,('observations','finite_visible_pairs','motion_fit_accepted','reason','measured_rotation_deg',
                    'fit_correspondences','fit_inliers','fit_rms_m','fit_failed_checks'))
                displacement=interval.get('landmark_displacements',{})
                if displacement:
                    entry['near_stationary_xy_pairs']=displacement.get('near_stationary_xy_pairs')
                    entry['median_xy_displacement_mm']=displacement.get('xy_displacement_mm',{}).get('median')
                result['interval_motion'].append(entry)
        if 'estimate' in value:
            result['estimate']=pick(value['estimate'],('accepted','reason','joint_origin_world_m','axis_world_unsigned',
                'joint_type_candidate','measured_rotation_deg','finite_visible_pairs','correspondences','inliers','rms_m',
                'bootstrap_spread90_m','radius_m','initialization','failed_checks','minimum_visible_point_radius_prior_m'))
            result['verification']=value.get('verification')
        if 'timing_seconds' in value:result['timing_seconds']=pick(value['timing_seconds'],('total','model_load','image_embedding','mask_prediction'))
    if 'applied_estimate_id' in value:
        result.update(pick(value,('applied_estimate_id','hinge_update_applied','motion_supported','provisional',
            'planner_model_updated','robot_motion_executed','direction_preserved','invalidated_old_plans',
            'source_observation_ids','source_control_steps','semantic_identity_verified','physical_contact_verified','scope')))
        result.update(pick(value,('axis_location_change_m','task_direction_verified_by_motion_fit','direction_note')))
    if isinstance(value.get('fit_diagnostics'), dict):
        result['fit_diagnostics'] = pick(value['fit_diagnostics'],
            ('cylinder_radius_m', 'cylinder_robust_residual_m', 'upright_face_rms_m',
             'cylinder_points', 'handle_on_opposite_observed_face', 'simulator_geometry_used',
             'geometry_source', 'source', 'reason'))
        if isinstance(value['fit_diagnostics'].get('motion_direction_check'), dict):
            result['fit_diagnostics']['motion_direction_check'] = pick(value['fit_diagnostics']['motion_direction_check'],
                ('source','image_hint_status','image_direction_xy','selected_axis','image_hint_overridden_by_depth',
                 'direction_conflict','depth_prior_axis','task_direction_verified'))
    if 'execution' in value:
        result['execution'] = pick(value['execution'], ('planned_commands', 'issued_commands',
            'all_commands_issued', 'terminated_on_simulator_success', 'termination_reason',
            'worker_seconds', 'wall_seconds', 'maximum_joint_reference_error_rad'))
    if 'motion_effect' in value:result['motion_effect']=value['motion_effect']
    if 'scene_observation_support' in value:result['scene_observation_support']=evidence(value['scene_observation_support'])
    if isinstance(value.get('result'), dict):
        result['planning'] = pick(value['result'], ('status', 'planner', 'grasp_constraint',
            'goal_angle_rad', 'terminal_progress_rad', 'total_seconds', 'exact_solution',
            'goal_proposals_tried', 'global_queries', 'grasp_pose_world_xyzw'))
    for name in ('scene_pose_update', 'scene_pose_observation'):
        if isinstance(value.get(name), dict):
            result[name] = motion(value[name])
    if isinstance(value.get('measurement'), dict):
        result['measurement'] = pick(value['measurement'], FIT_KEYS)
    if 'measurement_id' in value:
        result.update(pick(value,('measurement_id','kind','supported','control_steps','observation_ids',
            'normal_world','normal_toward_camera_world','centroid_world_m','values','intervals',
            'quantity_supported','same_capture','semantic_identity_verified','rms_m','inlier_fraction',
            'surface_spans_m','overlay','limitations')))
    if 'declared_at_control_step' in value:
        result.update(pick(value,('goal','criteria','declared_at_control_step','target_source')))
    if 'check_id' in value:
        result.update(pick(value,('check_id','supported','criteria','goal','control_steps','interpretation','scope')))
    if isinstance(value.get('evidence'), dict):
        result['evidence'] = evidence(value['evidence'])
    if 'solutions' in value:
        result['solution_count'] = len(value['solutions'])
    if 'post_closure_replanning_enabled' in value:
        result.update(pick(value,('post_closure_replanning_enabled','post_closure_replanning','stop_reason',
            'requested_degrees','commanded_degrees','observed_part_degrees','remaining_requested_degrees',
            'measured_closure_aperture_qpos','closure_evidence','joint_feedback','recovery_required','recovery_options',
            'planning_error','remaining_control_steps','total_seconds')))
    if 'joint_feedback' in value:
        result['joint_feedback']=[{**pick(r,('status','applied','new_model_predicts_better','observation_ids','evidence_file','geometry_epoch')),
            'fit_rms_m':r.get('estimate',{}).get('rms_m'),
            'verification_rms_m':(r.get('verification') or {}).get('axis_prediction_rms_m'),
            'previous_model_rms_m':r.get('previous_model_verification',{}).get('axis_prediction_rms_m')}
            for r in value['joint_feedback'][-2:]]
    if 'controller_joint_reference_limit_rad_s' in value:result.update(value)
    if 'stages' in value and 'skill' not in value:
        result['stages'] = []
        for stage in value['stages']:
            item = pick(stage, ('stage', 'task_success', 'episode_done', 'termination_reason', 'control_steps'))
            item.update(pick(stage,('grasp_evidence','motion_effect','scene_reference_control_step')))
            if isinstance(stage.get('observation'),dict):
                item['observation']=pick(stage['observation'],('observation_id','control_steps','grasp_evidence'))
            update = stage.get('scene_pose_update', stage.get('scene_pose_observation', {})) or {}
            fit = update.get('fit') or {}
            if 'landmark_displacements' in fit:
                item['landmark_displacements'] = fit['landmark_displacements']
                item['scene_pose_source'] = update.get('source')
            result['stages'].append(item)
    if 'practice_budget' in value:
        result['practice_budget'] = practice(value['practice_budget'])
    if 'granted_attempts' in value:
        result.update(practice(value))
    display['result'] = result
    return display


def present(root, operation, arguments, response, *, mode='python_brief'):
    """Save exact full result and a small display, with presentation-size telemetry."""
    root = Path(root)
    folder = root / 'robot_data' / 'interface'
    folder.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex
    target = folder / (name + '.json')
    full = json.dumps(response, indent=2, ensure_ascii=False) + '\n'
    target.write_text(full)
    display = compact_response(response)
    display.update(operation=operation, full_response=str(target.relative_to(root)))
    text = json.dumps(display, indent=2, ensure_ascii=False)
    record = {'time_unix': time.time(), 'operation': operation,
        'arguments': arguments, 'full_response': display['full_response'],
        'raw_bytes': len(full.encode()), 'display_bytes': len((full if mode == 'cli_full' else text).encode()),
        'presentation_mode': mode,
        'display': display, 'scope': 'Presentation bytes, not measured model tokens.'}
    with (root / 'robot_data' / 'interface-events.jsonl').open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    return display


def main(call, root, argv=None):
    root = Path(root)
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == 'schema':
        if len(args)!=2 or not re.fullmatch('[a-z_]+',args[1]):
            raise ValueError('Use: robot.py schema OPERATION')
        path=root/'tools/schemas'/(args[1]+'.json')
        if not path.is_file():raise ValueError('No schema for that operation; see tools.txt')
        print(path.read_text())
        return
    if not args or args[0] in {'help', '--help', '-h'}:
        name = args[1] if len(args) > 1 else None
        if name and not re.fullmatch('[a-z_]+', name):
            raise ValueError('Use an operation name listed in tools.txt')
        path = root / 'tools' / (name + '.txt') if name else root / 'tools.txt'
        if not path.exists():
            raise ValueError('No contract for that operation; see tools.txt')
        print(path.read_text())
        return
    full = args[0] == '--full'
    if full:
        args.pop(0)
    if not 1 <= len(args) <= 2:
        raise ValueError('Use: robot.py [--full] OPERATION [JSON arguments]')
    operation = args[0]
    arguments = json.loads(args[1]) if len(args) == 2 else {}
    if not isinstance(arguments, dict):
        raise ValueError('Arguments must be a JSON object')
    response = call(operation, **arguments)
    display = present(root, operation, arguments, response, mode='cli_full' if full else 'cli_compact')
    print(json.dumps(response if full else display, indent=2, ensure_ascii=False))


def install(work, document, *, attempts, initial_attempts, practice_mode, control_steps=600):
    """Keep the original manual and generate local operation-specific help."""
    work = Path(work)
    folder = work / 'tools'
    folder.mkdir(exist_ok=True)
    (folder / 'full.txt').write_text(document)
    matches = list(re.finditer(r'(?m)^- ([a-z][a-z_]+)(?:\(|:)', document))
    contracts = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(document)
        contracts.setdefault(match.group(1), []).append(document[match.start():end].strip())
    for name, chunks in contracts.items():
        (folder / (name + '.txt')).write_text('\n\n'.join(chunks) + '\n')
    # Status and transport helpers are local; their full contract precedes the list.
    for name, description in {
        'await_request': 'await_request(request_id): retrieve an existing pending reply. Never reissue a motion just to read its reply.',
        'episode_status': 'episode_status(): local snapshot of success, control/time budgets and in-flight requests; no robot motion.',
    }.items():
        if name not in contracts:
            contracts[name] = [description]
            (folder / (name + '.txt')).write_text(description + '\n')
    budget = f'\nBudget: {initial_attempts} initially granted attempts, maximum {attempts}; {control_steps} control steps each.\n'
    if practice_mode:
        budget += ('One continuing learner session; reset only after the current episode is terminal.\n'
                   'Check practice_status after each episode and follow its current learning_phase.\n')
    (work / 'tools.txt').write_text(INDEX + budget + '\nEnabled help topics: ' + ', '.join(sorted(contracts)) + '\n')
    from vlm_tamp.robot_contracts import export_schemas
    registry=export_schemas(folder/'schemas',contracts)
    (folder/'schema-index.json').write_text(json.dumps(registry,indent=2)+'\n')
    shutil.copyfile(__file__, work / 'robot_interface.py')
    return {'mode': 'compact-v1', 'full_document': 'tools/full.txt',
        'index_bytes': (work / 'tools.txt').stat().st_size,
        'full_document_bytes': len(document.encode()), 'default_rgb_previews': 2,
        'full_python_call_unchanged': True, 'numeric_tool_behavior_unchanged': True,
        'presentation_telemetry': 'robot_data/interface-events.jsonl'}
