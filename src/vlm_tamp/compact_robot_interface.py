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
from collections import Counter


INDEX = """Robot interface — compact view

This experiment exposes two synchronized RGB-D cameras: agentview and wrist.
Open both originals before choosing a contact view; then mark ONE visible face
in either original image. The fixed agentview supplies task context; wrist supplies
local contact detail. The wrist moves with the robot. No extra orbit views or
triangulation are needed. If neither view resolves the face, choose an observation
motion using the ordinary pose tools; it consumes the normal motion budget.
Use camera + observation_id + original contact_pixel_xy to identify a mark.
For hinge motion, use positive joint_degrees plus rotation_direction='cw'/'ccw'.
For plan_arc use positive degrees with that same rotation_direction field.
View from the positive hinge-axis end toward its origin: CCW positive, CW negative.
The hinge axis has its largest absolute world component positive (ties X,Y,Z).
Near-vertical means +Z; horizontal means +X or +Y. Point order, image arrows and
legacy motion_sign do not redefine it. The fitted line is never snapped upright.
CW/CCW does not mean opening/closing automatically. Inspect contact_effect in a
push preview: moving away from the selected face requires another observed face
for pushing or an engaged pulling contact. A reversed normal is not a new face.

Use: python3 robot.py OPERATION '{"argument": value}'
Read a contract before first use: python3 robot.py help OPERATION
Read its exact argument schema: python3 robot.py schema OPERATION
Use python3 robot.py --full OPERATION ... for an unabridged reply.
Full documentation: tools/full.txt. Read relevant sections, not the whole archive.

Goal, motion and evidence are separate. State the desired change between the
moving part and a fixed reference; a contact normal or hinge sign is not that goal.
For a revolute part, motion toward its pivot is not rotation around that pivot.
The returned positive_contact_tangent_world describes the declared model's positive
rotation only. Neither sign is automatically labeled as opening or closing.
When direction is uncertain, a short probe and fixed-view comparison can resolve
it before a long stroke. Inspect feedback.visual_comparison after motion; hand
travel, stroke_executed and continuation availability do not establish goal progress.
For a combined contact action, that comparison covers the whole call. Inspect
phase_visual_comparisons to separate approach, stroke and withdrawal: an approach
can move the object in a different direction from the requested stroke.
The requested push/pull direction describes the stroke, not incidental motion
during approach. Do not learn that direction's effect from the whole-call change alone.

Python analysis: from robot import call, brief
call(operation, **arguments) returns the ORIGINAL full response for calculations.
brief(operation, **arguments) returns the same compact display as the CLI.
Both issue exactly one request. Do not reissue a motion to get more detail:
read the returned full_response JSON file instead. Print selected calculation
results rather than whole histories, point arrays or every camera diagnostic.

Observe: reset, observe, episode_status, inspect_video
inspect_video(start_control_step, end_control_step, max_frames=8) retrieves a
chronological strip of recorded agentview RGB. Optional region_xyxy crops the
original images. Use it to inspect motion between captures; no new physics or
sensor capture is performed. Equal start/end requests one recorded control step.
Geometry: ground, refresh_contact, set_model, inspect_scene
Visual pointing when enabled: use point(observation_id, camera, query) to locate
new visual contacts or functional parts with the local Molmo2 model. Describe the
part and intended action (grasp, push, hook, track or place). Inspect its returned
overlay; use contact_pixel_xy in the existing contact tools or seed segment_parts.
You choose motion direction and the complete moving-part mask. A point is not a
grasp pose, hinge estimate or proof of contact. Reuse established contacts.
ground accepts moving_part_id/contact_part_id from current segment_parts masks.
push and hook_pull accept those same IDs. select_surface_contact accepts
contact_part_id. Reuse the selected part; region_xyxy is optional with a part ID.
Omit regions.contact_box_xyxy for door-only grounding. A failed handle fit can
preserve a usable door/joint model; read grounding_status and handle_fit.
Door-only geometry supports tracking and joint-linked push/hook, not a pinch.
Contact positioning: select_surface_contact, plan_contact_pose
plan_contact_pose(point_world_m, contact_name) converts an observed surface point into a
calibrated fingertip/pad pose. plan_pose instead takes the TCP origin; never
copy a surface point directly into that TCP position.
Progress measurements when enabled: measure_surface, compare_surface_geometry
Track: start_tracking, measure_tracking, apply_tracking
Apply measured part pose: apply_part_motion (from track_part_motion/track_video_motion)
Plan: plan_grasp, plan_arc, plan_joint, ik
Act: execute, gripper, finish
Settings: motion_settings (effective limits, speeds and collision policy)
Combined planning: plan_sequence, execute_sequence
Grasp planning: eight native GraspGen proposals, ranked by learned score; one
closest valid IK and one approach query per pose. No angle/height/depth sweeps.
The network receives the selected target cloud. Check the proposed open hand
against the whole observed workspace before IK; then check the arm and path.
execute of plan_grasp automatically closes after measured approach arrival, for the
planned close_steps. Use close_gripper=false for approach only. No automatic lift;
closure alone does not prove retention. gripper remains available for manual control.
Fingers may contact any environment surface; arm/palm stay checked.
For arcs, point mode constrains only the grip point and leaves wrist orientation
free. It does not carry the whole hand rigidly with the door. Choose rigid to
request that attachment relation, or jaw_axis to retain only the closing axis.
The planner's returned path is used without an extra dense validation pass.
Free approaches use measured progress and report execution.arrival.goal_reached.
Targeted push first plans to the selected face's outward side with finger
collisions checked, then makes a straight contact approach allowing finger contact.
Contact IK is solved first; approach travel is the minimum outward translation
of that solved hand to clear the face plane, with zero added distance/padding.
Point-only pushes use the opposite of the requested push direction as approach side.
Preview approach_plan/approach_images show this route; they do not verify task intent.
Sending all nominal commands does not establish arrival. A targeted push or
sequence closure starts only after its approach has physically reached its goal.
If RGB-D tracks the selected target moving during approach, targeted contact
reacquires that same point before the stroke. contact_handoff reports this work.
Reaching a contact point alone leaves wrist orientation free and does not prove
physical engagement; plan_contact_pose accepts your explicit quaternion when needed.
For a resolved surface face, push and plan_contact_pose also accept
contact_constraint='surface_normal' and its signed outward surface_normal_world.
Use the normal of your selected observed face, for example normal_toward_camera_world
from measure_surface (read-only) or select_surface_contact (installs a surface scene).
cuRobo aligns the chosen pad face against it while
leaving twist free. This chooses orientation, not opening/closing or push direction.
Supply direction_world or signed joint_degrees separately. Inspect the normal and
motion arrows together; moving away from the selected face can lose contact.
Do not flip a normal to label a motion 'closing': select the intended observed face.
A remote camera can see the face opposite the robot. The preview reports the
current contact's signed offset from that face's plane: positive is its outward
side, negative the other side. This is geometric evidence, not a path veto.
Inner jaw faces point across the finger gap; distal pad faces point along the
finger length. These are different contact surfaces, not interchangeable labels.
For goal assessment, compare the moving part with an appropriate fixed reference.
An unsigned plane angle (0..90 degrees) alone cannot determine opening/closing;
it folds angles above 90 degrees. Camera-facing normals also depend on the face selected.
Reset lists each contact's configured collision masks. Active pad surfaces include
inner faces and distal faces. Body-mesh tip references can be passive: contype=0
does not initiate contacts, so they may pass through other passive surfaces.
Use the configured pad faces for contact with such surfaces; no collision flags
or simulator physics are changed by choosing a contact name.
Contact skills when enabled: push_stroke, push, continue_push, hook_pull.
push_stroke(direction_world, distance_m) moves the current hand directly in
simulation, holding its aperture. No object model, prior contact or search is
required. Use it to attempt a push from the current pose, including after a
target-acquisition failure. It does not claim the hand is touching an object.
push/hook_pull execute by default; preview=true has no motion. An unchanged
preview is reused by execution. continue_push_available=true describes executable
robot motion, not established contact or a recommendation to repeat it.
continue_push(direction_world, distance_m) reuses retained geometry and the scene.
Targeted push/hook_pull require contact_pixel_xy, a contact_name from reset,
and an explicit world direction or signed associated
joint increment, including previews. Surface normals do not specify task intent.
reference_direction_world is a geometric reference, not a recommendation about
the task. Measured motion along a request can still move away from the goal.
No new region, hinge model or preview is needed. Distance is the commanded hand
stroke, executed as one stroke by default; measured object motion is separate.
Other arm/gripper motion clears target association; push_stroke remains available.
push modes: controlled; push_and_release. Controlled world-direction/distance
strokes use physical execution feedback without a predicted-scene or object-tracking
veto. Joint-progress requests use 1 cm increments by default and stop continuation
when object motion is unresolved. Internal contact feedback uses RGBDHingeTracker
optical flow; CoTracker3 is a separate explicit part-motion tool.
World distance requests command hand travel as one stroke by default.
Measured tracking lag holds the same reference; an unreached reference returns
physical feedback without issuing the rest blindly. The final point goal must
be reached for completion. No extra collision check is added. Free approaches, hooks and grasped planning retain their scene
collision policy. Wrist RGB-D is included in push feedback. No extra dense validation.
For observed-scene planners, points inside the measured current robot geometry are removed
from the collision map with zero padding. Future candidate poses cannot clear points.
Other enabled operations are listed below. help is local and makes no robot call.

Each compact reply preserves success/errors and provides full_response for all
omitted fields. Observation previews list at most two RGB images; other cameras,
robot joints, depth arrays and calibration remain in the full response/files.
Open useful images with your image viewer. More views are available when needed.
Depth geometry, IK and collision search run in the numerical tools. Use Python
with NumPy/SciPy/Pillow for your own analysis; versions: analysis-environment.json.
Ground requires an exact contact_pixel_xy and two depth-supported hinge pixels.
It uses no upright/cylinder prior. The line is your hypothesis; inspect other views
or measured motion. plan_arc and plan_sequence require an explicit grasp_constraint.

Plans do not move the robot. Execute explicitly. Use fresh plans after scene or
robot changes. A solved plan does not establish grasp retention or task success.
all_commands_issued means the references were sent. execution.endpoint_tracking
reports whether the hand actually arrived and its position/orientation error.
Use that measured result before an action that assumes arrival, such as closure.
Stop motion immediately on task_success=true. Read episode_status for actual
control and time budgets. A planning failure is feedback, not episode completion.
task_success=false means incomplete; it does not mean the last action failed
or had the wrong direction. Compare observed state changes with the goal.
Execute requests sequentially; await_request retrieves an already pending reply.
Scene updates label measured RGB-D motion versus prediction assuming contact.
Inspect those labels and uncertainty. Read full diagnostics when they matter.
Failed contact searches can return rejection images. Red is the native planner's
reported sphere, blue an observed cloud point, and yellow the requested touch
projection. These depict an unexecuted candidate, not a measured collision.
cuRobo checks environment and self-collisions when solving the initial contact pose.
After acquisition, push strokes use IK without an environment collision constraint;
the simulator determines the physical contact response. Robot self-collision and
the existing joint/speed limits remain enabled. An executed push is not proof of
task progress; inspect observed object motion and native task success.
"""


def pick(value, keys):
    return {key: value[key] for key in keys if key in value} if isinstance(value, dict) else {}


def contact_search_summary(diagnostic: dict) -> dict:
    """Keep native endpoint blockers visible without printing every joint vector."""
    attempts=diagnostic.get('attempts',[]);outcomes=Counter();rejections=Counter();examples=[];seen=set();insertions=[]
    for attempt in attempts:
        planning=attempt.get('planning',attempt.get('path_result',{}))
        status=('contact_approach_ik_failed' if attempt.get('contact_approach_failure')
            else attempt.get('reason',planning.get('status','unreported')))
        if attempt.get('contact_approach_failure') and len(insertions)<4:
            failure=attempt['contact_approach_failure']
            insertions.append({**pick(failure,('reason','completed_waypoints','requested_waypoints')),
                'ik_diagnostic':pick(failure.get('ik_diagnostic',{}),('minimum_position_error_m',
                    'constraint_feasible_seeds','local_solver_success','scene_collision_checked','self_collision_checked'))})
        outcomes[status]+=1
        rejected=False
        for endpoint,check in planning.get('endpoints',{}).items():
            if check.get('accepted') is not False:continue
            rejected=True
            part=check.get('blocking_robot_part');surface=check.get('blocking_surface')
            detail=(f'{part} / {surface}' if part or surface else
                'robot self-collision' if check.get('self_rejections') else
                'joint bounds' if check.get('bounds_rejections') else 'endpoint rejected; blocker not reported')
            label=endpoint+': '+detail;rejections[label]+=1
            if label not in seen and len(examples)<4:
                seen.add(label)
                examples.append({'endpoint':endpoint,**pick(check,('blocking_robot_part','blocking_surface',
                    'blocking_sphere_world_m','joint_bound_violations','self_rejections','bounds_rejections'))})
        if not rejected:rejections[status]+=1
    return {**pick(diagnostic,('search_complete','global_infeasibility_proved','physical_infeasibility_proved',
        'timing_seconds','search_work','selected_pixel','selected_contact','planning_stage','push_stroke_issued','ik_diagnostic')),
        'attempted_candidates':len(attempts),'search_outcomes':dict(outcomes),
        'rejection_counts':dict(rejections),'blocking_examples':examples,'contact_approach_failures':insertions}


STATE_KEYS = ('instruction', 'attempt', 'control_steps', 'remaining_control_steps',
              'remaining_wall_seconds', 'task_success', 'episode_done',
              'termination_reason', 'needs_initial_reset', 'observation_id',
              'infrastructure_failure','valid_task_attempt',
              'eef_pose_world_xyzw', 'gripper_qpos', 'sensor_arrays', 'camera_calibration',
              'in_flight_requests', 'control_state_is_last_completed_request',
              'image_coordinate_system','contact_landmarks_in_tcp')
FIT_KEYS = ('accepted', 'pose_accepted', 'source', 'reason', 'correspondences', 'inliers',
            'rotation_rad', 'rms_m', 'landmark_span_m', 'hinge_world_m',
            'bootstrap_spread90_m', 'radius_m', 'simulator_geometry_used', 'landmark_displacements')


def observation(value):
    result = pick(value, STATE_KEYS)
    if 'grasp_evidence' in value:result['grasp_evidence']=value['grasp_evidence']
    if 'camera_overview' in value:result['camera_overview']=value['camera_overview']
    images = value.get('images')
    if isinstance(images, dict):
        names = [n for n in ('agentview', 'wrist') if n in images]
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


def skill_feedback(value: dict) -> dict:
    """Put the meaning of motion results before detailed numerical diagnostics."""
    latest=value.get('part_motion_evidence',{}).get('latest',{})
    current=value.get('observation',{}).get('control_steps')
    arrival=value.get('motion_arrival',value.get('approach_arrival',{}))
    return {'task_goal':value.get('task_instruction'),
        'requested_rotation_direction':value.get('requested_rotation_direction'),
        'requested_motion_scope':'manipulation_stroke; approach/repositioning has separate effects',
        'rotation_view_convention':value.get('rotation_view_convention'),
        'contact_effect':value.get('selection_geometry',{}).get('contact_effect'),
        'task_success':value.get('task_success'),
        'task_progress_assessment':'completed' if value.get('task_success') else 'not_measured_by_motion_tool',
        'requested_direction_world':value.get('contact',{}).get('direction_world'),
        'requested_hand_distance_m':value.get('requested_distance_m'),
        'robot_motion_executed':value.get('robot_motion_executed',False),
        'hand_reference_reached':arrival.get('goal_reached'),
        'measured_hand_travel_along_request_m':value.get('measured_contact_progress_m'),
        'object_motion_status':value.get('object_motion_status','unmeasured'),
        'object_motion_tracker_class':value.get('object_motion_tracker_class'),
        'joint_progress_requires_object_tracking':value.get('joint_progress_requires_object_tracking'),
        'object_progress_is_execution_gate':value.get('object_progress_is_execution_gate'),
        'object_measurement_reason':latest.get('reason'),
        'object_measurement_control_step':latest.get('control_step'),
        'current_control_step':current,
        'object_measurement_is_current':current is not None and latest.get('control_step')==current,
        'visual_comparison':value.get('visual_comparison'),
        'phase_visual_comparisons':value.get('phase_visual_comparisons',[]),
        'interpretation':('A preview issues no motion.' if value.get('preview') else
            'Hand travel is not object progress. The whole-call images include approach/repositioning; '
            'use the stroke phase images to assess the requested direction separately. Compare the moving '
            'part against its fixed goal reference; unresolved tracking does not confirm task direction.')}


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
            'requested_relative_angle_deg', 'start_collision_free', 'start_rejection','current_robot_point_filter','goal_proposals_tried',
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
    if 'skill' in value:display['feedback']=skill_feedback(value)
    elif 'direction_contract' in value:
        display['feedback']={'grounding':'caller-supplied geometric hypothesis',
            'requested_task_motion':value['direction_contract'].get('requested_task_motion'),
            'task_direction_verified':False,
            'surface_plane_rms_m':value.get('fit_diagnostics',{}).get('surface_plane_rms_m'),
            'interpretation':'Grounding lifts your selected pixels; it does not verify the joint identity or which sign achieves the task.'}
    result = observation(value)
    if 'joint_limit_recovery' in value:result['joint_limit_recovery']=value['joint_limit_recovery']
    if 'grasp_stage_contract' in value:result['grasp_stage_contract']=value['grasp_stage_contract']
    if 'grasp_execution' in value:result['grasp_execution']=value['grasp_execution']
    if 'skill' in value:
        result.update(pick(value,('skill','mode','preview','status','stop_reason','diagnostic',
            'task_instruction','contact_constraint','selected_surface_normal_world','planner_rejection_views','visual_comparison',
            'input_mode','contact_reference_reused','engagement_status','continuation_scope','scene_reused','preview_reused','source_preview',
            'observation_cloud_reused','full_scene_builds','contact_searches','continuation',
            'tracking_reference_reused','tracking_reference_camera_count',
            'contact_motion_policy','stroke_scene_collision_veto','collision_check_scope','object_progress_is_execution_gate',
            'joint_progress_requires_object_tracking','object_motion_tracker_class',
            'object_motion_status','object_progress_reference_control_step','planning_limit',
            'part_motion_evidence',
            'continue_push_available','next_operation','grasp_scene_refresh_required','scene_refresh_scope',
            'commanded_contact_distance_m','attempted_stroke_distance_m','measured_contact_progress_m','contact_progress_error_m','stroke_distance_tolerance_m','hand_travel_shortfall_m',
            'robot_motion_executed','requested_contact_speed_m_s','requested_distance_m','requested_joint_degrees',
            'requested_rotation_direction','rotation_view_convention',
            'measured_object_motion_along_request','progress_units','motion_measurement_scope','direction_source','task_direction_verified',
            'contact','momentum','report','images','timing_seconds',
            'observed_joint_change_degrees','observed_joint_change_source','progress_measured_at_control_step',
            'estimated_approach_control_steps','estimated_approach_simulation_seconds','remaining_skill_seconds',
            'scene_refresh_required','next_geometry_interfaces','retained_scene_reference_control_step','feedback_cameras',
            'budget_feasibility_scope','nominal_requested_motion_controls','nominal_cost_assumption','preview_artifacts','robot_valid_plan_available','control_budget_feasible','minimum_following_control_steps',
            'collision_policy','environment_collision_predictions','cartesian_path_deviation','simulator_physical_contacts_enabled','current_robot_point_filter',
            'perception_recoveries','recovery_options','physical_infeasibility_proved','release_evidence','approach_arrival','contact_handoff','motion_arrival','motion_settings','retained_joint_estimate_id','motion_warnings','contact_warnings','release_motion_observed','release_motion_reference')))
        result['stage_count']=len(value.get('stages',[]))
        if value.get('contact_orientation'):
            result['contact_orientation']=pick(value['contact_orientation'],('surface_normal_world',
                'desired_robot_contact_normal_world','robot_contact_normal_world','normal_error_degrees',
                'requested_motion_dot_outward_normal','current_contact_signed_plane_offset_m','pose_source',
                'pose_control_step','surface_pose_current','surface_reference_control_step','wrist_twist','joint_prediction'))
        for field in ('approach_plan','approach_images'):
            if field in value:result[field]=value[field]
        if value.get('selection_geometry'):
            result['selection_geometry']=pick(value['selection_geometry'],('surface_normal_world','contact_effect',
                'initial_robot_contact_signed_plane_offset_m','requested_motion_dot_outward_normal',
                'same_plane_side_camera_candidates','normal_source','contact_role'))
        result['feedback_count']=sum('object_motion' in s for s in value.get('stages',[]))
        result['stages']=[pick(s,('stage','artifact_id','control_steps','measured_contact_speed',
            'robot_reference_tracking','arrival','visual_comparison',
            'requested_motion_m','requested_contact_speed_m_s','observed_contact_progress_m','engagement','object_motion',
            'planning_timing_seconds','scene_update','material_point_offset_m','physical_motion','control_budget_truncated')) for s in value.get('stages',[])[-3:]]
        diagnostic=value.get('diagnostic',{})
        if 'attempts' in diagnostic:
            result['diagnostic']=contact_search_summary(diagnostic)
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
            'reuse_existing_contact','measured_surface_distance_m','region_candidate_index','search_work','contact_pose_selection','contact_model','contact_constraint'))
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
        'grounding_status','handle_fit','handle_model_available','selected_parts',
        'point_world_m', 'normal_toward_camera_world', 'radius_m', 'surface_fit_residual_m',
        'configured_pad_centers_in_tcp', 'contact_landmarks_in_tcp','contact_pose','contact_permission', 'object_motion_predicted',
        'grasp_or_attachment_required', 'palm_arm_and_other_surface_checks_retained',
        'hinge_update_applied', 'scene_motion_applied', 'reference_control_step',
        'cameras_with_features', 'correspondences')))
    if 'observation' in value:
        result['observation'] = observation(value['observation'])
        if 'grasp_evidence' in value:result['observation'].pop('grasp_evidence',None)
    if 'model' in value:
        result['model'] = pick(value['model'], ('camera', 'joint_family',
            'joint_origin_world_m', 'joint_axis_world', 'source_anchor_world_m',
            'panel_normal_world', 'observed_handle_radius_m', 'constraint','task_motion_intent',
            'task_direction_verified','generic_part_joint_model'))
    if 'pointing_backend' in value:
        result.update(pick(value,('pointing_backend','status','query','observation_id','camera',
            'points','images','raw_text','request_file','model_output_file','timing_seconds',
            'robot_motion_executed','planner_model_updated','scope')))
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
             'geometry_source', 'source', 'reason','target_selection','handle_fit','selected_parts',
             'surface_plane_rms_m','hinge_depth_samples_world_m','contact_point_world_m',
             'moving_observed_points','contact_observed_points','axis_prior','shape_completion'))
        if isinstance(value['fit_diagnostics'].get('motion_direction_check'), dict):
            result['fit_diagnostics']['motion_direction_check'] = pick(value['fit_diagnostics']['motion_direction_check'],
                ('source','image_hint_status','image_direction_xy','selected_axis',
                 'task_direction_status','coordinate_convention_only','positive_rotation_meaning',
                 'task_direction_verified','fixed_joint_preserved'))
    if 'execution' in value:
        result['execution'] = pick(value['execution'], ('planned_commands', 'issued_commands',
            'all_commands_issued', 'terminated_on_simulator_success', 'termination_reason',
            'worker_seconds', 'wall_seconds', 'maximum_joint_reference_error_rad','endpoint_tracking',
            'arrival','arrival_scope','terminated_on_tracking_error','terminated_on_control_budget','nominal_path_commands'))
    if 'motion_effect' in value:result['motion_effect']=value['motion_effect']
    if 'selected_frame_count' in value:
        result.update(pick(value,('available','camera','requested_control_interval','available_control_interval',
            'recorded_frames_in_interval','frames','selected_frame_count','comparison_image','source',
            'new_sensor_capture','robot_motion_issued','depth_available','interpretation')))
    if value.get('available') is False and 'robot_motion_issued' in value:
        result.update(pick(value,('available','reason','available_control_interval','new_sensor_capture','robot_motion_issued')))
    if 'visual_comparison' in value:result['visual_comparison']=value['visual_comparison']
    if 'phase_visual_comparisons' in value:result['phase_visual_comparisons']=value['phase_visual_comparisons']
    if 'scene_observation_support' in value:result['scene_observation_support']=evidence(value['scene_observation_support'])
    if isinstance(value.get('result'), dict):
        result['planning'] = pick(value['result'], ('status', 'planner', 'grasp_constraint',
            'goal_angle_rad', 'terminal_progress_rad', 'total_seconds', 'exact_solution',
            'goal_proposals_tried', 'global_queries', 'grasp_pose_world_xyzw','execution_contract'))
        if value['result'].get('contact_pose'):
            result['contact_pose']=pick(value['result']['contact_pose'],('contact_name','contact_role',
                'contact_constraint','orientation_constraint','point_world_m','surface_normal_world',
                'contact_orientation','gripper_aperture_qpos','tcp_pose_role','tcp_pose_world_xyzw'))
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
            item.update(pick(stage,('grasp_evidence','motion_effect','scene_reference_control_step','artifact_id')))
            if 'execution' in stage:
                item['execution']=pick(stage['execution'],('issued_commands','planned_commands','termination_reason','arrival'))
            if stage.get('execution',{}).get('endpoint_tracking'):
                item['endpoint_tracking']=stage['execution']['endpoint_tracking']
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
