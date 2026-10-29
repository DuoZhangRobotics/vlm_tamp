"""Actor-visible diagnostics from the current numerical planning query."""
import math
from collections import Counter


def summarize_pose_failure(attempts, native_scene=None):
    """Identify the actual failed stage of a free-pose query from public geometry."""
    statuses=Counter(a['result'].get('status','unknown') for a in attempts)
    reasons=Counter();examples=[]
    cover=(native_scene or {}).get('gripper_cover',[])
    for attempt in attempts:
        for stage,check in attempt['result'].get('endpoints',{}).items():
            if check.get('accepted') is not False:continue
            part=int(check.get('last_rejected_part',-1));surface=int(check.get('last_rejected_surface',-1))
            if check.get('joint_bound_violations'):
                robot_part,reason='robot','joint_bounds'
            else:
                robot_part=(f'robot0_link{part-2000}_collision' if 2001<=part<=2007 else
                    f'robot_arm_sphere_{part}' if 0<=part<37 else
                    next((r['robot_geom_name'] for r in cover if r['sphere_range'][0]==part-1000),'configured_gripper')
                    if 1000<=part<2000 else 'robot')
                reason={0:'stationary_observed_geometry',1:'moving_observed_geometry',
                    2:'selected_contact_region'}.get(surface,'robot_self_collision_or_bounds')
            key=f'{stage}: {robot_part} / {reason}';reasons[key]+=1
            if not any(x['key']==key for x in examples):
                examples.append({'key':key,'stage':stage,'robot_part':robot_part,'reason':reason,
                    'joint_bound_violations':check.get('joint_bound_violations',[])})
    return {'status':'no_robot_valid_ik' if not attempts else 'pose_endpoint_rejected' if reasons else 'pose_path_search_failed',
        'module':'full_pose_ik' if not attempts else 'native_scene.endpoint_admission' if reasons else 'native_scene.path_search',
        'robot_valid_ik_solutions':len(attempts),'endpoint_rejection_counts':dict(reasons),
        'endpoint_rejection_examples':examples,'search_stage_counts':dict(statuses),
        'physical_motion_issued':False,
        'interpretation':'Collision checks use observed geometry. Endpoint rejection is distinct from path-search failure; neither proves every alternative impossible.'}


def summarize_arc_failure(result,native_scene=None):
    goals=result.get('goal_proposals',[])
    projected=[g for g in goals if g.get('projection',{}).get('accepted')]
    checked=[g for g in projected if g.get('scene') is not None]
    positions=[g['projection']['position_error_m'] for g in goals
               if math.isfinite(g.get('projection',{}).get('position_error_m',float('inf')))]
    orientations=[g['projection']['jaw_inequality_error']/.15 for g in goals
                  if math.isfinite(g.get('projection',{}).get('jaw_inequality_error',float('inf')))]
    reasons=Counter(reason for g in goals for reason in
        g.get('rejection_reasons',[g.get('reason')]) if reason)
    examples=[]
    for reason in reasons:
        matching=[g for g in goals if reason in g.get('rejection_reasons',[g.get('reason')])]
        g=min(matching,key=lambda g:g.get('projection',{}).get('position_error_m',
            g.get('gpu_input_position_error_m',float('inf'))))
        p=g.get('projection',{})
        examples.append({'module':('partial_pose_arc.wrist_policy' if reason=='wrist_excursion_limit'
            else 'partial_pose_arc.endpoint_projection' if reason=='endpoint_constraint_projection_failed'
            else 'native_scene.joint_bounds' if reason=='joint_bounds'
            else 'native_scene.collision' if reason=='scene_collision'
            else 'curobo_v2_arc.candidate_admission'),
            'reason':reason,'count':reasons[reason],
            'joint_bound_violations':g.get('scene',{}).get('joint_bound_violations',[]),
            'input_position_error_mm':1000*g['gpu_input_position_error_m'] if 'gpu_input_position_error_m' in g else None,
            'final_position_error_mm':1000*p['position_error_m'] if 'position_error_m' in p else None,
            'allowed_position_error_mm':1000*p.get('position_tolerance_m',g.get('position_tolerance_m'))
                if p.get('position_tolerance_m',g.get('position_tolerance_m')) is not None else None,
            'wrist_reorientation_deg':g.get('wrist_excursion_deg'),
            'allowed_wrist_reorientation_deg':g.get('wrist_limit_deg',result.get('planning_wrist_limit_deg')),
            'weighted_angular_error':p.get('jaw_inequality_error'),
            'allowed_weighted_angular_error':p.get('angular_weighted_tolerance',1e-6) if p else None,
            'orientation_error_deg':p.get('orientation_error_deg'),
            'allowed_orientation_error_deg':math.degrees(p.get('orientation_tolerance_rad') or 1e-6/.15)
                if p and p.get('orientation_tolerance_applied',True) else None,
            'consequence':'Candidate was not offered to the path planner.'})
    roles={};parts={};previous=result.get('start',{})
    cover=(native_scene or {}).get('gripper_cover',[])
    for goal in goals:
        scene=goal.get('scene')
        if scene is None:continue
        if not scene.get('accepted'):
            if scene.get('bounds_rejections',0)>previous.get('bounds_rejections',0):
                role,part='joint_bounds','robot'
            elif scene.get('self_rejections',0)>previous.get('self_rejections',0):
                role,part='robot_self_collision','robot'
            else:
                role={0:'stationary_observed_geometry',1:'moving_noncontact_geometry',
                      2:'selected_contact_region'}.get(int(scene.get('last_rejected_surface',-1)),'unknown')
                number=int(scene.get('last_rejected_part',-1))
                part=f'robot_arm_sphere_{number}' if 0<=number<37 else 'unknown'
                if number>=1000:
                    part=next((r['robot_geom_name'] for r in cover
                        if r['sphere_range'][0]==number-1000),'configured_gripper')
                if 2001<=number<=2007:
                    part=f'robot0_link{number-2000}_collision'
            roles[role]=roles.get(role,0)+1;parts[part]=parts.get(part,0)+1
        previous=scene
    return {'status':result.get('status'),'grasp_constraint':result.get('grasp_constraint','point'),
        'requested_relative_angle_deg':math.degrees(result.get('goal_angle_rad',0.)),
        'start_collision_free':result.get('start',{}).get('accepted'),
        'goal_proposals_tried':len(goals),'goal_projection_successes':len(projected),
        'collision_free_goals':result.get('admissible_goals',0),
        'projected_goals_rejected_by_collision':sum(not g['scene']['accepted'] for g in checked),
        'first_blocking_surface_by_rejected_goal':roles,
        'first_blocking_robot_part_by_rejected_goal':parts,
        'minimum_endpoint_position_error_m':min(positions,default=None),
        'minimum_orientation_or_jaw_residual':min(orientations,default=None),
        'candidate_rejection_counts':dict(reasons),
        'blocking_constraint_examples':examples,
        'last_attempted_continuation_angle_deg':math.degrees(result['last_progress_rad'])
            if 'last_progress_rad' in result else result.get('attempted_angle_deg'),
        'planning_reached_angle_deg':math.degrees(result['terminal_progress_rad'])
            if 'terminal_progress_rad' in result else None,
        'failure_explanation':result.get('failure'),
        'search_seconds':result.get('total_seconds'),
        'interpretation':'Finite search diagnostics, not a proof of infeasibility; no physical motion was issued.',
        'simulator_object_state_used':False}
