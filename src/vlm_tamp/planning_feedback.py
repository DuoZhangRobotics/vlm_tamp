"""Actor-visible diagnostics from the current numerical planning query."""
import math


def summarize_arc_failure(result,native_scene=None):
    goals=result.get('goal_proposals',[])
    projected=[g for g in goals if g.get('projection',{}).get('accepted')]
    checked=[g for g in projected if g.get('scene') is not None]
    positions=[g['projection']['position_error_m'] for g in goals
               if math.isfinite(g.get('projection',{}).get('position_error_m',float('inf')))]
    orientations=[g['projection']['jaw_inequality_error']/.15 for g in goals
                  if math.isfinite(g.get('projection',{}).get('jaw_inequality_error',float('inf')))]
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
        'last_attempted_continuation_angle_deg':math.degrees(result['last_progress_rad'])
            if 'last_progress_rad' in result else None,
        'search_seconds':result.get('total_seconds'),
        'interpretation':'Finite search diagnostics, not a proof of infeasibility; no physical motion was issued.',
        'simulator_object_state_used':False}
