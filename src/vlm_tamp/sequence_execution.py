"""Execute a grasp intent with fresh planning after physical closure."""
import copy
import time
import numpy as np

from .coupled_scene import RobotVolumes
from .execution_limits import remaining_controls
from .grasp_observation import grasp_observation
from .signed_contact_motion import plan_signed_arc
from .timed_execution import schedule,gripper_arm_reference


def measured_robot_scene(service):
    scene=copy.copy(service.scene)
    scene.robot=RobotVolumes(service.ik.chain,service.state['metadata']['robot_collision_volumes_world'],
        service.state['metadata']['joint_positions'])
    scene.config={**scene.config,'grasp_contact_depth_offset_m':getattr(service,'active_grasp_contact_offset_m',0.)}
    scene.skill_contact_surface='fingertips'
    return scene


def execute_sequence(service,plan_id,*,close_steps=40,speed=.35,approach_speed=None,hold=6,
                     manipulation_gripper=1,feedback_degrees=15.,calibrate_joint=True,open_steps=20):
    plan=service.plans[plan_id];extra=plan['extra']
    if plan['kind']!='sequence':raise ValueError(f"Plan {plan_id} is {plan['kind']}, not a sequence. Use execute for this plan; an approach requires gripper=-1, followed by closure at arrival. Use plan_sequence to request automatic staging.")
    if plan['control_steps']!=service.state['control_steps'] or plan['epoch']!=service.epoch:
        raise ValueError('Sequence approach belongs to an earlier state/model; request a fresh approach')
    approach_speed=float(extra.get('approach_speed',.6) if approach_speed is None else approach_speed)
    if (type(close_steps) is not int or close_steps<1 or
            not 0<float(speed)<=1 or not 0<approach_speed<=1 or int(hold)<0
            or type(open_steps) is not int or open_steps<1 or manipulation_gripper not in [0,1] or not np.isfinite(feedback_degrees) or feedback_degrees<=0):
        raise ValueError('Invalid sequence timing, feedback increment or gripper command')
    refs=schedule(plan['path'],maximum_joint_speed=approach_speed,endpoint_hold_steps=int(hold))
    opening_required=bool(extra.get('opening_required'))
    if (open_steps if opening_required else len(refs)+close_steps)>remaining_controls(service.state):
        raise ValueError('The approach and closure do not fit the remaining control budget')
    records=[];last={};started=time.perf_counter();commanded=0.;observed=0.;observed_complete=True
    service.active_grasp_contact_offset_m=float(extra.get('contact_depth_offset_m',0.))
    result={'plan_id':plan_id,'stages':records,'post_closure_replanning_enabled':True,'post_closure_replanning':False,
        'requested_degrees':extra['goal_degrees'],'commanded_degrees':0.,'observed_part_degrees':None,
        'approach_speed_rad_s':approach_speed,'manipulation_speed_rad_s':float(speed),
        'physical_grasp_verified':False,'episode_done':False,'task_success':False}

    def issue(references,name,gripper):
        nonlocal last
        last=service.issue(references,gripper,f'{plan_id}-{name}',articulated=name=='manipulation')
        records.append({'stage':name,'artifact_id':service.calls,**last})
        result.update(task_success=bool(last.get('task_success')),episode_done=bool(last.get('episode_done')),
            observation=last.get('observation'),grasp_evidence=last.get('grasp_evidence'))
        return result['episode_done'] or result['task_success']

    try:
        if opening_required:
            q,_=gripper_arm_reference(service.state['metadata']['joint_positions'],service.ik.chain.limits)
            if issue(np.repeat(q[None],open_steps,axis=0),'opening',-1):return result
            service.calls+=1
            options=extra.get('selection_options',{})
            from .hinged_contact_tools import grasp_timing_options
            try:
                selected=service._select_contact(extra['goal_degrees'],options.get('max_candidates',24),options.get('seed',0),
                    {**grasp_timing_options(approach_speed,int(hold),close_steps),
                     'maximum_grasp_search_seconds':float(options.get('budget_seconds',30)),
                     'environment_collision_policy':extra.get('collision_policy','veto')})
            except (ValueError,RuntimeError) as error:
                result.update(stop_reason='approach_replanning_after_opening_incomplete',planning_error=str(error),recovery_required=True)
                return result
            extra={**extra,'grasp_joints':selected['joints'],'contact_depth_offset_m':selected.get('contact_depth_offset_m',0.)}
            service.active_grasp_contact_offset_m=float(extra['contact_depth_offset_m'])
            refs=schedule(selected['approach'],maximum_joint_speed=approach_speed,endpoint_hold_steps=int(hold))
            result['approach_replanned_after_opening']=True
            service.calls+=1
            if len(refs)+close_steps>remaining_controls(service.state):
                result.update(stop_reason='approach_and_closure_exceed_remaining_controls',recovery_required=True);return result
        if issue(refs,'approach',-1):return result
        service.calls+=1
        # Continue holding the intended approach reference during closure; the
        # arm may still be settling. Replanning uses the measured end state.
        target=np.clip(np.asarray(extra['grasp_joints']),service.ik.chain.limits[:,0],service.ik.chain.limits[:,1])
        recovery={'performed':False,'reference':'planned approach endpoint; measured closure state used for the next plan'}
        if issue(np.repeat(target[None],close_steps,axis=0),'closure',1):return result
        result['closure_joint_limit_recovery']=recovery
        result['measured_closure_aperture_qpos']=service.state['metadata']['gripper_qpos']
        result['closure_evidence']=grasp_observation(service.state['metadata'],service.spec)
        if result['closure_evidence'].get('status')=='near_closed_without_bar_width_support':
            result.update(stop_reason='grasp_not_established',recovery_required=True);return result
        requested=float(extra['goal_degrees']);direction=np.sign(requested);ineffective=0;calibration_done=not calibrate_joint
        manipulation_count=0;calibration_pair=None
        feedback_records=[]
        while abs(commanded)<abs(requested)-1e-9:
            if service.episode_status()['episode_done']:
                result.update(stop_reason=service.episode_status()['termination_reason'],episode_done=True);break
            amount=direction*min(min(3.,float(feedback_degrees)) if calibrate_joint and not calibration_done and manipulation_count<3 else float(feedback_degrees),abs(requested-commanded))
            service.calls+=1;folder=service.folder('measured-grasp-plan');phase=time.perf_counter()
            result['post_closure_replanning']=True
            try:
                path,report=plan_signed_arc(service.ik,measured_robot_scene(service),service.record,
                    service.state['metadata'],np.asarray(service.state['metadata']['joint_positions']),amount,folder,
                    {**service.config,'grasp_constraint':extra['grasp_constraint'],
                        'environment_collision_policy':extra.get('collision_policy',service.config.get('environment_collision_policy','veto'))})
            except (ValueError,RuntimeError) as error:
                result.update(stop_reason='post_closure_planning_incomplete',planning_error=str(error),
                    physical_infeasibility_proved=False,recovery_required=True);break
            elapsed=time.perf_counter()-phase
            final=abs(commanded+amount)>=abs(requested)-1e-9
            refs=schedule(path,maximum_joint_speed=float(speed),endpoint_hold_steps=int(hold) if final else 0)
            if len(refs)>remaining_controls(service.state):
                result.update(stop_reason='next_complete_arc_exceeds_control_budget',required_controls=len(refs),
                    remaining_control_steps=remaining_controls(service.state),recovery_required=True);break
            service.calls+=1
            terminal=issue(refs,'manipulation',manipulation_gripper)
            records[-1].update(planned_degrees=amount,planning_seconds=elapsed,
                planning_report=report,robot_geometry_source='measured post-closure/current aperture')
            commanded+=amount;result['commanded_degrees']=commanded
            manipulation_count+=1
            measured=last.get('motion_effect',{}).get('observed_region_rotation_deg')
            if measured is None:observed_complete=False
            else:observed+=float(measured)
            result['observed_part_degrees']=observed if observed_complete else None
            if terminal:break
            if calibrate_joint and not calibration_done:
                available=[r for r in getattr(service,'motion_feedback',[]) if r['label']==f'{plan_id}-manipulation']
                if manipulation_count==2 and len(available)>=2 and available[0].get('tracker') is not None:
                    original_tracker=available[0]['tracker']
                    original_tracker.observe(service.cameras,control_step=service.state['control_steps'])
                    first_points,current_points=original_tracker.last_pairs
                    calibration_pair={'reference_observation_id':available[0]['reference_observation_id'],
                        'current_observation_id':available[-1]['current_observation_id'],
                        'first':np.asarray(first_points).copy(),'second':np.asarray(current_points).copy()}
                # A small checking residual over two tiny intervals can still
                # support a badly located hinge. Use the combined first-to-second
                # baseline, checked against a third observed interval, before
                # installing an automatic revision. This changes observation
                # conditioning; it adds no collision padding or motion veto.
                if manipulation_count>=3 and calibration_pair is not None and len(available)>=2:
                    from .joint_feedback import revise_from_feedback
                    pair=([calibration_pair,available[-1]] if calibration_pair is not None and
                        calibration_pair['current_observation_id']==available[-1]['reference_observation_id'] else available[-2:])
                    revision=revise_from_feedback(service,pair)
                    result.setdefault('joint_feedback',[]).append(revision)
                    calibration_done=bool(revision.get('motion_supported'))
            evidence=last.get('grasp_evidence',{})
            if evidence.get('status') in {'gripper_gap_collapsed_during_manipulation','near_closed_without_bar_width_support'}:
                result.update(stop_reason='grasp_engagement_lost',recovery_required=True);break
            stationary=last.get('motion_effect',{}).get('status')=='selected_region_stationary_despite_hand_motion'
            ineffective=ineffective+1 if stationary else 0
            if ineffective>=2:
                result.update(stop_reason='selected_part_not_following',recovery_required=True);break
        result.setdefault('stop_reason','native_task_success' if result['task_success'] else 'requested_reference_motion_complete')
    finally:
        result['total_seconds']=time.perf_counter()-started
        result.setdefault('stop_reason','native_task_success' if result.get('task_success') else
            last.get('termination_reason','episode_terminated') if result.get('episode_done') else 'partial_execution_returned')
        if result.get('recovery_required'):
            result['recovery_options']=['observe current part and contact','push from current visible surface',
                'plan_grasp with a local low-cost approach','refresh_contact while retaining fixed joint parameters',
                'after re-establishing engagement, consider a shorter feedback_degrees interval when the model drifts']
            result['remaining_requested_degrees']=float(extra['goal_degrees'])-commanded
        result['remaining_control_steps']=remaining_controls(service.state)
    return result
