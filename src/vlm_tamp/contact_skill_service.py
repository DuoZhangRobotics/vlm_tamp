"""Bounded physical contact skills integrated with the public robot service."""
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .contact_skills import (Budget,SkillStop,ContactTracker,build_contact_scene,
    cartesian_path,plan_engagement,plan_contact_stroke,measured_contact_speed,refresh_collision_observations,
    fresh_collision_snapshot,skill_native_scene,current_collision_witness,predicted_scene,unit)
from .coupled_storage import write_json,read_json
from .rekep_libero.geometry import pose_matrix,matrix_pose,transform_points
from .timed_execution import schedule_contact
from .execution_limits import control_limit

ROOT=Path(__file__).resolve().parents[2]


def qualification_sources():
    paths=['src/vlm_tamp/contact_skills.py','src/vlm_tamp/contact_skill_service.py',
        'src/vlm_tamp/timed_execution.py','src/vlm_tamp/mcvamp_contact_numerical.py',
        'src/vlm_tamp/native/mcvamp_contact_numerical.cc','src/vlm_tamp/rigid_hinge_tracking.py',
        'src/vlm_tamp/profiled_sim_worker.py','src/vlm_tamp/profiled_sim_client.py',
        'src/vlm_tamp/rekep_libero/sim_worker_v7.py','src/vlm_tamp/coupled_worker.py',
        'src/vlm_tamp/coupled_scene.py','scripts/serve_subscription_robot.py']
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def momentum_status():
    path=Path(os.environ.get('VLM_TAMP_MOMENTUM_QUALIFICATION',
        str(ROOT/'results/contact-skills-20261028/momentum-qualification.json')))
    report=read_json(path) if path.is_file() else {}
    current=bool(report and report.get('sources_sha256')==qualification_sources())
    okay=bool(report.get('execution_qualified') and current)
    reason=('qualification_missing' if not report else
        'recorded_execution_not_qualified' if not report.get('execution_qualified') else
        'qualification_stale' if not current else 'qualified_simulation_execution')
    return {'available':True,'validated':okay,'reason':'authorized_simulation_experiment',
        'validation_reason':reason,'qualification_sources_current':current,
        'supported_speeds_m_s':[.03,.06,.12],
        'execution_scope':'simulation contact experiments; physical contacts remain active',
        'task_success_guaranteed':False}


def _overlay(service,camera,region,target_region,point,direction,dest):
    view=service.cameras[camera];img=Image.fromarray(view['rgb']).copy();draw=ImageDraw.Draw(img)
    draw.rectangle(region,outline='#ffcc00',width=3)
    if target_region:draw.rectangle(target_region,outline='#00ffff',width=2)
    t=np.linalg.inv(np.asarray(view['camera_to_world']));k=np.asarray(view['intrinsic'])
    p=transform_points(np.asarray([point,np.asarray(point)+.05*np.asarray(direction)]),t)
    if np.all(p[:,2]>0):
        xy=p@k.T;xy=xy[:,:2]/xy[:,2,None]
        draw.line([tuple(x) for x in xy],fill='#ff5555',width=4)
        x,y=xy[0];draw.ellipse((x-5,y-5,x+5,y+5),fill='#ffff00')
    img.save(dest)


def feedback_camera_names(tracker,requested,available,previous=()):
    """Choose current reference views; never reuse the original ranking blindly."""
    references=tracker.tracker.references
    ranked=sorted((name for name in references if name in available),
        key=lambda name:-len(references[name]['pixels']))
    names=list(dict.fromkeys(([requested] if requested in available else [])+ranked))[:4]
    # Keep ordering stable when the captured set itself has not changed.
    return list(previous) if set(names)==set(previous) else names


def run_skill(service,kind,*,observation_id,camera,region_xyxy,target_region_xyxy=None,
              direction_world=None,distance_m=.03,joint_degrees=None,joint_model_epoch=None,
              mode='controlled',contact_speed_m_s=.03,contact_surface='fingertips',
              preview=False,budget_seconds=120.,collision_policy='veto',approach_speed_rad_s=.6,
              stroke_length_m=None,search_seconds=None,stroke_plan_seconds=None,max_ineffective_strokes=2,withdrawal_distance_m=.015):
    started=time.perf_counter();timing=Counter();stages=[];executed=False
    if service.sim is None:raise ValueError('Reset before using contact skills')
    if observation_id!=service.latest_observation['observation_id']:
        raise ValueError('Use the latest observation_id; contact requests cannot reuse stale pixels')
    if service.latest_observation['control_steps']!=service.state['control_steps']:
        raise ValueError('Capture the current physical state before selecting contact')
    initial_control_step=int(service.state['control_steps'])
    control_steps_limit=control_limit(service.state,getattr(service,'config',None))
    if mode not in {'controlled','push_and_release'} or kind not in {'push','hook_pull'}:raise ValueError('Unknown contact skill/mode')
    if collision_policy not in {'advisory','veto'}:raise ValueError('collision_policy must be advisory or veto')
    if kind=='hook_pull' and (mode!='controlled' or contact_surface!='fingertips'):
        raise ValueError('Hook-pull uses fingertip engagement and controlled strokes')
    if contact_surface not in {'fingertips','palm'}:raise ValueError('Choose fingertips or explicit palm contact')
    if not np.isfinite(contact_speed_m_s) or contact_speed_m_s<=0:raise ValueError('Contact speed must be finite and positive')
    if not np.isfinite(approach_speed_rad_s) or not 0<approach_speed_rad_s<=1:raise ValueError('Approach reference speed must be in (0,1] rad/s')
    if not np.isfinite(distance_m) or distance_m<=0:raise ValueError('Distance must be finite and positive')
    if not np.isfinite(budget_seconds) or not 0<budget_seconds<=1800:raise ValueError('Budget must be (0,1800] seconds')
    stroke_length_m=float((.03 if mode=='push_and_release' else .01) if stroke_length_m is None else stroke_length_m)
    if not np.isfinite(stroke_length_m) or stroke_length_m<=0:raise ValueError('Stroke length must be finite and positive')
    search_seconds=float(budget_seconds/4 if search_seconds is None else search_seconds)
    stroke_plan_seconds=float(budget_seconds/24 if stroke_plan_seconds is None else stroke_plan_seconds)
    if not np.isfinite([search_seconds,stroke_plan_seconds]).all() or min(search_seconds,stroke_plan_seconds)<=0:raise ValueError('Planning budgets must be positive')
    if type(max_ineffective_strokes) is not int or max_ineffective_strokes<1:raise ValueError('Use a positive ineffective-stroke count')
    if not np.isfinite(withdrawal_distance_m) or withdrawal_distance_m<=0:raise ValueError('Withdrawal distance must be positive')
    if joint_degrees is not None:
        if direction_world is not None:raise ValueError('Choose direction_world or joint_degrees, not both')
        if not np.isfinite(joint_degrees) or joint_degrees==0:raise ValueError('Joint increment must be finite and nonzero')
        if joint_model_epoch!=service.epoch:raise ValueError('joint_model_epoch must identify the current explicitly associated model')
        if service.scene is None or service.contact_scene_kind!='articulated':raise ValueError('An active articulated model is required')
        if service.scene_reference_control_step!=service.state['control_steps']:raise ValueError('Update the articulated scene to the current observation before requesting joint motion')
    remaining=service.episode_status().get('remaining_wall_seconds')
    budget=Budget(min(budget_seconds,remaining) if remaining is not None else budget_seconds)
    root=service.public/f'attempt{service.attempt:02d}'/f'skill{service.calls:03d}'
    root.mkdir(parents=True,exist_ok=False)
    result={'skill':kind,'mode':mode,'requested_contact_speed_m_s':contact_speed_m_s,
        'requested_distance_m':distance_m if joint_degrees is None else None,
        'requested_joint_degrees':joint_degrees,'preview':preview,
        'motion_settings':{'approach_speed_rad_s':approach_speed_rad_s,'stroke_length_m':stroke_length_m,
            'initial_search_seconds':search_seconds,'subsequent_plan_seconds':stroke_plan_seconds,'max_ineffective_strokes':max_ineffective_strokes,'withdrawal_distance_m':withdrawal_distance_m},
        'task_success':False,'robot_motion_executed':False,'stages':stages,
        'momentum':momentum_status(),'physical_contact_verified':False,
        'feedback_source':'selected RGB-D views and robot proprioception',
        'simulator_object_geometry_used':False,'fixed_joint_preserved':True}
    result['collision_policy']=collision_policy
    result['environment_collision_predictions']={'count':0,'examples':[],
        'all_environment_states_checked':False,'collision_free_certificate':False}
    result['simulator_physical_contacts_enabled']=True
    try:
        budget.check()
        phase=time.perf_counter()
        scene,evidence,selected=build_contact_scene(service.cameras,service.state['metadata'],service.ik.chain,
            camera,region_xyxy,target_region_xyxy,contact_surface,direction_world,joint_degrees,service.scene,skill_kind=kind)
        if kind=='hook_pull' and direction_world is None and joint_degrees is None:
            evidence['direction_world']=list(evidence['normal_toward_camera_world'])
        scene.skill_collision_policy=collision_policy
        scene.skill_approach_speed_rad_s=approach_speed_rad_s
        scene.skill_approach_hold_steps=0
        scene.skill_contact_collision_log=result['environment_collision_predictions']
        result['cartesian_path_deviation']={'diagnostic_threshold_m':.006,'maximum_m':0.,'samples_over_threshold':0,
            'execution_veto_applied':collision_policy=='veto'}
        scene.skill_path_deviation_log=result['cartesian_path_deviation']
        timing['perception_seconds']+=time.perf_counter()-phase
        result['contact']=evidence;direction=unit(evidence['direction_world'])
        def following_controls(contact_point):
            if mode!='push_and_release':return 1
            length=min(stroke_length_m,distance_m)
            if joint_degrees is not None:
                radius=np.linalg.norm(np.cross(scene.axis,np.asarray(contact_point)-scene.origin))
                angle=min(abs(np.radians(joint_degrees)),stroke_length_m/max(radius,.01))
                length=2*radius*np.sin(angle/2)
            # Commanded stages: a 1 cm slow contact test, the requested pulse,
            # 15 mm withdrawal, and one second of free coasting.
            return (7 if collision_policy=='veto' else 0)+max(1,int(np.ceil(length/contact_speed_m_s*20)))+2+int(np.ceil(withdrawal_distance_m/.03*20))+20
        _overlay(service,camera,region_xyxy,target_region_xyxy,evidence['point_world_m'],direction,root/'contact.png')
        result['images']=[str((root/'contact.png').relative_to(service.workspace))]
        phase=time.perf_counter()
        search_budget=budget.child(search_seconds)
        try:engagement=plan_engagement(service.ik,scene,service.record,service.state['metadata'],evidence,selected,kind,search_budget,
            maximum_approach_controls=lambda p:max(0,control_steps_limit-service.state['control_steps']-following_controls(p)))
        except SkillStop as error:
            timing['contact_search_seconds']+=time.perf_counter()-phase
            if error.reason=='budget_exhausted':raise SkillStop('contact_search_budget_exhausted',**error.evidence)
            raise
        timing['contact_search_seconds']+=time.perf_counter()-phase
        if collision_policy=='advisory' and budget.remaining>2.:
            phase=time.perf_counter()
            try:
                with skill_native_scene(scene,service.record,service.state['metadata'],service.state['metadata']['joint_positions']) as checker:
                    forecast=checker.validate(engagement['path'][-1])
                if not forecast['accepted']:
                    log=result['environment_collision_predictions'];log['count']+=1
                    log['examples'].append({'stage':'planned_contact_pose',
                        'blocking_robot_part':forecast.get('blocking_robot_part'),
                        'blocking_surface':forecast.get('blocking_surface'),'execution_veto_applied':False})
            except Exception as error:
                result['environment_collision_predictions'].setdefault('diagnostic_errors',[]).append(str(error))
            timing['advisory_collision_prediction_seconds']+=time.perf_counter()-phase
        result['search']={k:v for k,v in engagement.items() if k not in {'path','references','free_references','insertion_references','rotation','contact_local','contact_world','insertion_direction'}}
        result['search']['ik_seconds_nested']=engagement['timing_seconds'].get('ik_seconds',0.)
        result['contact'].update(robot_contact_surface=engagement['contact_name'],
            contact_world_m=engagement['contact_world'].tolist(),contact_local_m=engagement['contact_local'].tolist(),
            insertion_direction_world=engagement['insertion_direction'].tolist(),
            gripper_aperture_qpos=service.state['metadata']['gripper_qpos'])
        direction=unit(evidence['direction_world'])
        _overlay(service,camera,region_xyxy,target_region_xyxy,engagement['contact_world'],direction,root/'contact.png')
        result['estimated_approach_control_steps']=len(engagement['references'])
        result['robot_valid_plan_available']=True
        result['estimated_approach_simulation_seconds']=len(engagement['references'])/20
        np.save(root/'approach-path.npy',engagement['path'])
        np.save(root/'approach-references.npy',engagement['references'])
        result['preview_artifacts']={'joint_path':str((root/'approach-path.npy').relative_to(service.workspace)),
            'timed_references':str((root/'approach-references.npy').relative_to(service.workspace))}
        next_controls=following_controls(engagement['contact_world'])
        result['minimum_following_control_steps']=next_controls
        result['control_budget_feasible']=len(engagement['references'])+next_controls<=control_steps_limit-service.state['control_steps']
        result['budget_feasibility_scope']='approach plus initial bounded contact/release procedure; requested object progress is uncertain'
        nominal_distance=(distance_m if joint_degrees is None else abs(np.radians(joint_degrees))*np.linalg.norm(np.cross(scene.axis,engagement['contact_world']-scene.origin)))
        result['nominal_requested_motion_controls']=int(np.ceil(nominal_distance/contact_speed_m_s*20))
        result['nominal_cost_assumption']='one-to-one part/contact travel, before recovery or controller lag; not an admission veto'
        if preview:
            result.update(status='preview_ready' if result['control_budget_feasible'] else 'preview_budget_limited',
                stop_reason='preview_only',predicted_contact_only=True)
            return result
        budget.check()
        if len(engagement['references'])+next_controls>control_steps_limit-service.state['control_steps']:
            raise SkillStop('insufficient_control_budget',approach_steps=len(engagement['references']),
                remaining=control_steps_limit-service.state['control_steps'],minimum_following_controls=next_controls)
        contact_local=engagement['contact_local'];point=engagement['contact_world'].copy()
        original_point=point.copy();cumulative=np.eye(4);progress=0.;ineffective=0;unwrapped_angle=0.
        tracked_target=scene.moving_reference.copy()
        phase=time.perf_counter()
        tracker=ContactTracker(service.cameras,tracked_target,service.state['control_steps'])
        timing['tracker_initialization_seconds']+=time.perf_counter()-phase
        view_names=[]
        def refresh_feedback_views(reference_tracker):
            nonlocal view_names
            selected_views=feedback_camera_names(reference_tracker,camera,service.cameras,view_names)
            if selected_views!=view_names:
                result.setdefault('feedback_camera_changes',[]).append({
                    'control_steps':int(service.state['control_steps']),'cameras':selected_views})
            view_names=selected_views;result['feedback_cameras']=list(view_names)
        refresh_feedback_views(tracker)

        def issue(refs,label,**stage_fields):
            nonlocal executed
            budget.check()
            if len(refs)>control_steps_limit-service.state['control_steps']:raise SkillStop('control_budget_exhausted',required_steps=len(refs))
            service.calls+=1;before=len(service.events);q=service.state['metadata']['joint_positions']
            phase=time.perf_counter()
            reply=service.issue(refs,0,label,camera_names=view_names,track_scene=False)
            executed=True;timing['execution_and_capture_seconds']+=time.perf_counter()-phase
            events=service.events[before:]
            phase=time.perf_counter()
            stage={'stage':label,'artifact_id':service.calls,'control_steps':service.state['control_steps'],
                'execution':reply['execution'],'measured_contact_speed':measured_contact_speed(events,service.ik.chain if service.ik else chain,contact_local,q)}
            stage.update(stage_fields)
            issued=int(reply['execution']['issued_commands'])
            desired=pose_matrix(chain.forward(refs[issued-1] if issued else q))
            actual=pose_matrix(chain.forward(service.state['metadata']['joint_positions']))
            stage['robot_reference_tracking']={'position_error_m':float(np.linalg.norm(actual[:3,3]-desired[:3,3])),
                'orientation_error_deg':float(np.degrees(Rotation.from_matrix(desired[:3,:3].T@actual[:3,:3]).magnitude())),
                'reference':'last issued robot command; not an object-motion or grasp certificate'}
            timing['speed_measurement_seconds']+=time.perf_counter()-phase
            stages.append(stage);result['observation']=reply['observation']
            result['episode_done']=bool(reply.get('episode_done'))
            if reply.get('task_success'):
                result.update(task_success=True,status='task_success',stop_reason='native_task_success')
                raise SkillStop('native_task_success')
            if reply.get('episode_done'):raise SkillStop(reply.get('termination_reason','episode_budget_exhausted'))
            budget.check()
            return stage

        recovered_steps=set()
        def observe_motion(tracker):
            phase=time.perf_counter();fit=tracker.observe(service.cameras,service.state['control_steps'])
            timing['motion_tracking_seconds']+=time.perf_counter()-phase
            step=int(service.state['control_steps'])
            if not fit['accepted'] and len(service.cameras)<=4 and step not in recovered_steps and budget.remaining>0:
                recovered_steps.add(step);budget.check();service.calls+=1
                phase=time.perf_counter()
                try:result['observation']=service.capture()
                except Exception as error:
                    timing['additional_view_recovery_seconds']+=time.perf_counter()-phase
                    result.setdefault('perception_recoveries',[]).append({'control_steps':step,
                        'recovered':False,'error_type':type(error).__name__,'error':str(error)})
                    return fit
                timing['additional_view_recovery_seconds']+=time.perf_counter()-phase
                phase=time.perf_counter();retried=tracker.observe(service.cameras,step)
                timing['motion_tracking_seconds']+=time.perf_counter()-phase
                result.setdefault('perception_recoveries',[]).append({'control_steps':step,
                    'original_reason':fit.get('reason'),'original_pairs':fit.get('pairs'),
                    'recovered':bool(retried['accepted']),'retry_pairs':retried.get('pairs'),
                    'current_observation_id':result['observation']['observation_id']})
                fit=retried
            return fit

        chain=service.ik.chain  # retained FK after native success closes the IK process
        measured=None
        if not engagement.get('reuse_existing_contact'):
            issue(engagement['free_references'],'free-approach')
            measured=observe_motion(tracker)
            stages[-1]['object_motion']=measured
            if not measured['accepted']:raise SkillStop('object_motion_unresolved',measurement=measured)
            approach_transform=np.asarray(measured['transform'])
            approach_object_displacement=np.linalg.norm(transform_points([point],approach_transform)[0]-point)
            stages[-1]['uncommanded_object_motion_m']=float(approach_object_displacement)
            if approach_object_displacement>.08:
                if collision_policy=='veto':
                    raise SkillStop('target_moved_beyond_local_recovery',observed_displacement_m=float(approach_object_displacement),maximum_m=.08)
                result.setdefault('motion_warnings',[]).append({'reason':'large_measured_motion_during_approach',
                    'observed_displacement_m':float(approach_object_displacement),'pose_update_source':'accepted independent RGB-D fit'})

        def update_motion(fit):
            nonlocal scene,point,cumulative,tracked_target,direction,progress,unwrapped_angle
            transform=np.asarray(fit['transform']);r=transform[:3,:3];t=transform[:3,3]
            if joint_degrees is not None:
                drift=float(np.linalg.norm(r@scene.origin+t-scene.origin))
                if drift>.006:raise SkillStop('motion_inconsistent_with_fixed_joint',fixed_hinge_displacement_m=drift,maximum_m=.006)
            old=point.copy();point=r@point+t;cumulative=transform@cumulative
            scene=copy.copy(scene)
            scene.moving_reference=transform_points(scene.moving_reference,transform)
            scene.contact_reference=transform_points(scene.contact_reference,transform)
            phase=time.perf_counter()
            scene,occupancy=refresh_collision_observations(scene,service.cameras,service.state['metadata'])
            timing['occupancy_update_seconds']+=time.perf_counter()-phase
            stages[-1]['occupancy_update']=occupancy
            tracked_target=scene.moving_reference
            if joint_degrees is None:
                progress=float((point-original_point)@direction)
            else:
                unwrapped_angle+=float(Rotation.from_matrix(r).as_rotvec()@scene.axis)
                progress=float(unwrapped_angle*np.sign(joint_degrees))
                direction=unit(np.sign(joint_degrees)*np.cross(scene.axis,point-scene.origin))
            result['progress_measured_at_control_step']=int(service.state['control_steps'])
            return float((point-old)@direction)

        if measured is not None:update_motion(measured)
        # Replan the short insertion from the actual servo endpoint. A nominal
        # reference endpoint does not establish physical arrival at standoff.
        if not engagement.get('reuse_existing_contact'):
            phase=time.perf_counter();tracker=ContactTracker(service.cameras,tracked_target,service.state['control_steps'])
            refresh_feedback_views(tracker)
            timing['tracker_initialization_seconds']+=time.perf_counter()-phase
            meta=service.state['metadata'];start=np.asarray(meta['joint_positions'])
            insertion_goal=np.eye(4);insertion_goal[:3,:3]=engagement['rotation']
            insertion_goal[:3,3]=point-engagement['rotation']@contact_local
            nested=Counter();phase=time.perf_counter()
            insertion=cartesian_path(service.ik,scene,service.record,meta,start,matrix_pose(insertion_goal),budget.child(stroke_plan_seconds),timing=nested)
            timing['stroke_planning_seconds']+=time.perf_counter()-phase
            stage=issue(schedule_contact(insertion,chain,contact_local,contact_speed_m_s=.03,endpoint_hold_steps=3),'contact-insertion',
                requested_contact_speed_m_s=.03,planning_timing_seconds=dict(nested))
            measured=observe_motion(tracker);stage['object_motion']=measured
            if not measured['accepted']:raise SkillStop('object_motion_unresolved',measurement=measured)
            update_motion(measured)
        target=distance_m if joint_degrees is None else abs(np.radians(joint_degrees))
        scene.skill_collision_policy=collision_policy
        scene.skill_contact_collision_log=result['environment_collision_predictions']
        release_contact_ready=collision_policy=='advisory'
        stroke=-1
        while True:
            stroke+=1
            budget.check()
            if mode=='controlled' and progress>=target-(.001 if joint_degrees is None else np.radians(.5)):
                result.update(status='requested_progress_reached',stop_reason='requested_progress_reached');break
            phase=time.perf_counter()
            with skill_native_scene(scene,service.record,service.state['metadata'],service.state['metadata']['joint_positions']) as check:
                start_check=check.validate(service.state['metadata']['joint_positions'])
            timing['current_state_check_seconds']+=time.perf_counter()-phase
            if not start_check['accepted'] and collision_policy=='advisory':
                log=result['environment_collision_predictions'];log['count']+=1
                if len(log['examples'])<12:log['examples'].append({'stage':'current_pose',
                    'blocking_robot_part':start_check.get('blocking_robot_part'),
                    'blocking_surface':start_check.get('blocking_surface'),'execution_veto_applied':False})
            if not start_check['accepted'] and start_check.get('static_rejections',0)>0 and collision_policy=='veto':
                start_check['overlap_evidence']=current_collision_witness(scene,service.state['metadata'],start_check)
                budget.check();service.calls+=1;phase=time.perf_counter()
                result['observation']=service.capture()
                scene,refresh=fresh_collision_snapshot(scene,service.cameras,service.state['metadata'])
                tracked_target=scene.moving_reference
                with skill_native_scene(scene,service.record,service.state['metadata'],service.state['metadata']['joint_positions']) as check:
                    after=check.validate(service.state['metadata']['joint_positions'])
                if not after['accepted']:after['overlap_evidence']=current_collision_witness(scene,service.state['metadata'],after)
                timing['full_observation_recovery_seconds']+=time.perf_counter()-phase
                result.setdefault('observation_refreshes',[]).append({'artifact_id':service.calls,
                    'control_steps':service.state['control_steps'],'before':start_check,'after':after,'evidence':refresh})
                budget.check()
                if not after['accepted']:raise SkillStop('current_observation_start_collision',diagnostic=after,
                    explanation='A full current sensor snapshot still rejects the measured robot configuration; no obstacle was bypassed.')
            phase=time.perf_counter();tracker=ContactTracker(service.cameras,tracked_target,service.state['control_steps'])
            refresh_feedback_views(tracker)
            timing['tracker_initialization_seconds']+=time.perf_counter()-phase
            meta=service.state['metadata'];start=np.asarray(meta['joint_positions']);tcp=pose_matrix(chain.forward(start))
            hand_point=tcp[:3,3]+tcp[:3,:3]@contact_local
            # Detect lost engagement geometrically, without assuming a pinch.
            separation=float(np.linalg.norm(hand_point-point))
            if separation>.025:
                if collision_policy=='veto':raise SkillStop('contact_geometry_lost',landmark_separation_m=separation,maximum_m=.025)
                result.setdefault('contact_warnings',[]).append({'reason':'material_contact_point_prediction_changed',
                    'landmark_separation_m':separation,'physical_contact_loss_certified':False,
                    'interpretation':'Pushing may slide to another surface point; a bounded simulation stroke is permitted and measured independently.'})
            acquiring_release_contact=mode=='push_and_release' and not release_contact_ready
            stroke_speed=.03 if acquiring_release_contact else contact_speed_m_s
            length=(.01 if acquiring_release_contact else
                min(stroke_length_m,distance_m) if mode=='push_and_release' else
                min(stroke_length_m,distance_m-progress if joint_degrees is None else stroke_length_m))
            length=min(length,(control_steps_limit-service.state['control_steps'])*stroke_speed/20.)
            if length<=0:raise SkillStop('control_budget_exhausted')
            if kind=='hook_pull' and stroke==0:length=min(length,.005)
            angle=0.;delta=length*direction
            if joint_degrees is not None:
                radius=np.linalg.norm(np.cross(scene.axis,point-scene.origin))
                angle=np.sign(joint_degrees)*min(target if mode=='push_and_release' else target-progress,length/max(radius,.01),np.pi/2)
                rotation=Rotation.from_rotvec(scene.axis*angle).as_matrix()
                delta=scene.origin+rotation@(point-scene.origin)-point
            goal=tcp.copy();goal[:3,3]+=delta
            scene.skill_contact_collision_log['stage']='contact_test' if acquiring_release_contact else 'contact_stroke'
            planning=budget.child(stroke_plan_seconds);nested=Counter();phase=time.perf_counter()
            try:path,search=plan_contact_stroke(service.ik,scene,service.record,meta,contact_local,delta,planning,angle=angle,timing=nested)
            except SkillStop as error:
                timing['stroke_planning_seconds']+=time.perf_counter()-phase
                if error.reason=='budget_exhausted':raise SkillStop('stroke_planning_budget_exhausted',search_complete=False,timing_seconds=dict(nested))
                raise
            timing['stroke_planning_seconds']+=time.perf_counter()-phase
            if collision_policy=='advisory' and budget.remaining>2.:
                phase=time.perf_counter();log=scene.skill_contact_collision_log
                try:
                    predicted=predicted_scene(scene,delta,angle)
                    with skill_native_scene(predicted,service.record,meta,start) as checker:
                        prediction=checker.validate(path[-1])
                    if not prediction['accepted']:
                        log['count']+=1
                        if len(log['examples'])<12:log['examples'].append({'stage':log['stage'],
                            'blocking_robot_part':prediction.get('blocking_robot_part'),
                            'blocking_surface':prediction.get('blocking_surface'),'execution_veto_applied':False})
                except Exception as error:
                    log.setdefault('diagnostic_errors',[]).append(str(error))
                timing['advisory_collision_prediction_seconds']+=time.perf_counter()-phase
            refs=schedule_contact(path,chain,contact_local,contact_speed_m_s=stroke_speed,
                endpoint_hold_steps=2 if mode=='push_and_release' and not acquiring_release_contact else 0)
            stage=issue(refs,'push-contact-test' if acquiring_release_contact else
                'hook-engagement-test' if kind=='hook_pull' and stroke==0 else f'{kind}-stroke-{stroke+1}',
                requested_contact_speed_m_s=stroke_speed,posture_search=search,
                planning_timing_seconds=dict(nested),requested_motion_m=float(np.linalg.norm(delta)))
            fit=observe_motion(tracker);stage['object_motion']=fit
            finish_release=mode=='push_and_release' and collision_policy=='advisory' and not acquiring_release_contact
            motion_applied=False;moved=None
            if fit['accepted']:
                try:moved=update_motion(fit);motion_applied=True
                except SkillStop as error:
                    if not finish_release:raise
                    result.setdefault('motion_warnings',[]).append({'reason':error.reason,**error.evidence})
            elif not finish_release:raise SkillStop('object_motion_unresolved',measurement=fit)
            else:result.setdefault('motion_warnings',[]).append({'reason':'object_motion_unresolved','measurement':fit})
            stage['observed_contact_progress_m']=moved
            stage['engagement']={'status':'object_motion_unresolved' if moved is None else
                'object_followed_contact' if moved>.001 else 'no_supported_object_progress',
                'pinch_assumed':False,'force_or_tactile_measurement':False}
            result['observed_progress']=progress if motion_applied else None
            result['progress_units']='radians' if joint_degrees is not None else 'metres'
            if kind=='hook_pull' and stroke==0 and moved<=.001:raise SkillStop('hook_engagement_not_established',measured_progress_m=moved)
            if acquiring_release_contact:
                if moved<=.001:raise SkillStop('push_contact_not_established',measured_progress_m=moved,
                    explanation='The slow contact test did not establish observed part motion; no faster pulse was issued.')
                release_contact_ready=True
                continue
            ineffective=ineffective+1 if moved is not None and moved<=.001 else 0
            if ineffective>=max_ineffective_strokes:raise SkillStop('no_progress_after_stroke_budget',measured_progress_m=moved,consecutive_ineffective_strokes=ineffective)
            if mode=='push_and_release':
                scene.skill_contact_collision_log['stage']='withdrawal'
                phase=time.perf_counter()
                # If the pulse pose fit was unresolved, keep the original
                # target correspondences. Do not seed a tracker on a stale
                # target cloud projected into the post-pulse image.
                release_tracker=(ContactTracker(service.cameras,tracked_target,service.state['control_steps'])
                    if motion_applied else tracker)
                refresh_feedback_views(release_tracker)
                result['release_motion_reference']='after_pulse' if motion_applied else 'before_pulse'
                timing['tracker_initialization_seconds']+=time.perf_counter()-phase
                meta=service.state['metadata'];start=np.asarray(meta['joint_positions']);tcp=pose_matrix(chain.forward(start))
                retreat=tcp.copy();retreat[:3,3]-=withdrawal_distance_m*direction
                planning=budget.child(stroke_plan_seconds);nested=Counter();phase=time.perf_counter()
                path=cartesian_path(service.ik,scene,service.record,meta,start,matrix_pose(retreat),planning,timing=nested)
                timing['stroke_planning_seconds']+=time.perf_counter()-phase
                refs=schedule_contact(path,chain,contact_local,contact_speed_m_s=.03,endpoint_hold_steps=0)
                stage=issue(refs,'push-release-withdrawal',planning_timing_seconds=dict(nested),requested_contact_speed_m_s=.03)
                result['release_evidence']={'requested_withdrawal_m':withdrawal_distance_m,
                    'measured_tcp_travel_m':float(np.linalg.norm(pose_matrix(chain.forward(service.state['metadata']['joint_positions']))[:3,3]-tcp[:3,3])),
                    'physical_disengagement_verified':False,
                    'interpretation':'Completed withdrawal is measured robot motion; inspect current RGB-D for remaining contact. A longer withdrawal can be requested.'}
                # The simulator advances only on env.step. A capture or model
                # reasoning pause does not let an unheld door coast.
                hold=np.repeat(np.asarray(service.state['metadata']['joint_positions'])[None],20,axis=0)
                stage=issue(hold,'push-release-coasting')
                fit=observe_motion(release_tracker);stage['object_motion']=fit
                if fit['accepted']:
                    try:
                        stage['observed_contact_progress_m']=update_motion(fit)
                        result['observed_progress']=progress
                    except SkillStop as error:
                        if collision_policy!='advisory':raise
                        result.setdefault('motion_warnings',[]).append({'reason':error.reason,**error.evidence})
                        result['observed_progress']=None
                result['release_motion_observed']=bool(fit['accepted'])
                result.update(status='push_and_release_complete',stop_reason='bounded_release_stroke_complete',
                    final_object_displacement_prediction='unknown; observe coasting and possible rebound')
                break
    except SkillStop as error:
        result.update(stop_reason=error.reason,diagnostic=error.evidence)
        if error.reason!='native_task_success':result['status']='returned_without_task_success'
        if error.reason in {'object_motion_unresolved','motion_inconsistent_with_fixed_joint','no_progress_after_stroke_budget','hook_engagement_not_established'}:
            result['recovery_options']=['select an opaque currently observed region','use a current surface push without a hinge assumption','apply an independently measured part/joint update']
            result['physical_infeasibility_proved']=False
    finally:
        # A contact skill does not silently install its hypothetical target
        # motion into the grasp pipeline. Old geometry must be refreshed.
        executed=executed or int(service.state['control_steps'])!=initial_control_step
        if executed and service.sim is not None:
            from .observed_joint_scene import retain_observed_articulation
            retain_observed_articulation(service)
            service.scene=None;service.spec=None;service.contact_scene_kind='surface'
            service.surface_reference_step=-1;service.tracker=None
            service.scene_reference_control_step=None
            service.epoch+=1
        result['scene_refresh_required']=bool(executed and service.sim is not None)
        if result['scene_refresh_required']:
            retained=service.retained_articulated_model or {}
            if retained.get('generic_part_joint_model'):
                result['next_geometry_interfaces']=['segment_parts','apply_joint_estimate with current_part_id and reuse_fixed_joint=true','select_surface_contact']
                result['retained_joint_estimate_id']=retained.get('joint_estimate_id')
            else:result['next_geometry_interfaces']=['refresh_contact' if retained else 'ground','select_surface_contact']
            history=getattr(service,'retained_articulated_scene',None)
            if history is not None:
                result['retained_scene_reference_control_step']=history['reference_control_step']
                result['next_geometry_interfaces'].insert(0,'apply_part_motion with matching reference and current observed tracks')
        result['robot_motion_executed']=executed
        result['remaining_skill_seconds']=budget.remaining
        if joint_degrees is not None and result.get('observed_progress') is not None:
            result['observed_joint_change_degrees']=float(np.degrees(result['observed_progress'])*np.sign(joint_degrees))
            result['observed_joint_change_source']='RGB-D part pose projected onto the installed signed axis; not native joint state'
        result['feedback_count']=sum('object_motion' in s for s in stages)
        result['timing_seconds']={'total':time.perf_counter()-started,**dict(timing)}
        result['timing_seconds']['other_seconds']=max(0.,result['timing_seconds']['total']-sum(timing.values()))
        result['report']=str((root/'result.json').relative_to(service.workspace))
        write_json(root/'result.json',result)
        write_json(service.episode/f'{root.name}-result.json',result)
    return result
