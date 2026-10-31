"""Bounded physical contact skills integrated with the public robot service."""
import copy
import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .contact_skills import (Budget,SkillStop,ContactTracker,build_contact_scene,
    cartesian_path,plan_engagement,plan_contact_stroke,measured_contact_speed,advance_contact_scene,
    skill_native_scene,predicted_scene,unit)
from .push_contact_state import (observed_contact_cloud,preview_key,active_contact,
    contact_summary,resumed_engagement,copied_preview,clear_motion_context,current_hand_reference)
from .current_robot_scene import clear_current_robot_scene
from .coupled_storage import write_json,read_json
from .rekep_libero.geometry import pose_matrix,matrix_pose,transform_points
from .timed_execution import schedule_contact
from .execution_limits import control_limit
from .robot_contracts import require_motion_intent

ROOT=Path(__file__).resolve().parents[2]


def planning_allowances(search_seconds=None,stroke_plan_seconds=None):
    """Local work costs do not shrink when the caller shortens the whole skill."""
    search=float(30. if search_seconds is None else search_seconds)
    stroke=float(5. if stroke_plan_seconds is None else stroke_plan_seconds)
    if not np.isfinite([search,stroke]).all() or min(search,stroke)<=0:
        raise ValueError('Planning budgets must be positive')
    return search,stroke


@contextmanager
def planning_stage(parent,seconds,stage,result,timing,timing_key,*,accept_finished_work=False):
    """Account for failed searches as well as successful plans, under one deadline."""
    child=parent.child(seconds);nested=Counter();started=time.perf_counter()
    row={'stage':stage,'requested_seconds':float(seconds),'effective_seconds':child.seconds,
        'capped_by_remaining_skill_budget':child.seconds<float(seconds),'status':'running'}
    result.setdefault('planning_stages',[]).append(row)
    try:
        child.check()
        yield child,nested
        if not accept_finished_work:child.check()
        row['status']='completed'
    except SkillStop as error:
        row.update(status='stopped',stop_reason=error.reason)
        if error.reason.endswith('budget_exhausted'):
            reason={'initial_contact_search':'contact_search_budget_exhausted',
                'contact_insertion':'contact_insertion_budget_exhausted',
                'contact_stroke':'stroke_planning_budget_exhausted',
                'withdrawal':'withdrawal_planning_budget_exhausted'}[stage]
            row['stop_reason']=reason
            # Keep this same row in the evidence; the finally block completes it.
            raise SkillStop(reason,**{**error.evidence,'search_complete':False,
                'planning_stage':row,'physical_infeasibility_proved':False}) from error
        error.evidence['planning_stage']=row
        raise
    except Exception:
        row['status']='error'
        raise
    finally:
        elapsed=time.perf_counter()-started
        timing[timing_key]+=elapsed
        row.update(elapsed_seconds=elapsed,remaining_local_seconds=child.remaining,
            remaining_skill_seconds=parent.remaining,timing_seconds=dict(nested))
        if row.get('stop_reason','').endswith('budget_exhausted'):
            row['expired_budget_scope']='skill_or_episode' if parent.remaining<=0 else 'local_planning_stage'


def qualification_sources():
    paths=['src/vlm_tamp/contact_skills.py','src/vlm_tamp/contact_skill_service.py','src/vlm_tamp/contact_collision_evidence.py',
        'src/vlm_tamp/current_robot_scene.py','src/vlm_tamp/push_contact_state.py',
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
    if region is not None:draw.rectangle(region,outline='#ffcc00',width=3)
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
    counts=getattr(tracker,'last_view_counts',{})
    ranked=sorted((name for name in references if name in available),
        key=lambda name:-counts.get(name,len(references[name]['pixels'])))
    names=list(dict.fromkeys(([requested] if requested in available else [])+
        (['wrist'] if 'wrist' in available else [])+ranked))[:4]
    # Keep ordering stable when the captured set itself has not changed.
    return list(previous) if set(names)==set(previous) else names


def run_skill(service,kind,*,observation_id=None,camera=None,region_xyxy=None,target_region_xyxy=None,
              direction_world=None,distance_m=.03,joint_degrees=None,joint_model_epoch=None,
              mode='controlled',contact_speed_m_s=.03,contact_surface='fingertips',
              preview=False,budget_seconds=120.,collision_policy='veto',approach_speed_rad_s=.6,
              stroke_length_m=None,search_seconds=None,stroke_plan_seconds=None,max_ineffective_strokes=2,withdrawal_distance_m=.015,
              continuation=False):
    started=time.perf_counter();timing=Counter();stages=[];executed=False
    physical_push=kind=='push' and mode=='controlled'
    hand_distance_request=physical_push and joint_degrees is None
    if service.sim is None:raise ValueError('Reset before using contact skills')
    require_motion_intent(kind,{'direction_world':direction_world,'joint_degrees':joint_degrees})
    context=None
    if continuation:
        try:context=active_contact(service)
        except ValueError:context=current_hand_reference(service,unit(direction_world))
        if kind!='push' or mode!='controlled' or joint_degrees is not None or preview:
            raise ValueError('Continued pushing uses direction/distance and executes controlled strokes')
        direction_world=unit(direction_world).tolist()
        observation_id=service.latest_observation['observation_id']
        camera=context['camera'] if context['camera'] in service.cameras else next(iter(service.cameras))
        region_xyxy=target_region_xyxy=None
        contact_surface=context['contact_surface']
        if contact_speed_m_s is None:contact_speed_m_s=context['contact_speed_m_s']
        if stroke_length_m is None:stroke_length_m=distance_m
        if stroke_plan_seconds is None:stroke_plan_seconds=context['stroke_plan_seconds']
    if observation_id!=service.latest_observation['observation_id']:
        raise ValueError('Use the latest observation_id; contact requests cannot reuse stale pixels')
    if not continuation and service.latest_observation['control_steps']!=service.state['control_steps']:
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
    stroke_length_m=float((distance_m if hand_distance_request else .03 if mode=='push_and_release' else .01) if stroke_length_m is None else stroke_length_m)
    if not np.isfinite(stroke_length_m) or stroke_length_m<=0:raise ValueError('Stroke length must be finite and positive')
    search_seconds,stroke_plan_seconds=planning_allowances(search_seconds,stroke_plan_seconds)
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
        'direction_source':'explicit_world_direction' if joint_degrees is None else 'signed_associated_joint_increment',
        'task_direction_verified':False,
        'motion_measurement_scope':'Movement along the requested direction or joint sign; not progress toward the task goal.',
        'simulator_object_geometry_used':False,'fixed_joint_preserved':True}
    result.update(input_mode='continue_contact' if continuation else 'acquire_contact',
        contact_reused=bool(continuation),scene_reused=bool(continuation),preview_reused=False,
        full_scene_builds=0,contact_searches=0)
    associated=context.get('object_target_associated',True) if context else True
    result['object_motion_status']='not_measured_yet' if associated else 'unmeasured'
    if not associated:result.update(input_mode='current_hand',contact_reused=False,scene_reused=False)
    result['collision_policy']=collision_policy
    result['contact_motion_policy']='physical_execution_feedback' if physical_push else 'planned_contact'
    result['stroke_scene_collision_veto']=not physical_push and collision_policy=='veto'
    result['object_progress_is_execution_gate']=not physical_push
    result['environment_collision_predictions']={'count':0,'examples':[],
        'all_environment_states_checked':False,'collision_free_certificate':False}
    result['simulator_physical_contacts_enabled']=True
    scene=engagement=contact_local=point=None
    can_continue=geometry_current=False
    scene_control_step=context.get('scene_control_step',initial_control_step) if context else initial_control_step
    initial_direction=None;cumulative=np.eye(4)
    commanded_distance=0.
    cache_key=None
    try:
        budget.check()
        phase=time.perf_counter()
        cached=None
        if context is not None:
            scene=copy.copy(context['scene']);evidence=copy.deepcopy(context['evidence'])
            evidence.update(point_world_m=context['contact_world'].tolist(),
                direction_world=direction_world,joint_associated=False)
            engagement=resumed_engagement(context)
            engagement['path']=np.repeat(np.asarray(service.state['metadata']['joint_positions'])[None],2,axis=0)
            timing['contact_context_reuse_seconds']+=time.perf_counter()-phase
        else:
            cache_key=preview_key(service,{'kind':kind,'camera':camera,'region':list(region_xyxy),
                'target_region':list(target_region_xyxy) if target_region_xyxy is not None else None,
                'direction':np.round(unit(direction_world),12).tolist() if direction_world is not None else None,
                'distance_m':distance_m,'joint_degrees':joint_degrees,'mode':mode,'contact_surface':contact_surface,
                'contact_speed_m_s':contact_speed_m_s,'approach_speed_rad_s':approach_speed_rad_s,
                'stroke_length_m':stroke_length_m,'withdrawal_distance_m':withdrawal_distance_m})
            saved=getattr(service,'push_preview_cache',None)
            cached=saved if saved is not None and saved['key']==cache_key else None
            if cached:
                scene,evidence,engagement=copied_preview(cached)
                result.update(preview_reused=True,scene_reused=True,source_preview=cached['report'])
                timing['preview_reuse_seconds']+=time.perf_counter()-phase
            else:
                observed,reused=observed_contact_cloud(service)
                result['observation_cloud_reused']=reused
                scene,evidence,selected=build_contact_scene(service.cameras,service.state['metadata'],service.ik.chain,
                    camera,region_xyxy,target_region_xyxy,contact_surface,direction_world,joint_degrees,service.scene,
                    skill_kind=kind,observed=observed)
                result['full_scene_builds']=1
                timing['perception_seconds']+=time.perf_counter()-phase
        geometry_current=bool(context.get('object_pose_current',True)) if context else True
        scene.skill_collision_policy=collision_policy
        scene.skill_physical_contact_execution=physical_push
        scene.skill_approach_speed_rad_s=approach_speed_rad_s
        scene.skill_approach_hold_steps=0
        scene.skill_contact_collision_log=result['environment_collision_predictions']
        result['cartesian_path_deviation']={'diagnostic_threshold_m':.006,'maximum_m':0.,'samples_over_threshold':0,
            'execution_veto_applied':collision_policy=='veto' and not physical_push}
        scene.skill_path_deviation_log=result['cartesian_path_deviation']
        result['contact']=evidence;direction=unit(evidence['direction_world'])
        initial_direction=direction.copy()
        def collision_scene():
            query=clear_current_robot_scene(scene,service.state['metadata'],service.state['control_steps'])
            evidence=query.current_robot_filter
            timing['current_robot_filter_seconds']+=evidence['elapsed_seconds']
            result['current_robot_point_filter']={'updates':result.get('current_robot_point_filter',{}).get('updates',0)+1,
                'last_control_step':evidence['reference_control_step'],'last_removed_points':evidence['total_removed_points'],
                'padding_m':0.,'future_candidate_poses_used':False,
                'last_counts_by_cloud':{key:value['removed_points'] for key,value in evidence['clouds'].items()}}
            return query
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
        planning_scene=None
        if engagement is None:
            planning_scene=collision_scene();result['contact_searches']=1
            with planning_stage(budget,search_seconds,'initial_contact_search',result,timing,'contact_search_seconds') as (search_budget,nested):
                engagement=plan_engagement(service.ik,planning_scene,service.record,service.state['metadata'],evidence,selected,kind,search_budget,
                    maximum_approach_controls=lambda p:max(0,control_steps_limit-service.state['control_steps']-following_controls(p)))
                nested.update(engagement.get('timing_seconds',{}))
            # Hook selection can refine association metadata, independently of
            # the raw observed obstacle clouds retained for feedback.
            scene.contact_reference=planning_scene.contact_reference
        result['contact_reused']=bool(engagement.get('reuse_existing_contact'))
        if collision_policy=='advisory' and budget.remaining>2.:
            phase=time.perf_counter()
            try:
                with skill_native_scene(planning_scene or collision_scene(),service.record,service.state['metadata'],service.state['metadata']['joint_positions']) as checker:
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
            service.push_preview_cache={'key':cache_key,'scene':copy.copy(scene),
                'evidence':copy.deepcopy(evidence),'engagement':engagement,
                'report':str((root/'result.json').relative_to(service.workspace))}
            result.update(status='preview_ready' if result['control_budget_feasible'] else 'preview_budget_limited',
                stop_reason='preview_only',predicted_contact_only=True)
            return result
        budget.check()
        if len(engagement['references'])+next_controls>control_steps_limit-service.state['control_steps']:
            raise SkillStop('insufficient_control_budget',approach_steps=len(engagement['references']),
                remaining=control_steps_limit-service.state['control_steps'],minimum_following_controls=next_controls)
        contact_local=engagement['contact_local'];point=engagement['contact_world'].copy()
        reused_pose=bool(engagement.get('reuse_existing_contact') or engagement.get('reuse_current_hand_pose'))
        can_continue=kind=='push' and reused_pose
        original_point=point.copy();cumulative=np.eye(4);progress=0.
        ineffective=int(context.get('ineffective_strokes',0)) if context else 0
        unwrapped_angle=0.
        tracked_target=scene.moving_reference.copy()
        phase=time.perf_counter()
        tracker=context.get('tracker') if context else None
        result['tracking_reference_reused']=tracker is not None and associated
        if tracker is None:tracker=ContactTracker(service.cameras,tracked_target,service.state['control_steps'])
        result['tracking_reference_camera_count']=len(tracker.tracker.references)
        timing['tracker_initialization_seconds']+=time.perf_counter()-phase
        view_names=[]
        def refresh_feedback_views(reference_tracker):
            nonlocal view_names
            available=set(service.cameras)|set(reference_tracker.tracker.references)
            selected_views=feedback_camera_names(reference_tracker,camera,available,view_names)
            if selected_views!=view_names:
                result.setdefault('feedback_camera_changes',[]).append({
                    'control_steps':int(service.state['control_steps']),'cameras':selected_views})
            view_names=selected_views;result['feedback_cameras']=list(view_names)
        refresh_feedback_views(tracker)

        def issue(refs,label,**stage_fields):
            nonlocal executed,geometry_current
            budget.check()
            remaining_controls=control_steps_limit-service.state['control_steps']
            if len(refs)>remaining_controls:
                if physical_push and label.startswith('push-stroke') and remaining_controls>0:
                    stage_fields['control_budget_truncated']=True;refs=refs[:remaining_controls]
                else:raise SkillStop('control_budget_exhausted',required_steps=len(refs))
            service.calls+=1;before=len(service.events);q=service.state['metadata']['joint_positions']
            phase=time.perf_counter()
            clear_motion_context(service)
            feedback=None
            if physical_push and label.startswith('push-stroke'):
                from .push_execution_feedback import contact_point
                feedback={'contact_local_m':contact_local.tolist(),
                    'reference_points_world_m':[contact_point(chain.forward(q),contact_local).tolist() for q in refs]}
            reply=service.issue(refs,0,label,camera_names=view_names,track_scene=False,
                **({'push_feedback':feedback} if feedback is not None else {}))
            if reply['execution']['issued_commands']:geometry_current=False
            executed=True;timing['execution_and_capture_seconds']+=time.perf_counter()-phase
            events=service.events[before:]
            phase=time.perf_counter()
            stage={'stage':label,'artifact_id':service.calls,'control_steps':service.state['control_steps'],
                'execution':reply['execution'],'measured_contact_speed':measured_contact_speed(events,service.ik.chain if service.ik else chain,contact_local,q)}
            stage.update(stage_fields)
            if reply['execution'].get('push_feedback'):stage['physical_motion']=reply['execution']['push_feedback']
            issued=int(reply['execution']['issued_commands'])
            desired=pose_matrix(chain.forward(refs[issued-1] if issued else q))
            actual=pose_matrix(chain.forward(service.state['metadata']['joint_positions']))
            stage['robot_reference_tracking']={'position_error_m':float(np.linalg.norm(actual[:3,3]-desired[:3,3])),
                'orientation_error_deg':float(np.degrees(Rotation.from_matrix(desired[:3,:3].T@actual[:3,:3]).magnitude())),
                'reference':'last issued robot command; not an object-motion or grasp certificate'}
            if feedback is not None:
                actual_point=actual[:3,3]+actual[:3,:3]@contact_local
                commanded_point=desired[:3,3]+desired[:3,:3]@contact_local
                result['commanded_contact_distance_m']=float((commanded_point-initial_hand)@initial_direction)
                result['measured_contact_progress_m']=float((actual_point-initial_hand)@initial_direction)
            timing['speed_measurement_seconds']+=time.perf_counter()-phase
            stages.append(stage);result['observation']=reply['observation']
            result['episode_done']=bool(reply.get('episode_done'))
            if reply.get('task_success'):
                result.update(task_success=True,status='task_success',stop_reason='native_task_success')
                raise SkillStop('native_task_success')
            if reply.get('episode_done'):raise SkillStop(reply.get('termination_reason','episode_budget_exhausted'))
            # Finish feedback for an issued stroke even if its capture crossed
            # the deadline. Never start a new stroke with an expired budget.
            return stage

        recovered_steps=set()
        def observe_motion(tracker):
            phase=time.perf_counter();fit=tracker.observe(service.cameras,service.state['control_steps'])
            timing['motion_tracking_seconds']+=time.perf_counter()-phase
            step=int(service.state['control_steps'])
            if not fit['accepted'] and not physical_push and len(service.cameras)<=4 and step not in recovered_steps and budget.remaining>0:
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
        if not reused_pose:
            issue(engagement['free_references'],'free-approach')
            measured=observe_motion(tracker)
            stages[-1]['object_motion']=measured
            if not measured['accepted']:
                if not physical_push:raise SkillStop('object_motion_unresolved',measurement=measured)
                result.setdefault('motion_warnings',[]).append({'reason':'object_motion_unresolved','measurement':measured})
            approach_transform=np.asarray(measured['transform']) if measured['accepted'] else np.eye(4)
            approach_object_displacement=np.linalg.norm(transform_points([point],approach_transform)[0]-point)
            stages[-1]['uncommanded_object_motion_m']=float(approach_object_displacement)
            if approach_object_displacement>.08:
                if collision_policy=='veto' and not physical_push:
                    raise SkillStop('target_moved_beyond_local_recovery',observed_displacement_m=float(approach_object_displacement),maximum_m=.08)
                result.setdefault('motion_warnings',[]).append({'reason':'large_measured_motion_during_approach',
                    'observed_displacement_m':float(approach_object_displacement),'pose_update_source':'accepted independent RGB-D fit'})

        def update_motion(fit):
            nonlocal scene,point,cumulative,tracked_target,direction,progress,unwrapped_angle,geometry_current,scene_control_step
            transform=np.asarray(fit['transform']);r=transform[:3,:3];t=transform[:3,3]
            joint_mismatch=None
            if joint_degrees is not None:
                drift=float(np.linalg.norm(r@scene.origin+t-scene.origin))
                if drift>.006:
                    joint_mismatch={'fixed_hinge_displacement_m':drift,'maximum_m':.006}
                    if kind!='push' or mode!='controlled':
                        raise SkillStop('motion_inconsistent_with_fixed_joint',**joint_mismatch)
            old=point.copy();point=r@point+t;cumulative=transform@cumulative
            phase=time.perf_counter()
            scene,update=advance_contact_scene(scene,transform,service.cameras,service.state['control_steps'])
            timing['moving_geometry_update_seconds']+=time.perf_counter()-phase
            stages[-1]['scene_update']=update
            geometry_current=True
            scene_control_step=int(service.state['control_steps'])
            tracked_target=scene.moving_reference
            if joint_degrees is None:
                progress=float((point-original_point)@direction)
            else:
                unwrapped_angle+=float(Rotation.from_matrix(r).as_rotvec()@scene.axis)
                progress=float(unwrapped_angle*np.sign(joint_degrees))
                direction=unit(np.sign(joint_degrees)*np.cross(scene.axis,point-scene.origin))
            result['progress_measured_at_control_step']=int(service.state['control_steps'])
            if joint_mismatch is not None:
                # An unsupported hinge does not erase an independently measured
                # part pose. Stop the joint-angle request, retaining the scene
                # for a direction-only continuation from the reached contact.
                result['joint_model_supported']=False
                raise SkillStop('motion_inconsistent_with_fixed_joint',**joint_mismatch,
                    measured_part_pose_retained=True,direction_only_continuation_supported=bool(can_continue))
            return float((point-old)@direction)

        if measured is not None and measured['accepted']:update_motion(measured)
        # Replan the short insertion from the actual servo endpoint. A nominal
        # reference endpoint does not establish physical arrival at standoff.
        if not reused_pose:
            refresh_feedback_views(tracker)
            meta=service.state['metadata'];start=np.asarray(meta['joint_positions'])
            insertion_goal=np.eye(4);insertion_goal[:3,:3]=engagement['rotation']
            insertion_goal[:3,3]=point-engagement['rotation']@contact_local
            planning_scene=copy.copy(scene) if physical_push else collision_scene()
            if physical_push:planning_scene.skill_collision_policy='advisory'
            with planning_stage(budget,stroke_plan_seconds,'contact_insertion',result,timing,'stroke_planning_seconds',accept_finished_work=physical_push) as (planning,nested):
                insertion=cartesian_path(service.ik,planning_scene,service.record,meta,start,matrix_pose(insertion_goal),planning,timing=nested,allow_partial=physical_push)
            stage=issue(schedule_contact(insertion,chain,contact_local,contact_speed_m_s=.03,endpoint_hold_steps=3),'contact-insertion',
                requested_contact_speed_m_s=.03,planning_timing_seconds=dict(nested))
            measured=observe_motion(tracker);stage['object_motion']=measured
            can_continue=kind=='push'
            if measured['accepted']:update_motion(measured)
            elif not physical_push:raise SkillStop('object_motion_unresolved',measurement=measured)
            else:result.setdefault('motion_warnings',[]).append({'reason':'object_motion_unresolved','measurement':measured})
        # Measure the requested push from contact arrival, not from the initial
        # free approach. A continuation normally executes one complete stroke.
        initial_direction=direction.copy();cumulative=np.eye(4)
        initial_hand=pose_matrix(chain.forward(service.state['metadata']['joint_positions']))
        initial_hand=initial_hand[:3,3]+initial_hand[:3,:3]@contact_local
        target=distance_m if joint_degrees is None else abs(np.radians(joint_degrees))
        result['object_progress_reference_control_step']=scene_control_step
        scene.skill_collision_policy=collision_policy
        scene.skill_contact_collision_log=result['environment_collision_predictions']
        release_contact_ready=collision_policy=='advisory'
        stroke=-1
        while True:
            stroke+=1
            if hand_distance_request and commanded_distance>=distance_m-1e-10:
                error=float(result.get('measured_contact_progress_m',0.)-distance_m)
                result.update(status='executed',stop_reason='stroke_executed',contact_progress_error_m=error,
                    hand_travel_shortfall_m=max(0.,-error),object_progress_is_execution_gate=False)
                break
            if not hand_distance_request and mode=='controlled' and progress>=target-(.001 if joint_degrees is None else np.radians(.5)):
                result.update(status='requested_progress_reached',stop_reason='requested_progress_reached');break
            budget.check()
            # The measured robot is the starting fact, not another observation
            # gate. Correct the collision cloud once, then plan against it.
            planning_scene=scene if physical_push else collision_scene()
            refresh_feedback_views(tracker)
            meta=service.state['metadata'];start=np.asarray(meta['joint_positions']);tcp=pose_matrix(chain.forward(start))
            hand_point=tcp[:3,3]+tcp[:3,:3]@contact_local
            # A push can slide across the part. Separation from the original
            # material point is diagnostic, not a new contact-loss veto.
            separation=float(np.linalg.norm(hand_point-point))
            if kind=='hook_pull' and separation>.025 and collision_policy=='veto':
                raise SkillStop('contact_geometry_lost',landmark_separation_m=separation,maximum_m=.025)
            acquiring_release_contact=mode=='push_and_release' and not release_contact_ready
            stroke_speed=.03 if acquiring_release_contact else contact_speed_m_s
            measured_before=float((hand_point-initial_hand)@initial_direction)
            length=(.01 if acquiring_release_contact else
                min(stroke_length_m,distance_m) if mode=='push_and_release' else
                min(stroke_length_m,distance_m-measured_before if hand_distance_request else
                    distance_m-progress if joint_degrees is None else stroke_length_m))
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
            with planning_stage(budget,stroke_plan_seconds,'contact_stroke',result,timing,'stroke_planning_seconds',accept_finished_work=physical_push) as (planning,nested):
                path,search=plan_contact_stroke(service.ik,planning_scene,service.record,meta,contact_local,delta,planning,angle=angle,timing=nested,physical_feedback=physical_push)
            if collision_policy=='advisory' and not physical_push and budget.remaining>2.:
                phase=time.perf_counter();log=scene.skill_contact_collision_log
                try:
                    predicted=predicted_scene(planning_scene,delta,angle)
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
                planning_timing_seconds=dict(nested),requested_motion_m=float(np.linalg.norm(delta)),
                material_point_offset_m=separation)
            if hand_distance_request:commanded_distance=measured_before+float(delta@initial_direction)
            else:commanded_distance+=float(np.linalg.norm(delta))
            hand=pose_matrix(chain.forward(service.state['metadata']['joint_positions']))
            hand=hand[:3,3]+hand[:3,:3]@contact_local
            if not physical_push:result['commanded_contact_distance_m']=commanded_distance
            result['measured_contact_progress_m']=float((hand-initial_hand)@initial_direction)
            fit=observe_motion(tracker);stage['object_motion']=fit
            finish_release=mode=='push_and_release' and collision_policy=='advisory' and not acquiring_release_contact
            motion_applied=False;moved=None
            if fit['accepted']:
                can_continue=kind=='push'
                try:moved=update_motion(fit);motion_applied=True
                except SkillStop as error:
                    if not finish_release:raise
                    result.setdefault('motion_warnings',[]).append({'reason':error.reason,**error.evidence})
            elif not finish_release and not physical_push:raise SkillStop('object_motion_unresolved',measurement=fit)
            elif associated:result.setdefault('motion_warnings',[]).append({'reason':'object_motion_unresolved','measurement':fit})
            if physical_push:can_continue=True
            stage['observed_contact_progress_m']=moved
            stage['engagement']={'status':'object_motion_unresolved' if moved is None else
                'object_followed_contact' if moved>.001 else 'no_supported_object_progress',
                'pinch_assumed':False,'force_or_tactile_measurement':False}
            result['observed_progress']=progress if motion_applied else None
            result['progress_units']='radians' if joint_degrees is not None else 'metres'
            result['object_motion_status']='measured' if motion_applied else 'unresolved' if associated else 'unmeasured'
            if stage['execution'].get('termination_reason')=='measured_push_stall':
                raise SkillStop('physical_stall',feedback=stage['execution'].get('push_feedback'),
                    execution_attempted=True,scene_prediction_veto=False)
            if physical_push and search.get('reachable_prefix'):
                result.update(status='reachable_prefix_executed',stop_reason='reachable_prefix_executed',
                    planning_limit=search['reachable_prefix']);break
            if physical_push and joint_degrees is not None and not motion_applied:
                result.update(status='motion_feedback_unresolved',stop_reason='motion_feedback_unresolved');break
            if kind=='hook_pull' and stroke==0 and moved<=.001:raise SkillStop('hook_engagement_not_established',measured_progress_m=moved)
            if acquiring_release_contact:
                if moved<=.001:raise SkillStop('push_contact_not_established',measured_progress_m=moved,
                    explanation='The slow contact test did not establish observed part motion; no faster pulse was issued.')
                release_contact_ready=True
                continue
            ineffective=ineffective+1 if moved is not None and moved<=.001 else 0
            if ineffective>=max_ineffective_strokes and not hand_distance_request:raise SkillStop('no_progress_after_stroke_budget',measured_progress_m=moved,consecutive_ineffective_strokes=ineffective)
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
                planning_scene=collision_scene()
                with planning_stage(budget,stroke_plan_seconds,'withdrawal',result,timing,'stroke_planning_seconds') as (planning,nested):
                    path=cartesian_path(service.ik,planning_scene,service.record,meta,start,matrix_pose(retreat),planning,timing=nested)
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
        if error.reason=='physical_stall':
            result['recovery_options']=['inspect wrist/global views and measured motion',
                'change direction, distance or speed with push_stroke or continue_push',
                'reposition the hand if the current contact is unsuitable']
        elif error.reason.startswith('contact_search'):
            result['recovery_options']=['push_stroke can attempt motion directly from the current hand pose without contact acquisition',
                'choose another approach/region when repositioning to a target is needed']
        if error.reason in {'object_motion_unresolved','motion_inconsistent_with_fixed_joint','no_progress_after_stroke_budget','hook_engagement_not_established'}:
            if error.reason in {'object_motion_unresolved','hook_engagement_not_established'}:can_continue=False
            result['recovery_options']=['select an opaque currently observed region','use a current surface push without a hinge assumption','apply an independently measured part/joint update']
            if error.reason=='no_progress_after_stroke_budget' and kind=='push':
                result['recovery_options'].insert(0,'change direction, distance or speed with continue_push; the tracked scene is retained, while effective engagement remains unverified')
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
        if (kind=='push' and mode=='controlled' and not preview and can_continue
                and service.sim is not None and not result.get('episode_done') and not result.get('task_success')):
            evidence=copy.deepcopy(result['contact'])
            if not associated:
                point=np.asarray(service.state['metadata']['eef_pose_world_xyzw'][:3],float)
            evidence['point_world_m']=point.tolist()
            suggested=unit(cumulative[:3,:3]@initial_direction)
            evidence['direction_world']=suggested.tolist()
            service.active_push_contact={'contact_id':context['contact_id'] if context else f'push-{service.attempt}-{root.name}',
                'scene':scene,'evidence':evidence,'contact_local':contact_local.copy(),'contact_world':point.copy(),
                'contact_name':engagement['contact_name'],'contact_surface':contact_surface,'camera':camera,
                'direction_world':suggested,'control_step':int(service.state['control_steps']),
                'contact_speed_m_s':contact_speed_m_s,'stroke_length_m':stroke_length_m,
                'stroke_plan_seconds':stroke_plan_seconds,'ineffective_strokes':ineffective,'tracker':tracker,
                'object_pose_current':geometry_current,'scene_control_step':scene_control_step}
            service.active_push_contact['object_target_associated']=associated
        retained=getattr(service,'active_push_contact',None)
        result['continuation']=contact_summary(retained)
        result['continue_push_available']=bool(retained)
        if retained and not preview:
            result['grasp_scene_refresh_required']=result['scene_refresh_required']
            result['scene_refresh_required']=False
            result['next_operation']='continue_push'
            result['scene_refresh_scope']='Direction-only pushing reuses the robot contact reference. Object pose freshness is reported separately; grasp/joint-model planning requires current grounding.'
        result['robot_motion_executed']=executed
        result['remaining_skill_seconds']=budget.remaining
        if joint_degrees is not None and result.get('observed_progress') is not None:
            result['observed_joint_change_degrees']=float(np.degrees(result['observed_progress'])*np.sign(joint_degrees))
            result['observed_joint_change_source']='RGB-D part pose projected onto the installed signed axis; not native joint state'
        result['feedback_count']=sum('object_motion' in s for s in stages)
        if result.get('observed_progress') is not None:
            result['measured_object_motion_along_request']=result['observed_progress']
        result['timing_seconds']={'total':time.perf_counter()-started,**dict(timing)}
        result['timing_seconds']['other_seconds']=max(0.,result['timing_seconds']['total']-sum(timing.values()))
        result['report']=str((root/'result.json').relative_to(service.workspace))
        write_json(root/'result.json',result)
        write_json(service.episode/f'{root.name}-result.json',result)
    return result
