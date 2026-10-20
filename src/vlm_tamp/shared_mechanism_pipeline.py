"""Observed-progress drawer execution using the shared contact interface.

Only start a new plan after the previous nominal trajectory is fully issued.
Progress comes from RGB-D registration, with an explicitly logged proprioceptive
fallback. Native success and object joint values are consumed only after close.
"""
from pathlib import Path
import time
import traceback
import numpy as np

from .coupled_storage import read_json,write_json,compact_episode
from .cohort_execution import read_cohort_observation
from .reliable_mechanisms import segment
from .reliability_geometry import SurfaceScene
from .shared_contact_geometry import grounded_geometry,expand_contact,closing_target
from .shared_contact_planning import select_contact
from .shared_feedback_client import FeedbackClient,execute_shared
from .rekep_libero.mechanism_geometry import refine_contact
from .rekep_libero.task_ik_curobo import TaskIK
from .timed_execution import schedule
from .mcvamp_mechanism import clear_native_cache

CoupledClient=FeedbackClient


class ObservedProgress:
    def __init__(self,axis):
        self.axis=np.asarray(axis,float);self.axis/=np.linalg.norm(self.axis)
        self.progress=0.;self.observed_progress=0.;self.records=[]

    def update(self,tracking,hand_delta):
        if tracking.get('accepted'):
            delta=np.asarray(tracking['translation_world_m'])
            self.observed_progress+=float(delta@self.axis)
            self.progress=self.observed_progress;source='RGB-D handle displacement'
        else:
            delta=np.asarray(hand_delta)
            self.progress+=float(delta@self.axis);source='robot-motion fallback; object progress unobserved'
        self.records.append({'source':source,'progress_m':self.progress,
            'observed_progress_m':self.observed_progress,'tracking_accepted':bool(tracking.get('accepted')),
            'execution_veto':False})
        return self.progress


def run_episode(task,index,folder,config,resources):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    sim=CoupledClient(folder/'simulator.log');ik=None;state=None;events=[];phases=[];stage='reset'
    started=time.perf_counter();calls=resources.new_api_calls
    result={'task':task['label'],'official_task_name':task['name'],'suite':task['suite'],
        'initial_state':index,'seed':0,'status':'failed','official_task_success':False,
        'execution_veto_checks':0,'execution_time_collision_checks':False,
        'execution_time_progress_vetoes':False,'planning_scene_admission_enabled':True,
        'assisted_attachment':False,'physical_contacts_enabled':True,
        'planner':'VAMP approach and MC-VAMP constrained line','native_state_used_for_control':False}

    def issue(path,label,gripper,hold=6,reserve=0):
        nonlocal state
        refs=schedule(np.asarray(path,float),maximum_joint_speed=.35,endpoint_hold_steps=hold)
        available=600-int(state['control_steps'])-reserve
        if available<=hold:raise ValueError('Insufficient budget for a complete next trajectory')
        if len(refs)>available:
            refs=schedule(np.asarray(path,float),maximum_joint_speed=.35*(len(refs)-hold)/(available-hold)*1.001,endpoint_hold_steps=hold)
        out=folder/f'path-{len(phases):02d}-{label}';out.mkdir();np.save(out/'references.npy',refs)
        state,done=execute_shared(sim,state,refs,gripper=gripper,event_sink=events.append)
        phases.append({'label':label,**{k:v for k,v in done.items() if k!='events'}})
        write_json(folder/'phases.json',phases)

    try:
        state=sim.reset(task['name'],index,folder/'initial',suite_name=task['suite'])
        state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,output_dir=folder/'opened')
        stage='rgb_grounding'
        choices=resources.grounding(task['label'],index,folder/'opened',state['metadata'],folder/'grounding')
        cameras,metadata=read_cohort_observation(folder/'opened/obs.npz',state['metadata']);name=choices['camera']
        with np.load(folder/'grounding'/name/'keypoints.npz') as values:bundle={k:values[k].copy() for k in values.files}
        bundle.update(cameras=cameras,meta=metadata,main_camera=name)
        stage='observed_geometry'
        cloud,mask=refine_contact(bundle,choices,resources.ensure_perception(),folder)
        mechanism,panel=grounded_geometry(bundle,choices,cloud,mask,resources.ensure_perception(),folder)
        if mechanism.kind!='prismatic':raise ValueError('Observed-progress line runner requires prismatic semantics')
        spec=mechanism.spec;cloud=expand_contact(cameras,spec,choices,cloud)
        spec['source_anchor_world_m']=np.median(cloud,axis=0).tolist()
        if spec['motion']=='close':spec=closing_target(cameras,state['metadata'],spec,cloud,config)
        write_json(folder/'task-spec.json',spec);np.save(folder/'grasp-cloud.npy',cloud)
        generated=resources.grasps(cloud,folder/'grasps');record=sim.robot_chain();write_json(folder/'robot-chain.json',record)
        import vamp
        empty=vamp.Environment();ik=TaskIK(record,config,lambda q:vamp.panda.validate(np.asarray(q).tolist(),empty,True),output=folder,sim=sim)
        scene=SurfaceScene(cameras,state['metadata'],spec,cloud,panel,config);write_json(folder/'scene-scope.json',scene.record())
        stage='contact_mode_planning'
        chosen,approach=select_contact(generated,cloud,spec,scene,record,ik,state,sim,folder,config)
        write_json(folder/'chosen-grasp.json',chosen)
        state=sim.enable_joint_tracking(kp=300.)
        stage='approach';issue(approach,'approach',-1,hold=20,reserve=180)
        stage='physical_contact';q=np.asarray(chosen['solution_joints']);issue([q,q],'contact',1,hold=40,reserve=140)
        before_hand=np.asarray(state['metadata']['eef_pose_world_xyzw'][:3]);state=sim.observe(folder/'closed')
        progress=ObservedProgress(spec['joint_axis_world'])
        tracking=sim.depth_records[-1]['tracking'] if sim.depth_records else {'accepted':False}
        progress.update(tracking,np.zeros(3))
        views,_=read_cohort_observation(folder/'closed/obs.npz',state['metadata'])
        transform=np.eye(4)
        if tracking.get('accepted'):transform[:3,3]=tracking['translation_world_m']
        scene=SurfaceScene(views,state['metadata'],spec,cloud,panel,config,previous=scene,transform=transform)
        tolerance=float(config['scene_voxel_spacing_m']);target=float(spec['target_progress'])
        for number in range(48):
            remaining=target-progress.progress
            if remaining<=tolerance:
                result['policy_goal_complete']=True;break
            stage='planning_between_complete_trajectories'
            # Closing uses one bounded seating segment at the observed face.
            seat=spec['motion']=='close' and remaining<=.020
            amount=min(.025,max(remaining,tolerance))
            q=np.asarray(state['metadata']['joint_positions'])
            path,info=segment(scene,record,state['metadata'],ik,q,amount,folder/f'plan-{number:02d}',seconds=8.)
            if path is None:raise ValueError('No complete constrained line segment: '+info['reason'])
            before_hand=np.asarray(state['metadata']['eef_pose_world_xyzw'][:3])
            stage='complete_manipulation_trajectory';issue(path,f'manipulate-{number:02d}',1,hold=6)
            state=sim.observe(folder/f'observation-{number:02d}')
            hand_delta=np.asarray(state['metadata']['eef_pose_world_xyzw'][:3])-before_hand
            tracking=sim.depth_records[-1]['tracking'] if sim.depth_records else {'accepted':False}
            progress.update(tracking,hand_delta)
            write_json(folder/'observed-progress.json',{'target_m':target,'tolerance_m':tolerance,'records':progress.records})
            if seat:
                result.update(policy_goal_complete=True,completion_policy='one fully issued seating segment near observed cabinet face');break
            transform=np.eye(4);transform[:3,3]=tracking['translation_world_m'] if tracking.get('accepted') else hand_delta
            views,_=read_cohort_observation(folder/f'observation-{number:02d}/obs.npz',state['metadata'])
            scene=SurfaceScene(views,state['metadata'],spec,cloud,panel,config,previous=scene,transform=transform)
        else:raise ValueError('Maximum completed manipulation segments reached')
        result.update(status='completed_planned_trajectories',observed_goal_progress_m=progress.progress)
        sim.observe(folder/'final')
    except Exception as error:
        result.update(failure_stage=stage,failure=str(error));(folder/'error.txt').write_text(traceback.format_exc())
    finally:
        if ik:ik.close()
        sim.close();clear_native_cache()
        actions=read_json(folder/'initial/actions.json') if (folder/'initial/actions.json').exists() else []
        result.update(official_task_success=any(a['evaluation_task_success'] for a in actions),
            final_task_success=bool(actions and actions[-1]['evaluation_task_success']),control_steps=len(actions),
            all_issued_trajectories_completed=all(p['all_commands_issued'] for p in phases),
            completed_trajectories=len(phases),new_api_calls=resources.new_api_calls-calls,
            episode_wall_seconds=time.perf_counter()-started,pipeline_status=result['status'])
        if result['official_task_success']:result['status']='task_success'
        write_json(folder/'tracking.json',events);write_json(folder/'summary.json',result)
        result['storage']=compact_episode(folder);write_json(folder/'summary.json',result)
        resources.cleanup_proposals()
    return result
