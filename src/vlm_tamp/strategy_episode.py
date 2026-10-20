"""A bounded VLM/tool loop on one continuous physical drawer episode.

Strategies are chosen outside this module. The numerical tools retain their
shared geometry/contact priors; this pilot does not claim autonomous geometry
model discovery. Native simulator success is read only after execution ends.
"""
from pathlib import Path
import hashlib
import time
import traceback
import numpy as np

from .cohort_execution import read_cohort_observation
from .coupled_storage import read_json,write_json,compact_episode
from .shared_contact import contact_menu
from .shared_contact_geometry import grounded_geometry,expand_contact,closing_target
from .shared_contact_planning import select_contact
from .shared_feedback_client import FeedbackClient,execute_shared
from .shared_mechanism_pipeline import ObservedProgress
from .reliability_geometry import SurfaceScene
from .reliable_mechanisms import segment
from .rekep_libero.mechanism_geometry import refine_contact
from .rekep_libero.task_ik_curobo import TaskIK
from .timed_execution import schedule
from .mcvamp_mechanism import clear_native_cache
from .strategy_feedback import planner_feedback,feedback_packet,render_feedback,save_packet


class StrategyTools:
    def __init__(self,task,index,folder,config,resources):
        self.task=task;self.index=index;self.folder=Path(folder);self.config=config;self.resources=resources
        self.sim=None;self.ik=None;self.state=None;self.scene=None;self.current_choice=None
        self.events=[];self.phases=[];self.history=[];self.observe_number=0;self.plan_number=0
        self.goal_reached=False;self.stage='reset';self.latest_pose=None

    def prepare(self):
        self.sim=FeedbackClient(self.folder/'simulator.log',variant='reactive')
        self.state=self.sim.reset(self.task['name'],self.index,self.folder/'initial',suite_name=self.task['suite'])
        self.state=self.sim.action(self.state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,output_dir=self.folder/'opened')
        self.stage='visual_grounding'
        choices=self.resources.grounding(self.task['label'],self.index,self.folder/'opened',self.state['metadata'],self.folder/'grounding')
        self.views,metadata=read_cohort_observation(self.folder/'opened/obs.npz',self.state['metadata'])
        self.previous_views=self.views
        with np.load(self.folder/'grounding'/choices['camera']/'keypoints.npz') as a:bundle={k:a[k].copy() for k in a.files}
        bundle.update(cameras=self.views,meta=metadata,main_camera=choices['camera'])
        self.stage='sensor_geometry'
        self.cloud,mask=refine_contact(bundle,choices,self.resources.ensure_perception(),self.folder)
        mechanism,self.panel=grounded_geometry(bundle,choices,self.cloud,mask,self.resources.ensure_perception(),self.folder)
        if mechanism.kind!='prismatic':raise ValueError('This initial strategy pilot exposes the prismatic motion tool')
        self.spec=mechanism.spec;self.cloud=expand_contact(self.views,self.spec,choices,self.cloud)
        self.spec['source_anchor_world_m']=np.median(self.cloud,axis=0).tolist()
        if self.spec['motion']=='close':self.spec=closing_target(self.views,self.state['metadata'],self.spec,self.cloud,self.config)
        write_json(self.folder/'task-spec.json',self.spec);np.save(self.folder/'grasp-cloud.npy',self.cloud)
        self.generated=self.resources.grasps(self.cloud,self.folder/'grasps')
        self.record=self.sim.robot_chain();write_json(self.folder/'robot-chain.json',self.record)
        import vamp
        empty=vamp.Environment()
        self.ik=TaskIK(self.record,self.config,lambda q:vamp.panda.validate(np.asarray(q).tolist(),empty,True),output=self.folder,sim=self.sim)
        self.scene=SurfaceScene(self.views,self.state['metadata'],self.spec,self.cloud,self.panel,self.config)
        self.progress=ObservedProgress(self.spec['joint_axis_world'])
        self.state=self.sim.enable_joint_tracking(kp=300.)
        write_json(self.folder/'scene-scope.json',self.scene.record())
        self.geometry=self.geometry_record()

    def geometry_record(self):
        _,evidence=contact_menu(self.cloud,self.spec,self.state['metadata'],
            np.vstack([self.scene.static_points,self.scene.moving_reference]),
            depth_resolution=self.config['scene_voxel_spacing_m'],mode_order=['pinch','press_drag','push'])
        return evidence

    def issue(self,path,label,gripper,*,hold=6,reserve=0):
        refs=schedule(np.asarray(path,float),maximum_joint_speed=.35,endpoint_hold_steps=hold)
        available=600-int(self.state['control_steps'])-reserve
        if available<=hold:raise ValueError('Insufficient control budget for a complete trajectory')
        if len(refs)>available:
            refs=schedule(np.asarray(path,float),maximum_joint_speed=.35*(len(refs)-hold)/(available-hold)*1.001,endpoint_hold_steps=hold)
        output=self.folder/f'path-{len(self.phases):02d}-{label}';output.mkdir()
        np.save(output/'references.npy',refs)
        self.state,done=execute_shared(self.sim,self.state,refs,gripper=gripper,event_sink=self.events.append)
        self.phases.append({'label':label,**{k:v for k,v in done.items() if k!='events'}})
        write_json(self.folder/'phases.json',self.phases)

    def observe(self,*,before_hand=None):
        folder=self.folder/f'observation-{self.observe_number:02d}';self.observe_number+=1
        before_records=len(self.sim.depth_records)
        self.state=self.sim.observe(folder)
        self.previous_views=self.views
        self.views,_=read_cohort_observation(folder/'obs.npz',self.state['metadata'])
        tracking=self.sim.depth_records[-1]['tracking'] if len(self.sim.depth_records)>before_records else {'accepted':False}
        hand=np.asarray(self.state['metadata']['eef_pose_world_xyzw'][:3])
        hand_delta=np.zeros(3) if before_hand is None else hand-np.asarray(before_hand)
        if self.current_choice is not None:
            self.progress.update(tracking,hand_delta)
            transform=np.eye(4)
            transform[:3,3]=tracking['translation_world_m'] if tracking.get('accepted') else hand_delta
            self.scene=SurfaceScene(self.views,self.state['metadata'],self.spec,self.cloud,self.panel,self.config,previous=self.scene,transform=transform)
            self.cloud=self.scene.contact_reference.copy()
            self.spec['source_anchor_world_m']=np.median(self.cloud,axis=0).tolist()
            write_json(self.folder/'observed-progress.json',{'target_m':self.spec['target_progress'],
                'tolerance_m':self.config['scene_voxel_spacing_m'],'records':self.progress.records})
        self.geometry=self.geometry_record()
        return {'tracking_accepted':bool(tracking.get('accepted')),
            'tracking_coverage':tracking.get('coverage_6mm'),
            'tracking_residual_m':tracking.get('percentile75_m'),
            'observed_target_translation_world_m':tracking.get('translation_world_m') if tracking.get('accepted') else None,
            'measured_hand_translation_world_m':hand_delta.tolist()}

    def motion(self,extent,folder):
        if self.current_choice is None:return {'status':'no_active_contact','manipulation_executed':False}
        start=self.progress.observed_progress;completed=0;last={};last_plan={}
        for _ in range(1 if extent=='probe' else 4):
            remaining=float(self.spec['target_progress'])-self.progress.progress
            tolerance=float(self.config['scene_voxel_spacing_m'])
            if remaining<=tolerance:
                self.goal_reached=True;break
            if 600-int(self.state['control_steps'])<=12:break
            seating=self.spec['motion']=='close' and remaining<=.020
            amount=min(.025,max(remaining,tolerance))
            q=np.asarray(self.state['metadata']['joint_positions']);self.stage='motion_planning'
            output=self.folder/f'plan-{self.plan_number:02d}';self.plan_number+=1
            path,info=segment(self.scene,self.record,self.state['metadata'],self.ik,q,amount,output,seconds=8.)
            last_plan=planner_feedback(output)
            if path is None:
                return {'status':'planning_failed','phase':'manipulation','reason':info['reason'],
                    'manipulation_executed':bool(completed),'completed_segments':completed,
                    'object_progress_delta_m':self.progress.observed_progress-start if last.get('tracking_accepted') else None,**last,**last_plan}
            before=np.asarray(self.state['metadata']['eef_pose_world_xyzw'][:3]);self.stage='execution'
            self.issue(path,f'manipulate-{self.plan_number:02d}',1);completed+=1
            last=self.observe(before_hand=before)
            if seating:
                self.goal_reached=True;break
        if float(self.spec['target_progress'])-self.progress.progress<=float(self.config['scene_voxel_spacing_m']):
            self.goal_reached=True
        return {'status':'motion_completed','completed_segments':completed,'manipulation_executed':bool(completed),
            'object_progress_delta_m':self.progress.observed_progress-start if last.get('tracking_accepted') else None,
            'estimated_goal_reached':self.goal_reached,**last,**last_plan}

    def try_contact(self,decision,folder):
        # New contact searches use the current physical state, never a reset.
        if self.current_choice is not None:
            self.sim.context=None
            q=np.asarray(self.state['metadata']['joint_positions'])
            self.issue([q,q],'release',-1,hold=8,reserve=180)
            self.observe(before_hand=np.asarray(self.state['metadata']['eef_pose_world_xyzw'][:3]))
            self.current_choice=None
            self.generated=self.resources.grasps(self.cloud,folder/'grasps')
        self.sim.execution_calls=0;self.sim.depth_hand=None
        self.stage='contact_planning'
        try:
            chosen,approach=select_contact(self.generated,self.cloud,self.spec,self.scene,self.record,self.ik,
                self.state,self.sim,folder,self.config,strategy=decision)
        except ValueError:
            feedback=planner_feedback(folder)
            self.latest_pose=feedback['last_proposed_pose_world_xyzw']
            return {'status':'planning_failed','phase':'contact','manipulation_executed':False,**feedback}
        self.current_choice={'contact_mode':decision['contact_mode'],'approach_tilt_deg':decision['approach_tilt_deg']}
        self.latest_pose=chosen['tcp_pose_world_xyzw'];write_json(folder/'chosen-grasp.json',chosen)
        self.stage='approach';self.issue(approach,'approach',-1,hold=20,reserve=180)
        self.stage='contact';q=np.asarray(chosen['solution_joints']);self.issue([q,q],'contact',1,hold=40,reserve=140)
        self.observe()
        result=self.motion(decision['motion_extent'],folder)
        return {**result,'selected_strategy':self.current_choice,'contact_retention_inferred_from_motion_only':True}

    def dispatch(self,decision,folder):
        op=decision['operation']
        if op=='try_contact':return self.try_contact(decision,folder)
        if op=='continue_motion':return self.motion(decision['motion_extent'],folder)
        if op=='reobserve':return {'status':'observation_updated','manipulation_executed':False,**self.observe()}
        return {'status':'agent_finished','manipulation_executed':False,'native_success_used':False}


def run_episode(task,index,folder,config,resources,policy,*,maximum_rounds=5):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    tools_class=StrategyTools
    if config.get('strategy_tool_version')=='v2':
        from .recovery_episode import RecoveryTools
        tools_class=RecoveryTools
    if config.get('strategy_tool_version')=='v3':
        from .region_episode import RegionTools
        tools_class=RegionTools
        if config.get('separate_camera_inputs',False):
            from .region_multiview import SeparateRegionTools
            tools_class=SeparateRegionTools
    elif config.get('collision_evidence_views',False):
        from .region_episode import EvidenceTools
        tools_class=EvidenceTools
    session=tools_class(task,index,folder,config,resources);started=time.perf_counter();before_calls=resources.new_api_calls
    result={'task':task['label'],'initial_state':index,'official_task_name':task['name'],'suite':task['suite'],
        'status':'failed','official_task_success':False,'seed':0,'maximum_decision_rounds':maximum_rounds,
        'execution_veto_checks':0,'execution_time_collision_checks':False,'execution_time_progress_vetoes':False,
        'planning_scene_admission_enabled':True,'physical_contacts_enabled':True,'assisted_attachment':False,
        'native_state_used_for_control':False,'automatic_contact_mode_switch':False,'scene_resets':1}
    try:
        session.prepare()
        for number in range(maximum_rounds):
            if session.state['control_steps']>=600:break
            current=folder/f'decisions/round-{number:02d}';current.mkdir(parents=True)
            metadata={**session.state['metadata'],'control_steps':session.state['control_steps']}
            packet=feedback_packet(session.spec['instruction'],metadata,session.spec,session.progress,session.geometry,
                session.history,round_index=number,maximum_rounds=maximum_rounds,goal_reached=session.goal_reached,
                active_contact=session.current_choice)
            if hasattr(session,'extend_feedback'):packet=session.extend_feedback(packet)
            save_packet(packet,current)
            image=current/'feedback-sheet.png'
            if hasattr(session,'render_feedback'):session.render_feedback(image)
            else:render_feedback(session.previous_views,session.views,session.cloud,session.spec,metadata,image,pose=session.latest_pose)
            session.stage='strategy_decision'
            episode_key=hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:20]
            decision=policy.choose(packet,image,current,f"strategy-{config.get('strategy_tool_version','v1')}-{episode_key}-round-{number:02d}")
            print({'event':'strategy','case':folder.name,'round':number,'operation':decision['operation'],
                   'mode':decision['contact_mode'],'tilt':decision['approach_tilt_deg']},flush=True)
            try:feedback=session.dispatch(decision,current)
            except ValueError as error:
                if 'budget' not in str(error).lower():raise
                feedback={'status':'tool_budget_exhausted','phase':session.stage,
                          'reason':'The requested complete motion cannot fit in the remaining control budget',
                          'manipulation_executed':False}
            session.history.append({'decision':decision,'result':feedback})
            write_json(current/'tool-result.json',feedback);write_json(folder/'strategy-history.json',session.history)
            if decision['operation']=='finish':break
        result.update(status='decision_loop_finished',estimated_goal_reached=session.goal_reached)
        session.sim.observe(folder/'final')
    except Exception as error:
        result.update(failure_stage=session.stage,failure=str(error));(folder/'error.txt').write_text(traceback.format_exc())
    finally:
        if session.ik:session.ik.close()
        if session.sim:session.sim.close()
        clear_native_cache()
        actions=read_json(folder/'initial/actions.json') if (folder/'initial/actions.json').exists() else []
        result.update(official_task_success=any(a['evaluation_task_success'] for a in actions),
            final_task_success=bool(actions and actions[-1]['evaluation_task_success']),control_steps=len(actions),
            all_issued_trajectories_completed=all(p['all_commands_issued'] for p in session.phases),
            completed_trajectories=len(session.phases),decisions=len(session.history),strategy_api_calls=policy.calls,
            grounding_api_calls=resources.new_api_calls-before_calls,episode_wall_seconds=time.perf_counter()-started)
        if result['official_task_success']:result['status']='task_success'
        elif 'failure' not in result:
            result['failure']='Native task predicate not reached; no additional retries after the declared decision/control budget'
        write_json(folder/'tracking.json',session.events);write_json(folder/'summary.json',result)
        result['storage']=compact_episode(folder);write_json(folder/'summary.json',result)
        resources.cleanup_proposals()
    return result
