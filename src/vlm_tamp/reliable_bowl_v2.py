"""Timed physical trials for existing bowl, drawer and control adapters.

All collision admission happens while preparing a complete segment. Execution
uses the shared reference-time joint servo, with no perception/retention veto.
The microwave adapter remains MC-VAMP; these adapters use cuRobo IK continuation
and sampled observed-surface/payload admission, explicitly reported separately.
"""
import hashlib
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .coupled_client import CoupledClient
from .coupled_storage import read_json,write_json,compact_episode
from .timed_execution import execute,schedule
from .rekep_libero.geometry import pose_matrix,transform_points
from try_rekep_bowl_v3 import read_observation_v3
from .rekep_libero.program import parse_program
from .rekep_libero.task_geometry_v4 import referenced_points,bind_bowl_goal
from .rekep_libero.task_geometry_v5 import compile_scene
from .rekep_libero.task_ik_curobo import TaskIK
from .rekep_libero.task_ik_v5 import manipulation_poses
from .rekep_libero.physical_tasks_v4 import candidate_variants
from .rekep_libero.grasp_selection_curobo import admit_posture
from .rekep_libero.mechanism_planning import choose_grasp as choose_mechanism,mechanism_path_check,ranked_postures
from .rekep_libero.mechanism_geometry import ground_geometry,refine_contact,expand_handle_surface,observe_progress,Mechanism
from .rekep_libero.scene_state_v11 import ObservedScene as OriginalScene
from .rekep_libero.contact_policy_v11 import ContactPolicy as OriginalPolicy
from .rekep_libero.mechanism_scene import MechanismScene
from .rekep_libero.collision_paths_v7 import check_joint_path,bowl_transform,drawer_transform
from .rekep_libero.object_motion_v3 import observed_object_cloud,verify_observed_transport
from .rekep_libero.object_tracking_v4 import register_closed_surface

ROOT=Path(__file__).resolve().parents[2]


class PlacementContactPolicy(OriginalPolicy):
    def __init__(self,volumes,spec,config,support):
        super().__init__(volumes,spec,config)
        self.selected_support=np.asarray(support,float).reshape(-1,3)
    def local_panel_allowed(self,points,volume,contact_points,stage,transform):
        result=super().local_panel_allowed(points,volume,contact_points,stage,transform)
        if (self.spec['task_type']=='pick_place' and stage in {'place','release','retreat_contact'}
                and volume['robot_geom_name'] in self.finger_names and len(self.selected_support)):
            result|=cKDTree(self.selected_support).query(np.asarray(points).reshape(-1,3))[0]<=self.support_tolerance
        return result
    def record(self,task_type):
        result=super().record(task_type)
        result['finger_selected_destination_contact_during_placement']=task_type=='pick_place'
        result['arm_palm_or_unrelated_surface_exempt']=False
        return result


class ObservedScene(OriginalScene):
    def __init__(self,cameras,metadata,*args,**kwargs):
        super().__init__(cameras,metadata,*args,**kwargs)
        self.policy=PlacementContactPolicy(metadata['robot_collision_volumes_world'],self.region.spec,self.config,self.target_points)
    def refresh(self,cameras,metadata,transform):
        result=super().refresh(cameras,metadata,transform)
        if self.region.task_type!='pick_place':return result
        from .rekep_libero.scene_state_v7 import hull_volume,voxel_points
        # Occlusion can split one bowl into disconnected visible patches.
        # Bind current depth to the tracked initial surface, not to a robot sweep.
        seed=transform_points(self.tracking_reference,transform)
        points=self.static_points
        near=cKDTree(seed).query(points)[0]<.012
        near&=points[:,2]>self.region.support_height+.0015
        if len(self.target_points):near&=cKDTree(self.target_points).query(points)[0]>self.policy.support_tolerance
        observed=points[near]
        if len(observed):
            canonical=transform_points(observed,np.linalg.inv(pose_matrix(transform)))
            self.moving_reference=voxel_points(np.vstack([self.moving_reference,canonical]),self.spacing)
            self.region.contact_reference=self.moving_reference.copy()
            self.payload_volume=hull_volume(self.moving_reference,'predicted_moving_source')
            self.static_points=points[~near];self.static_tree=cKDTree(self.static_points)
        self.disconnected_surface_reconciliation={'reassigned_current_depth_points':len(observed),
            'tracked_initial_surface_distance_m':.012,'selected_plate_excluded':True,
            'table_plane_excluded':True,'robot_sweep_used':False,'simulator_instance_ids_used':False}
        result['disconnected_surface_reconciliation']=self.disconnected_surface_reconciliation
        return result
    def record(self):
        result=super().record();result['disconnected_surface_reconciliation']=getattr(self,'disconnected_surface_reconciliation',None)
        return result
    def check(self,*args,**kwargs):
        result=super().check(*args,**kwargs)
        if self.region.task_type=='pick_place':
            for item in result.get('contact_permissions_applied',[]):
                if item['rule']=='finger_local_panel_support':item['rule']='finger_selected_placement_support'
        return result


def read_cohort_observation(path,metadata):
    """Keep synchronized robot-only depth priors for scene construction."""
    return read_observation_v3(Path(path).parent,metadata)


def legacy_grounding(task,index,observation,config,output):
    """Rebind cached RGB roles to current depth after bounded evidence checks.

    Old full raw snapshots were intentionally pruned. Retained RGB, 3-D
    landmarks and selected surface arrays suffice for independent verification.
    """
    source=ROOT/f'results/rekep-libero-v2/libero_goal-{task:02d}-state{index:03d}-seed0'
    cameras,metadata=read_cohort_observation(observation/'obs.npz',observation/'metadata.json')
    main=read_json(source/'perception/perception.json')['main_camera']
    with np.load(source/'perception/keypoints.npz') as data:bundle={k:data[k].copy() for k in data.files}
    old=np.asarray(Image.open(source/'perception/rgb.png'));new=cameras[main]['rgb'];world=cameras[main]['world_points']
    mask=np.load(source/'stage1-grasps/selected_mask.npy')
    task_type=read_json(source/'program/metadata.json')['task_type']
    program=parse_program((source/'program/output_raw.txt').read_text(),len(bundle['points_world']),task_type,attachment_cost=lambda i:1.)
    ids=sorted(set(program.grasp_keypoints+referenced_points(program.stages[1]))-{-1})
    color=np.abs(old[mask].astype(float)-new[mask].astype(float))
    pixel=bundle['pixels_yx'][ids];points=world[tuple(pixel.T)]
    delta=float(np.max(np.linalg.norm(points-bundle['points_world'][ids],axis=1)))
    prior=ROOT/'results/physical-tasks-v6-eval'/('bowl' if task==8 else 'drawer')/f'goal{task:02d}-state{index:03d}'
    reference=np.load(prior/'source-cloud.npy');current=world[mask];current=current[np.isfinite(current).all(1)]
    surface_delta=float(np.quantile(cKDTree(reference).query(current)[0],.99))
    evidence={'rgb_mean':float(color.mean()),'rgb_max':float(color.max()),'landmark_max_delta_m':delta,
        'source_surface_p99_delta_m':surface_delta,'new_api_calls':0,'manual_role_rebinding':False,
        'source':str(source),'depth_reference':str(prior/'source-cloud.npy')}
    write_json(output/'cache-evidence.json',evidence)
    if (color.mean()>config['cache_rgb_mean_channel_tolerance'] or color.max()>config['cache_rgb_max_channel_tolerance']
            or delta>config['cache_world_tolerance_m'] or surface_delta>config['cache_world_tolerance_m']):
        raise ValueError('Cached RGB roles do not match the fresh selected RGB-D evidence')
    bundle.update(cameras=cameras,meta=metadata,main_camera=main,points_world=world[tuple(bundle['pixels_yx'].T)])
    shutil.copytree(source/'program',output/'cached-program')
    shutil.copyfile(source/'perception/numbered.png',output/'numbered.png')
    return bundle,program,current,mask


def limited_proposals(generated,calibration,anchor,maximum):
    """Bound the selection search, preserving all original model candidates."""
    distance={}
    for index,variant,pose in candidate_variants(generated,calibration):
        d=float(np.linalg.norm(np.asarray(pose)[:3]-anchor))
        distance[index]=min(distance.get(index,float('inf')),d)
    indices=sorted(distance,key=lambda i:(distance[i],-float(generated['scores'][i]),i))[:maximum]
    return {**generated,'retained_indices':np.asarray(indices,int)}


def select_legacy(generated,cloud,spec,calibration,ik,state,sim,scene,output,config):
    anchor=np.asarray(spec['source_anchor_world_m']);tree=cKDTree(cloud);eligible=[];rejected=[]
    selected=limited_proposals(generated,calibration,anchor,config['maximum_candidate_grasps'])
    for index,variant,pose in candidate_variants(selected,calibration):
        distance=float(np.linalg.norm(pose[:3]-anchor));row={'index':index,'variant':variant,
            'score':float(generated['scores'][index]),'keypoint_distance_m':distance,'tcp_pose_world_xyzw':pose.tolist()}
        if distance>.10 or tree.query(pose[:3])[0]>.025:continue
        if spec['task_type']=='open_middle_drawer':
            R=pose_matrix(pose)[:3,:3]
            if -R[:,2]@spec['pull_axis_world']<.65 or abs(R[:,0]@spec['handle_long_axis_world'])>.6:continue
        eligible.append(row)
    eligible.sort(key=lambda r:(r['keypoint_distance_m'],-r['score']))
    batches=[];started=time.perf_counter();tested=0
    for ordinal,row in enumerate(eligible):
        if time.perf_counter()-started>config['maximum_grasp_search_seconds']:break
        solutions=ik.solve_many(row['tcp_pose_world_xyzw'],state['metadata']['joint_positions'],maximum=8)
        batches.append((ordinal,row,solutions))
    chosen=None
    for ordinal,row,rank,grasp in ranked_postures(batches):
        if tested>=config['maximum_total_ik_postures'] or time.perf_counter()-started>config['maximum_grasp_search_seconds']:break
        tested+=1
        admitted,failure=admit_posture(grasp,spec,ik,state['metadata']['joint_positions'],sim,scene,state['metadata']['gripper_qpos'])
        if failure:rejected.append({**row,'ik_rank':rank,**failure})
        else:
            chosen={**row,**admitted,'ik_rank':rank,'tested_postures':tested,'fixed_grasp_replay':False};break
        write_json(output/'grasp-search-progress.json',{'tested':tested,'eligible':len(eligible),'seconds':time.perf_counter()-started})
    write_json(output/'grasp-admission.json',{'selected':chosen,'rejected':rejected,'tested':tested,'eligible':len(eligible),
        'detector_retained':len(generated['retained_indices']),'candidate_budget':config['maximum_candidate_grasps']})
    if chosen is None:raise ValueError('Bounded fresh grasp/IK search found no admitted approach and manipulation')
    return chosen


def run_episode(task,index,folder,config,resources):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    sim=CoupledClient(folder/'simulator.log');ik=None;state=None;events=[];phases=[];observations=[];stage='reset'
    started=time.perf_counter();calls_before=resources.new_api_calls
    result={'task':task['label'],'official_task_name':task['name'],'suite':task['suite'],'initial_state':index,'seed':0,
        'status':'failed','official_task_success':False,'new_api_calls':0,'fixed_grasp_replay':False,
        'planner':'cuRobo multi-IK continuation + sampled observed-scene admission',
        'sampling_search_planner_used':False,'execution_veto_checks':0,'assisted_attachment':False,
        'execution_time_collision_checks':False,'execution_time_constraint_checks':False,
        'execution_time_progress_vetoes':False,'physical_contacts_enabled':True}
    def issue(path,label,gripper=1,hold=6,reserve=0):
        nonlocal state
        path=np.asarray(path,float)
        if len(path)==1:path=np.vstack([path,path])
        references=schedule(path,maximum_joint_speed=.35,endpoint_hold_steps=hold)
        available=600-int(state['control_steps'])-reserve
        if available<=hold:raise ValueError('Insufficient episode budget for another complete segment')
        if len(references)>available:
            speed=.35*(len(references)-hold)/(available-hold)*1.001
            references=schedule(path,maximum_joint_speed=speed,endpoint_hold_steps=hold)
        destination=folder/f'path-{len(phases):02d}-{label}';destination.mkdir()
        np.save(destination/'path.npy',path);np.save(destination/'references.npy',references)
        state,done=execute(sim,state,references,gripper=gripper,event_sink=events.append)
        phases.append({'label':label,**{k:v for k,v in done.items() if k!='events'}})
        write_json(folder/'phases.json',phases)
    try:
        state=sim.reset(task['name'],index,folder/'initial',suite_name=task['suite'])
        state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,output_dir=folder/'opened')
        stage='rgb_grounding'
        legacy='legacy_index' in task
        if legacy:
            bundle,program,source,mask=legacy_grounding(task['legacy_index'],index,folder/'opened',config,folder)
            spec,cloud,target,target_mask,masks=compile_scene(bundle,program,source,mask,config)
            cameras=bundle['cameras'];mechanism=None
        else:
            choices=resources.grounding(task['label'],index,folder/'opened',state['metadata'],folder/'grounding')
            from .rekep_libero.mechanism_execution import cached_grounding
            # The prepared grounding already holds the matched current bundle.
            cameras,metadata=read_cohort_observation(folder/'opened/obs.npz',state['metadata']);name=choices['camera']
            with np.load(folder/'grounding'/name/'keypoints.npz') as arrays:bundle={k:arrays[k].copy() for k in arrays.files}
            bundle.update(cameras=cameras,meta={**metadata,'main_camera':name},main_camera=name)
            stage='numerical_geometry'
            cloud,mask=refine_contact(bundle,choices,resources.ensure_perception(),folder)
            mechanism,panel_mask=ground_geometry(bundle,choices,cloud,mask,resources.ensure_perception(),folder)
            cloud=expand_handle_surface(cameras,mechanism.spec,choices,cloud)
            mechanism.spec['source_anchor_world_m']=np.median(cloud,axis=0).tolist();spec=mechanism.spec
            target=None;masks={'source':mask}
        write_json(folder/'task-spec.json',spec);np.save(folder/'grasp-cloud.npy',cloud)
        result['geometric_path_constraints']=([] if spec['task_type']=='pick_place' else
            ['point_on_line' if legacy or mechanism.kind=='prismatic' else 'point_on_circle_with_coupled_orientation'])
        stage='robot_and_scene_geometry'
        record=sim.robot_chain();write_json(folder/'robot-chain.json',record)
        import vamp
        empty=vamp.Environment();self_valid=lambda q:vamp.panda.validate(np.asarray(q).tolist(),empty,True)
        ik=TaskIK(record,config,self_valid,output=folder,sim=sim)
        if legacy:
            scene=ObservedScene(cameras,state['metadata'],spec,cloud,masks['source'],bundle['main_camera'],masks.get('parent'),target,config)
        else:scene=MechanismScene(cameras,state['metadata'],spec,cloud,panel_mask,config)
        scene.save(folder/'initial-scene.npz');write_json(folder/'scene-scope.json',scene.record())
        stage='grasp_generation'
        # Feed GraspGen the complete fused observed body, not only the source-view patch.
        grasp_input=scene.moving_reference if spec['task_type']=='pick_place' else cloud
        np.save(folder/'grasp-input-cloud.npy',grasp_input)
        generated=resources.grasps(grasp_input,folder/'grasps')
        write_json(folder/'grasp-input-provenance.json',{'source':'fused RGB-D selected moving body',
            'source_view_points':len(cloud),'detector_input_points':len(grasp_input),'simulator_object_geometry_used':False})
        stage='fresh_grasp_selection'
        if legacy:chosen=select_legacy(generated,cloud,spec,state['metadata']['robot_calibration'],ik,state,sim,scene,folder,config)
        else:
            limited=limited_proposals(generated,state['metadata']['robot_calibration'],np.asarray(spec['source_anchor_world_m']),config['maximum_candidate_grasps'])
            chosen=choose_mechanism(limited,cloud,state['metadata']['robot_calibration'],ik,state['metadata']['joint_positions'],
                sim,scene,mechanism,state['metadata']['gripper_qpos'],folder,write_json)
        write_json(folder/'chosen-grasp.json',chosen)
        state=sim.enable_joint_tracking(kp=300.);write_json(folder/'controller-switch.json',state['controller_change'])
        stage='timed_approach';issue(chosen['approach']['joint_postures'],'approach',-1,hold=20,reserve=100)
        stage='physical_closure';q=np.asarray(chosen['solution_joints']);issue([q,q],'close',1,hold=40,reserve=80)
        state=sim.observe(folder/'closed');closed_pose=np.asarray(state['metadata']['eef_pose_world_xyzw']);closed_q=np.asarray(state['metadata']['joint_positions'])
        cameras,_=read_cohort_observation(folder/'closed/obs.npz',state['metadata'])
        observed=observed_object_cloud(cameras,cloud,np.eye(4),state['metadata']['robot_collision_volumes_world'])
        closure=register_closed_surface(cloud,observed);write_json(folder/'closure-registration.json',closure)
        closure_transform=np.asarray(closure['transform_initial_to_closed_world']) if closure.get('verified') else np.eye(4)
        if not closure.get('verified'):observations.append({'stage':'closure','measurement_unavailable':True,'execution_veto':False})
        try:scene.refresh(cameras,state['metadata'],closure_transform)
        except ValueError as error:observations.append({'stage':'closure_scene_refresh','reason':str(error),'retained_previous_scene':True})
        if legacy and task['legacy_index']==8:
            support=transform_points([spec['object_support_point_initial_world_m']],closure_transform)[0]
            local=transform_points([support],np.linalg.inv(pose_matrix(closed_pose)))[0]
            goal=np.asarray(spec['goal']['target_world_m']);lift=support+[0,0,config['lift_distance_m']]
            over=goal.copy();over[2]=max(lift[2],goal[2]+config['lift_distance_m'])
            knots=[support,lift,over,goal]
            transform=bowl_transform(closed_pose,closure_transform)
            for label,a,b in zip(['lift','transport','place'],knots[:-1],knots[1:]):
                stage='planning_'+label
                scene.save(folder/(label+'-planning-scene.npz'))
                points=np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/.015))+1))
                planned=ik.point_path(points,local,state['metadata']['joint_positions'])
                if not planned['accepted']:raise ValueError('No bounded IK path for '+label)
                qs=np.vstack([state['metadata']['joint_positions'],[s['joints'] for s in planned['solutions']]])
                checked=check_joint_path(scene,sim,ik,qs,gripper_qpos=state['metadata']['gripper_qpos'],
                    transform_at_pose=transform,stage_at_pose=lambda p,l=label:'place' if l=='place' else 'carry',check_payload=True,config=config)
                write_json(folder/(label+'-planning.json'),{'ik':planned,'collision':checked})
                if not checked['accepted']:raise ValueError('Observed robot/payload collision in planned '+label)
                stage='timed_'+label;issue(qs,label,hold=8,reserve=65)
                if label=='lift':
                    state=sim.observe(folder/'lifted');views,_=read_cohort_observation(folder/'lifted/obs.npz',state['metadata'])
                    motion=transform(state['metadata']['eef_pose_world_xyzw'])
                    surface=observed_object_cloud(views,cloud,motion,state['metadata']['robot_collision_volumes_world'])
                    retention=verify_observed_transport(cloud,surface,motion,state['metadata']['gripper_qpos'])
                    write_json(folder/'grasp-retention.json',retention);result['grasp_retention_verified']=retention['verified']
            stage='physical_release';q=np.asarray(state['metadata']['joint_positions']);issue([q,q],'release',-1,hold=40,reserve=12)
            stage='release_settle';issue([q,q],'settle',-1,hold=10)
        else:
            if legacy:
                mechanism=Mechanism({'joint_family':'prismatic','joint_axis_world':spec['pull_axis_world'],
                    'joint_origin_world_m':spec['source_anchor_world_m'],'target_progress':config['drawer_maximum_command_travel_m']})
            chunk=.025 if mechanism.kind=='prismatic' else np.radians(8.)
            count=int(np.ceil(mechanism.goal/chunk));previous=0.
            for number,end in enumerate(np.linspace(0,mechanism.goal,count+1)[1:]):
                stage='planning_mechanism_segment'
                scene.save(folder/f'mechanism-planning-scene-{number:02d}.npz')
                target_poses=mechanism.path(closed_pose,start=previous,end=float(end))
                planned=ik.path(target_poses[1:],state['metadata']['joint_positions'])
                if not planned['accepted']:raise ValueError('No bounded IK continuation along grounded mechanism')
                qs=np.vstack([state['metadata']['joint_positions'],[p['joints'] for p in planned['solutions']]])
                checked=mechanism_path_check(scene,sim,ik,qs,mechanism,closed_pose,state['metadata']['gripper_qpos'],closure_transform)
                write_json(folder/f'mechanism-plan-{number:02d}.json',{'ik':planned,'collision_and_constraint':checked})
                if not checked['accepted']:raise ValueError('New complete mechanism segment fails planning-time collision/constraint admission')
                stage='timed_mechanism';issue(qs,f'manipulate-{number:02d}',hold=6)
                state=sim.observe(folder/f'observation-{number:02d}');views,_=read_cohort_observation(folder/f'observation-{number:02d}/obs.npz',state['metadata'])
                hand=mechanism.progress_of_pose(closed_pose,state['metadata']['eef_pose_world_xyzw'])
                measured=observe_progress(transform_points(cloud,closure_transform),mechanism,views,
                    state['metadata']['robot_collision_volumes_world'],max(0.,min(hand,mechanism.goal+.07)),previous)
                observations.append({'control_step':state['control_steps'],'commanded_progress':float(end),
                    'measurement':measured,'execution_veto':False})
                progress=float(measured['observed_progress']) if measured.get('tracking_valid') else float(end)
                progress=float(np.clip(progress,-.025,mechanism.goal+.09))
                try:scene.refresh(views,state['metadata'],mechanism.transform(progress)@closure_transform)
                except ValueError as error:observations[-1]['scene_refresh_failed']=str(error)
                previous=float(end)
                write_json(folder/'observations.json',observations)
        result['status']='completed_planned_trajectories';sim.observe(folder/'final')
    except Exception as error:
        result.update(failure_stage=stage,failure=str(error));(folder/'error.txt').write_text(traceback.format_exc())
    finally:
        if ik:ik.close()
        sim.close()
        actions=read_json(folder/'initial/actions.json') if (folder/'initial/actions.json').exists() else []
        result.update(official_task_success=any(a['evaluation_task_success'] for a in actions),
            final_task_success=bool(actions and actions[-1]['evaluation_task_success']),control_steps=len(actions),
            completed_trajectories=len(phases),all_issued_trajectories_completed=all(p['all_commands_issued'] for p in phases),
            new_api_calls=resources.new_api_calls-calls_before,episode_wall_seconds=time.perf_counter()-started,
            pipeline_status=result['status'])
        if result['official_task_success']:result['status']='task_success'
        write_json(folder/'tracking.json',events);write_json(folder/'observations.json',observations)
        write_json(folder/'summary.json',result);result['storage']=compact_episode(folder);write_json(folder/'summary.json',result)
    return result
