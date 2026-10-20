"""Cached VLM grounding -> whole-motion Atlas planning -> guarded joint tracking."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from try_rekep_bowl_v3 import save,read_observation_v3
from vlm_tamp.coupled_client import CoupledClient
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.coupled_manifold import CoupledHandlePoint
from vlm_tamp.coupled_local_projection import project_nearby
from vlm_tamp.coupled_tracking import phase_arrival
from vlm_tamp.coupled_resources import CoupledResources,api_snapshot,fresh_task_template
from vlm_tamp.coupled_grasp_capture import observed_pad_capture
from vlm_tamp.coupled_storage import POLICY as STORAGE_POLICY,compact_episode,write_json
from vlm_tamp.coupled_pipeline import terminal_angle,complete_motion_selection,isolated_plan,supported_handle_anchors,closing_normal_residual,validated_grasp_reference,handle_anchor_menu,partial_goal_candidates
from vlm_tamp.handle_contact import ground_handle,require_arrival,translated_surface,fit_upright_hinge,upright_plane
from vlm_tamp.rekep_libero.geometry import pose_matrix,matrix_pose,transform_points
from vlm_tamp.rekep_libero.mechanism_geometry import project
from vlm_tamp.rekep_libero.mechanism_execution import cached_grounding
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pad_local(metadata):
    inverse=np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']))
    return np.mean([transform_points(v['vertices_world'],inverse).mean(0)
        for v in metadata['robot_collision_volumes_world'] if v['robot_geom_name'].endswith('_pad_collision')],axis=0)


def current_progress(problem,q):
    matrix=pose_matrix(problem.chain.forward(q))
    point=matrix[:3,3]+matrix[:3,:3]@problem.pad_local
    first=problem.reference_anchor-problem.origin;now=point-problem.origin
    first-=problem.axis*(first@problem.axis);now-=problem.axis*(now@problem.axis)
    return float(np.arctan2(problem.axis@np.cross(first,now),first@now))


def observed_angle(origin,axis,first,last):
    a=np.asarray(first)-origin;b=np.asarray(last)-origin
    a=a-axis*(a@axis);b=b-axis*(b@axis)
    return float(np.arctan2(axis@np.cross(a,b),a@b))


def world_pad(metadata):
    pose=pose_matrix(metadata['eef_pose_world_xyzw'])
    return pose[:3,3]+pose[:3,:3]@pad_local(metadata)


def fresh_mask(cameras,spec,cloud,old_scene=None,theta=0.):
    view=cameras[spec['camera']];world=np.asarray(view['world_points'])
    signed=np.abs((world-np.asarray(spec['front_plane']['point_world_m']))@np.asarray(spec['panel_normal_world']))
    mask=np.isfinite(world).all(2)&(signed<.008)
    if old_scene is not None:
        pixels=project(transform_points(old_scene.moving_reference,old_scene.transform(theta)),view)
        pixels=pixels[np.isfinite(pixels).all(1)]
        low,high=np.quantile(pixels,[.01,.99],axis=0)
        yy,xx=np.indices(mask.shape);mask&=(xx>=low[0]-4)&(xx<=high[0]+4)&(yy>=low[1]-4)&(yy<=high[1]+4)
        choices={'camera':spec['camera'],'moving_panel_box_xyxy':[float(low[0]-4),float(low[1]-4),float(high[0]+4),float(high[1]+4)]}
    else:choices=None
    return mask,choices


class PlanningFailure(ValueError):pass
class ModeBindingFailure(ValueError):pass


def fixed_contact_plan(ik,state,cameras,spec,cloud,mask,choices,goal,folder,config,static_prior=None,warm_path=None,contact_mode='aligned',require_valid_goal=False):
    folder.mkdir(parents=True,exist_ok=False)
    save(folder/'request.json',{'task_spec':spec,'choices':choices,'goal_angle_rad':goal,
        'metadata':state['metadata'],'configuration':config,'contact_mode':contact_mode})
    np.save(folder/'handle-cloud.npy',cloud);np.save(folder/'panel-mask.npy',mask)
    robot=RobotVolumes(ik.chain,state['metadata']['robot_collision_volumes_world'],state['metadata']['joint_positions'])
    panel_prior=static_prior.panel_prior_for(spec,cloud) if static_prior is not None else None
    scene=Scene(cameras,state['metadata'],spec,cloud,mask,choices,config,robot=robot,
                panel_completion_prior=panel_prior)
    if static_prior is not None:scene.retain_static(static_prior)
    local_pad=pad_local(state['metadata'])
    pad=world_pad(state['metadata'])
    # Validate the selected bar and finite pad support, then preserve the grip offset.
    if contact_mode=='aligned':
        try:_,support=supported_handle_anchors([pad],cloud,state['metadata'])
        except ValueError as error:raise ModeBindingFailure('Aligned finite-pad support unavailable: '+str(error)) from error
    else:support={'full_bar_height_support_certified':False,'source':'bounded-tilt observed capture and continued retention'}
    capture=observed_pad_capture(state['metadata'],cloud,spec.get('observed_handle_radius_m',.018),axis=spec['joint_axis_world'])
    anchor,attachment=validated_grasp_reference(pad,spec['source_anchor_world_m'],
        maximum_distance=capture['maximum_axis_distance_m'])
    observed_axis=np.asarray(attachment['observed_axis_point_world_m'])
    binding_error=attachment['observed_pad_to_handle_distance_m']
    problem=CoupledHandlePoint(ik.chain,spec['joint_origin_world_m'],spec['joint_axis_world'],
        pose_matrix(state['metadata']['eef_pose_world_xyzw']),pad_local=local_pad,reference_anchor=anchor,
        align_axis=False,jaw_axis_local=[1.,0.,0.] if contact_mode=='aligned' else None,
        jaw_tilt_limit=None if contact_mode=='aligned' else config.get('maximum_jaw_tilt_rad',np.pi/3),
        orientation_cone=np.pi/2,progress_bounds=(-.03,goal+.03),joint_margin=.025,
        position_tolerance=.001,orientation_tolerance=.01)
    save(folder/'inputs.json',{'task_spec':spec,'choices':choices,'goal_angle_rad':goal,
        'metadata':state['metadata'],'configuration':config,'pad_local':local_pad,
        'observed_anchor_world_m':observed_axis,'reference_anchor_world_m':anchor,
        'contact_mode':contact_mode,'constraint_codimension':problem.codimension,
        'jaw_tilt_limit_rad':getattr(problem,'jaw_tilt_limit',None),
        'grasp_attachment':attachment,'finite_pad_support':support,'observed_grasp_capture':capture,'state_bounds':problem.bounds})
    np.save(folder/'handle-cloud.npy',cloud);np.save(folder/'panel-mask.npy',mask)
    start=project_nearby(problem,np.r_[state['metadata']['joint_positions'],0.],fixed_progress=0.)
    correction=float(np.max(np.abs(start['state'][:7]-state['metadata']['joint_positions'])))
    save(folder/'binding.json',{'observed_pad_to_handle_distance_m':binding_error,
        'measured_to_projected_start_max_joint_correction_rad':correction,
        'local_projection':{k:v for k,v in start.items() if k!='state'},
        'measured_pad_world_m':pad,'observed_anchor_world_m':observed_axis,
        'grasp_attachment':attachment,'finite_pad_support':support,'observed_grasp_capture':capture})
    if not start['accepted']:raise ModeBindingFailure('Measured grasp cannot bind the maintained-contact manifold: '+start['reason'])
    if correction>.04:raise ValueError('Measured grasp needs a substantial unplanned start projection')
    for w in np.linspace(0,1,8):
        q=np.asarray(state['metadata']['joint_positions'])+w*(start['state'][:7]-state['metadata']['joint_positions'])
        if not ik.self_valid(q) or not scene.check(q,0.)['accepted']:
            raise ValueError('Measured-to-projected contact connector is not clear')
    save(folder/'scene.json',scene.record())
    goal_state=np.r_[start['state'][:7],goal]
    if warm_path is not None and len(warm_path)>1:
        path,result=isolated_plan(problem,start['state'],np.r_[start['state'][:7],goal],scene,
            folder/'warm-start',save,seconds=60.,strategy='warmstart',warm_path=warm_path,
            warm_maximum_joint_correction_rad=.2)
        if result.get('exact_solution') and len(path)>1:
            np.save(folder/'path.npy',path);save(folder/'result.json',result)
            return problem,scene,path
        # These are hypothetical references after a model update. The measured
        # binding above keeps its separate 0.04 rad limit and checked connector.
        endpoint=project_nearby(problem,warm_path[-1],fixed_progress=goal,
                               maximum_joint_correction_rad=.2)
        endpoint_valid=(endpoint['accepted'] and endpoint['weighted_residual_norm']<=1e-5
                        and ik.self_valid(endpoint['state'][:7])
                        and scene.check(endpoint['state'][:7],goal)['accepted'])
        save(folder/'goal-reuse.json',{'valid':endpoint_valid,
            'hypothetical_proposal_joint_correction_limit_rad':.2,
            'projection':{k:v for k,v in endpoint.items() if k!='state'}})
        if endpoint_valid:
            goal_state=endpoint['state']
            path,result=isolated_plan(problem,start['state'],goal_state,scene,
                folder/'global-atlas',save,seconds=config.get('maximum_global_plan_seconds',60.),strategy='atlas')
            if result.get('exact_solution') and len(path)>1:
                np.save(folder/'path.npy',path);save(folder/'result.json',result)
                return problem,scene,path
    if (not problem.satisfied(goal_state) or problem.errors(goal_state)['weighted_residual_norm']>1e-5):
        candidates,diagnostics=partial_goal_candidates(problem,start['state'][:7],goal,ik.self_valid,scene)
        save(folder/'partial-goal-proposals.json',{'candidates':candidates,'diagnostics':diagnostics})
        if not candidates and require_valid_goal:raise PlanningFailure('No validated endpoint found for this horizon and contact mode')
        for number,candidate in enumerate(candidates[:2]):
            if config.get('prefer_milestone_plans',False):
                path,result=isolated_plan(problem,start['state'],candidate,scene,
                    folder/f'partial-goal-milestones-{number}',save,seconds=60.,strategy='milestones',seed=1)
                if result.get('exact_solution') and len(path)>1:
                    np.save(folder/'path.npy',path);save(folder/'result.json',result)
                    return problem,scene,path
            path,result=isolated_plan(problem,start['state'],candidate,scene,
                folder/f'partial-goal-atlas-{number}',save,seconds=config.get('maximum_global_plan_seconds',60.),strategy='atlas')
            if result.get('exact_solution') and len(path)>1:
                np.save(folder/'path.npy',path);save(folder/'result.json',result)
                return problem,scene,path
        if candidates:goal_state=candidates[0]
    for seed in [1,2,3]:
        path,result=isolated_plan(problem,start['state'],goal_state,scene,
            folder/f'milestone-seed-{seed}',save,seconds=60.,strategy='milestones',seed=seed)
        if result.get('exact_solution') and len(path)>1:
            np.save(folder/'path.npy',path);save(folder/'result.json',result)
            return problem,scene,path
    raise PlanningFailure('No complete remaining contact trajectory found by constrained planning')


def contact_horizons(total_goal,config):
    maximum=float(config['contact_segment_angle_rad'])
    fallback=float(config.get('contact_fallback_segment_angle_rad',maximum/2))
    if not np.isfinite([total_goal,maximum,fallback]).all() or not 0<fallback<maximum or total_goal<=0:
        raise ValueError('Positive goal and strictly decreasing bounded horizons are required')
    return [float(total_goal),*[value for value in [maximum,fallback] if value<total_goal-1e-8]]


def adaptive_contact_plan(ik,state,cameras,spec,cloud,mask,choices,total_goal,folder,config,static_prior=None,warm_path=None):
    """Prefer a complete task plan; otherwise execute a checked bounded horizon."""
    folder.mkdir(parents=True,exist_ok=False);failures=[]
    horizons=contact_horizons(total_goal,config)
    for horizon in horizons:
        for mode in ['aligned','tilted']:
            attempt=folder/f'{mode}-{len(failures):02d}'
            warm=warm_path.copy() if warm_path is not None and len(warm_path)>1 else None
            if warm is not None and warm[-1,7]>1e-8:warm[:,7]*=horizon/warm[-1,7]
            try:
                problem,scene,path=fixed_contact_plan(ik,state,cameras,spec,cloud,mask,choices,horizon,
                    attempt,config,static_prior,warm,contact_mode=mode,
                    require_valid_goal=horizon>config['contact_segment_angle_rad']+1e-8)
            except (PlanningFailure,ModeBindingFailure) as error:
                failures.append({'horizon_rad':horizon,'contact_mode':mode,'reason':str(error)});continue
            for name in ['path.npy','request.json','inputs.json','binding.json','scene.json','result.json','handle-cloud.npy','panel-mask.npy']:
                shutil.copyfile(attempt/name,folder/name)
            save(folder/'horizon-selection.json',{'full_task_remaining_rad':total_goal,'selected_horizon_rad':horizon,
                'contact_mode':mode,'failed_attempts':failures})
            return problem,scene,path
    save(folder/'horizon-selection.json',{'full_task_remaining_rad':total_goal,'failed_attempts':failures})
    raise PlanningFailure('No checked full or bounded contact plan was found in either grasp mode')


def tracking_reference(ik,scene,problem,path,index,q,theta):
    """Look ahead along the prepared path, checking the short connector."""
    best=None
    for j in range(index,min(len(path),index+16)):
        target=path[j];delta=float(np.max(np.abs(target[:7]-q)))
        if delta>.048 or abs(target[7]-theta)>.10:break
        clear=True
        for w in np.linspace(0.,1.,max(1,int(np.ceil(delta/.01)))+1):
            qs=q+w*(target[:7]-q);angle=theta+w*(target[7]-theta)
            errors=problem.errors(np.r_[qs,angle])
            if (not ik.self_valid(qs) or not scene.check(qs,angle,execution=True)['accepted']
                    or errors['position_error_m']>.012 or errors['jaw_axis_error_rad']>np.deg2rad(2.)
                    or (getattr(problem,'jaw_tilt_limit',None) is not None
                        and errors['measured_jaw_tilt_rad']>problem.jaw_tilt_limit+np.deg2rad(2.))):
                clear=False;break
        if not clear:break
        best=j
    if best is None:raise ValueError('No clear connector to the next planned reference')
    return best


def joint_approach(sim,state,ik,scene,chosen,folder):
    raw=np.asarray(chosen['approach']['joint_postures']);pieces=[]
    for a,b in zip(raw[:-1],raw[1:]):
        n=max(1,int(np.ceil(np.max(np.abs(b-a))/.015)))
        pieces.extend(a+w*(b-a) for w in np.linspace(0,1,n+1)[:-1])
    path=np.vstack([pieces,raw[-1]]);np.save(folder/'approach-joint-path.npy',path)
    index=0;records=[];stalled=0
    while index<len(path):
        if state['control_steps']>=600:raise ValueError('Control budget exhausted during planned approach')
        q=np.asarray(state['metadata']['joint_positions']);best=None
        for j in range(index,min(len(path),index+24)):
            delta=float(np.max(np.abs(path[j]-q)))
            if delta>.048:break
            clear=True
            for w in np.linspace(0,1,max(1,int(np.ceil(delta/.01)))+1):
                qs=q+w*(path[j]-q)
                contact=np.linalg.norm(ik.chain.forward(qs)[:3]-chosen['pose_world_xyzw'][:3])<.02
                if not ik.self_valid(qs) or not scene.check(qs,0.,contact=contact,execution=True)['accepted']:
                    clear=False;break
            if not clear:break
            best=j
        if best is None:raise ValueError('No clear approach connector to the prepared joint path')
        state=sim.joint_action(path[best],gripper=0,max_steps=1)
        q=np.asarray(state['metadata']['joint_positions'])
        contact=np.linalg.norm(np.asarray(state['metadata']['eef_pose_world_xyzw'])[:3]-chosen['pose_world_xyzw'][:3])<.02
        checked=scene.check_volumes(state['metadata']['robot_collision_volumes_world'],0.,contact=contact,execution=True)
        records.append({'control_step':state['control_steps'],'reference':best,'scene_accepted':checked['accepted'],
                        'joint_error_rad':float(np.max(np.abs(q-path[best])))})
        save(folder/'approach-tracking.json',records)
        if not checked['accepted']:
            save(folder/'approach-collision.json',checked);raise ValueError('Measured joint approach lost observed clearance')
        closest=index+int(np.argmin(np.linalg.norm(path[index:best+1]-q,axis=1)))
        if closest>index:index=closest;stalled=0
        elif best==len(path)-1 and np.max(np.abs(q-path[-1]))<.003:index=len(path)
        else:
            stalled+=1
            if stalled>25:raise ValueError('Joint approach stalled')
    return state


def episode(task,folder,config,plan_only=False,state_index=1,resources=None):
    folder.mkdir(parents=True,exist_ok=False)
    sim=CoupledClient(folder/'simulator.log');ik=None;stage='reset'
    tracking=[];observations=[];replans=[]
    resources=resources or CoupledResources()
    summary={'task':task['label'],'initial_state':state_index,'seed':0,
        'development_only':state_index in {0,1},'split':'development' if state_index in {0,1} else 'held_out','new_api_calls':0,
        'hierarchical_vlm_used':False,'whole_motion_preflight_required':False,'each_contact_segment_preflight_required':True,
        'joint_trajectory_tracking':True,'assisted_attachment':False,'official_task_success':False}
    try:
        state=sim.reset(task['name'],state_index,folder/'initial',seed=0)
        state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,output_dir=folder/'opened')
        reference=ROOT/'results/articulated-physics-pilot'/(task['label']+'-state001')
        stage='grounding'
        choices=resources.grounding(task['label'],state_index,folder/'opened',state['metadata'],folder/'grounding')
        if choices['joint_family']!='revolute' or choices['motion']!=task['label'].split('_')[0]:
            raise ValueError('RGB grounding chose an incompatible task primitive')
        cameras,_=read_observation_v3(folder/'opened',state['metadata'])
        stage='sensor_geometry'
        template=fresh_task_template(read(reference/'task-spec.json'),choices)
        mechanism,cloud,mask,anchors,evidence=ground_handle(cameras,choices,template,state['metadata'])
        anchors,pad_support=handle_anchor_menu(anchors[0][:2],cloud,state['metadata'])
        save(folder/'pad-support.json',pad_support)
        spec=mechanism.spec;spec['observed_handle_radius_m']=evidence['cylinder_radius_m']
        goal,goal_evidence=terminal_angle(cameras,state['metadata'],spec,cloud)
        task_goal=goal
        if spec['motion']=='close':spec['terminal_panel_normal_world']=goal_evidence['selected']['normal_world']
        else:spec['terminal_panel_normal_world']=(Rotation.from_rotvec(goal*np.asarray(spec['joint_axis_world'])).as_matrix()@np.asarray(spec['panel_normal_world'])).tolist()
        goal=min(task_goal,config['contact_segment_angle_rad'])
        save(folder/'task-spec.json',spec);save(folder/'geometry.json',evidence);save(folder/'goal.json',{'angle_rad':task_goal,'initial_segment_angle_rad':goal,**goal_evidence})
        np.save(folder/'handle-cloud.npy',cloud)
        record=sim.robot_chain();save(folder/'robot-chain.json',record)
        import vamp
        empty=vamp.Environment();self_valid=lambda q:vamp.panda.validate(np.asarray(q).tolist(),empty,True)
        ik=TaskIK(record,config,self_valid,output=folder,sim=sim)
        opened_robot=RobotVolumes(ik.chain,state['metadata']['robot_collision_volumes_world'],state['metadata']['joint_positions'])
        aperture=float(evidence['cylinder_radius_m'])+.001
        predicted=sim.fk(state['metadata']['joint_positions'],include_volumes=True,gripper_qpos=[aperture,-aperture])
        closed_robot=RobotVolumes(ik.chain,predicted['robot_collision_volumes_world'],state['metadata']['joint_positions'])
        open_scene=Scene(cameras,state['metadata'],spec,cloud,mask,choices,config,robot=opened_robot)
        closed_scene=Scene(cameras,state['metadata'],spec,cloud,mask,choices,config,robot=closed_robot)
        save(folder/'initial-scene.json',closed_scene.record())
        stage='grasp_generation'
        generated=resources.grasps(cloud,folder/'grasps')
        stage='complete_motion_grasp_selection'
        selected=None
        for horizon,name in [(task_goal,'full-task'),(goal,'bounded-segment')]:
            selection=folder/('selection-'+name);selection.mkdir()
            try:
                selected=complete_motion_selection(generated,anchors,state['metadata'],ik,open_scene,closed_scene,
                    spec,horizon,selection,save,contact_model='jaw_contact',maximum_seconds=180 if name=='full-task' else 300,
                    maximum_poses=12 if name=='full-task' else 36,require_valid_goal=name=='full-task')
            except ValueError as error:
                if not str(error).startswith('No tested grasp'):raise
                save(selection/'failure.json',{'reason':str(error)});continue
            goal=horizon;shutil.copyfile(selection/'grasp-search.json',folder/'grasp-search.json');break
        if selected is None:raise PlanningFailure('No checked initial grasp and contact path found')
        chosen,problem,path=selected
        goal_record=read(folder/'goal.json');goal_record['selected_initial_horizon_rad']=goal;save(folder/'goal.json',goal_record)
        save(folder/'chosen-grasp.json',{k:v for k,v in chosen.items() if k!='pose'})
        summary['grasp_source']=chosen.get('source')
        summary['tested_grasp_pose_index']=chosen.get('pose_index')
        np.save(folder/'pregrasp-complete-path.npy',path)
        summary['complete_path_found_before_grasp']=bool(abs(goal-task_goal)<1e-8)
        summary['initial_contact_segment_found_before_grasp']=True
        summary['requested_task_angle_rad']=task_goal
        if plan_only:
            summary['status']='planned_only';return summary
        stage='approach'
        state=sim.enable_joint_tracking(kp=300.);save(folder/'controller-switch.json',state['controller_change'])
        state=joint_approach(sim,state,ik,open_scene,chosen,folder)
        state=sim.observe(folder/'at-grasp')
        authorization=require_arrival(chosen['pose_world_xyzw'],state['metadata']['eef_pose_world_xyzw'])
        save(folder/'closure-authorization.json',{'control_step':state['control_steps'],**authorization})
        stage='physical_grasp'
        state=sim.joint_action(chosen['joints'],gripper=1,max_steps=min(40,600-state['control_steps']),output_dir=folder/'closed')
        cameras,_=read_observation_v3(folder/'closed',state['metadata'])
        observed=translated_surface(cloud,cameras,state['metadata']['robot_collision_volumes_world'],np.zeros(3))
        if not observed['accepted']:raise ValueError('Selected handle cannot be tracked after closure')
        shift=np.asarray(observed['translation_world_m']);old_centre=np.median(cloud,axis=0);cloud=cloud+shift
        closure_angle=observed_angle(np.asarray(spec['joint_origin_world_m']),np.asarray(spec['joint_axis_world']),old_centre,np.median(cloud,axis=0))
        spec=copy.deepcopy(spec)
        rotation=Rotation.from_rotvec(closure_angle*np.asarray(spec['joint_axis_world'])).as_matrix()
        spec['front_plane']['point_world_m']=((np.asarray(spec['front_plane']['point_world_m'])-old_centre)@rotation.T+np.median(cloud,axis=0)).tolist()
        spec['panel_normal_world']=(rotation@np.asarray(spec['panel_normal_world'])).tolist()
        spec['source_anchor_world_m']=(np.asarray(spec['source_anchor_world_m'])+shift).tolist()
        goal=max(.02,goal-closure_angle)
        stage='postclosure_full_replan'
        warm=path.copy();warm[:,7]*=goal/path[-1,7]
        problem,scene,path=adaptive_contact_plan(ik,state,cameras,spec,cloud,mask,choices,goal,folder/'plan-00',config,
                                              static_prior=closed_scene,warm_path=warm)
        initial_cloud=cloud.copy();last_pad=world_pad(state['metadata'])
        centres=[np.median(cloud,axis=0)];last_sensor_step=state['control_steps'];plan_number=0
        observed_reference_centre=centres[0].copy()
        index,holds=0,0
        stage='planned_joint_tracking'
        while index<len(path):
            if state['control_steps']>=600:raise ValueError('600 control step budget exhausted')
            current=np.asarray(state['metadata']['joint_positions'])
            actual_theta=current_progress(problem,current)
            command_index=tracking_reference(ik,scene,problem,path,index,current,actual_theta)
            reference_state=path[command_index];qref=reference_state[:7]
            if np.max(np.abs(qref-current))>.049:raise ValueError('Planned reference exceeds the bounded joint servo step')
            state=sim.joint_action(qref,gripper=1,max_steps=1)
            q=np.asarray(state['metadata']['joint_positions']);theta=current_progress(problem,q)
            model_errors=problem.errors(np.r_[q,theta])
            near=path[max(0,index-6):min(len(path),command_index+3),:7]
            tube=float(np.min(np.max(np.abs(near-q),axis=1)))
            check=scene.check_volumes(state['metadata']['robot_collision_volumes_world'],theta,execution=True)
            buffer=scene.check_volumes(state['metadata']['robot_collision_volumes_world'],theta)
            event={'control_step':state['control_steps'],'plan':plan_number,
                'contact_mode':'aligned' if problem.jaw_axis_local is not None else 'tilted','reference_index':command_index,
                'path_progress_index':index,
                'reference_theta':float(reference_state[7]),'estimated_theta':theta,
                'joint_reference_error_rad':float(np.max(np.abs(q-qref))),'path_tube_error_rad':tube,
                'model_errors':model_errors,'scene_accepted':check['accepted'],
                'planning_buffer_accepted':buffer['accepted'],
                'planning_buffer_violations':buffer['collisions']}
            tracking.append(event);save(folder/'tracking.json',tracking)
            if not check['accepted']:
                save(folder/'tracking-collision.json',check);raise ValueError('Measured robot violates the planned observed scene')
            if not self_valid(q) or tube>.04:raise ValueError('Joint tracking left the validated path neighbourhood')
            if model_errors['position_error_m']>.012:raise ValueError('Measured hand/door model residual exceeds tracking allowance')
            if model_errors['jaw_axis_error_rad']>np.deg2rad(2.) or (getattr(problem,'jaw_tilt_limit',None) is not None and model_errors['measured_jaw_tilt_rad']>problem.jaw_tilt_limit+np.deg2rad(2.)):
                raise ValueError('Measured jaws no longer remain perpendicular to the observed bar')
            if model_errors['orientation_cone_angle_rad']>problem.orientation_cone+.08:
                raise ValueError('Measured gripper left its configured orientation allowance')
            window=path[index:min(len(path),command_index+2),:7]
            closest=index+int(np.argmin(np.linalg.norm(window-q,axis=1)))
            if closest>index:
                index=closest;holds=0
            elif command_index==len(path)-1 and phase_arrival(path,q,theta,
                    joint_tolerance=config['joint_path_endpoint_tolerance_rad'])['arrived']:
                index=len(path);holds=0
            else:
                holds+=1
            stalled=holds>20
            endpoint_candidate=(command_index==len(path)-1 and
                event['joint_reference_error_rad']<config['joint_path_endpoint_tolerance_rad'])
            if state['control_steps']-last_sensor_step>=config['sensor_interval_steps'] or index==len(path) or stalled or (endpoint_candidate and holds==1):
                obs_folder=folder/f'observation-{len(observations):02d}';state=sim.observe(obs_folder)
                cameras,_=read_observation_v3(obs_folder,state['metadata'])
                pad=world_pad(state['metadata'])
                observed=translated_surface(cloud,cameras,state['metadata']['robot_collision_volumes_world'],pad-last_pad)
                if not observed['accepted']:raise ValueError('RGB-D tracking lost the handle during joint trajectory execution')
                cloud=cloud+np.asarray(observed['translation_world_m']);centres.append(np.median(cloud,axis=0))
                sensor_theta=observed_angle(problem.origin,problem.axis,observed_reference_centre,centres[-1])
                hand_theta=current_progress(problem,np.asarray(state['metadata']['joint_positions']))
                if abs(sensor_theta-hand_theta)>.04:
                    save(folder/'rejected-observation.json',{'control_step':state['control_steps'],
                        'tracking':observed,'sensor_theta_rad':sensor_theta,'hand_theta_rad':hand_theta})
                    raise ValueError('Observed door motion disagrees with the hand-implied contact progress')
                fit=fit_upright_hinge(centres)
                finger_shift=float(np.linalg.norm(pad_local(state['metadata'])-problem.pad_local))
                geometry_changed=finger_shift>config['grasp_geometry_replan_shift_m']
                observations.append({'control_step':state['control_steps'],'tracking':observed,'hinge_fit':fit,
                    'observed_door_progress_rad':sensor_theta,'hand_implied_progress_rad':hand_theta,
                    'configured_pad_midpoint_shift_m':finger_shift,'joint_tracking_stalled':stalled})
                save(folder/'observations.json',observations)
                last_pad=pad;last_sensor_step=state['control_steps']
                at_endpoint=(index==len(path) or endpoint_candidate)
                maximum_replans=config['maximum_model_and_goal_replans']
                if (index<len(path) and (fit['accepted'] or geometry_changed or stalled) and plan_number<maximum_replans) or at_endpoint:
                    updated=np.asarray(fit['hinge_world_m']) if fit['accepted'] else problem.origin
                    change=float(np.linalg.norm((updated-problem.origin)[:2]))
                    if (.006<change<.08 and not at_endpoint) or ((at_endpoint or geometry_changed or stalled) and change<.08):
                        current_theta=current_progress(problem,np.asarray(state['metadata']['joint_positions']))
                        remaining=max(.02,float(path[-1,7]-current_theta))
                        next_spec=copy.deepcopy(spec)
                        transform=problem.transform(current_theta)
                        next_spec['joint_origin_world_m']=updated.tolist()
                        next_spec['panel_normal_world']=(transform[:3,:3]@np.asarray(spec['panel_normal_world'])).tolist()
                        next_spec['front_plane']['point_world_m']=((np.asarray(spec['front_plane']['point_world_m'])-
                            observed_reference_centre)@transform[:3,:3].T+centres[-1]).tolist()
                        next_spec['source_anchor_world_m']=(np.asarray(spec['source_anchor_world_m'])+
                            centres[-1]-observed_reference_centre).tolist()
                        next_mask,next_choices=fresh_mask(cameras,next_spec,cloud,scene,current_theta)
                        support=np.asarray(cameras[next_spec['camera']]['world_points'])[next_mask]
                        face=upright_plane(support,np.asarray(cameras[next_spec['camera']]['camera_to_world'])[:3,3])
                        normal=face['normal']
                        if normal@np.asarray(next_spec['panel_normal_world'])<0:normal=-normal
                        next_spec['panel_normal_world']=normal.tolist()
                        next_spec['front_plane']['point_world_m']=face['point'].tolist()
                        if 'terminal_panel_normal_world' in next_spec:
                            desired=np.asarray(next_spec['terminal_panel_normal_world'])
                            visual_remaining=closing_normal_residual(normal,desired,problem.axis)
                            remaining=max(.002,visual_remaining)
                            if at_endpoint:
                                summary['observed_goal_residual_rad']=visual_remaining
                                if spec['motion']=='close':summary['observed_closing_residual_rad']=visual_remaining
                                save(folder/'visual-goal-check.json',{'control_step':state['control_steps'],
                                    'remaining_closing_angle_rad':visual_remaining,'plane_rms_m':face['rms'],
                                    'plane_support_points':int(face['inliers'].sum()),
                                    'correction_requested':visual_remaining>.003,
                                    'replan_budget_available':plan_number<maximum_replans,
                                    'source':'RGB-D moving-panel normal versus fixed task target normal',
                                    'evaluation_geometry_used':False})
                        if at_endpoint and visual_remaining<=.003:
                            index=len(path)
                        meaningful_endpoint=(not endpoint_candidate or phase_arrival(path,q,theta,
                            joint_tolerance=config['joint_path_endpoint_tolerance_rad'])['arrived'])
                        should_replan=((not at_endpoint and (.006<change or geometry_changed or stalled))
                            or (at_endpoint and visual_remaining>.003 and (meaningful_endpoint or stalled or geometry_changed)))
                        if should_replan and (fit['accepted'] or geometry_changed or stalled) and plan_number<maximum_replans:
                            plan_number+=1
                            warm=None
                            if at_endpoint:
                                q=np.asarray(state['metadata']['joint_positions'])
                                warm=np.array([np.r_[q,0.],np.r_[q,remaining]])
                            elif path[-1,7]-current_theta>1e-5:
                                warm=path[index:].copy()
                                warm[:,7]=(warm[:,7]-current_theta)*remaining/(path[-1,7]-current_theta)
                                warm=warm[(warm[:,7]>=-.03)&(warm[:,7]<=remaining+.03)]
                            reason=('observed finger geometry changed' if geometry_changed else
                                'joint tracking stalled' if stalled else
                                'contact segment endpoint' if at_endpoint else 'observed hinge model changed')
                            stage='visual_goal_full_replan' if at_endpoint else 'feedback_full_replan'
                            problem,scene,path=adaptive_contact_plan(ik,state,cameras,next_spec,cloud,next_mask,next_choices,
                                remaining,folder/f'plan-{plan_number:02d}',config,static_prior=scene,warm_path=warm)
                            replans.append({'control_step':state['control_steps'],
                                'reason':reason,'configured_pad_midpoint_shift_m':finger_shift,
                                'estimated_hinge_world_m':next_spec['joint_origin_world_m'],
                                'contact_mode':'aligned' if problem.jaw_axis_local is not None else 'tilted',
                                'overall_task_remaining_rad':visual_remaining,
                                'hinge_change_m':change,'remaining_angle_rad':remaining})
                            save(folder/'replans.json',replans);spec=next_spec;index=0;holds=0
                            stage='planned_joint_tracking'
                            observed_reference_centre=centres[-1].copy()
                        elif at_endpoint and index==len(path) and visual_remaining>.003:
                            raise ValueError('Observed task goal remains unresolved after bounded segment replanning')
                if stalled and holds>20:
                    raise ValueError('Joint tracking stall could not be recovered by a complete checked replan')
        summary.update(status='completed_planned_trajectory',plans_executed=plan_number+1,
            maximum_path_tube_error_rad=max(t['path_tube_error_rad'] for t in tracking))
        sim.observe(folder/'final')
    except Exception as error:
        summary.update(status='failed',failure_stage=stage,failure=str(error))
        (folder/'error.txt').write_text(traceback.format_exc())
    finally:
        if ik:ik.close()
        sim.close()
        actions=read(folder/'initial/actions.json') if (folder/'initial/actions.json').exists() else []
        provenance=folder/'grounding/input-provenance.json'
        if provenance.exists():summary['new_api_calls']=read(provenance).get('new_api_calls',0)
        summary.update(control_steps=len(actions),official_task_success=any(a['evaluation_task_success'] for a in actions),
            final_task_success=bool(actions and actions[-1]['evaluation_task_success']),replans=len(replans))
        summary.setdefault('status','interrupted')
        write_json(folder/'summary.json',summary)
        summary['storage']=compact_episode(folder)
        write_json(folder/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--tasks',nargs='+',choices=['open_microwave','close_microwave'],default=['open_microwave','close_microwave'])
    parser.add_argument('--states',nargs='+',type=int,default=[1])
    parser.add_argument('--frozen-protocol')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();output=ROOT/args.output;output.mkdir(parents=True,exist_ok=False)
    if len(set(args.states))!=len(args.states) or min(args.states)<0:
        raise ValueError('Use unique nonnegative official initial-state indices')
    if args.frozen_protocol:
        pinned=read(ROOT/args.frozen_protocol)['sources_sha256']
        if any(digest(ROOT/p)!=h for p,h in pinned.items()):raise ValueError('Evaluation sources differ from the frozen protocol')
    baseline=read(ROOT/'results/handle-contact-development/pilot-v20/protocol.json')['source_sha256']
    if any(digest(ROOT/p)!=h for p,h in baseline.items()):raise ValueError('Frozen reactive baseline sources changed')
    config=read(ROOT/'configs/articulated_libero.json')
    config.update(scene_voxel_spacing_m=.003,observed_clearance_margin_m=.003,noncontact_tool_clearance_m=.002,
        execution_clearance_margin_m=.001,joint_path_endpoint_tolerance_rad=.006,
        grasp_geometry_replan_shift_m=.0015,maximum_model_and_goal_replans=32,
        contact_segment_angle_rad=float(np.deg2rad(20.)),contact_fallback_segment_angle_rad=float(np.deg2rad(10.)),sensor_interval_steps=5,
        maximum_jaw_tilt_rad=float(np.deg2rad(60.)),maximum_global_plan_seconds=20.,prefer_milestone_plans=True,
        complete_planar_panel=True,panel_completion_spacing_m=.005,panel_completion_maximum_points=100000,
        panel_completion_extent_source='selected_plane_pixels',
        allow_palm_panel_brace=False,allow_palm_handle_contact=False,
        allow_pad_handle_contact=True,allow_finger_handle_contact=True,
        maximum_target_contact_penetration_m=.003)
    tasks=[t for t in config['tasks'] if t['label'] in args.tasks]
    ledger=ROOT/'results/articulated-api-v1/ledger.json';ledger_hash=digest(ledger);api_before=api_snapshot()
    sources={**baseline,**{str(p.relative_to(ROOT)):digest(p) for pattern in ['src/vlm_tamp/coupled_*.py','scripts/*coupled*.py'] for p in ROOT.glob(pattern)}}
    for name in sources:
        target=output/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    save(output/'protocol.json',{'sources_sha256':sources,'config':config,'paid_grounding_allowed':True,'api_budget_usd':2.,'api_retries':0,
        'api_snapshot_before':api_before,'evaluation_states':args.states,'development_states':[0,1],
        'execution_thread_environment':{key:os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        'api_ledger_sha256_before':ledger_hash,'reactive_baseline_preserved':True,
        'full_motion_required_before_grasp':False,'complete_contact_segment_required_before_grasp':True,
        'contact_segment_angle_deg':20.,'task_opening_angle_deg':90.,'sensor_interval_steps':5,'joint_controller':'official JOINT_POSITION for planned approach and manipulation',
        'simulator_scene_geometry_used_for_planning':False,'hierarchical_vlm_used':False,
        'contact_constraint':'grasp point follows observed hinge arc; aligned jaws preferred, bounded-tilt mode when needed; observed capture and RGB-D retention guard',
        'horizon_policy':'prefer complete task path; checked 20-degree then 10-degree segments when needed',
        'fallback_contact_segment_angle_deg':10.,
        'maximum_jaw_tilt_deg':60.,
        'physical_jaw_angle_tolerance_deg':2.,
        'grasp_binding':'actual measured grip point after observed bar proximity and finite pad support checks',
        'joint_endpoint_tolerance_rad':config['joint_path_endpoint_tolerance_rad'],
        'visual_closing_correction_threshold_rad':.003,'visual_closing_correction_maximum_rad':.05,
        'maximum_model_and_goal_replans':config['maximum_model_and_goal_replans'],
        'grasp_geometry_replan_shift_m':config['grasp_geometry_replan_shift_m'],
        'tracking_stall_requires_fresh_observation_and_complete_replan':True,
        'minimum_phase_progress_fraction':.9,
        'moving_panel_collision_model':'bounded panel inferred once and transported rigidly by observed normals and handle translation; fresh measured points retained',
        'planner':'OMPL Atlas/RRTConnect','continuous_collision_certificate':False,
        'storage_retention':STORAGE_POLICY})
    save(output/'manifest.json',{'planned_episodes':[{'task':t['label'],'state':i} for t in tasks for i in args.states],
        'development_only':all(i in {0,1} for i in args.states),'all_failures_in_denominator':True,'plan_only':args.plan_only})
    rows=[];resources=CoupledResources()
    def checkpoint():
        after=api_snapshot()
        save(output/'summary.json',{'planned':len(tasks)*len(args.states),'completed':len(rows),'episodes':rows,
            'native_successes':sum(r['official_task_success'] for r in rows),
            'new_api_calls':sum(r['new_api_calls'] for r in rows),
            'api_ledger_unchanged':digest(ledger)==ledger_hash,'api_cost_policy_satisfied':after['cumulative_conservative_spend_usd']<=2.,
            'cumulative_api_spend_usd':after['cumulative_conservative_spend_usd'],'remaining_api_budget_usd':after['remaining_usd']})
    checkpoint()
    for task in tasks:
        for index in args.states:
            rows.append(episode(task,output/f'{task["label"]}-state{index:03d}',config,args.plan_only,index,resources))
            checkpoint();print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__':main()
