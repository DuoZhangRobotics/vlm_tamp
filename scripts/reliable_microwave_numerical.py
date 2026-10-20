"""MC-VAMP microwave plans followed by complete, unguarded simulator execution."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from try_rekep_bowl_v3 import read_observation_v3
from vlm_tamp.coupled_client import CoupledClient
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.coupled_manifold import CoupledHandlePoint
from vlm_tamp.coupled_pipeline import terminal_angle,handle_anchor_menu,grasp_poses,goal_pose_menu,closing_normal_residual
from vlm_tamp.coupled_resources import CoupledResources,fresh_task_template,api_snapshot
from vlm_tamp.coupled_storage import read_json,write_json,compact_episode,preserve_scene,POLICY
from vlm_tamp.handle_contact import translated_surface,fit_upright_hinge,upright_plane
from vlm_tamp.reliable_handle import ground_handle
from run_coupled_door import fresh_mask,observed_angle
from vlm_tamp.rekep_libero.mechanism_execution import cached_grounding
from vlm_tamp.rekep_libero.geometry import pose_matrix,matrix_pose
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK
from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
from vlm_tamp.timed_execution import schedule,execute
from vlm_tamp.short_motion_policy import motion_horizons,ObservedHingeHistory
from vlm_tamp.rigid_hinge_tracking import RGBDHingeTracker,measured_pose

BASE=ROOT/'results/coupled-planning-resumed-20261015'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):write_json(path,data)


def baseline_folder(label,index):
    case=f'{label}-state{index:03d}'
    if index==1:return BASE/('final-development-'+label.split('_')[0])/case
    return BASE/'frozen-evaluation/runs'/case/case


def virtual_metadata(state,robot,chain,q):
    result=dict(state['metadata']);result.update(joint_positions=np.asarray(q).tolist(),
        eef_pose_world_xyzw=chain.forward(q).tolist(),robot_collision_volumes_world=robot.forward(q))
    return result


def constraint_model(chain,spec,q,pad,goal,aligned=True):
    return CoupledHandlePoint(chain,spec['joint_origin_world_m'],spec['joint_axis_world'],pose_matrix(chain.forward(q)),
        pad_local=pad,align_axis=False,jaw_axis_local=[1.,0.,0.] if aligned else None,
        jaw_tilt_limit=None if aligned else np.pi/3,orientation_cone=np.pi/2,
        progress_bounds=(-.03,goal+.03),joint_margin=.025,position_tolerance=.001,orientation_tolerance=.01)


def goal_candidates(ik,context,spec,start,goal,*,maximum=4):
    """IK proposes endpoints; MC-VAMP checks their point-cloud occupancy."""
    model=constraint_model(ik.chain,spec,start,context.pad_local,goal,context.record['aligned_jaws'])
    candidates=[];records=[]
    def admit(q,source):
        q,projection=context.project(q)
        check=context.validate(q) if projection['accepted'] else None
        error=context.progress(q)-goal
        okay=bool(projection['accepted'] and abs(error)<.005 and check['accepted'])
        records.append({'source':'endpoint_admission','proposal_source':source,'joints':q,
            'projection':projection,'scene':check,'accepted':okay,'goal_error_rad':error})
        if okay and not any(np.max(np.abs(q-other))<.03 for other in candidates):candidates.append(q)
    previous=np.asarray(start,float)
    # Continuation proposes a terminal posture, not an executed controller path.
    # The actual connecting path is always searched and validated by MC-VAMP.
    for angle in np.linspace(0,goal,max(2,int(np.ceil(goal/.08))+1))[1:]:
        projected=model.project(np.r_[previous,angle],fixed_progress=float(angle),max_iterations=75)
        records.append({'source':'terminal_posture_continuation','angle_rad':float(angle),
            'accepted':projected['accepted'],'position_error_m':projected['position_error_m']})
        if not projected['accepted']:break
        previous=projected['state'][:7]
    else:
        admit(previous,'terminal_posture_continuation')
    for target,angles in goal_pose_menu(model,goal,pose_matrix(ik.chain.forward(start)),context.pad_local,
                                        spec['joint_axis_world'],allow_roll=True):
        if len(candidates)>=maximum:break
        solutions=ik.solve_many(matrix_pose(target),start,maximum=8)
        records.append({'source':'curobo_terminal_pose','angles':angles,'returned':len(solutions)})
        for solution in solutions:
            q=np.asarray(solution['joints'])
            admit(q,'curobo_terminal_pose')
            if len(candidates)>=maximum:break
    # Rejected endpoints do not consume the feasible-goal quota. Previously
    # four colliding proposals prevented searching the remaining wrist poses.
    candidates.sort(key=lambda q:np.linalg.norm(q-start))
    return candidates,records


def contact_plan(ik,scene,record,metadata,spec,start,goal,folder,*,seconds=8.,modes=(True,False),backend='endpoint_menu',search_options=None):
    if backend=='partial_pose_arc':
        from vlm_tamp.numerical_arc_execution import contact_arc
        return contact_arc(scene,record,metadata,start,goal,folder,search_options or {})
    folder.mkdir(parents=True,exist_ok=False);attempts=[]
    preserve_scene(scene,folder)
    compact_metadata={key:metadata[key] for key in ['joint_positions','eef_pose_world_xyzw']}
    compact_metadata['robot_collision_volumes_world']=[v for v in metadata['robot_collision_volumes_world']
        if 'gripper' in v['robot_geom_name']]
    save(folder/'planning-input.json',{'task_spec':spec,'metadata':compact_metadata,'reference_joints':start,
        'goal_angle_rad':goal,'maximum_solve_seconds':seconds,'planning_only':True})
    for aligned in modes:
        with NativeScene(scene,record,metadata,start,goal,aligned=aligned) as context:
            begin=context.validate(start)
            if not begin['accepted']:
                attempts.append({'aligned_jaws':aligned,'status':'invalid_start','scene':begin});continue
            goals,proposals=goal_candidates(ik,context,spec,start,goal)
            save(folder/('goals-aligned.json' if aligned else 'goals-tilted.json'),proposals)
            for number,qgoal in enumerate(goals[:3]):
                path,result=context.plan(start,qgoal,seconds=seconds)
                result.update(aligned_jaws=aligned,goal_joints=qgoal.tolist())
                attempts.append(result)
                save(folder/'attempts.json',attempts)
                if result['exact_solution']:
                    progress=[context.progress(q) for q in path]
                    np.save(folder/'path.npy',path);np.save(folder/'door-progress.npy',progress)
                    save(folder/'native-scene.json',context.record)
                    preserve_scene(scene,folder)
                    save(folder/'result.json',result)
                    return path,result
    save(folder/'attempts.json',attempts)
    raise ValueError('MC-VAMP found no complete contact path to the full task endpoint')


def select_grasp(generated,anchors,state,ik,record,open_scene,closed_scene,spec,goal,folder,config):
    folder.mkdir();rows,pad=grasp_poses(generated,anchors,np.asarray(spec['panel_normal_world']),state['metadata'])
    initial=np.asarray(state['metadata']['joint_positions']);attempts=[]
    before=time.perf_counter()
    with NativeScene(open_scene,record,state['metadata'],initial,goal,articulated=False) as approach_scene:
        opening=approach_scene.validate(initial)
        save(folder/'approach-initial-scene.json',{'record':approach_scene.record,'start':opening})
        if not opening['accepted']:raise ValueError('Initial arm is in collision in the MC-VAMP observed point-cloud model')
        for ordinal,row in enumerate(rows[:config['maximum_grasp_poses']]):
            if time.perf_counter()-before>config['maximum_grasp_search_seconds']:break
            solutions=ik.solve_many(row['pose'],initial,maximum=8)
            solutions.sort(key=lambda s:(s['posture_cost'],-s['joint_reserve_rad']))
            for rank,solution in enumerate(solutions[:8]):
                q=np.asarray(solution['joints'])
                predicted=virtual_metadata(state,closed_scene.robot,ik.chain,q)
                with NativeScene(closed_scene,record,predicted,q,goal,aligned=False) as contact_scene:
                    check=contact_scene.validate(q)
                attempt={'pose_index':ordinal,'ik_rank':rank,'source':row['source'],'joints':q.tolist(),
                    'start_scene':check,'status':'start_admitted' if check['accepted'] else 'start_collision'}
                attempts.append(attempt)
                save(folder/'grasp-attempts.json',attempts)
                if not check['accepted']:continue
                # Plan to a measured 7 cm standoff before approaching the bar.
                target=np.asarray(solution['pose_world_xyzw']).copy()
                target[:3]-=.07*pose_matrix(target)[:3,2]
                pregrasps=ik.solve_many(target,q,maximum=8)
                pregrasps.sort(key=lambda s:np.linalg.norm(np.asarray(s['joints'])-q))
                approach=None
                for pre in pregrasps[:3]:
                    qpre=np.asarray(pre['joints'])
                    first,r1=approach_scene.plan(initial,qpre,constrained=False,seconds=3.)
                    if not r1['exact_solution']:continue
                    second,r2=approach_scene.plan(qpre,q,constrained=False,seconds=3.)
                    if r2['exact_solution']:
                        approach=np.vstack([first,second[1:]])
                        attempt['approach_plans']=[r1,r2];break
                if approach is None:attempt['status']='approach_planning_failed';continue
                approach_steps=len(schedule(approach,maximum_joint_speed=.30,endpoint_hold_steps=60))
                available=600-int(state['control_steps'])-40-200
                if approach_steps>available:
                    attempt.update(status='approach_exceeds_episode_budget',approach_steps=approach_steps,available_approach_steps=available);continue
                attempt['approach_steps']=approach_steps
                try:
                    path,result=contact_plan(ik,closed_scene,record,predicted,spec,q,goal,
                        folder/f'contact-pose{ordinal:02d}-ik{rank}',seconds=config['maximum_contact_plan_seconds'],
                        backend=config.get('planning_backend','endpoint_menu'),search_options=config)
                except ValueError as error:
                    attempt.update(status='contact_planning_failed',reason=str(error));continue
                attempt.update(status='selected',contact_plan=result)
                save(folder/'grasp-attempts.json',attempts)
                selected={**row,'pose_world_xyzw':solution['pose_world_xyzw'],'joints':q.tolist(),
                    'pose_index':ordinal,'ik_rank':rank,'approach':approach,'planned_contact':path,
                    'search_seconds':time.perf_counter()-before}
                save(folder/'selected.json',{k:v for k,v in selected.items() if k not in {'approach','planned_contact'}})
                np.save(folder/'approach.npy',approach);np.save(folder/'contact.npy',path)
                return selected
    save(folder/'grasp-attempts.json',attempts)
    raise ValueError('No candidate has a complete MC-VAMP approach and full-task contact path')


def remaining_plan(ik,scene,record,metadata,spec,remaining,folder,config):
    """Finish each selected horizon in execution; never replace an active path."""
    folder.mkdir(parents=True,exist_ok=False);failures=[]
    horizons=motion_horizons(remaining,config)
    for index,horizon in enumerate(horizons):
        try:
            path,result=contact_plan(ik,scene,record,metadata,spec,np.asarray(metadata['joint_positions']),
                horizon,folder/f'horizon-{index}',seconds=config['maximum_contact_plan_seconds'],
                backend=config.get('planning_backend','endpoint_menu'),search_options=config)
        except ValueError as error:
            failures.append({'horizon_rad':horizon,'reason':str(error)});continue
        save(folder/'horizon-selection.json',{'full_task_remaining_rad':remaining,'selected_horizon_rad':horizon,
            'failed_attempts':failures})
        np.save(folder/'path.npy',path);save(folder/'result.json',result)
        return path,result,horizon
    save(folder/'horizon-selection.json',{'full_task_remaining_rad':remaining,'failed_attempts':failures})
    raise ValueError('MC-VAMP found no complete trajectory for the requested motion horizons')


def episode(task,index,folder,config,resources,*,plan_only=False):
    folder.mkdir(parents=True,exist_ok=False)
    sim=CoupledClient(folder/'simulator.log');ik=None;stage='reset';events=[];phases=[];observations=[]
    before=time.perf_counter()
    scene_type=Scene
    if config.get('reconcile_handle_roles',False):
        from vlm_tamp.arc_contact_scene import ArcContactScene
        scene_type=ArcContactScene
    result={'task':task['label'],'initial_state':index,'seed':0,'status':'interrupted',
        'official_task_success':False,'new_api_calls':0,'execution_time_collision_checks':False,
        'execution_time_constraint_checks':False,'execution_time_progress_vetoes':False,
        'assisted_attachment':False,'physical_contacts_enabled':True,'planner':'MC-VAMP',
        'all_planned_commands_issued':False}
    try:
        state=sim.reset(task['name'],index,folder/'initial',seed=0)
        state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,output_dir=folder/'opened')
        source=baseline_folder(task['label'],index)
        result['baseline_episode']=str(source)
        stage='frozen_grounding'
        if config.get('shared_cohort_grounding',False):
            choices=resources.grounding(task['label'],index,folder/'opened',state['metadata'],folder/'grounding')
        elif not (source/'grounding/grounding.json').exists():
            failure=read_json(source/'summary.json')
            result['shared_baseline_input_failure']=True
            raise ValueError('Frozen input unavailable: '+failure.get('failure','no completed VLM grounding'))
        else:
            _,choices,_,_=cached_grounding(source/'grounding',folder/'opened',state['metadata'],folder/'grounding')
            save(folder/'grounding/input-provenance.json',{'new_api_calls':0,'source':str(source/'grounding'),
                'rgb_and_selected_depth_verified':True,'baseline_grounding_reused':True})
        stage='sensor_geometry'
        cameras,_=read_observation_v3(folder/'opened',state['metadata'])
        template=fresh_task_template(read_json(ROOT/'results/articulated-physics-pilot'/f'{task["label"]}-state001/task-spec.json'),choices)
        mechanism,cloud,mask,anchors,evidence=ground_handle(cameras,choices,template,state['metadata'])
        anchors,support=handle_anchor_menu(anchors[0][:2],cloud,state['metadata'])
        spec=mechanism.spec;spec['observed_handle_radius_m']=evidence['cylinder_radius_m']
        goal,goal_evidence=terminal_angle(cameras,state['metadata'],spec,cloud)
        terminal_normal=(np.asarray(goal_evidence['selected']['normal_world']) if spec['motion']=='close' else
            Rotation.from_rotvec(goal*np.asarray(spec['joint_axis_world'])).as_matrix()@np.asarray(spec['panel_normal_world']))
        save(folder/'task-spec.json',spec);save(folder/'geometry.json',evidence);save(folder/'pad-support.json',support)
        save(folder/'goal.json',{'angle_rad':goal,**goal_evidence});np.save(folder/'handle-cloud.npy',cloud)
        np.save(folder/'panel-mask.npy',mask)
        record=sim.robot_chain();save(folder/'robot-chain.json',record)
        import vamp
        empty=vamp.Environment()
        self_valid=lambda q:vamp.panda.validate(np.asarray(q).tolist(),empty,True)
        ik=TaskIK(record,config,self_valid,output=folder,sim=sim)
        opened_robot=RobotVolumes(ik.chain,state['metadata']['robot_collision_volumes_world'],state['metadata']['joint_positions'])
        aperture=float(evidence['cylinder_radius_m'])+.001
        predicted=sim.fk(state['metadata']['joint_positions'],include_volumes=True,gripper_qpos=[aperture,-aperture])
        closed_robot=RobotVolumes(ik.chain,predicted['robot_collision_volumes_world'],state['metadata']['joint_positions'])
        open_scene=scene_type(cameras,state['metadata'],spec,cloud,mask,choices,config,robot=opened_robot)
        closed_scene=scene_type(cameras,state['metadata'],spec,cloud,mask,choices,config,robot=closed_robot)
        preserve_scene(closed_scene,folder)
        stage='grasp_generation'
        generated=resources.grasps(cloud,folder/'grasps')
        stage='mcvamp_grasp_and_full_motion_planning'
        selected=None;selection_start=time.perf_counter()
        searches=[(goal,'full-task',24 if config.get('full_task_only',False) else 6)]
        if not config.get('full_task_only',False):searches.append((min(goal,np.deg2rad(20.)),'bounded-segment',24))
        if config.get('motion_policy')=='short' and not config.get('full_task_only',False):
            searches=[(h,f'short-{number}',24) for number,h in enumerate(motion_horizons(goal,config))]
        if config.get('replay_grasp_source'):
            from vlm_tamp.arc_execution import frozen_grasp
            source=Path(config['replay_grasp_source']);replay_summary=read_json(source/'summary.json')
            if replay_summary['task']!=task['label'] or replay_summary['initial_state']!=index:
                raise ValueError('Frozen grasp does not belong to this requested task and initial state')
            selected,initial_horizon=frozen_grasp(source,state,ik,record,open_scene,closed_scene,folder/'grasp-replay')
            result.update(selected_initial_horizon_rad=initial_horizon,frozen_grasp_replay=True,
                original_grasp_source=str(source),full_task_path_planned_before_grasp=False)
            searches=[]
        for initial_horizon,name,limit in searches:
            try:
                selected=select_grasp(generated,anchors,state,ik,record,open_scene,closed_scene,spec,initial_horizon,
                    folder/('selection-'+name),{**config,'maximum_grasp_poses':limit})
            except ValueError as error:
                save(folder/('selection-'+name)/'failure.json',{'reason':str(error)});continue
            result['selected_initial_horizon_rad']=initial_horizon;break
        if selected is None:raise ValueError('No MC-VAMP grasp/approach/contact segment was found')
        np.save(folder/'pregrasp-complete-path.npy',selected['planned_contact'])
        save(folder/'chosen-grasp.json',{k:v for k,v in selected.items() if k not in {'approach','planned_contact'}})
        result.update(grasp_source=selected['source'],complete_path_found_before_grasp=bool(abs(initial_horizon-goal)<1e-8),
            initial_search_seconds=time.perf_counter()-selection_start)
        if plan_only:result['status']='planned_only';return result
        # Grasping can itself move the door. Track that measured motion rather
        # than discarding it and starting hinge calibration after closure.
        closure_tracker=None
        if config.get('hinge_estimator','circle')=='hybrid':
            closure_tracker=RGBDHingeTracker(cameras,open_scene.moving_reference,prior=spec['joint_origin_world_m'],
                reference_step=state['control_steps'],output=folder/'preclosure-landmark-reference.npz')
        stage='timed_approach'
        state=sim.enable_joint_tracking(kp=300.)
        save(folder/'controller-switch.json',state['controller_change'])
        approach=schedule(selected['approach'],maximum_joint_speed=.30,endpoint_hold_steps=60)
        np.save(folder/'approach-references.npy',approach)
        state,execution=execute(sim,state,approach,gripper=-1,event_sink=events.append)
        save(folder/'approach-execution.json',{k:v for k,v in execution.items() if k!='events'})
        stage='physical_grasp'
        closure=np.repeat(np.asarray(selected['joints'])[None],40,axis=0)
        state,_=execute(sim,state,closure,gripper=1,event_sink=events.append)
        state=sim.observe(folder/'closed')
        # A new plan may use a postclosure observation. Once issued, it is
        # executed in full without further checks or replanning.
        cameras,_=read_observation_v3(folder/'closed',state['metadata'])
        tracking=translated_surface(cloud,cameras,state['metadata']['robot_collision_volumes_world'],np.zeros(3))
        closure_rigid=(closure_tracker.observe(cameras,control_step=state['control_steps'],
            output=folder/'closure-landmark-matches.npz') if closure_tracker else {'accepted':False,'reason':'disabled'})
        closure_pose=measured_pose(closure_rigid)
        save(folder/'closure-rigid-tracking.json',closure_rigid)
        save(folder/'closure-observation.json',tracking)
        if tracking['accepted'] or closure_pose is not None:
            old_centre=np.median(cloud,axis=0)
            if closure_pose is not None:
                rotation=closure_pose[:3,:3];shift=closure_pose[:3,3]
                cloud=cloud@rotation.T+shift
                closure_angle=float(closure_rigid['rotation_rad']*np.asarray(spec['joint_axis_world'])[2])
            else:
                shift=np.asarray(tracking['translation_world_m']);cloud=cloud+shift
                closure_angle=observed_angle(np.asarray(spec['joint_origin_world_m']),np.asarray(spec['joint_axis_world']),
                    old_centre,np.median(cloud,axis=0))
                rotation=Rotation.from_rotvec(closure_angle*np.asarray(spec['joint_axis_world'])).as_matrix()
            spec=copy.deepcopy(spec)
            spec['front_plane']['point_world_m']=(rotation@(
                np.asarray(spec['front_plane']['point_world_m'])-old_centre)+np.median(cloud,axis=0)).tolist()
            spec['panel_normal_world']=(rotation@np.asarray(spec['panel_normal_world'])).tolist()
            spec['source_anchor_world_m']=(rotation@(
                np.asarray(spec['source_anchor_world_m'])-old_centre)+np.median(cloud,axis=0)).tolist()
            mask,choices=fresh_mask(cameras,spec,cloud,closed_scene,closure_angle)
        hinge_refined=False
        if closure_rigid['accepted'] and np.linalg.norm(np.asarray(closure_rigid['hinge_world_m'])[:2]-np.asarray(spec['joint_origin_world_m'])[:2])<.08:
            spec=copy.deepcopy(spec);spec['joint_origin_world_m']=closure_rigid['hinge_world_m'];hinge_refined=True
        closed_robot=RobotVolumes(ik.chain,state['metadata']['robot_collision_volumes_world'],state['metadata']['joint_positions'])
        scene=scene_type(cameras,state['metadata'],spec,cloud,mask,choices,config,robot=closed_robot)
        if closure_pose is not None and config.get('reconcile_handle_roles'):
            scene.observed_motion_transform=closure_pose
        scene.retain_static(closed_scene)
        rigid_tracker=None;last_rigid_pose=np.eye(4);last_rigid_step=state['control_steps']
        rigid_reference_cloud=cloud.copy()
        if config.get('hinge_estimator','circle')=='hybrid':
            rigid_tracker=RGBDHingeTracker(cameras,scene.moving_reference,prior=spec['joint_origin_world_m'],
                reference_step=state['control_steps'],output=folder/'rigid-landmark-reference.npz')
        centres=[np.median(cloud,axis=0)]
        hinge_history=ObservedHingeHistory()
        hinge_history.add(centres[0],observed=tracking['accepted'] or closure_pose is not None,control_step=state['control_steps'])
        save(folder/'hinge-calibration.json',{**hinge_history.record(refined=hinge_refined),'closure_rigid_fit':closure_rigid})
        remaining=max(.002,closing_normal_residual(np.asarray(spec['panel_normal_world']),terminal_normal,np.asarray(spec['joint_axis_world'])))
        for phase in range(64 if config.get('motion_policy')=='short' else 16):
            stage='mcvamp_planning_between_completed_trajectories'
            if state['control_steps']>=600:raise ValueError('Episode control-step budget exhausted between completed trajectories')
            phase_folder=folder/f'plan-{phase:02d}'
            plan_start=time.perf_counter()
            path,planning,horizon=remaining_plan(ik,scene,record,state['metadata'],spec,remaining,phase_folder,
                {**config,'hinge_refinement_accepted':hinge_refined})
            cycle=time.perf_counter()-plan_start
            stage='timed_full_trajectory'
            hold=(6 if hinge_refined else 4) if config.get('motion_policy')=='short' else 12
            references=schedule(path,maximum_joint_speed=.35,endpoint_hold_steps=hold)
            available=600-state['control_steps']
            if len(references)>available:
                moving=len(references)-hold
                if available<=hold:raise ValueError('Insufficient episode budget to schedule another complete trajectory')
                speed=.35*moving/(available-hold)
                references=schedule(path,maximum_joint_speed=speed*1.001,endpoint_hold_steps=hold)
            np.save(phase_folder/'execution-references.npy',references)
            q_before=np.asarray(state['metadata']['joint_positions'])
            tcp_before=pose_matrix(ik.chain.forward(q_before))
            pad_before=tcp_before[:3,3]+tcp_before[:3,:3]@context_pad(state['metadata'])
            state,execution=execute(sim,state,references,gripper=1,event_sink=events.append)
            phase_record={'phase':phase,'full_task_remaining_before_rad':remaining,'planned_horizon_rad':horizon,
                'planning_cycle_seconds':cycle,'planning':planning,
                **{k:v for k,v in execution.items() if k!='events'}}
            phases.append(phase_record);save(folder/'phases.json',phases)
            result.update(all_planned_commands_issued=True,completed_contact_trajectories=len(phases),
                manipulation_commands_issued=sum(p['issued_commands'] for p in phases),
                execution_wall_seconds=sum(p['wall_seconds'] for p in phases),
                maximum_joint_reference_error_rad=max(p['maximum_joint_reference_error_rad'] for p in phases))
            # The entire trajectory is over before any perception or next-plan
            # decision. Tracking/model errors never interrupt its references.
            stage='perception_between_completed_trajectories'
            obs_folder=folder/f'observation-{phase:02d}'
            state=sim.observe(obs_folder);cameras,_=read_observation_v3(obs_folder,state['metadata'])
            rigid={'accepted':False,'reason':'disabled'};measured_transform=None;current_pose=None
            if rigid_tracker is not None:
                rigid=rigid_tracker.observe(cameras,control_step=state['control_steps'],
                    output=phase_folder/'rigid-landmark-matches.npz')
                current_pose=measured_pose(rigid)
                if current_pose is not None:
                    previous_step=observations[-1]['control_step'] if observations else rigid_tracker.reference_step
                    if last_rigid_step==previous_step:measured_transform=current_pose@np.linalg.inv(last_rigid_pose)
                    last_rigid_pose=current_pose;last_rigid_step=state['control_steps']
            tcp=pose_matrix(ik.chain.forward(state['metadata']['joint_positions']))
            pad_after=tcp[:3,3]+tcp[:3,:3]@context_pad(state['metadata'])
            observed=translated_surface(cloud,cameras,state['metadata']['robot_collision_volumes_world'],pad_after-pad_before)
            old_centre=np.median(cloud,axis=0)
            if current_pose is not None:
                cloud=rigid_reference_cloud@current_pose[:3,:3].T+current_pose[:3,3]
                observed={**observed,'translation_tracker_accepted':observed['accepted'],'accepted':True,
                    'handle_pose_source':'persistent measured RGB-D rigid motion',
                    'translation_world_m':(np.median(cloud,axis=0)-old_centre).tolist()}
            else:
                translation=(np.asarray(observed['translation_world_m']) if observed['accepted'] else
                    (Rotation.from_rotvec(horizon*np.asarray(spec['joint_axis_world'])).as_matrix()@(
                        old_centre-np.asarray(spec['joint_origin_world_m']))+np.asarray(spec['joint_origin_world_m'])-old_centre))
                cloud=cloud+translation
            centre=np.median(cloud,axis=0);centres.append(centre);translation=centre-old_centre
            theta=(float(np.arctan2(measured_transform[1,0],measured_transform[0,0])*np.asarray(spec['joint_axis_world'])[2])
                if measured_transform is not None else observed_angle(np.asarray(spec['joint_origin_world_m']),
                    np.asarray(spec['joint_axis_world']),old_centre,centre))
            new_spec=copy.deepcopy(spec)
            rotation=Rotation.from_rotvec(theta*np.asarray(spec['joint_axis_world'])).as_matrix()
            new_spec['front_plane']['point_world_m']=(rotation@(
                np.asarray(spec['front_plane']['point_world_m'])-old_centre)+centre).tolist()
            new_spec['panel_normal_world']=(rotation@np.asarray(spec['panel_normal_world'])).tolist()
            new_spec['source_anchor_world_m']=(rotation@(
                np.asarray(spec['source_anchor_world_m'])-old_centre)+centre).tolist()
            circle_fit=hinge_history.add(centre,observed=observed['accepted'],control_step=state['control_steps'])
            fit=rigid if rigid.get('accepted') else circle_fit
            if rigid_tracker is not None and hinge_refined and not rigid.get('accepted') and circle_fit.get('accepted'):
                if np.linalg.norm(np.asarray(circle_fit['hinge_world_m'])[:2]-np.asarray(spec['joint_origin_world_m'])[:2])>.02:
                    fit={'accepted':False,'reason':'circle_fit_disagrees_with_refined_hinge'}
            if fit['accepted'] and np.linalg.norm(np.asarray(fit['hinge_world_m'])[:2]-np.asarray(spec['joint_origin_world_m'])[:2])<.08:
                new_spec['joint_origin_world_m']=fit['hinge_world_m']
                hinge_refined=True
            result.update(hinge_refinement_used=hinge_refined,observed_hinge_samples=len(hinge_history.points))
            save(folder/'hinge-calibration.json',{**hinge_history.record(refined=hinge_refined),
                'last_fit':fit,'last_circle_fit':circle_fit,'last_rigid_fit':rigid})
            next_mask,next_choices=fresh_mask(cameras,new_spec,cloud,scene,theta)
            support=np.asarray(cameras[new_spec['camera']]['world_points'])[next_mask]
            try:
                face=upright_plane(support,np.asarray(cameras[new_spec['camera']]['camera_to_world'])[:3,3])
                normal=face['normal']
                if normal@np.asarray(new_spec['panel_normal_world'])<0:normal=-normal
                new_spec['panel_normal_world']=normal.tolist();new_spec['front_plane']['point_world_m']=face['point'].tolist()
            except ValueError:
                normal=np.asarray(new_spec['panel_normal_world'])
            visual_remaining=closing_normal_residual(normal,terminal_normal,np.asarray(new_spec['joint_axis_world']))
            remaining=max(.002,visual_remaining)
            observations.append({'control_step':state['control_steps'],'after_completed_trajectory':phase,
                'tracking':observed,'hinge_fit':fit,'remaining_angle_rad':visual_remaining,
                'rigid_tracking':rigid,
                'no_execution_veto':True})
            save(folder/'observations.json',observations)
            if visual_remaining<=.003:
                result.update(policy_goal_complete=True,observed_goal_residual_rad=visual_remaining);break
            robot=RobotVolumes(ik.chain,state['metadata']['robot_collision_volumes_world'],state['metadata']['joint_positions'])
            next_scene=scene_type(cameras,state['metadata'],new_spec,cloud,next_mask,next_choices,config,robot=robot)
            if measured_transform is not None and config.get('reconcile_handle_roles'):
                next_scene.observed_motion_transform=measured_transform
            next_scene.retain_static(scene)
            spec,mask,choices,scene=new_spec,next_mask,next_choices,next_scene
            # Refresh material features after a measured arc, before the
            # original texture becomes occluded or leaves the optical-flow basin.
            if current_pose is not None and abs(rigid.get('rotation_rad',0.))>=np.radians(8):
                rigid_tracker=RGBDHingeTracker(cameras,scene.moving_reference,prior=spec['joint_origin_world_m'],
                    reference_step=state['control_steps'],output=phase_folder/'rebased-landmark-reference.npz')
                rigid_reference_cloud=cloud.copy();last_rigid_pose=np.eye(4);last_rigid_step=state['control_steps']
                save(phase_folder/'landmark-rebase.json',{'control_step':state['control_steps'],
                    'measured_rotation_rad':rigid['rotation_rad'],'reason':'renew observed material features after an 8-degree arc'})
        else:raise ValueError('Maximum completed contact trajectories reached')
        result.update(status='completed_planned_trajectories')
        sim.observe(folder/'final')
    except Exception as error:
        result.update(status='failed',failure_stage=stage,failure=str(error))
        (folder/'error.txt').write_text(traceback.format_exc())
    finally:
        if ik:ik.close()
        sim.close()
        clear_native_cache()
        save(folder/'tracking.json',events)
        actions=read_json(folder/'initial/actions.json') if (folder/'initial/actions.json').exists() else []
        result.update(control_steps=len(actions),official_task_success=any(a['evaluation_task_success'] for a in actions),
            final_task_success=bool(actions and actions[-1]['evaluation_task_success']),episode_wall_seconds=time.perf_counter()-before,
            execution_veto_checks=sum(e['execution_veto_checks'] for e in events))
        result['pipeline_status']=result['status']
        result['task_outcome']='success' if result['official_task_success'] else 'failure'
        if result['official_task_success']:
            result['status']='task_success'
            if 'failure' in result:
                result['continuation_stop_reason']=result.pop('failure')
                result['continuation_stop_stage']=result.pop('failure_stage')
        save(folder/'summary.json',result)
        result['storage']=compact_episode(folder)
        save(folder/'summary.json',result)
    return result


def context_pad(metadata):
    from run_coupled_door import pad_local
    return pad_local(metadata)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--tasks',nargs='+',choices=['open_microwave','close_microwave'],default=['open_microwave','close_microwave'])
    parser.add_argument('--states',nargs='+',type=int,default=[1,11,21,31,41])
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--full-task-only',action='store_true',help='Require full remaining task path; disable 20/10 degree fallback')
    parser.add_argument('--backend',choices=['endpoint_menu','partial_pose_arc'],default='partial_pose_arc')
    parser.add_argument('--motion-policy',choices=['short','full','legacy'],default='short')
    parser.add_argument('--hinge-estimator',choices=['hybrid','circle'],default='hybrid')
    parser.add_argument('--reconcile-handle-roles',action='store_true')
    parser.add_argument('--replay-grasp-source',help='Explicit development diagnostic; reuse a validated grasp/approach, plan the pull afresh')
    parser.add_argument('--arc-seconds',type=float,default=60.)
    parser.add_argument('--arc-proposals',type=int,default=128)
    args=parser.parse_args()
    if not np.isfinite(args.arc_seconds) or args.arc_seconds<=0 or not 1<=args.arc_proposals<=512:
        parser.error('Use a positive finite arc budget and 1..512 joint proposals')
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=False)
    frozen=read_json(BASE/'frozen-evaluation/protocol.json')['sources_sha256']
    if any(digest(ROOT/name)!=value for name,value in frozen.items()):
        raise ValueError('Frozen baseline sources changed')
    config=read_json(ROOT/'configs/articulated_libero.json')
    config.update(complete_planar_panel=False,scene_voxel_spacing_m=.003,
        maximum_grasp_poses=24,maximum_grasp_search_seconds=240.,maximum_contact_plan_seconds=8.,
        full_task_only=args.full_task_only or args.motion_policy=='full',planning_backend=args.backend,
        motion_policy='full' if args.full_task_only else args.motion_policy,
        hinge_estimator=args.hinge_estimator,
        reconcile_handle_roles=args.reconcile_handle_roles,
        replay_grasp_source=str((ROOT/args.replay_grasp_source).resolve()) if args.replay_grasp_source else None,
        arc_search_seconds=args.arc_seconds,arc_goal_proposals=args.arc_proposals)
    if args.reconcile_handle_roles:config['handle_association_m']=.008
    tasks=[t for t in config['tasks'] if t['label'] in args.tasks]
    sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),
        ROOT/'src/vlm_tamp/mcvamp_microwave.py',ROOT/'src/vlm_tamp/timed_execution.py',
        ROOT/'src/vlm_tamp/native/mcvamp_microwave.cc',ROOT/'scripts/build_mcvamp_microwave.py']}
    sources['src/vlm_tamp/short_motion_policy.py']=digest(ROOT/'src/vlm_tamp/short_motion_policy.py')
    sources['src/vlm_tamp/rigid_hinge_tracking.py']=digest(ROOT/'src/vlm_tamp/rigid_hinge_tracking.py')
    if args.backend=='partial_pose_arc' or args.reconcile_handle_roles:
        for name in ['src/vlm_tamp/partial_pose_arc.py','src/vlm_tamp/arc_contact_scene.py',
                     'src/vlm_tamp/arc_execution.py']:
            sources[name]=digest(ROOT/name)
    for name in sources:
        dest=output/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    ledger=ROOT/'results/articulated-api-v1/ledger.json';ledger_before=digest(ledger)
    save(output/'protocol.json',{'sources_sha256':sources,'baseline_frozen_sources_sha256':frozen,
        'baseline_sources_unchanged':True,'config':config,'initial_states':args.states,'seed':0,
        'settling_actions':5,'maximum_control_steps':600,'grounding':'observation-validated frozen baseline inputs',
        'new_api_calls_allowed':False,'api_ledger_sha256_before':ledger_before,'reasoning':'high',
        'planning_backend':args.backend,'end_effector_quaternion_prescribed':args.backend!='partial_pose_arc',
        'frozen_grasp_replay_source':config['replay_grasp_source'],
        'initial_grasp_replay_is_development_diagnostic':bool(config['replay_grasp_source']),
        'native_build':read_json(ROOT/'.runtime/mcvamp-microwave/build.json'),
        'execution':'timed complete references through official joint PD; no runtime collision/residual/progress vetoes',
        'postclosure_replan_allowed':True,'replanning_during_trajectory':False,
        'horizon_policy':('full remaining task only; no bounded fallback' if config['full_task_only'] else
            '3-degree calibration motions, then 8-degree motions after observed hinge refinement; 2/1 degree fallback'
            if config['motion_policy']=='short' else 'prefer complete remaining task; otherwise 20/10 degree segments'),
        'hinge_fit_observations_only':True,
        'full_trajectory_definition':'every reference of each selected complete planned segment is issued before observing/replanning',
        'task_and_pipeline_status_separate':True,'native_goal_is_evaluation_only':True,
        'frozen_before_first_episode':True,
        'execution_thread_environment':{key:os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        'planning_collision':'native CAPT observed static/moving surfaces, calibrated gripper cover, intended finger-handle contact',
        'inferred_solid_panel_used':False,'physical_grasp_only':True,'simulator_object_state_policy_input':False,
        'comparison_limit':'planner, execution policy and collision model change together; not a planner-only speed ablation',
        'storage_retention':POLICY})
    manifest=[{'task':t['label'],'state':i} for t in tasks for i in args.states]
    save(output/'manifest.json',{'planned_episodes':manifest,'all_failures_in_denominator':True})
    results=[];resources=CoupledResources()
    for task in tasks:
        for index in args.states:
            if any(digest(ROOT/name)!=value for name,value in sources.items()):
                raise ValueError('Experiment sources changed after freezing')
            result=episode(task,index,output/f'{task["label"]}-state{index:03d}',config,resources,plan_only=args.plan_only)
            results.append(result)
            save(output/'summary.json',{'planned':len(manifest),'completed':len(results),'episodes':results,
                'native_successes':sum(r['official_task_success'] for r in results),'api_ledger_unchanged':digest(ledger)==ledger_before,
                'api_snapshot':api_snapshot(),'baseline_sources_unchanged':all(digest(ROOT/n)==h for n,h in frozen.items())})
            print(json.dumps(result,default=lambda value:value.tolist()),flush=True)

if __name__=='__main__':main()
