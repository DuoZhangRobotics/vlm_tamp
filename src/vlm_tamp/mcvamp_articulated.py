"""Fresh-grasp drawer/knob trials with MC-VAMP and full timed execution."""
import copy
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .coupled_client import CoupledClient
from .coupled_storage import read_json,write_json,compact_episode
from .cohort_execution import read_cohort_observation,legacy_grounding,limited_proposals
from .mcvamp_mechanism import NativeScene,clear_native_cache
from .mechanism_surface import SurfaceScene,refine_control_axis,observed_control_rotation
from .timed_execution import execute,schedule
from .rekep_libero.geometry import pose_matrix,matrix_pose,transform_points
from .rekep_libero.task_geometry_v5 import compile_scene
from .rekep_libero.mechanism_geometry import Mechanism,refine_contact,ground_geometry,expand_handle_surface
from .rekep_libero.task_ik_curobo import TaskIK
from .rekep_libero.physical_tasks_v4 import candidate_variants
from .rekep_libero.collision_paths_v7 import estimated_contact_aperture
from .rekep_libero.mechanism_planning import ranked_postures
from .rekep_libero.scene_state_v7 import observed_points


def metadata_at(sim,metadata,q,aperture):
    geometry=sim.fk(q,include_volumes=True,gripper_qpos=aperture)
    return {**metadata,**geometry,'joint_positions':np.asarray(q).tolist()}


def segment(scene,record,metadata,ik,q,progress,output,*,seconds=8.):
    """Find a complete constrained path; only terminal poses come from IK."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    spec={**scene.spec,'target_progress':float(progress)};mechanism=Mechanism(spec)
    target=mechanism.pose(ik.chain.forward(q),float(progress))
    attempts=[];before=time.perf_counter()
    with NativeScene(scene,record,metadata,q,progress,output=output) as context:
        initial=context.validate(q)
        if not initial['accepted']:
            result={'accepted':False,'reason':'observed_robot_collision_at_start','initial':initial,'attempts':[]}
            write_json(output/'planning.json',result);return None,result
        goals=ik.solve_many(target,q,maximum=8)
        for solution in goals:
            projected,projection=context.project(solution['joints'])
            achieved=context.progress(projected)
            if not projection['accepted'] or abs(achieved-progress)>(.001 if mechanism.kind=='prismatic' else .01):
                attempts.append({'reason':'terminal_projection','projection':projection,'progress':achieved});continue
            path,planned=context.plan(q,projected,seconds=seconds,iterations=5000)
            attempts.append(planned)
            if planned['exact_solution']:
                residuals=np.asarray([context.residual(x) for x in path])
                result={'accepted':True,'progress':progress,'terminal_progress':achieved,
                    'maximum_position_residual_m':float(np.linalg.norm(residuals[:,:3],axis=1).max()),
                    'maximum_orientation_residual_rad':float(np.linalg.norm(residuals[:,3:6],axis=1).max()/.15),
                    'native_planning':planned,'attempts':attempts,'seconds':time.perf_counter()-before}
                np.save(output/'path.npy',path);write_json(output/'planning.json',result);return path,result
            if time.perf_counter()-before>seconds*2:break
    result={'accepted':False,'reason':'no_complete_constrained_path','attempts':attempts,'seconds':time.perf_counter()-before}
    write_json(output/'planning.json',result);return None,result


def select_grasp(generated,cloud,spec,scene,record,ik,state,sim,folder,config):
    # Apply contact-direction eligibility before spending the 24-candidate budget.
    # Proximity alone can rank upside-down grasps ahead of every usable pose.
    proposals=generated
    anchor=np.asarray(spec['source_anchor_world_m']);normal=np.asarray(spec['panel_normal_world'])
    tree=cKDTree(cloud);rows=[];rejected=[]
    _,sv,vt=np.linalg.svd(cloud-cloud.mean(0),full_matrices=False)
    elongated=sv[0]>2*sv[1]
    for index,variant,pose in candidate_variants(proposals,state['metadata']['robot_calibration']):
        row={'index':index,'variant':variant,'pose':pose.tolist(),'score':float(generated['scores'][index]),
             'anchor_distance_m':float(np.linalg.norm(pose[:3]-anchor)),'source':'GraspGen Panda'}
        r=pose_matrix(pose)[:3,:3]
        if tree.query(pose[:3])[0]>.025 or row['anchor_distance_m']>.12:
            rejected.append({**row,'reason':'outside_contact_region'});continue
        if -r[:,2]@normal<.65 or (elongated and abs(r[:,0]@vt[0])>.65):
            rejected.append({**row,'reason':'contact_approach_or_closing_direction'});continue
        rows.append(row)
    rows.sort(key=lambda r:(r['anchor_distance_m'],-r['score']))
    kept=[]
    for row in rows:
        if row['index'] not in kept and len(kept)<config['maximum_candidate_grasps']:kept.append(row['index'])
    rows=[row for row in rows if row['index'] in kept]
    batches=[];q0=np.asarray(state['metadata']['joint_positions']);open_qpos=state['metadata']['gripper_qpos']
    for ordinal,row in enumerate(rows):
        solutions=ik.solve_many(row['pose'],q0,maximum=8)
        if not solutions:rejected.append({**row,'reason':'no_robot_valid_ik'})
        batches.append((ordinal,row,solutions))
    tested=0;chosen=None
    with NativeScene(scene,record,state['metadata'],q0,.03,articulated=False) as open_context:
        for ordinal,row,rank,solution in ranked_postures(batches):
            if tested>=config['maximum_total_ik_postures']:break
            tested+=1;q=np.asarray(solution['joints']);evidence={**row,'ik_rank':rank,'joints':q.tolist()}
            write_json(folder/'grasp-search-progress.json',{'tested_ik_postures':tested,'eligible_poses':len(rows),'candidate_index':row['index']})
            opened=open_context.validate(q)
            if not opened['accepted']:
                rejected.append({**evidence,'reason':'open_grasp_scene_collision','diagnostic':opened});continue
            aperture=estimated_contact_aperture(sim,q,cloud,open_qpos)
            if aperture is None:
                rejected.append({**evidence,'reason':'no_observed_bilateral_pad_aperture'});continue
            closed_metadata=metadata_at(sim,state['metadata'],q,aperture)
            with NativeScene(scene,record,closed_metadata,q,.03,articulated=False) as closed_context:
                closed=closed_context.validate(q)
            if not closed['accepted']:
                rejected.append({**evidence,'reason':'closure_scene_collision','diagnostic':closed});continue
            trial=folder/f'candidate-{tested:03d}';trial.mkdir()
            probe=.025 if spec['joint_family']=='prismatic' else np.deg2rad(6)
            probe_path,probe_info=segment(scene,record,closed_metadata,ik,q,probe,trial/'probe',seconds=4.)
            if probe_path is None:
                rejected.append({**evidence,'reason':'manipulation_probe_not_plannable','planning':probe_info});continue
            # Search the free approach rather than rejecting a straight joint interpolation.
            target=np.asarray(solution['pose_world_xyzw']);pre=target.copy()
            pre[:3]-=.07*pose_matrix(target)[:3,2]
            pre_solutions=ik.solve_many(pre,q,maximum=4);approach=None;approach_attempts=[]
            for pre_solution in pre_solutions:
                pre_q=np.asarray(pre_solution['joints'])
                first,info=open_context.plan(q0,pre_q,constrained=False,seconds=6.,iterations=5000)
                approach_attempts.append(info)
                if not info['exact_solution']:continue
                last,info=open_context.plan(pre_q,q,constrained=False,seconds=4.,iterations=3000)
                approach_attempts.append(info)
                if info['exact_solution']:
                    approach=np.vstack([first,last[1:]]);break
            write_json(trial/'approach.json',{'attempts':approach_attempts})
            if approach is None:
                rejected.append({**evidence,'reason':'no_complete_approach','attempts':approach_attempts});continue
            chosen={**evidence,'solution_joints':q.tolist(),'tcp_pose_world_xyzw':target.tolist(),
                'predicted_gripper_qpos':aperture,'tested_ik_postures':tested,'fresh_grasp':True,
                'probe':probe_info,'approach_planner':'VAMP RRTC','manipulation_planner':'MC-VAMP CRRTC'}
            np.save(folder/'approach-path.npy',approach);break
    write_json(folder/'grasp-admission.json',{'selected':chosen,'rejected_candidates':rejected,
        'tested_ik_postures':tested,'eligible_poses':len(rows),'posture_budget':config['maximum_total_ik_postures']})
    if chosen is None:raise ValueError('No candidate with complete grasp approach and constrained probe within search budget')
    return chosen,np.load(folder/'approach-path.npy')


def run_episode(task,index,folder,config,resources):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    sim=CoupledClient(folder/'simulator.log');ik=None;state=None;phases=[];events=[];observations=[];stage='reset'
    before=time.perf_counter();calls_before=resources.new_api_calls
    result={'task':task['label'],'official_task_name':task['name'],'suite':task['suite'],'initial_state':index,'seed':0,
        'status':'failed','official_task_success':False,'planner':'MC-VAMP CRRTC with measured-surface collision',
        'execution_veto_checks':0,'execution_time_collision_checks':False,'execution_time_constraint_checks':False,
        'execution_time_progress_vetoes':False,'fixed_grasp_replay':False,'assisted_attachment':False,
        'physical_contacts_enabled':True,'new_api_calls':0,'moving_solid_hull_used':False}
    def issue(path,label,gripper=1,hold=6,reserve=0):
        nonlocal state
        path=np.asarray(path,float)
        if len(path)==1:path=np.vstack([path,path])
        refs=schedule(path,maximum_joint_speed=.35,endpoint_hold_steps=hold)
        available=600-int(state['control_steps'])-reserve
        if available<=hold:raise ValueError('Insufficient budget for another complete trajectory')
        if len(refs)>available:
            speed=.35*(len(refs)-hold)/(available-hold)*1.001
            refs=schedule(path,maximum_joint_speed=speed,endpoint_hold_steps=hold)
        out=folder/f'path-{len(phases):02d}-{label}';out.mkdir()
        np.save(out/'path.npy',path);np.save(out/'references.npy',refs)
        state,done=execute(sim,state,refs,gripper=gripper,event_sink=events.append)
        phases.append({'label':label,**{k:v for k,v in done.items() if k!='events'}})
        write_json(folder/'phases.json',phases)
    try:
        state=sim.reset(task['name'],index,folder/'initial',suite_name=task['suite'])
        state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,output_dir=folder/'opened')
        stage='rgb_grounding'
        if 'legacy_index' in task:
            bundle,program,source,mask=legacy_grounding(task['legacy_index'],index,folder/'opened',config,folder)
            spec,cloud,_,_,masks=compile_scene(bundle,program,source,mask,config);cameras=bundle['cameras']
            world=cameras[bundle['main_camera']]['world_points'];panel=masks['parent']&(np.abs(world[:,:,2]-np.median(cloud[:,2]))<.035)
            spec.update(joint_family='prismatic',joint_axis_world=spec['pull_axis_world'],
                joint_origin_world_m=spec['source_anchor_world_m'],target_progress=config['drawer_maximum_command_travel_m'],
                panel_normal_world=spec['pull_axis_world'],camera=bundle['main_camera'],panel_depth_association_m=.020)
        else:
            choices=resources.grounding(task['label'],index,folder/'opened',state['metadata'],folder/'grounding')
            cameras,metadata=read_cohort_observation(folder/'opened/obs.npz',state['metadata']);name=choices['camera']
            with np.load(folder/'grounding'/name/'keypoints.npz') as a:bundle={k:a[k].copy() for k in a.files}
            bundle.update(cameras=cameras,meta=metadata,main_camera=name)
            stage='numerical_geometry'
            cloud,mask=refine_contact(bundle,choices,resources.ensure_perception(),folder)
            mechanism,panel=ground_geometry(bundle,choices,cloud,mask,resources.ensure_perception(),folder)
            cloud=expand_handle_surface(cameras,mechanism.spec,choices,cloud);spec=mechanism.spec
            if spec['motion']=='turn_on':spec,cloud=refine_control_axis(cameras,choices,spec,cloud)
            spec['source_anchor_world_m']=np.median(cloud,axis=0).tolist()
        if spec.get('motion')=='close' and spec['joint_family']=='prismatic':
            spec['nominal_target_progress']=spec['target_progress']
            spec['closing_seating_allowance_m']=.005
            spec['target_progress']+=.005
        write_json(folder/'task-spec.json',spec);np.save(folder/'grasp-cloud.npy',cloud)
        generated=resources.grasps(cloud,folder/'grasps');record=sim.robot_chain();write_json(folder/'robot-chain.json',record)
        import vamp
        empty=vamp.Environment();ik=TaskIK(record,config,lambda q:vamp.panda.validate(np.asarray(q).tolist(),empty,True),output=folder,sim=sim)
        scene=SurfaceScene(cameras,state['metadata'],spec,cloud,panel,config);write_json(folder/'scene-scope.json',scene.record())
        stage='fresh_grasp_and_mcvamp_planning';chosen,approach=select_grasp(generated,cloud,spec,scene,record,ik,state,sim,folder,config)
        write_json(folder/'chosen-grasp.json',chosen)
        state=sim.enable_joint_tracking(kp=300.);write_json(folder/'controller-switch.json',state['controller_change'])
        stage='timed_approach';issue(approach,'approach',-1,hold=20,reserve=180)
        stage='physical_closure';q=np.asarray(chosen['solution_joints']);issue([q,q],'close',1,hold=40,reserve=140)
        state=sim.observe(folder/'closed');cameras,_=read_cohort_observation(folder/'closed/obs.npz',state['metadata'])
        scene=SurfaceScene(cameras,state['metadata'],spec,cloud,panel,config,previous=scene)
        initial_pose=np.asarray(state['metadata']['eef_pose_world_xyzw']);commanded=0.;plan_number=0
        # Unknown knob direction is tested physically. The native joint value is
        # absent here; it is read only after all execution for evaluation.
        if spec['joint_family']=='revolute':
            stage='control_direction_probe';base_axis=np.asarray(spec['joint_axis_world']);reference=scene.contact_reference.copy()
            outcomes=[];current_offset=0.
            for sign in [1.,-1.]:
                amount=np.deg2rad(8.)-sign*current_offset
                local_spec={**spec,'joint_axis_world':(sign*base_axis).tolist()}
                scene.spec=local_spec;scene.axis=sign*base_axis
                q=np.asarray(state['metadata']['joint_positions']);meta=state['metadata']
                path,info=segment(scene,record,meta,ik,q,amount,folder/f'plan-{plan_number:02d}',seconds=8.);plan_number+=1
                if path is None:
                    outcomes.append({'sign':sign,'planned':False,'planning':info});continue
                issue(path,f'probe-{int(sign):+d}',hold=12,reserve=90)
                state=sim.observe(folder/f'probe-{int(sign):+d}');views,_=read_cohort_observation(folder/f'probe-{int(sign):+d}/obs.npz',state['metadata'])
                observed,_=observed_points(views,state['metadata'],spacing=.002,robot_depth_tolerance=config['robot_self_depth_tolerance_m'])
                tracking=observed_control_rotation(reference,observed,spec['joint_origin_world_m'],base_axis)
                outcomes.append({'sign':sign,'planned':True,'tracking':tracking})
                current_offset=float(Rotation.from_matrix(pose_matrix(state['metadata']['eef_pose_world_xyzw'])[:3,:3]@pose_matrix(initial_pose)[:3,:3].T).as_rotvec()@base_axis)
                # Re-associate from the initial scene using observed endpoints.
                scene=SurfaceScene(views,state['metadata'],spec,cloud,panel,config,previous=scene,
                    transform=Mechanism({**local_spec,'target_progress':amount}).transform(amount))
                if tracking.get('valid') and sign*tracking['angle_rad']>np.deg2rad(3):break
            supported=[o for o in outcomes if o.get('planned') and o['tracking'].get('valid') and o['sign']*o['tracking']['angle_rad']>np.deg2rad(3)]
            direction=supported[-1]['sign'] if supported else next((o['sign'] for o in reversed(outcomes) if o.get('planned')),None)
            if direction is None:raise ValueError('Neither bounded direction probe has a complete plan')
            result['control_direction_observed']=bool(supported)
            result['control_direction_reversed']=bool(direction<0)
            spec={**spec,'joint_axis_world':(direction*base_axis).tolist()};scene.spec=spec;scene.axis=direction*base_axis
            commanded=max(0.,direction*current_offset)
            write_json(folder/'direction-probe.json',{'attempts':outcomes,'selected_sign':direction,
                'observed_direction_supported':bool(supported),'fallback':'bounded opposite-direction exploration',
                'simulator_task_success_or_joint_state_used':False})
        chunk=.025 if spec['joint_family']=='prismatic' else np.deg2rad(8.)
        while commanded<spec['target_progress']-1e-6:
            amount=min(chunk,spec['target_progress']-commanded)
            stage='mcvamp_planning_between_complete_trajectories';q=np.asarray(state['metadata']['joint_positions'])
            path,info=segment(scene,record,state['metadata'],ik,q,amount,folder/f'plan-{plan_number:02d}',seconds=8.);plan_number+=1
            if path is None:raise ValueError('No complete MC-VAMP mechanism segment: '+info['reason'])
            stage='timed_mechanism';issue(path,f'manipulate-{plan_number:02d}',hold=6)
            commanded+=amount;state=sim.observe(folder/f'observation-{plan_number:02d}')
            views,_=read_cohort_observation(folder/f'observation-{plan_number:02d}/obs.npz',state['metadata'])
            transform=Mechanism({**spec,'target_progress':amount}).transform(amount)
            scene=SurfaceScene(views,state['metadata'],spec,cloud,panel,config,previous=scene,transform=transform)
            observations.append({'control_step':state['control_steps'],'commanded_progress':commanded,'execution_veto':False})
            write_json(folder/'observations.json',observations)
        result['status']='completed_planned_trajectories';sim.observe(folder/'final')
    except Exception as error:
        result.update(failure_stage=stage,failure=str(error));(folder/'error.txt').write_text(traceback.format_exc())
    finally:
        if ik:ik.close()
        sim.close();clear_native_cache()
        actions=read_json(folder/'initial/actions.json') if (folder/'initial/actions.json').exists() else []
        result.update(official_task_success=any(a['evaluation_task_success'] for a in actions),
            final_task_success=bool(actions and actions[-1]['evaluation_task_success']),control_steps=len(actions),
            completed_trajectories=len(phases),all_issued_trajectories_completed=all(p['all_commands_issued'] for p in phases),
            new_api_calls=resources.new_api_calls-calls_before,episode_wall_seconds=time.perf_counter()-before,
            pipeline_status=result['status'])
        if result['official_task_success']:result['status']='task_success'
        write_json(folder/'tracking.json',events);write_json(folder/'observations.json',observations);write_json(folder/'summary.json',result)
        result['storage']=compact_episode(folder);write_json(folder/'summary.json',result)
        # Newly generated cache duplicates are unnecessary once exact proposals
        # and checkpoint provenance have been retained inside this episode.
        resources.cleanup_proposals()
    return result
