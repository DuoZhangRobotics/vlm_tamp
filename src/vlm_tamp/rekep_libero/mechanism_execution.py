"""Physical grasp/line/hinge execution with sensor-only mechanism feedback."""

import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image

from try_rekep_bowl_v3 import read_observation_v3, save
from .articulated_grounding import ground
from .collision_paths_v7 import check_joint_path
from .geometry import transform_points
from .mechanism_geometry import ground_geometry, observe_progress,refine_contact,expand_handle_surface
from .mechanism_planning import choose_grasp, mechanism_path_check,panel_probe_admission
from .mechanism_scene import MechanismScene
from .object_motion_v3 import observed_object_cloud
from .object_tracking_v4 import register_closed_surface
from .perception import _read_observation
from .sim_client_articulated import ArticulatedClient
from .task_ik_curobo import TaskIK

ROOT=Path(__file__).resolve().parents[3]


def cached_grounding(source,observation,metadata,output):
    source=Path(source)
    record=json.loads((source/'grounding.json').read_text())
    choices=record['choices']
    cameras,meta=_read_observation(Path(observation)/'obs.npz',metadata)
    name=choices['camera']
    reference=np.asarray(Image.open(source/name/'rgb.png'))
    differences=np.abs(cameras[name]['rgb'].astype(float)-reference)
    if differences.mean()>.25 or differences.max()>16:
        raise ValueError('Prototype RGB cache does not match current observation')
    shutil.copytree(source,output)
    with np.load(source/name/'keypoints.npz') as data:
        bundle={key:data[key].copy() for key in data.files}
    mask=np.load(source/'contact-mask.npy')
    cloud=np.load(source/'contact-cloud.npy')
    # Verify selected sensor geometry before reusing the paid RGB role decision.
    from scipy.spatial import cKDTree
    support=cameras[name]['world_points'][mask]
    if np.quantile(cKDTree(cloud).query(support)[0],.95)>.008:
        raise ValueError('Prototype depth cache does not match current selected surface')
    bundle.update(cameras=cameras,meta={**meta,'main_camera':name},main_camera=name)
    return bundle,choices,cloud,mask


def run_episode(label,task,state_index,output,config,perception,graspgen,development=False,reuse_run=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    sim=ArticulatedClient(output/'simulator.log')
    summary={'task':label,'official_task_name':task,'suite':'libero_90','initial_state':state_index,
        'seed':0,'version':config['version'],'development':development,'official_task_success':False,
        'grasp_retention_verified':False,'failure_stage':None,'status':'running',
        'curobo_ik_used':False,'curobo_motiongen_used':False,'simulator_scene_state_used':False,
        'assisted_grasping':False,'physical_contacts_enabled':True,'collision_free_guarantee':False}
    summary.update(scene_mode=config.get('scene_mode','strict'),
                   software_scene_admission_enforced=config.get('scene_mode','strict')=='strict')
    stage,start='reset',time.monotonic()
    commands,checks,feedback=[],[],[]
    ik=None
    try:
        state=sim.reset(task,state_index,output/'initial')
        state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,output_dir=output/'opened')
        stage='grounding'
        if reuse_run is not None:
            reuse_folder=Path(reuse_run)/f'{label}-state{state_index:03d}'
            bundle,choices,cloud,mask=cached_grounding(reuse_folder/'grounding',output/'opened',state['metadata'],output/'grounding')
            summary['grounding_reused_from_strict_pilot']=True
        elif development and state_index==0:
            source=ROOT/f'results/articulated-development/grounding-v3/{label}'
            bundle,choices,cloud,mask=cached_grounding(source,output/'opened',state['metadata'],output/'grounding')
            summary['grounding_response_reused_from_prototype']=True
        else:
            bundle,choices,cloud,mask=ground(perception,output/'opened',state['metadata'],output/'grounding',
                                          f'{config["version"]}-{label}-state{state_index:03d}')
        stage='numerical_geometry'
        cloud,mask=refine_contact(bundle,choices,perception,output)
        mechanism,panel_mask=ground_geometry(bundle,choices,cloud,mask,perception,output)
        before_expansion=len(cloud)
        cloud=expand_handle_surface(bundle['cameras'],mechanism.spec,choices,cloud)
        mechanism.spec['source_anchor_world_m']=np.median(cloud,axis=0).tolist()
        save(output/'task-spec.json',mechanism.spec)
        save(output/'contact-refinement.json',{'initial_rgb_surface_points':before_expansion,
            'multi_view_contact_points':len(cloud),'uses_vlm_contact_pixel':True,
            'selected_rgb_box_padding_pixels':3,'minimum_observed_panel_protrusion_m':.004,
            'maximum_observed_panel_protrusion_m':.06,'simulator_scene_state_used':False})
        np.save(output/'grasp-cloud.npy',cloud)
        np.save(output/'contact-mask.npy',mask)
        spec=mechanism.spec
        summary.update(joint_family=mechanism.kind,motion=spec['motion'],target_progress=mechanism.goal,
                       progress_units=spec['progress_units'],selected_camera=choices['camera'])
        stage='grasp_generation'
        if reuse_run is not None:
            from .cache_evidence_v7 import grasp_proposal_agreement
            evidence=grasp_proposal_agreement(cloud,np.load(reuse_folder/'grasp-cloud.npy'),config)
            save(output/'grasp-cache-evidence.json',evidence)
            if not evidence['accepted']:
                raise ValueError('Paired grasp input does not match strict pilot')
            shutil.copytree(reuse_folder/'grasps',output/'grasps')
            with np.load(output/'grasps/grasp_candidates.npz') as data:
                generated={key:data[key].copy() for key in ['poses_world_base','scores','retained_indices']}
            summary['graspgen_candidates_reused']=True
        else:
            generated=graspgen.generate(cloud,output/'grasps')
        record=sim.robot_chain()
        save(output/'robot-chain.json',record)
        import vamp
        environment=vamp.Environment()
        self_valid=lambda q:vamp.panda.validate(np.asarray(q).tolist(),environment,True)
        stage='ik_calibration'
        ik=TaskIK(record,config,self_valid,output=output,sim=sim)
        summary['curobo_ik_used']=True
        cameras,_=read_observation_v3(output/'opened',state['metadata'])
        stage='scene_reconstruction'
        scene=MechanismScene(cameras,state['metadata'],spec,cloud,panel_mask,config)
        scene.save(output/'initial-scene.npz')
        save(output/'scene-scope.json',scene.record())
        initial=scene.check(state['metadata']['robot_collision_volumes_world'],state['metadata']['eef_pose_world_xyzw'])
        save(output/'initial-clearance.json',initial)
        if not initial['accepted']:
            raise ValueError('Initial measured robot posture fails observed scene admission')
        stage='moving_geometry_admission'
        panel_check=panel_probe_admission(scene,state['metadata'],mechanism)
        save(output/'moving-panel-probe.json',panel_check)
        if not panel_check['accepted']:
            raise ValueError('Predicted moving-panel envelope overlaps stationary surfaces; the observed geometry model rejects the motion')
        stage='grasp_admission'
        chosen=choose_grasp(generated,cloud,state['metadata']['robot_calibration'],ik,state['metadata']['joint_positions'],
            sim,scene,mechanism,state['metadata']['gripper_qpos'],output,save)
        save(output/'chosen-grasp.json',chosen)
        summary.update(selected_grasp_index=chosen['index'],selected_grasp_variant=chosen['variant'],
                       selected_ik_alternative=chosen['ik_alternative'],tested_ik_postures=chosen['tested_ik_postures'])
        closure_transform=np.eye(4)
        context={'phase':'approach','transform':lambda pose:np.eye(4),
                 'stage':lambda pose:'grasp' if np.linalg.norm(np.asarray(pose)[:3]-np.asarray(chosen['fk_pose_world_xyzw'])[:3])<=.012 else 'approach'}

        def actual_check(label_name):
            pose=state['metadata']['eef_pose_world_xyzw']
            checked=scene.check(state['metadata']['robot_collision_volumes_world'],pose,
                stage=context['stage'](pose),moving_transform=context['transform'](pose),
                check_payload=context['phase']=='pull')
            checks.append({'kind':'actual_action_boundary','label':label_name,'control_steps':state['control_steps'],'result':checked})
            save(output/'collision-checks.json',checks)
            if not checked['accepted']:
                raise ValueError('Actual measured posture fails contact-aware scene admission')

        def move(pose,joints,steps,gripper=0,folder=None):
            nonlocal state
            remaining=600-state['control_steps']
            if remaining<=0:
                raise ValueError('600-control-step budget exhausted')
            checked=check_joint_path(scene,sim,ik,[state['metadata']['joint_positions'],joints],
                gripper_qpos=state['metadata']['gripper_qpos'],transform_at_pose=context['transform'],
                stage_at_pose=context['stage'],check_payload=context['phase']=='pull',config=config)
            checks.append({'kind':'command_preflight','label':stage,'result':checked,'control_steps':state['control_steps']})
            if not checked['accepted']:
                save(output/'collision-checks.json',checks)
                raise ValueError('Requested motion fails sampled contact-aware scene admission')
            state=sim.action(pose,gripper=gripper,max_steps=min(steps,remaining),nullspace_joint_target=joints,
                nullspace_kp=config['nullspace_kp'],output_dir=output/folder if folder else None)
            commands.append({'stage':stage,'requested_pose_world_xyzw':np.asarray(pose).tolist(),
                'requested_joint_posture':joints,'actual_pose_world_xyzw':state['metadata']['eef_pose_world_xyzw'],
                'actual_joints':state['metadata']['joint_positions'],'control_steps':state['control_steps'],
                'pose_reached':state['reached_pose']})
            save(output/'progress.json',{'stage':stage,'control_steps':state['control_steps']})
            if context['phase']=='approach':
                actual_check(stage)

        stage='approach_execution'
        approach=chosen['approach']
        for pose,q in zip(approach['poses_world_xyzw'][1:],approach['joint_postures'][1:]):
            move(pose,q,8)
        context.update(phase='grasp',stage=lambda pose:'grasp')
        grasp=chosen['fk_pose_world_xyzw']
        q=chosen['solution_joints']
        move(grasp,q,20,gripper=-1,folder='at-grasp')
        stage='grasp_closure'
        move(grasp,q,60,gripper=1,folder='closed')
        closed_pose=np.asarray(state['metadata']['eef_pose_world_xyzw'])
        stage='closure_tracking'
        cameras,_=read_observation_v3(output/'closed',state['metadata'])
        observed=observed_object_cloud(cameras,cloud,np.eye(4),state['metadata']['robot_collision_volumes_world'])
        closure=register_closed_surface(cloud,observed)
        save(output/'closure-registration.json',closure)
        if not closure['verified']:
            raise ValueError('Selected contact surface cannot be tracked through closure')
        closure_transform=np.asarray(closure['transform_initial_to_closed_world'])
        closed_cloud=transform_points(cloud,closure_transform)
        scene.refresh(cameras,state['metadata'],closure_transform)
        scene.save(output/'scene-closed.npz')
        context.update(transform=lambda pose:closure_transform)
        actual_check('observed_closure')
        stage='manipulation_preflight'
        poses=mechanism.path(closed_pose,end=min(mechanism.goal,.03 if mechanism.kind=='prismatic' else .20))
        path=ik.path(poses[1:],state['metadata']['joint_positions'])
        if not path['accepted']:
            save(output/'postclosure-path.json',path)
            raise ValueError('Closed grasp has no continuous bounded IK path along the inferred mechanism')
        qs=[state['metadata']['joint_positions']]+[row['joints'] for row in path['solutions']]
        checked=mechanism_path_check(scene,sim,ik,qs,mechanism,closed_pose,state['metadata']['gripper_qpos'],closure_transform)
        save(output/'postclosure-path.json',{'path':path,'check':checked})
        if not checked['accepted']:
            raise ValueError('Postclosure mechanism path fails observed collision/manifold admission')
        context.update(phase='pull',stage=lambda pose:'pull',
            transform=lambda pose:mechanism.transform(mechanism.progress_of_pose(closed_pose,pose))@closure_transform)
        stage='mechanism_execution'
        previous,previous_hand=0.,0.
        radius=max(.03,float(np.linalg.norm(np.median(cloud,axis=0)-mechanism.origin)))
        chunk=.025 if mechanism.kind=='prismatic' else .20
        count=max(1,int(np.ceil(mechanism.goal/chunk)))
        current=0.
        for index,target in enumerate(np.linspace(0.,mechanism.goal,count+1)[1:]):
            subposes=mechanism.path(closed_pose,start=current,end=float(target))[1:]
            for pose in subposes:
                solution=ik.solve(pose,state['metadata']['joint_positions'],seed_count=2)
                if solution is None:
                    raise ValueError('Measured robot state has no IK continuation along the mechanism')
                move(solution['pose_world_xyzw'],solution['joints'],8,gripper=1)
                residual=mechanism.residual(closed_pose,state['metadata']['eef_pose_world_xyzw'])
                if residual['position_residual_m']>.015 or residual['orientation_residual_rad']>.10:
                    raise ValueError('Physical controller departed from the estimated mechanism manifold')
            current=float(target)
            folder=f'mechanism-observation-{index:02d}'
            state=sim.observe(output/folder)
            cameras,_=read_observation_v3(output/folder,state['metadata'])
            hand=mechanism.progress_of_pose(closed_pose,state['metadata']['eef_pose_world_xyzw'])
            measured=observe_progress(closed_cloud,mechanism,cameras,state['metadata']['robot_collision_volumes_world'],
                                      max(0.,min(hand,mechanism.goal+.07)),previous)
            measured.update(control_steps=state['control_steps'],commanded_progress=current)
            feedback.append(measured)
            save(output/'mechanism-feedback.json',feedback)
            if not measured['tracking_valid']:
                raise ValueError('RGB-D mechanism tracking failed or its angle is unobservable')
            tolerance=.015 if mechanism.kind=='prismatic' else .015/radius
            if abs(measured['hand_surface_error'])>tolerance:
                raise ValueError('Observed mechanism did not follow the hand; retention not established')
            gap=sum(abs(v) for v in state['metadata']['gripper_qpos'])
            if not .002<gap<.079:
                raise ValueError('Gripper gap does not support retained contact')
            summary['grasp_retention_verified']=True
            previous=float(measured['observed_progress'])
            previous_hand=hand
            transform=mechanism.transform(previous)@closure_transform
            scene.refresh(cameras,state['metadata'],transform)
            scene.save(output/f'scene-{index:02d}.npz')
            context['transform']=lambda pose,baseline=previous,hand0=previous_hand:mechanism.transform(
                baseline+mechanism.progress_of_pose(closed_pose,pose)-hand0)@closure_transform
            actual_check('observed_mechanism_update')
        summary.update(observed_progress=previous,actual_hand_progress=previous_hand,
                       status='completed_commands')
        sim.observe(output/'final')
    except Exception as error:
        summary.update(status='failed',failure_stage=stage,failure=str(error),exception_type=type(error).__name__)
        (output/'error-traceback.txt').write_text(traceback.format_exc())
    finally:
        try:
            if ik is not None:
                ik.close()
        finally:
            closed=sim.close()
        summary.update(control_steps=closed.get('control_steps',0),video=closed.get('video_path'),
                       elapsed_seconds=time.monotonic()-start)
        actions=json.loads((output/'initial/actions.json').read_text()) if (output/'initial/actions.json').exists() else []
        summary['official_task_success']=any(row['evaluation_task_success'] for row in actions)
        summary['final_task_success']=bool(actions and actions[-1]['evaluation_task_success'])
        save(output/'command-log.json',commands)
        save(output/'collision-checks.json',checks)
        save(output/'mechanism-feedback.json',feedback)
        if 'scene' in locals():
            save(output/'collision-prediction-events.json',scene.prediction_events)
            save(output/'final-scene-scope.json',scene.record())
        save(output/'summary.json',summary)
        print(json.dumps(summary),flush=True)
    return summary
