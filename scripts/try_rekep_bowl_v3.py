"""Cached-grounding physical bowl pilot with calibrated multiview geometry."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def save(path, value):
    def clean(item):
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return str(item)
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(x) for x in item]
        if isinstance(item, Path):
            return str(item)
        return item
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), indent=2) + '\n')


def read_observation_v3(observation,metadata):
    from vlm_tamp.rekep_libero.perception import _read_observation
    cameras,metadata=_read_observation(Path(observation)/'obs.npz',metadata)
    with np.load(Path(observation)/'obs.npz',allow_pickle=False) as arrays:
        for name,camera in cameras.items():
            key=name+'_robot_depth_prior'
            if key in arrays.files:
                camera['known_robot_depth_prior']=arrays[key]
    return cameras,metadata


def snapshot_protocol(output,args):
    """Save exact code/configuration used by each new development attempt."""
    source_files=list((ROOT/'src/vlm_tamp/rekep_libero').glob('*.py'))
    source_files += [Path(__file__),ROOT/'configs/rekep_libero_v2.json',
                     ROOT/'requirements-rekep-libero.txt']
    hashes={}
    for source in source_files:
        relative=source.relative_to(ROOT)
        destination=output/'source-snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    packages={}
    for name in ('numpy','scipy','torch','open3d','trimesh','timm','Pillow'):
        try:
            packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name]=None
    frozen=json.loads((ROOT/'results/rekep-libero-v2/frozen-protocol.json').read_text())
    save(output/'protocol.json',{'development_only':True,'configuration':vars(args),
         'sources_sha256':hashes,'worker_python':sys.executable,'worker_packages':packages,
         'upstream_frozen_protocol':str(ROOT/'results/rekep-libero-v2/frozen-protocol.json'),
         'pinned_upstreams':{k:v for k,v in frozen.items() if k!='files_sha256'},
         'new_api_calls':0,'official_benchmark_repeated':False})


def execution_clearance(state,calibration,field,target_cloud):
    """Check the actual observed robot posture independently of its IK target."""
    from vlm_tamp.rekep_libero.geometry import pose_matrix,transform_points
    from vlm_tamp.rekep_libero.grasp_v3 import _collision,contact_face_flags
    metadata=state['metadata']
    tool=np.asarray(calibration['tool_collision_points_tcp'])
    faces,_=contact_face_flags(calibration)
    tool_check=_collision(field,transform_points(tool,pose_matrix(metadata['eef_pose_world_xyzw'])),
                          faces,target_cloud)
    arm=np.asarray(metadata['robot_collision_points_world'])
    arm_faces=np.asarray(metadata.get('robot_collision_contact_face_flags',np.zeros(len(arm),bool)),bool)
    arm_check=_collision(field,arm,arm_faces,target_cloud)
    return {'accepted':tool_check['accepted'] and arm_check['accepted'],
            'control_steps':state['control_steps'],'actual_pose_world_xyzw':metadata['eef_pose_world_xyzw'],
            'actual_joints':metadata['joint_positions'],'tool':tool_check,'arm':arm_check}


def cached_bundle(source, observation):
    metadata = json.loads((observation / 'metadata.json').read_text())
    cameras, metadata = read_observation_v3(observation,metadata)
    with np.load(source / 'perception/keypoints.npz', allow_pickle=False) as data:
        bundle = {name: data[name] for name in data.files}
    bundle.update(cameras=cameras, meta=metadata, main_camera='agentview',
                  numbered_image=str(source / 'perception/numbered.png'))
    reference = np.asarray(Image.open(source / 'open-gripper/agentview.png'))
    observed = cameras['agentview']['rgb']
    selected_mask = np.load(source / 'stage1-grasps/selected_mask.npy', allow_pickle=False)
    exact = np.array_equal(reference, observed)
    source_matches = np.array_equal(reference[selected_mask], observed[selected_mask])
    target_y, target_x = bundle['pixels_yx'][16]
    target_patch = np.s_[max(0,target_y-8):target_y+9, max(0,target_x-8):target_x+9]
    target_matches = np.array_equal(reference[target_patch], observed[target_patch])
    if not (source_matches and target_matches):
        raise RuntimeError('Cached bowl/plate evidence changed; no new API request is permitted in this pilot')
    points = cameras['agentview']['world_points']
    bundle['points_world'] = points[tuple(bundle['pixels_yx'].T)]
    return bundle, {'full_primary_rgb_identical': exact, 'source_mask_rgb_identical': source_matches,
                    'plate_anchor_rgb_identical': target_matches, 'source_keypoint':22, 'plate_keypoint':16,
                    'new_api_calls':0, 'manually_rebound_landmarks':False}


def source_cloud(bundle, source):
    from scipy.spatial import cKDTree
    mask = np.load(source / 'stage1-grasps/selected_mask.npy', allow_pickle=False)
    cloud = bundle['cameras']['agentview']['world_points'][mask]
    reference = cloud.copy()
    tree = cKDTree(cloud)
    for name, camera in bundle['cameras'].items():
        if name == 'agentview':
            continue
        xyz = camera['world_points'].reshape(-1,3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        cloud = np.r_[cloud, xyz[tree.query(xyz)[0] <= .008]]
    # The RGB-selected object's measured extent transfers the region to the
    # additional views. A thin observed support-plane band is excluded so
    # nearby tabletop pixels cannot become target-contact exceptions.
    low, high = reference.min(axis=0)-.002, reference.max(axis=0)+.002
    low[2] = reference.min(axis=0)[2]+.0015
    for camera in bundle['cameras'].values():
        xyz = camera['world_points'].reshape(-1,3)
        inside=np.isfinite(xyz).all(axis=1)&(xyz>=low).all(axis=1)&(xyz<=high).all(axis=1)
        cloud=np.r_[cloud,xyz[inside]]
    rounded = np.floor(cloud / .002).astype(int)
    return cloud[np.unique(rounded,axis=0,return_index=True)[1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/rekep-bowl-pilot-v3/attempt-01')
    parser.add_argument('--filter-only',action='store_true')
    parser.add_argument('--candidate-limit',type=int,default=100)
    parser.add_argument('--candidate-indices',type=int,nargs='+',default=None,
                        help='Development subset of cached detector indices; does not constitute benchmark coverage')
    parser.add_argument('--approach-limit',type=int,default=12)
    parser.add_argument('--waypoint-steps',type=int,default=8)
    parser.add_argument('--servo-scale',type=float,default=1.)
    parser.add_argument('--follow-ik-posture',action='store_true')
    parser.add_argument('--precise-controller',action='store_true')
    parser.add_argument('--closure-steps',type=int,default=120)
    parser.add_argument('--nullspace-kp',type=float,default=10.)
    args=parser.parse_args()
    from vlm_tamp.rekep_libero.geometry import pose_matrix, interpolate_path
    from vlm_tamp.rekep_libero.grasp_v3 import (build_collision_fields, LocalObservedField,
        filter_candidates, validate_pose_path, _collision)
    from vlm_tamp.rekep_libero.sim_client_v3 import SimClientV3

    output=ROOT/args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve an existing pilot; choose a new output directory')
    output.mkdir(parents=True)
    snapshot_protocol(output,args)
    source=ROOT/'results/rekep-libero-v2/libero_goal-08-state001-seed0'
    sim=SimClientV3(log_path=output/'simulator.log')
    summary={'version':'bowl-pilot-3.0','task_index':8,'initial_state':1,'seed':0,
             'new_api_calls':0,'official_task_success':False,'physical_grasp_attempts':0,
             'grasp_lift_success':False,'cached_grounding_source':str(source),
             'provided_inventory':False,'assisted_grasping':False,'status':'running'}
    summary['controller_configuration']={'waypoint_steps':args.waypoint_steps,
                                         'servo_scale':args.servo_scale,
                                         'follow_validated_ik_posture':args.follow_ik_posture,
                                         'precise_controller':args.precise_controller,
                                         'closure_steps':args.closure_steps,
                                         'nullspace_kp':args.nullspace_kp}
    precision=({'position_tolerance':.001,'orientation_tolerance':.01,'posture_tolerance':.01}
               if args.precise_controller else {})
    before=time.monotonic()
    try:
        state=sim.reset(8,1,output/'initial',seed=0)
        state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,
                         output_dir=output/'opened')
        bundle,evidence=cached_bundle(source,output/'opened')
        save(output/'cached-evidence.json',evidence)
        from vlm_tamp.rekep_libero.target_region_v3 import infer_multiview_target_cloud,load_mobile_sam_predictor
        source_mask=np.load(source/'stage1-grasps/selected_mask.npy',allow_pickle=False)
        cloud,target_masks,region=infer_multiview_target_cloud(bundle,source_mask,
                         load_mobile_sam_predictor(),output/'target-region')
        np.save(output/'selected_cloud.npy',cloud)
        save(output/'target-region.json',region)
        calibration=state['metadata']['robot_calibration']
        calibration.update(graspgen_finger_attach_z_m=.0584,physical_finger_attach_z_m=.0524)
        try:
            from vlm_tamp.rekep_libero.dense_tool_v3 import densify_tool_calibration
            calibration,density=densify_tool_calibration(calibration,
                state['metadata']['robot_collision_volumes_world'],state['metadata']['eef_pose_world_xyzw'])
            save(output/'tool-density.json',density)
        except ImportError:
            if not args.filter_only:
                raise RuntimeError('Dense physical tool validation must be available before execution')
        save(output/'calibration.json',calibration)
        coarse,fine,fields=build_collision_fields(list(bundle['cameras'].values()),cloud,
                state['metadata']['robot_collision_volumes_world'],
                robot_visual_volumes=state['metadata'].get('robot_visual_volumes_world',[]),
                robot_material_artifact=state['metadata'].get('robot_visual_meshes_artifact'))
        coarse.save(output/'coarse-sdf.npz');fine.save(output/'fine-sdf.npz')
        save(output/'collision-fields.json',fields)
        from vlm_tamp.rekep_libero.contact_region_v3 import RGBTargetContactField
        coarse=RGBTargetContactField(coarse,bundle['cameras'],target_masks)
        fine=RGBTargetContactField(fine,bundle['cameras'],target_masks)
        field=LocalObservedField(coarse,fine)
        current_points=np.asarray(state['metadata']['robot_collision_points_world'])
        current_check=_collision(field,current_points,np.zeros(len(current_points),bool),None)
        save(output/'current-pose-clearance.json',current_check)
        if not current_check['accepted']:
            raise RuntimeError('The actual current robot pose still fails environment clearance')
        with np.load(source/'stage1-grasps/grasp_candidates.npz',allow_pickle=False) as data:
            generated={'poses_world_base':data['poses_world_base'],'scores':data['scores'],
                       'retained_indices':data['retained_indices'][:args.candidate_limit]}
        if args.candidate_indices is not None:
            retained=set(map(int,generated['retained_indices']))
            if not set(args.candidate_indices)<=retained:
                raise ValueError('Development indices must belong to the originally retained detector candidates')
            generated['retained_indices']=np.asarray(args.candidate_indices,dtype=int)
            summary['development_candidate_subset']=list(args.candidate_indices)
        selection=filter_candidates(generated,bundle['points_world'][22],cloud,calibration,sim,coarse,fine,
                                    initial_joints=state['metadata']['joint_positions'])
        save(output/'grasp-selection.json',selection)
        summary.update(feasible_solutions=len(selection['feasible_candidates']),
                       rejected_pose_variants=len(selection['rejected_candidates']),
                       rejection_counts=dict(Counter(x['reason'] for x in selection['rejected_candidates'])))
        print(json.dumps({'feasible':summary['feasible_solutions'],'rejections':summary['rejection_counts']}),flush=True)
        if args.filter_only or not selection['accepted']:
            summary['status']='filter_only' if args.filter_only else 'no_feasible_grasp'
            return
        # The contact diagnostic is deliberately separate from an end-to-end
        # ReKep benchmark: all interpolated poses receive full-arm validation.
        selected=None
        unique=set()
        for candidate in selection['feasible_candidates']:
            key=(candidate['index'],candidate['variant'])
            if key in unique:
                continue
            unique.add(key)
            target=np.asarray(candidate['tcp_pose_world_xyzw'])
            standoff=target.copy();standoff[:3]-=.07*pose_matrix(target)[:3,2]
            current=np.asarray(state['metadata']['eef_pose_world_xyzw'])
            path=interpolate_path(np.array([current,standoff,target]))
            checked=validate_pose_path(path,calibration,sim,coarse,fine_sdf=fine,
                        initial_joints=state['metadata']['joint_positions'],target_points=cloud,allow_target_contact=True)
            save(output/f'approach-{candidate["index"]}-{candidate["variant"]}.json',dict(checked,poses_world_xyzw=path))
            print(json.dumps({'approach_index':candidate['index'],'variant':candidate['variant'],
                              'accepted':checked['accepted'],'failure':checked.get('failure'),
                              'sample_index':checked.get('sample_index')}),flush=True)
            if checked['accepted']:
                selected=candidate
                break
            if len(unique)>=args.approach_limit:
                break
        if selected is None:
            summary.update(status='approach_paths_rejected',failure=checked['failure'],checked_grasps=len(unique))
            return
        save(output/'chosen-grasp.json',selected)
        summary.update(selected_index=selected['index'],selected_variant=selected['variant'])
        command_path=(np.asarray(checked['fk_poses_world_xyzw']) if args.follow_ik_posture else path)
        save(output/'approach-command-poses.json',{
             'poses_world_xyzw':command_path,
             'mode':'checked IK forward-kinematic poses' if args.follow_ik_posture else 'nominal interpolated poses',
             'nominal_detector_target':target})
        target=command_path[-1]
        from vlm_tamp.rekep_libero.program import parse_program
        grasp_program=parse_program((source/'program/output_raw.txt').read_text(),len(bundle['points_world']),
                                    'pick_place',attachment_cost=lambda i:1.)
        residuals={constraint.name:constraint(target[:3],bundle['points_world'])
                   for constraint in grasp_program.stages[0].subgoal_constraints}
        save(output/'cached-grasp-function-residuals.json',{
             'residuals':residuals,'all_satisfied':all(value<=0 for value in residuals.values()),
             'grasp_goal_mode':'fixed detector pose development diagnostic',
             'stage1_distance_function_enforced':False,
             'reason':'GraspGen selects an object-region grasp; cached landmark remains a semantic object anchor'})
        execution_checks=[]
        for waypoint_index,waypoint in enumerate(command_path[1:],start=1):
            posture=({'nullspace_joint_target':checked['solution_joints'][waypoint_index]}
                     if args.follow_ik_posture else {})
            state=sim.action(waypoint,gripper=0,max_steps=args.waypoint_steps,
                             servo_scale=args.servo_scale,nullspace_kp=args.nullspace_kp,**posture,**precision)
            actual_check=execution_clearance(state,calibration,field,cloud)
            execution_checks.append(dict(actual_check,waypoint_index=waypoint_index))
            save(output/'approach-execution-clearance.json',execution_checks)
            if not actual_check['accepted']:
                raise RuntimeError('Actual robot posture failed the checked approach clearance')
            if not state['reached_pose']:
                raise RuntimeError('Physical controller did not reach a checked approach waypoint')
        state=sim.action(target,gripper=1,max_steps=args.closure_steps,servo_scale=args.servo_scale,
                         nullspace_kp=args.nullspace_kp,output_dir=output/'closed',**posture,**precision)
        summary['physical_grasp_attempts']=int(state['executed_steps']>0)
        summary['closed_gripper_qpos']=state['metadata']['gripper_qpos']
        summary.update(status='closed_gripper',selected_index=selected['index'],selected_variant=selected['variant'])
        from vlm_tamp.rekep_libero.object_motion_v3 import observed_object_cloud,verify_observed_transport
        from vlm_tamp.rekep_libero.geometry import transform_points
        closed_tcp=np.asarray(state['metadata']['eef_pose_world_xyzw'])
        closed_cameras,_=read_observation_v3(output/'closed',state['metadata'])
        lift_coarse,lift_fine,lift_meta=build_collision_fields(list(closed_cameras.values()),cloud,
            state['metadata']['robot_collision_volumes_world'],
            robot_visual_volumes=state['metadata'].get('robot_visual_volumes_world',[]),
            robot_material_artifact=state['metadata'].get('robot_visual_meshes_artifact'))
        save(output/'lift-fields.json',lift_meta)
        lift_goal=closed_tcp.copy();lift_goal[2]+=.05
        lift_path=interpolate_path(np.array([closed_tcp,lift_goal]))
        closed_cal=state['metadata']['robot_calibration']
        closed_cal,_=densify_tool_calibration(closed_cal,state['metadata']['robot_collision_volumes_world'],closed_tcp)
        lift_check=validate_pose_path(lift_path,closed_cal,sim,lift_coarse,fine_sdf=lift_fine,
            initial_joints=state['metadata']['joint_positions'],target_points=cloud,allow_target_contact=True)
        save(output/'lift-path.json',dict(lift_check,poses_world_xyzw=lift_path))
        if not lift_check['accepted']:
            summary.update(status='lift_path_rejected',failure=lift_check['failure'])
            return
        lift_commands=(np.asarray(lift_check['fk_poses_world_xyzw']) if args.follow_ik_posture else lift_path)
        for waypoint_index,waypoint in enumerate(lift_commands[1:],start=1):
            posture=({'nullspace_joint_target':lift_check['solution_joints'][waypoint_index]}
                     if args.follow_ik_posture else {})
            state=sim.action(waypoint,gripper=0,max_steps=args.waypoint_steps,
                             servo_scale=args.servo_scale,nullspace_kp=args.nullspace_kp,**posture,**precision)
            if not state['reached_pose']:
                raise RuntimeError('Physical controller did not reach a checked lift waypoint')
        state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=1,max_steps=20,
                         output_dir=output/'lifted-held')
        cameras,_=read_observation_v3(output/'lifted-held',state['metadata'])
        motion=pose_matrix(state['metadata']['eef_pose_world_xyzw'])@np.linalg.inv(pose_matrix(closed_tcp))
        observed=observed_object_cloud(cameras,cloud,motion,state['metadata']['robot_collision_volumes_world'])
        np.save(output/'observed-lifted-cloud.npy',observed)
        retention=verify_observed_transport(cloud,observed,motion,state['metadata']['gripper_qpos'])
        save(output/'lift-retention.json',retention)
        summary.update(grasp_lift_success=retention['verified'],status='grasp_lift_success' if retention['verified'] else 'grasp_not_retained')
        if not retention['verified']:
            return
        # Use the cached ReKep numerical placement functions and native motion
        # optimizers only after observed physical retention is established.
        from vlm_tamp.rekep_libero.program import parse_program
        from vlm_tamp.rekep_libero.planning import PlanningAdapter
        program=parse_program((source/'program/output_raw.txt').read_text(),len(bundle['points_world']),
                              'pick_place',attachment_cost=lambda i:0. if i==22 else 1.)
        points=bundle['points_world'].copy()
        points[22]=transform_points(points[22][None],np.asarray(retention['transform_world']))[0]
        carried=transform_points(cloud,np.asarray(retention['transform_world']))
        place_coarse,place_fine,place_meta=build_collision_fields(list(cameras.values()),carried,
            state['metadata']['robot_collision_volumes_world'],
            robot_visual_volumes=state['metadata'].get('robot_visual_volumes_world',[]),
            robot_material_artifact=state['metadata'].get('robot_visual_meshes_artifact'))
        place_coarse.exclude_attached_region(carried);place_fine.exclude_attached_region(carried)
        place_field=LocalObservedField(place_coarse,place_fine)
        place_cal=state['metadata']['robot_calibration']
        place_cal,_=densify_tool_calibration(place_cal,state['metadata']['robot_collision_volumes_world'],
                                            state['metadata']['eef_pose_world_xyzw'])
        place_cal['tool_collision_finger_flags']=place_cal['tool_collision_contact_face_flags']
        moving=np.zeros(len(points),bool);moving[22]=True
        optimizer=PlanningAdapter(sim,place_cal,sampling_maxfun=1000,maxiter=100)
        planned=optimizer.solve_stage(program.stages[1],state['metadata']['eef_pose_world_xyzw'],
                     state['metadata']['joint_positions'],points,moving,place_field,target_points=carried)
        save(output/'rekep-placement-plan.json',planned)
        if not planned['accepted']:
            summary.update(status='placement_plan_rejected',failure=planned['failure'])
            return
        placement=np.asarray(planned['poses_world_xyzw'])
        place_checked=validate_pose_path(placement,place_cal,sim,place_coarse,fine_sdf=place_fine,
            initial_joints=state['metadata']['joint_positions'],target_points=carried,allow_target_contact=True)
        save(output/'placement-path-check.json',place_checked)
        if not place_checked['accepted']:
            summary.update(status='placement_path_rejected',failure=place_checked['failure'])
            return
        place_commands=(np.asarray(place_checked['fk_poses_world_xyzw']) if args.follow_ik_posture else placement)
        for waypoint_index,waypoint in enumerate(place_commands[1:],start=1):
            posture=({'nullspace_joint_target':place_checked['solution_joints'][waypoint_index]}
                     if args.follow_ik_posture else {})
            state=sim.action(waypoint,gripper=0,max_steps=args.waypoint_steps,
                             servo_scale=args.servo_scale,nullspace_kp=args.nullspace_kp,**posture,**precision)
            if not state['reached_pose']:
                raise RuntimeError('Physical controller did not reach a checked placement waypoint')
        state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,
                         output_dir=output/'released')
        summary.update(status='task_success' if state['evaluation']['task_success'] else 'released_without_task_success',
                       official_task_success=bool(state['evaluation']['task_success']))
    except Exception as error:
        summary.update(status='error',failure=str(error),exception_type=type(error).__name__)
        (output/'error-traceback.txt').write_text(traceback.format_exc())
    finally:
        closed=sim.close()
        if isinstance(closed,dict):
            summary['control_steps']=closed.get('control_steps',0)
            summary['official_task_success']=bool(closed.get('evaluation',{}).get('task_success'))
            summary['video']=closed.get('video_path')
        actions_path=output/'initial/actions.json'
        if actions_path.is_file():
            actual_actions=json.loads(actions_path.read_text())
            summary['final_task_predicate']=summary['official_task_success']
            summary['official_task_success']=any(action['evaluation_task_success'] for action in actual_actions)
            summary['physical_grasp_attempts']=int(any(action['control'][-1]>0 for action in actual_actions))
        summary['elapsed_seconds']=time.monotonic()-before
        save(output/'summary.json',summary)
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
