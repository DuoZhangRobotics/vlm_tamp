"""Observed surface contacts for free-motion manipulation without a grasp."""
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .coupled_scene import Scene,RobotVolumes
from .coupled_storage import write_json,preserve_scene
from .hinged_contact_tools import NativeScene
from .rekep_libero.geometry import pose_matrix,matrix_pose,transform_points
from .rekep_libero.scene_state_v7 import observed_points


class ObservedContactPatchScene(Scene):
    def record(self):
        return {**super().record(),'schema_version':'observed-contact-patch-1.0',
            'moving_geometry':'bounded observed contact patch; all other observed points remain obstacles',
            'upright_handle_mount_shape_prior':None,'articulation_estimated':False,
            'object_motion_predicted':False,'execution_guard_policy':'planning only; no execution-time veto',
            'allow_palm_handle_contact':False,'allow_palm_panel_brace':False,
            'allow_finger_handle_contact':True}


def contact_snapshot(cameras,metadata,camera,region_xyxy,chain,*,radius_m=.02,observed=None):
    if camera not in cameras:raise ValueError('Choose an observed camera')
    if not np.isfinite(radius_m) or not .005<=radius_m<=.04:
        raise ValueError('Contact-patch radius must be 0.005..0.04 metres')
    view=cameras[camera];world=np.asarray(view['world_points']);h,w=world.shape[:2]
    box=np.asarray(region_xyxy,float)
    if box.shape!=(4,) or not np.isfinite(box).all() or not (
            0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
        raise ValueError(f'Selected region {box.tolist()} is outside {camera} ({w}×{h}); use original camera pixels')
    yy,xx=np.indices((h,w));selected=(xx>=box[0])&(xx<=box[2])&(yy>=box[1])&(yy<=box[3])
    candidate=world[selected&np.isfinite(world).all(2)]
    occupied,filter_record=(observed_points(cameras,metadata,spacing=.003) if observed is None else observed)
    if len(occupied)==0 or len(candidate)<8:raise ValueError('Insufficient observed surface points')
    finite_candidates=len(candidate)
    candidate=candidate[cKDTree(occupied).query(candidate,distance_upper_bound=np.nextafter(.0035,np.inf))[0]<=.0035]
    if len(candidate)<8:raise ValueError(f'Selected region has {len(candidate)} eligible non-robot surface points after filtering ({finite_candidates} finite samples; need 8). Inspect the selected object region.')
    median=np.median(candidate,axis=0)
    center=candidate[np.argmin(np.linalg.norm(candidate-median,axis=1))]
    patch=candidate[np.linalg.norm(candidate-center,axis=1)<=radius_m]
    if len(patch)<8:raise ValueError('Contact patch has too little measured support')
    _,_,vt=np.linalg.svd(patch-patch.mean(0),full_matrices=False)
    normal=vt[-1]
    origin=pose_matrix(view['camera_to_world'])[:3,3]
    if normal@(origin-center)<0:normal=-normal
    in_patch=cKDTree(patch).query(occupied,distance_upper_bound=np.nextafter(.0035,np.inf))[0]<=.0035
    contact=occupied[in_patch];static=occupied[~in_patch]
    if len(contact)<3:raise ValueError('Contact patch disappeared during scene assembly')
    # These reference-frame fields are required by Scene/NativeScene. No
    # articulation is estimated or used for this free-motion snapshot.
    spec={'joint_origin_world_m':center.tolist(),'joint_axis_world':normal.tolist(),
        'front_plane':{'point_world_m':center.tolist()},'panel_normal_world':normal.tolist(),
        'scene_kind':'observed_finger_contact_patch','articulation_estimated':False}
    robot=RobotVolumes(chain,metadata['robot_collision_volumes_world'],metadata['joint_positions'])
    scene=ObservedContactPatchScene.from_points(static,contact,contact,spec,robot=robot,
        reference_volumes=metadata['robot_collision_volumes_world'])
    scene.self_filter=filter_record
    scene.contact_policy_kind='observed_finger_patch'
    inverse=np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']))
    pads={v['robot_geom_name']:transform_points(v['vertices_world'],inverse).mean(0).tolist()
          for v in metadata['robot_collision_volumes_world'] if v['robot_geom_name'].endswith('_pad_collision')}
    record={'camera':camera,'region_xyxy':box.tolist(),'point_world_m':center.tolist(),
        'normal_toward_camera_world':normal.tolist(),'radius_m':radius_m,
        'selected_observed_points':len(contact),'other_observed_points':len(static),
        'surface_fit_residual_m':float(np.median(np.abs((patch-center)@normal))),
        'configured_pad_centers_in_tcp':pads,'grasp_or_attachment_required':False,
        'contact_permission':'configured fingers versus this bounded observed patch only',
        'palm_arm_and_other_surface_checks_retained':True,
        'object_motion_predicted':False,'simulator_object_geometry_used':False}
    return scene,record


def contact_pose_candidates(metadata,point_world_m,quaternion_xyzw=None,contact_name=None):
    """Convert an observed contact point to TCP poses using current hand geometry."""
    from .contact_skills import contact_landmarks
    point=np.asarray(point_world_m,float)
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Use a finite world contact point')
    current=pose_matrix(metadata['eef_pose_world_xyzw'])
    rotations=[current[:3,:3]] if quaternion_xyzw is None else [pose_matrix([0.,0.,0.,*quaternion_xyzw])[:3,:3]]
    if quaternion_xyzw is None:
        for axis in np.eye(3):
            for angle in [-np.pi/4,np.pi/4]:rotations.append(Rotation.from_rotvec(axis*angle).as_matrix()@current[:3,:3])
    landmarks=contact_landmarks(metadata,'fingertips')
    if contact_name is not None:landmarks=[c for c in landmarks if c['name']==contact_name]
    if not landmarks:raise ValueError('Choose a configured fingertip/pad from contact_landmarks_in_tcp')
    candidates=[]
    for contact in landmarks:
        for rotation in rotations:
            pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=point-rotation@contact['point']
            candidates.append({'contact_name':contact['name'],'contact_local_m':contact['point'].tolist(),
                'point_world_m':point.tolist(),'tcp_pose_world_xyzw':matrix_pose(pose).tolist(),
                'gripper_aperture_qpos':metadata.get('gripper_qpos'),
                'conversion':'TCP translation = requested contact point - rotation × calibrated contact offset'})
    return sorted(candidates,key=lambda c:np.linalg.norm(np.asarray(c['tcp_pose_world_xyzw'][:3])-current[:3,3])
        +.04*Rotation.from_matrix(current[:3,:3].T@pose_matrix(c['tcp_pose_world_xyzw'])[:3,:3]).magnitude())


def plan_contact_pose(ik,scene,record,metadata,point,folder,*,quaternion_xyzw=None,contact_name=None,budget_seconds=30.):
    """Plan to a finger contact point; do not ask the actor to solve TCP offsets."""
    folder=Path(folder);folder.mkdir();start=np.asarray(metadata['joint_positions'])
    candidates=contact_pose_candidates(metadata,point,quaternion_xyzw,contact_name)
    deadline=time.perf_counter()+float(budget_seconds);attempts=[];timing={}
    with NativeScene(scene,record,metadata,start,.05,articulated=False) as context:
        for candidate in candidates:
            if time.perf_counter()>=deadline:break
            pose=candidate['tcp_pose_world_xyzw'];attempt={'contact_pose':candidate};attempts.append(attempt)
            before=time.perf_counter();hand=context.validate_hand_pose(pose)
            timing['hand_scene_filter_seconds']=timing.get('hand_scene_filter_seconds',0.)+time.perf_counter()-before
            attempt['hand_check']=hand
            if not hand['accepted']:
                attempt.update(reason='contact_hand_collision',diagnostic=hand);continue
            before=time.perf_counter();solutions=ik.solve_many(pose,start,maximum=8)
            timing['ik_seconds']=timing.get('ik_seconds',0.)+time.perf_counter()-before
            solutions=sorted(solutions,key=lambda row:np.linalg.norm(np.asarray(row['joints'])-start))
            attempt['robot_valid_ik_solutions']=len(solutions);attempt['endpoint_checks']=[]
            attempt['reason']='no_scene_valid_ik' if solutions else 'no_contact_ik'
            for solution in solutions:
                check=context.validate(solution['joints']);attempt['endpoint_checks'].append(check)
                if not check['accepted']:continue
                remaining=deadline-time.perf_counter()
                if remaining<=0:break
                before=time.perf_counter();path,result=context.plan(start,solution['joints'],constrained=False,seconds=min(8.,remaining))
                timing['approach_planning_seconds']=timing.get('approach_planning_seconds',0.)+time.perf_counter()-before
                attempt['path_result']=result
                if result['exact_solution']:
                    attempt['reason']='solved'
                    np.save(folder/'path.npy',path);write_json(folder/'attempts.json',attempts)
                    return path,{**result,'contact_pose':candidate,'requested_pose_world_xyzw':pose,
                        'grasp_required':False,'articulation_constraint_used':False,
                        'search_timing_seconds':timing,'physical_contact_verified':False}
                # One closest scene-valid arm solution per calibrated landmark.
                attempt['reason']='approach_search_incomplete'
                break
    complete=len(attempts)==len(candidates) and time.perf_counter()<deadline
    diagnostic={'status':'contact_pose_candidates_exhausted' if complete else 'contact_pose_search_incomplete',
        'search_complete':complete,'physical_infeasibility_proved':False,'attempts':attempts,'timing_seconds':timing}
    write_json(folder/'attempts.json',attempts)
    error=ValueError('Contact-pose planning: '+diagnostic['status']);error.planning_diagnostic=diagnostic
    raise error


def plan_pose(ik,scene,record,metadata,pose,folder):
    folder=Path(folder);folder.mkdir()
    preserve_scene(scene,folder)
    start=np.asarray(metadata['joint_positions']);attempts=[]
    write_json(folder/'planning-input.json',{'reference_joints':start,
        'requested_pose_world_xyzw':pose,'planning_only':True,
        'metadata':{key:metadata[key] for key in ['joint_positions','eef_pose_world_xyzw','robot_collision_volumes_world']}})
    solutions=ik.solve_many(pose,start)
    with NativeScene(scene,record,metadata,start,.05,articulated=False) as context:
        for index,solution in enumerate(solutions):
            path,result=context.plan(start,solution['joints'],constrained=False,seconds=8.)
            attempts.append({'ik_index':index,'goal_joints':solution['joints'],'result':result})
            if result['exact_solution']:
                np.save(folder/'path.npy',path)
                result={**result,'requested_pose_world_xyzw':np.asarray(pose).tolist(),
                    'ik_index':index,'grasp_required':False,'articulation_constraint_used':False}
                write_json(folder/'result.json',result);write_json(folder/'attempts.json',attempts)
                write_json(folder/'native-scene.json',context.record)
                return path,result
        write_json(folder/'native-scene.json',context.record)
    write_json(folder/'attempts.json',attempts)
    from .planning_feedback import summarize_pose_failure
    diagnostic=summarize_pose_failure(attempts,context.record)
    error=ValueError(f"Free-pose planning: {diagnostic['status']}; {len(solutions)} robot-valid IK solutions. "
        f"Endpoint rejections: {diagnostic['endpoint_rejection_counts']}; search outcomes: {diagnostic['search_stage_counts']}.")
    error.planning_diagnostic=diagnostic
    raise error
