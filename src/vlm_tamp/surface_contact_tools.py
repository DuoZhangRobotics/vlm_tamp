"""Observed surface contacts for free-motion manipulation without a grasp."""
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from .coupled_scene import Scene,RobotVolumes
from .coupled_storage import write_json,preserve_scene
from .hinged_contact_tools import NativeScene
from .rekep_libero.geometry import pose_matrix,matrix_pose,transform_points
from .rekep_libero.scene_state_v7 import observed_points
from .part_perception import region_mask,projected_selection,usable_pixels


class ObservedContactPatchScene(Scene):
    def record(self):
        return {**super().record(),'schema_version':'observed-contact-patch-1.0',
            'moving_geometry':'bounded observed contact patch; all other observed points remain obstacles',
            'upright_handle_mount_shape_prior':None,'articulation_estimated':False,
            'object_motion_predicted':False,'execution_guard_policy':'planning only; no execution-time veto',
            'allow_palm_handle_contact':False,'allow_palm_panel_brace':False,
            'allow_finger_handle_contact':True}


def selected_region_points(view,region,occupied,selected_mask=None):
    mask,box=region_mask(view,region,selected_mask)
    world=np.asarray(view['world_points']);candidate=world[mask&np.isfinite(world).all(2)]
    if len(candidate) and len(occupied):
        candidate=candidate[cKDTree(occupied).query(candidate,distance_upper_bound=np.nextafter(.0035,np.inf))[0]<=.0035]
    else:candidate=np.empty((0,3))
    return candidate,box


def selected_pixel_point(view, pixel_xy, region=None, selected_mask=None):
    """Lift the caller's exact RGB-D sample; never snap to a different surface."""
    pixel=np.asarray(pixel_xy,float)
    if pixel.shape!=(2,) or not np.isfinite(pixel).all() or not np.array_equal(pixel,np.rint(pixel)):
        raise ValueError('contact_pixel_xy must contain two integer original-image pixel coordinates')
    x,y=pixel.astype(int);world=np.asarray(view['world_points']);h,w=world.shape[:2]
    if not (0<=x<w and 0<=y<h):raise ValueError('Contact pixel is outside the selected camera image')
    if not usable_pixels(view)[y,x]:
        raise ValueError('The selected contact pixel has no measured non-robot depth; choose another pixel explicitly')
    if region is not None or selected_mask is not None:
        mask,_=region_mask(view,region,selected_mask)
        if not mask[y,x]:raise ValueError('Contact pixel is outside the selected target region/mask')
    return world[y,x].copy()


def contact_snapshot(cameras,metadata,camera,region_xyxy,chain,*,radius_m=.02,observed=None,
                     selected_mask=None,selected_points=None,contact_pixel_xy=None):
    if camera not in cameras:raise ValueError('Choose an observed camera')
    if not np.isfinite(radius_m) or radius_m<=0:
        raise ValueError('Contact-patch radius must be finite and positive')
    view=cameras[camera]
    occupied,filter_record=(observed_points(cameras,metadata,spacing=.003) if observed is None else observed)
    if selected_points is None:candidate,box=selected_region_points(view,region_xyxy,occupied,selected_mask)
    else:candidate,box=np.asarray(selected_points),list(region_xyxy)
    center=selected_pixel_point(view,contact_pixel_xy,region_xyxy,selected_mask)
    patch=candidate[np.linalg.norm(candidate-center,axis=1)<=radius_m]
    if len(patch)<3:raise ValueError('At least three depth samples are needed to report a surface normal')
    _,_,vt=np.linalg.svd(patch-patch.mean(0),full_matrices=False)
    normal=vt[-1]
    origin=pose_matrix(view['camera_to_world'])[:3,3]
    if normal@(origin-center)<0:normal=-normal
    in_patch=cKDTree(patch).query(occupied,distance_upper_bound=np.nextafter(.0035,np.inf))[0]<=.0035
    if selected_mask is not None:in_patch &= projected_selection(occupied,view,box,selected_mask,padding_pixels=0)
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
    record={'camera':camera,'region_xyxy':box,'contact_pixel_xy':list(contact_pixel_xy),
        'point_source':'exact caller-selected RGB-D pixel','point_world_m':center.tolist(),
        'normal_toward_camera_world':normal.tolist(),'radius_m':radius_m,
        'selected_observed_points':len(contact),'other_observed_points':len(static),
        'surface_fit_residual_m':float(np.median(np.abs((patch-center)@normal))),
        'configured_pad_centers_in_tcp':pads,'grasp_or_attachment_required':False,
        'contact_permission':'fingers may contact any environment surface; the patch selects the requested target',
        'palm_arm_and_other_surface_checks_retained':True,
        'object_motion_predicted':False,'simulator_object_geometry_used':False}
    return scene,record


def contact_pose_candidates(metadata,point_world_m,quaternion_xyzw=None,contact_name=None):
    """Describe exactly one caller-selected robot contact, without pose menus."""
    from .contact_skills import contact_landmarks
    point=np.asarray(point_world_m,float)
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Use a finite world contact point')
    landmarks=contact_landmarks(metadata,'palm' if contact_name and 'hand_collision' in contact_name else 'fingertips')
    contact=next((c for c in landmarks if c['name']==contact_name),None)
    if contact is None:raise ValueError('Choose contact_name explicitly from contact_landmarks_in_tcp')
    current=pose_matrix(metadata['eef_pose_world_xyzw'])
    rotation=current[:3,:3] if quaternion_xyzw is None else pose_matrix([0.,0.,0.,*quaternion_xyzw])[:3,:3]
    pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=point-rotation@contact['point']
    return [{'contact_name':contact['name'],'contact_local_m':contact['point'].tolist(),
        'robot_collision_geom':contact['robot_collision_geom'],
        'physical_collision_masks':contact['physical_collision_masks'],
        'initiates_simulator_contacts':contact['initiates_simulator_contacts'],
        'point_world_m':point.tolist(),'tcp_pose_world_xyzw':matrix_pose(pose).tolist(),
        'orientation_constraint':'free' if quaternion_xyzw is None else 'caller_quaternion',
        'gripper_aperture_qpos':metadata.get('gripper_qpos'),
        'conversion':'TCP translation = requested contact point - rotation × calibrated contact offset'}]


def contact_solutions(ik,metadata,point,contact_name,quaternion_xyzw=None):
    candidate=contact_pose_candidates(metadata,point,quaternion_xyzw,contact_name)[0]
    start=np.asarray(metadata['joint_positions'])
    if quaternion_xyzw is None:
        solutions=ik.solve_contact_many(point,candidate['contact_local_m'],start,
            pose_matrix(metadata['eef_pose_world_xyzw'])[:3,:3])
    else:
        solutions=ik.solve_many(candidate['tcp_pose_world_xyzw'],start)
    return candidate,solutions


def plan_contact_pose(ik,scene,record,metadata,point,folder,*,quaternion_xyzw=None,contact_name=None,budget_seconds=30.):
    """cuRobo endpoint IK followed by native planning; no extra collision pass."""
    folder=Path(folder);folder.mkdir();start=np.asarray(metadata['joint_positions'])
    began=time.perf_counter();deadline=began+float(budget_seconds);attempts=[]
    candidate,solutions=contact_solutions(ik,metadata,point,contact_name,quaternion_xyzw)
    timing={'ik_seconds':time.perf_counter()-began,'approach_planning_seconds':0.}
    with NativeScene(scene,record,metadata,start,.05,articulated=False) as context:
        for solution in solutions:
            remaining=deadline-time.perf_counter()
            if remaining<=0:break
            before=time.perf_counter()
            path,result=context.plan(start,solution['joints'],constrained=False,seconds=min(8.,remaining))
            timing['approach_planning_seconds']+=time.perf_counter()-before
            attempts.append({'joints':solution['joints'],'path_result':result})
            if result['exact_solution']:
                candidate['tcp_pose_world_xyzw']=ik.chain.forward(path[-1]).tolist()
                np.save(folder/'path.npy',path);write_json(folder/'attempts.json',attempts)
                return path,{**result,'contact_pose':candidate,
                    'requested_point_world_m':np.asarray(point).tolist(),
                    'grasp_required':False,'articulation_constraint_used':False,
                    'search_timing_seconds':timing,'physical_contact_verified':False}
    diagnostic={'status':'contact_pose_search_incomplete' if time.perf_counter()>=deadline else
        'contact_ik_failed' if not solutions else 'contact_path_not_found',
        'search_complete':len(attempts)==len(solutions),'physical_infeasibility_proved':False,
        'attempts':attempts,'timing_seconds':timing,'contact_pose':candidate}
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
