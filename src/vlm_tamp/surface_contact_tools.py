"""Observed surface contacts for free-motion manipulation without a grasp."""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .coupled_scene import Scene,RobotVolumes
from .coupled_storage import write_json,preserve_scene
from .hinged_contact_tools import NativeScene
from .rekep_libero.geometry import pose_matrix,transform_points
from .rekep_libero.scene_state_v7 import observed_points


class ObservedContactPatchScene(Scene):
    def record(self):
        return {**super().record(),'schema_version':'observed-contact-patch-1.0',
            'moving_geometry':'bounded observed contact patch; all other observed points remain obstacles',
            'upright_handle_mount_shape_prior':None,'articulation_estimated':False,
            'object_motion_predicted':False,'execution_guard_policy':'planning only; no execution-time veto',
            'allow_palm_handle_contact':False,'allow_palm_panel_brace':False,
            'allow_finger_handle_contact':True}


def contact_snapshot(cameras,metadata,camera,region_xyxy,chain,*,radius_m=.02):
    if camera not in cameras:raise ValueError('Choose an observed camera')
    if not np.isfinite(radius_m) or not .005<=radius_m<=.04:
        raise ValueError('Contact-patch radius must be 0.005..0.04 metres')
    view=cameras[camera];world=np.asarray(view['world_points']);h,w=world.shape[:2]
    box=np.asarray(region_xyxy,float)
    if box.shape!=(4,) or not np.isfinite(box).all() or not (
            0<=box[0]<box[2]<w and 0<=box[1]<box[3]<h):
        raise ValueError('Select a bounded pixel region in the current image')
    yy,xx=np.indices((h,w));selected=(xx>=box[0])&(xx<=box[2])&(yy>=box[1])&(yy<=box[3])
    candidate=world[selected&np.isfinite(world).all(2)]
    occupied,filter_record=observed_points(cameras,metadata,spacing=.003)
    if len(occupied)==0 or len(candidate)<8:raise ValueError('Insufficient observed surface points')
    candidate=candidate[cKDTree(occupied).query(candidate)[0]<=.0035]
    if len(candidate)<8:raise ValueError('Selected pixels lack non-robot depth support')
    median=np.median(candidate,axis=0)
    center=candidate[np.argmin(np.linalg.norm(candidate-median,axis=1))]
    patch=candidate[np.linalg.norm(candidate-center,axis=1)<=radius_m]
    if len(patch)<8:raise ValueError('Contact patch has too little measured support')
    _,_,vt=np.linalg.svd(patch-patch.mean(0),full_matrices=False)
    normal=vt[-1]
    origin=pose_matrix(view['camera_to_world'])[:3,3]
    if normal@(origin-center)<0:normal=-normal
    in_patch=cKDTree(patch).query(occupied)[0]<=.0035
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
