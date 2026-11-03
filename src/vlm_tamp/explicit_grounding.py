"""Literal caller selections lifted through calibrated RGB-D.

No vertical-door, cylindrical-handle, mount or panel-completion prior. A hinge
annotation is a caller hypothesis, not recognition of a hidden physical joint.
"""
import copy
import numpy as np
from scipy.spatial import cKDTree

from .coupled_scene import Scene, RobotVolumes
from .part_perception import region_mask, usable_pixels, projected_selection
from .rekep_libero.geometry import pose_matrix
from .rekep_libero.scene_state_v7 import observed_points
from .surface_contact_tools import selected_pixel_point
from .joint_rotation import canonical_axis, AXIS_CONVENTION, VIEW_CONVENTION


def ground_pixels(cameras, metadata, choices, *, fixed_joint=None, panel_mask=None, contact_mask=None):
    view=cameras[choices['camera']]
    selected,panel_box=region_mask(view,choices['moving_panel_box_xyxy'],panel_mask)
    world=np.asarray(view['world_points']);valid=usable_pixels(view)
    panel=world[selected & valid]
    if len(panel)<3:raise ValueError('Selected moving part needs at least three measured depth points')
    box=choices.get('contact_box_xyxy')
    handle_selected=box is not None or contact_mask is not None
    contact_selection=(region_mask(view,box,contact_mask)[0] if handle_selected else selected)
    contact=world[contact_selection & valid]
    anchor=selected_pixel_point(view,choices['contact_pixel_xy'],
        box if handle_selected else panel_box,contact_mask if handle_selected else panel_mask)
    center=panel.mean(0);_,_,basis=np.linalg.svd(panel-center,full_matrices=False)
    normal=basis[-1]
    if normal@(pose_matrix(view['camera_to_world'])[:3,3]-center)<0:normal=-normal
    if fixed_joint is None:
        hinge=np.asarray([selected_pixel_point(view,p) for p in choices['hinge_pixels_xy']])
        axis=hinge[1]-hinge[0];length=float(np.linalg.norm(axis))
        if length==0:raise ValueError('The two hinge pixels resolve to the same 3D point')
        axis=canonical_axis(axis);origin=hinge.mean(0)
        source={'source':'caller hinge pixels lifted through measured depth; canonical world-axis sign',
            'task_direction_status':'unknown','coordinate_convention_only':True,
            'task_direction_verified':False,'axis_sign_convention':AXIS_CONVENTION,
            'legacy_image_hint_recorded_only':choices.get('initial_image_direction_xy')}
        source.update(selected_axis=axis.tolist(),positive_rotation_meaning=VIEW_CONVENTION)
    else:
        origin=np.asarray(fixed_joint['joint_origin_world_m'])
        axis=canonical_axis(fixed_joint['joint_axis_world'])
        hinge=None;source=copy.deepcopy(fixed_joint.get('direction_source',{}))
        source.update(selected_axis=axis.tolist(),axis_sign_convention=AXIS_CONVENTION,
            positive_rotation_meaning=VIEW_CONVENTION)
    spec={
        'schema_version':'articulated-explicit-pixels-1.0','task_type':'articulated',
        'joint_family':'revolute','camera':choices['camera'],
        'joint_origin_world_m':origin.tolist(),'joint_axis_world':axis.tolist(),
        'source_anchor_world_m':anchor.tolist(),'front_plane':{'point_world_m':center.tolist()},
        'panel_normal_world':normal.tolist(),'direction_source':source,
        'generic_part_joint_model':not handle_selected,'explicit_pixel_geometry':True,
        'geometry_source':'caller pixels and masks with measured RGB-D; no shape prior',
        'hinge_evidence':{'status':'caller_hinge_hypothesis','semantic_identity_verified':False,
            'calibrated_accuracy_guarantee':False},
    }
    if fixed_joint is not None:
        for key in ('joint_estimate_id','joint_estimation_evidence','hinge_evidence'):
            if key in fixed_joint:spec[key]=copy.deepcopy(fixed_joint[key])
    evidence={'contact_pixel_xy':list(choices['contact_pixel_xy']),
        'contact_point_world_m':anchor.tolist(),'hinge_depth_samples_world_m':None if hinge is None else hinge.tolist(),
        'axis_prior':None,'handle_shape_prior':None,'shape_completion':False,
        'moving_observed_points':len(panel),'contact_observed_points':len(contact),
        'surface_plane_rms_m':float(np.sqrt(np.mean(((panel-center)@normal)**2))),
        'surface_normal_role':'reported geometry only; not a push/pull direction',
        'motion_direction_check':source,'fixed_joint_preserved':fixed_joint is not None,
        'simulator_object_geometry_used':False}
    return {'spec':spec,'cloud':contact,'moving_points':panel,'mask':contact_selection,
        'moving_mask':selected,
        'anchors':[anchor],'choices':copy.deepcopy(choices),'evidence':evidence}


def selected_scene(candidate, cameras, metadata, chain, config):
    """Tag only measured selected samples, using the existing observation grid."""
    occupied,filter_record=observed_points(cameras,metadata,spacing=.003)
    contact=np.asarray(candidate['cloud']);moving=np.asarray(candidate['moving_points'])
    on_moving=cKDTree(moving).query(occupied,distance_upper_bound=np.nextafter(.0035,np.inf))[0]<=.0035
    on_contact=cKDTree(contact).query(occupied,distance_upper_bound=np.nextafter(.0035,np.inf))[0]<=.0035
    from .part_perception import box_of
    view=cameras[candidate['choices']['camera']]
    on_moving &= projected_selection(occupied,view,box_of(candidate['moving_mask']),candidate['moving_mask'],padding_pixels=0)
    on_contact &= projected_selection(occupied,view,box_of(candidate['mask']),candidate['mask'],padding_pixels=0)
    on_moving |= on_contact
    robot=RobotVolumes(chain,metadata['robot_collision_volumes_world'],metadata['joint_positions'])
    scene=Scene.from_points(occupied[~on_moving],occupied[on_moving],occupied[on_contact],candidate['spec'],
        config=copy.deepcopy(config),robot=robot,reference_volumes=metadata['robot_collision_volumes_world'])
    scene.self_filter=filter_record
    return scene
