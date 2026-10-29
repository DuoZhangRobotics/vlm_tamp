"""Gripper-width evidence from calibrated robot geometry, never native contacts."""
import numpy as np
from .rekep_libero.geometry import pose_matrix,transform_points


def pad_gap_m(metadata):
    if 'eef_pose_world_xyzw' not in metadata or not metadata.get('robot_collision_volumes_world'):return None
    inverse=np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']))
    pads=[]
    for volume in metadata.get('robot_collision_volumes_world',[]):
        if volume['robot_geom_name'].endswith('_pad_collision'):
            points=transform_points(volume['vertices_world'],inverse)
            pads.append([float(points[:,0].min()),float(points[:,0].max())])
    if len(pads)!=2:return None
    pads.sort()
    return float(pads[1][0]-pads[0][1])


def grasp_observation(metadata,spec=None,*,before=None,manipulation=False):
    gap=pad_gap_m(metadata);previous=pad_gap_m(before) if before is not None else None
    if gap is None:return {'available':False,'reason':'two calibrated pads unavailable'}
    collapsed=bool(manipulation and previous is not None and previous>.006
        and gap<.003 and gap<.25*previous)
    near_closed=bool(spec and spec.get('observed_handle_radius_m',0)>=.005 and gap<.003)
    return {'available':True,'source':'current robot proprioception and configured finger-pad geometry',
        'pad_gap_mm':1000*gap,'previous_pad_gap_mm':1000*previous if previous is not None else None,
        'selected_bar_diameter_mm':2000*spec['observed_handle_radius_m']
            if spec and 'observed_handle_radius_m' in spec else None,
        'status':'gripper_gap_collapsed_during_manipulation' if collapsed else
            'near_closed_without_bar_width_support' if near_closed else 'grasp_not_verified',
        'physical_contacts_observed':False,
        'interpretation':('The gap collapsed toward fully closed during the manipulation. The object may no longer be between the pads; inspect contact before assuming attachment.'
            if collapsed else 'The pads nearly meet despite the selected rigid bar model. This does not support a pinch grasp of that bar; verify contact or part identity.'
            if near_closed else 'A close command is not proof of a grasp. Compare measured pad gap with the intended contact and observed object motion.')}
