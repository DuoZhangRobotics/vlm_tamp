"""Collision witnesses from observed endpoints and calibrated robot geometry.

These are explanations of an existing rejection, not extra path validation or
physical contact measurements. They never change collision permissions.
"""
import numpy as np
from scipy.spatial import cKDTree

from .rekep_libero.geometry import pose_matrix,transform_points
from .rekep_libero.scene_state_v7 import voxel_points


CONTACT_ASSOCIATION_M=.0035


def selected_contact_points(scene):
    """The exact association used by native contact-scene admission."""
    return cKDTree(scene.contact_reference).query(scene.moving_reference)[0]<=CONTACT_ASSOCIATION_M


def current_collision_witness(scene,metadata,diagnostic):
    name=diagnostic.get('blocking_robot_part')
    volumes=metadata.get('robot_collision_volumes_world',[])
    volume=next((v for v in volumes if v['robot_geom_name']==name),None)
    if volume is None or volume.get('type')!='convex':
        return {'available':False,'reason':'no_matching_configured_convex_volume'}
    surface=int(diagnostic.get('last_rejected_surface',-1))
    if surface not in (0,1,2):return {'available':False,'reason':'not_an_observed_scene_collision'}
    if surface==0:points=voxel_points(scene.static_points,.0075)
    else:
        selected=selected_contact_points(scene)
        points=voxel_points(scene.moving_reference[selected if surface==2 else ~selected],.003 if surface==2 else .005)
    vertices=np.asarray(volume['vertices_world']);center=vertices.mean(0)
    ids=cKDTree(points).query_ball_point(center,np.linalg.norm(vertices-center,axis=1).max())
    nearby=points[ids];planes=np.asarray(volume['planes_world'],float)
    planes=planes/np.linalg.norm(planes[:,:3],axis=1)[:,None]
    distances=np.max(nearby@planes[:,:3].T+planes[:,3],axis=1) if len(nearby) else np.empty(0)
    inside=distances<=0
    # Finger bodies overlap the pads. On the selected patch only the proximal
    # fragment is blocking in fingertip/palm mode, matching the native split.
    mode=getattr(scene,'skill_contact_surface',None) or 'fingertips'
    if surface==2 and 'gripper' in name:
        if mode=='grasped_fingers' and 'finger' in name:inside[:]=False
        elif 'finger' in name and not name.endswith('_pad_collision'):
            inverse=np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']))
            pads=[transform_points(v['vertices_world'],inverse) for v in volumes
                if v['robot_geom_name'].endswith('_pad_collision')]
            cut=min(p[:,2].min() for p in pads)-.001 if pads else 0.
            inside&=transform_points(nearby,inverse)[:,2]<=cut
        elif name.endswith('_pad_collision') or mode=='palm':inside[:]=False
    witnesses=nearby[inside];depths=-distances[inside]*1000
    contact_distance=cKDTree(scene.contact_reference).query(witnesses)[0]*1000 if len(witnesses) else []
    return {'available':True,'robot_part':name,'blocking_surface':diagnostic.get('blocking_surface'),
        'observed_points_inside_volume':int(inside.sum()),
        'overlap_depth_mm_range':[float(depths.min()),float(depths.max())] if len(depths) else None,
        'distance_to_selected_contact_mm_range':[float(np.min(contact_distance)),float(np.max(contact_distance))] if len(witnesses) else None,
        'sample_points_world_m':witnesses[:3].tolist(),'contact_association_m':CONTACT_ASSOCIATION_M,
        'added_collision_padding_m':0.,'physical_contact_measurement':False,
        'source':'reported surface partition versus calibrated convex robot volume at measured joints',
        'interpretation':'Observed-model overlap, not proof of physical penetration. Transport and depth error can affect this estimate.'}
