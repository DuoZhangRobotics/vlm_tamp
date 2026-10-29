"""Initialize a joint-linked observed part without assuming a grasp handle."""
import copy
import numpy as np
from scipy.spatial import cKDTree

from .contact_skills import fit_object_motion,unit
from .coupled_scene import Scene,RobotVolumes
from .part_perception import usable_pixels
from .rekep_libero.geometry import transform_points
from .rekep_libero.scene_state_v7 import observed_points


def retain_observed_articulation(service):
    """Keep timestamped observations for explicit recovery, never active planning."""
    if getattr(service,'spec',None) is None:return
    service.retained_articulated_model=copy.deepcopy(service.spec)
    reference=getattr(service,'scene_reference_control_step',None)
    if (getattr(service,'scene',None) is not None and reference is not None and
            getattr(service,'contact_scene_kind',None)=='articulated'):
        service.retained_articulated_scene={'scene':copy.copy(service.scene),
            'spec':copy.deepcopy(service.spec),'anchors':copy.deepcopy(service.anchors),
            'choices':copy.deepcopy(getattr(service,'choices',{})),
            'mask':getattr(service,'mask',None),'reference_control_step':int(reference)}


def current_part_points(engine,part_id,track_id,current_step):
    part=engine.parts[part_id];view=engine._view(part['observation_id'],part['camera'])
    points=np.asarray(view['world_points'])[np.asarray(part['mask'],bool) & usable_pixels(view)]
    source_step=int(engine._record(part['observation_id'])['control_steps'])
    evidence={'part_id':part_id,'source_observation_id':part['observation_id'],'source_control_step':source_step}
    if source_step==current_step:return points,evidence
    track=engine.tracks.get(track_id)
    if track is None or track['part_id']!=part_id:
        raise ValueError('Select a current part mask or supply tracks carrying this part to the current capture')
    ids=track['observation_ids'];current=next((i for i,oid in enumerate(ids) if int(engine._record(oid)['control_steps'])==current_step),None)
    if current is None or part['observation_id'] not in ids:
        raise ValueError('The part pose is stale; segment the current part and pass current_part_id')
    reference=ids.index(part['observation_id']);valid=np.asarray(track['valid'])[reference] & np.asarray(track['valid'])[current]
    fit=fit_object_motion(np.asarray(track['world'])[reference,valid],np.asarray(track['world'])[current,valid])
    if not fit['accepted']:raise ValueError('Tracked part pose is unresolved; select a current part mask instead')
    evidence.update(current_observation_id=ids[current],current_pose_source='independent measured RGB-D tracks',pose_fit=fit)
    return transform_points(points,np.asarray(fit['transform'])),evidence


def install_joint_part(service,estimate,current_part_id=None,motion_sign=1):
    engine=service._perception_tools;part_id=current_part_id or estimate['part_id']
    if part_id not in engine.parts:raise ValueError('Use a registered observed part ID')
    if motion_sign not in [-1,1]:raise ValueError('motion_sign must be -1 or +1 relative to the fitted unsigned axis')
    points,evidence=current_part_points(engine,part_id,estimate['track_id'],int(service.state['control_steps']))
    if len(points)<12:raise ValueError('Selected part has insufficient measured depth')
    occupied,_=observed_points(service.cameras,service.state['metadata'],spacing=.003)
    owned=cKDTree(points).query(occupied,distance_upper_bound=.004)[0]<.004
    moving=occupied[owned]
    if len(moving)<12:raise ValueError('The current RGB-D capture does not support this registered part pose')
    center=np.median(moving,axis=0);_,_,basis=np.linalg.svd(moving-center,full_matrices=False);normal=basis[-1]
    camera=engine.parts[part_id]['camera']
    if camera not in service.cameras:camera=next(iter(service.cameras))
    camera_origin=np.asarray(service.cameras[camera]['camera_to_world'])[:3,3]
    if normal@(camera_origin-center)<0:normal=-normal
    fit=estimate['estimate'];axis=motion_sign*unit(fit['axis_world_unsigned'])
    spec={'joint_origin_world_m':list(fit['joint_origin_world_m']),'joint_axis_world':axis.tolist(),
        'front_plane':{'point_world_m':center.tolist()},'panel_normal_world':normal.tolist(),
        'generic_part_joint_model':True,'part_id':part_id,'joint_estimate_id':estimate['estimate_id'],
        'hinge_evidence':{'status':'registered_motion_estimate','estimate_id':estimate['estimate_id'],
            'motion_supported':bool(estimate.get('motion_supported')),'semantic_identity_verified':False},
        'motion':'signed relative rotation; task direction is selected by the caller'}
    robot=RobotVolumes(service.ik.chain,service.state['metadata']['robot_collision_volumes_world'],service.state['metadata']['joint_positions'])
    scene=Scene.from_points(occupied[~owned],moving,moving,spec,config=copy.deepcopy(service.config),robot=robot,
        reference_volumes=service.state['metadata']['robot_collision_volumes_world'])
    service.scene=scene;service.spec=spec;service.cloud=moving;service.anchors=[center.tolist()]
    service.mask=engine.parts[part_id]['mask'];service.choices={'camera':camera}
    service.tracker=None;service.contact_scene_kind='articulated'
    service.scene_reference_control_step=int(service.state['control_steps']);service.epoch+=1
    return {**service.export_geometry(evidence),'planner_model_updated':True,'robot_motion_executed':False,
        'applied_estimate_id':estimate['estimate_id'],'joint_model_from_registered_part':True,
        'handle_model_required':False,'task_direction_verified_by_motion_fit':False,
        'direction_note':'Positive increments follow the exported signed axis; no opening/closing sign is inferred.'}
