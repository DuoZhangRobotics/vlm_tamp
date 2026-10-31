"""Planning clouds consistent with an already measured physical robot state.

Call at a measured-state planning boundary, before creating hypothetical poses.
The returned scene is a query copy; sensor data and perception references stay
available. Predicted continuations transport the cleaned points without clearing
any additional points against candidate robot configurations.
"""
import copy
import time

import numpy as np
from scipy.spatial import cKDTree

from .coupled_scene import RobotVolumes


def filter_summary(record):
    if not record:return None
    return {key:record.get(key) for key in ('source','reference_control_step','padding_m',
        'total_removed_points','elapsed_seconds')} | {
        'removed_by_cloud':{key:value['removed_points'] for key,value in record.get('clouds',{}).items()}}


def points_outside_current_robot(points,metadata):
    """Subtract exact current robot volumes, with no padding or future sweep."""
    points=np.asarray(points,float).reshape(-1,3)
    volumes=metadata['robot_collision_volumes_world']
    tree=cKDTree(points);removed=np.zeros(len(points),bool);by_part={}
    for volume in volumes:
        if volume['type']=='convex':
            vertices=np.asarray(volume['vertices_world'],float);center=vertices.mean(0)
            ids=np.asarray(tree.query_ball_point(center,np.linalg.norm(vertices-center,axis=1).max()),int)
            planes=np.asarray(volume['planes_world'],float)
            inside=ids[np.max(points[ids]@planes[:,:3].T+planes[:,3],axis=1)<=0] if len(ids) else ids
        elif volume['type']=='sphere':
            inside=np.asarray(tree.query_ball_point(volume['center_world'],float(volume['radius'])),int)
        else:raise ValueError('Unsupported configured robot volume: '+volume['type'])
        removed[inside]=True
        if len(inside):by_part[volume['robot_geom_name']]=int(len(inside))
    kept=points[~removed];kept.setflags(write=False)
    return kept,{'input_points':len(points),'removed_points':int(removed.sum()),
        'remaining_points':len(kept),'points_inside_each_robot_part':by_part}


def clear_current_robot_scene(scene,metadata,control_step=None):
    """Prepare one shared collision world from the actual measured start.

    Contact references locate the intended contact; they are not extra obstacle
    points. Keep them for association and grounding, while filtering every cloud
    that is actually indexed as collision geometry.
    """
    started=time.perf_counter();result=copy.copy(scene);counts={}
    for name in ('static_points','moving_reference','inferred_panel_reference'):
        points=getattr(scene,name,np.empty((0,3)))
        filtered,counts[name]=points_outside_current_robot(points,metadata)
        setattr(result,name,filtered)
    result.static_tree=cKDTree(result.static_points)
    result.moving_tree=cKDTree(result.moving_reference)
    result.contact_tree=cKDTree(result.contact_reference)
    result.collision_reference=np.vstack([result.moving_reference,result.inferred_panel_reference])
    result.collision_reference.setflags(write=False)
    result.collision_tree=cKDTree(result.collision_reference)
    result._moving_cache={}
    result._reference_contact_distance=result.contact_tree.query(result.collision_reference)[0]
    result._reference_panel_distance=np.abs((result.collision_reference-result.panel_origin)@result.panel_normal)
    if getattr(scene,'robot',None) is not None:
        result.robot=RobotVolumes(scene.robot.chain,metadata['robot_collision_volumes_world'],metadata['joint_positions'])
    result.current_robot_filter={'source':'measured_current_robot_geometry',
        'reference_control_step':control_step,'joint_positions':list(metadata['joint_positions']),
        'gripper_qpos':list(metadata.get('gripper_qpos',[])),
        'padding_m':0.,'future_candidate_poses_used':False,'raw_observations_modified':False,
        'contact_reference_is_association_only':True,'clouds':counts,
        'total_removed_points':sum(r['removed_points'] for r in counts.values()),
        'elapsed_seconds':time.perf_counter()-started}
    return result
