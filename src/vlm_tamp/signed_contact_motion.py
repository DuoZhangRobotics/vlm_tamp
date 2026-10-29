"""Signed joint requests, decomposed into bounded planning coordinates.

The observed scene is never mutated by a hypothetical continuation. Large
requests use a new reference frame per segment rather than wrapping atan2.
"""
import copy
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .coupled_storage import write_json,preserve_scene
from .rekep_libero.geometry import pose_matrix,transform_points


def signed_scene(scene,degrees):
    if not np.isfinite(degrees) or degrees==0:raise ValueError('A finite nonzero signed angle is required')
    result=copy.copy(scene);result.spec=copy.deepcopy(scene.spec)
    result.axis=np.asarray(scene.axis,float)*np.sign(degrees)
    result.spec['joint_axis_world']=result.axis.tolist()
    result._moving_cache={}
    return result


def predicted_rotation(scene,angle):
    from .scene_pose_sync import transport_scene
    r=Rotation.from_rotvec(np.asarray(scene.axis)*angle).as_matrix();t=np.eye(4)
    t[:3,:3]=r;t[:3,3]=scene.origin-r@scene.origin
    result=transport_scene(scene,t,source='hand_motion_prediction',reference_step=0)
    result.pose_update['scope']='planning continuation only; never installed as measured scene'
    return result


def plan_signed_arc(ik,scene,record,metadata,start,degrees,folder,config):
    from .arc_execution import contact_arc
    query=signed_scene(scene,degrees)
    angle=abs(float(np.radians(degrees)));folder=Path(folder)
    if angle<=np.pi/2+1e-12:
        path,result=contact_arc(query,record,metadata,start,angle,folder,config,ik=ik)
        return path,{**result,'requested_signed_degrees':degrees,'signed_axis_world':query.axis.tolist()}
    folder.mkdir(parents=True,exist_ok=False);preserve_scene(scene,folder)
    started=time.perf_counter();deadline=started+float(config.get('arc_search_seconds',20.))
    remaining=angle;paths=[];reports=[];q=np.asarray(start);meta=copy.deepcopy(metadata)
    while remaining>1e-10:
        seconds=deadline-time.perf_counter()
        if seconds<=0:
            report={'status':'search_budget_exhausted','search_complete':False,'requested_signed_degrees':degrees,
                'planned_degrees':float(np.degrees(angle-remaining))*np.sign(degrees),'segments':reports}
            write_json(folder/'result.json',report);raise ValueError('Signed-arc search budget exhausted before the complete request was planned')
        chunk=min(np.pi/2,remaining)
        path,result=contact_arc(query,record,meta,q,chunk,folder/f'segment{len(paths):03d}',
            {**config,'arc_search_seconds':seconds},ik=ik)
        paths.append(path if not paths else path[1:]);reports.append(result);q=path[-1]
        remaining-=chunk
        if remaining>1e-10:
            query=predicted_rotation(query,chunk)
            meta={**meta,'joint_positions':q.tolist(),'eef_pose_world_xyzw':np.asarray(ik.chain.forward(q)).tolist(),
                'robot_collision_volumes_world':query.robot.forward(q)}
    path=np.vstack(paths);result={'status':'solved','exact_solution':True,'requested_signed_degrees':degrees,
        'signed_axis_world':query.axis.tolist(),'segments':reports,'total_seconds':time.perf_counter()-started,
        'observed_scene_mutated':False,'progress_unwrapped_by_segments':True}
    np.save(folder/'path.npy',path);write_json(folder/'result.json',result)
    return path,result
