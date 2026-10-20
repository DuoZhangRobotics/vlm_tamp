"""Observed rigid-object transport evidence; no simulator object state."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import pose_matrix, transform_points


def remove_robot_points(points, volumes):
    points=np.asarray(points,float)
    self_points=np.zeros(len(points),bool)
    for volume in volumes:
        if volume.get('type')!='convex':
            continue
        vertices=np.asarray(volume['vertices_world'])
        local=((points>=vertices.min(axis=0)-1e-6).all(axis=1)
               &(points<=vertices.max(axis=0)+1e-6).all(axis=1)&~self_points)
        indices=np.flatnonzero(local)
        planes=np.asarray(volume['planes_world'])
        for offset in range(0,len(indices),20000):
            batch=indices[offset:offset+20000]
            self_points[batch]=np.all(points[batch]@planes[:,:3].T+planes[:,3]<=1e-6,axis=1)
    return points[~self_points]


def observed_object_cloud(cameras, reference, predicted_transform, robot_volumes):
    reference=np.asarray(reference,float)
    predicted=transform_points(reference,predicted_transform)
    low=np.minimum(reference.min(axis=0),predicted.min(axis=0))-.025
    high=np.maximum(reference.max(axis=0),predicted.max(axis=0))+.025
    # Avoid treating the original tabletop plane as observed object motion.
    low[2]=max(low[2],reference.min(axis=0)[2]+.0005)
    clouds=[]
    for camera in cameras.values():
        points=np.asarray(camera['world_points']).reshape(-1,3)
        keep=np.isfinite(points).all(axis=1)&(points>=low).all(axis=1)&(points<=high).all(axis=1)
        clouds.append(points[keep])
    observed=np.vstack(clouds)
    observed=remove_robot_points(observed,robot_volumes)
    if len(observed):
        cells=np.floor(observed/.002).astype(int)
        observed=observed[np.unique(cells,axis=0,return_index=True)[1]]
    return observed


def _score(template, tree):
    distances=tree.query(template)[0]
    return {'coverage_6mm':float(np.mean(distances<.006)),
            'median_m':float(np.median(distances)), 'percentile75_m':float(np.quantile(distances,.75)),
            'percentile90_m':float(np.quantile(distances,.90))}


def verify_observed_transport(reference, observed, predicted_transform, gripper_qpos,
                              *,minimum_motion=.035,empty_gap=.001):
    """Compare following versus stationary geometry and fit a bounded rigid pose."""
    reference=np.asarray(reference,float);observed=np.asarray(observed,float)
    predicted_transform=pose_matrix(predicted_transform)
    result={'verified':False,'source':'RGB-D rigid surface transport and physical closure',
            'reference_points':len(reference),'observed_points':len(observed),
            'physical_gap_m':float(np.sum(np.abs(gripper_qpos))),'new_reference_state_used':False}
    if len(reference)<75 or len(observed)<75 or not np.isfinite(observed).all():
        return dict(result,failure='insufficient_observed_object_support')
    tree=cKDTree(observed)
    following=transform_points(reference,predicted_transform)
    stationary_score=_score(reference,tree);predicted_score=_score(following,tree)
    transform=predicted_transform.copy()
    for _ in range(15):
        moved=transform_points(reference,transform)
        distance,index=tree.query(moved)
        keep=distance<.015
        if keep.sum()<75:
            break
        a,b=moved[keep],observed[index[keep]]
        ac,bc=a.mean(axis=0),b.mean(axis=0)
        u,_,vt=np.linalg.svd((a-ac).T@(b-bc))
        rotation=vt.T@u.T
        if np.linalg.det(rotation)<0:
            vt[-1]*=-1;rotation=vt.T@u.T
        delta=np.eye(4);delta[:3,:3]=rotation;delta[:3,3]=bc-rotation@ac
        proposed=delta@transform
        relative=proposed@np.linalg.inv(predicted_transform)
        if np.linalg.norm(relative[:3,3])>.015 or Rotation.from_matrix(relative[:3,:3]).magnitude()>.25:
            break
        transform=proposed
        if np.linalg.norm(delta[:3,3])<1e-5 and Rotation.from_matrix(rotation).magnitude()<1e-4:
            break
    score=_score(transform_points(reference,transform),tree)
    motion=float(np.linalg.norm(transform_points(reference.mean(axis=0)[None],transform)[0]-reference.mean(axis=0)))
    following_better=(score['percentile75_m']+.0015<stationary_score['percentile75_m'])
    gap=result['physical_gap_m']
    verified=(empty_gap+.001<gap<.079 and motion>=minimum_motion
              and score['coverage_6mm']>=.60 and score['percentile75_m']<=.009 and following_better)
    result.update(verified=bool(verified),stationary=stationary_score,predicted=predicted_score,
                  aligned=score,observed_motion_m=motion,following_better=bool(following_better),
                  transform_world=transform.tolist(),minimum_motion_m=minimum_motion,
                  empty_gap_calibration_m=empty_gap)
    if not verified:
        result['failure']='observed_surfaces_do_not_establish_retained_transport'
    return result
