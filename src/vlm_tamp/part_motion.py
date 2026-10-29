"""Observed part tracking and independently initialized upright articulation.

Only visible tracks with valid measured depth enter fits. Hand trajectories and
old hinge guesses are never substituted for observed object motion.
"""
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .rigid_hinge_tracking import estimate_rigid_hinge
from .part_perception import usable_pixels


def query_points(view,mask,maximum=200):
    mask=np.asarray(mask,bool)&usable_pixels(view)
    gray=cv2.cvtColor(view['rgb'],cv2.COLOR_RGB2GRAY)
    points=cv2.goodFeaturesToTrack(gray,maxCorners=maximum,qualityLevel=.006,minDistance=5,
                                  mask=mask.astype(np.uint8)*255,blockSize=5)
    points=points.reshape(-1,2).tolist() if points is not None else []
    # Both backends receive exactly this same deterministic query set.
    y,x=np.where(mask)
    if len(x):
        for index in np.linspace(0,len(x)-1,min(len(x),maximum*4),dtype=int):
            point=np.array([x[index],y[index]],float)
            if not points or np.min(np.linalg.norm(np.asarray(points)-point,axis=1))>=5:
                points.append(point.tolist())
            if len(points)>=maximum:break
    return np.asarray(points,np.float32).reshape(-1,2)


def _lk_forward(views,queries):
    count=len(queries);tracks=np.full((len(views),count,2),np.nan,np.float32)
    visible=np.zeros((len(views),count),bool)
    if not count:return tracks,visible
    tracks[0]=queries;visible[0]=True
    for frame in range(1,len(views)):
        previous=cv2.cvtColor(views[frame-1]['rgb'],cv2.COLOR_RGB2GRAY)
        current=cv2.cvtColor(views[frame]['rgb'],cv2.COLOR_RGB2GRAY)
        active=np.flatnonzero(visible[frame-1])
        if not len(active):continue
        first=tracks[frame-1,active].reshape(-1,1,2)
        following,status,_=cv2.calcOpticalFlowPyrLK(previous,current,first,None,winSize=(25,25),maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005))
        if following is None:continue
        back,reverse,_=cv2.calcOpticalFlowPyrLK(current,previous,following,None,winSize=(25,25),maxLevel=3)
        if back is None:continue
        okay=status.ravel().astype(bool)&reverse.ravel().astype(bool)&(np.linalg.norm(back[:,0]-first[:,0],axis=1)<.8)
        finite=np.isfinite(following[:,0]).all(1)
        xy=np.rint(np.where(finite[:,None],following[:,0],-100)).astype(int)
        h,w=current.shape;inside=finite&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        supported=np.zeros(len(xy),bool)
        supported[inside]=usable_pixels(views[frame])[xy[inside,1],xy[inside,0]]
        okay&=supported
        tracks[frame,active]=following[:,0];visible[frame,active]=okay
    return tracks,visible


def lk_tracks(views,queries,reference_index=0):
    """Track around any observed reference, retaining chronological outputs."""
    if not 0<=reference_index<len(views):raise ValueError('Reference frame is outside the clip')
    if reference_index==0:return _lk_forward(views,queries)
    before,visible_before=_lk_forward(list(reversed(views[:reference_index+1])),queries)
    after,visible_after=_lk_forward(views[reference_index:],queries)
    return (np.concatenate([before[::-1][:-1],after],axis=0),
            np.concatenate([visible_before[::-1][:-1],visible_after],axis=0))


class CoTrackerBackend:
    def __init__(self,device='cuda'):
        self.device=device;self.model=None

    def predict(self,views,queries,reference_index=0):
        import torch
        if not 0<=reference_index<len(views):raise ValueError('Reference frame is outside the clip')
        root=Path(__file__).resolve().parents[2]
        source=root/'.runtime/cotracker';checkpoint=root/'.runtime/cotracker-checkpoints/scaled_offline.pth'
        timing={};start=time.perf_counter()
        if self.model is None:
            if not source.is_dir() or not checkpoint.is_file():raise ValueError('CoTracker3 runtime/checkpoint is unavailable')
            sys.path.insert(0,str(source))
            from cotracker.predictor import CoTrackerPredictor
            self.model=CoTrackerPredictor(checkpoint=str(checkpoint),offline=True,v2=False).eval().to(self.device)
            if self.device.startswith('cuda'):torch.cuda.synchronize()
            timing['model_load']=time.perf_counter()-start
        else:timing['model_load']=0.
        if not len(queries):return np.empty((len(views),0,2)),np.empty((len(views),0),bool),timing
        start=time.perf_counter()
        video=torch.from_numpy(np.stack([v['rgb'] for v in views])).permute(0,3,1,2)[None].float().to(self.device)
        prompt=torch.from_numpy(np.c_[np.full(len(queries),reference_index),queries].astype(np.float32))[None].to(self.device)
        with torch.inference_mode():tracks,visible=self.model(video,queries=prompt,backward_tracking=reference_index>0)
        if self.device.startswith('cuda'):torch.cuda.synchronize()
        timing['inference']=time.perf_counter()-start
        return tracks[0].cpu().numpy(),visible[0].cpu().numpy().astype(bool),timing

    def close(self):self.model=None


def lift_tracks(views,tracks,visible):
    """Use each frame's measured world points; never lift an occluded prediction."""
    world=np.full((*tracks.shape[:2],3),np.nan,float);valid=np.zeros(tracks.shape[:2],bool)
    for frame,view in enumerate(views):
        height,width=view['rgb'].shape[:2];points=tracks[frame]
        finite=np.isfinite(points).all(1)
        xy=np.rint(np.where(finite[:,None],points,-100)).astype(int)
        inside=finite&visible[frame]&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
        indices=np.flatnonzero(inside);selected=xy[indices]
        indices=indices[usable_pixels(view)[selected[:,1],selected[:,0]]]
        valid[frame,indices]=True
        world[frame,indices]=np.asarray(view['world_points'])[xy[indices,1],xy[indices,0]]
    return world,valid


def pair_points(world,valid,first,second):
    okay=valid[first]&valid[second]&np.isfinite(world[first]).all(1)&np.isfinite(world[second]).all(1)
    return world[first,okay],world[second,okay]


def fit_upright(world,valid,first,second):
    a,b=pair_points(world,valid,first,second)
    # The visible points need not be at the handle's lever arm. In particular,
    # near-axis features may still have adequate span and angular information.
    # Keep the original tracker default; independent perception uses no positive
    # minimum radius and retains all other observability/residual gates.
    try:result=estimate_rigid_hinge(a,b,prior=None,seed=42,minimum_radius_m=0.)
    except (ValueError,np.linalg.LinAlgError) as error:
        result={'accepted':False,'pose_accepted':False,'reason':'degenerate_motion_geometry','detail':str(error)}
    def finite(value):
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        if isinstance(value,list):return [finite(v) for v in value]
        if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
        return value
    if any(isinstance(v,(float,np.floating)) and not np.isfinite(v) for v in result.values()):
        result.update(accepted=False,reason='nonfinite_motion_geometry')
    result=finite(result)
    result.update(initialization='independent observed-motion fit; no old hinge prior',
        axis_world_unsigned=[0.,0.,1.],joint_type_candidate='upright_revolute',
        measured_rotation_deg=float(np.degrees(result['rotation_rad'])) if result.get('rotation_rad') is not None else None,
        finite_visible_pairs=len(a),fit_frame_indices=[first,second],
        semantic_identity_verified=False,physical_contact_verified=False)
    result['minimum_visible_point_radius_prior_m']=0.
    angle=abs(result['measured_rotation_deg']) if result['measured_rotation_deg'] is not None else None
    measured_checks={
        'six_moving_landmarks':result.get('correspondences',0)>=6,
        'observable_rigid_pose':bool(result.get('pose_accepted')),
        'xy_span_at_least_60mm':result['landmark_span_m']>=.06 if result.get('landmark_span_m') is not None else None,
        'rotation_between_1_5_and_70deg':1.5<=angle<70 if angle is not None else None,
        'rigid_rms_below_2mm':result['rms_m']<.002 if result.get('rms_m') is not None else None,
        'bootstrap_spread_below_15mm':result['bootstrap_spread90_m']<.015 if result.get('bootstrap_spread90_m') is not None else None,
        'visible_point_radius_below_700mm':0<result['radius_m']<.70 if result.get('radius_m') is not None else None,
    }
    result['observability_checks']=measured_checks
    result['failed_checks']=[name for name,value in measured_checks.items() if value is False]
    if result.get('accepted'):
        result['joint_origin_world_m']=result['hinge_world_m']
    return result


def verify_upright_axis(world,valid,first,second,origin):
    """Test a fixed axis against a separate observed motion interval."""
    a,b=pair_points(world,valid,first,second)
    measured=fit_upright(world,valid,first,second)
    report={'accepted':False,'frame_indices':[first,second],'finite_visible_pairs':len(a),
        'axis_used_for_motion_fit':False,'measured_rotation_deg':measured['measured_rotation_deg']}
    # Validation of a FIXED axis requires observed rigid rotation, not another
    # independently admitted pivot. Visible points may be close to the axis even
    # when an earlier, wider view initialized it well.
    if not measured.get('pose_accepted') or abs(measured.get('measured_rotation_deg') or 0)<1.5:
        return {**report,'reason':'verification_motion_not_observable','motion_fit_reason':measured.get('reason')}
    rotation=np.asarray(measured['rotation_xy']);translation=np.asarray(measured['translation_xy_m'])
    movement=np.linalg.norm(b[:,:2]-a[:,:2],axis=1)
    rigid_error=np.linalg.norm(a[:,:2]@rotation.T+translation-b[:,:2],axis=1)
    good=(movement>.0015)&(movement<.15)&(np.abs(b[:,2]-a[:,2])<.004)&(rigid_error<.0025)
    if good.sum()<6:return {**report,'reason':'insufficient_verified_moving_points'}
    origin=np.asarray(origin,float)[:2]
    predicted=(a[good,:2]-origin)@rotation.T+origin
    residual=np.linalg.norm(predicted-b[good,:2],axis=1)
    rms=float(np.sqrt(np.mean(residual**2)));p90=float(np.quantile(residual,.90))
    return {**report,'accepted':rms<=.003 and p90<=.005,
        'reason':'held_out_motion_supports_axis' if rms<=.003 and p90<=.005 else 'axis_disagrees_with_held_out_motion',
        'moving_inliers':int(good.sum()),'axis_prediction_rms_m':rms,'axis_prediction_p90_m':p90,
        'verification_pivot_difference_m':float(np.linalg.norm(np.asarray(measured['hinge_world_m'])[:2]-origin)),
        'scope':'Axis prediction checked on a separate interval. The interval rotation and rigid inliers are measured without using the proposed axis.'}


def estimate_and_verify(world,valid,fit_indices,verify_indices=None):
    fit=fit_upright(world,valid,*fit_indices)
    result={'status':'insufficient_evidence','estimate':fit,'verification':None,
        'motion_supported':False,'planner_model_updated':False,'robot_motion_executed':False}
    if not fit['accepted']:return result
    result['status']='motion_estimate_unverified'
    if verify_indices is not None:
        if max(verify_indices)<=max(fit_indices) or min(verify_indices)<max(fit_indices):
            raise ValueError('Verification must use a later motion interval not used to fit the axis')
        verification=verify_upright_axis(world,valid,*verify_indices,fit['joint_origin_world_m'])
        inconclusive=verification.get('reason') in {
            'verification_motion_not_observable','insufficient_verified_moving_points'}
        result.update(verification=verification,motion_supported=verification['accepted'],
                      status='motion_supported' if verification['accepted'] else
                          'verification_inconclusive' if inconclusive else 'verification_failed')
    return result
