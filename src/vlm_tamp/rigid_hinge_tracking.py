"""Sensor-only upright hinge estimation from persistent RGB-D door landmarks.

Pyramidal optical flow proposes image correspondences. Their measured world
points estimate an SE(2) rigid motion; stationary background, robot pixels and
inconsistent matches are rejected. No simulator articulation enters this code.
"""
import numpy as np
from scipy.spatial import cKDTree
import cv2


def fit_se2(first,second):
    first=np.asarray(first,float);second=np.asarray(second,float)
    a=first.mean(0);b=second.mean(0)
    u,_,vt=np.linalg.svd((first-a).T@(second-b))
    R=vt.T@u.T
    if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
    return R,b-R@a


def estimate_rigid_hinge(first,second,*,prior=None,seed=0):
    first=np.asarray(first,float).reshape(-1,3);second=np.asarray(second,float).reshape(-1,3)
    valid=np.isfinite(first).all(1)&np.isfinite(second).all(1)
    movement=np.linalg.norm(second[:,:2]-first[:,:2],axis=1)
    valid&=(movement>.0015)&(movement<.15)&(np.abs(second[:,2]-first[:,2])<.004)
    first,second=first[valid],second[valid]
    result={'accepted':False,'pose_accepted':False,'source':'persistent RGB-D landmarks; robust upright SE2 motion',
        'simulator_geometry_used':False,'correspondences':len(first)}
    if len(first)<6:return {**result,'reason':'insufficient_moving_landmarks'}
    rng=np.random.default_rng(seed);best=None
    for _ in range(350):
        ids=rng.choice(len(first),2,replace=False)
        if np.linalg.norm(first[ids[0],:2]-first[ids[1],:2])<.04:continue
        R,t=fit_se2(first[ids,:2],second[ids,:2])
        error=np.linalg.norm(first[:,:2]@R.T+t-second[:,:2],axis=1)
        inliers=error<.0025
        if best is None or inliers.sum()>best.sum():best=inliers
    if best is None or best.sum()<6:return {**result,'reason':'no_consistent_rigid_motion'}
    R,t=fit_se2(first[best,:2],second[best,:2]);error=np.linalg.norm(first[:,:2]@R.T+t-second[:,:2],axis=1)
    best=error<.0025
    R,t=fit_se2(first[best,:2],second[best,:2])
    error=np.linalg.norm(first[:,:2]@R.T+t-second[:,:2],axis=1)
    angle=float(np.arctan2(R[1,0],R[0,0]));span=float(np.max(np.linalg.norm(first[best,None,:2]-first[None,best,:2],axis=2)))
    result.update(inliers=int(best.sum()),rotation_rad=angle,landmark_span_m=span,
        rms_m=float(np.sqrt(np.mean(error[best]**2))),rotation_xy=R.tolist(),translation_xy_m=t.tolist())
    result['pose_accepted']=bool(span>=.06 and result['rms_m']<.002 and abs(angle)<np.radians(70))
    if abs(angle)<np.radians(1.5) or span<.06:return {**result,'reason':'insufficient_rotation_or_spatial_baseline'}
    origin=np.linalg.solve(np.eye(2)-R,t)
    pivots=[];indices=np.flatnonzero(best)
    for _ in range(60):
        sample=rng.choice(indices,len(indices),replace=True)
        rr,tt=fit_se2(first[sample,:2],second[sample,:2])
        if abs(np.linalg.det(np.eye(2)-rr))>1e-5:pivots.append(np.linalg.solve(np.eye(2)-rr,tt))
    spread=float(np.quantile(np.linalg.norm(np.asarray(pivots)-origin,axis=1),.90)) if pivots else float('inf')
    radius=float(np.median(np.linalg.norm(first[best,:2]-origin,axis=1)))
    result.update(hinge_world_m=[*origin.tolist(),float(np.mean(first[best,2]))],
        bootstrap_spread90_m=spread,bootstrap_is_geometric_stability_not_calibrated_uncertainty=True,
        radius_m=radius)
    okay=(spread<.015 and .06<radius<.70 and abs(angle)<np.radians(70) and result['rms_m']<.002)
    if prior is not None:okay=okay and np.linalg.norm(origin-np.asarray(prior)[:2])<.10
    result.update(accepted=bool(okay),reason='accepted' if okay else 'unstable_or_inconsistent_hinge')
    return result


def measured_pose(result):
    """A measured door pose may be observable before its pivot is stable."""
    if not result.get('pose_accepted',False):return None
    transform=np.eye(4)
    transform[:2,:2]=result['rotation_xy'];transform[:2,3]=result['translation_xy_m']
    return transform


def robot_mask(camera):
    depth=np.asarray(camera['depth']).squeeze();prior=camera.get('known_robot_depth_prior')
    if prior is None:return np.zeros(depth.shape,bool)
    prior=np.asarray(prior).squeeze()
    return np.isfinite(prior)&(prior>0)&(np.abs(depth-prior)<.0003)


class RGBDHingeTracker:
    def __init__(self,cameras,moving_points,*,prior,reference_step=0,output=None):
        self.prior=np.asarray(prior,float);self.references={};self.reference_step=reference_step
        tree=cKDTree(np.asarray(moving_points,float))
        for name in ['agentview','frontview','arm_side','arm_back','arm_top']:
            if name not in cameras:continue
            view=cameras[name];xyz=np.asarray(view['world_points']);finite=np.isfinite(xyz).all(2)
            mask=np.zeros(finite.shape,bool);mask[finite]=tree.query(xyz[finite])[0]<.004
            mask&=~robot_mask(view)
            mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))
            gray=cv2.cvtColor(view['rgb'],cv2.COLOR_RGB2GRAY)
            pixels=cv2.goodFeaturesToTrack(gray,maxCorners=350,qualityLevel=.006,minDistance=5,mask=mask*255,blockSize=5)
            if pixels is None:continue
            self.references[name]={'gray':gray,'points':xyz.copy(),'pixels':pixels,'mask':mask}
        if output is not None:
            data={}
            for name,view in self.references.items():
                for key,value in view.items():data[name+'_'+key]=value
            np.savez_compressed(output,**data)

    def observe(self,cameras,*,control_step,output=None):
        first=[];second=[];views=[]
        for name,reference in self.references.items():
            view=cameras[name];gray=cv2.cvtColor(view['rgb'],cv2.COLOR_RGB2GRAY)
            original=reference['pixels']
            tracked,status,_=cv2.calcOpticalFlowPyrLK(reference['gray'],gray,original,None,winSize=(25,25),maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005))
            if tracked is None:continue
            back,reverse,_=cv2.calcOpticalFlowPyrLK(gray,reference['gray'],tracked,None,winSize=(25,25),maxLevel=3)
            a=original.reshape(-1,2);b=tracked.reshape(-1,2)
            valid=status.ravel().astype(bool)&reverse.ravel().astype(bool)&(np.linalg.norm(back.reshape(-1,2)-a,axis=1)<.8)
            h,w=gray.shape;valid&=(b[:,0]>=2)&(b[:,0]<w-2)&(b[:,1]>=2)&(b[:,1]<h-2)
            mask=robot_mask(view);p0=[];p1=[]
            for x,y in zip(a[valid],b[valid]):
                ix,iy=np.rint(x).astype(int);jx,jy=np.rint(y).astype(int)
                if mask[jy,jx]:continue
                old=reference['points'][iy,ix];new=view['world_points'][jy,jx]
                if np.isfinite(old).all() and np.isfinite(new).all():p0.append(old);p1.append(new)
            first.extend(p0);second.extend(p1);views.append({'camera':name,'tracked_pairs':len(p0)})
        a=np.asarray(first,float).reshape(-1,3);b=np.asarray(second,float).reshape(-1,3)
        result=estimate_rigid_hinge(a,b,prior=self.prior)
        result.update(camera_counts=views,reference_control_step=self.reference_step,control_step=int(control_step))
        if output is not None:np.savez_compressed(output,reference_world=a,current_world=b)
        return result
