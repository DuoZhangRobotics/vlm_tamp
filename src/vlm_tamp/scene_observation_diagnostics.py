"""Read-only, multi-view depth support for a current world-space scene model."""
import numpy as np
from PIL import Image,ImageDraw

from .rekep_libero.geometry import pose_matrix,transform_points


def depth_support(points,cameras,*,tolerance_m=.005,maximum_points=1200):
    points=np.asarray(points,float).reshape(-1,3)
    points=points[np.isfinite(points).all(1)]
    if len(points)>maximum_points:
        points=points[np.linspace(0,len(points)-1,maximum_points,dtype=int)]
    supported=np.zeros(len(points),int);free=np.zeros(len(points),int);observed=np.zeros(len(points),int)
    views={}
    for name,view in cameras.items():
        optical=transform_points(points,np.linalg.inv(pose_matrix(view['camera_to_world'])))
        projected=optical@np.asarray(view['intrinsic'],float).T
        uv=np.full((len(points),2),-1,int)
        forward=optical[:,2]>1e-6
        uv[forward]=np.rint(projected[forward,:2]/projected[forward,2:3]).astype(int)
        depth=np.asarray(view['depth']).squeeze();h,w=depth.shape
        valid=forward&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        ids=np.flatnonzero(valid);xy=uv[ids]
        values=depth[xy[:,1],xy[:,0]]
        usable=np.isfinite(values)&(values>0)
        if 'world_points' in view:
            usable&=np.isfinite(np.asarray(view['world_points'])[xy[:,1],xy[:,0]]).all(1)
        if 'far_m' in view:usable&=values<float(view['far_m'])*.99
        ids=ids[usable];values=values[usable]
        delta=values-optical[ids,2]
        support=np.abs(delta)<=tolerance_m;contradiction=delta>tolerance_m
        supported[ids]+=support;free[ids]+=contradiction;observed[ids]+=1
        views[name]={'usable_projections':len(ids),'supporting':int(support.sum()),
                     'in_observed_free_space':int(contradiction.sum()),
                     'behind_measured_surface':int((delta < -tolerance_m).sum())}
    status=np.zeros(len(points),np.uint8)
    status[(supported==0)&(free>=2)]=2
    status[supported>0]=1
    total=max(1,len(points))
    result={'sampled_points':len(points),'supported_fraction':float(np.count_nonzero(status==1)/total),
        'multi_view_free_space_conflict_fraction':float(np.count_nonzero(status==2)/total),
        'occluded_or_unresolved_fraction':float(np.count_nonzero(status==0)/total),
        'depth_tolerance_m':tolerance_m,'minimum_conflicting_views':2,'camera_evidence':views,
        'execution_veto':False,'model_changed':False,'simulator_object_state_used':False,
        'interpretation':'Geometric depth support, not semantic identity or a contact-retention certificate. Occlusion, transparency and calibration can affect this evidence.'}
    return result,points,status


def overlay(points,status,view,destination):
    canvas=Image.fromarray(np.asarray(view['rgb'],np.uint8)).convert('RGB')
    draw=ImageDraw.Draw(canvas)
    optical=transform_points(points,np.linalg.inv(pose_matrix(view['camera_to_world'])))
    projected=optical@np.asarray(view['intrinsic']).T
    colors={0:(155,155,155),1:(40,210,110),2:(245,60,50)}
    for i in range(len(points)):
        if optical[i,2]<=0:continue
        x,y=projected[i,:2]/projected[i,2]
        if not (0<=x<canvas.width and 0<=y<canvas.height):continue
        draw.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill=colors[int(status[i])])
    draw.rectangle((0,0,canvas.width,44),fill=(20,25,30))
    draw.text((7,5),'MODEL POINTS: green depth support; red free-space conflict',fill='white')
    draw.text((7,22),'gray occluded/unknown; evidence combines available views',fill='white')
    canvas.save(destination)
