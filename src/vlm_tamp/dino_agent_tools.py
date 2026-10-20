"""Optional appearance proposals and correspondence diagnostics for a VLM agent.

DINO sees RGB only. Measured depth supplies output positions and excludes robot
pixels. These proposals have no semantic labels or calibrated probabilities.
"""
import numpy as np


def valid_grid(world, depth, box=None, robot_depth=None, maximum=4096):
    world=np.asarray(world);depth=np.asarray(depth).squeeze();h,w=depth.shape
    if world.shape!=(h,w,3):raise ValueError('World points and depth must share the RGB grid')
    bounds=np.array([0,0,w-1,h-1] if box is None else box,float)
    if bounds.shape!=(4,) or not np.isfinite(bounds).all():raise ValueError('Expected four finite image bounds')
    x0,y0,x1,y1=np.rint(bounds).astype(int)
    if not (0<=x0<x1<w and 0<=y0<y1<h):raise ValueError('Region must lie inside the original image')
    step=max(2,int(np.ceil(np.sqrt((x1-x0+1)*(y1-y0+1)/maximum))))
    yy,xx=np.meshgrid(np.arange(y0,y1+1,step),np.arange(x0,x1+1,step),indexing='ij')
    pixels=np.column_stack([xx.ravel(),yy.ravel()]);x,y=pixels.T
    valid=np.isfinite(world[y,x]).all(1)&np.isfinite(depth[y,x])&(depth[y,x]>.01)&(depth[y,x]<5.)
    if robot_depth is not None:
        prior=np.asarray(robot_depth).squeeze()[y,x]
        valid&=~(np.isfinite(prior)&(prior>0)&(np.abs(depth[y,x]-prior)<.0003))
    return pixels[valid],step


def normalized(values):
    values=np.asarray(values,np.float32)
    return values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-12)


def propose_landmarks(features, world, depth, *, box=None, count=8, robot_depth=None):
    if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=32:
        raise ValueError('Choose 1..32 candidate landmarks')
    pixels,step=valid_grid(world,depth,box,robot_depth)
    if not len(pixels):return {'candidates':[],'reason':'no_valid_observed_depth','grid_stride_px':step}
    x,y=pixels.T;appearance=normalized(np.asarray(features)[y,x]);positions=np.asarray(world)[y,x]
    scaled=(positions-positions.min(0))/np.maximum(np.ptp(positions,axis=0),1e-6)
    center=pixels.mean(0);selected=[];distance=np.full(len(pixels),np.inf)
    index=int(np.argmin(np.sum((pixels-center)**2,axis=1)))
    for _ in range(min(count,len(pixels))):
        selected.append(index)
        current=.65*(1-appearance@appearance[index])+.35*np.sum((scaled-scaled[index])**2,axis=1)/3
        distance=np.minimum(distance,current)
        close=np.linalg.norm(pixels-pixels[index],axis=1)<10
        distance[close]=-np.inf
        if not np.isfinite(distance).any():break
        index=int(np.argmax(distance))
    rows=[{'id':i+1,'pixel_xy':pixels[index].tolist(),
           'world_m':positions[index].astype(float).tolist(),
           'optical_depth_m':float(np.asarray(depth).squeeze()[pixels[index,1],pixels[index,0]])}
          for i,index in enumerate(selected)]
    return {'candidates':rows,'grid_stride_px':step,
        'method':'Deterministic farthest-point proposals mixing DINO appearance distance and normalized measured 3D spread',
        'appearance_weight':.65,'spatial_weight':.35,'semantic_labels_provided':False,
        'candidate_is_not_a_verified_hinge_or_grasp':True}


def match_landmarks(reference_features,current_features,reference_world,current_world,
                    reference_depth,current_depth,pixels,*,box=None,reference_robot=None,current_robot=None):
    queries=np.asarray(pixels,float)
    if queries.ndim!=2 or queries.shape[1]!=2 or not 1<=len(queries)<=24 or not np.isfinite(queries).all():
        raise ValueError('Provide 1..24 finite reference pixels')
    h,w=np.asarray(reference_depth).squeeze().shape
    queries=np.rint(queries).astype(int)
    if (queries<0).any() or (queries[:,0]>=w).any() or (queries[:,1]>=h).any():raise ValueError('Reference pixels lie outside their image')
    targets,stride=valid_grid(current_world,current_depth,box,current_robot)
    references,_=valid_grid(reference_world,reference_depth,None,reference_robot)
    if not len(targets) or not len(references):return {'matches':[],'reason':'no_valid_observed_depth'}
    target_features=normalized(np.asarray(current_features)[targets[:,1],targets[:,0]])
    reference_samples=normalized(np.asarray(reference_features)[references[:,1],references[:,0]])
    rows=[]
    for number,pixel in enumerate(queries,1):
        x,y=pixel;point=np.asarray(reference_world)[y,x]
        depth=float(np.asarray(reference_depth).squeeze()[y,x])
        if not np.isfinite(point).all() or not np.isfinite(depth) or not .01<depth<5.:
            rows.append({'id':number,'reference_pixel_xy':pixel.tolist(),'candidates':[],'reason':'invalid_reference_depth'});continue
        if reference_robot is not None:
            rp=float(np.asarray(reference_robot).squeeze()[y,x])
            if np.isfinite(rp) and rp>0 and abs(depth-rp)<.0003:
                rows.append({'id':number,'reference_pixel_xy':pixel.tolist(),'candidates':[],'reason':'reference_pixel_is_robot'});continue
        scores=target_features@normalized(np.asarray(reference_features)[y,x])
        order=np.argsort(-scores,kind='stable');chosen=[]
        for index in order:
            if any(np.linalg.norm(targets[index]-targets[j])<16 for j in chosen):continue
            chosen.append(int(index))
            if len(chosen)==3:break
        best=chosen[0];back=int(np.argmax(reference_samples@target_features[best]))
        cycle=float(np.linalg.norm(references[back]-pixel))
        margin=float(scores[best]-scores[chosen[1]]) if len(chosen)>1 else None
        candidates=[]
        for index in chosen:
            tx,ty=targets[index];matched=np.asarray(current_world)[ty,tx]
            candidates.append({'pixel_xy':[int(tx),int(ty)],'world_m':matched.astype(float).tolist(),
                'cosine_similarity':float(scores[index]),'observed_displacement_m':float(np.linalg.norm(matched-point))})
        rows.append({'id':number,'reference_pixel_xy':pixel.tolist(),'reference_world_m':point.astype(float).tolist(),
            'candidates':candidates,'reciprocal_pixel_error_px':cycle,'nonlocal_similarity_margin':margin,
            'ambiguous':bool(cycle>16 or margin is None or margin<.02)})
    return {'matches':rows,'target_grid_stride_px':stride,
        'scores_are_not_calibrated_probabilities':True,
        'ambiguity_rule':'Reciprocal error >16 pixels or nonlocal cosine margin <0.02; heuristic, not a guarantee',
        'model_or_scene_updated':False}
