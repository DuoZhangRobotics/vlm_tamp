"""Generic image/depth evidence that two selected planar patches move rigidly."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


def rotation_angle(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))))


def rigid_correspondences(source,target,tolerance_m=.008,reference_point=None):
    source=np.asarray(source,float);target=np.asarray(target,float)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3:raise ValueError('Matched finite three-dimensional points required')
    finite=np.isfinite(source).all(1)&np.isfinite(target).all(1);source=source[finite];target=target[finite]
    if len(source)<6:return {'supported':False,'reason':'insufficient_material_correspondences','correspondences':len(source)}
    def fit(a,b):
        ac=a.mean(0);bc=b.mean(0);u,_,vt=np.linalg.svd((a-ac).T@(b-bc))
        correction=np.eye(3);correction[-1,-1]=np.linalg.det(vt.T@u.T)
        rotation=vt.T@correction@u.T;translation=bc-rotation@ac
        return rotation,translation
    rng=np.random.default_rng(0);best=np.zeros(len(source),bool)
    for _ in range(256):
        chosen=rng.choice(len(source),3,replace=False)
        if np.linalg.norm(np.cross(source[chosen[1]]-source[chosen[0]],source[chosen[2]]-source[chosen[0]]))<1e-5:continue
        r,t=fit(source[chosen],target[chosen]);inliers=np.linalg.norm(source@r.T+t-target,axis=1)<tolerance_m
        if inliers.sum()>best.sum():best=inliers
    if best.sum()<6:return {'supported':False,'reason':'no_consistent_rigid_correspondences','correspondences':len(source),'inliers':int(best.sum())}
    r,t=fit(source[best],target[best]);errors=np.linalg.norm(source@r.T+t-target,axis=1)
    best=errors<tolerance_m
    if best.sum()<6:return {'supported':False,'reason':'rigid_refit_lost_support','correspondences':len(source),'inliers':int(best.sum())}
    spans=[]
    for points in [source[best],target[best]]:
        _,_,v=np.linalg.svd(points-points.mean(0),full_matrices=False)
        spans.append(np.ptp((points-points.mean(0))@v[:2].T,axis=0).tolist())
    supported=bool(best.sum()>=6 and best.mean()>=.6 and min(np.asarray(spans).ravel())>=.025)
    rms=float(np.sqrt(np.mean(errors[best]**2)))
    reference=source[best].mean(0) if reference_point is None else np.asarray(reference_point,float)
    orientation_spreads=[];displacements=[];chosen=np.flatnonzero(best)
    for _ in range(32):
        subset=rng.choice(chosen,max(6,int(len(chosen)*.8)),replace=False)
        rb,tb=fit(source[subset],target[subset])
        orientation_spreads.append(rotation_angle(rb@r.T))
        displacements.append(float(np.linalg.norm(rb@reference+tb-reference)))
    angle=rotation_angle(r)
    spread=float(np.quantile(orientation_spreads,.95))+float(np.degrees(np.arctan2(rms,max(min(np.asarray(spans).ravel()),1e-9))))
    displacement=float(np.linalg.norm(r@reference+t-reference))
    return {'supported':supported,'reason':'rigid_material_support' if supported else 'insufficient_correspondence_support',
        'correspondences':len(source),'inliers':int(best.sum()),'inlier_fraction':float(best.mean()),
        'rms_m':rms,'support_spans_m':spans,
        'rotation':r.tolist(),'translation_m':t.tolist(),'inlier_indices':np.flatnonzero(best).tolist(),
        'inlier_distance_m':tolerance_m,'rigid_rotation_deg':angle,'orientation_stability_radius_deg':spread,
        'rigid_rotation_interval_deg':[max(0,angle-spread),min(180,angle+spread)],
        'material_displacement_m':displacement,
        'material_displacement_interval_m':[max(0,float(np.quantile(displacements,.05))-rms),float(np.quantile(displacements,.95))+rms],
        'displacement_reference_world_m':reference.tolist()}


def selected_frame(record,measurement):
    p=Path(record['sensors'])
    if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sensor_sha256']:raise ValueError('Captured sensor data changed')
    with np.load(p) as data:
        cam=measurement['camera'];rgb=data[cam+'_rgb'];xyz=data[cam+'_world_points'];depth=data[cam+'_depth'];prior=data.get(cam+'_known_robot_depth_prior')
    h,w=depth.shape;polygon=Image.new('1',(w,h));ImageDraw.Draw(polygon).polygon([tuple(p) for p in measurement['polygon_xy']],fill=1)
    mask=np.asarray(polygon)&np.isfinite(xyz).all(-1)&np.isfinite(depth)&(depth>0)
    if prior is not None:mask&=~(np.isfinite(prior)&(prior>0)&(np.abs(depth-prior)<.012))
    mask&=np.abs((xyz-np.asarray(measurement['centroid_world_m']))@np.asarray(measurement['normal_world']))<measurement['inlier_distance_threshold_m']
    return rgb,xyz,mask


def associate_surfaces(first_record,first,second_record,second,*,pixels_first_xy=None,pixels_second_xy=None):
    rgb1,xyz1,mask1=selected_frame(first_record,first);rgb2,xyz2,mask2=selected_frame(second_record,second)
    if (pixels_first_xy is None)!=(pixels_second_xy is None):raise ValueError('Supply both matched pixel arrays or neither')
    if pixels_first_xy is None:
        sift=cv2.SIFT_create(nfeatures=1800)
        k1,d1=sift.detectAndCompute(cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY),mask1.astype(np.uint8)*255)
        k2,d2=sift.detectAndCompute(cv2.cvtColor(rgb2,cv2.COLOR_RGB2GRAY),mask2.astype(np.uint8)*255)
        if d1 is None or d2 is None or len(d1)<2 or len(d2)<2:
            return {'supported':False,'reason':'insufficient_image_features_on_selected_planes','matching_source':'SIFT'},np.empty((0,3)),np.empty((0,3))
        matcher=cv2.BFMatcher();forward=matcher.knnMatch(d1,d2,k=2);back=matcher.knnMatch(d2,d1,k=2)
        reverse={pair[0].queryIdx:pair[0].trainIdx for pair in back if len(pair)==2 and pair[0].distance<.75*pair[1].distance}
        matches=[pair[0] for pair in forward if len(pair)==2 and pair[0].distance<.75*pair[1].distance and reverse.get(pair[0].trainIdx)==pair[0].queryIdx]
        p1=np.asarray([k1[m.queryIdx].pt for m in matches]).reshape(-1,2)
        p2=np.asarray([k2[m.trainIdx].pt for m in matches]).reshape(-1,2);origin='mutual ratio-tested SIFT'
    else:
        p1=np.asarray(pixels_first_xy,float);p2=np.asarray(pixels_second_xy,float);origin='agent-supplied current-image correspondences'
        if p1.shape!=p2.shape or p1.ndim!=2 or p1.shape[1]!=2 or not 6<=len(p1)<=64 or not np.isfinite(p1).all() or not np.isfinite(p2).all():raise ValueError('Supply six to sixty-four finite matched pixel pairs')
    points1=[];points2=[];pixels1=[];pixels2=[];seen1=set();seen2=set()
    for a,b in zip(p1,p2):
        x1,y1=np.rint(a).astype(int);x2,y2=np.rint(b).astype(int)
        if not (0<=y1<mask1.shape[0] and 0<=x1<mask1.shape[1] and 0<=y2<mask2.shape[0] and 0<=x2<mask2.shape[1]):continue
        key1=(x1//3,y1//3);key2=(x2//3,y2//3)
        if key1 in seen1 or key2 in seen2 or not mask1[y1,x1] or not mask2[y2,x2]:continue
        seen1.add(key1);seen2.add(key2);pixels1.append([int(x1),int(y1)]);pixels2.append([int(x2),int(y2)]);points1.append(xyz1[y1,x1]);points2.append(xyz2[y2,x2])
    a=np.asarray(points1).reshape(-1,3);b=np.asarray(points2).reshape(-1,3)
    result=rigid_correspondences(a,b,reference_point=first['centroid_world_m']);result['matching_source']=origin
    result.update(pixels_first_xy=pixels1,pixels_second_xy=pixels2)
    if result.get('rotation'):
        n=np.asarray(result['rotation'])@np.asarray(first['normal_world']);m=np.asarray(second['normal_world'])
        angle=float(np.degrees(np.arccos(np.clip(abs(n@m),0,1))))
        result['transformed_normal_disagreement_deg']=angle
        plane_margin=sum(np.degrees(np.arctan2(x['rms_m'],max(min(x['surface_spans_m']),1e-9))) for x in [first,second])
        tolerance=max(3,result.get('orientation_stability_radius_deg',0)+plane_margin)
        result['normal_agreement_tolerance_deg']=float(tolerance)
        if angle>tolerance:result.update(supported=False,reason='rigid_matches_disagree_with_selected_planes')
    result['limitations']=['Geometrically consistent feature matches are association evidence, not semantic object recognition.',
        'Repeated appearance, occlusion and weak texture can produce ambiguity or unavailable evidence.']
    return result,a,b
