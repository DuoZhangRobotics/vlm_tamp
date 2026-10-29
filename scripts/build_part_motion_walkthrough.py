"""Render real masks, query features, tracks and fitted-motion support.

The inlier trace below reproduces the existing fitter for display, then checks
its counts/transform against the production result. It never supplies a policy
or an estimate to Luna. Original artifacts are read-only inputs.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.part_motion import fit_upright,estimate_and_verify
from vlm_tamp.rigid_hinge_tracking import fit_se2


def encoded(image,kind='JPEG'):
    image=image.copy();image.thumbnail((288,288),Image.Resampling.NEAREST if kind=='PNG' else Image.Resampling.LANCZOS)
    if kind=='PNG':image=image.quantize(colors=16,method=Image.Quantize.FASTOCTREE,dither=Image.Dither.NONE)
    stream=io.BytesIO()
    image.save(stream,format=kind,**({'quality':45,'optimize':True} if kind=='JPEG' else {'optimize':True}))
    return 'data:image/'+('jpeg' if kind=='JPEG' else 'png')+';base64,'+base64.b64encode(stream.getvalue()).decode()


def finite_list(value,decimals=3):
    value=np.asarray(value)
    if value.ndim:return [finite_list(v,decimals) for v in value]
    return round(float(value),decimals) if np.isfinite(value) else None


def mask_runs(mask):
    rows=[]
    for y,row in enumerate(mask):
        edges=np.flatnonzero(np.diff(np.r_[False,row,False].astype(np.int8)))
        rows.extend([int(y),int(a),int(b-a)] for a,b in zip(edges[::2],edges[1::2]))
    return rows


def trace_membership(world,valid,a,b,fit):
    """Exact membership of the fitter's final selected set, in global query IDs."""
    good=valid[a]&valid[b]&np.isfinite(world[a]).all(1)&np.isfinite(world[b]).all(1)
    ids=np.flatnonzero(good);first=world[a,ids];second=world[b,ids]
    movement=np.linalg.norm(second[:,:2]-first[:,:2],axis=1)
    usable=(movement>.0015)&(movement<.15)&(np.abs(second[:,2]-first[:,2])<.004)
    classes=np.zeros(world.shape[1],np.uint8);classes[ids]=1
    ids=ids[usable];first=first[usable];second=second[usable];classes[ids]=2
    assert len(ids)==fit['correspondences']
    if len(ids)<6:return classes
    rng=np.random.default_rng(42);best=None
    for _ in range(350):
        chosen=rng.choice(len(first),2,replace=False)
        if np.linalg.norm(first[chosen[0],:2]-first[chosen[1],:2])<.04:continue
        rotation,translation=fit_se2(first[chosen,:2],second[chosen,:2])
        residual=np.linalg.norm(first[:,:2]@rotation.T+translation-second[:,:2],axis=1)
        selected=residual<.0025
        if best is None or selected.sum()>best.sum():best=selected
    if best is None or best.sum()<6:return classes
    rotation,translation=fit_se2(first[best,:2],second[best,:2])
    best=np.linalg.norm(first[:,:2]@rotation.T+translation-second[:,:2],axis=1)<.0025
    rotation,translation=fit_se2(first[best,:2],second[best,:2])
    assert int(best.sum())==fit['inliers']
    np.testing.assert_allclose(rotation,fit['rotation_xy'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(translation,fit['translation_xy_m'],rtol=0,atol=1e-12)
    classes[ids[best]]=3
    return classes


def project(points,calibration):
    points=np.asarray(points,float).reshape(-1,3)
    transform=np.asarray(calibration['camera_to_world']);k=np.asarray(calibration['intrinsic'])
    camera=(points-transform[:3,3])@transform[:3,:3]
    screen=camera@k.T
    good=screen[:,2]>0
    result=np.full((len(points),2),np.nan)
    result[good]=screen[good,:2]/screen[good,2,None]
    return result


def build_case(name,work,track_path,mask_path,association_path,cameras,fit_pairs,box,images,asset_lookup,depth_scales,subset=None):
    manifest=json.loads((work/'observations/manifest.json').read_text())['observations']
    if subset is not None:manifest=[manifest[i] for i in subset]
    with np.load(track_path,allow_pickle=False) as src:tracks={k:src[k] for k in src.files}
    world,valid=tracks['world_points'],tracks['valid']
    with np.load(mask_path,allow_pickle=False) as src:chosen=src['usable_mask']
    with np.load(association_path,allow_pickle=False) as src:masks={k:src[k] for k in cameras}
    masks[cameras[0]]=chosen
    calibrations=[];frame_assets=[];depths={k:[] for k in cameras};rgb={k:[] for k in cameras}
    evidence_hashes={}
    for frame in manifest:
        cp=work/frame['camera_calibration'];sp=work/frame['sensor_arrays']
        calibrations.append(json.loads(cp.read_text()))
        for p in [cp,sp]:evidence_hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(sp,allow_pickle=False) as arrays:
            for camera in cameras:
                rgb[camera].append(arrays[camera+'_rgb'])
                depths[camera].append(arrays[camera+'_depth'].squeeze())
    depth_ranges={}
    for camera in cameras:
        sample=np.concatenate([d[::8,::8].ravel() for d in depths[camera]])
        sample=sample[np.isfinite(sample)&(sample>0)]
        if camera not in depth_scales:depth_scales[camera]=np.quantile(sample,[0.01,0.99]).tolist()
        depth_ranges[camera]=depth_scales[camera]
    for i,frame in enumerate(manifest):
        row={}
        for camera in cameras:
            array=rgb[camera][i];depth=depths[camera][i];lo,hi=depth_ranges[camera]
            # One shared metric scale per camera; invalid depth is transparent.
            gray=np.rint(255*(1-np.clip((depth-lo)/(hi-lo),0,1))).astype(np.uint8)
            rgba=np.stack([gray,gray,gray,np.where(np.isfinite(depth)&(depth>0),255,0).astype(np.uint8)],-1)
            fingerprint=hashlib.sha256(array.tobytes()+rgba.tobytes()).hexdigest()
            if fingerprint not in asset_lookup:
                asset_lookup[fingerprint]=len(images)
                images.append({'rgb':encoded(Image.fromarray(array)), 'depth':encoded(Image.fromarray(rgba),'PNG')})
            row[camera]=asset_lookup[fingerprint]
        frame_assets.append(row)
    fits=[fit_upright(world,valid,*pair) for pair in fit_pairs]
    membership=[trace_membership(world,valid,*pair,fit) for pair,fit in zip(fit_pairs,fits)]
    checks=[]
    for i in range(len(fit_pairs)-1):
        a,b=fit_pairs[i];c,d=fit_pairs[i+1]
        result=estimate_and_verify(world,valid,[a,b],[c,d]);verification=result.get('verification') or {}
        prediction=np.full_like(world[d],np.nan);supported=np.zeros(world.shape[1],bool)
        if result['estimate'].get('accepted') and fits[i+1].get('pose_accepted') and abs(fits[i+1].get('measured_rotation_deg') or 0)>=1.5:
            rot=np.asarray(fits[i+1]['rotation_xy']);translation=np.asarray(fits[i+1]['translation_xy_m'])
            first,second=world[c],world[d]
            movement=np.linalg.norm(second[:,:2]-first[:,:2],axis=1)
            rigid_error=np.linalg.norm(first[:,:2]@rot.T+translation-second[:,:2],axis=1)
            supported=valid[c]&valid[d]&(movement>.0015)&(movement<.15)&(np.abs(second[:,2]-first[:,2])<.004)&(rigid_error<.0025)
            origin=np.asarray(result['estimate']['joint_origin_world_m'])[:2]
            prediction[supported]=first[supported]
            prediction[supported,:2]=(first[supported,:2]-origin)@rot.T+origin
            assert supported.sum()==verification['moving_inliers']
        checks.append({'status':result['status'],'reason':verification.get('reason'),
            'rms_mm':1000*verification['axis_prediction_rms_m'] if 'axis_prediction_rms_m' in verification else None,
            'p90_mm':1000*verification['axis_prediction_p90_m'] if 'axis_prediction_p90_m' in verification else None,
            'rotation_deg':verification.get('measured_rotation_deg'),'support':supported,'predictions':prediction,'end':d})
    camera_data={};offset=0
    z=world[:,:,2][valid];zlo,zhi=np.quantile(z,[.01,.99])
    for camera in cameras:
        pixels=tracks[camera+'_pixels'];count=pixels.shape[1];sl=slice(offset,offset+count)
        axes=[];verifications=[]
        for i,fit in enumerate(fits):
            origin=fit.get('joint_origin_world_m')
            axes.append(finite_list(project([[origin[0],origin[1],zlo],[origin[0],origin[1],zhi]],calibrations[fit_pairs[i][1]][camera])) if origin else None)
        for i,check in enumerate(checks):
            origin=fits[i].get('joint_origin_world_m')
            axis=finite_list(project([[origin[0],origin[1],zlo],[origin[0],origin[1],zhi]],calibrations[check['end']][camera])) if origin else None
            verifications.append({'support':check['support'][sl].astype(int).tolist(),
                'predicted_pixels':finite_list(project(check['predictions'][sl],calibrations[check['end']][camera])), 'axis':axis})
        all_xy=np.concatenate([p[v] for p,v in zip(pixels,tracks[camera+'_depth_valid'])])
        bounds=[float(all_xy[:,0].min()-28),float(all_xy[:,1].min()-28),float(all_xy[:,0].max()+28),float(all_xy[:,1].max()+28)]
        bounds=[max(0,bounds[0]),max(0,bounds[1]),min(512,bounds[2]),min(512,bounds[3])]
        camera_data[camera]={'pixels':finite_list(pixels,2),'valid':tracks[camera+'_depth_valid'].astype(int).tolist(),
            'tracker_visible':tracks[camera+'_tracker_visible'].astype(int).tolist(),
            'queries':finite_list(tracks[camera+'_queries'],2),'mask':mask_runs(masks[camera]),
            'query_count':count,'crop':bounds,'classes':[v[sl].tolist() for v in membership],
            'axes':axes,'verifications':verifications,'depth_range_m':finite_list(depth_ranges[camera])}
        offset+=count
    assert offset==world.shape[1]
    return {'name':name,'frame_assets':frame_assets,'steps':[f['control_steps'] for f in manifest],
        'reference':len(manifest)-1,'cameras':camera_data,'box':box,'fit_pairs':fit_pairs,
        'fits':[{'eligible':f['correspondences'],'inliers':f.get('inliers',0),'visible':f['finite_visible_pairs'],
            'rms_mm':1000*f['rms_m'] if 'rms_m' in f else None,'rotation_deg':f.get('measured_rotation_deg'),
            'accepted':f['accepted'],'reason':f.get('reason')} for f in fits],
        'checks':[{k:v for k,v in c.items() if k not in ['support','predictions']} for c in checks],
        'input_hashes':evidence_hashes}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    original=ROOT/'results/luna-part-motion-tools-20261027/radius-prior-followup/trials/clip2/agent-workspace'
    mask=original/'robot_data/perception/001-segment/part001-0.npz'
    association=original/'robot_data/perception/002-association/masks.npz'
    cameras=['agentview','frontview','arm_side'];images=[];lookup={};depth_scales={}
    definitions=[('Original: 5 captures',original,original/'robot_data/perception/003-tracks/tracks.npz',[[0,1],[1,2],[2,3],[3,4]],None)]
    comparison=out/'density-comparison/summary.json'
    if comparison.exists():
        summary=json.loads(comparison.read_text())
        dense=out/'dense-capture';manifest=json.loads((dense/'observations/manifest.json').read_text())['observations']
        steps=[f['control_steps'] for f in manifest];anchors=[steps.index(s) for s in [221,242,259,277,278]]
        definitions.append(('Replay: 5 captures',dense,out/'density-comparison/sparse/tracks.npz',[[0,1],[1,2],[2,3],[3,4]],anchors))
        definitions.append(('Replay: 16 captures',dense,out/'density-comparison/dense/tracks.npz',list(zip(anchors[:-1],anchors[1:])),None))
    cases=[];audit=[]
    for name,work,track,pairs,subset in definitions:
        case=build_case(name,work,track,mask,association,cameras,pairs,[105,270,320,435],images,lookup,depth_scales,subset)
        audit.append({'name':name,'input_hashes':case.pop('input_hashes'),'fit_trace_counts_verified':True,
            'fit_pairs':pairs,'fitted_membership_source':'Exact RANSAC reconstruction checked against production counts and transforms'})
        cases.append(case)
    data={'cases':cases,'images':images,'cameras':cameras,'source_size':512,
        'scope':'Recorded public RGB-D. Original mask and query identities preserved. Upright axis is a supplied assumption.'}
    template=(ROOT/'templates/part_motion_walkthrough.html').read_text()
    fragment=template.replace('__PIPELINE_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    path=out/'perception-walkthrough.html';path.write_text(fragment)
    assert path.stat().st_size<1_000_000, 'Inline view exceeds its size budget'
    (out/'visualization-audit.json').write_text(json.dumps({'cases':audit,'preview_image_width':288,
        'depth_preview':'16 grayscale bins for display only; metric values and fit calculations use full precision.',
        'original_coordinate_width':512,'bytes':path.stat().st_size,'case_count':len(cases),
        'world_geometry_supplied_to_actor':False,'input_files_modified':False},indent=2)+'\n')
    print(json.dumps({'fragment':str(path),'bytes':path.stat().st_size,'cases':len(cases),'unique_images':len(images)}))

if __name__=='__main__':main()
