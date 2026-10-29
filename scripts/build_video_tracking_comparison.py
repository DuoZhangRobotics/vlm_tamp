"""Visualize dense RGB tracking without inventing intermediate depth."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_part_motion_walkthrough import encoded,finite_list,mask_runs,project,trace_membership

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'results/luna-part-motion-tools-20261027/radius-prior-followup/trials/clip2/agent-workspace'

def build(run):
    protocol=json.loads((run/'protocol.json').read_text())
    grade=json.loads((run/'grading-private.json').read_text())
    manifest=json.loads((PUBLIC/'observations/manifest.json').read_text())['observations']
    depth_steps=protocol['depth_control_steps']
    anchors=[];calibrations=[];source_hashes={}
    for row in manifest:
        path=PUBLIC/row['sensor_arrays'];cal=PUBLIC/row['camera_calibration']
        for p in [path,cal]:source_hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as z:anchors.append({k:z['agentview_'+k] for k in ['rgb','depth','world_points','known_robot_depth_prior']})
        calibrations.append(json.loads(cal.read_text())['agentview'])
    samples=np.concatenate([a['depth'][::8,::8].ravel() for a in anchors]);samples=samples[np.isfinite(samples)&(samples>0)]
    lo,hi=np.quantile(samples,[.01,.99]);depth_images={}
    for step,view in zip(depth_steps,anchors):
        depth=view['depth'].squeeze();valid=np.isfinite(depth)&(depth>0)
        gray=np.rint(255*(1-np.clip((np.where(valid,depth,hi)-lo)/(hi-lo),0,1))).astype(np.uint8)
        rgba=np.stack([gray,gray,gray,np.where(valid,255,0).astype(np.uint8)],-1)
        depth_images[step]=encoded(Image.fromarray(rgba),'PNG')
    with np.load(PUBLIC/'robot_data/perception/001-segment/part001-0.npz') as z:mask=z['usable_mask']
    images=[];assets={};cases=[];audit=[]
    for condition,name in [('video5','Video: 5 RGB + 5 depth'),('video58','Video: 58 RGB + 5 depth'),('png5','PNG control: 5 RGB + 5 depth')]:
        result=json.loads((run/condition/'result.json').read_text())
        with np.load(run/condition/'tracks.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        world,valid=arrays['world_points'],arrays['valid'];pixels=arrays['pixels'];visible=arrays['tracker_visible']
        indices=arrays['anchor_frame_indices'].astype(int).tolist();steps=arrays['frame_control_steps'].astype(int).tolist()
        assert arrays['anchor_control_steps'].tolist()==depth_steps
        assert [steps[i] for i in indices]==depth_steps
        valid_at_frame=np.zeros(visible.shape,bool);valid_at_frame[indices]=valid
        depth_available=[s in depth_steps for s in steps]
        frame_assets=[]
        for frame,step in enumerate(steps):
            rgb=anchors[frame]['rgb'] if condition=='png5' else np.asarray(Image.open(run/'decoded'/f'step{step:03d}.png').convert('RGB'))
            fingerprint=(hashlib.sha256(rgb.tobytes()).hexdigest(),step if step in depth_steps else None)
            if fingerprint not in assets:
                assets[fingerprint]=len(images)
                images.append({'rgb':encoded(Image.fromarray(rgb)), 'depth':depth_images.get(step)})
            frame_assets.append({'agentview':assets[fingerprint]})
        fits=result['fits'];classes=[trace_membership(world,valid,i,i+1,f) for i,f in enumerate(fits)]
        z_values=world[:,:,2][valid];zlo,zhi=np.quantile(z_values,[.01,.99])
        def axis_for(fit,anchor):
            origin=fit.get('joint_origin_world_m')
            if origin is None:return None
            return finite_list(project([[origin[0],origin[1],zlo],[origin[0],origin[1],zhi]],calibrations[anchor]))
        axes=[axis_for(fit,i+1) for i,fit in enumerate(fits)]
        checks=[];verification_draw=[]
        for i,r in enumerate(result['checks']):
            verification=r.get('verification') or {};supported=np.zeros(world.shape[1],bool)
            predicted=np.full_like(world[i+2],np.nan)
            if 'moving_inliers' in verification:
                measured=fits[i+1];rot=np.asarray(measured['rotation_xy']);translation=np.asarray(measured['translation_xy_m'])
                first,second=world[i+1],world[i+2]
                movement=np.linalg.norm(second[:,:2]-first[:,:2],axis=1)
                error=np.linalg.norm(first[:,:2]@rot.T+translation-second[:,:2],axis=1)
                supported=valid[i+1]&valid[i+2]&(movement>.0015)&(movement<.15)&(np.abs(second[:,2]-first[:,2])<.004)&(error<.0025)
                assert supported.sum()==verification['moving_inliers']
                origin=np.asarray(r['estimate']['joint_origin_world_m'])[:2]
                predicted[supported]=first[supported]
                predicted[supported,:2]=(first[supported,:2]-origin)@rot.T+origin
            checks.append({'status':r['status'],'reason':verification.get('reason'),
                'rms_mm':1000*verification['axis_prediction_rms_m'] if 'axis_prediction_rms_m' in verification else None,
                'p90_mm':1000*verification['axis_prediction_p90_m'] if 'axis_prediction_p90_m' in verification else None,
                'rotation_deg':verification.get('measured_rotation_deg'),'end':indices[i+2]})
            verification_draw.append({'support':supported.astype(int).tolist(),
                'predicted_pixels':finite_list(project(predicted,calibrations[i+2])), 'axis':axis_for(fits[i],i+2)})
        supported_xy=np.concatenate([p[v] for p,v in zip(pixels[indices],valid)])
        bounds=[max(0,float(supported_xy[:,0].min()-28)),max(0,float(supported_xy[:,1].min()-28)),
            min(512,float(supported_xy[:,0].max()+28)),min(512,float(supported_xy[:,1].max()+28))]
        public_fits=[]
        for fit,graded in zip(fits,grade['conditions'][condition]['fits']):
            public_fits.append({'eligible':fit['correspondences'],'inliers':fit.get('inliers',0),'visible':fit['finite_visible_pairs'],
                'rms_mm':1000*fit['rms_m'] if 'rms_m' in fit else None,'rotation_deg':fit.get('measured_rotation_deg'),
                'accepted':fit['accepted'],'reason':fit.get('reason'),
                'axis_error_mm':graded['metrics']['axis_position_error_mm'] if graded['metrics'] else None})
        camera={'pixels':finite_list(pixels,2),'valid':valid_at_frame.astype(int).tolist(),
            'tracker_visible':visible.astype(int).tolist(),'queries':finite_list(arrays['queries'],2),
            'query_count':len(arrays['queries']),'mask':mask_runs(mask),'crop':bounds,
            'classes':[v.tolist() for v in classes],'axes':axes,'verifications':verification_draw,
            'depth_range_m':finite_list([lo,hi]),'depth_available':depth_available}
        cases.append({'name':name,'frame_assets':frame_assets,'steps':steps,'reference':len(steps)-1,
            'cameras':{'agentview':camera},'box':[105,270,320,435],
            'fit_pairs':list(zip(indices[:-1],indices[1:])),'fits':public_fits,'checks':checks,
            'sparse_depth':True,'depth_steps':depth_steps,'rgb_source':'png' if condition=='png5' else 'mp4',
            'origin_kind':'numerical_comparison','selection_scope':'Original Luna mask and queries held fixed; segmentation was not rerun.',
            'tracking_wall_seconds':result['timing']['median_tracking_wall_seconds']})
        audit.append({'condition':condition,'rgb_frames':len(steps),'depth_frames':sum(depth_available),
            'depth_frame_indices':indices,'empty_depth_frames':len(steps)-sum(depth_available),
            'fit_trace_memberships_verified':True,'query_coordinates_sha256':hashlib.sha256(arrays['queries'].tobytes()).hexdigest()})
    data={'title':'Dense RGB tracking with five measured depth captures','cases':cases,'images':images,
        'cameras':['agentview'],'source_size':512,'default_state':{'stage':'track','frame':4,'interval':1},
        'scope':'Controlled single-camera comparison. Native axis errors are grader-only, after prediction.'}
    template=(ROOT/'templates/part_motion_walkthrough.html').read_text()
    text=template.replace('__PIPELINE_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    path=run/'video-tracking-comparison.html';path.write_text(text)
    assert path.stat().st_size<1_000_000,'Preview exceeds inline size budget'
    (run/'visualization-audit.json').write_text(json.dumps({'conditions':audit,'shared_depth_input_hashes':source_hashes,
        'bytes':path.stat().st_size,'previews_only':'RGB images compressed to 288px; depth preview uses 16 grayscale bins.',
        'missing_depth_policy':'No depth preview or 3D point is generated for RGB-only timestamps; hollow markers show 2D tracks.',
        'original_artifacts_modified':False},indent=2)+'\n')
    print(json.dumps({'fragment':str(path),'bytes':path.stat().st_size,'unique_images':len(images),'conditions':len(cases)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();build(args.run.resolve())
