"""Task-independent RGB-D measurements and explicit, observation-based completion.

No simulator state, task names, motion planning, or action execution are used.
The acting agent chooses the goal, thresholds, and material surface selections.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def plane_fit(points):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<40 or not np.isfinite(points).all():
        raise ValueError('Need at least forty finite measured surface points')
    rng=np.random.default_rng(0)
    sample=points[rng.choice(len(points),min(len(points),2500),replace=False)]
    best=None;count=0;threshold=.003
    for _ in range(180):
        a,b,c=sample[rng.choice(len(sample),3,replace=False)]
        normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
        if length<1e-10:continue
        normal/=length;mask=np.abs((sample-a)@normal)<threshold
        if mask.sum()>count:best=mask;count=mask.sum()
    if best is None or count<40:raise ValueError('No supported planar surface in selection')
    inliers=sample[best]
    center=inliers.mean(0);_,singular,vt=np.linalg.svd(inliers-center,full_matrices=False)
    normal=vt[-1]
    if normal[np.argmax(np.abs(normal))]<0:normal=-normal
    residual=np.abs((points-center)@normal);mask=residual<threshold
    inliers=points[mask]
    # Two-dimensional support prevents a line or narrow edge from masquerading
    # as an accurately measured plane. This is a sensor quality check only.
    widths=np.ptp((inliers-center)@vt[:2].T,axis=0)
    rms=float(np.sqrt(np.mean(residual[mask]**2)))
    supported=bool(mask.mean()>=.55 and min(widths)>=.025 and rms<=threshold)
    normals=[]
    for _ in range(48):
        subset=inliers[rng.choice(len(inliers),min(len(inliers),300),replace=True)]
        _,_,basis=np.linalg.svd(subset-subset.mean(0),full_matrices=False)
        n=basis[-1];normals.append(n if n@normal>=0 else -n)
    result={'supported':supported,'normal_world':normal.tolist(),'centroid_world_m':center.tolist(),
        'selected_points':len(points),'inlier_points':len(inliers),'inlier_fraction':float(mask.mean()),
        'rms_m':rms,'surface_spans_m':widths.tolist(),'inlier_distance_threshold_m':threshold,
        'bootstrap_normals':np.asarray(normals).tolist(),
        'limitations':['A planar background can also fit; inspect the overlay and identify the intended material surface.',
                      'Resampling spread measures fit stability, not calibrated semantic or sensor uncertainty.']}
    return result,mask


def compare_planes(first,second):
    a=np.asarray(first['normal_world']);b=np.asarray(second['normal_world'])
    angle=float(np.degrees(np.arccos(np.clip(abs(a@b),0,1))))
    aa=np.asarray(first['bootstrap_normals']);bb=np.asarray(second['bootstrap_normals'])
    samples=np.degrees(np.arccos(np.clip(np.abs(aa@bb.T),0,1)))
    margin=sum(np.degrees(np.arctan2(x['rms_m'],max(min(x['surface_spans_m']),1e-9))) for x in [first,second])
    interval=[max(0,float(np.quantile(samples,.05))-margin),min(90,float(np.quantile(samples,.95))+margin)]
    distance=float(np.linalg.norm(np.asarray(first['centroid_world_m'])-np.asarray(second['centroid_world_m'])))
    return {'supported':bool(first['supported'] and second['supported']),
        'values':{'plane_angle_deg':angle,'centroid_distance_m':distance},
        'intervals':{'plane_angle_deg':interval},
        'limitations':['Plane angle is unsigned and between zero and ninety degrees; it is not total articulation rotation.',
            'Both selections must represent the same material surface; the tool does not recognize objects.',
            'Centroid distance depends on selected regions and is not an object displacement certificate.',
            'Angle bounds combine resampling spread and residual/support margin; they are not calibrated confidence bounds.']}


class CompletionEvidence:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.contract=None;self.measurements={};self.checks={};self.surface_records={};self.association_edges=[]

    def save(self,name,value):
        (self.folder/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
        return copy.deepcopy(value)

    def set_contract(self,goal,criteria,control_steps):
        if self.contract is not None:raise ValueError('Completion criteria are already fixed for this episode')
        if control_steps>60:raise ValueError('Declare the completion goal before the first physical action')
        if not isinstance(goal,str) or not goal.strip() or not isinstance(criteria,list) or not criteria:
            raise ValueError('State the intended goal and at least one measurable criterion')
        ids=set()
        for item in criteria:
            if not isinstance(item,dict) or not item.get('id') or item['id'] in ids:raise ValueError('Unique criterion ids required')
            ids.add(item['id'])
            if item.get('quantity') not in {'plane_angle_deg','rigid_rotation_deg','material_displacement_m'}:raise ValueError('Unsupported observed quantity')
            if not item.get('rationale'):raise ValueError('Explain how the criterion follows from the instruction')
            if not any(k in item for k in ('minimum','maximum')):raise ValueError('Specify a quantitative goal range')
            for key in ['minimum','maximum']:
                if key in item and (type(item[key]) not in (int,float) or not np.isfinite(item[key])):raise ValueError('Finite numeric goal bounds required')
            if item.get('minimum',-np.inf)>item.get('maximum',np.inf):raise ValueError('Inverted goal range')
        self.contract={'goal':goal,'criteria':copy.deepcopy(criteria),'declared_at_control_step':control_steps,
                       'target_source':'Acting agent interpretation; no task-specific target or native threshold supplied.'}
        return self.save('contract',self.contract)

    def surface(self,record,observation_id,camera,polygon_xy):
        path=Path(record['sensors'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sensor_sha256']:
            raise ValueError('Sensor evidence changed since capture')
        with np.load(path) as arrays:
            rgb=arrays[camera+'_rgb'];xyz=arrays[camera+'_world_points'];depth=arrays[camera+'_depth']
            prior=arrays.get(camera+'_known_robot_depth_prior')
        poly=np.asarray(polygon_xy,float);h,w=depth.shape
        if poly.ndim!=2 or poly.shape[1]!=2 or not 3<=len(poly)<=32 or not np.isfinite(poly).all():raise ValueError('Use three to thirty-two polygon vertices')
        if np.any(poly<0) or np.any(poly[:,0]>=w) or np.any(poly[:,1]>=h):raise ValueError('Polygon outside original image')
        selection=Image.new('1',(w,h));ImageDraw.Draw(selection).polygon([tuple(x) for x in poly],fill=1)
        valid=np.asarray(selection)&np.isfinite(xyz).all(-1)&np.isfinite(depth)&(depth>0)
        if prior is not None:valid&=~(np.isfinite(prior)&(prior>0)&(np.abs(depth-prior)<.0003))
        ys,xs=np.where(valid);fit,mask=plane_fit(xyz[valid])
        token=f'm{len(self.measurements)+1:03d}'
        overlay=Image.fromarray(rgb).convert('RGB');draw=ImageDraw.Draw(overlay)
        for j in np.linspace(0,len(xs)-1,min(1000,len(xs))).astype(int):
            x,y=int(xs[j]),int(ys[j]);draw.ellipse((x-1,y-1,x+1,y+1),fill=(20,220,100) if mask[j] else (250,70,70))
        draw.line([tuple(x) for x in np.vstack([poly,poly[:1]])],fill=(255,210,20),width=2)
        overlay.save(self.folder/(token+'.png'))
        result={**fit,'measurement_id':token,'kind':'surface','observation_ids':[observation_id],
                'control_steps':record['control_steps'],'camera':camera,'polygon_xy':poly.tolist(),
                'overlay':str(self.folder/(token+'.png'))}
        if record.get('calibration'):
            calibration_path=Path(record['calibration']);raw=calibration_path.read_bytes()
            expected=record.get('calibration_sha256')
            if expected is not None and hashlib.sha256(raw).hexdigest()!=expected:
                raise ValueError('Camera calibration changed since capture')
            calibration=json.loads(raw)[camera]
            normal=np.asarray(fit['normal_world']).copy()
            origin=np.asarray(calibration['camera_to_world'])[:3,3]
            if normal@(origin-np.asarray(fit['centroid_world_m']))<0:normal=-normal
            result['normal_toward_camera_world']=normal.tolist()
        self.measurements[token]=result
        self.surface_records[token]=copy.deepcopy(record)
        return self.save(token,result)

    def geometry_relation(self,first_id,second_id,control_steps):
        """Compare current surfaces, including different parts, without claiming motion."""
        first=self.measurements[first_id];second=self.measurements[second_id]
        if first['kind']!='surface' or second['kind']!='surface':raise ValueError('Choose two surface measurements')
        if first['control_steps']!=control_steps or second['control_steps']!=control_steps:
            raise ValueError('Select both surfaces from the current physical state')
        comparison=compare_planes(first,second)
        normal=np.asarray(second.get('normal_toward_camera_world',second['normal_world']))
        offset=float((np.asarray(first['centroid_world_m'])-second['centroid_world_m'])@normal)
        values={**comparison['values'],'signed_centroid_to_reference_plane_m':offset,
            'absolute_centroid_to_reference_plane_m':abs(offset)}
        if 'normal_toward_camera_world' in first and 'normal_toward_camera_world' in second:
            dot=float(np.clip(np.asarray(first['normal_toward_camera_world'])@normal,-1,1))
            values['visible_normal_angle_deg']=float(np.degrees(np.arccos(dot)))
        token=f'm{len(self.measurements)+1:03d}'
        result={'measurement_id':token,'kind':'surface_geometry','sources':[first_id,second_id],
            'control_steps':control_steps,'observation_ids':first['observation_ids']+second['observation_ids'],
            'same_capture':first['observation_ids']==second['observation_ids'],
            'supported':comparison['supported'],'values':values,'intervals':comparison['intervals'],
            'semantic_identity_verified':False,'object_motion_claimed':False,
            'limitations':['Selections may be different parts; their goal relationship is the caller interpretation.',
                'Unsigned plane angle is 0..90 degrees. Visible-normal angle depends on which faces/cameras were selected.',
                'Centroid distances depend on the selected patches. These measurements do not replace native task success.']}
        self.measurements[token]=result
        return self.save(token,result)

    def association_path(self,first_id,second_id):
        from .surface_association import rotation_angle
        queue=[(first_id,np.eye(3),np.zeros(3),0.,[first_id])];visited={first_id}
        for node,rotation,translation,spread,path in queue:
            for a,b,edge in self.association_edges:
                if node==a:target=b;r=np.asarray(edge['rotation']);t=np.asarray(edge['translation_m'])
                elif node==b:target=a;r=np.asarray(edge['rotation']).T;t=-r@np.asarray(edge['translation_m'])
                else:continue
                if target in visited:continue
                visited.add(target);rr=r@rotation;tt=r@translation+t
                ss=spread+edge['orientation_stability_radius_deg'];pp=path+[target]
                if target==second_id:
                    angle=rotation_angle(rr)
                    return {'supported':True,'reason':'chain_of_verified_material_matches','measurement_path':pp,
                        'rotation':rr.tolist(),'translation_m':tt.tolist(),'orientation_stability_radius_deg':ss,
                        'rigid_rotation_deg':angle,'rigid_rotation_interval_deg':[max(0,angle-ss),min(180,angle+ss)],
                        'limitations':['Identity is propagated through earlier supported rigid correspondences; drift and mistaken associations remain possible.',
                            'The rotation margin sums fitted per-edge stability estimates; it is not calibrated uncertainty.']}
                queue.append((target,rr,tt,ss,pp))
        return None

    def compare(self,first_id,second_id,pixels_first_xy=None,pixels_second_xy=None):
        from .surface_association import associate_surfaces
        first=self.measurements[first_id];second=self.measurements[second_id]
        if first['kind']!='surface' or second['kind']!='surface':raise ValueError('Compare two surface measurements')
        result=compare_planes(first,second);token=f'm{len(self.measurements)+1:03d}'
        association,a,b=associate_surfaces(self.surface_records[first_id],first,self.surface_records[second_id],second,
            pixels_first_xy=pixels_first_xy,pixels_second_xy=pixels_second_xy)
        np.savez_compressed(self.folder/(token+'-correspondences.npz'),source_world=a,target_world=b)
        if association['supported']:
            self.association_edges.append((first_id,second_id,association))
        else:
            chain=self.association_path(first_id,second_id)
            if chain is not None:
                normal=np.asarray(chain['rotation'])@np.asarray(first['normal_world'])
                error=float(np.degrees(np.arccos(np.clip(abs(normal@np.asarray(second['normal_world'])),0,1))))
                margin=sum(np.degrees(np.arctan2(x['rms_m'],max(min(x['surface_spans_m']),1e-9))) for x in [first,second])
                chain['endpoint_normal_disagreement_deg']=error
                if error>max(3,chain['orientation_stability_radius_deg']+margin):
                    chain.update(supported=False,reason='material_chain_inconsistent_with_endpoint_planes')
                chain['direct_match_attempt']=association;association=chain
        plane_supported=bool(result['supported'] and association['supported'])
        result['association']=association
        result['supported']=bool(association['supported'])
        result['quantity_supported']={'plane_angle_deg':plane_supported,'rigid_rotation_deg':False,'material_displacement_m':False}
        if 'rigid_rotation_deg' in association:
            result['values']['rigid_rotation_deg']=association['rigid_rotation_deg']
            result['intervals']['rigid_rotation_deg']=association['rigid_rotation_interval_deg']
            result['quantity_supported']['rigid_rotation_deg']=bool(association['supported'])
        if 'material_displacement_m' in association:
            result['values']['material_displacement_m']=association['material_displacement_m']
            result['intervals']['material_displacement_m']=association['material_displacement_interval_m']
            result['quantity_supported']['material_displacement_m']=bool(association['supported'])
        result.update(measurement_id=token,kind='comparison',sources=[first_id,second_id],
                      observation_ids=first['observation_ids']+second['observation_ids'],
                      control_steps=max(first['control_steps'],second['control_steps']))
        self.measurements[token]=result
        return self.save(token,result)

    def check(self,evidence,control_steps,interpretation):
        if self.contract is None:raise ValueError('Declare a completion contract first')
        if not isinstance(interpretation,str) or not interpretation.strip():raise ValueError('Explain surface identity, visibility, and remaining uncertainty')
        rows=[]
        for criterion in self.contract['criteria']:
            mid=evidence.get(criterion['id']);measurement=self.measurements.get(mid,{})
            quantity=criterion['quantity'];value=measurement.get('values',{}).get(quantity)
            bounds=measurement.get('intervals',{}).get(quantity)
            fresh=measurement.get('control_steps')==control_steps
            # A centroid displacement lacks a correspondence/error certificate;
            # retain it as a diagnostic rather than invent confidence bounds.
            supported=bool(measurement.get('quantity_supported',{}).get(quantity,measurement.get('supported')) and fresh and bounds is not None)
            met=bool(supported and bounds[0]>=criterion.get('minimum',-np.inf) and bounds[1]<=criterion.get('maximum',np.inf))
            rows.append({'criterion':criterion['id'],'measurement_id':mid,'value':value,'bounds':bounds,
                         'fresh':fresh,'supported':supported,'met':met})
        token=f'check{len(self.checks)+1:03d}'
        result={'check_id':token,'supported':all(x['met'] for x in rows),'criteria':rows,
                'control_steps':control_steps,'interpretation':interpretation,'native_success_checked':False,
                'semantic_surface_identity_verified':False,
                'note':'This checks the declared measured conditions, not the native task or semantic correctness of selections.'}
        self.checks[token]=result
        return self.save(token,result)

    def finish_claim(self,claim,check_id,reason,control_steps,*,advisory=False):
        if claim not in {'completed','incomplete','uncertain'}:raise ValueError('Explicit completed/incomplete/uncertain claim required')
        if not isinstance(reason,str) or not reason.strip():raise ValueError('Give the observed reason for stopping')
        check=self.checks.get(check_id,{})
        supported=bool(check.get('supported') and check.get('control_steps')==control_steps)
        accepted=advisory or claim!='completed' or supported
        result={'claim':claim,'check_id':check_id,'reason':reason,'control_steps':control_steps,
                'finish_allowed':accepted,'native_success_used':False,
                'completion_policy':'advisory' if advisory else 'required_measurement',
                'current_measurement_supported':supported,
                'feedback':('Claim recorded.' if accepted else 'Completion is unsupported or stale. Obtain current measurements, continue if appropriate, or explicitly finish as uncertain/incomplete. No motion has been interrupted.')}
        if advisory:result['feedback']='Agent judgment recorded for independent evaluation; quantitative measurement is advisory.'
        self.save('finish-claim',result)
        return result
