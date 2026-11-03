"""Observer-only provenance for the pixels actually supplied to robot tools.

Images are existing captures. No inference, simulation, geometry correction or
robot request is performed. Continuations never inherit an old pixel as a new
annotation. Optional evaluator references are read only from a separate export.
"""
import math
import os
from pathlib import Path

from PIL import Image
from report_contact_skills import contact_image, read


OPERATIONS = {'segment_parts', 'ground', 'ground_multiview', 'refresh_contact',
              'select_surface_contact', 'push', 'hook_pull', 'continue_push',
              'push_stroke', 'plan_contact_pose'}


def public_file(run, relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:return None
    bases=[run/'agent-workspace',run/'workspace']
    command=read(run/'agent-command.json',[])
    if isinstance(command,list):
        bases += [Path(command[i+1]) for i,v in enumerate(command[:-1]) if v in {'-C','--cd'}]
    for base in bases:
        path=base/relative
        if path.is_file() and path.resolve().is_relative_to(base.resolve()):return path
    return None


def vector(value, length):
    if not isinstance(value,(list,tuple)) or len(value)!=length:return None
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in value):return None
    return list(value)


def project(point, camera):
    """Calibrated world-to-image projection; no depth or visibility claim."""
    if vector(point,3) is None or not camera:return None
    transform=camera.get('camera_to_world');intrinsic=camera.get('intrinsic')
    if (not isinstance(transform,list) or len(transform)!=4 or any(vector(row,4) is None for row in transform)
            or not isinstance(intrinsic,list) or len(intrinsic)!=3 or any(vector(row,3) is None for row in intrinsic)):return None
    delta=[point[j]-transform[j][3] for j in range(3)]
    local=[sum(transform[j][i]*delta[j] for j in range(3)) for i in range(3)]
    if local[2]<=0:return None
    p=[sum(row[j]*local[j] for j in range(3)) for row in intrinsic]
    if not all(math.isfinite(x) for x in p) or abs(p[2])<1e-12:return None
    return [p[0]/p[2],p[1]/p[2]]


def axis_projection(model, camera):
    origin=vector(model.get('joint_origin_world_m'),3)
    axis=vector(model.get('joint_axis_world',model.get('axis_world')),3)
    if origin is None or axis is None:return None
    points=[project([origin[j]+t*axis[j] for j in range(3)],camera) for t in [-.1,.1]]
    return points if all(p is not None for p in points) else None


def selection_history(study, run, events):
    study,run=Path(study),Path(run)
    records=[];parts={};models={};attempt=1;latest=None;captures={}
    operator=read(run/'selection-evaluation-reference.json',{}) or {}

    def relative(path):return os.path.relpath(path,study) if path is not None else None

    def capture(oid,camera):
        key=(attempt,oid,camera)
        if key in captures:return captures[key]
        data={'image':None,'image_size':None,'control_step':None,'calibration':{}}
        if type(oid) is int and isinstance(camera,str) and camera.replace('_','').isalnum():
            prefix=f'robot_data/attempt{attempt:02d}/obs{oid:03d}'
            path=contact_image(run,prefix+'/'+camera+'.png',False)
            data['image']=relative(path)
            if path is not None:
                try:
                    with Image.open(path) as picture:data['image_size']=list(picture.size)
                except OSError:data['image']=None
            state=public_file(run,prefix+'/robot.json')
            calibration=public_file(run,prefix+'/calibration.json')
            data['control_step']=(read(state,{}) or {}).get('control_steps') if state else None
            data['calibration']=(read(calibration,{}) or {}).get(camera,{}) if calibration else {}
        captures[key]=data
        return data

    for event in events:
        request=event.get('request',{});op=request.get('operation');args=request.get('arguments',{})
        if not isinstance(args,dict):args={}
        response=event.get('response',{});result=response.get('result',{})
        if op=='reset':
            attempt=result.get('attempt',attempt)
            latest=None;models={};parts={}
        if op=='segment_parts':
            for part in result.get('parts',[]):parts[part['part_id']]=part
        if op in OPERATIONS:
            selections=([('regions',args.get('regions',{})),('comparison_regions',args.get('comparison_regions',{})),
                         ('check_view',args.get('check_view',{}))] if op=='ground_multiview' else
                        [('regions',args.get('regions',{}))] if op=='ground' else [('',args)])
            for variant,selected in selections:
                if not isinstance(selected,dict):continue
                camera=selected.get('camera')
                camera_source='request'
                if camera is None and op in {'continue_push','push_stroke','plan_contact_pose'}:
                    camera=result.get('contact',{}).get('camera') or 'agentview'
                    camera_source='display view; no camera selected in this call'
                oid=args.get('observation_id',latest)
                current=capture(oid,camera)
                calibration=current['calibration']
                raw_pixel=selected.get('contact_pixel_xy')
                pixel=vector(raw_pixel,2)
                hinge=selected.get('hinge_pixels_xy')
                hinge=([vector(p,2) for p in hinge] if isinstance(hinge,list) and len(hinge)==2 else None)
                if hinge is not None and any(p is None for p in hinge):hinge=None
                contact=result.get('contact',{})
                model=result.get('model',{})
                point=vector(contact.get('point_world_m',result.get('point_world_m',model.get('source_anchor_world_m',args.get('point_world_m')))),3)
                if variant in {'comparison_regions','check_view'}:point=None
                record_id=f'a{attempt}-e{event.get("sequence")}-{variant or op}'
                row={'id':record_id,'event':event.get('sequence'),'operation':op,'variant':variant or None,
                    'attempt':attempt,'observation_id':oid,'camera':camera,'camera_source':camera_source,
                    'control_step':current['control_step'],'image':current['image'],'image_size':current['image_size'],
                    'contact_pixel_xy':pixel,'raw_contact_pixel_xy':raw_pixel,'hinge_pixels_xy':hinge,
                    'pixel_source':'Luna request' if pixel is not None else 'no contact pixel supplied in this call',
                    'world_point_m':point,'projected_world_point_xy':project(point,calibration) if point is not None else None,
                    'preview':args.get('preview',False),'ok':response.get('ok'),
                    'outcome':result.get('stop_reason',response.get('error')) or ('model updated' if result.get('planner_model_updated') else 'recorded'),
                    'motion_executed':result.get('robot_motion_executed',False),'contact_name':args.get('contact_name'),
                    'contact_reference_origin_control':contact.get('geometry_observed_at_control_step'),
                    'regions':[],'masks':[],'segmentation_points':[],'notes':[]}
                prompts=selected.get('points_xy') if op=='segment_parts' else None
                if isinstance(prompts,list) and len(prompts)<=12:
                    labels=selected.get('point_labels')
                    if labels is None:labels=[1]*len(prompts)
                    if not isinstance(labels,list):labels=[]
                    for i,prompt in enumerate(prompts):
                        if vector(prompt,2) is not None:
                            label=labels[i] if i<len(labels) else None
                            row['segmentation_points'].append({'xy':prompt,'include':label if type(label) is int and label in [0,1] else None})
                arrival=result.get('approach_arrival') or result.get('motion_arrival') or result.get('execution',{}).get('arrival')
                if arrival:row['arrival']={'goal_reached':arrival.get('goal_reached'),'goal_error_m':arrival.get('goal',{}).get('position_error_m')}
                after=result.get('observation',{})
                if type(after.get('observation_id')) is int:
                    result_capture=capture(after['observation_id'],camera)
                    row['result_image']=result_capture['image']
                    row['result_control_step']=result_capture['control_step']
                for key,label in [('box_xyxy','Luna segmentation prompt box'),('region_xyxy','Luna contact region'),
                    ('target_region_xyxy','Luna moving-part region'),('contact_box_xyxy','Luna contact region'),
                    ('moving_panel_box_xyxy','Luna moving-part region')]:
                    box=vector(selected.get(key),4)
                    if box is not None:row['regions'].append({'label':label,'box_xyxy':box,'source':'Luna request'})
                if op=='segment_parts':
                    mask_parts=[(p,'SAM proposal') for p in result.get('parts',[])]
                else:
                    mask_parts=[(parts[p],role) for key,role in [('contact_part_id','Contact mask'),('moving_part_id','Moving-part mask')]
                                if isinstance(p:=args.get(key),str) and p in parts]
                seen=set()
                for part,role in mask_parts:
                    if part['part_id'] in seen:continue
                    seen.add(part['part_id'])
                    overlay=contact_image(run,part.get('overlay',''),False)
                    same_capture=part.get('observation_id')==oid and part.get('camera')==camera
                    row['masks'].append({'part_id':part['part_id'],'role':role,'label':part.get('caller_label'),
                        'image':relative(overlay),'same_capture':same_capture,'observation_id':part.get('observation_id'),
                        'camera':part.get('camera'),'bounds':part.get('mask_box_xyxy')})
                if not row['regions'] and row['masks']:
                    for part in row['masks']:
                        if part['same_capture'] and vector(part['bounds'],4) is not None:
                            row['regions'].append({'label':'SAM mask bounds: '+part['part_id'],'box_xyxy':part['bounds'],'source':'SAM output'})
                direction=vector(args.get('direction_world'),3)
                direction_source='Luna world-direction request'
                if direction is None and args.get('joint_degrees') is not None:
                    direction=vector(contact.get('direction_world'),3)
                    direction_source='tool-computed tangent for Luna’s signed joint request'
                row.update(direction_world=direction,direction_source=direction_source if direction is not None else None,
                    requested_distance_m=result.get('requested_distance_m',args.get('distance_m')),
                    requested_joint_degrees=result.get('requested_joint_degrees',args.get('joint_degrees')),
                    requested_rotation_direction=args.get('rotation_direction'),
                    rotation_view_convention=result.get('rotation_view_convention'),
                    contact_effect=result.get('selection_geometry',{}).get('contact_effect'))
                if point is not None and direction is not None:
                    length=math.sqrt(sum(x*x for x in direction))
                    if length:
                        end=project([point[j]+.05*direction[j]/length for j in range(3)],calibration)
                        start=project(point,calibration)
                        if start is not None and end is not None:
                            row['direction_arrow']={'start':start,'end':end,'world_guide_m':.05}
                hint=vector(selected.get('initial_image_direction_xy'),2)
                if hint is not None and pixel is not None and sum(x*x for x in hint)>0:
                    length=math.sqrt(sum(x*x for x in hint))
                    row['image_direction_hint']={'start':pixel,'end':[pixel[j]+35*hint[j]/length for j in range(2)]}
                if op in {'continue_push','push_stroke'}:
                    row['notes'].append('No new pixel was selected. Any blue marker is a projected hand/contact reference, not a Luna click.')
                    row['notes'].append('A commanded world direction does not establish opening/closing intent or physical contact.')
                    if row['contact_reference_origin_control'] is not None:
                        row['notes'].append('The tool’s contact/reference record originated at control '+str(row['contact_reference_origin_control'])+'.')
                if not current['image']:row['notes'].append('The exact source image is unavailable; no later capture is substituted.')
                if op=='plan_contact_pose':row['notes'].append('This request supplies a 3D world point, not image coordinates; the blue marker is a calibrated projection.')
                if type(args.get('joint_model_epoch')) is int:
                    selected_model=models.get(args['joint_model_epoch'])
                    row['joint_model_epoch']=args['joint_model_epoch']
                    row['model_reference']=({k:selected_model[k] for k in ('event','id','camera','observation_id')} if selected_model else None)
                    if selected_model:row['model_axis_line']=axis_projection(selected_model['model'],calibration)
                if variant in {'','regions'} and model and result.get('geometry_epoch') is not None:
                    row['geometry_epoch']=result['geometry_epoch']
                    row['model_axis_line']=axis_projection(model,calibration)
                    models[result['geometry_epoch']]={'event':event['sequence'],'id':record_id,
                        'camera':camera,'observation_id':oid,'model':model}
                reference=(operator.get('states',{}).get(str(current['control_step']))
                    if operator.get('attempt',1)==attempt else None)
                if reference:
                    row['evaluator_axis_line']=axis_projection(reference,calibration)
                    row['evaluator_reference_scope']='Operator-only simulator reference; never supplied to Luna or planning.'
                row['summary']=f'E{row["event"]} · {op}'+(' preview' if row['preview'] else '')+f' · obs {oid} · {camera}'
                records.append(row)
        observed=result.get('observation')
        if not isinstance(observed,dict) and op in {'reset','observe'}:observed=result
        if isinstance(observed,dict) and type(observed.get('observation_id')) is int:
            latest=observed['observation_id']
    for row in records:
        if row.get('geometry_epoch') is not None:
            row['referenced_by']=[{'event':r['event'],'id':r['id'],'operation':r['operation'],'preview':r['preview'],'ok':r['ok']}
                for r in records if (r.get('model_reference') or {}).get('id')==row['id']]
            if not row['referenced_by']:row['notes'].append('No recorded push/hook request in this history references this model epoch; later hypotheses may replace it.')
    return records
