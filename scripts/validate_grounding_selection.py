"""Recheck current recorded grounding inputs without simulation or actor hints."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw

from vlm_tamp.handle_contact import ground_panel
from vlm_tamp.reliable_handle import ground_handle
from vlm_tamp.rekep_libero.mechanism_geometry import project

ROOT=Path(__file__).resolve().parents[1]


def recorded_input(root):
    events=[json.loads(line) for line in (root/'tool-events.jsonl').open()]
    event=next(row for row in events if row['request']['operation']=='ground')
    choices=copy.deepcopy(event['request']['arguments']['regions']);choices['motion']='close'
    initial=next(row for row in events if row['request']['operation']=='reset')['response']['result']
    observation=root/'agent-workspace'/Path(initial['sensor_arrays']).parent
    metadata=json.loads((root/'attempt01/initial/metadata.json').read_text())
    calibration=json.loads((observation/'calibration.json').read_text())
    with np.load(observation/'sensors.npz',allow_pickle=False) as arrays:
        cameras={name:{**values,'robot_depth_prior':metadata['cameras'][name]['robot_depth_prior'],
            **{key:arrays[name+'_'+key] for key in ['rgb','depth','world_points','known_robot_depth_prior']}}
            for name,values in calibration.items()}
    segmentation=next(row['response']['result'] for row in events if row['request']['operation']=='segment_parts' and row['response']['ok'])
    part=next(p for p in segmentation['parts'] if p['camera']==choices['camera'])
    with np.load(root/'agent-workspace'/part['mask_file']) as arrays:mask=arrays['usable_mask']
    return cameras,choices,metadata,mask,event


def run(output):
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for state in [17,49]:
        root=ROOT/f'results/luna-current-closing-checks-20261031/state{state:03d}/trials/baseline-r1-close_microwave-state{state:03d}-600'
        cameras,choices,metadata,mask,old=recorded_input(root)
        original=copy.deepcopy(choices);template={'joint_family':'revolute','target_progress':float(np.pi/2)}
        row={'state':state,'source':str(root.relative_to(ROOT)),'unchanged_regions':original,'fits':{}}
        for mode in ['box','part_mask','door_only']:
            started=time.perf_counter();options={'panel_mask':mask} if mode=='part_mask' else {}
            fitter=ground_panel if mode=='door_only' else ground_handle
            mechanism,cloud,_,_,evidence=fitter(cameras,choices,template,metadata,**options)
            row['fits'][mode]={'seconds':time.perf_counter()-started,'model':mechanism.spec,'evidence':evidence}
            assert choices==original
            if mode!='door_only':
                assert evidence['cylinder_robust_residual_m']<.0035
                assert evidence['central_bar_refinement']['radial_residual_m']<.0025
            if mode=='box':
                camera=cameras[choices['camera']];base=Image.fromarray(camera['rgb']);w,h=base.size
                picture=Image.new('RGB',(2*w,h+45),'white');picture.paste(base,(0,0));picture.paste(base,(w,0))
                draw=ImageDraw.Draw(picture)
                draw.rectangle(choices['moving_panel_box_xyxy'],outline='lime',width=2)
                draw.rectangle(choices['contact_box_xyxy'],outline='orange',width=2)
                for x,y in project(cloud,camera)[::max(1,len(cloud)//1500)]:
                    if 0<=x<w and 0<=y<h:draw.ellipse((x+w-1,y-1,x+w+1,y+1),fill='yellow')
                draw.text((8,h+6),'Original Luna selections: door green; handle orange',fill='black')
                draw.text((w+8,h+6),f'Observed handle fit: {evidence["cylinder_robust_residual_m"]*1000:.3f} mm residual',fill='black')
                picture.save(output/f'state{state:03d}.png')
        previous=old['response'].get('result',{}).get('fit_diagnostics',old['response'].get('planning_diagnostic',{}))
        row.update(previous_ok=old['response']['ok'],previous_error=old['response'].get('error'),
            previous_residual_m=previous.get('cylinder_robust_residual_m',previous.get('radial_residual_m')))
        if state==17:
            old_model=old['response']['result']['model'];new=row['fits']['box']['model']
            for key in ['source_anchor_world_m','joint_origin_world_m','joint_axis_world']:
                np.testing.assert_allclose(new[key],old_model[key],atol=1e-12,rtol=0)
        rows.append(row)
    report={'cases':rows,'simulator_controls_executed':0,'source':'saved public RGB-D, calibration and original Luna selections',
        'actor_receives_diagnostic_coordinates':False,'scope':'Fitting regression only; no task-success claim'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    body='<h1>Grounding selection regression</h1><p>Unchanged Luna selections; saved observations only. No robot motion or successful coordinates supplied to actors.</p><table><tr><th>State</th><th>Old error (mm)</th><th>Corrected box fit</th><th>Existing mask fit</th></tr>'
    for row in rows:
        values=[row['previous_residual_m'],*[row['fits'][m]['evidence']['cylinder_robust_residual_m'] for m in ['box','part_mask']]]
        body+=f'<tr><td>{row["state"]}</td>'+''.join(f'<td>{1000*v:.3f}</td>' for v in values)+'</tr>'
    body+='</table>'
    for row in rows:body+=f'<h2>State {row["state"]}</h2><img src="state{row["state"]:03d}.png" alt="Original selections and corrected observed handle fit">'
    body+='<p><a href="summary.json">Full measurements and provenance</a></p>'
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Grounding diagnostics</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;color:#203249}td,th{padding:10px;text-align:left}img{max-width:100%}</style>'+body)
    print(json.dumps({'output':str(output),'recorded_cases_passed':len(rows)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
