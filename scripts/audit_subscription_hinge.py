"""Post-run hinge diagnosis. Simulator truth here is evaluation-only.

Never import this module into the policy or numerical robot service.
"""
import argparse
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from vlm_tamp.rigid_hinge_tracking import estimate_rigid_hinge


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    run=(ROOT/args.run).resolve()
    if not (run/'agent-exit.json').exists():raise ValueError('Audit requires a completed agent run')
    events=[json.loads(line) for line in (run/'tool-events.jsonl').read_text().splitlines()]
    episode=0; rows=[]; current=None; groundings=[]; fits=[]; axis_requests=[]
    for row in events:
        op=row['request']['operation'];result=row['response'].get('result',{})
        if op=='reset' and row['response']['ok']:
            episode+=1
            native=read(run/f'attempt{episode:02d}/initial/evaluation_contact_diagnostics.json')['steps']
            def joint(step):return next(j for j in step['evaluation_only_articulation_state'] if 'microwave' in j['joint_name'])
            first,last=joint(native[0]),joint(native[-1])
            anchor=np.asarray(first['anchor_world_m']);axis=np.asarray(first['axis_world'])
            current={'attempt':episode,'true_hinge_world_m':anchor.tolist(),
                'final_opening_degrees':float(np.degrees(abs(last['position']))),
                'maximum_opening_degrees':float(max(np.degrees(abs(joint(s)['position'])) for s in native))}
            rows.append(current)
        if op=='ground' and row['response']['ok']:
            model=result['model'];delta=np.asarray(model['joint_origin_world_m'])-anchor
            groundings.append({'attempt':episode,'event':row['sequence'],
                'estimated_hinge_world_m':model['joint_origin_world_m'],
                'perpendicular_hinge_error_m':float(np.linalg.norm(delta-axis*(delta@axis))),
                'axis_world':model['joint_axis_world'],
                'regions':row['request']['arguments']['regions']})
        if op=='measure_tracking' and row['response']['ok']:
            original=result['measurement'];path=run/'agent-workspace'/result['correspondences']
            with np.load(path) as data:
                no_prior=estimate_rigid_hinge(data['reference_world'],data['current_world'],prior=None)
            candidate=no_prior.get('hinge_world_m');error=None
            if candidate is not None:
                delta=np.asarray(candidate)-anchor;error=float(np.linalg.norm(delta-axis*(delta@axis)))
            fits.append({'attempt':episode,'event':row['sequence'],
                'original_hinge_accepted':original['accepted'],
                'same_fit_accepted_without_initial_pivot_prior':no_prior['accepted'],
                'fit_hinge_world_m':candidate,'perpendicular_hinge_error_m':error,
                'rotation_degrees':float(np.degrees(no_prior.get('rotation_rad',0))),
                'inliers':no_prior.get('inliers'),'rms_m':no_prior.get('rms_m'),
                'reason_without_prior':no_prior['reason']})
        if op=='set_model':
            axis_requests.append({'event':row['sequence'],'requested':row['request']['arguments'],
                'returned_axis':result.get('model',{}).get('joint_axis_world')})
    prior_rejections=[f for f in fits if not f['original_hinge_accepted'] and f['same_fit_accepted_without_initial_pivot_prior']]
    audit={'evaluation_only':True,'provided_to_agent':False,'physical_actions_executed':0,
        'episodes':rows,'grounding_hypotheses':groundings,'tracking_fits':fits,
        'fits_rejected_only_by_initial_pivot_prior':prior_rejections,
        'axis_change_requests':axis_requests,
        'scope':'Offline diagnostic, not a tested policy improvement. Removing a prior can also admit bad fits.'}
    (run/'hinge-diagnosis-evaluation-only.json').write_text(json.dumps(audit,indent=2))
    # Original RGB remains unchanged; the SVG plots numeric evaluation overlays.
    first=groundings[0];camera=first['regions']['camera']
    metadata=read(run/'attempt01/opened/metadata.json')['cameras'][camera]
    inverse=np.linalg.inv(np.asarray(metadata['camera_to_world']));K=np.asarray(metadata['intrinsic'])
    def projected_line(origin):
        points=np.array([origin,origin],float);points[:,2]=[.94,1.09]
        optical=points@inverse[:3,:3].T+inverse[:3,3]
        uv=optical@K.T;uv=uv[:,:2]/uv[:,2:]
        return ' '.join(f'{x:.2f},{y:.2f}' for x,y in uv)
    actual=projected_line(rows[0]['true_hinge_world_m'])
    predicted=projected_line(first['estimated_hinge_world_m'])
    section='<article><h2>Evaluation-only mechanism diagnosis</h2>'
    section+='<p>Orange: the initial predicted hinge. Cyan: the simulator hinge, used only after the run for evaluation. These overlays and true coordinates were never supplied to Luna.</p>'
    section+=f'<svg viewBox="0 0 512 512" style="max-width:600px;width:100%"><image href="../attempt01/opened/{camera}.png" width="512" height="512"/><polyline points="{actual}" fill="none" stroke="#00eaff" stroke-width="5"/><polyline points="{predicted}" fill="none" stroke="#ff9d00" stroke-width="5"/></svg>'
    section+=f'<p>The first hinge-line error was {first["perpendicular_hinge_error_m"]*100:.1f} cm. '
    section+=f'{len(prior_rejections)} motion fits failed the initial-pivot proximity condition while passing the same numerical fit checks without that prior. This diagnosis does not establish that removing the prior would complete the task.</p>'
    section+='<details><summary>Native outcomes, measured fits, and axis-update audit</summary><pre>'+html.escape(json.dumps(audit,indent=2))+'</pre></details></article>'
    page=run/'report/index.html';body=page.read_text()
    page.write_text(body.replace('<article><h2>Agent-authored findings</h2>',section+'<article><h2>Agent-authored findings</h2>'))
    print(json.dumps({'episodes':rows,'first_hinge_error_m':first['perpendicular_hinge_error_m'],
        'prior_only_rejections':prior_rejections,'axis_changes':axis_requests},indent=2))


if __name__=='__main__':main()
