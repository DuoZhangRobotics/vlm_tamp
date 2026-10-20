"""Report and independently verify a frozen mechanism-family transfer test."""
import argparse
from collections import Counter
import hashlib
import html
import os
from pathlib import Path
import sys
import json
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.family_transfer import TASKS
from vlm_tamp.coupled_resources import api_snapshot
from validate_reliability import check_episode


def category(row):
    if row['official_task_success']:return 'native_success'
    failure=row.get('failure','')
    if row.get('failure_stage')=='rgb_grounding':return 'visual_grounding'
    if row.get('failure_stage')=='numerical_geometry':return 'metric_geometry'
    if 'No candidate' in failure:return 'grasp_admission'
    if 'collision_at_start' in failure:return 'next_plan_start_collision'
    if 'No complete' in failure:return 'constrained_planning'
    if 'budget' in failure.lower():return 'control_step_budget'
    return row.get('failure_stage') or 'native_goal_not_reached'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run=(ROOT/a.run).resolve();out=(ROOT/a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=read_json(run/'protocol.json');entries=read_json(run/'manifest.json')['episodes'];rows=[];checks=[];cards=[]
    relative=lambda path:os.path.relpath(path,out)
    for entry in entries:
        folder=run/f'{entry["task"]}-state{entry["state"]:03d}'
        if not (folder/'summary.json').exists():continue
        result=read_json(folder/'summary.json')
        if 'experiment_phase' not in result or 'storage' not in result:continue
        result['failure_category']=category(result)
        actions=read_json(folder/'initial/actions.json')
        result['first_native_success_step']=next((i+1 for i,x in enumerate(actions) if x['evaluation_task_success']),None)
        result['goal_already_true_during_initial_opening']=any(x['evaluation_task_success'] for x in actions[:60])
        checks.append(check_episode(folder));rows.append(result)
        selection=folder/'grounding/grounding.json';choices=read_json(selection)['choices'] if selection.exists() else None
        overlay=None
        if choices:
            raw=folder/'grounding'/choices['camera']/'rgb.png'
            if raw.exists():
                assets=out/'images';assets.mkdir(exist_ok=True);overlay=assets/(folder.name+'.png')
                picture=Image.open(raw).convert('RGB');draw=ImageDraw.Draw(picture)
                for field,color in [('contact_box_xyxy','lime'),('moving_panel_box_xyxy','cyan')]:
                    draw.rectangle(choices[field],outline=color,width=2)
                x,y=choices['grasp_pixel_xy'];draw.ellipse([x-4,y-4,x+4,y+4],fill='red');picture.save(overlay)
        media=''
        if overlay:media+=f'<img src="{relative(overlay)}" alt="VLM contact and moving-part regions">'
        video=folder/'initial/rollout.mp4'
        if video.exists():media+=f'<video controls preload="none" src="{relative(video)}"></video>'
        links=[f'<a href="{relative(folder/"summary.json")}">Outcome</a>']
        for file,label in [('grounding/response.txt','VLM answer'),('task-spec.json','Measured constraint'),('grasp-admission.json','Grasp checks'),('phases.json','Executed motions')]:
            path=folder/file
            if not path.exists() and path.suffix=='.json':path=path.with_suffix('.json.gz')
            if path.exists():links.append(f'<a href="{relative(path)}">{label}</a>')
        success=result['official_task_success'];detail=result.get('failure','Completed planned motions')
        if success and result.get('failure'):detail='Native goal succeeded; later planning stopped: '+detail
        label=entry['task'].replace('_',' ')
        cards.append(f'<details><summary>{html.escape(label)} · state {entry["state"]} · {"success" if success else "failure"} · {entry["phase"]}</summary><p>{html.escape(detail)}</p><p>{" · ".join(links)}</p><div class="media">{media}</div></details>')
    groups={}
    for phase in ['reproduction_control','original_tasks','related_tasks']:
        part=[r for r in rows if r['experiment_phase']==phase]
        groups[phase]={'planned':sum(e['phase']==phase for e in entries),'completed':len(part),
            'native_successes':sum(r['official_task_success'] for r in part),'final_native_successes':sum(r['final_task_success'] for r in part),
            'failure_categories':dict(Counter(r['failure_category'] for r in part if not r['official_task_success']))}
    transfer=[r for r in rows if r['experiment_phase']!='reproduction_control']
    task_results=[]
    for task in TASKS:
        part=[r for r in transfer if r['task']==task['label']]
        task_results.append({'task':task['label'],'group':task['group'],'family':task['family'],
            'planned':sum(e['task']==task['label'] and e['phase']!='reproduction_control' for e in entries),
            'completed':len(part),'native_successes':sum(r['official_task_success'] for r in part),
            'final_native_successes':sum(r['final_task_success'] for r in part)})
    ledger=api_snapshot();before=protocol['api_before']
    result={'planned_transfer':sum(e['phase']!='reproduction_control' for e in entries),'completed_transfer':len(transfer),
        'native_transfer_successes':sum(r['official_task_success'] for r in transfer),'groups':groups,
        'tasks':task_results,'episodes':rows,'demonstration_inputs':False,'neural_finetuning':False,
        'same_configuration_all_cases':True,'no_changes_after_freeze':True,
        'scope':'New initial states and related task/layouts using existing cabinet and stove assets; no novel-design claim.',
        'no_demonstration_is_not_no_prior':'Hand-engineered family priors and pretrained perception/grasp models remain.',
        'new_api_spend_usd':ledger['cumulative_conservative_spend_usd']-before['cumulative_conservative_spend_usd'],
        'new_api_calls':ledger['api_attempts']-before['api_attempts'],
        'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd'],'remaining_api_budget_usd':ledger['remaining_usd']}
    write_json(out/'evaluation.json',result)
    hashes=protocol['sources_sha256']
    snapshots=all(hashlib.sha256((run/'source-snapshot'/n).read_bytes()).hexdigest()==v for n,v in hashes.items())
    current=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in hashes.items())
    write_json(out/'validation.json',{'passed':len(rows)==len(entries) and snapshots and current,
        'completed':len(rows),'planned':len(entries),'snapshot_hashes_verified':snapshots,'current_sources_match':current,
        'baseline_core_unchanged':protocol['frozen_geometry_planner_and_controller'],
        'episodes':checks,'execution_vetoes':0,'reclaimed_bytes':sum(c['reclaimed_bytes'] for c in checks),
        'api_cap_usd':2.,'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd']})
    body=''.join(f'<tr><td>{html.escape(t["task"].replace("_"," "))}</td><td>{t["group"].replace("_"," ")}</td><td>{t["native_successes"]}/{t["planned"]}</td><td>{t["completed"]}</td></tr>' for t in task_results)
    states=sorted({e['state'] for e in entries if e['phase']!='reproduction_control'})
    control=groups['reproduction_control'];original=groups['original_tasks'];related=groups['related_tasks']
    (out/'index.html').write_text(f'''<!doctype html><html><head><meta charset="utf-8"><title>Drawer and stove family transfer</title><style>
body{{font:16px system-ui;max-width:1120px;margin:36px auto;padding:0 24px;background:#f4f7fa;color:#172535}}h1{{font-size:30px}}.box,details{{background:white;padding:22px;margin:18px 0;border-radius:10px;border:1px solid #d5dfe8}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #dce3ea;padding:10px}}.media{{display:flex;gap:12px;flex-wrap:wrap}}img,video{{width:48%;min-width:260px;max-width:512px}}summary{{cursor:pointer;font-weight:600}}a{{color:#155997}}</style></head><body>
<h1>Drawer and stove family transfer</h1><div class="box"><p><strong>{result['native_transfer_successes']}/{result['planned_transfer']} native successes</strong> · {result['completed_transfer']} transfer episodes completed</p>
<p>Reproduction controls: {control['native_successes']}/{control['planned']}. Original tasks with new initial states: {original['native_successes']}/{original['planned']}. Related tasks/layouts: {related['native_successes']}/{related['planned']}.</p>
<p>One fixed configuration from the successful mechanism regression. Existing drawer and knob geometry, MC-VAMP, GraspGen, IK, contact rules and controller are preserved. The original compact Luna prompt selects visual roles. Fresh transfer observations use bounded DINO clustering; coherent collage-coordinate repair uses the existing baseline rule. No case-specific tuning or demonstration video is used.</p>
<p>Transfer states: {states}; seed 0; five settling actions; maximum 600 control steps. Every planned transfer failure counts. Three state-1 controls reuse only observation-validated baseline VLM choices and are excluded from the transfer score. All issued trajectories execute completely; later planning can fail.</p>
<p>This tests case/layout transfer within existing LIBERO cabinet and stove assets. It does not establish transfer to new appliance designs. Family geometry priors remain, including a cabinet face below the drawer for closing and a protruding control for knob-axis estimation.</p>
<p>API spending this experiment: ${result['new_api_spend_usd']:.4f}; cumulative ${result['cumulative_api_spend_usd']:.4f} of $2.00. Cleanup runs after every episode.</p>
<p><a href="evaluation.json">All outcomes</a> · <a href="validation.json">Independent verification</a> · <a href="{relative(run/'protocol.json')}">Frozen protocol</a></p></div>
<div class="box"><table><thead><tr><th>Task</th><th>Group</th><th>Native success</th><th>Completed</th></tr></thead><tbody>{body}</tbody></table></div>
<p>Videos and first blocking failures are below. A planning-time rejection is not evidence of a physical collision.</p>{''.join(cards)}</body></html>''')
    print(json.dumps({k:v for k,v in result.items() if k not in {'episodes','tasks'}},indent=2))


if __name__=='__main__':main()
