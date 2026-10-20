"""Report frozen interaction experiments without dropping pre-motion failures."""
import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import sys
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from validate_reliability import check_episode


def failure_category(row):
    message=row.get('failure','')
    if row['official_task_success']:return 'native_success'
    if 'unresolved interaction' in message:return 'goal_or_model_unresolved'
    if '0..511' in message:return 'image_coordinate_frame'
    if 'distinct endpoints required' in message:return 'output_schema'
    if 'cylindrical bar' in message or 'bar endpoints' in message:return 'selected_shape_not_supported_by_depth'
    if 'connected depth support' in message:return 'source_region_not_supported_by_depth'
    if 'No candidate with complete grasp' in message:return 'no_admitted_grasp_plan'
    if row.get('failure_stage')=='vlm_selection_and_depth_fitting':return 'other_visual_geometry_failure'
    return row.get('failure_stage','native_goal_false')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run=(ROOT/a.run).resolve();out=(ROOT/a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=read_json(run/'protocol.json');manifest=read_json(run/'manifest.json')['episodes']
    hashes=protocol['sources_sha256'];snapshot_ok=all(hashlib.sha256((run/'source-snapshot'/n).read_bytes()).hexdigest()==v for n,v in hashes.items())
    current_ok=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in hashes.items())
    cases=[];checks=[];cards=[]
    def url(path):return os.path.relpath(path,out)
    for entry in manifest:
        folder=run/f'{entry["task"]}-state{entry["state"]:03d}'
        if not (folder/'summary.json').exists():
            cases.append({**entry,'pending':True});continue
        row=read_json(folder/'summary.json');check=check_episode(folder);checks.append(check)
        selection=read_json(folder/'grounding/selection.json') if (folder/'grounding/selection.json').exists() else None
        routing=read_json(folder/'routing.json') if (folder/'routing.json').exists() else None
        row.update(valid_selection=selection is not None,fitted_geometry=(folder/'grounding/fitted-spec.json').exists(),routing=routing)
        row['failure_category']=failure_category(row)
        cases.append(row);label=entry['task'].replace('_',' ')
        status='Native success' if row['official_task_success'] else 'Failure'
        detail=html.escape(row.get('failure',row.get('continuation_stop_reason','Completed planned motion')))
        media=[]
        overlay=folder/'grounding/grounding-overlay.png'
        if not overlay.exists() and selection is not None:
            raw=folder/'grounding'/selection['camera']/'rgb.png'
            if raw.exists():
                assets=out/'images';assets.mkdir(exist_ok=True);overlay=assets/(folder.name+'-selection.png')
                picture=Image.open(raw).convert('RGB');draw=ImageDraw.Draw(picture)
                for box,color in [(selection['contact']['box_xyxy'],'lime'),(selection['moving']['box_xyxy'],'cyan'),
                                  (selection['mechanism']['support_box_xyxy'],'orange'),(selection['goal']['reference_box_xyxy'],'yellow')]:
                    if box is not None:draw.rectangle(box,outline=color,width=2)
                x,y=selection['contact']['pixel_xy'];draw.ellipse([x-4,y-4,x+4,y+4],fill='red')
                picture.save(overlay)
        if overlay.exists():media.append(f'<img src="{url(overlay)}" alt="VLM regions and fitted axes">')
        elif (folder/'opened/frontview.png').exists():media.append(f'<img src="{url(folder/"opened/frontview.png")}" alt="Observed scene">')
        if (folder/'initial/rollout.mp4').exists():media.append(f'<video controls preload="none" src="{url(folder/"initial/rollout.mp4")}"></video>')
        semantic='' if selection is None else f'<p>Selected: {html.escape(selection["contact"]["shape"])} · {html.escape(selection["mechanism"]["family"])} · {html.escape(selection["goal"]["relation"])}</p>'
        links=[f'<a href="{url(folder/"summary.json")}">Outcome</a>']
        for file,title in [('selection.json','VLM choices'),('fit-evidence.json','Measured fit'),('prompt.txt','Exact prompt'),('response.txt','Raw answer')]:
            target=folder/'grounding'/file
            if target.exists():links.append(f'<a href="{url(target)}">{title}</a>')
        cards.append(f'<article><h2>{html.escape(label)} · state {entry["state"]}</h2><b>{status}</b><p>{detail}</p>{semantic}<p>{" · ".join(links)}</p><div class="media">{"".join(media)}</div></article>')
    completed=[r for r in cases if not r.get('pending')];successes=sum(r['official_task_success'] for r in completed)
    errors=Counter(r.get('failure_stage','native_goal_false') for r in completed if not r['official_task_success'])
    categories=Counter(r['failure_category'] for r in completed)
    ledger=read_json(ROOT/'results/articulated-api-v1/ledger.json')
    states=sorted({r['state'] for r in manifest})
    result={'planned':len(manifest),'completed':len(completed),'native_successes':successes,
        'final_native_successes':sum(r['final_task_success'] for r in completed),
        'valid_visual_selections':sum(r['valid_selection'] for r in completed),
        'completed_geometry_fits':sum(r['fitted_geometry'] for r in completed),
        'episodes_with_joint_commands':sum(c['joint_commands']>0 for c in checks),
        'failure_stages':dict(errors),'failure_categories':dict(categories),'cases':cases,
        'scope':'One state per task held out from success-first tuning; seven known tasks and two new drawer task/layout combinations. Not new object assets. The historical search checks physical outcome summaries, not every old image-only dataset.',
        'development_observation':'Microwave opening state 1 for prompt/interface development; state 2 for placement integration; state 3 for an inherited DINO NaN-loop regression. All attempts retained.',
        'new_api_spend_usd':ledger['cumulative_conservative_spend_usd']-protocol['api_before']['cumulative_conservative_spend_usd'],
        'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd'],'remaining_api_budget_usd':ledger['remaining_usd']}
    write_json(out/'evaluation.json',result)
    write_json(out/'validation.json',{'passed':len(completed)==len(manifest) and snapshot_ok and current_ok,
        'frozen_snapshot_verified':snapshot_ok,'current_sources_match':current_ok,'episodes':checks,
        'execution_vetoes':0,'reclaimed_bytes':sum(r['storage']['bytes_reclaimed_this_pass'] for r in completed)})
    opening=f'''<!doctype html><html><head><meta charset="utf-8"><title>Shared VLM interaction interface</title>
<style>body{{font:16px system-ui;max-width:1120px;margin:36px auto;padding:0 22px;background:#f4f6f9;color:#172535}}h1{{font-size:30px}}article,.intro{{background:white;padding:22px;margin:20px 0;border-radius:12px;border:1px solid #d5dfe8}}.media{{display:flex;gap:16px;flex-wrap:wrap}}img,video{{width:48%;min-width:260px;max-width:512px}}a{{color:#155997}}code{{background:#eef1f5}}</style></head><body>
<h1>Shared VLM interaction interface</h1><div class="intro"><p><strong>{successes}/{len(manifest)} native LIBERO successes</strong> · {len(completed)} completed cases</p>
<p>One prompt selects contact shape, mechanism, moving region and goal reference. Depth fits the numeric geometry. The planner is selected from predicted physical properties, without a task-name dispatch rule. Every planned case remains in the denominator.</p>
<p>{result['valid_visual_selections']} valid VLM selections · {result['completed_geometry_fits']} completed fits · {result['episodes_with_joint_commands']} cases issued joint trajectories. A completed fit is not evidence that the chosen physical model was correct. Failure counts identify the first blocking check; some outputs contain more than one issue.</p>
<p>Development history is separate: <a href="{url(ROOT/'results/interaction-development-open-20261016/grounding/selection.json')}">initial interface</a>, <a href="{url(ROOT/'results/interaction-development-face-20261016/grounding/selection.json')}">explicit face patch</a>, <a href="{url(ROOT/'results/interaction-heldout-20261016/administrative-stop.json')}">state-2 integration failures and unstarted cases</a>, <a href="{url(ROOT/'results/interaction-placement-smoke-20261016/bowl_on_plate-state002/summary.json')}">corrected placement smoke test</a>, and <a href="{url(ROOT/'results/interaction-final-state3-20261016/administrative-stop.json')}">state-3 clustering failure</a>. These are not selected successful attempts in the final score.</p>
<p>This is an initial generalization experiment on state(s) {html.escape(str(states))} of seven known tasks and two additional bottom-drawer tasks. The additional tasks use existing LIBERO cabinet assets; there is no novel-object claim. No rules were adjusted using these held-out outcomes.</p>
<p>Remaining assumptions: finite primitive library; RGB-D masks and observed-surface collision models; a quarter-turn rotational opening goal; robot-hand-sized sliding opening; a two-standard-deviation closing allowance capped by robot aperture. The mature partial-pose hinge tracker supports upright bars and hinges within 3°; other orientations use the full-pose mechanism planner, whose broader physical reliability is unproven. DINO weights remain unchanged; clustering normalizes constant dimensions to zero, skips empty clusters, and stops after at most 100 iterations.</p>
<p>All issued trajectories execute fully. Native success is evaluation-only. Luna high uses a 12,000-token cap, a durable cache, no automatic retries and the cumulative $2 ceiling. This cohort spent ${result['new_api_spend_usd']:.4f}; cumulative ${result['cumulative_api_spend_usd']:.4f}.</p>
<p><a href="evaluation.json">All outcomes</a> · <a href="validation.json">Independent verification</a> · <a href="{url(run/'protocol.json')}">Frozen protocol</a> · <a href="{url(ROOT/'configs/prompts/interaction_choices_v1.txt')}">Shared prompt</a></p></div>'''
    (out/'index.html').write_text(opening+''.join(cards)+'</body></html>')
    print(json.dumps({k:v for k,v in result.items() if k not in {'cases','development_observation'}},indent=2))


if __name__=='__main__':main()
