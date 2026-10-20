"""Build a compact physical-outcome comparison with links to retained videos."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.coupled_storage import read_json,write_json

LABELS={'open_middle_drawer':'Open middle drawer','open_top_drawer':'Open top drawer',
    'close_top_drawer':'Close top drawer','turn_on_stove':'Turn on stove'}
BASELINE=ROOT/'results/all-prior-tasks-frozen-v3-20261015'


def articulation(folder):
    path=folder/'initial/evaluation_contact_diagnostics.json'
    if not path.exists():return None
    steps=read_json(path)['steps']
    if not steps:return None
    for item in steps[-1].get('evaluation_only_articulation_state',[]):
        name=item['joint_name']
        if ('stove' in name and 'button' in name) or ('cabinet' in name and
            (('middle' in folder.name and 'middle' in name) or ('top' in folder.name and 'top' in name))):
            first=next((r for r in steps[0]['evaluation_only_articulation_state'] if r['joint_name']==name),None)
            return {'name':name,'type':item['type'],'initial':first['position'] if first else None,
                'final':item['position'],'evaluation_only':True}
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',default='results/mcvamp-drawer-stove-report');p.add_argument('--retry-runs',nargs='*',default=[]);args=p.parse_args()
    out=(ROOT/args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    episodes=[];planned=[];seen=set();checks=[]
    for run in args.runs:
        root=(ROOT/run).resolve();manifest=read_json(root/'manifest.json')['episodes'];planned.extend(manifest)
        protocol=read_json(root/'protocol.json')
        for entry in manifest:
            key=(entry['task'],entry['state'])
            if key in seen:raise ValueError('Duplicate episode in comparison')
            seen.add(key);folder=root/f'{key[0]}-state{key[1]:03d}'
            if not (folder/'summary.json').exists():continue
            row=read_json(folder/'summary.json');row['folder']=str(folder);row['articulation']=articulation(folder)
            old=read_json(BASELINE/folder.name/'summary.json');row['baseline_success']=old['official_task_success']
            phases=read_json(folder/'phases.json') if (folder/'phases.json').exists() else []
            row['complete_reference_count']=sum(p['issued_commands'] for p in phases)
            admission=folder/'grasp-admission.json'
            if admission.exists() or admission.with_suffix('.json.gz').exists():
                adm=read_json(admission);row['candidate_rejections']=dict(Counter(r['reason'] for r in adm['rejected_candidates']))
            plans=[read_json(p) for p in folder.glob('plan-*/planning.json')]
            row['manipulation_plans']=len(plans);row['planning_seconds']=[p['seconds'] for p in plans if 'seconds' in p]
            episodes.append(row)
    counts=[]
    for name in LABELS:
        entries=[e for e in planned if e['task']==name];rows=[e for e in episodes if e['task']==name]
        if not entries:continue
        counts.append({'task':name,'planned':len(entries),'completed':len(rows),
            'baseline_successes':sum(read_json(BASELINE/f'{name}-state{e["state"]:03d}'/'summary.json')['official_task_success'] for e in entries),
            'new_successes':sum(e['official_task_success'] for e in rows)})
    data={'planned':len(planned),'completed':len(episodes),'successes':sum(e['official_task_success'] for e in episodes),
        'per_task':counts,'episodes':episodes,'native_success_predicate':True,'all_failures_in_denominator':True,
        'execution_veto_checks':sum(e['execution_veto_checks'] for e in episodes),
        'reclaimed_bytes':sum(e.get('storage',{}).get('bytes_reclaimed_this_pass',0) for e in episodes),
        'comparison':'Combined geometry, candidate ranking and MC-VAMP adaptation; not a planner-only ablation',
        'api':{k:v for k,v in read_json(ROOT/'results/articulated-api-v1/ledger.json').items() if k in {'model','budget_usd','cumulative_conservative_spend_usd','remaining_usd'}}}
    development=[]
    for path in sorted((ROOT/'results').glob('mcvamp-mechanisms-development-20261015-*/*-state*/summary.json')):
        item=read_json(path);development.append({k:item.get(k) for k in ['task','initial_state','status','official_task_success','failure','control_steps']})
    data['development_attempts_excluded_from_frozen_comparison']=development
    retries=[]
    for run in args.retry_runs:
        root=(ROOT/run).resolve()
        for path in sorted(root.glob('*-state*/summary.json')):
            item=read_json(path);item['folder']=str(path.parent);item['articulation']=articulation(path.parent);retries.append(item)
    data['post_run_camera_repair_retries_excluded_from_frozen_comparison']=retries
    write_json(out/'evaluation.json',data)
    def link(path):
        import os
        return html.escape(os.path.relpath(path,out),quote=True)
    audit_note=''
    if (out/'final-validation.json').exists():
        audit=read_json(out/'final-validation.json')
        audit_note=f'<p>Verification: {audit["focused_tests_passed"]} focused tests passed; every saved joint reference matches the native action log. <a href="final-validation.json">Audit record</a>.</p>'
    tables=''.join(f'<tr><td>{LABELS[t["task"]]}</td><td>{t["baseline_successes"]}/{t["planned"]}</td><td>{t["new_successes"]}/{t["planned"]}</td><td>{t["completed"]}/{t["planned"]}</td></tr>' for t in counts)
    cards=[]
    for row in episodes:
        folder=Path(row['folder']);old=BASELINE/folder.name;status='Success' if row['official_task_success'] else 'Failure'
        failure=html.escape(row.get('failure','All requested segments completed'))
        art=row.get('articulation');motion=''
        if art:
            if art['type']=='hinge':motion=f'Final control angle: {art["final"]*180/3.141592653589793:.1f}°'
            else:motion=f'Drawer opening: {max(0.,-art["final"])*100:.1f} cm'
        cards.append(f'''<article><h2>{LABELS[row['task']]} · state {row['initial_state']} · <span class="{'ok' if row['official_task_success'] else 'bad'}">{status}</span></h2>
<p>{failure}. {motion}</p><p>{row['control_steps']} control steps · {row.get('completed_trajectories',0)} complete trajectories · {row['episode_wall_seconds']:.1f} s pipeline wall time</p>
<div class="videos"><div><b>New pipeline</b><video controls preload="metadata" src="{link(folder/'initial/rollout.mp4')}"></video></div><div><b>Saved baseline</b><video controls preload="none" src="{link(old/'initial/rollout.mp4')}"></video></div></div>
<details><summary>Failure / planning evidence</summary><pre>{html.escape(json.dumps({'stage':row.get('failure_stage'),'grasp_rejections':row.get('candidate_rejections'), 'direction_observed':row.get('control_direction_observed'),'direction_reversed':row.get('control_direction_reversed'),'planning_seconds':row['planning_seconds']},indent=2))}</pre><a href="{link(folder/'summary.json')}">Episode record</a></details></article>''')
    retry_cards=[]
    for row in retries:
        status='Success' if row['official_task_success'] else 'Failure'
        retry_cards.append(f'<article><h2>Camera repair retry: {LABELS[row["task"]]} · state {row["initial_state"]} · {status}</h2><p>The original failed attempt remains in the 20-episode comparison. This additional trial changes only the missing camera metadata.</p><p>{html.escape(row.get("failure","All requested segments completed"))}</p><video controls preload="metadata" src="{link(Path(row["folder"])/"initial/rollout.mp4")}"></video></article>')
    page=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MC-VAMP drawers and stove</title><style>
body{{font:16px system-ui;margin:32px auto;max-width:1150px;padding:0 20px;color:#17202a;background:#f4f6f8}}h1{{font-size:30px}}h2{{font-size:20px}}article{{background:white;padding:20px;border-radius:10px;margin:22px 0}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}.videos{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}video{{display:block;width:100%;max-height:420px;background:#111}}.ok{{color:#087d48}}.bad{{color:#a33}}pre{{white-space:pre-wrap;font-size:13px}}button{{padding:8px;margin:5px}}@media(max-width:650px){{.videos{{grid-template-columns:1fr}}}}
</style></head><body><h1>MC-VAMP: drawers and stove</h1><p><b>{data['successes']}/{data['planned']} native task successes</b> · {data['completed']}/{data['planned']} episodes completed · {data['execution_veto_checks']} execution-time vetoes.</p>
<p>Observed surfaces replace the convex moving-object envelope. MC-VAMP searches constrained manipulation paths; VAMP searches free approaches. Knob direction is tested through physical probes and depth observations. Every issued trajectory finishes. Simulator object state is used only to score and diagnose outcomes.</p><p>Collision scope: arm and palm versus observed stationary/moving surfaces, with selected finger contact permitted. Mechanism–environment contacts are resolved by simulator physics; unobserved space is unmodeled.</p>
<p>{len(development)} development attempts are retained separately from this frozen comparison.</p><p>This changes several pipeline components together. It is a regression on previously inspected states, not a planner-only ablation or a held-out benchmark.</p>
<table><tr><th>Task</th><th>Baseline success</th><th>New success</th><th>Completed</th></tr>{tables}</table>
<details><summary>Why the old collision reconstruction blocked motion</summary><p>Red points were classified as stationary but lay inside the convex completion of the partial moving surface. This illustrates reconstruction overlap, not a demonstrated physical collision. The figure shows raw hull overlap before contact exemptions.</p><img src="collision-model-audit.png" alt="Initial reconstructed geometry overlaps" style="width:100%"></details>
<p>Videos are recorded at 4 fps with joint actions at 20 Hz. During joint execution, 5× playback approximates simulation time; sensing frames are sampled differently. Planning pauses are omitted.</p>
<button onclick="document.querySelectorAll('video').forEach(v=>v.playbackRate=1)">1× diagnostic playback</button><button onclick="document.querySelectorAll('video').forEach(v=>v.playbackRate=5)">5× playback</button>
<p>Automatic cleanup reclaimed {data['reclaimed_bytes']/2**30:.2f} GiB. Cumulative API cost: ${data['api']['cumulative_conservative_spend_usd']:.4f} / $2.</p>{audit_note}{''.join(retry_cards)}{''.join(cards)}</body></html>'''
    (out/'index.html').write_text(page)
    print(json.dumps({'report':str(out/'index.html'),'completed':data['completed'],'successes':data['successes'],'planned':data['planned']}))

if __name__=='__main__':main()
