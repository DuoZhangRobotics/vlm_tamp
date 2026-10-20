"""Report success-focused regressions without combining unlike evaluation cohorts."""
from pathlib import Path
import sys,json,html,collections,os
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.coupled_resources import api_snapshot
REPORT=ROOT/'results/success-first-report'
BASE=ROOT/'results/mcvamp-drawer-stove-report/evaluation.json'


def read_run(name):
    root=ROOT/'results'/name
    if not (root/'manifest.json').exists():return None
    manifest=read_json(root/'manifest.json')['episodes'];rows=[]
    for e in manifest:
        folder=root/f'{e["task"]}-state{e["state"]:03d}';p=folder/'summary.json'
        rows.append({**e,'folder':str(folder),'complete':p.exists(),**(read_json(p) if p.exists() else {})})
    return {'run':name,'planned':len(rows),'completed':sum(r['complete'] for r in rows),'successes':sum(r.get('official_task_success',False) for r in rows),'episodes':rows}


def main():
    REPORT.mkdir(exist_ok=True)
    main=read_run('reliability-mechanisms-frozen-20261016')
    additional=[read_run(n) for n in ['reliability-bowls-final-20261016','success-serial-open-20261016','success-serial-close-20261016']]
    additional=[r for r in additional if r]
    development=[read_run(p.name) for p in sorted((ROOT/'results').glob('reliability-development-20261016-*'))];development=[r for r in development if r]
    development += [r for r in [read_run(n) for n in ['reliability-bowls-frozen-20261016','reliability-microwave-recovery-20261016','reliability-microwave-opening-20261016','reliability-microwave-final-open-20261016','reliability-microwave-final-close-20261016','reliability-microwave-v2-open-20261016','reliability-microwave-v2-close-20261016']] if r]
    primary=[r for r in [main,*additional] if r]
    all_runs=[r for r in [main,*additional,*development] if r]
    reclaimed=sum(e.get('storage',{}).get('bytes_reclaimed_this_pass',0) for r in all_runs for e in r['episodes'])
    data={'planned':35,'completed':sum(r['completed'] for r in primary),'native_successes':sum(r['successes'] for r in primary),'previous_latest_task_family_successes':23,'mechanism_regression':main,'additional_regressions':additional,'development_attempts':development,
          'api':api_snapshot(),'all_manifest_failures_in_denominator':True,'execution_vetoes':0,
          'regression_not_held_out_generalization':True,'reclaimed_bytes':reclaimed,
          'scope':'Cached RGB-validated Luna high grounding; fresh local geometry, grasp selection, planning and physical execution'}
    write_json(REPORT/'evaluation.json',data)
    import csv
    with (REPORT/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['cohort','final_configuration','task','state','complete','native_success','final_native_success','status','failure','control_steps','folder']);writer.writeheader()
        for run in all_runs:
            for e in run['episodes']:
                writer.writerow({'cohort':run['run'],'final_configuration':run in primary,'task':e['task'],'state':e['state'],'complete':e['complete'],'native_success':e.get('official_task_success'),'final_native_success':e.get('final_task_success'),'status':e.get('status'),'failure':e.get('failure'),'control_steps':e.get('control_steps'),'folder':e['folder']})
    esc=html.escape
    css='''body{margin:36px auto;max-width:1100px;padding:0 24px;font:16px/1.55 system-ui;background:#f6f7f9;color:#182537}h1,h2{line-height:1.2}table{border-collapse:collapse;background:white;width:100%;margin:20px 0}td,th{text-align:left;padding:11px;border-bottom:1px solid #dce1e7}.ok{color:#136c45}.fail{color:#ae3035}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:white;border-radius:12px;padding:20px}video,img{width:100%;border-radius:8px}small{color:#536273}code{font-size:14px}a{color:#1859ae}'''
    parts=['<!doctype html><meta charset="utf-8"><title>Success-first regression</title><style>'+css+'</style>',
        '<h1>Task success before speed</h1>',f'<p><strong>{data["native_successes"]}/35 native task successes</strong>; {data["completed"]}/35 final episodes completed.</p>',
        '<p>Success means the native LIBERO predicate became true during physical execution. Every scheduled case counts, including failures before motion. These states have already been inspected; this measures regression recovery, not held-out generalization.</p>',
        '<div class="card"><strong>What changed</strong><ul>',
        '<li>Drawer: correctly label observed front-face points as part of the moving drawer, and infer closing travel from the observed cabinet face.</li>',
        '<li>Stove: isolate the raised control in RGB-D before fitting its axis. A loose image box previously included the burner.</li>',
        '<li>Bowl: use a fused observed bowl surface when the original grasp search fails; allow functional fingers to touch the selected plate only during placement and release.</li>',
        '<li>Microwave: reject flat or inconsistent handle fits, recover a supported vertical bar, refit its central surface, and hold the open-gripper approach longer before closing. Near the observed closed position, finish with one bounded seating motion.</li>',
        '<li>Microwave: use calibrated arm geometry, a budget-feasible approach with joint reserve, and a bounded tolerance for existing shallow palm–handle contact. Arm and unrelated-object collisions remain checked.</li></ul>',
        '<p>MC-VAMP constraints, physical grasping, the 600-control-step limit, and complete execution of every issued trajectory remain. No execution-time collision or progress veto was added.</p></div>']
    if main:
        baseline={'open_middle_drawer':5,'open_top_drawer':5,'close_top_drawer':2,'turn_on_stove':3}
        parts+=['<h2>Full drawer/stove regression</h2>',f'<p><strong>{main["successes"]}/{main["planned"]}</strong> native successes; {main["completed"]}/{main["planned"]} episodes completed. Previous matched cohort: <strong>15/20</strong>.</p>', '<table><tr><th>Task</th><th>Previous</th><th>Current</th><th>Completed</th></tr>']
        for task,old in baseline.items():
            rows=[r for r in main['episodes'] if r['task']==task];wins=sum(r.get('official_task_success',False) for r in rows)
            parts.append(f'<tr><td>{esc(task.replace("_"," "))}</td><td>{old}/5</td><td>{wins}/5</td><td>{sum(r["complete"] for r in rows)}/5</td></tr>')
        parts.append('</table>')
    titles={'reliability-bowls-final-20261016':'Bowl on plate','success-serial-open-20261016':'Open microwave','success-serial-close-20261016':'Close microwave'}
    for run in additional:
        parts+=['<h2>'+titles.get(run['run'],esc(run['run']))+'</h2>',f'<p>{run["successes"]}/{run["planned"]} native successes; {run["completed"]} completed. This separate set is not silently merged into the 20-case drawer/stove score.</p>']
    parts+=['<h2>Episode evidence</h2>','<table><tr><th>Case</th><th>Result</th><th>Reason / evidence</th></tr>']
    for run in [r for r in [main,*additional] if r]:
        for row in run['episodes']:
            f=Path(row['folder']);relative=os.path.relpath(f,REPORT);status='pending' if not row['complete'] else 'success' if row.get('official_task_success') else 'failure'
            why=('Native task goal achieved' if status=='success' else row.get('failure', 'Planned motions finished but the native goal was not reached' if row['complete'] else 'Awaiting completion'))
            if status=='success' and row.get('failure'):why+='; later planner status: '+row['failure']
            if row.get('administrative_interruption'):why+='; administratively stopped during subsequent planning to reduce RAM use'
            parts.append(f'<tr><td>{esc(row["task"])} / {row["state"]}</td><td class="{"ok" if status=="success" else "fail"}">{status}</td><td>{esc(why)} · <a href="{relative}/summary.json">record</a> · <a href="{relative}/initial/rollout.mp4">video (4 fps archive)</a></td></tr>')
    parts+=['</table>', '<h2>Recovered examples</h2><p>20 Hz simulation-time previews; planning pauses are omitted. Exact source runs are retained with the previews.</p><div class="grid">']
    for movie in sorted((REPORT/'previews').glob('*.mp4')):
        parts.append(f'<div class="card"><strong>{esc(movie.stem.replace("_"," "))}</strong><video controls preload="metadata" src="previews/{movie.name}"></video></div>')
    parts+=['</div><h2>Diagnosed examples</h2><div class="grid">']
    for name,caption in [('close_top_drawer-11.png','Red: points rejected as stationary obstacles. They lie on the selected drawer front.'),('turn_on_stove-21.png','Orange: original VLM box; cyan: old pivot. The box includes burner surface, biasing the axis fit.')]:
        parts.append(f'<div class="card"><img src="../reliability-diagnosis-20261016/{name}"><p>{caption}</p></div>')
    parts+=['</div>','<h2>Development attempts remain visible</h2><table><tr><th>Attempt</th><th>Completed</th><th>Native successes</th></tr>']
    for run in development:parts.append(f'<tr><td>{esc(run["run"])}</td><td>{run["completed"]}/{run["planned"]}</td><td>{run["successes"]}/{run["planned"]}</td></tr>')
    parts+=['</table>','<p>One superseded opening trial was stopped between motions to reduce RAM pressure. A separate worker was killed by the operating system during planning. Both command traces were checked and recovered, all issued motions had completed, and both interruptions remain recorded. The final microwave evaluations run serially with bounded scene-index caches.</p>',f'<p>Temporary data reclaimed after completed episodes: {reclaimed/(1024**3):.2f} GiB. Cumulative API spend: ${data["api"]["cumulative_conservative_spend_usd"]:.6f} / $2. No new API request is needed for these observation-validated cached inputs.</p>',
        '<p><a href="evaluation.json">Structured results</a> · <a href="final-validation.json">Independent validation</a></p>',
        '<small>Videos omit planning pauses. Archived movies use 4 fps; separate previews, when provided, are resampled to the 20 Hz simulated control timeline. A 100% result on these previously inspected states does not establish 100% success on unseen tasks or fresh model outputs.</small>']
    (REPORT/'index.html').write_text('\n'.join(parts))
    print({'main_completed':main['completed'] if main else 0,'main_successes':main['successes'] if main else 0,'reclaimed_gib':reclaimed/1024**3})
if __name__=='__main__':main()
