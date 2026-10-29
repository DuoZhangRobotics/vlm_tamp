"""Live, local report for restriction fixes, reference tracking and recovery."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
from report_contact_skills import read,timing_table,skill_card

ROOT=Path(__file__).resolve().parents[1]


def profile_run(run,events,actor_wall):
    operations=Counter();issues=Counter();ik=Counter();grasp=Counter();contact=Counter();arcs=Counter()
    for event in events:
        operations[event['request']['operation']]+=event.get('wall_seconds',0.)
        result=event.get('response',{}).get('result',{})
        if event['request']['operation'] in {'push','hook_pull'}:contact.update(result.get('timing_seconds',{}))
        for stage in result.get('stages',[]):
            if 'planning_seconds' in stage:arcs['post_closure_arc_planning_seconds']+=stage['planning_seconds']
    for file in run.glob('attempt*/*-issue-timing.json*'):
        data=read(file,{})
        issues.update(data.get('phases',{}))
    for file in run.glob('attempt*/curobo-queries.json*'):
        for row in read(file,[]):
            for name in ['wall_seconds','worker_rpc_seconds','cpu_postprocessing_seconds','elapsed_seconds']:
                if name in row:ik[name]+=row[name]
    for file in run.glob('attempt*/*/selection/selected.json*'):
        grasp.update(read(file,{}).get('search_timing_seconds',{}))
    wall=sum(operations.values())
    return {'actor_wall_seconds':actor_wall,'service_wall_seconds':wall,
        'outside_service_seconds':max(0,actor_wall-wall) if actor_wall is not None else None,
        'outside_service_scope':'Luna calls, transport and local analysis; not isolated model reasoning time',
        'service_seconds_by_operation':dict(operations),'nested_execution_and_observation_seconds':dict(issues),
        'nested_full_pose_ik_seconds':dict(ik),'nested_grasp_search_seconds':dict(grasp),
        'nested_contact_skill_seconds':dict(contact),'nested_sequence_planning_seconds':dict(arcs),
        'nested_sections_overlap_do_not_sum':True}


def esc(x):return html.escape(str(x))


def link(path,base,label):return f'<a href="{esc(os.path.relpath(path,base))}">{esc(label)}</a>'


def render(study,embed=True):
    study=Path(study).resolve();study.mkdir(parents=True,exist_ok=True)
    body='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Luna recovery: restrictions and control budgets</title><style>
body{font:16px system-ui;color:#172b43;background:#f3f6fa;max-width:1180px;margin:24px auto;padding:0 18px}
article{background:white;border:1px solid #c9d6e4;padding:18px;margin:18px 0;border-radius:10px}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #dae3ed}
pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto;font-size:12px}
img,video{max-width:100%;max-height:480px}a{color:#1264a2}.note{background:#e8f1fc;padding:12px}button{padding:7px 14px}</style>
<h1>Luna recovery: restriction fixes and control budgets</h1><p class="note">Zero added collision padding. Physical contacts remain enabled. Native task success, contact retention and numerical feasibility are reported separately.</p>
<button onclick="location.reload()">Refresh</button>'''
    status=read(study/'implementation-status.json',{})
    body+='<article><h2>Implementation and validation</h2><p>'+esc(', '.join(f'{x["id"]}: {x["status"]}' for x in status.get('steps',[])))+'</p>'
    body+=link(ROOT/'docs/restriction_audit_20261029.md',study,'Original audit and plan')+' · '+link(study/'implementation-status.json',study,'Current work record')+'</article>'
    baseline=read(study/'trials/budget-only-close49-1200-r1/evaluation-private.json',{}).get('attempts',[])
    if baseline:
        a=baseline[0]
        body+=f'<article><h2>Budget-only comparison</h2><p>With the previous numerical policies and frozen memory, closing state 49 reached native success at control <strong>{a.get("first_native_success_control_step",a.get("control_steps"))}</strong> under a 1,200-control limit. The prior 600-control evaluation failed. This new actor knew its larger budget; it is not a claim that its decisions were identical to the prior run.</p></article>'
    trial_rows=[];cards=[]
    for run in sorted((study/'trials').glob('*')) if (study/'trials').exists() else []:
        if not run.is_dir():continue
        protocol=read(run/'protocol.json',{});agent=read(run/'agent-exit.json',{})
        evaluation=read(run/'evaluation-private.json',{}).get('attempts',[])
        interruption=read(run/'operator-interruption.json',{})
        status_row=read(run/'live-episode-status.json',{})
        events=[]
        try:
            for line in (run/'tool-events.jsonl').read_text().splitlines():
                try:events.append(json.loads(line))
                except ValueError:pass
        except OSError:pass
        service_seconds=sum(e.get('wall_seconds',0) for e in events)
        wall=agent.get('wall_seconds');calls=Counter(e['request']['operation'] for e in events)
        profiling=profile_run(run,events,wall)
        (run/'timing-breakdown.json').write_text(json.dumps(profiling,indent=2)+'\n')
        if evaluation:
            for a in evaluation:
                trial_rows.append((run.name,str(a.get('task',protocol.get('case',{}).get('task'))),
                    a.get('maximum_control_steps',protocol.get('maximum_control_steps_per_attempt')),a.get('official_task_success'),
                    a.get('control_steps'),a.get('success_by_600_controls'),wall,service_seconds))
        else:trial_rows.append((run.name,protocol.get('case',{}).get('task','starting'),protocol.get('maximum_control_steps_per_attempt'),
            'interrupted' if interruption else 'running',status_row.get('control_steps'),None,wall,service_seconds))
        card=f'<article><h2>{esc(run.name)}</h2><p>Tools selected: {esc(dict(calls))}.</p>'
        if interruption:card+='<p>Infrastructure/implementation interruption: '+esc(interruption.get('reason'))+'. Excluded from task-success counts.</p>'
        card+='<p>'+link(run/'tool-events.jsonl',study,'Tool requests and responses')+' · '+link(run/'agent-events.jsonl',study,'Luna input/output')+' · '+link(run/'agent-prompt.txt',study,'Initial prompt')+'</p>'
        for movie in sorted(run.glob('attempt*/initial/rollout.mp4')):
            card+=f'<details><summary>{esc(movie.parent.parent.name)} video</summary><video controls preload="none" src="{esc(os.path.relpath(movie,study))}"></video></details>'
        if wall is not None:card+=f'<p>Actor wall: {wall:.1f} s; robot services: {service_seconds:.1f} s; outside services: {max(0,wall-service_seconds):.1f} s. Outside includes model, transport and local analysis.</p>'
        else:card+=f'<p>Current controls: {esc(status_row.get("control_steps"))}; remaining: {esc(status_row.get("remaining_control_steps"))}; last operation: {esc(status_row.get("last_completed_operation"))}.</p>'
        card+='<details open><summary>Time by robot service (inclusive)</summary>'+timing_table(profiling['service_seconds_by_operation'])+'</details>'
        card+='<details><summary>Execution, observation, grasp search and IK breakdown</summary><p>These are nested measurements inside service time. Sections overlap; do not add them together. Missing measurements are not zero-cost work.</p>'
        for key in ['nested_execution_and_observation_seconds','nested_grasp_search_seconds','nested_full_pose_ik_seconds','nested_contact_skill_seconds','nested_sequence_planning_seconds']:
            card+='<h3>'+esc(key.replace('_',' '))+'</h3>'+timing_table(profiling[key])
        card+='</details>'+link(run/'timing-breakdown.json',study,'Timing JSON')
        for event in events:
            if event['request']['operation'] in {'push','hook_pull'}:
                result=event.get('response',{}).get('result',{})
                card+='<details><summary>'+esc(f"{event['request']['operation']} event {event['sequence']}: {result.get('stop_reason','see log')}")+'</summary>'
                card+=skill_card(f'Event {event["sequence"]}',result,run/'tool-events.jsonl',study,{})+'</details>'
        cards.append(card+'</article>')
    if trial_rows:
        body+='<article><h2>Luna evaluations</h2><table><tr><th>Condition</th><th>Task</th><th>Budget</th><th>Native success</th><th>Controls</th><th>By 600</th><th>Actor wall</th><th>Services</th></tr>'
        for name,task,budget,success,steps,by600,wall,svc in trial_rows:
            body+='<tr>'+''.join(f'<td>{esc(v)}</td>' for v in [name,task,budget,success,steps,by600,f'{wall:.1f} s' if wall is not None else 'running',f'{svc:.1f} s'])+'</tr>'
        body+='</table></article>'
    retimes=[];sequences=[]
    for file in sorted(study.glob('*/result.json')):
        r=read(file,{})
        if r.get('mode')=='retime':retimes.append((file.parent,r))
        elif r.get('mode')=='sequence':sequences.append((file.parent,r))
    if retimes:
        body+='<article><h2>Identical approach path: actual physics comparison</h2><p>The historical prefix is replayed under its original adapter. Only the tested approach timing/reference adapter changes; closure and the subsequent recorded motion are retained.</p><table><tr><th>Run</th><th>Reference adapter</th><th>Requested rad/s</th><th>Approach controls</th><th>Endpoint position error</th><th>Orientation error</th><th>Gap after closing</th></tr>'
        for folder,r in retimes:
            body+='<tr>'+''.join(f'<td>{esc(x)}</td>' for x in [folder.name,r.get('tested_reference_mode','measured_delta'),r.get('requested_approach_speed_rad_s'),r.get('approach_controls'),f'{1000*r["approach_position_error_m"]:.2f} mm' if r.get('approach_position_error_m') is not None else r.get('error'),f'{r["approach_orientation_error_deg"]:.2f}°' if r.get('approach_orientation_error_deg') is not None else '',r.get('post_closure_grasp_evidence',{}).get('pad_gap_mm')])+'</tr>'
        body+='</table><p>Reference speed is distinct from actual joint speed. The old measured-relative 0.05-rad lead cap caused large tracking lag; the corrected adapter limits consecutive commanded references and uses the same official PD controller and gains.</p>'
        plot=study/'reference-tracking-comparison.png'
        if plot.exists():body+=f'<img loading="lazy" src="{plot.name}" alt="Requested versus actual tracking under the two reference adapters">'
        body+='</article>'
    for folder,r in sequences:
        e=r.get('execution',{});plan=r.get('plan',{});budget=plan.get('budget_timing_assumptions',{})
        body+=f'<article><h2>Physical sequence: {esc(folder.name)}</h2><p>Native success: {esc(r.get("native_success_seen"))}; controls: {esc(r.get("control_steps"))}; stop: <strong>{esc(e.get("stop_reason",r.get("error")))}</strong>.</p>'
        body+=f'<p>Approach: {esc(budget.get("approach_controls"))} controls. Replanned after closure: {esc(e.get("post_closure_replanning"))}. Joint revisions: {sum(bool(x.get("applied")) for x in e.get("joint_feedback",[]))}.</p>'
        movie=folder/'service/attempt01/initial/rollout.mp4'
        if movie.exists():body+=f'<video controls preload="none" src="{esc(os.path.relpath(movie,study))}"></video>'
        body+='<details><summary>Planner, measured closure, joint feedback and recovery</summary><pre>'+esc(json.dumps({'plan':plan,'execution':e},indent=2))+'</pre></details>'+link(folder/'result.json',study,'Complete diagnostic')+'</article>'
    body+=''.join(cards)+'</body>'
    (study/'watch.html').write_text(body)
    if embed:
        watch=ROOT/'results/luna-live-hinge-comparison-20261027/watch.html'
        if watch.exists():
            text=watch.read_text();begin='<!-- RESTRICTION_FIXES_BEGIN -->';end='<!-- RESTRICTION_FIXES_END -->'
            block=begin+f'<section id="restriction-fixes"><h2>Current recovery and planner tests</h2><p>{link(study/"watch.html",watch.parent,"Open current results, timings and videos")}</p><iframe src="{esc(os.path.relpath(study/"watch.html",watch.parent))}" title="Restriction fixes" style="width:100%;height:1100px;border:1px solid #cad5e0"></iframe></section>'+end
            if begin in text:text=text[:text.index(begin)]+block+text[text.index(end)+len(end):]
            else:text=text.replace('</body>',block+'</body>')
            watch.write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',default=str(ROOT/'results/restriction-fixes-20261029'))
    a=p.parse_args();render(a.study)
