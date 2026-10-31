"""Live, local report for restriction fixes, reference tracking and recovery."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
from report_contact_skills import read,timing_table,skill_card,read_events,file_stamp

ROOT=Path(__file__).resolve().parents[1]


def cached_profile_run(run,events,actor_wall):
    """Timing artifacts are complete before the corresponding event is appended."""
    source={'events':file_stamp(run/'tool-events.jsonl'),
            'actor_exit':file_stamp(run/'agent-exit.json'),'profile_version':2}
    source=json.loads(json.dumps(source))
    path=run/'timing-breakdown.json';previous=read(path,{})
    if previous.get('_profile_source')==source:return previous
    profiling=profile_run(run,events,actor_wall)
    profiling['_profile_source']=source
    path.write_text(json.dumps(profiling,indent=2)+'\n')
    return profiling


def profile_run(run,events,actor_wall):
    operations=Counter();issues=Counter();ik=Counter();grasp=Counter();contact=Counter();arcs=Counter()
    for event in events:
        operations[event['request']['operation']]+=event.get('wall_seconds',0.)
        result=event.get('response',{}).get('result',{})
        if event['request']['operation'] in {'push','continue_push','push_stroke','hook_pull'}:contact.update(result.get('timing_seconds',{}))
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
    grasp_phases=Counter()
    for file in run.glob('attempt*/*-grasp/profiling.json*'):
        grasp_phases.update(read(file,{}).get('phases',{}))
    arc_details=Counter()
    def arc_timing(result):
        if result.get('segments'):
            for segment in result['segments']:arc_timing(segment)
            return
        for key in ['goal_generation_seconds','native_connection_seconds','total_seconds']:
            if key in result:arc_details[key]+=result[key]
        for key in ['scene_setup_seconds','gpu_ik_seconds','equality_polish_seconds','fallback_setup_seconds']:
            if key in result.get('curobo_v2',{}):arc_details[key]+=result['curobo_v2'][key]
    for pattern in ['attempt*/*-arc/result.json*','attempt*/*-measured-grasp-plan/result.json*']:
        for file in run.glob(pattern):arc_timing(read(file,{}))
    wall=sum(operations.values())
    return {'actor_wall_seconds':actor_wall,'service_wall_seconds':wall,
        'outside_service_seconds':max(0,actor_wall-wall) if actor_wall is not None else None,
        'outside_service_scope':'Luna calls, transport and local analysis; not isolated model reasoning time',
        'service_seconds_by_operation':dict(operations),'nested_execution_and_observation_seconds':dict(issues),
        'nested_full_pose_ik_seconds':dict(ik),'nested_grasp_search_seconds':dict(grasp),
        'nested_grasp_generation_and_selection_seconds':dict(grasp_phases),
        'nested_arc_goal_ik_and_connection_seconds':dict(arc_details),
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
    if status.get('title'):
        body=body.replace('Luna recovery: restrictions and control budgets',esc(status['title']))
        body=body.replace('Luna recovery: restriction fixes and control budgets',esc(status['title']))
    body+='<article><h2>Implementation and validation</h2><p>'+esc(', '.join(f'{x["id"]}: {x["status"]}'.replace('_',' ') for x in status.get('steps',[])))+'</p>'
    body+=link(ROOT/'docs/restriction_audit_20261029.md',study,'Original audit and plan')+' · '+link(study/'implementation-status.json',study,'Current work record')+'</article>'
    plan=read(study/'varied-practice-plan.json',{})
    if plan.get('clean_start'):
        body+='<article><h2>Clean practice and memory comparison</h2><p>'+esc(f"One learner starts without inherited task memories or worked examples. {plan['initial_and_maximum_attempts']} practice episodes; {plan['control_steps']} controls per episode. Generic tool documentation remains available.")+'</p>'
        body+='<p>Fresh evaluation actors may write their own lessons and notebooks. Those files are archived; they are not fed to later actors in this comparison. Each memory condition receives either no prior notes or the same frozen practice guidelines.</p>'
        body+=link(study/'varied-practice-plan.json',study,'Declared experiment')+' · '+link(ROOT/'docs/clean_start_600_20261029.md',study,'Results and interpretation')+'</article>'
        for file in sorted(study.glob('*-practice-manifest.json')):
            experiment=read(file,{})
            if not experiment.get('plan',{}).get('clean_start'):continue
            rows=experiment.get('evaluations',[])
            expected=len(plan['later_evaluation_cases'])*len(plan['evaluation_memory_conditions'])
            body+='<article><h2>Fresh evaluations: memory versus no memory</h2><p>'+esc(f"{len(rows)}/{expected} episodes complete. Same code, initial states and control limits; a fresh actor for each case/condition. Condition order alternates between cases. One rollout per condition does not establish a general success rate.")+'</p>'
            body+='<table><tr><th>Case / logs</th><th>Memory</th><th>Native outcome</th><th>Controls</th><th>Total (s)</th><th>Services (s)</th><th>Outside (s)</th></tr>'
            for row in rows:
                timing=read(study/'trials'/row['run']/'timing-breakdown.json',{})
                total=row.get('actor_wall_seconds');services=timing.get('service_wall_seconds')
                outside=max(0,total-services) if total is not None and services is not None else None
                label=f"{row['task'].replace('_',' ')} / {row['state_index']}"
                body+=f'<tr><td><a href="#{esc(row["run"])}">{esc(label)}</a></td>'
                values=['None' if row['memory_condition']=='none' else 'Frozen practice notes',
                    'Success' if row['official_task_success'] else 'Failed',row['control_steps'],
                    *[f'{value:.1f}' if value is not None else 'unavailable' for value in [total,services,outside]]]
                body+=''.join('<td>'+esc(value)+'</td>' for value in values)+'</tr>'
            body+='</table><p>Outside-service time includes model calls, transport and local analysis.</p>'+link(file,study,'Complete comparison manifest')+'</article>'
    context=read(study/'context-size-audit.json',{})
    if context:
        body+='<article><h2>Memory size and conversation history</h2><p>'+esc(context['summary'])+'</p>'
        body+=link(study/'context-size-audit.json',study,'Measured sizes, context counters and sources')+'</article>'
    if (study/'hardware-conditions.json').exists():
        body+='<p>GPU 3 was shared with another workload during the later tests. Wall time includes the recorded hardware conditions. '+link(study/'hardware-conditions.json',study,'Hardware observation')+'</p>'
    final=read(study/'final-evaluation-summary.json',{})
    if final:
        body+='<article><h2>Final frozen evaluation</h2><p>'+esc(f"{final['completed_cases']}/{final['planned_cases']} cases complete; runtime commit {final['code_commit']}. One fresh Luna actor per case, with the same frozen learner memory.")+'</p>'
        counts=[]
        for group,label in [('development_regression','Development cases'),('additional_case','Additional cases excluded from this practice')]:
            value=final.get(group,{})
            counts.append(f"{label}: {value.get('native_successes',0)}/{value.get('completed',0)} native successes")
        body+='<p>'+esc('; '.join(counts))+'. These small counts do not establish a general success rate or isolate the effect of memory.</p>'
        body+='<table><tr><th>Case / logs</th><th>Group</th><th>Outcome</th><th>Controls</th><th>Total (s)</th><th>Services (s)</th><th>Outside (s)</th><th>Success stage</th></tr>'
        for row in final['rows']:
            label=f"{row['task'].replace('_',' ')} / {row['state_index']}"
            body+=f'<tr><td><a href="#{esc(row["run"])}">{esc(label)}</a></td>'
            values=['Development' if row['group']=='development_regression' else 'Additional',
                'Success' if row['native_success'] else 'Failed',row['controls'],
                f"{row['actor_wall_seconds']:.1f}",f"{row['service_seconds']:.1f}",
                f"{row['outside_service_seconds']:.1f}",row.get('success_stage') or '—']
            body+=''.join('<td>'+esc(value)+'</td>' for value in values)+'</tr>'
        body+='</table><p>Outside-service time includes Luna calls, transport and local analysis. Click a case for video, input/output and detailed profiling. A success during free approach does not validate the requested contact stroke or release.</p>'
        body+=link(study/'final-evaluation-summary.json',study,'Evaluation summary and provenance')+'</article>'
    plot=study/'task-progress-comparison.png'
    if plot.exists():body+=f'<article><h2>Observed task outcomes</h2><p>Offline simulator grading; these native angles were not actor or planner inputs. Each line is a separate rollout.</p><img src="{plot.name}" alt="Door opening versus controls across evaluation conditions"></article>'
    qualification=read(study/'qualification-summary.json',{})
    if qualification:
        body+='<article><h2>Local contact repeatability</h2><p>'+esc(qualification['scope'])+'</p><table><tr><th>Mode</th><th>Requested speed</th><th>Native closure</th><th>Retained closure</th><th>Contact-free coasts</th><th>Measured stroke peak*</th></tr>'
        for row in qualification['groups']:
            n=row['replays']
            if row['skill']=='hook_pull':continue
            body+='<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [row['mode'],f"{100*row['requested_contact_speed_m_s']:.0f} cm/s",f"{row['native_successes']}/{n}",f"{row['retained_native_successes']}/{n}",f"{row['all_environment_contact_free_coasts']}/{n}",f"{100*row['mean_peak_stroke_speed_m_s']:.2f} cm/s"])+'</tr>'
        body+='</table><p>*Mean of recorded stroke peaks. Controlled-mode data includes later ineffective strokes after closure.</p>'
        for row in qualification['groups']:
            if row['skill']=='hook_pull':body+=f'<p>Hook-pull motion: {row["skill_motion_goals_reached"]}/{row["replays"]}; measured {abs(row["observed_joint_degrees"][0]):.2f}° for a 3° request, minimum gripper joint span {1000*row["minimum_gripper_joint_span_m"]:.1f} mm.</p>'
        body+='<p>'+esc(qualification['contact_scope_note'])+'</p>'+link(study/'qualification-summary.json',study,'Measurements and provenance')+'</article>'
    retention=read(study/'retention-summary.json',{})
    if retention:
        body+='<article><h2>Success after withdrawal and coasting</h2><p>'+esc(retention['scope'])+'</p><table><tr><th>Withdrawal</th><th>Retained closure</th><th>No microwave contact</th><th>No environment contact</th><th>Peak remaining force</th></tr>'
        for row in retention['rows']:
            body+='<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [f"{100*row['withdrawal_m']:.1f} cm",row['retained_native_success'],row['microwave_contact_free_coast'],row['all_environment_contact_free_coast'],f"{row['peak_coast_force_n']:.3f} N"])+'</tr>'
        body+='</table>'+link(study/'retention-summary.json',study,'Full contact evidence')+'</article>'
    memory=read(study/'learner-memory-summary.json',{})
    if memory:
        body+='<article><h2>Frozen learner memory</h2><p>'+esc(memory.get('summary','Learner-authored procedural notes; geometry is re-estimated in each episode.'))+'</p>'
        if memory.get('memory_file'):body+=link(Path(memory['memory_file']),study,'Read the frozen guidelines')+' · '
        body+=link(study/'learner-memory-summary.json',study,'Practice outcomes, memory hash and provenance')+'</article>'
    for folder in sorted(study.glob('contact-requalification*')):
        if folder.is_dir() and (folder/'watch.html').exists():body+='<p>'+link(folder/'watch.html',study,'Contact replay details: '+folder.name)+'</p>'
    baseline=read(study/'trials/budget-only-close49-1200-r1/evaluation-private.json',{}).get('attempts',[])
    if baseline:
        a=baseline[0]
        body+=f'<article><h2>Budget-only comparison</h2><p>With the previous numerical policies and frozen memory, closing state 49 reached native success at control <strong>{a.get("first_native_success_control_step",a.get("control_steps"))}</strong> under a 1,200-control limit. The prior 600-control evaluation failed. This new actor knew its larger budget; it is not a claim that its decisions were identical to the prior run.</p></article>'
    trial_rows=[];practice_rows=[];cards=[]
    for run in sorted((study/'trials').glob('*')) if (study/'trials').exists() else []:
        if not run.is_dir():continue
        protocol=read(run/'protocol.json',{});agent=read(run/'agent-exit.json',{})
        evaluation=read(run/'evaluation-private.json',{}).get('attempts',[])
        interruption=read(run/'operator-interruption.json',{})
        status_row=read(run/'live-episode-status.json',{})
        events=read_events(run/'tool-events.jsonl')
        service_seconds=sum(e.get('wall_seconds',0) for e in events)
        wall=agent.get('wall_seconds');calls=Counter(e['request']['operation'] for e in events)
        profiling=cached_profile_run(run,events,wall)
        if evaluation:
            for a in evaluation:
                if protocol.get('learning_practice'):
                    practice_rows.append((f'{run.name} / attempt {a["attempt"]}',f'{a.get("task")} / state {a.get("state_index")}',
                        a.get('official_task_success'),a.get('control_steps'),a.get('attempt_wall_seconds')))
                    continue
                trial_rows.append((run.name,str(a.get('task',protocol.get('case',{}).get('task'))),
                    a.get('maximum_control_steps',protocol.get('maximum_control_steps_per_attempt')),a.get('official_task_success'),
                    a.get('control_steps'),a.get('success_by_600_controls'),wall,service_seconds))
        elif not protocol.get('learning_practice'):trial_rows.append((run.name,protocol.get('case',{}).get('task','starting'),protocol.get('maximum_control_steps_per_attempt'),
            'interrupted' if interruption else 'running',status_row.get('control_steps'),None,wall,service_seconds))
        card=f'<article id="{esc(run.name)}"><h2>{esc(run.name)}</h2><p>Tools selected: {esc(dict(calls))}.</p>'
        skills=[e for e in events if e['request']['operation'] in {'push','continue_push','push_stroke','hook_pull'}]
        card+=f'<p>Contact skill selections: {len(skills)}; previews: {sum(bool(e["request"].get("arguments",{}).get("preview")) for e in skills)}; calls with physical motion: {sum(bool(e.get("response",{}).get("result",{}).get("robot_motion_executed")) for e in skills)}. Availability, selection, execution and native task success are separate measurements.</p>'
        if interruption:card+='<p>Infrastructure/implementation interruption: '+esc(interruption.get('reason'))+'. Excluded from task-success counts.</p>'
        card+='<p>'+link(run/'tool-events.jsonl',study,'Tool requests and responses')+' · '+link(run/'agent-events.jsonl',study,'Luna input/output')+' · '+link(run/'agent-prompt.txt',study,'Initial prompt')+'</p>'
        for movie in sorted(run.glob('attempt*/initial/rollout.mp4')):
            card+=f'<details><summary>{esc(movie.parent.parent.name)} video</summary><video controls preload="none" src="{esc(os.path.relpath(movie,study))}"></video></details>'
        if wall is not None:card+=f'<p>Actor wall: {wall:.1f} s; robot services: {service_seconds:.1f} s; outside services: {max(0,wall-service_seconds):.1f} s. Outside includes model, transport and local analysis.</p>'
        else:card+=f'<p>Current controls: {esc(status_row.get("control_steps"))}; remaining: {esc(status_row.get("remaining_control_steps"))}; last operation: {esc(status_row.get("last_completed_operation"))}.</p>'
        card+='<details open><summary>Time by robot service (inclusive)</summary>'+timing_table(profiling['service_seconds_by_operation'])+'</details>'
        card+='<details><summary>Execution, observation, grasp search and IK breakdown</summary><p>These are nested measurements inside service time. Sections overlap; do not add them together. Missing measurements are not zero-cost work.</p>'
        for key in ['nested_execution_and_observation_seconds','nested_grasp_generation_and_selection_seconds','nested_grasp_search_seconds','nested_full_pose_ik_seconds','nested_arc_goal_ik_and_connection_seconds','nested_contact_skill_seconds','nested_sequence_planning_seconds']:
            card+='<h3>'+esc(key.replace('_',' '))+'</h3>'+timing_table(profiling[key])
        card+='</details>'+link(run/'timing-breakdown.json',study,'Timing JSON')
        analysis=read(run/'failure-analysis.json',{}) or read(run/'outcome-analysis.json',{})
        if analysis:
            card+='<p><strong>Outcome details:</strong> '+esc(' '.join(analysis.get('main_observed_failures',[])) or analysis.get('interpretation','See the measured evidence below.'))+'</p>'
            card+='<details><summary>Measured outcome evidence</summary><p>Offline simulator grading only; native quantities were never actor or planner inputs.</p><pre>'+esc(json.dumps(analysis,indent=2))+'</pre></details>'
        failed=[e for e in events if not e.get('response',{}).get('ok')]
        if failed:
            card+='<details><summary>Rejected requests and numerical failures</summary><table><tr><th>Event/tool</th><th>Reported cause</th></tr>'
            for e in failed:card+='<tr><td>'+esc(f"{e['sequence']} / {e['request']['operation']}")+'</td><td>'+esc(e['response'].get('error','See full log'))+'</td></tr>'
            card+='</table></details>'
        for event in events:
            if event['request']['operation'] in {'push','continue_push','push_stroke','hook_pull'}:
                result=event.get('response',{}).get('result',{})
                card+='<details><summary>'+esc(f"{event['request']['operation']} event {event['sequence']}: {result.get('stop_reason','see log')}")+'</summary>'
                card+=skill_card(f'Event {event["sequence"]}',result,run/'tool-events.jsonl',study,{})+'</details>'
        cards.append(card+'</article>')
    if trial_rows:
        body+='<article><h2>Luna evaluations</h2><table><tr><th>Condition</th><th>Task</th><th>Budget</th><th>Native success</th><th>Controls</th><th>By 600</th><th>Actor wall</th><th>Services</th></tr>'
        for name,task,budget,success,steps,by600,wall,svc in trial_rows:
            body+='<tr>'+''.join(f'<td>{esc(v)}</td>' for v in [name,task,budget,success,steps,by600,f'{wall:.1f} s' if wall is not None else 'running',f'{svc:.1f} s'])+'</tr>'
        body+='</table></article>'
    if practice_rows:
        body+='<article><h2>Luna practice (development)</h2><p>One persistent learner session. Episode wall time ends at terminal bookkeeping; whole-session time, including note writing, is reported in its card below.</p><table><tr><th>Attempt</th><th>Case</th><th>Native success</th><th>Controls</th><th>Episode wall</th></tr>'
        for name,task,success,steps,seconds in practice_rows:
            body+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [name,task,success,steps,f'{seconds:.1f} s' if seconds is not None else 'unknown'])+'</tr>'
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
