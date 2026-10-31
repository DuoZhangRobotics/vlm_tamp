"""Render contact-skill evidence in the existing Live Preview dashboard."""
import argparse
from collections import Counter
import gzip
import html
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
_EVENT_CACHE={}


def file_stamp(path):
    try:
        stat=Path(path).stat()
        return (stat.st_mtime_ns,stat.st_size)
    except OSError:
        return None


def read_events(path):
    """Reuse unchanged logs while a live dashboard polls the filesystem."""
    path=Path(path);stamp=file_stamp(path);key=str(path)
    previous=_EVENT_CACHE.get(key)
    if previous is not None and previous[0]==stamp:return previous[1]
    rows=[]
    if stamp is not None:
        with path.open() as stream:
            for line in stream:
                try:rows.append(json.loads(line))
                except ValueError:pass  # A writer may not have finished its last line.
    _EVENT_CACHE[key]=(stamp,rows)
    return rows


def brief_evidence(value,depth=0):
    """Keep readable diagnostics in HTML; large arrays stay in linked records."""
    if isinstance(value,dict):
        if depth>=4:return {'fields':list(value),'details':'See complete evidence'}
        return {k:brief_evidence(v,depth+1) for k,v in value.items()}
    if isinstance(value,list):
        if len(value)>8:
            return {'items':len(value),'preview':[brief_evidence(v,depth+1) for v in value[:2]],
                    'details':'See complete evidence'}
        return [brief_evidence(v,depth+1) for v in value]
    return value


def read(path,default=None):
    try:
        path=Path(path)
        if not path.exists() and path.suffix=='.json':path=path.with_suffix('.json.gz')
        return json.loads(gzip.open(path,'rt').read() if path.suffix=='.gz' else path.read_text())
    except (OSError,json.JSONDecodeError):return default


def esc(value):return html.escape(str(value))


def link(path,base,label):return f'<a href="{esc(os.path.relpath(path,base))}">{esc(label)}</a>'


def timing_table(value):
    if not value:return '<p>No timing measurement recorded.</p>'
    return '<table><tr><th>Measured phase</th><th>Seconds</th></tr>'+''.join(
        f'<tr><td>{esc(k.replace("_"," "))}</td><td>{v:.3f}</td></tr>'
        for k,v in value.items() if isinstance(v,(int,float)))+'</table>'


def robot_part(code):
    code=int(code)
    return f'arm link {code-2000}' if 2001<=code<=2007 else 'palm' if code==1000 else f'gripper volume {code}'


def contact_image(run,relative,replay):
    """Copy only a public contact PNG into the served tree during a live run."""
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix.lower()!='.png':return None
    candidate=run/('workspace' if replay else 'agent-workspace')/relative
    if candidate.is_file():return candidate
    if replay:return None
    command=read(run/'agent-command.json',[])
    if not isinstance(command,list):return None
    work=next((Path(command[i+1]) for i,v in enumerate(command[:-1]) if v in {'-C','--cd'}),None)
    if work is None:return None
    source=work/relative
    if not source.is_file() or not source.resolve().is_relative_to(work.resolve()):return None
    destination=run/'live-contact-overlays'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        if not destination.exists() or destination.stat().st_mtime_ns!=source.stat().st_mtime_ns:
            shutil.copy2(source,destination)
    except FileNotFoundError:
        # The finished-run archiver may move the public workspace concurrently.
        archived=run/'agent-workspace'/relative
        return archived if archived.is_file() else None
    return destination


def render(study,watch=None):
    study=Path(study).resolve();study.mkdir(parents=True,exist_ok=True);cards=[];summary=[];actor_cards=[]
    for path in sorted(study.glob('*/previews.json')):
        for row in read(path,[]):
            attempts=row.get('diagnostic',{}).get('attempts',[])
            rejected=Counter()
            for attempt in attempts:
                checks=attempt.get('endpoint_checks',[])+([attempt['diagnostic']] if 'diagnostic' in attempt else [])
                for check in checks:
                    if check.get('accepted',False):continue
                    if 'last_rejected_part' in check:
                        surface={0:'stationary geometry',1:'target outside contact patch',2:'selected contact patch'}.get(int(check['last_rejected_surface']),'unknown')
                        rejected[f'{robot_part(check["last_rejected_part"])} / {surface}']+=1
            outcome='Approach planned; physical contact unverified' if row.get('solved') else row.get('stop_reason','unknown')
            cards.append(f'<article><h2>Recorded selection {row["source_luna_event"]} — {esc(path.parent.name)}</h2>'
                f'<p><strong>{esc(outcome)}</strong> · {row.get("seconds",0):.2f} s · {esc(row.get("contact_surface"))}</p>'
                f'<p>Historical Luna selection at the near-closed checkpoint; this is a numerical preview, not task success.</p>'
                f'<p>Rejected geometry: {esc(dict(rejected))}</p>'
                +timing_table(row.get('plan',{}).get('timing_seconds',row.get('diagnostic',{}).get('timing_seconds',{})))
                +link(path,study,'Full candidates and diagnostics')+'</article>')
    for path in sorted(study.glob('*/replay.json')):
        replay=read(path,{});result=replay.get('skill_result',{});name=path.parent.name
        summary.append({'run':name,'type':'physical replay','native_success':replay.get('native_success_seen'),
            'final_success_after_settling':replay.get('final_task_success'),
            'withdrawal_executed':any(s.get('stage')=='push-release-withdrawal' for s in result.get('stages',[])),
            'stop_reason':result.get('stop_reason',replay.get('error'))})
        cards.append(skill_card(name,result,path,study,replay))
    for run in sorted((study/'trials').glob('*')) if (study/'trials').exists() else []:
        protocol=read(run/'protocol.json',{});evaluation=read(run/'evaluation-private.json',{})
        episodes=evaluation.get('attempts',[]);agent=read(run/'agent-exit.json',{})
        events=read_events(run/'tool-events.jsonl')
        tool_seconds=sum(e.get('wall_seconds',0) for e in events)
        actor_wall=agent.get('wall_seconds')
        timing={'service_calls_inclusive':tool_seconds}
        if actor_wall is not None:
            timing.update(actor_wall=actor_wall,actor_outside_service_approx=max(0,actor_wall-tool_seconds))
        selected=[e for e in events if e.get('request',{}).get('operation') in {'push','continue_push','push_stroke','hook_pull'}]
        calls=Counter(e['request']['operation'] for e in selected)
        previews=sum(bool(e['request'].get('arguments',{}).get('preview')) for e in selected)
        executed=sum(bool(e.get('response',{}).get('result',{}).get('robot_motion_executed')) for e in selected)
        skill_timing=Counter();nested_ik=0.
        for event in selected:
            result=event.get('response',{}).get('result',{})
            skill_timing.update(result.get('timing_seconds',{}))
            nested_ik+=result.get('search',{}).get('timing_seconds',{}).get('ik_seconds',0.)
            nested_ik+=sum(s.get('planning_timing_seconds',{}).get('ik_seconds',0.) for s in result.get('stages',[]))
        success=[e.get('official_task_success') for e in episodes]
        interruption=read(run/'operator-interruption.json',{})
        summary.append({'run':run.name,'type':'Luna','case':protocol.get('case'),
            'native_outcomes':success,'contact_tool_calls':dict(calls),'preview_calls':previews,
            'physically_executed_contact_calls':executed,'agent_exit':agent})
        last=events[-1] if events else {};last_result=last.get('response',{}).get('result',{})
        card=f'<article><h2>Luna — {esc(run.name)}</h2><p>Native outcomes: {esc(success) if episodes else "running / no completed attempt"}. '
        card+=(f'Tools available: {"push_stroke, push, continue_push, hook_pull, grasped manipulation" if protocol.get("surface_contact_tools") else "see protocol"}. '
            f'Selections: {esc(dict(calls))}; previews: {previews}; calls with physical contact-skill motion: {executed}.</p>'
            f'<p>Last returned tool: {esc(last.get("request",{}).get("operation","none"))}; '
            f'outcome: {esc(last_result.get("stop_reason",last_result.get("status","see log")))}.</p>'+timing_table(timing)
            +'<p>Service time includes numerical planning and observations. The outside-service remainder includes model, transport and local analysis; it is not pure reasoning time.</p>'
            +'<details><summary>Contact-skill phase totals</summary>'+timing_table(skill_timing)
            +f'<p>Recorded IK time nested inside search/planning: {nested_ik:.3f} s. Missing timing in interrupted stages is not counted as zero-cost work.</p></details>'
            +link(run/'agent-events.jsonl',study,'Luna input/output log')+' · '+link(run/'tool-events.jsonl',study,'Tool inputs/output'))
        if interruption:
            card+=f'<p class="note">Interrupted for implementation repair: {esc(interruption.get("reason"))}. This is an incomplete practice cohort; the interrupted attempt is not a model-capability finding.</p>'
        for filename,label in [('agent-workspace/findings.txt','Luna findings'),('agent-workspace/guidelines.json','Learner memory'),('agent-prompt.txt','Actor prompt')]:
            if (run/filename).is_file():card+=' · '+link(run/filename,study,label)
        for movie in sorted(run.glob('attempt*/initial/rollout.mp4')):
            card+=f'<details><summary>{esc(movie.parent.parent.name)} video</summary><video controls preload="none" src="{esc(os.path.relpath(movie,study))}"></video></details>'
        live=read(run/'live-episode-status.json',{})
        if not agent and not interruption and live:
            card+=f'<p>Current controls: {esc(live.get("control_steps"))}; remaining: {esc(live.get("remaining_control_steps"))}. Last completed operation: {esc(live.get("last_completed_operation"))}.</p>'
        actor_cards.append((run.stat().st_mtime,card+'</article>'))
        for event in selected:
            result=event.get('response',{}).get('result',{})
            cards.append(skill_card(f'{run.name} / event {event["sequence"]}',result,run/'tool-events.jsonl',study,{}))
    qualification=read(study/'momentum-qualification.json',{})
    body='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Luna contact skills — execution and timing</title><style>
body{font:16px system-ui;background:#f4f7fb;color:#192d42;margin:24px auto;max-width:1100px;padding:0 18px}article{background:white;border:1px solid #ccd7e3;border-radius:10px;padding:18px;margin:16px 0}h1{font-size:26px}h2{font-size:20px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #dde5ef;text-align:left;padding:7px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto;font-size:12px}img,video{max-width:100%;max-height:420px}a{color:#1765a4}.note{padding:12px;background:#e6eef9}button{padding:8px 16px}</style></head><body>
<h1>Push, hook-pull and momentum experiments</h1><p class="note">Planning success, measured engagement, task success and retained closure are separate outcomes. Failed runs remain visible. Preview rows issue no motion.</p>
<p><strong>Current policy: zero added collision padding.</strong> Buffered runs are historical diagnostics; the buffered v2 batch was stopped after user feedback.</p>
<button onclick="location.reload()">Refresh results</button>'''
    body+=f'<p>Strict contact qualification: <strong>{"passed" if qualification.get("execution_qualified") else "not established"}</strong>. Simulation experiments are available under either collision policy; physical contacts remain enabled. Qualification is reported separately from permission to execute.</p>'
    evaluation=read(study/'advisory-v4-evaluation-summary.json',{})
    practice=read(study/'frozen-memory/advisory-v4/provenance.json',{})
    if evaluation:
        body+='<article><h2>Latest Luna results with frozen practice memory</h2>'
        body+=f'<p>Practice: <strong>{practice.get("successes",0)}/{practice.get("attempts",0)} native successes</strong>. '
        body+=link(study/'frozen-memory/advisory-v4/guidelines.json',study,'Read Luna’s verbatim learned guidelines')+'. No successful coordinates were supplied or retained in this memory.</p>'
        body+='<table><tr><th>Fresh evaluation</th><th>Native success</th><th>Controls</th><th>Actor wall</th><th>Robot services</th><th>Outside services*</th></tr>'
        for row in evaluation.get('fresh_evaluations',[]):
            label=row['trial'].replace('advisory-v4-','').replace('_',' ')
            body+=f'<tr><td>{esc(label)}</td><td>{"yes" if row["native_success"] else "no"}</td><td>{row["controls"]}</td><td>{row["actor_wall_seconds"]:.1f} s</td><td>{row["service_seconds"]:.1f} s</td><td>{row["outside_service_seconds"]:.1f} s</td></tr>'
        body+='</table><p>*Outside services includes model calls, transport and local analysis. One fresh actor per previously used case; these are not unseen benchmark estimates. Normal episodes stop at first native success.</p>'
        failure=read(study/'trials/advisory-v4-close_microwave-state049/failure-analysis.json',{})
        if failure:
            body+='<p><strong>Why closing state 49 failed:</strong> the first push opened it another 3.3°. Reaching the grasp cost 315 controls. The arcs then closed it to 11.1° open, but the pad gap collapsed to approximately zero. Only 7 controls remained; the cheapest recovery approach found needed 38, before its push. These door angles are offline evaluation evidence, never actor input.</p>'
            body+=link(study/'trials/advisory-v4-close_microwave-state049/failure-analysis.json',study,'Full failure measurements')
        body+='</article>'
    retention=read(study/'actor-retention-comparison.json',[])
    if retention:
        body+='<article><h2>After Luna success: withdrawal and one-second coasting</h2><table><tr><th>Replayed actor decision</th><th>Still closed</th><th>No robot/environment force contact during coasting</th></tr>'
        for row in retention:
            body+=f'<tr><td>{esc(row["run"])}</td><td>{esc(row.get("final_task_success"))}</td><td>{esc(row.get("contact_free_coast"))}</td></tr>'
        body+='</table><p>One practice replay retained palm contact during coasting; that row does not certify fully disengaged closure. Normal Luna stopping was unchanged.</p></article>'
    hooks=read(study/'advisory-v4-hook-analysis.json',{}).get('rows',[])
    if hooks:
        body+=f'<article><h2>Local skill repeatability</h2><p>Non-pinching hook motion: <strong>{sum(r["criterion_met"] for r in hooks)}/{len(hooks)}</strong>, measured 2.79° for a 3° request, finger-pad contact only. Controlled closure: <strong>3/3</strong>; these controlled tests held the final arm pose during settling and included palm contact.</p></article>'
    comparison=read(study/'collision-advisory-3cm-comparison.json',[])
    if comparison:
        rows=[]
        for speed in sorted({r['requested_speed_m_s'] for r in comparison}):
            group=[r for r in comparison if r['requested_speed_m_s']==speed]
            times=[]
            for row in group:
                replay=read(study/row['run']/'replay.json',{})
                if row.get('first_success_control_step') is not None:
                    times.append((row['first_success_control_step']-replay['prepared_control_step'])/20)
            rows.append(f'<tr><td>{speed*100:g} cm/s</td><td>{sum(r["native_success"] for r in group)}/{len(group)}</td>'
                f'<td>{sum(r["retained_success"] for r in group)}/{len(group)}</td>'
                f'<td>{sum(times)/len(times):.2f} s</td>'
                f'<td>{100*sum(r["peak_contact_speed_m_s"] for r in group)/len(group):.2f} cm/s</td></tr>' if times else '')
        body+='<article><h2>Actual 3 cm commands with advisory collision predictions</h2><p>Matched physical replays from the same prepared contact pose. Physical contacts remained enabled. Closure occurred during the push and was checked again after withdrawal and one second of coasting. Motion time excludes initialization, planning and observations; per-run wall timing is shown below.</p><table><tr><th>Requested speed</th><th>Task success</th><th>Retained closure</th><th>Simulated time to success</th><th>Measured peak speed</th></tr>'+''.join(rows)+'</table></article>'
    body+=''.join(card for _,card in sorted(actor_cards,reverse=True))
    body+=''.join(cards) if cards else '<p>No completed checks yet.</p>'
    body+='<p>Timing categories inside a search are nested in its total; do not add inclusive totals twice. Geometry checks use observed RGB-D and known robot geometry.</p></body></html>'
    (study/'watch.html').write_text(body);(study/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if watch and Path(watch).exists():
        watch=Path(watch);text=watch.read_text();start='<!-- CONTACT_SKILLS_BEGIN -->';end='<!-- CONTACT_SKILLS_END -->'
        block=start+f'<section id="contact-skills"><h2>Push and hook-pull experiments</h2><p>{link(study/"watch.html",watch.parent,"Open contact skills, failures and timing")}</p><iframe title="Contact skill validation" src="{esc(os.path.relpath(study/"watch.html",watch.parent))}" style="width:100%;height:1100px;border:1px solid #ccd7e3"></iframe></section>'+end
        if start in text:text=text[:text.index(start)]+block+text[text.index(end)+len(end):]
        else:text=text.replace('</body>',block+'\n</body>') if '</body>' in text else text+block
        watch.write_text(text)


def skill_card(name,result,path,study,replay):
    text=f'<article><h2>{esc(name)}</h2><p><strong>{esc(result.get("stop_reason",replay.get("error","pending")))}</strong> · {esc(result.get("skill",""))} / {esc(result.get("mode",""))}</p>'
    if result.get('environment_collision_veto') is False or result.get('collision_policy')=='advisory':
        text+='<p><strong>Simulation contact experiment:</strong> environmental collision predictions are logged without vetoing the contact motion. Physical simulator contacts remain active.</p>'
    if result.get('measured_forward_hand_travel_m') is not None:
        text+=f'<p>Measured forward hand travel: {1000*result["measured_forward_hand_travel_m"]:.1f} mm. Predicted collisions ignored during this diagnostic: {result.get("ignored_collision_predictions",0)}.</p>'
    if replay:
        text+=f'<p>Native success seen: {esc(replay.get("native_success_seen","unmeasured"))}; final after diagnostic settling: {esc(replay.get("final_task_success","unmeasured"))}; controls: {esc(replay.get("control_steps","unknown"))}.</p>'
        if result.get('mode')=='controlled':
            text+='<p>Controlled diagnostic settling holds the final arm pose. It does not establish closure after withdrawal.</p>'
            if replay.get('native_success_seen') and result.get('stop_reason')=='no_progress_after_two_strokes':
                text+='<p>This diagnostic continued after native success; the later no-progress stop describes additional pushing after the task first succeeded.</p>'
        if result.get('skill')=='hook_pull':
            text+='<p>This hook test pulls in a closing-task scene. Its skill criterion is measured non-pinching pull progress; native closing success is a separate outcome.</p>'
        audit=replay.get('offline_contact_audit',{})
        if audit:
            parts=Counter(x['robot_part'] for x in audit.get('non_permitted_robot_parts',[]))
            text+=f'<p>Offline contact audit: {esc(dict(parts)) if parts else "no contacts on non-permitted robot parts recorded"}. Finger-body location still needs geometric verification when present.</p>'
        movie=path.parent/'service/attempt01/initial/rollout.mp4'
        if movie.exists():text+=f'<video controls preload="none" src="{esc(os.path.relpath(movie,study))}"></video>'
    for image in result.get('images',[]):
        candidate=contact_image(path.parent,image,bool(replay))
        if candidate is not None:text+=f'<img loading="lazy" alt="Selected contact and requested motion" src="{esc(os.path.relpath(candidate,study))}">'
    text+=f'<p>Requested contact speed: {esc(result.get("requested_contact_speed_m_s","—"))} m/s.</p>'
    if result.get('observed_progress') is not None:
        import math
        is_joint=result.get('progress_units')=='radians'
        measured=math.degrees(result['observed_progress']) if is_joint else result['observed_progress']*1000
        requested=abs(result.get('requested_joint_degrees',0)) if is_joint else (result.get('requested_distance_m') or 0)*1000
        text+=f'<p>Requested / measured object progress: {requested:.2f} / {measured:.2f} {"degrees" if is_joint else "mm"}. '
        text+=('The joint-progress stopping tolerance is 0.5°.' if is_joint else 'The distance stopping tolerance is 1 mm.')+'</p>'
    text+=f'<p>Visual feedback measurements: {result.get("feedback_count",sum("object_motion" in s for s in result.get("stages",[])))}.</p>'
    text+=timing_table(result.get('timing_seconds',{}))
    nested=Counter()
    run_root=path.parent/'service' if replay else path.parent
    for stage in result.get('stages',[]):
        nested['robot_execution_rpc_seconds']+=stage.get('execution',{}).get('wall_seconds',0)
        for timing_file in run_root.glob(f'attempt*/{stage["artifact_id"]:03d}-issue-timing.json*'):
            phases=read(timing_file,{}).get('phases',{})
            nested['capture_and_export_seconds']+=phases.get('capture_seconds',0)
            nested['trajectory_and_records_seconds']+=phases.get('trajectory_and_records_seconds',0)
    if nested:
        text+='<details open><summary>Execution / observation breakdown (nested in execution-and-capture total)</summary>'+timing_table(nested)+'</details>'
    text+='<details><summary>Search breakdown (nested)</summary>'+timing_table(result.get('search',{}).get('timing_seconds',{}))+'</details>'
    text+='<table><tr><th>Stage</th><th>Requested m/s</th><th>Measured mean / peak m/s</th><th>Object progress</th><th>Engagement</th></tr>'
    for stage in result.get('stages',[]):
        speed=stage.get('measured_contact_speed',{})
        label='Approach (contact possible in advisory mode)' if stage['stage']=='free-approach' else stage['stage']
        text+=f'<tr><td>{esc(label)}</td><td>{esc(stage.get("requested_contact_speed_m_s","—"))}</td><td>{esc(speed.get("mean_m_s","—"))} / {esc(speed.get("peak_m_s","—"))}</td><td>{esc(stage.get("observed_contact_progress_m","unresolved"))}</td><td>{esc(stage.get("engagement",{}).get("status","unverified"))}</td></tr>'
    evidence=brief_evidence({k:result.get(k) for k in ['contact','diagnostic','search']})
    text+='</table><details><summary>Contact and failure summary</summary><p>Full candidate arrays and stage records are in the linked evidence.</p><pre>'+esc(json.dumps(evidence,indent=2))+'</pre></details>'
    return text+link(path,study,'Complete evidence')+'</article>'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',default=str(ROOT/'results/contact-skills-20261028'))
    parser.add_argument('--watch',default=str(ROOT/'results/luna-live-hinge-comparison-20261027/watch.html'))
    args=parser.parse_args();render(args.study,args.watch)
