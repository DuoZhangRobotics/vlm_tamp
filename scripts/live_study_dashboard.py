"""Small static dashboard with rate-limited status and on-demand run details."""
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

from report_contact_skills import read, read_events, file_stamp, contact_image, brief_evidence
from report_restriction_fixes import cached_profile_run

INTERVAL_SECONDS = 60
_PUBLISH = {}
_DETAILS = {}

HTML = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Luna experiments</title><style>
:root{font:16px system-ui;color:#203249;background:#f3f6fa}body{max-width:1140px;margin:28px auto;padding:0 18px}h1{font-size:28px}h2{font-size:20px;margin-top:0}.muted{color:#607088;font-size:14px}section{background:white;border:1px solid #d6dfeb;border-radius:10px;padding:18px;margin:16px 0}button,select{padding:8px;font:inherit}a{color:#1462ad}table{width:100%;border-collapse:collapse}th,td{padding:8px;text-align:left;border-bottom:1px solid #e2e8f0;vertical-align:top}td:last-child{overflow-wrap:anywhere}.scroll{overflow:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.stats{display:flex;gap:22px;flex-wrap:wrap}.stats b{display:block;font-size:22px}img,video{max-width:100%;max-height:470px}video{width:100%;background:#111}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}details{margin:12px 0}summary{cursor:pointer}tr[data-run]{cursor:pointer}tr[data-run]:hover{background:#eef5fc}progress{width:100%}.warning{color:#964b00}.links a{display:inline-block;margin:5px 14px 5px 0}@media(max-width:720px){.grid{grid-template-columns:1fr}}
</style></head><body>
<h1>Luna experiments</h1>
<p id="status">Loading saved status…</p><p class="muted" id="stamp"></p>
<p><button id="refresh">Refresh status</button> <label><input id="follow" type="checkbox" checked> Follow current run</label></p>
<p class="muted">Small status updates once per minute while this tab is visible. Video playback and scroll position stay in place. Detailed logs open only when requested.</p>
<section><div id="totals" class="stats"></div><p id="message"></p><p id="studyLinks" class="links"></p></section>
<section><h2>Runs</h2><div class="scroll"><table><thead><tr><th>Run / case</th><th>Memory</th><th>Task outcome</th><th>Controls</th><th>Actor time</th></tr></thead><tbody id="runs"></tbody></table></div></section>
<section><h2 id="runTitle">Select a run</h2><p id="runStatus"></p><progress id="controls" max="600" value="0"></progress>
<p id="runLinks" class="links"></p>
<div class="grid"><div><h3>Recorded view</h3><img id="frame" alt="Last recorded robot view" hidden loading="lazy"><p id="frameNote" class="muted"></p></div>
<div><h3>Video</h3><select id="videoChoice" aria-label="Episode video"></select> <button id="loadVideo">Play selected video</button><p><video id="video" controls preload="none" hidden></video></p><p class="muted">Videos are not downloaded until selected for playback.</p></div></div>
<details id="contactDetails"><summary>Selected contact overlay</summary><img id="overlay" alt="Selected contact" hidden loading="lazy"></details>
<h3>Latest skill and feedback</h3><div id="skill"></div>
<h3>Time breakdown</h3><p id="timeScope" class="muted"></p><div id="times"></div>
<details><summary>Perception, grasp generation, IK, goal search and execution details</summary><p class="muted">These measurements are nested inside tool time and overlap. Do not add the sections together.</p><div id="nested"></div></details>
<h3>Recent reported issues</h3><p class="muted">A rejected request may have been recovered from. These records do not by themselves establish the cause of task failure.</p><div id="issues"></div>
</section><p id="health" class="muted"></p>
<script>
const el=id=>document.getElementById(id);let state=null,chosen=new URLSearchParams(location.search).get('run'),detail=null,lastDetail=null,busy=false;if(chosen)el('follow').checked=false;
const num=x=>typeof x==='number'?x.toFixed(1):'—';
function text(id,value){el(id).textContent=value??''}
function link(parent,label,path){if(!path)return;const a=document.createElement('a');a.textContent=label;a.href=path;a.target='_blank';a.rel='noopener';parent.append(a)}
function table(parent,rows){const t=document.createElement('table');for(const row of rows){const tr=t.insertRow();for(const v of row){const td=tr.insertCell();td.textContent=v??'—'}}parent.append(t)}
function image(id,path){const node=el(id);node.hidden=!path;if(path&&node.getAttribute('src')!==path)node.setAttribute('src',path);if(!path)node.removeAttribute('src')}
function playChoice(){const p=el('videoChoice').value;const v=el('video');v.hidden=!p;if(p&&v.getAttribute('src')!==p){v.src=p;v.load()}if(p)v.play().catch(()=>{})}
function showRun(d){detail=d;text('runTitle',d.label);text('runStatus',d.status_text);el('controls').max=d.maximum_control_steps||600;el('controls').value=d.control_steps||0;
 el('runLinks').replaceChildren();for(const [name,path] of Object.entries(d.links||{}))link(el('runLinks'),name,path);
 image('frame',d.frame);text('frameNote',d.frame?'Recorded evidence, not a live camera stream.':'Completed episode videos and image records are linked above.');el('overlay').dataset.path=d.contact_overlay||'';image('overlay',el('contactDetails').open?d.contact_overlay:null);
 const videos=JSON.stringify(d.videos||[]);if(el('videoChoice').dataset.list!==videos){const old=el('videoChoice').value;el('videoChoice').replaceChildren();for(const x of d.videos||[]){const o=document.createElement('option');o.value=x.path;o.textContent=x.label;el('videoChoice').append(o)}el('videoChoice').dataset.list=videos;if([...el('videoChoice').options].some(o=>o.value===old))el('videoChoice').value=old;if(el('video').paused){el('video').removeAttribute('src');el('video').hidden=true}}
 el('skill').replaceChildren();table(el('skill'),Object.entries(d.skill||{}));
 el('times').replaceChildren();const p=d.timing||{};text('timeScope',p.outside_service_scope||'Completed tool calls are timed; active requests may still be in progress.');
 table(el('times'),[['Actor total (s)',num(p.actor_wall_seconds)],['Robot tools, inclusive (s)',num(p.service_wall_seconds)],['Outside tools: Luna + transport + local analysis (s)',num(p.outside_service_seconds)]]);
 table(el('times'),Object.entries(p.service_seconds_by_operation||{}).map(([k,v])=>[k,num(v)+' s']));
 el('nested').replaceChildren();for(const [k,v] of Object.entries(p)){if(!k.startsWith('nested_')||typeof v!=='object'||!v)continue;const h=document.createElement('h4');h.textContent=k.replaceAll('_',' ');el('nested').append(h);table(el('nested'),Object.entries(v).map(([n,s])=>[n.replaceAll('_',' '),num(s)+' s']))}
 el('issues').replaceChildren();for(const issue of d.issues||[]){const box=document.createElement('details');const s=document.createElement('summary');s.textContent='Event '+issue.event+' · '+issue.operation+' · '+issue.reason;box.append(s);if(issue.diagnostic){const pre=document.createElement('pre');pre.textContent=JSON.stringify(issue.diagnostic,null,2);box.append(pre)}el('issues').append(box)}
}
async function selectRun(id,manual=false){if(manual)el('follow').checked=false;if(chosen!==id&&!el('video').paused)el('video').pause();chosen=id;const row=(state?.runs||[]).find(r=>r.id===id);if(!row)return;const key=row.detail+'?v='+row.detail_version;if(key===lastDetail)return;const r=await fetch(key,{cache:'no-cache'});if(!r.ok)throw Error('Run details are not available yet');const d=await r.json();if(chosen===id){showRun(d);lastDetail=key}}
async function refresh(){if(busy)return;busy=true;try{const r=await fetch('watch-status.json',{cache:'no-cache'});if(!r.ok)throw Error('Saved status is not available yet');state=await r.json();
 text('status',state.status_text);text('message',state.message);const age=(Date.now()/1000-state.checked_at);text('stamp','Last check: '+new Date(state.checked_at*1000).toLocaleString()+(state.running&&age>150?' · Status is stale; the publisher may have stopped.':''));
 el('totals').replaceChildren();for(const [name,v] of Object.entries(state.totals||{})){const div=document.createElement('div');const b=document.createElement('b');b.textContent=v.finished+' / '+v.planned;div.append(b,document.createTextNode(name)) ;el('totals').append(div)}
 el('studyLinks').replaceChildren();for(const [name,path] of Object.entries(state.links||{}))link(el('studyLinks'),name,path);
 el('runs').replaceChildren();for(const row of state.runs||[]){const tr=document.createElement('tr');tr.dataset.run=row.id;for(const v of [row.label,row.memory,row.outcome,row.controls,num(row.actor_seconds)+' s']){const td=document.createElement('td');td.textContent=v??'—';tr.append(td)}tr.onclick=()=>selectRun(row.id,true).catch(e=>text('runStatus',e.message));el('runs').append(tr)}
 text('health',state.health_text||'');const next=state.active_run||(state.runs||[]).at(-1)?.id;if(!chosen||(el('follow').checked&&el('video').paused))chosen=next;if(chosen)await selectRun(chosen);
 }catch(e){text('status','Open watch.html with VS Code Live Preview to load the saved status. '+e.message)}finally{busy=false}}
el('refresh').onclick=refresh;el('loadVideo').onclick=playChoice;el('contactDetails').ontoggle=()=>{if(el('contactDetails').open)image('overlay',el('overlay').dataset.path)};el('follow').onchange=refresh;
setInterval(()=>{if(!document.hidden&&state?.running)refresh()},60000);document.addEventListener('visibilitychange',()=>{if(!document.hidden&&state?.running)refresh()});refresh();
</script></body></html>'''


def _write(path,text):
    path=Path(path)
    if path.exists() and path.read_text()==text:return 0
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    temporary.write_text(text);temporary.replace(path)
    return len(text.encode())


def _json(path,value):
    return _write(path,json.dumps(value,separators=(',',':'),ensure_ascii=False)+'\n')


def _relative(path,study):
    return os.path.relpath(path,study)


def _live_process(active):
    if active.get('status')!='running':return False
    try:
        return b'run_subscription_agent_pilot.py' in Path(f'/proc/{int(active["pid"])}/cmdline').read_bytes()
    except (OSError,KeyError,ValueError):return False


def _run_data(study,run,entry,active):
    watched=['tool-events.jsonl','live-episode-status.json','agent-exit.json','evaluation-private.json',
             'operator-interruption.json','protocol.json']
    stamp=(active,*[file_stamp(run/name) for name in watched])
    previous=_DETAILS.get(str(run))
    if previous is not None and previous[0]==stamp:return previous[1],0
    events=read_events(run/'tool-events.jsonl');agent=read(run/'agent-exit.json',{})
    protocol=read(run/'protocol.json',{});status=read(run/'live-episode-status.json',{})
    attempts=read(run/'evaluation-private.json',{}).get('attempts',entry.get('attempts',[]))
    interruption=read(run/'operator-interruption.json',{})
    task=entry.get('task',protocol.get('case',{}).get('task','practice'))
    state=entry.get('state_index',protocol.get('case',{}).get('state_index',protocol.get('case',{}).get('state')))
    label=f'{task.replace("_"," ")} / {state}' if state is not None else run.name
    success=sum(bool(a.get('official_task_success')) for a in attempts)
    outcome=(f'{success}/{len(attempts)} success' if attempts else 'Interrupted' if interruption else 'In progress' if active else 'No terminal result')
    controls=attempts[-1].get('control_steps') if attempts else status.get('control_steps')
    issues=[];skills=[]
    for event in events:
        op=event['request']['operation'];response=event.get('response',{});value=response.get('result',{})
        if op in {'push','hook_pull','execute_sequence','execute','gripper'}:skills.append(event)
        reason=response.get('error') if response.get('ok') is False else value.get('stop_reason')
        if reason and reason!='preview_only':
            issues.append({'event':event['sequence'],'operation':op,'reason':str(reason)[:1800],
                           'diagnostic':brief_evidence(response.get('planning_diagnostic',value.get('diagnostic')))})
    last=skills[-1] if skills else {};args=last.get('request',{}).get('arguments',{})
    result=last.get('response',{}).get('result',{});progress=result.get('observed_progress')
    if progress is not None:
        progress=f'{math.degrees(progress):.2f} degrees' if result.get('progress_units')=='radians' else f'{progress*1000:.2f} mm'
    requested=result.get('requested_joint_degrees')
    requested=(f'{requested} degrees' if requested is not None else
               f"{1000*result['requested_distance_m']:.1f} mm" if result.get('requested_distance_m') is not None else '—')
    skill={'Operation':last.get('request',{}).get('operation','—'),'Mode':result.get('mode',args.get('mode','—')),
           'Requested contact speed (m/s)':result.get('requested_contact_speed_m_s','—'),
           'Requested motion':requested,
           'Measured object motion':progress or '—','Last skill feedback observations':result.get('feedback_count','—'),
           'Reported feedback count across tools':sum(e.get('response',{}).get('result',{}).get('feedback_count',0) for e in events),
           'Stop reason':result.get('stop_reason','—')}
    for stage in reversed(result.get('stages',[])):
        speed=stage.get('measured_contact_speed',{})
        if speed:skill.update({'Measured stage':stage.get('stage'),'Engagement':stage.get('engagement',{}).get('status','unverified'),
                              'Measured mean contact speed (m/s)':speed.get('mean_m_s'),
                              'Measured peak contact speed (m/s)':speed.get('peak_m_s')});break
    overlay=None
    for event in reversed(skills):
        for image in event.get('response',{}).get('result',{}).get('images',[]):
            candidate=contact_image(run,image,False)
            if candidate is not None:overlay=_relative(candidate,study);break
        if overlay:break
    frame=None
    if active:
        frames=sorted(run.glob('attempt*/initial/frames/step*.jpg'))
        if frames:frame=_relative(frames[-2] if len(frames)>1 else frames[-1],study)
    videos=[{'label':p.parent.parent.name,'path':_relative(p,study)} for p in sorted(run.glob('attempt*/initial/rollout.mp4'))]
    links={label:_relative(run/path,study) for label,path in
           [('Tool log','tool-events.jsonl'),('Luna input/output','agent-events.jsonl'),('Prompt','agent-prompt.txt'),
            ('Episode report','report/index.html'),('Findings','agent-workspace/findings.txt'),
            ('Learner notes','agent-workspace/guidelines.json'),('Timing JSON','timing-breakdown.json')]
           if (run/path).exists()}
    timing=cached_profile_run(run,events,agent.get('wall_seconds'))
    timing={k:v for k,v in timing.items() if not k.startswith('_')}
    reason=interruption.get('reason') or status.get('termination_reason') or status.get('last_completed_operation') or 'Waiting for first observation'
    data={'id':run.name,'label':label,'status_text':f'{outcome}; {controls if controls is not None else 0} controls. {reason}',
          'control_steps':controls,'maximum_control_steps':status.get('maximum_control_steps',600),
          'links':links,'frame':frame,'contact_overlay':overlay,'videos':videos,'skill':skill,
          'timing':timing,'issues':issues[-8:],'reported_issue_count':len(issues),
          'attempts':[{'attempt':a.get('attempt'),'task':a.get('task'),'state_index':a.get('state_index'),
                       'success':a.get('official_task_success'),'controls':a.get('control_steps')} for a in attempts]}
    path=run/'watch-data.json';written=_json(path,data)
    summary={'id':run.name,'label':label,'memory':entry.get('memory_condition','none'),
             'outcome':outcome,'controls':controls,'actor_seconds':agent.get('wall_seconds'),
             'detail':_relative(path,study),'detail_version':file_stamp(path)[0]}
    _DETAILS[str(run)]=(stamp,summary)
    return summary,written


def publish(study,*,force=False,now=None):
    """HTML is static. Running status publishes at most once/minute or on a phase boundary."""
    study=Path(study);now=time.monotonic() if now is None else now
    boundary=(file_stamp(study/'study-manifest.json'),file_stamp(study/'active-process.json'))
    previous=_PUBLISH.get(str(study))
    if previous and not force and previous['boundary']==boundary and now-previous['at']<INTERVAL_SECONDS:
        return {'published':False,'reason':'rate_limited','bytes_written':0}
    manifest=read(study/'study-manifest.json',{});active=read(study/'active-process.json',{})
    stopped=manifest.get('experiments_stopped_by_user',False);running=_live_process(active) and not stopped
    if previous and not force and not running and previous['boundary']==boundary:
        return {'published':False,'reason':'unchanged_inactive','bytes_written':0}
    started=time.perf_counter();html_bytes=_write(study/'watch.html',HTML);written=html_bytes
    completed=manifest.get('completed_runs',[]);entries={r['run']:r for r in completed}
    if active.get('run') not in entries and active.get('run'):entries[active['run']]={'run':active['run'],'memory_condition':'none'}
    runs=[]
    for name,entry in entries.items():
        run=study/'trials'/name
        if run.is_dir():
            row,n=_run_data(study,run,entry,running and name==active.get('run'));runs.append(row);written+=n
    counts=Counter(r['stage'] for r in completed);plan=manifest.get('plan',{})
    totals={key:{'finished':counts[key],'planned':count} for key,count in
            [('baseline',len(plan.get('baseline',[]))),('practice',1),('transfer',len(plan.get('transfer',[])))]}
    phase=manifest.get('phase','preflight')
    message=manifest.get('stop_reason','') if stopped else manifest.get('problem',{}).get('reason','')
    if not message:message=f"{plan.get('model','Luna')} / {plan.get('reasoning','high')} · {plan.get('control_steps',600)} controls per episode. Only public observations are supplied to the actor."
    status='Experiments stopped' if stopped else 'Running' if running else 'Complete' if manifest.get('completed') else 'Waiting / no live actor'
    if active.get('status')=='running' and not running and not stopped:message='The saved process is no longer alive; its last result is recorded below.'
    links={'Study protocol':'study-manifest.json','Declared cases':'plan.json'}
    if manifest.get('superseded_by'):links['Open the new rerun']=manifest['superseded_by']
    monitor=read(study/'resource-monitor.json',{})
    resource=read(Path(monitor['latest_sample']),{}) if monitor.get('latest_sample') else {}
    health=''
    if resource:
        health=f"Server I/O pressure (10-second average): {resource.get('io_full_avg10','unavailable')}%. Job memory: {resource.get('job_memory_gib','unavailable')} GiB. Resource measurements are separate from task outcomes."
    payload={'checked_at':time.time(),'running':running,'status_text':status+' · '+phase,
             'message':message,'totals':totals,'runs':runs,'active_run':active.get('run') if running else None,
             'links':links,'health_text':health,'publish_interval_seconds':INTERVAL_SECONDS}
    written+=_json(study/'watch-status.json',payload)
    result={'published':True,'html_bytes_written':html_bytes,'bytes_written':written,
            'seconds':time.perf_counter()-started,'runs':len(runs)}
    _PUBLISH[str(study)]={'at':now,'boundary':boundary}
    return result
