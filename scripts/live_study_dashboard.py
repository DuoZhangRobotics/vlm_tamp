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

TEMPLATES = Path(__file__).resolve().parents[1]/'templates'
HTML_PATH = TEMPLATES/'luna_watch.html'
CLIENT_PATH = TEMPLATES/'luna_watch.js'
HTML = HTML_PATH.read_text()
CLIENT_JS = CLIENT_PATH.read_text()



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
    videos=[]
    for movie in sorted(run.glob('attempt*/initial/rollout.mp4')):
        attempt_name=movie.parent.parent.name
        number=int(attempt_name.removeprefix('attempt'))
        completed=any(a.get('attempt')==number for a in attempts)
        videos.append({'label':attempt_name+(' · partial recording' if not completed else ''),
                       'path':_relative(movie,study),'partial':not completed})
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
             'detail':_relative(path,study),'detail_version':file_stamp(path)[0],'videos':videos}
    _DETAILS[str(run)]=(stamp,summary)
    return summary,written


def publish(study,*,force=False,now=None):
    """HTML is static. Running status publishes at most once/minute or on a phase boundary."""
    study=Path(study);now=time.monotonic() if now is None else now
    boundary=tuple(file_stamp(study/name) for name in ['study-manifest.json','active-process.json','capacity-wait.json'])
    previous=_PUBLISH.get(str(study))
    if previous and not force and previous['boundary']==boundary and now-previous['at']<INTERVAL_SECONDS:
        return {'published':False,'reason':'rate_limited','bytes_written':0}
    manifest=read(study/'study-manifest.json',{});active=read(study/'active-process.json',{})
    stopped=manifest.get('experiments_stopped_by_user',False);running=_live_process(active) and not stopped
    if previous and not force and not running and previous['boundary']==boundary:
        return {'published':False,'reason':'unchanged_inactive','bytes_written':0}
    started=time.perf_counter();html_bytes=_write(study/'watch.html',HTML_PATH.read_text())
    client_bytes=_write(study/'watch.js',CLIENT_PATH.read_text());written=html_bytes+client_bytes
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
    capacity=read(study/'capacity-wait.json',{})
    if capacity.get('waiting') and not stopped:
        status='Waiting for shared memory capacity'
        shared=capacity.get('shared_headroom_gib')
        shared=f'{shared:.1f} GiB' if isinstance(shared,(int,float)) else 'not limited'
        message=f"The next Luna actor has not started. Additional headroom needed: {capacity['additional_headroom_needed_gib']:.1f} GiB; available shared-pool headroom including clean inactive file cache: {shared}. Waiting does not consume the episode budget."
    if active.get('status')=='running' and not running and not stopped:message='The saved process is no longer alive; its last result is recorded below.'
    links={'Study protocol':'study-manifest.json','Declared cases':'plan.json'}
    if manifest.get('superseded_by'):links['Open the new rerun']=manifest['superseded_by']
    monitor=read(study/'resource-monitor.json',{})
    resource=read(Path(monitor['latest_sample']),{}) if monitor.get('latest_sample') else {}
    health=''
    if resource:
        label='Latest experiment sample' if running else 'Last recorded experiment sample before shutdown (not a current server reading)'
        memory=resource.get('job_memory_gib')
        memory=f'{memory:.1f}' if isinstance(memory,(int,float)) else 'unavailable'
        health=f"{label}: I/O pressure {resource.get('io_full_avg10','unavailable')}%; job memory {memory} GiB."
    payload={'checked_at':time.time(),'running':running,'status_text':status+' · '+phase,
             'message':message,'totals':totals,'runs':runs,'active_run':active.get('run') if running else None,
             'links':links,'health_text':health,'publish_interval_seconds':INTERVAL_SECONDS}
    written+=_json(study/'watch-status.json',payload)
    result={'published':True,'html_bytes_written':html_bytes,'client_bytes_written':client_bytes,'bytes_written':written,
            'seconds':time.perf_counter()-started,'runs':len(runs)}
    _PUBLISH[str(study)]={'at':now,'boundary':boundary}
    return result
