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


def issue_explanation(diagnostic):
    """Keep the failed module and cause visible without opening a large JSON log."""
    if not isinstance(diagnostic,dict):return None
    feedback=diagnostic.get('feedback',{})
    if feedback.get('stalled') and feedback.get('stall_window'):
        window=feedback['stall_window']
        return (f"Physical execution attempted {feedback.get('samples',0)} controls. Over the last "
            f"{window['simulation_seconds']:.1f}s, references advanced {1000*window['commanded_travel_m']:.1f} mm "
            f"but the hand advanced {1000*window['measured_directional_progress_m']:.2f} mm in that direction. "
            'This is measured directional stalling, not a point-cloud collision veto; the current hand can be commanded again.')
    start=diagnostic.get('start_rejection')
    if start:
        text=f"Start-state collision check: {start.get('blocking_robot_part','robot')} / {start.get('blocking_surface','bounds or self collision')}."
        depth=start.get('overlap_evidence',{}).get('overlap_depth_mm_range')
        if depth:text+=f' Observed-model overlap {depth[0]:.3f}–{depth[1]:.3f} mm.'
        return text+' No goal search was attempted.'
    stage=diagnostic.get('planning_stage')
    if stage:
        return (f"{stage['stage'].replace('_',' ')}: {stage.get('elapsed_seconds',0):.2f}s elapsed; "
            f"{stage.get('effective_seconds',0):.2f}s local allowance; "
            f"{stage.get('remaining_skill_seconds',0):.2f}s left in the whole skill. "
            'An incomplete search does not establish that the motion is impossible.')
    attempts=diagnostic.get('point_attempts',[])
    if attempts:
        counts=Counter(a.get('reason','unknown') for a in attempts)
        if set(counts)=={'wrong_side_contact'}:
            return f'Contact selection: all {len(attempts)} sampled points have a surface normal incompatible with the requested push direction. No robot motion was issued.'
        return 'Contact candidates: '+', '.join(f'{key}: {value}' for key,value in counts.items())+'.'
    return None


def _run_data(study,run,entry,active):
    watched=['tool-events.jsonl','live-episode-status.json','agent-exit.json','evaluation-private.json',
             'operator-interruption.json','protocol.json','evaluator-diagnostics.json']
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
    addenda=read(run/'evaluator-diagnostics.json',{}).get('events',{})
    issues=[];skills=[];contact_calls=[]
    for event in events:
        op=event['request']['operation'];response=event.get('response',{});value=response.get('result',{})
        if op in {'push','continue_push','push_stroke','hook_pull','execute_sequence','execute','gripper'}:skills.append(event)
        if op in {'push','continue_push','push_stroke','hook_pull'}:
            contact_calls.append({'event':event['sequence'],'operation':op,'mode':value.get('mode'),
                'stop_reason':value.get('stop_reason') or response.get('error'),
                'wall_seconds':event.get('wall_seconds'),'timing_seconds':value.get('timing_seconds',{}),
                'contact_reused':value.get('contact_reused'),'scene_reused':value.get('scene_reused'),
                'preview_reused':value.get('preview_reused'),'full_scene_builds':value.get('full_scene_builds'),
                'contact_searches':value.get('contact_searches')})
        reason=response.get('error') if response.get('ok') is False else value.get('stop_reason')
        if reason and reason not in {'preview_only','requested_progress_reached','requested_stroke_complete',
                'stroke_executed','requested_reference_motion_complete','native_task_success','bounded_release_stroke_complete'}:
            diagnostic=response.get('planning_diagnostic',value.get('diagnostic'))
            addendum=addenda.get(str(event['sequence']))
            issues.append({'event':event['sequence'],'operation':op,'reason':str(reason)[:1800],
                           'diagnostic':brief_evidence(diagnostic),'explanation':issue_explanation(diagnostic),
                           'evaluator_addendum':addendum})
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
    if result.get('skill'):
        skill.update({'Contact reused':result.get('contact_reused',False),'Scene reused':result.get('scene_reused',False),
            'Input mode':result.get('input_mode','targeted contact'),
            'Feedback cameras':', '.join(result.get('feedback_cameras',[])),
            'Contact stroke policy':result.get('contact_motion_policy','planned contact'),
            'Object motion measurement':result.get('object_motion_status','see feedback'),
            'Object pose current':result.get('continuation',{}).get('object_pose_current'),
            'Preview reused':result.get('preview_reused',False),'Full scene builds':result.get('full_scene_builds'),
            'Contact searches':result.get('contact_searches'),'Continuation available':result.get('continue_push_available',False),
            'Tracking reference reused':result.get('tracking_reference_reused',False),
            'Retained reference cameras':result.get('tracking_reference_camera_count'),
            'Commanded contact travel (m)':result.get('commanded_contact_distance_m'),
            'Measured contact progress (m)':result.get('measured_contact_progress_m')})
    cleared=result.get('current_robot_point_filter')
    if cleared:
        skill.update({'Last current-pose points cleared':cleared.get('last_removed_points',cleared.get('total_removed_points')),
            'Point-filter padding (m)':cleared.get('padding_m'),
            'Current-pose filter updates':cleared.get('updates',1)})
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
    wrist=None
    for event in reversed(events):
        value=event.get('response',{}).get('result',{})
        source=value.get('observation',value).get('images',{})
        if not isinstance(source,dict) or not source.get('wrist'):continue
        candidate=contact_image(run,source['wrist'],False)
        if candidate is not None:wrist=_relative(candidate,study);break
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
    if (study/'failure-summary.html').exists():links['Task failure explanations']='failure-summary.html'
    if (run/'evaluator-diagnostics.json').exists():links['Evaluator diagnosis']=_relative(run/'evaluator-diagnostics.json',study)
    timing=cached_profile_run(run,events,agent.get('wall_seconds'))
    timing={k:v for k,v in timing.items() if not k.startswith('_')}
    reason=interruption.get('reason') or status.get('termination_reason') or status.get('last_completed_operation') or 'Waiting for first observation'
    data={'id':run.name,'label':label,'status_text':f'{outcome}; {controls if controls is not None else 0} controls. {reason}',
          'control_steps':controls,'maximum_control_steps':status.get('maximum_control_steps',600),
          'links':links,'frame':frame,'wrist_view':wrist,'contact_overlay':overlay,'videos':videos,'skill':skill,
          'timing':timing,'contact_calls':contact_calls,'issues':issues[-8:],'reported_issue_count':len(issues),
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
    for interrupted in manifest.get('interrupted_runs',[]):
        if interrupted.get('run'):entries.setdefault(interrupted['run'],interrupted)
    if active.get('run') not in entries and active.get('run'):entries[active['run']]={'run':active['run'],'memory_condition':'none'}
    runs=[]
    for name,entry in entries.items():
        run=study/'trials'/name
        if run.is_dir():
            row,n=_run_data(study,run,entry,running and name==active.get('run'));runs.append(row);written+=n
    counts=Counter(r['stage'] for r in completed);plan=manifest.get('plan',{})
    totals={key:{'finished':counts[key],'planned':count} for key,count in
            [('baseline',len(plan.get('baseline',[]))),('practice',1),('transfer',len(plan.get('transfer',[])))]}
    scheduled=manifest.get('scheduled_stages')
    if scheduled:totals={key:value for key,value in totals.items() if key in scheduled}
    scored=[attempt for row in completed if not scheduled or row.get('stage') in scheduled
            for attempt in row.get('attempts',[])]
    task_outcomes={'completed':len(scored),'successes':sum(bool(a.get('official_task_success')) for a in scored)}
    phase=manifest.get('phase','preflight')
    message=manifest.get('stop_reason','') if stopped else manifest.get('problem',{}).get('reason','')
    if not message:message=f"{plan.get('model','Luna')} / {plan.get('reasoning','high')} · {plan.get('control_steps',600)} controls per episode. Only public observations are supplied to the actor."
    status='Experiments stopped' if stopped else 'Running' if running else 'Complete' if manifest.get('completed') else 'Waiting / no live actor'
    if not running and scheduled==['baseline'] and manifest.get('baseline_completed'):status='Complete'
    capacity=read(study/'capacity-wait.json',{})
    if capacity.get('waiting') and not stopped:
        status='Waiting for shared memory capacity'
        shared=capacity.get('shared_headroom_gib')
        shared=f'{shared:.1f} GiB' if isinstance(shared,(int,float)) else 'not limited'
        message=f"The next Luna actor has not started. Additional headroom needed: {capacity['additional_headroom_needed_gib']:.1f} GiB; available shared-pool headroom including clean inactive file cache: {shared}. Waiting does not consume the episode budget."
    if active.get('status')=='running' and not running and not stopped:message='The saved process is no longer alive; its last result is recorded below.'
    links={'Study protocol':'study-manifest.json','Declared cases':'plan.json'}
    if (study/'failure-summary.html').exists():links['Task failure explanations']='failure-summary.html'
    if manifest.get('repair_report'):links['Contact repair tests']=manifest['repair_report']
    if manifest.get('superseded_by'):links['Open the new rerun']=manifest['superseded_by']
    monitor=read(study/'resource-monitor.json',{})
    resource=read(Path(monitor['latest_sample']),{}) if monitor.get('latest_sample') else {}
    health=''
    if resource:
        label='Latest experiment sample' if running else 'Last recorded experiment sample before shutdown (not a current server reading)'
        memory=resource.get('job_memory_gib')
        memory=f'{memory:.1f}' if isinstance(memory,(int,float)) else 'unavailable'
        peak=resource.get('job_memory_peak_gib')
        peak=f'{peak:.1f}' if isinstance(peak,(int,float)) else 'unavailable'
        health=f"{label}: I/O pressure {resource.get('io_full_avg10','unavailable')}%; job memory {memory} GiB, peak {peak} GiB. Watchdog sampling: every 2 seconds."
        summary={k:resource.get(k) for k in ('time','io_full_avg10','io_some_avg10','memory_full_avg10',
            'job_memory_full_avg10','job_memory_gib','job_memory_peak_gib','job_anon_and_shmem_gib',
            'job_memory_max_events','job_memory_oom_kills','host_available_memory_gib',
            'shared_headroom_including_clean_inactive_file_gib','health_read_seconds')}
        summary.update(sample_scope='memory is this experiment job; I/O pressure is host-wide',
            running=running,sample_interval_seconds=2,stop_policy=monitor.get('policy'))
        written+=_json(study/'resource-summary.json',summary)
        links['Memory and I/O measurements']='resource-summary.json'
    payload={'checked_at':time.time(),'running':running,'status_text':status+' · '+phase,
             'message':message,'totals':totals,'task_outcomes':task_outcomes,'runs':runs,'active_run':active.get('run') if running else None,
             'links':links,'health_text':health,'publish_interval_seconds':INTERVAL_SECONDS}
    written+=_json(study/'watch-status.json',payload)
    result={'published':True,'html_bytes_written':html_bytes,'client_bytes_written':client_bytes,'bytes_written':written,
            'seconds':time.perf_counter()-started,'runs':len(runs)}
    _PUBLISH[str(study)]={'at':now,'boundary':boundary}
    return result
