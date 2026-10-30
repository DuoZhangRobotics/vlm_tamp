"""Live study overview around the existing detailed robot timing/video report."""
from collections import Counter
import html
import json
from pathlib import Path

from report_restriction_fixes import render as render_details
from report_contact_skills import read
from vlm_tamp.coupled_storage import write_json


def observed_issues(run):
    events=[]
    try:
        for line in (run/'tool-events.jsonl').read_text().splitlines():
            try:events.append(json.loads(line))
            except ValueError:pass
    except OSError:pass
    rows=[];counts=Counter();gap_reports=[]
    for event in events:
        op=event['request']['operation'];response=event.get('response',{});value=response.get('result',{})
        if response.get('ok') is False:
            group=('grounding/observation' if op in {'ground','ground_multiview','refresh_contact','set_model','segment_parts','track_part_motion','track_video_motion'}
                else 'grasp search' if op in {'plan_grasp','plan_sequence'}
                else 'motion planning' if op in {'plan_arc','plan_pose','plan_joint'}
                else 'tool contract/execution')
            rows.append({'event':event['sequence'],'operation':op,'module':group,
                'reported_reason':response.get('error'),'diagnostic':response.get('planning_diagnostic',response.get('request_diagnostic'))})
            counts[group]+=1
        evidence=value.get('grasp_evidence') or value.get('closure_evidence') or {}
        if evidence.get('status') in {'near_closed_without_bar_width_support','gripper_gap_collapsed_during_manipulation'}:
            gap_reports.append({'event':event['sequence'],'operation':op,**evidence})
        reason=value.get('stop_reason')
        if reason:
            rows.append({'event':event['sequence'],'operation':op,'module':'skill/sequence feedback','reported_reason':reason})
    search_times=Counter()
    for path in run.glob('attempt*/*-grasp/selection/search-summary.json'):
        search_times.update(read(path,{}).get('search_timing_seconds',{}))
    result={'tool_calls':len(events),'tool_selections':dict(Counter(e['request']['operation'] for e in events)),
        'failed_requests_by_module':dict(counts),'issues':rows,'grasp_gap_reports':gap_reports,
        'all_logged_grasp_search_seconds':dict(search_times),
        'scope':'Reported issues may have been recovered from; this is evidence, not automatic root-cause attribution.'}
    write_json(run/'observed-issues.json',result)
    return result


def render(study):
    study=Path(study);render_details(study,embed=False)
    manifest=read(study/'study-manifest.json',{});rows=manifest.get('completed_runs',[])
    plan=manifest.get('plan',{});phase=manifest.get('phase','preflight')
    counts=Counter(r['stage'] for r in rows)
    intro=f'''<article><h2>Frozen full-task study</h2><p><b>Phase: {html.escape(phase)}</b>. Luna/high; 600 controls and 30 minutes per episode; GPU 3.</p>
<p>Regression: {counts['baseline']}/12 fresh no-memory actors. Practice: {counts['practice']}/1 persistent six-episode learner. Transfer: {counts['transfer']}/8 fresh paired memory/no-memory actors.</p>
<p>Only generic, current tool contracts are supplied. Evaluation notebooks are archived and never fed to subsequent actors. Held-out states are excluded from this study's practice and regression, not claimed globally unseen.</p>
<p><a href="study-manifest.json">Protocol, frozen sources and progress</a> · <a href="plan.json">Declared cases</a></p></article>'''
    if manifest.get('problem'):
        intro+='<article><h2>Run interruption</h2><pre>'+html.escape(json.dumps(manifest['problem'],indent=2))+'</pre></article>'
    grouped={}
    for row in rows:
        run=study/'trials'/row['run']
        if run.exists():observed_issues(run)
        if row['stage']!='baseline':continue
        key=(row['task'],row['state_index']);grouped.setdefault(key,[]).append(row)
    if grouped:
        intro+='<article><h2>Regression results</h2><table><tr><th>Case</th><th>Successes / completed</th><th>Controls</th></tr>'
        for (task,state),items in grouped.items():
            intro+=f'<tr><td>{html.escape(task)} / {state}</td><td>{sum(x["official_task_success"] for x in items)}/{len(items)}</td><td>'+', '.join(str(x['control_steps']) for x in items)+'</td></tr>'
        intro+='</table></article>'
    active=read(study/'active-process.json',{})
    if active.get('status')=='running':
        run=study/'trials'/active['run'];status=read(run/'live-episode-status.json',{})
        intro+='<article><h2>Current actor</h2><p>'+html.escape(active['run'])+'</p><pre>'+html.escape(json.dumps(status,indent=2))+'</pre>'
        frames=sorted(run.glob('attempt*/initial/frames/step*.jpg'))
        if frames:
            frame=frames[-2] if len(frames)>1 else frames[-1]
            intro+=f'<img src="{frame.relative_to(study)}" alt="Latest recorded robot view">'
        intro+='</article>'
    page=study/'watch.html';text=page.read_text();at=text.index('</style>')+len('</style>')
    text=text[:at]+intro+text[at:]
    if not manifest.get('completed'):
        text+='''<script>setTimeout(()=>{if(!Array.from(document.querySelectorAll('video')).some(v=>!v.paused))location.reload();},15000);</script>'''
    page.write_text(text)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);render(p.parse_args().study)
