"""Publish the compact live dashboard and retain optional issue summaries."""
from collections import Counter
from pathlib import Path

from report_contact_skills import read,read_events,file_stamp
from vlm_tamp.coupled_storage import write_json
_ISSUE_STAMPS={}


def observed_issues(run):
    stamp=file_stamp(run/'tool-events.jsonl')
    if _ISSUE_STAMPS.get(str(run))==stamp and (run/'observed-issues.json').exists():return
    events=read_events(run/'tool-events.jsonl')
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
    _ISSUE_STAMPS[str(run)]=stamp
    return result



def render(study, *, force=False):
    from live_study_dashboard import publish
    return publish(study,force=force)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    render(p.parse_args().study,force=True)
