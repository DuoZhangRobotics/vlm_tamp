"""Summarize completed frozen evaluations; native quantities are grading only."""
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.coupled_storage import read_json,write_json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--study',default=str(ROOT/'results/clean-start-600-20261029'))
parser.add_argument('--tag',default='clean-v1')
parser.add_argument('--code-commit',required=True,help='Recorded runtime revision for the evaluated cohort')
args=parser.parse_args()
STUDY=Path(args.study).resolve()
manifest=read_json(STUDY/(args.tag+'-practice-manifest.json'))
manifest['rows']=manifest.get('evaluations',[])
practice_cases=read_json(STUDY/manifest['plan']['practice_cases'])
regressions={(case['task'],case['state_index']) for case in practice_cases}
rows=[]
for row in manifest['rows']:
    run=STUDY/'trials'/row['run']
    timing=read_json(run/'timing-breakdown.json')
    events=[json.loads(line) for line in (run/'tool-events.jsonl').read_text().splitlines()]
    plans={e['response']['result']['plan_id']:e['response']['result'] for e in events
        if e['response'].get('result',{}).get('plan_id') and e['request']['operation'].startswith('plan_')}
    def event_stages(event):
        result=event['response'].get('result',{})
        if result.get('stages'):return result['stages']
        if 'execution' not in result or 'observation' not in result:return []
        operation=event['request']['operation']
        plan=plans.get(event['request'].get('arguments',{}).get('plan_id'),{})
        return [{'stage':f"{plan['kind']}-execution" if plan.get('kind') else operation,
            'control_steps':result['observation'].get('control_steps'),'execution':result['execution'],
            'observation':result['observation']}]
    counts=Counter(e['request']['operation'] for e in events)
    stop_reasons=Counter(e.get('response',{}).get('result',{}).get('stop_reason') for e in events)
    stop_reasons.pop(None,None);stop_reasons.pop('preview_only',None)
    physical=[e for e in events if e['response'].get('result',{}).get('robot_motion_executed')]
    successes=[e for e in events if e['request']['operation'] in {'push','hook_pull','execute','execute_sequence','gripper'}
        and e['response'].get('result',{}).get('task_success')]
    success_tool=successes[0]['request']['operation'] if successes else None
    stages=event_stages(successes[0]) if successes else []
    success_stage=stages[-1]['stage'] if stages else success_tool
    diagnostics=read_json(run/'attempt01/initial/evaluation_contact_diagnostics.json')['steps']
    angles={}
    for frame in diagnostics:
        if frame['step']<60:continue
        joint=next(j for j in frame['evaluation_only_articulation_state'] if j['joint_name']=='microwave_1_microjoint')
        angles[frame['step']]=max(0.,-math.degrees(joint['position']))
    stage_timeline=[]
    for event in events:
        result=event['response'].get('result',{})
        arguments=event['request'].get('arguments',{})
        for stage in event_stages(event):
            end=stage.get('control_steps',stage.get('observation',{}).get('control_steps'))
            issued=stage.get('execution',{}).get('issued_commands')
            if end is None or issued is None:continue
            start=end-issued
            stage_timeline.append({
                'event':event['sequence'],'operation':event['request']['operation'],
                'stage':stage.get('stage'),'start_control':start,'end_control':end,
                'requested_mode':arguments.get('mode'),'requested_joint_degrees':arguments.get('joint_degrees'),
                'opening_before_degrees':angles.get(start),'opening_after_degrees':angles.get(end),
                'measured_contact_speed':stage.get('measured_contact_speed'),
                'tool_stop_reason':result.get('stop_reason'),
            })
    write_json(run/'physical-stage-timeline.json',{'offline_grading_only':True,
        'native_state_sent_to_actor_or_planner':False,'rows':stage_timeline})
    facts=[]
    if not row['official_task_success']:
        facts.append(f"Native task remained incomplete: {row['reason']} at {row['control_steps']} controls.")
        if angles:
            first,last=angles[min(angles)],angles[max(angles)]
            facts.append(f"Offline door opening: {first:.2f}° initially, {last:.2f}° finally; observed range {min(angles.values()):.2f}–{max(angles.values()):.2f}°.")
        facts.append('Recorded tool stop reasons: '+', '.join(f'{reason} ({count})' for reason,count in stop_reasons.items())+'.')
    elif success_stage:
        facts.append(f"Native success occurred during {success_stage} in a {success_tool} call.")
    success_evidence=None
    if successes and stages:
        terminal=stages[-1]
        last_control=terminal.get('control_steps',row['control_steps'])
        issued=terminal.get('execution',{}).get('issued_commands',0)
        start_control=last_control-issued
        interval=[f for f in diagnostics if start_control < f['step'] <= last_control]
        force_by_pair={}
        for frame in interval:
            for contact in frame.get('robot_involved_contacts',[]):
                force=float(contact.get('normal_force_n',0.))
                if force<=1e-6:continue
                pair=' / '.join(contact['geom_names'])
                force_by_pair[pair]=max(force_by_pair.get(pair,0.),force)
        success_result=successes[0]['response']['result']
        success_evidence={
            'selected_mode':success_result.get('mode'),
            'selected_contact_surface':success_result.get('contact',{}).get('contact_surface'),
            'requested_contact_speed_m_s':success_result.get('requested_contact_speed_m_s'),
            'success_stage_start_control':start_control,'success_stage_end_control':last_control,
            'success_stage_measured_contact_landmark_speed':terminal.get('measured_contact_speed'),
            'door_opening_before_success_stage_degrees':angles.get(start_control),
            'door_opening_at_success_degrees':angles.get(last_control),
            'peak_normal_force_by_contact_pair_during_success_stage_n':force_by_pair,
            'post_success_withdrawal_or_retention_test_in_this_episode':False,
        }
        if success_stage=='free-approach':
            peak=terminal.get('measured_contact_speed',{}).get('peak_m_s')
            facts.append('The requested contact stroke and planned release had not begun; this does not validate that requested mode.')
            if peak is not None:facts.append(f'Measured approach contact-landmark peak speed was {100*peak:.2f} cm/s.')
        target_pairs=[pair for pair in force_by_pair if 'microwave_' in pair]
        if any('robot0_link' in pair for pair in target_pairs):
            facts.append('Offline contacts include an arm link against the microwave during that stage.')
        if any('gripper0_hand_collision' in pair for pair in target_pairs):
            facts.append('Offline contacts include the palm against the microwave during that stage.')
    analysis={'native_success':row['official_task_success'],'offline_evaluation_only':True,
        'native_state_sent_to_actor_or_planner':False,'main_observed_failures':facts if not row['official_task_success'] else [],
        'interpretation':' '.join(facts),'success_tool':success_tool,'success_stage':success_stage,
        'tool_selection_counts':dict(counts),'contact_calls_with_physical_motion':len(physical),'stop_reason_counts':dict(stop_reasons),
        'success_evidence':success_evidence,
        'error_examples':[{'event':e['sequence'],'operation':e['request']['operation'],'error':e['response'].get('error'),
            'diagnostic':e['response'].get('planning_diagnostic')} for e in events if not e['response'].get('ok')]}
    write_json(run/'outcome-analysis.json',analysis)
    rows.append({'run':row['run'],'task':row['task'],'state_index':row['state_index'],
        'memory_condition':row['memory_condition'],
        'group':'development_regression' if (row['task'],row['state_index']) in regressions else 'additional_case',
        'native_success':row['official_task_success'],'controls':row['control_steps'],
        'success_by_600_controls':row['success_by_600_controls'],'reason':row['reason'],
        'actor_wall_seconds':row['actor_wall_seconds'],'service_seconds':timing['service_wall_seconds'],
        'outside_service_seconds':timing['outside_service_seconds'],'success_tool':success_tool,'success_stage':success_stage,
        'offline_door_opening_degrees':{'initial':angles[min(angles)],'final':angles[max(angles)],
            'minimum':min(angles.values()),'maximum':max(angles.values())} if angles else None,
        'tool_selection_counts':dict(counts),
        'sources_unchanged':read_json(run/'source-integrity.json')['sources_unchanged_during_run']})
report={'code_commit':args.code_commit,'manifest_file':args.tag+'-practice-manifest.json','frozen_memory_sha256':manifest['frozen_memory_sha256'],
    'planned_cases':len(manifest['plan']['later_evaluation_cases'])*len(manifest['plan']['evaluation_memory_conditions']),'completed_cases':len(rows),'cohort_completed':manifest.get('completed',False),
    'one_fresh_actor_per_case_and_condition':True,'additional_cases_not_used_in_this_practice_not_claimed_globally_unseen':True,
    'runtime_and_budgets_held_constant_between_conditions':True,'model_sampling_not_deterministically_paired':True,'rows':rows}
for group in ['development_regression','additional_case']:
    chosen=[r for r in rows if r['group']==group]
    report[group]={'completed':len(chosen),'native_successes':sum(r['native_success'] for r in chosen)}
report['memory_conditions']={}
for condition in manifest['plan']['evaluation_memory_conditions']:
    chosen=[row for row in rows if row['memory_condition']==condition]
    report['memory_conditions'][condition]={'completed':len(chosen),'native_successes':sum(row['native_success'] for row in chosen)}
write_json(STUDY/'memory-comparison-summary.json',report)
print(json.dumps({key:report[key] for key in ['completed_cases','cohort_completed','development_regression','additional_case']}))
