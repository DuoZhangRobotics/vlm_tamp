"""Summarize completed transfer evidence, with simulator truth evaluation-only."""
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.tool_event_audit import operation_name
RUN=ROOT/'results/shared-tools-transfer-20261020'
OUT=ROOT/'results/shared-tools-transfer-report-20261020'


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    rows=[]
    for state in [1,11,21,31]:
        folder=RUN/'luna'/f'state{state:03d}'
        summary=read(folder/'attempt01/summary.json')
        if summary is None or not (folder/'remembered-agent-result.json').exists():continue
        events=[json.loads(line) for line in (folder/'tool-events.jsonl').read_text().splitlines()]
        counts=Counter(operation_name(event) for event in events)
        failures=[];support=[];requests=[];plans={}
        for event in events:
            request=event['request'];response=event['response'];result=response.get('result',{})
            op=operation_name(event)
            if op.startswith('plan_'):
                requests.append({'operation':op,'arguments':request.get('arguments',{}),
                    'rpc_ok':response.get('ok'),'planned':result.get('planned',response.get('ok'))})
                if response.get('ok') and result.get('plan_id'):
                    plans[result['plan_id']]={'operation':op,**request.get('arguments',{})}
            if not response.get('ok') or result.get('planned') is False:
                failures.append({'operation':op,'error':response.get('error'),
                    'planning_status':response.get('planning_diagnostic',{}).get('status',
                        result.get('result',{}).get('status'))})
            if op=='inspect_scene' and response.get('ok'):
                support.append(result['evidence'])
        diagnostics=read(folder/'attempt01/initial/evaluation_contact_diagnostics.json',{}).get('steps',[])
        final_articulation=diagnostics[-1].get('evaluation_only_articulation_state',[]) if diagnostics else []
        door=[x for x in final_articulation if x['type']=='hinge' and 'microjoint' in x['joint_name']]
        angles={}
        for sample in diagnostics:
            joints=[x for x in sample.get('evaluation_only_articulation_state',[])
                    if x['type']=='hinge' and 'microjoint' in x['joint_name']]
            if joints:angles[sample['step']]=abs(math.degrees(joints[-1]['position']))
        robot_calls=read(folder/'remembered-agent-result.json',{})
        updates=[]
        for file in sorted((folder/'attempt01').glob('*-scene-pose-update.json')):
            record=read(file)
            updates.append({'source':record.get('source'),'control_step':record.get('reference_control_step'),
                            'hand_predicted_angle_deg':math.degrees(record.get('hand_predicted_angle_rad',0.))})
        executions=read(folder/'attempt01/executions.json',[])
        step=summary['control_steps']-sum(x['issued_commands'] for x in executions)
        motion=[]
        for execution in executions:
            end=step+execution['issued_commands'];before=angles.get(step);after=angles.get(end)
            estimate=next((x for x in updates if x['control_step']==end),{})
            motion.append({'label':execution['label'],'start_step':step,'end_step':end,
                'planned_commands':execution['planned_commands'],'issued_commands':execution['issued_commands'],
                'requested_angle_deg':plans.get(execution['label'].split('-')[0],{}).get('degrees'),
                'maximum_joint_reference_error_rad':execution.get('maximum_joint_reference_error_rad'),
                'execution_veto_checks':execution.get('execution_veto_checks'),
                'all_commands_issued':execution.get('all_commands_issued'),
                'evaluation_only_door_delta_deg':after-before if before is not None and after is not None else None,
                'hand_predicted_angle_deg':estimate.get('hand_predicted_angle_deg')})
            step=end
        row={'state':state,'native_success':summary['official_task_success'],
            'control_steps':summary['control_steps'],'physical_segments':summary['requested_trajectories'],
            'tool_call_counts':dict(counts),'planning_requests':requests,'failed_queries':failures,
            'scene_inspections':support,'scene_updates':updates,'executed_motion':motion,
            'evaluation_only_final_door_angle_deg':abs(math.degrees(door[-1]['position'])) if door else None,
            'separately_billed_api_calls':robot_calls.get('api_calls'),
            'isolation_passed':robot_calls.get('isolation_passed'),
            'agent_report':str(folder/'agent-last.txt')}
        rows.append(row)
    OUT.mkdir(exist_ok=True)
    (OUT/'transfer-evidence-audit.json').write_text(json.dumps({
        'cases':rows,'articulation_truth_read_after_episode':True,
        'evaluation_truth_supplied_to_agent':False,'policy_or_memory_updated':False},indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['state','native_success','control_steps',
        'evaluation_only_final_door_angle_deg','isolation_passed']} for r in rows]))


if __name__=='__main__':
    main()
