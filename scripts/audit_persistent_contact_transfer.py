"""Audit completed episodes using native logs; never send evaluator state to Luna."""
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.tool_event_audit import operation_name

RUN=ROOT/'results/persistent-contact-transfer-v2-20261020'
OUT=ROOT/'results/persistent-contact-transfer-report-20261020'


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    rows=[]
    for state in [1,11,21,31]:
        folder=RUN/f'state{state:03d}'
        if not (folder/'remembered-agent-result.json').exists():continue
        episodes=read(folder/'evaluation-private.json',{}).get('attempts',[])
        assert len(episodes)==1
        episode=episodes[0];events=[json.loads(x) for x in (folder/'tool-events.jsonl').read_text().splitlines()]
        actions=read(folder/'attempt01/initial/actions.json',[])
        native=any(a['evaluation_task_success'] for a in actions)
        assert native==episode['official_task_success']
        assert len(actions)==episode['control_steps']<=600
        first=next((i for i,a in enumerate(actions) if a['evaluation_task_success']),None)
        post_success=len(actions)-first-1 if first is not None else 0
        assert post_success==0 and episode['execution_veto_checks']==0
        counts=Counter(operation_name(x) for x in events);mode='articulated';plans={};surface_motions=[]
        diagnostics=read(folder/'attempt01/initial/evaluation_contact_diagnostics.json',{}).get('steps',[])
        angles={}
        for sample in diagnostics:
            joints=[j for j in sample.get('evaluation_only_articulation_state',[])
                    if j['type']=='hinge' and 'microjoint' in j['joint_name']]
            if joints:angles[sample['step']]=-math.degrees(joints[0]['position'])
        for event in events:
            op=operation_name(event);response=event['response'];result=response.get('result',{})
            if not response.get('ok'):continue
            if op=='ground':mode='articulated'
            elif op=='select_surface_contact':mode='surface'
            if op.startswith('plan_') and result.get('plan_id'):plans[result['plan_id']]=mode
            if op=='execute':
                token=event['request'].get('arguments',{}).get('plan_id')
                if plans.get(token)=='surface':
                    execution=result['execution'];end=result['observation']['control_steps']
                    start=end-execution['issued_commands']
                    surface_motions.append({'plan_id':token,'start_step':start,'end_step':end,
                        'issued_commands':execution['issued_commands'],
                        'evaluation_only_door_change_deg':angles[end]-angles[start]
                            if end in angles and start in angles else None,
                        'interpretation':'Observed change during the motion; not a causal contact certificate'})
        continuation=read(folder/'continuation.json',{}).get('turns',[])
        identities={t['thread_id'] for t in continuation}
        assert len(identities)==1
        protocol=read(folder/'protocol.json',{})
        workspace=Path(protocol['workspace'])
        audit=read(folder/'audit.json',{})
        assert audit['filesystem_isolation']['passed']
        assert audit['separately_billed_api_calls']==0
        assert all(r['references_match'] for r in audit['trajectory_audit'])
        client=folder/'agent-workspace/robot_data/transport-events.jsonl'
        client_events=[json.loads(x) for x in client.read_text().splitlines()] if client.exists() else []
        rows.append({'state':state,'native_success':native,'steps':len(actions),
            'reason':episode['reason'],'premature_end':episode.get('premature_end'),
            'normal_success_or_budget_end':native or episode['reason'] in
                {'control_budget_exhausted','wall_budget_exhausted'},
            'remaining_wall_seconds_at_end':episode.get('remaining_wall_seconds'),
            'same_thread_across_turns':len(identities)==1,'agent_turns':len(continuation),
            'backend_call_counts':dict(counts),'client_call_count':len(client_events),
            'duplicate_or_overlapping_calls_refused':sum(x.get('error_type')=='RequestInFlight' for x in client_events),
            'early_finish_requests_rejected':sum(operation_name(x)=='finish' and
                x['response'].get('result',{}).get('must_continue',False) for x in events),
            'surface_mode_executions':surface_motions,
            'final_door_angle_deg_evaluation_only':angles.get(len(actions)),
            'post_success_steps':post_success,'execution_veto_checks':0,
            'temporary_workspace_removed':not workspace.exists(),
            'isolation_passed':True,'api_calls':0})
    OUT.mkdir(exist_ok=True)
    (OUT/'completed-episode-audit.json').write_text(json.dumps({
        'cases':rows,'evaluation_geometry_read_only_after_episode':True,
        'evaluation_geometry_sent_to_agent':False,'prior_memory_updated':False},indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['state','native_success','steps','reason',
        'agent_turns','final_door_angle_deg_evaluation_only']} for r in rows]))


if __name__=='__main__':
    main()
