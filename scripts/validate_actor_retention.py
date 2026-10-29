"""Separate diagnostic: replay an actor success, withdraw, and observe rebound.

Normal Luna success stopping is unchanged. The withdrawal uses the actor's
recorded public push direction and calibrated robot kinematics. Native contacts
and success labels are read only afterwards for grading.
"""
import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.contact_skills import Budget,cartesian_path,unit,measured_contact_speed
from vlm_tamp.profiled_sim_client import execute_batch
from vlm_tamp.rekep_libero.geometry import pose_matrix,matrix_pose
from vlm_tamp.timed_execution import schedule_contact
from replay_subscription_episode import source_commands,source_reference_mode,source_case
from serve_subscription_robot import RobotService


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--attempt',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--withdrawal-distance',type=float,default=.015)
    p.add_argument('--planning-seconds',type=float,default=5.);a=p.parse_args()
    if not np.isfinite([a.withdrawal_distance,a.planning_seconds]).all() or min(a.withdrawal_distance,a.planning_seconds)<=0:
        raise ValueError('Withdrawal distance and planning budget must be finite and positive')
    source=Path(a.run).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    work=out/'workspace';work.mkdir();case=source_case(source,a.attempt)
    import json
    attempt=0;success=None
    for line in (source/'tool-events.jsonl').read_text().splitlines():
        e=json.loads(line);op=e['request']['operation'];response=e['response']
        if op=='reset' and response['ok']:attempt+=1
        r=response.get('result',{})
        if attempt==a.attempt and op=='push' and r.get('task_success'):success=r
    if success is None:raise ValueError('No successful public push in the selected actor attempt')
    service=RobotService(out/'service',work,case['state_index'],1,task=case['task'],
        surface_contact_tools=True,stop_on_success=False,numerical_backend='curobo_v2',
        reference_mode=source_reference_mode(source,a.attempt),maximum_control_steps=case['maximum_control_steps']+1000)
    report={'source_run':str(source),'source_attempt':a.attempt,'diagnostic_only':True,
        'normal_luna_termination_changed':False,'source_reference_mode':source_reference_mode(source,a.attempt),
        'diagnostic_control_allowance':case['maximum_control_steps']+1000,'simulator_state_assignment_used':False,
        'withdrawal_m':a.withdrawal_distance,'coasting_seconds':1.,'native_labels_used_only_for_posthoc_grading':True}
    stages=[];timing=Counter();began=time.perf_counter()
    try:
        service.dispatch('reset',{})
        for command in source_commands(source,a.attempt):
            service.state,_=execute_batch(service.sim,service.state,command['references'],gripper=command['gripper'])
        old=read_json(source/f'attempt{a.attempt:02d}/initial/actions.json')[-1]
        before=np.asarray(service.state['metadata']['joint_positions'])
        error=float(np.max(np.abs(before-old['joint_positions'])))
        report.update(prefix_joint_error_rad=error,prefix_control_step=service.state['control_steps'])
        if error>1e-6:raise RuntimeError('Physical prefix no longer matches the recorded actor state')
        chain=service.ik.chain;contact=np.asarray(success['contact']['contact_local_m'])
        direction=unit(success['contact']['direction_world']);target=pose_matrix(chain.forward(before))
        target[:3,3]-=a.withdrawal_distance*direction;report['withdrawal_direction_world']=(-direction).tolist()
        phase=time.perf_counter();nested=Counter()
        path=cartesian_path(service.ik,SimpleNamespace(skill_collision_policy='advisory'),service.record,
            service.state['metadata'],before,matrix_pose(target),Budget(a.planning_seconds),timing=nested)
        timing['withdrawal_planning_seconds']=time.perf_counter()-phase
        references=schedule_contact(path,chain,contact,contact_speed_m_s=.03)
        service.calls+=1;phase=time.perf_counter();previous=len(service.events)
        reply=service.issue(references,0,'push-release-withdrawal',track_scene=False,
            camera_names=['agentview','frontview','arm_wide_back'])
        timing['withdrawal_execution_and_capture_seconds']=time.perf_counter()-phase
        stages.append({'stage':'push-release-withdrawal','artifact_id':service.calls,
            'control_steps':service.state['control_steps'],'execution':reply['execution'],
            'requested_contact_speed_m_s':.03,'planning_timing_seconds':dict(nested),
            'measured_contact_speed':measured_contact_speed(service.events[previous:],chain,contact,before)})
        report['coast_start_control_step']=service.state['control_steps']
        q=np.asarray(service.state['metadata']['joint_positions']);refs=np.repeat(q[None],20,axis=0)
        service.calls+=1;phase=time.perf_counter()
        reply=service.issue(refs,0,'push-release-coasting',track_scene=False,camera_names=['agentview','frontview','arm_wide_back'])
        timing['coasting_execution_and_capture_seconds']=time.perf_counter()-phase
        stages.append({'stage':'push-release-coasting','artifact_id':service.calls,
            'control_steps':service.state['control_steps'],'execution':reply['execution']})
        report.update(control_steps=service.state['control_steps'],settling_control_steps=20)
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['skill_result']={'skill':'push','mode':'actor_success_retention','stages':stages,
            'requested_contact_speed_m_s':.03,'collision_policy':'advisory',
            'timing_seconds':{'total_with_replay':time.perf_counter()-began,**dict(timing)}}
        if service.sim:service.finish_attempt('actor_retention_diagnostic')
        initial=out/'service/attempt01/initial'
        actions=read_json(initial/'actions.json') if (initial/'actions.json').exists() or (initial/'actions.json.gz').exists() else []
        contacts=(read_json(initial/'evaluation_contact_diagnostics.json')['steps']
            if (initial/'evaluation_contact_diagnostics.json').exists() or (initial/'evaluation_contact_diagnostics.json.gz').exists() else [])
        successful=[r['step'] for r in actions if r['evaluation_task_success']]
        force_contacts=[{'step':r['step'],'geoms':c['geom_names'],'normal_force_n':c['normal_force_n']}
            for r in contacts if r['step']>report.get('coast_start_control_step',float('inf'))
            for c in r['robot_involved_contacts'] if sum(c['robot_flags'])==1 and c['normal_force_n']>0]
        report.update(first_native_success_step=min(successful) if successful else None,
            native_success_seen=bool(successful),final_task_success=bool(actions and actions[-1]['evaluation_task_success']),
            force_bearing_robot_environment_contacts_during_coast=force_contacts,
            contact_free_coast=bool(report.get('settling_control_steps')==20 and not force_contacts))
        report['retained_success_verified']=bool(report.get('settling_control_steps')==20 and report['final_task_success'])
        report['skill_result']['stop_reason']='retained_after_withdrawal' if report['retained_success_verified'] else 'closure_not_retained_or_diagnostic_incomplete'
        write_json(out/'replay.json',report)
    print({k:report[k] for k in ['source_attempt','prefix_joint_error_rad','first_native_success_step','final_task_success','contact_free_coast']},flush=True)


if __name__=='__main__':main()
