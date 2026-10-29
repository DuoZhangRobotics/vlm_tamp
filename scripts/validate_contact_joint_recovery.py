"""Replay a measured joint-limit deadlock and test physical gripper recovery."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.contact_skill_service import qualification_sources
from vlm_tamp.mcvamp_contact_numerical import JOINT_ADMISSION_RESERVE_RAD
from vlm_tamp.profiled_sim_client import execute_batch
from replay_subscription_episode import source_commands
from serve_subscription_robot import RobotService


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True);parser.add_argument('--attempt',type=int,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();source=Path(args.source).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False);work=out/'workspace';work.mkdir()
    case=read_json(source/'protocol.json')['case']
    service=RobotService(out/'service',work,case['state_index'],1,task=case['task'],
        surface_contact_tools=True,stop_on_success=False,numerical_backend='curobo_v2',reference_mode='measured_delta')
    report={'source_attempt':args.attempt,'historical_command_replay':True,'fresh_luna_evaluation':False,
        'source_hashes':qualification_sources(),'state_assignment_used':False}
    try:
        service.dispatch('reset',{})
        for command in source_commands(source,args.attempt):
            service.state,_=execute_batch(service.sim,service.state,command['references'],gripper=command['gripper'])
        old=read_json(source/f'attempt{args.attempt:02d}/initial/actions.json')[-1]
        before=np.asarray(service.state['metadata']['joint_positions']);limits=service.ik.chain.limits.copy()
        admitted=limits+np.array([JOINT_ADMISSION_RESERVE_RAD,-JOINT_ADMISSION_RESERVE_RAD])
        report['checkpoint_joint_error_rad']=float(np.max(np.abs(before-old['joint_positions'])))
        if report['checkpoint_joint_error_rad']>1e-6:raise RuntimeError('Physical replay differs from measured checkpoint')
        report['before_joints']=before.tolist();report['admitted_limits']=admitted.tolist()
        service.state=service.sim.set_joint_reference_mode('slew_limited')
        service.calls+=1;service.capture();started=time.perf_counter()
        reply=service.dispatch('gripper',{'close':False,'steps':10})
        after=np.asarray(service.state['metadata']['joint_positions'])
        report.update(after_joints=after.tolist(),control_steps=service.state['control_steps'],
            recovered=bool(np.all(after>=admitted[:,0]) and np.all(after<=admitted[:,1])),response=reply)
        report['skill_result']={'skill':'joint_limit_recovery','mode':'physical_inward_reference',
            'stop_reason':'recovered' if report['recovered'] else 'still_outside_limits',
            'timing_seconds':{'total':time.perf_counter()-started},'diagnostic':reply['joint_limit_recovery']}
    finally:
        if service.sim:service.finish_attempt('joint_limit_recovery_diagnostic')
        write_json(out/'replay.json',report)
    print({k:report.get(k) for k in ['checkpoint_joint_error_rad','recovered','control_steps']},flush=True)


if __name__=='__main__':main()
