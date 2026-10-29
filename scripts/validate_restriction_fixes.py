"""Matched physics checks for approach timing and measured-closure planning.

Historical actor regions/paths are diagnostic inputs only. Fresh Luna runs do
not receive them. The original physical command prefix is always replayed.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import shutil
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.profiled_sim_client import execute_batch
from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.timed_execution import schedule
from vlm_tamp.grasp_observation import grasp_observation
from replay_subscription_episode import source_commands
from serve_subscription_robot import RobotService


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--ground-event',type=int,required=True)
    p.add_argument('--mode',choices=['retime','sequence'],required=True)
    p.add_argument('--grasp-constraint',choices=['point','jaw_axis','rigid'],default='point')
    p.add_argument('--reference-mode',choices=['measured_delta','slew_limited'],default='slew_limited')
    p.add_argument('--speed',type=float,default=.6);p.add_argument('--degrees',type=float,default=120.)
    a=p.parse_args();source=Path(a.source).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    work=out/'workspace';work.mkdir();case=read_json(source/'protocol.json')['case']
    hashes={}
    for path in [*ROOT.glob('src/vlm_tamp/**/*.py'),*ROOT.glob('src/vlm_tamp/native/*.cc'),
                 *ROOT.glob('scripts/*.py'),ROOT/'configs/articulated_libero.json']:
        relative=path.relative_to(ROOT);target=out/'source-snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'source-hashes.json',hashes)
    events=[json.loads(l) for l in (source/'tool-events.jsonl').read_text().splitlines()]
    event=next(e for e in events if e['sequence']==a.ground_event)
    if event['request']['operation']!='ground' or not event['response']['ok']:raise ValueError('Use a successful recorded grounding event')
    checkpoint=event['response']['result']['scene_reference_control_step']
    service=RobotService(out/'service',work,case['state_index'],1,task=case['task'],
        maximum_control_steps=1200,stop_on_success=a.mode=='sequence',surface_contact_tools=True,reference_mode='measured_delta')
    report={'mode':a.mode,'source':str(source),'historical_selection_only':True,'fresh_luna':False,
        'checkpoint':checkpoint,'maximum_control_steps':1200,'tested_reference_mode':a.reference_mode,'prefix_reference_mode':'measured_delta','requested_approach_speed_rad_s':a.speed}
    started=time.perf_counter()
    try:
        service.dispatch('reset',{})
        for command in source_commands(source,1):
            if service.state['control_steps']==checkpoint:break
            if service.state['control_steps']+len(command['references'])>checkpoint:raise ValueError('Checkpoint would cut an issued source trajectory')
            service.state,_=execute_batch(service.sim,service.state,command['references'],gripper=command['gripper'])
        states=[read_json(f) for f in (source/'agent-workspace/robot_data/attempt01').glob('obs*/robot.json')]
        expected=next(s for s in states if s['control_steps']==checkpoint)['joint_positions']
        report['prefix_joint_error_rad']=float(np.max(np.abs(np.asarray(expected)-service.state['metadata']['joint_positions'])))
        if report['prefix_joint_error_rad']>1e-6:raise RuntimeError('Physical prefix does not match recorded proprioception')
        service.state=service.sim.set_joint_reference_mode(a.reference_mode)
        service.reference_mode=a.reference_mode
        service.calls+=1;service.capture();service.dispatch('ground',event['request']['arguments'])
        if a.mode=='sequence':
            report['plan']=service.dispatch('plan_sequence',{'degrees':a.degrees,'approach_speed':a.speed,'budget_seconds':30,'grasp_constraint':a.grasp_constraint})
            report['execution']=service.dispatch('execute_sequence',{'plan_id':report['plan']['plan_id'],
                'approach_speed':a.speed,'speed':.35,'feedback_degrees':15})
        else:
            folder=source/'attempt01/013-grasp/selection'
            path=np.load(folder/'approach.npy');refs=schedule(path,maximum_joint_speed=a.speed,endpoint_hold_steps=6)
            chain=service.ik.chain;first=len(service.events);before=service.state['metadata']['joint_positions']
            service.calls+=1;report['approach']=service.issue(refs,-1,'retimed-approach')
            actual=pose_matrix(chain.forward(service.state['metadata']['joint_positions']));desired=pose_matrix(chain.forward(path[-1]))
            report.update(approach_controls=len(refs),approach_position_error_m=float(np.linalg.norm(actual[:3,3]-desired[:3,3])),
                approach_orientation_error_deg=float(np.degrees(Rotation.from_matrix(desired[:3,:3].T@actual[:3,:3]).magnitude())))
            qs=np.asarray([before,*[e['measured_joints'] for e in service.events[first:]]])
            report['measured_peak_joint_speed_rad_s']=float(np.abs(np.diff(qs,axis=0)).max()*20)
            selected=read_json(folder/'selected.json');refs=np.repeat(np.asarray(selected['joints'])[None],40,axis=0)
            service.calls+=1;report['closure']=service.issue(refs,1,'matched-closure')
            report['post_closure_grasp_evidence']=grasp_observation(service.state['metadata'],service.spec)
            refs=schedule(np.load(folder/'contact.npy'),maximum_joint_speed=.35,endpoint_hold_steps=6)
            service.calls+=1;report['probe']=service.issue(refs,0,'matched-60deg-motion',articulated=True)
        report['control_steps']=service.state['control_steps']
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        report['wall_seconds_before_cleanup']=time.perf_counter()-started
        if service.sim:service.finish_attempt('restriction_fix_diagnostic')
        initial=out/'service/attempt01/initial'
        if (initial/'actions.json').exists() or (initial/'actions.json.gz').exists():
            actions=read_json(initial/'actions.json');success=[r['step'] for r in actions if r['evaluation_task_success']]
            report.update(native_success_seen=bool(success),first_success_control_step=min(success) if success else None,
                final_task_success=bool(actions and actions[-1]['evaluation_task_success']))
        if (initial/'evaluation_contact_diagnostics.json').exists() or (initial/'evaluation_contact_diagnostics.json.gz').exists():
            rows=read_json(initial/'evaluation_contact_diagnostics.json')['steps'];angles={}
            for row in rows:
                hinge=next((j for j in row.get('evaluation_only_articulation_state',[]) if j['type']=='hinge' and not j['joint_name'].startswith(('robot','gripper'))),None)
                if hinge:angles[str(row['step'])]=float(hinge['position'])
            report['evaluation_only_articulation_positions_rad']=angles
        write_json(out/'result.json',report)
    print({k:report.get(k) for k in ['mode','requested_approach_speed_rad_s','approach_controls','approach_position_error_m','native_success_seen','control_steps','error']},flush=True)


if __name__=='__main__':main()
