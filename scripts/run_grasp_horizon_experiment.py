"""Matched physical horizon/attachment ablation using a verified grasp prefix.

This is a numerical-component experiment, not a fresh agent evaluation.
The source grasp is replayed only after matching all initial sensor arrays.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
import numpy as np
from serve_subscription_robot import RobotService
from replay_subscription_episode import source_commands,compress_observations
from vlm_tamp.coupled_storage import read_json,write_json

CONDITIONS={
    'baseline-15':dict(degrees=15,grasp_constraint='point',scene_pose_sync='none'),
    'synced-point-15':dict(degrees=15,grasp_constraint='point',scene_pose_sync='rgbd'),
    'synced-point-90':dict(degrees=90,grasp_constraint='point',scene_pose_sync='rgbd'),
    'synced-rigid-15':dict(degrees=15,grasp_constraint='rigid',scene_pose_sync='rgbd'),
    'synced-rigid-90':dict(degrees=90,grasp_constraint='rigid',scene_pose_sync='rgbd'),
    'synced-jaw-90':dict(degrees=90,grasp_constraint='jaw_axis',scene_pose_sync='rgbd'),
}


def run_case(source,folder,condition,repetition,*,seconds=20.,proposals=32,speed=.35):
    folder.mkdir()
    work=folder/'temporary-workspace';work.mkdir()
    service=None;error=None;began=time.perf_counter();plans=[];checks={}
    settings=CONDITIONS[condition]
    events=[json.loads(x) for x in (source/'tool-events.jsonl').read_text().splitlines()]
    ground=next(e['request']['arguments'] for e in events if e['request']['operation']=='ground')
    first_arc=next(e['sequence'] for e in events if e['request']['operation']=='plan_arc')
    commands=[c for c in source_commands(source,1) if c['event']<first_arc]
    source_query=source/f'attempt01/{first_arc:03d}-arc/planning-input.json'
    expected=read_json(source_query)
    protocol=read_json(source/'protocol.json')
    try:
        service=RobotService(folder,work,protocol['case']['state_index'],1,stop_on_success=True)
        service.config.update(scene_pose_sync=settings['scene_pose_sync'],
            grasp_constraint=settings['grasp_constraint'],arc_search_seconds=seconds,arc_goal_proposals=proposals)
        def call(op,**kwargs):
            start=time.perf_counter()
            try:
                result=service.dispatch(op,kwargs);response={'ok':True,'result':result}
                return result
            except Exception as exc:
                response={'ok':False,'error':str(exc)};raise
            finally:
                with (folder/'tool-events.jsonl').open('a') as f:
                    f.write(json.dumps({'sequence':service.calls,'request':{'operation':op,'arguments':kwargs},
                        'response':response,'wall_seconds':time.perf_counter()-start},
                        default=lambda x:x.tolist())+'\n')
        call('reset')
        with np.load(folder/'attempt01/opened/obs.npz') as a,np.load(source/'attempt01/opened/obs.npz') as b:
            checks={k:bool(k in b and np.array_equal(a[k],b[k],equal_nan=True)) for k in a.files}
        if not all(checks.values()):raise ValueError('Initial sensor arrays differ; frozen grasp replay forbidden')
        call('ground',**ground)
        for command in commands:
            service.calls+=1
            service.issue(command['references'],command['gripper'],'matched-prefix')
        error_q=float(np.max(np.abs(np.asarray(service.state['metadata']['joint_positions'])-expected['reference_joints'])))
        write_json(folder/'matched-start.json',{'sensor_arrays_equal':checks,'maximum_postgrasp_joint_difference_rad':error_q,
            'source_query':str(source_query),'source_prefix':[{'event':c['event'],'sha256':hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()} for c in commands]})
        if error_q>1e-5:raise ValueError('Measured postgrasp joints differ from the matched source')
        for n in range(12):
            if service.episode_success:break
            planned=call('plan_arc',degrees=settings['degrees'],refined=True,grasp_constraint=settings['grasp_constraint'])
            plans.append({'plan_id':planned['plan_id'],'result':planned['result']})
            call('execute',plan_id=planned['plan_id'],gripper=1,speed=speed,hold=6)
        if not service.episode_success:error='no_success_after_12_complete_segments'
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        (folder/'error.txt').write_text(traceback.format_exc())
    finally:
        if service is not None:service.finish_attempt('diagnostic_complete' if error is None else 'diagnostic_failure')
        before=sum(p.stat().st_size for p in work.rglob('*') if p.is_file())
        shutil.rmtree(work)
        compressed=compress_observations(folder)
        write_json(folder/'cleanup.json',{'removed_temporary_workspace_bytes':before,
            'observation_compression':compressed,'source_results_modified':False})
    result=read_json(folder/'attempt01/summary.json') if (folder/'attempt01/summary.json').exists() else {'official_task_success':False,'control_steps':0}
    result.update(condition=condition,repetition=repetition,configuration=settings,error=error,
        wall_seconds=time.perf_counter()-began,successful_plans=len(plans),physical_execution=True,
        fresh_agent_evaluation=False,matched_initial_sensors=bool(checks and all(checks.values())))
    write_json(folder/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='results/subscription-success-first-continuation-20261019')
    p.add_argument('--output',required=True)
    p.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS))
    p.add_argument('--repetitions',type=int,default=1)
    p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--proposals',type=int,default=32)
    p.add_argument('--speed',type=float,default=.35)
    args=p.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'results') or out.exists():raise ValueError('Use a new results directory')
    out.mkdir()
    manifest=[{'condition':c,'repetition':r} for r in range(1,args.repetitions+1) for c in args.conditions]
    write_json(out/'protocol.json',{'manifest':manifest,'source':str((ROOT/args.source).resolve()),
        'conditions':CONDITIONS,'maximum_steps':600,'stop_on_simulator_success':True,
        'execution_time_collision_veto':False,'api_calls':0,'seconds':args.seconds,'goal_proposals':args.proposals,
        'speed':args.speed,'scope':'Matched component ablation; saved grasp prefix, no new agent decisions',
        'every_started_attempt_retained':True})
    rows=[]
    for entry in manifest:
        print(json.dumps({'event':'started',**entry}),flush=True)
        result=run_case(ROOT/args.source,out/f'{entry["condition"]}-r{entry["repetition"]}',**entry,
            seconds=args.seconds,proposals=args.proposals,speed=args.speed)
        rows.append(result);write_json(out/'summary.json',{'planned':len(manifest),'completed':len(rows),'episodes':rows})
        print(json.dumps({'event':'completed',**entry,'success':result['official_task_success'],
            'steps':result['control_steps'],'error':result['error']}),flush=True)


if __name__=='__main__':main()
