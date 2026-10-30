"""Matched public-input grasp component test; no fresh actor or learned memory.

Reuses one resident model across fresh simulator episodes. Every started case
is retained. Source inputs are Luna's previously published RGB selections.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import time
import traceback

import numpy as np

from serve_subscription_robot import RobotService, ROOT
from vlm_tamp.coupled_resources import CoupledResources
from vlm_tamp.coupled_storage import read_json,write_json


def run_case(source,folder,resources,*,samples=8,method='graspgen_direct',collision_policy='veto',seed=0):
    folder.mkdir();work=folder/'workspace';work.mkdir()
    protocol=read_json(source/'protocol.json');case=protocol['case']
    service=RobotService(folder,work,case['state_index'],1,task=case['task'],stop_on_success=True)
    service.resources=resources
    service.config.update(grasp_selection_method=method,graspgen_samples=samples,
        graspgen_topk=samples,maximum_grasp_poses=samples,environment_collision_policy=collision_policy)
    events=[json.loads(x) for x in (source/'tool-events.jsonl').read_text().splitlines()]
    plan_event=next(e for e in events if e['request']['operation'] in {'plan_grasp','plan_sequence'})
    ground=next(e['request']['arguments'] for e in events if e['request']['operation']=='ground' and e['response']['ok'])
    source_args=plan_event['request']['arguments'];speed=source_args.get('approach_speed',.6)
    if plan_event['request']['operation']=='plan_grasp':speed=source_args.get('speed',.6)
    rows=[];began=time.perf_counter();result={};failure=None
    def call(op,**args):
        t=time.perf_counter();print(json.dumps({'case':folder.name,'operation':op,'phase':'start'}),flush=True)
        try:
            value=service.dispatch(op,args);response={'ok':True,'result':value};return value
        except Exception as error:
            response={'ok':False,'error':str(error),'diagnostic':getattr(error,'planning_diagnostic',None)};raise
        finally:
            row={'sequence':service.calls,'request':{'operation':op,'arguments':args},
                'response':response,'wall_seconds':time.perf_counter()-t}
            rows.append(row)
            with (folder/'tool-events.jsonl').open('a') as out:
                out.write(json.dumps(row,default=lambda x:x.tolist())+'\n')
            print(json.dumps({'case':folder.name,'operation':op,'ok':response['ok'],'seconds':row['wall_seconds']}),flush=True)
    try:
        call('reset')
        with np.load(folder/'attempt01/opened/obs.npz') as a,np.load(source/'attempt01/opened/obs.npz') as b:
            checks={k:bool(k in b and np.array_equal(a[k],b[k],equal_nan=True)) for k in a.files}
            # This component replay supplies fixed public region coordinates.
            # Grounding and GraspGen consume metric depth/point arrays, not RGB
            # intensities. Retain rendering differences without treating them
            # as changed numerical input (there is no new visual actor here).
            required=[k for k in a.files if not k.endswith('_rgb')]
            differences={k:{'changed_values':int(np.count_nonzero(a[k]!=b[k])),
                'maximum_absolute_difference':float(np.nanmax(np.abs(a[k].astype(float)-b[k].astype(float))))}
                for k in a.files if k in b and not checks[k]}
        result['initial_sensors_equal']=checks
        result['replay_input_match']={'consumed_inputs_equal':all(checks[k] for k in required),
            'required_arrays':required,'all_sensor_arrays_equal':all(checks.values()),
            'unused_rgb_differences':differences,
            'scope':'Exact metric arrays; RGB differences recorded, fixed public pixel regions, no new visual actor'}
        if not all(checks[k] for k in required):raise ValueError('Consumed sensor input mismatch; recorded selection cannot be reused')
        call('ground',**ground)
        plan=call('plan_grasp',max_candidates=samples,seed=seed,speed=speed,budget_seconds=30.)
        result['plan']=plan
        call('execute',plan_id=plan['plan_id'],gripper=-1,speed=speed,hold=6)
        call('gripper',close=True,steps=40)
        result['after_closure']={'gripper_qpos':service.state['metadata']['gripper_qpos'],
            'controls':service.state['control_steps']}
        if not service.episode_success:
            probe=call('plan_arc',degrees=10.,grasp_constraint='jaw_axis')
            result['probe_plan']=probe
            call('execute',plan_id=probe['plan_id'],gripper=1,speed=.35,hold=6)
            result['after_probe']={'gripper_qpos':service.state['metadata']['gripper_qpos'],
                'controls':service.state['control_steps']}
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}'
        (folder/'error.txt').write_text(traceback.format_exc())
    finally:
        service.finish_attempt('grasp_component_test_complete')
    result.update(error=failure,case=case,method=method,samples=samples,grasp_seed=seed,
        wall_seconds=time.perf_counter()-began,fresh_actor_evaluation=False,
        scope='Approach, pinch, then requested 10-degree probe; not a full-task evaluation',
        stages=[{'operation':r['request']['operation'],'ok':r['response']['ok'],'seconds':r['wall_seconds']} for r in rows])
    write_json(folder/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--collision-policy',choices=['advisory','veto'],default='veto')
    parser.add_argument('--reuse-grasps-from',type=Path)
    parser.add_argument('--cases',nargs='+',default=['open_microwave-state037','open_microwave-state013','close_microwave-state049'])
    args=parser.parse_args();out=(ROOT/args.output).resolve();out.mkdir(exist_ok=False)
    resources=CoupledResources(out/'grasp-cache')
    write_json(out/'protocol.json',{'samples':args.samples,'grasp_seed':args.seed,'cases':args.cases,'method':'graspgen_direct',
        'source_study':'clean-start-600-20261029','one_model_across_cases':True,
        'one_ik_and_route_per_pose':True,'additional_dense_validation':False,'maximum_control_steps':600,
        'collision_padding_m':0.,'environment_collision_policy':args.collision_policy,'all_started_cases_retained':True,
        'frozen_grasp_cache_source':str(args.reuse_grasps_from) if args.reuse_grasps_from else None})
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[ROOT/'src/vlm_tamp/native/mcvamp_contact_numerical.cc',
        ROOT/'scripts/serve_subscription_robot.py',Path(__file__).resolve()]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write_json(out/'source-hashes.json',hashes)
    results=[]
    for case in args.cases:
        if args.reuse_grasps_from:
            # finish_attempt intentionally clears temporary caches. Restore the
            # immutable archived generation, not that empty disposable cache.
            saved=next((args.reuse_grasps_from/case).glob('attempt01/*-grasp/generation'))
            provenance=read_json(saved/'input-provenance.json')
            if provenance.get('grasp_seed',0)!=args.seed:raise ValueError('Frozen generation has a different seed')
            key=provenance['cache_key']
            shutil.copytree(saved,resources.cache/('graspgen-'+key))
        source=ROOT/'results/clean-start-600-20261029/trials'/f'clean-v1-frozen-{case}-600'
        results.append(run_case(source,out/case,resources,samples=args.samples,collision_policy=args.collision_policy,seed=args.seed))
        write_json(out/'summary.json',{'completed':len(results),'planned':len(args.cases),'results':results})
        print(json.dumps({'completed':case,'error':results[-1]['error']}),flush=True)
    changed=[name for name,digest in hashes.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    write_json(out/'source-integrity.json',{'passed':not changed,'changed_files':changed})


if __name__=='__main__':main()
