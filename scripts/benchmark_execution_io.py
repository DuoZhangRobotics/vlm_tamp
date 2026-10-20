"""Engineering equivalence replay: same saved references, faster I/O only."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.profiled_sim_client import ProfiledClient,execute_batch
from vlm_tamp.timed_execution import execute
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.coupled_storage import read_json,write_json,compact_episode


def array_hashes(path):
    with np.load(path) as a:
        return {k:{'sha256':hashlib.sha256(np.ascontiguousarray(a[k]).tobytes()).hexdigest(),
                   'shape':list(a[k].shape),'dtype':str(a[k].dtype)} for k in a.files}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['baseline','fast'],required=True);args=p.parse_args()
    source=(ROOT/args.source).resolve();out=(ROOT/args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    policy_files=['src/vlm_tamp/coupled_worker.py','src/vlm_tamp/timed_execution.py','src/vlm_tamp/profiled_sim_worker.py','src/vlm_tamp/profiled_sim_client.py','src/vlm_tamp/rekep_libero/sim_worker_v7.py','scripts/benchmark_execution_io.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in policy_files}
    import shutil
    for n in policy_files:
        dest=out/'source-snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    write_json(out/'execution-protocol.json',{'sources_sha256':hashes,'same_controller_references':True,'same_physics':True,'engineering_equivalence_only':True})
    original=read_json(source/'summary.json');trace=Timeline();sim=ProfiledClient(out/'simulator.log',fast=args.mode=='fast',timeline=trace)
    events=[];phases=[];captures=[];result={'mode':args.mode,'source':str(source),'task':original['task'],
        'state':original['initial_state'],'engineering_reference_replay':True,'fresh_grasp_benchmark':False,'execution_veto_checks':0}
    before=time.perf_counter()
    try:
        with trace.span('reset_and_initial_observation'):
            state=sim.reset(original['official_task_name'],original['initial_state'],out/'initial',suite_name=original['suite'])
        with trace.span('initial_open_and_observe'):
            state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,output_dir=out/'opened')
        captures.append({'label':'opened','arrays':array_hashes(out/'opened/obs.npz')})
        result['opened_sensor_arrays_match_source']=captures[-1]['arrays']==array_hashes(source/'opened/obs.npz')
        state=sim.enable_joint_tracking(kp=300.)
        original_actions=read_json(source/'initial/actions.json');offset=60
        for i,path in enumerate(sorted(source.glob('path-*/references.npy'))):
            refs=np.load(path);label=path.parent.name.split('-',2)[2]
            gripper=int(original_actions[offset]['control'][-1]);offset+=len(refs)
            target=out/path.parent.name;target.mkdir();np.save(target/'references.npy',refs)
            with trace.span('execute_segment',label=label,commands=len(refs)):
                state,done=(execute_batch if args.mode=='fast' else execute)(sim,state,refs,gripper=gripper,event_sink=events.append)
            phases.append({'label':label,**{k:v for k,v in done.items() if k!='events'}})
            write_json(out/'phases.json',phases)
            if label!='approach':
                capture=out/f'observation-{i:02d}'
                with trace.span('observation_boundary',label=label):state=sim.observe(capture)
                captures.append({'label':label,'arrays':array_hashes(capture/'obs.npz')})
        result['status']='completed'
    except Exception as error:
        result.update(status='failed',failure=str(error));(out/'error.txt').write_text(traceback.format_exc())
    finally:
        with trace.span('worker_finalize'):sim.close()
        actions=read_json(out/'initial/actions.json') if (out/'initial/actions.json').exists() else []
        expected=read_json(source/'initial/actions.json');same_length=len(actions)==len(expected)
        errors={}
        for k in ['joint_positions','gripper_qpos','control','eef_pose_world_xyzw']:
            pairs=[(a[k],b[k]) for a,b in zip(actions,expected) if k in a and k in b]
            errors[k]=max((float(np.max(np.abs(np.asarray(a)-b))) for a,b in pairs),default=0.)
        success_equal=same_length and [a['evaluation_task_success'] for a in actions]==[a['evaluation_task_success'] for a in expected]
        native=any(a['evaluation_task_success'] for a in actions)
        references=np.vstack([np.load(p) for p in sorted(out.glob('path-*/references.npy'))]) if phases else np.empty((0,7))
        issued=np.asarray([a['target_joints'] for a in actions if 'target_joints' in a]).reshape(-1,7)
        result.update(official_task_success=native,final_task_success=bool(actions and actions[-1]['evaluation_task_success']),
            control_steps=len(actions),same_command_count=same_length,evaluation_success_sequence_equal=success_equal,
            maximum_errors_versus_source=errors,all_saved_references_issued=bool(np.array_equal(references,issued)),
            execution_wall_seconds=sum(x['wall_seconds'] for x in phases),joint_commands=len(issued),wall_seconds=time.perf_counter()-before,
            controller_and_physics_equivalent=bool(same_length and success_equal and max(errors.values(),default=0)<1e-8))
        write_json(out/'tracking.json',events);write_json(out/'sensor-array-hashes.json',captures)
        write_json(out/'rpc-profile.json',sim.rpc_events);trace.save(out/'timing.json');write_json(out/'summary.json',result)
        result['storage']=compact_episode(out);write_json(out/'summary.json',result)
        if args.mode=='fast':
            from vlm_tamp.profiled_pipeline import compress_retained_observations
            compress_retained_observations(out);result=read_json(out/'summary.json')
    print({k:result[k] for k in ['mode','task','official_task_success','control_steps','controller_and_physics_equivalent','execution_wall_seconds','joint_commands','wall_seconds']})
    if not result['controller_and_physics_equivalent']:raise SystemExit(2)

if __name__=='__main__':main()
