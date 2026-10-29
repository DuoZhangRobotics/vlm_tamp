"""Replay issued physical commands, separating repeatability from agent decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from vlm_tamp.profiled_sim_client import ProfiledClient,execute_batch
from vlm_tamp.coupled_storage import read_json,write_json,compact_episode


def compress_observations(root):
    records=[]
    for path in Path(root).rglob('obs.npz'):
        with zipfile.ZipFile(path) as src:
            if all(z.compress_type!=zipfile.ZIP_STORED for z in src.infolist()):continue
            expected={n:hashlib.sha256(src.read(n)).hexdigest() for n in src.namelist()}
            temporary=path.with_suffix('.compressed.npz')
            with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as dst:
                for n in src.namelist():dst.writestr(n,src.read(n))
        with zipfile.ZipFile(temporary) as check:
            assert expected=={n:hashlib.sha256(check.read(n)).hexdigest() for n in check.namelist()}
        records.append({'path':str(path),'saved_bytes':path.stat().st_size-temporary.stat().st_size,
                        'uncompressed_member_sha256':expected})
        temporary.replace(path)
    return records


def source_commands(run,attempt):
    events=[json.loads(s) for s in (run/'tool-events.jsonl').read_text().splitlines()]
    current=0;commands=[]
    for event in events:
        op=event['request']['operation'];args=event['request'].get('arguments',{})
        if op=='reset' and event['response']['ok']:current+=1
        if current!=attempt or op not in {'execute','gripper','execute_sequence','push','hook_pull'} or not event['response']['ok']:continue
        if op in {'execute_sequence','push','hook_pull'}:
            for stage in event['response'].get('result',{}).get('stages',[]):
                artifact=stage['artifact_id'];name=stage['stage']
                label=f'{args["plan_id"]}-{name}' if op=='execute_sequence' else name
                path=run/f'attempt{attempt:02d}'/f'{artifact:03d}-issued-{label}.npy'
                g=({'opening':-1,'approach':-1,'closure':1,'manipulation':args.get('manipulation_gripper',1)}[name]
                   if op=='execute_sequence' else 0)
                commands.append({'event':artifact,'path':str(path),'references':np.load(path),'gripper':g})
            continue
        label=args['plan_id'] if op=='execute' else ('close' if args.get('close',True) else 'open')
        path=run/f'attempt{attempt:02d}'/f'{event["sequence"]:03d}-issued-{label}.npy'
        refs=np.load(path)
        g=args.get('gripper',1) if op=='execute' else (1 if args.get('close',True) else -1)
        commands.append({'event':event['sequence'],'path':str(path),'references':refs,'gripper':g})
    if not commands:raise ValueError('No issued trajectories')
    return commands


def source_reference_mode(run,attempt=1):
    protocol=read_json(Path(run)/'protocol.json') if (Path(run)/'protocol.json').exists() else {}
    if protocol.get('controller_reference_mode'):return protocol['controller_reference_mode']
    path=Path(run)/f'attempt{attempt:02d}/initial/controller-changes.json'
    if path.exists():
        changes=read_json(path)
        if changes and changes[0].get('reference_mode'):return changes[0]['reference_mode']
    return 'measured_delta'


def run_replays(source,attempt,output,repetitions=5):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(exist_ok=False)
    commands=source_commands(source,attempt)
    protocol=read_json(source/'protocol.json');state_index=protocol['case']['state_index']
    write_json(output/'protocol.json',{'source':str(source),'source_attempt':attempt,
        'task':'KITCHEN_SCENE7_open_the_microwave','state_index':state_index,'seed':0,
        'repetitions':repetitions,'maximum_steps':600,'initialization':'Same reset, settling, 60-step open hold, then joint servo kp=300',
        'commands':[{'event':x['event'],'gripper':x['gripper'],'references':len(x['references']),
                     'sha256':hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()} for x in commands],
        'scope':'Exact physical command replay, no agent reasoning or generalization test'})
    records=[]
    for n in range(1,repetitions+1):
        folder=output/f'replay{n:02d}';folder.mkdir();sim=None;tracking=[];error=None;checks={};executions=[]
        started=time.perf_counter()
        try:
            sim=ProfiledClient(folder/'simulator.log')
            state=sim.request('reset',task_name='KITCHEN_SCENE7_open_the_microwave',state_index=state_index,
                output_dir=str(folder/'initial'),seed=0,suite_name='libero_90',scene_variant=protocol.get('scene_variant','nominal'))
            state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,output_dir=folder/'opened')
            with np.load(folder/'opened/obs.npz') as a,np.load(source/f'attempt{attempt:02d}/opened/obs.npz') as b:
                checks={k:bool(k in b and np.array_equal(a[k],b[k],equal_nan=True)) for k in a.files}
            state=sim.enable_joint_tracking(kp=300.,reference_mode=source_reference_mode(source,attempt))
            for command in commands:
                state,result=execute_batch(sim,state,command['references'],gripper=command['gripper'],event_sink=tracking.append)
                executions.append({k:v for k,v in result.items() if k!='events'})
            sim.observe(folder/'final')
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
        finally:
            if sim:sim.close()
        actions=read_json(folder/'initial/actions.json') if (folder/'initial/actions.json').exists() else []
        expected=np.vstack([x['references'] for x in commands])
        actual=np.asarray([x['reference_joints'] for x in tracking]).reshape(-1,7)
        row={'repetition':n,'official_task_success':any(x['evaluation_task_success'] for x in actions),
             'final_task_success':bool(actions and actions[-1]['evaluation_task_success']),
             'control_steps':len(actions),'issued_references_match':bool(np.array_equal(expected,actual)),
             'initial_observation_matches_source':checks,'execution_veto_checks':0,'error':error,
             'wall_seconds':time.perf_counter()-started,'all_issued_trajectories_completed':all(x['all_commands_issued'] for x in executions)}
        write_json(folder/'tracking.json',tracking);write_json(folder/'summary.json',row)
        row['storage']=compact_episode(folder)
        row['lossless_compression']=compress_observations(folder)
        write_json(folder/'summary.json',row);records.append(row)
        write_json(output/'summary.json',{'planned':repetitions,'completed':len(records),'records':records,
            'native_successes':sum(r['official_task_success'] for r in records),
            'final_successes':sum(r['final_task_success'] for r in records)})
        print(json.dumps({'replay':n,'native_success':row['official_task_success'],'final_success':row['final_task_success'],
            'exact_references':row['issued_references_match'],'initial_equal':all(checks.values()),'error':error}),flush=True)
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--attempt',type=int,default=2)
    p.add_argument('--output',required=True);p.add_argument('--repetitions',type=int,default=5);a=p.parse_args()
    run_replays(a.source,a.attempt,a.output,a.repetitions)
