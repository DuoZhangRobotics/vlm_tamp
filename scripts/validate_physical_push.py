"""Replay a public robot-command prefix, then test current-hand physical pushes.

Evaluator-only: no Luna calls, object geometry, saved grasp solution, or object
pose model enters the new stroke. The input prefix establishes a matched physical
state. These measurements are local repeatability, not fresh-policy task success.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from serve_subscription_robot import RobotService
from replay_subscription_episode import source_commands,source_case,source_reference_mode
from supervise_luna_dashboard_study import sample,cgroup_path,require_memory_limit
from run_restriction_study import source_hashes
from vlm_tamp.coupled_storage import read_json,write_json,compact_public_observations

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    source=args.source.resolve();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'results'):raise ValueError('Diagnostic output must be under this checkout/results')
    group=cgroup_path();limit=require_memory_limit(group)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or os.environ.get('MUJOCO_EGL_DEVICE_ID')!='3':raise ValueError('GPU/EGL device 3 only')
    events={e['sequence']:e for e in (json.loads(line) for line in (source/'tool-events.jsonl').read_text().splitlines())}
    case=source_case(source,1)
    commands=[c for c in source_commands(source,1) if c['event']<=args.through_event]
    event=events[args.through_event];observation=event['response']['result']['observation']['observation_id']
    expected=read_json(source/f'attempt01/{observation:03d}-observation/metadata.json')
    direction=(args.direction if args.direction is not None else
        events[args.direction_event]['request']['arguments']['direction_world'])
    direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
    output.mkdir(exist_ok=False)
    frozen=source_hashes()
    write_json(output/'protocol.json',{'scope':'matched prefix and physical push diagnostic; no fresh-policy claim',
        'source':str(source),'case':case,'through_event':args.through_event,
        'prefix_sha256':[hashlib.sha256(Path(c['path']).read_bytes()).hexdigest() for c in commands],
        'distance_m':args.distance,'maximum_strokes':args.strokes,'repeats':args.repeats,
        'direction_world':direction.tolist(),'expected_outcome':args.expect,'memory_limit_bytes':limit,
        'object_model_supplied_to_stroke':False,'new_model_calls':0,'source_sha256':frozen})
    stop=threading.Event();samples=[]
    def monitor():
        while not stop.is_set():samples.append(sample(group));stop.wait(2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();rows=[]
    try:
        for repeat in range(1,args.repeats+1):
            folder=output/f'replay{repeat:02d}';folder.mkdir();workspace=folder/'agent-workspace';workspace.mkdir()
            service=RobotService(folder,workspace,state_index=case['state_index'],task=case['task'],attempts=1,
                stop_on_success=args.expect=='success',surface_contact_tools=True,maximum_control_steps=600,
                reference_mode=source_reference_mode(source))
            row={'repeat':repeat,'calls':[],'passed':False}
            try:
                service.dispatch('reset',{})
                for index,command in enumerate(commands):
                    service.calls+=1
                    service.issue(command['references'],command['gripper'],f'recorded-prefix-{index}',track_scene=False)
                error=float(np.max(np.abs(np.asarray(service.state['metadata']['joint_positions'])-expected['joint_positions'])))
                row['initial_robot_max_error_rad']=error;row['initial_control_step']=service.state['control_steps']
                if error>1e-7:raise ValueError('The physical prefix did not reproduce the recorded robot state')
                if service.active_push_contact is not None or service.scene is not None:raise ValueError('Direct stroke unexpectedly inherited an object/contact model')
                for index in range(args.strokes):
                    request={'direction_world':direction.tolist(),'distance_m':args.distance,
                        'contact_speed_m_s':.03,'budget_seconds':120.,'stroke_plan_seconds':15.}
                    began=time.perf_counter();result=service.dispatch('push_stroke',request)
                    row['calls'].append(result)
                    with (folder/'tool-events.jsonl').open('a') as stream:
                        json.dump({'sequence':service.calls,'request':{'operation':'push_stroke','arguments':request},
                            'response':{'ok':True,'result':result},'wall_seconds':time.perf_counter()-began},stream,
                            default=lambda value:value.tolist() if hasattr(value,'tolist') else str(value));stream.write('\n')
                    write_json(folder/'progress.json',row)
                    print({'repeat':repeat,'stroke':index+1,'controls':service.state['control_steps'],
                        'stop':result.get('stop_reason'),'hand_m':result.get('measured_contact_progress_m'),
                        'task_success':result.get('task_success'),'seconds':result.get('timing_seconds',{}).get('total')},flush=True)
                    if result.get('task_success') or result.get('episode_done') or result.get('stop_reason')=='physical_stall':break
                row['native_task_success']=any(r.get('task_success') for r in row['calls'])
                row['passed']=(row['native_task_success'] if args.expect=='success' else
                    bool(row['calls'] and row['calls'][-1].get('stop_reason')=='physical_stall'
                        and row['calls'][-1].get('robot_motion_executed')))
            except Exception as error:row['error']=f'{type(error).__name__}: {error}'
            finally:
                service.finish_attempt('diagnostic_complete')
                compact_public_observations(workspace,finished=True,run=folder)
                rows.append(row);write_json(folder/'summary.json',row)
                write_json(output/'summary.json',{'completed':len(rows),'passed':sum(r['passed'] for r in rows),'records':rows})
            if not row['passed']:break
    finally:
        stop.set();thread.join(3);write_json(output/'resource-samples.json',samples)
        write_json(output/'resource-summary.json',{'samples':len(samples),
            'peak_job_memory_gib':max((s['job_memory_gib'] for s in samples),default=0),
            'maximum_io_full_avg10':max((s['io_full_avg10'] for s in samples),default=0),
            'maximum_memory_full_avg10':max((s['memory_full_avg10'] for s in samples),default=0)})
    unchanged=frozen==source_hashes();write_json(output/'source-integrity.json',{'sources_unchanged':unchanged})
    return unchanged and len(rows)==args.repeats and all(r['passed'] for r in rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--through-event',type=int,required=True);parser.add_argument('--direction-event',type=int)
    parser.add_argument('--direction',type=float,nargs=3);parser.add_argument('--distance',type=float,default=.1)
    parser.add_argument('--strokes',type=int,default=5);parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--expect',choices=['success','stall'],default='success');args=parser.parse_args()
    if args.direction is None and args.direction_event is None:parser.error('Supply --direction or --direction-event')
    raise SystemExit(0 if run(args) else 1)
