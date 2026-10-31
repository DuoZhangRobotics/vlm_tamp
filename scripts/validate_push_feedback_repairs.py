"""Matched, evaluator-only checks for short-stroke and contact-feedback repairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import threading
import time

import numpy as np

from serve_subscription_robot import RobotService
from replay_subscription_episode import source_case,source_commands,source_reference_mode
from run_restriction_study import source_hashes
from supervise_luna_dashboard_study import cgroup_path,sample,require_memory_limit,pressure_stop
from vlm_tamp.coupled_storage import read_json,write_json,compact_public_observations

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    output=args.output.resolve();output.mkdir(exist_ok=False)
    if not output.is_relative_to(ROOT/'results'):raise ValueError('Use this checkout/results')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or os.environ.get('MUJOCO_EGL_DEVICE_ID')!='3':raise ValueError('GPU/EGL 3 only')
    group=cgroup_path();require_memory_limit(group);samples=[];stop=threading.Event()
    frozen=source_hashes();write_json(output/'protocol.json',{'source':str(args.source.resolve()),
        'scope':'Recorded public requests and matched physical prefix; not fresh Luna task success',
        'source_sha256':frozen,'controller_and_collision_policy_unchanged':True})
    def monitor():
        while not stop.is_set():
            row=sample(group);samples.append(row)
            if pressure_stop(samples):
                write_json(output/'resource-stop.json',{'reason':pressure_stop(samples),'samples':samples[-3:]})
                os.kill(os.getpid(),signal.SIGTERM)
            stop.wait(2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();records=[]
    try:
        for kind,name in [('opening','baseline-r1-open_microwave-state037-600'),('closing','baseline-r1-close_microwave-state017-600')]:
            if args.case and kind!=args.case:continue
            source=args.source/'trials'/name;case=source_case(source,1)
            events={e['sequence']:e for e in map(json.loads,(source/'tool-events.jsonl').read_text().splitlines())}
            destination=output/kind;destination.mkdir();row={'case':kind,'calls':[],'passed':False}
            with tempfile.TemporaryDirectory(prefix='vlm-feedback-repair-') as temporary:
                folder=Path(temporary);workspace=folder/'agent-workspace';workspace.mkdir()
                service=RobotService(folder,workspace,state_index=case['state_index'],task=case['task'],attempts=1,
                    stop_on_success=True,surface_contact_tools=True,maximum_control_steps=600,
                    reference_mode=source_reference_mode(source))
                def call(operation,arguments):
                    started=time.perf_counter()
                    try:response={'ok':True,'result':service.dispatch(operation,arguments)}
                    except Exception as error:response={'ok':False,'error':str(error),'planning_diagnostic':getattr(error,'planning_diagnostic',{})}
                    event={'sequence':service.calls,'request':{'operation':operation,'arguments':arguments},
                        'response':response,'wall_seconds':time.perf_counter()-started}
                    with (folder/'tool-events.jsonl').open('a') as stream:
                        stream.write(json.dumps(event,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')
                    value=response.get('result',{})
                    summary={'operation':operation,'ok':response['ok'],'stop_reason':value.get('stop_reason',response.get('error')),
                        'seconds':event['wall_seconds'],'controls':service.state['control_steps'],
                        'timing_seconds':value.get('timing_seconds',{}),'search':value.get('search',{}).get('search_work',{}),
                        'part_motion_evidence':value.get('part_motion_evidence'),
                        'hand_progress_m':value.get('measured_contact_progress_m')}
                    if value.get('diagnostic'):summary['search']=value['diagnostic'].get('search_work',{})
                    row['calls'].append(summary);write_json(destination/'progress.json',row)
                    print(json.dumps({k:v for k,v in summary.items() if k not in {'part_motion_evidence','search','timing_seconds'}}),flush=True)
                    return response
                try:
                    initial=call('reset',{})
                    if not initial['ok']:raise RuntimeError(initial['error'])
                    if kind=='opening':
                        call('ground',events[3]['request']['arguments'])
                        for number in [5,8]:
                            request={**events[number]['request']['arguments'],'observation_id':service.latest_observation['observation_id'],
                                'joint_model_epoch':service.epoch}
                            call('push',request)
                        call('select_surface_contact',{'camera':'frontview','region_xyxy':[322,230,337,300]})
                        point=read_json(service.episode/f'{service.calls:03d}-surface-contact/selection.json')['point_world_m']
                        positioned=call('plan_contact_pose',{'point_world_m':point})
                        row['contact_pose_planned']=positioned['ok']
                        commands=[c for c in source_commands(source,1) if c['event']<=30]
                        for index,command in enumerate(commands):
                            service.calls+=1;service.issue(command['references'],command['gripper'],f'recorded-prefix-{index}',track_scene=False)
                        expected=next(r for r in read_json(source/'attempt01/initial/actions.json') if r['step']==251)
                        row['prefix_robot_error_rad']=float(np.max(np.abs(np.asarray(service.state['metadata']['joint_positions'])-expected['joint_positions'])))
                        if service.state['control_steps']!=251 or row['prefix_robot_error_rad']>1e-7:raise RuntimeError('Recorded physical prefix mismatch')
                        result=call('push_stroke',events[57]['request']['arguments'])['result']
                        row.update(returned_at_control=service.state['control_steps'],strokes=len(result.get('stages',[])),
                            physical_stall=result.get('stop_reason')=='physical_stall',
                            retained_controls=600-service.state['control_steps'])
                        row['passed']=bool(result['robot_motion_executed'] and service.state['control_steps']<400
                            and result.get('stop_reason') in {'physical_stall','stroke_executed','reachable_prefix_executed'})
                    else:
                        request={**events[6]['request']['arguments'],'observation_id':service.latest_observation['observation_id']}
                        result=call('push',request)['result']
                        row['part_motion_measured_after_stroke']=bool(result.get('stages',[])[-1].get('object_motion',{}).get('accepted'))
                        row['passed']=bool(result.get('robot_motion_executed') and result.get('stop_reason')=='stroke_executed')
                except Exception as error:row['error']=f'{type(error).__name__}: {error}'
                finally:
                    service.finish_attempt('diagnostic_complete')
                    compact_public_observations(workspace,finished=True,run=folder)
                    # Preserve small inspection artifacts, not RGB-D array dumps.
                    shutil.copytree(workspace,destination/'agent-workspace',ignore=shutil.ignore_patterns('*.npz','*.npy'))
                    shutil.copy2(folder/'tool-events.jsonl',destination/'tool-events.jsonl')
                    for filename in ['rollout.mp4','actions.json','evaluation_contact_diagnostics.json','video-frames.json']:
                        source_file=folder/'attempt01/initial'/filename
                        if source_file.exists():shutil.copy2(source_file,destination/filename)
                    private_path=folder/'attempt01/initial/evaluation_contact_diagnostics.json'
                    if private_path.exists():
                        private=read_json(private_path)
                        angles=[-np.degrees(next(j['position'] for j in r['evaluation_only_articulation_state'] if j['joint_name']=='microwave_1_microjoint')) for r in private['steps']]
                        if angles:row['native_open_angle_degrees']={'initial':angles[0],'final':angles[-1],'minimum':min(angles),'maximum':max(angles)}
                        row['native_angles_are_evaluator_only']=True
                    write_json(destination/'summary.json',row);records.append(row)
                    write_json(output/'summary.json',{'records':records,'passed':sum(r['passed'] for r in records)})
    finally:
        stop.set();thread.join(3)
        write_json(output/'resource-summary.json',{'samples':len(samples),
            'peak_job_memory_gib':max((s['job_memory_gib'] for s in samples),default=0),
            'maximum_io_full_avg10':max((s['io_full_avg10'] for s in samples),default=0),
            'maximum_memory_full_avg10':max((s['memory_full_avg10'] for s in samples),default=0)})
        write_json(output/'source-integrity.json',{'sources_unchanged':frozen==source_hashes()})
    return bool(records) and all(r['passed'] for r in records)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--case',choices=['opening','closing'])
    raise SystemExit(0 if run(parser.parse_args()) else 1)
