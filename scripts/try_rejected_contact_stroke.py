"""User-authorized simulation experiment: execute a scene-rejected 3 cm stroke.

Environment collision predictions are logged, never used as an execution veto.
The physical simulator retains its contacts, dynamics and joint actuators.
This is separate from the actor's validated contact-skill interface.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from serve_subscription_robot import RobotService
from replay_subscription_episode import source_commands,source_reference_mode
from vlm_tamp.contact_skills import build_contact_scene,predicted_scene,skill_native_scene,measured_contact_speed
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.profiled_sim_client import execute_batch
from vlm_tamp.rekep_libero.geometry import pose_matrix,matrix_pose
from vlm_tamp.timed_execution import schedule_contact

ORIGINAL=ROOT/'results/perception-contract-repair-20261028/trials/close_microwave-state017-perception-repaired'
PREPARED=ROOT/'results/contact-skills-20261028/zero-padding-v1-push-push_and_release-12cms-r1'


def robot_only_path(ik,start,delta):
    """Continuous Cartesian IK; environment geometry cannot veto this test."""
    transform=pose_matrix(ik.chain.forward(start));path=[np.asarray(start)];timing=[]
    for fraction in np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(delta)/.003)))+1)[1:]:
        pose=transform.copy();pose[:3,3]+=fraction*delta;began=time.perf_counter()
        solutions=ik.solve_many(matrix_pose(pose),path[-1])
        if not solutions:raise RuntimeError(f'No robot-valid IK at stroke fraction {fraction:.3f}')
        solution=min(solutions,key=lambda s:np.linalg.norm(np.asarray(s['joints'])-path[-1]))
        path.append(np.asarray(solution['joints']));timing.append(time.perf_counter()-began)
    return np.asarray(path),{'ik_seconds':sum(timing),'queries':len(timing),
        'environment_collision_veto':False,'robot_joint_bounds_and_self_checks_retained':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--speed',type=float,default=.12);parser.add_argument('--distance',type=float,default=.03)
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if not 0<args.distance<=.03 or not 0<args.speed<=.12:raise ValueError('This experiment uses a stroke up to 3 cm and reference speed up to 12 cm/s')
    work=out/'workspace';work.mkdir();service=RobotService(out/'service',work,17,1,
        task='close_microwave',surface_contact_tools=True,stop_on_success=False,reference_mode=source_reference_mode(ORIGINAL))
    original=read_json(PREPARED/'replay.json')['skill_result'];contact=original['contact']
    direction=np.asarray(contact['direction_world']);direction/=np.linalg.norm(direction)
    local=np.asarray(contact['contact_local_m']);stages=[];started=time.perf_counter()
    result={'skill':'push','mode':'simulation_collision_veto_bypassed','preview':False,
        'requested_contact_speed_m_s':args.speed,'requested_distance_m':args.distance,
        'robot_motion_executed':False,'environment_collision_veto':False,
        'simulator_physical_contacts_enabled':True,'added_collision_padding_m':0.,'stages':stages,
        'contact':contact,'task_success':False}
    record={'checkpoint':459,'prepared_control_step':523,'fresh_luna_evaluation':False,
        'explicit_user_authorization':'Execute the rejected 3 cm stroke in simulation instead of treating predicted contact as a veto',
        'skill_result':result,'source_preparation':str(PREPARED)}
    source=out/'source-snapshot';source.mkdir()
    for relative in ['scripts/try_rejected_contact_stroke.py','src/vlm_tamp/timed_execution.py',
                     'src/vlm_tamp/coupled_worker.py','src/vlm_tamp/contact_skills.py']:
        dest=source/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dest)
    try:
        service.dispatch('reset',{})
        for command in source_commands(ORIGINAL,1):
            if service.state['control_steps']==459:break
            if service.state['control_steps']+len(command['references'])>459:raise RuntimeError('Prefix must end on an original trajectory boundary')
            service.state,_=execute_batch(service.sim,service.state,command['references'],gripper=command['gripper'])
        for stage in original['stages']:
            if stage['stage'] not in {'free-approach','contact-insertion'}:continue
            path=PREPARED/f'service/attempt01/{stage["artifact_id"]:03d}-issued-{stage["stage"]}.npy'
            service.state,_=execute_batch(service.sim,service.state,np.load(path),gripper=0)
            expected=read_json(PREPARED/f'service/attempt01/{stage["artifact_id"]:03d}-observation/metadata.json')
        error=float(np.max(abs(np.asarray(service.state['metadata']['joint_positions'])-expected['joint_positions'])))
        record['prepared_joint_error_rad']=error
        if service.state['control_steps']!=523 or error>1e-6:raise RuntimeError('The prepared physical state was not reproduced')
        record['source_reference_mode']=source_reference_mode(ORIGINAL)
        service.state=service.sim.set_joint_reference_mode('slew_limited')
        record['tested_reference_mode']='slew_limited'
        service.calls+=1;result['initial_observation']=service.capture()
        chain=service.ik.chain
        path,planning=robot_only_path(service.ik,service.state['metadata']['joint_positions'],args.distance*direction)
        shadow=[];shadow_started=time.perf_counter()
        try:
            scene,_,_=build_contact_scene(service.cameras,service.state['metadata'],chain,contact['camera'],
                contact['region_xyxy'],contact.get('target_region_xyxy'),'fingertips',direction)
            for fraction,q in zip(np.linspace(0,1,len(path)),path):
                predicted=predicted_scene(scene,args.distance*direction*fraction)
                with skill_native_scene(predicted,service.record,service.state['metadata'],path[0]) as checker:
                    check=checker.validate(q)
                shadow.append({'stroke_fraction':float(fraction),**check,'execution_veto_applied':False})
        except Exception as error:
            shadow.append({'diagnostic_error':str(error),'execution_veto_applied':False})
        result['shadow_collision_checks']=shadow
        result['shadow_collision_seconds']=time.perf_counter()-shadow_started
        result['ignored_collision_predictions']=sum(not r.get('accepted',True) for r in shadow)

        def issue(references,label):
            service.calls+=1;before=len(service.events);q=service.state['metadata']['joint_positions']
            reply=service.issue(references,0,label,track_scene=False,camera_names=['agentview','arm_wide_back','frontview'])
            stages.append({'stage':label,'artifact_id':service.calls,'control_steps':service.state['control_steps'],
                'execution':reply['execution'],'measured_contact_speed':measured_contact_speed(service.events[before:],chain,local,q)})
            result['observation']=reply['observation'];result['robot_motion_executed']=True

        # Two endpoint controls let the unchanged position servo approach its
        # commanded endpoint; no contact/progress/collision veto is consulted.
        references=schedule_contact(path,chain,local,contact_speed_m_s=args.speed,endpoint_hold_steps=2)
        np.save(out/'requested-stroke-path.npy',path)
        initial=pose_matrix(chain.forward(service.state['metadata']['joint_positions']))
        initial_point=initial[:3,3]+initial[:3,:3]@local
        issue(references,'three-centimetre-stroke')
        after=pose_matrix(chain.forward(service.state['metadata']['joint_positions']))
        after_point=after[:3,3]+after[:3,:3]@local
        result['measured_forward_hand_travel_m']=float((after_point-initial_point)@direction)
        stages[-1]['requested_contact_speed_m_s']=args.speed
        stages[-1]['planning_timing_seconds']=planning
        retreat,retreat_timing=robot_only_path(service.ik,service.state['metadata']['joint_positions'],-.015*direction)
        issue(schedule_contact(retreat,chain,local,contact_speed_m_s=.03,endpoint_hold_steps=0),'withdrawal')
        stages[-1]['planning_timing_seconds']=retreat_timing
        hold=np.repeat(np.asarray(service.state['metadata']['joint_positions'])[None],20,axis=0)
        issue(hold,'coasting-one-second')
        actions=read_json(service.episode/'initial/actions.json')
        contacts=read_json(service.episode/'initial/evaluation_contact_diagnostics.json')['steps']
        first=next((r['step'] for r in actions if r['step']>523 and r['evaluation_task_success']),None)
        force_rows=[]
        for row in contacts:
            if row['step']<=523:continue
            for item in row['robot_involved_contacts']:
                if sum(item['robot_flags'])!=1:continue
                force_rows.append({'control_step':row['step'],'geometry_names':item['geom_names'],
                    'signed_distance_m':item['distance_m'],'normal_force_n':item.get('normal_force_n',0.)})
        record.update(native_success_seen=first is not None,first_success_control_step=first,
            final_task_success=bool(actions[-1]['evaluation_task_success']),control_steps=service.state['control_steps'],
            settling_control_steps=20,retained_success_verified=bool(actions[-1]['evaluation_task_success']),
            offline_contacts=force_rows,peak_robot_environment_normal_force_n=max((r['normal_force_n'] for r in force_rows),default=0.))
        result.update(task_success=record['native_success_seen'],status='diagnostic_executed',stop_reason='stroke_withdrawal_and_coasting_complete')
    except Exception as error:
        record.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result['timing_seconds']={'total':time.perf_counter()-started}
        write_json(out/'replay.json',record)
        if service.sim:service.finish_attempt('user_authorized_collision_experiment_complete')
    print(json.dumps({k:v for k,v in record.items() if k not in {'skill_result','offline_contacts'}}),flush=True)


if __name__=='__main__':main()
