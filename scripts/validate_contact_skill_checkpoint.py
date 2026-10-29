"""Use retained Luna annotations to preview/replay contact skills at a checkpoint.

This numerical diagnostic never turns its historical selections into the fresh
actor's instructions. Replay verifies the entire original command prefix.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.contact_skills import build_contact_scene,plan_engagement,Budget,SkillStop
from vlm_tamp.hinged_contact_tools import tool_configuration
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK

SOURCE=ROOT/'results/perception-contract-repair-20261028/trials/close_microwave-state017-perception-repaired'


def saved_observation(source,observation=27):
    folder=source/f'agent-workspace/robot_data/attempt01/obs{observation:03d}'
    metadata=read_json(source/f'attempt01/{observation:03d}-observation/metadata.json')
    calibration=read_json(folder/'calibration.json');cameras={}
    with np.load(folder/'sensors.npz') as arrays:
        for name,cal in calibration.items():
            view={**metadata['cameras'][name],**cal}
            for key in ['rgb','depth','world_points','known_robot_depth_prior']:
                if name+'_'+key in arrays:view[key]=arrays[name+'_'+key].copy()
            cameras[name]=view
    return cameras,metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',default=str(SOURCE))
    p.add_argument('--events',nargs='+',type=int,default=[33,36]);p.add_argument('--surface',choices=['fingertips','palm'],default='fingertips')
    p.add_argument('--kind',choices=['push','hook_pull'],default='push');p.add_argument('--replay',action='store_true')
    p.add_argument('--mode',choices=['controlled','push_and_release'],default='controlled')
    p.add_argument('--speed',type=float,default=.03);p.add_argument('--request-file')
    p.add_argument('--checkpoint',type=int,choices=[60,459],default=459)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    from vlm_tamp.contact_skill_service import qualification_sources
    write_json(out/'source-hashes.json',qualification_sources())
    for relative in [*qualification_sources(),'scripts/validate_contact_skill_checkpoint.py','configs/articulated_libero.json']:
        dest=out/'source-snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,dest)
    source=Path(a.source).resolve();events=[json.loads(s) for s in (source/'tool-events.jsonl').read_text().splitlines()]
    if a.replay:
        return physical_replay(a,out,source,events)
    cameras,metadata=saved_observation(source,27 if a.checkpoint==459 else 1);record=read_json(source/'attempt01/robot-chain.json')
    import vamp
    empty=vamp.Environment();config=tool_configuration(read_json(ROOT/'configs/articulated_libero.json'))
    ik=TaskIK(record,config,lambda q:vamp.panda.validate(np.asarray(q).tolist(),empty,True),output=out)
    rows=[]
    try:
        for event_id in a.events:
            event=next(e for e in events if e['sequence']==event_id);args=event['request']['arguments']
            if event['request']['operation']=='ground':
                regions=args['regions'];args={'camera':regions['camera'],'region_xyxy':regions['contact_box_xyxy']}
            began=time.perf_counter();row={'source_luna_event':event_id,'contact_surface':a.surface,'kind':a.kind,
                'historical_annotation_only':True,'checkpoint':a.checkpoint,'physical_execution':False}
            try:
                scene,evidence,selected=build_contact_scene(cameras,metadata,ik.chain,args['camera'],args['region_xyxy'],None,a.surface,skill_kind=a.kind)
                row['contact']=evidence
                if a.kind=='hook_pull':evidence['direction_world']=(-np.asarray(evidence['direction_world'])).tolist()
                engagement=plan_engagement(ik,scene,record,metadata,evidence,selected,a.kind,Budget(30.))
                row.update(solved=True,contact=evidence,
                    plan={k:v for k,v in engagement.items() if k not in {'path','references'}})
                np.save(out/f'event{event_id}-approach.npy',engagement['references'])
            except SkillStop as error:row.update(solved=False,stop_reason=error.reason,diagnostic=error.evidence)
            row['seconds']=time.perf_counter()-began;rows.append(row);write_json(out/'previews.json',rows)
            print(json.dumps({k:v for k,v in row.items() if k not in {'plan','contact','diagnostic'}}),flush=True)
    finally:ik.close()


def physical_replay(args,out,source,events):
    from serve_subscription_robot import RobotService
    from replay_subscription_episode import source_commands
    from vlm_tamp.profiled_sim_client import execute_batch
    work=out/'workspace';work.mkdir()
    service=RobotService(out/'service',work,17,1,task='close_microwave',surface_contact_tools=True,
        stop_on_success=False,numerical_backend='curobo_v2',reference_mode='measured_delta')
    service.contact_skill_diagnostic=True
    record={'checkpoint':args.checkpoint,'historical_annotation_only':True,'fresh_luna_evaluation':False}
    try:
        service.dispatch('reset',{})
        for command in source_commands(source,1):
            if service.state['control_steps']>=args.checkpoint:break
            count=args.checkpoint-service.state['control_steps']
            service.state,_=execute_batch(service.sim,service.state,command['references'][:count],gripper=command['gripper'])
        expected=read_json(source/f'attempt01/{27 if args.checkpoint==459 else 1:03d}-observation/metadata.json')['joint_positions']
        error=float(np.max(np.abs(np.asarray(expected)-service.state['metadata']['joint_positions'])))
        record['checkpoint_joint_error_rad']=error
        if error>1e-6:raise ValueError('Physical replay failed the checkpoint agreement test')
        service.state=service.sim.set_joint_reference_mode('slew_limited')
        service.calls+=1;obs=service.capture()
        if args.request_file:request=read_json(args.request_file)
        else:
            event=next(e for e in events if e['sequence']==args.events[0]);old=event['request']['arguments']
            request={'camera':old['camera'],'region_xyxy':old['region_xyxy'],'distance_m':.12,
                'contact_surface':args.surface,'mode':args.mode,'contact_speed_m_s':args.speed}
        request['observation_id']=obs['observation_id']
        grounding=request.pop('_recorded_ground_regions',None)
        derive_pull=request.pop('_derive_joint_pull',False)
        if grounding:
            service.dispatch('ground',{'regions':grounding})
        if derive_pull:
            point=np.median(service.scene.contact_reference,axis=0)
            outward=np.asarray(service.scene.panel_normal)
            tangent=np.cross(service.scene.axis,point-service.scene.origin)
            request.update(joint_degrees=float(3*np.sign(outward@tangent)),joint_model_epoch=service.epoch)
        if args.kind=='push':request.update(mode=args.mode,contact_speed_m_s=args.speed)
        record['request']=request
        record['skill_result']=service.dispatch(args.kind,request)
        # Diagnostic-only settling. Normal actor native-success stopping is
        # unchanged. Finish the complete release before testing retention.
        if service.sim and 600-service.state['control_steps']>=20:
            refs=np.repeat(np.asarray(service.state['metadata']['joint_positions'])[None],20,axis=0)
            service.calls+=1;record['settling']=service.issue(refs,0,'diagnostic-settle',track_scene=False,
                camera_names=['agentview','frontview'])
            record['settling_control_steps']=20
        else:record['settling_control_steps']=0
        actions=read_json(service.episode/'initial/actions.json')
        contacts=read_json(service.episode/'initial/evaluation_contact_diagnostics.json')['steps']
        violations=[];contact_counts={}
        for step in contacts:
            if step['step']<=args.checkpoint:continue
            for contact in step['robot_involved_contacts']:
                flags=contact['robot_flags']
                if sum(flags)!=1:continue
                name=contact['geom_names'][flags.index(True)]
                contact_counts[name]=contact_counts.get(name,0)+1
                if 'finger' not in name and not(name=='gripper0_hand_collision' and request.get('contact_surface')=='palm'):
                    violations.append({'control_step':step['step'],'robot_part':name,
                        'signed_distance_m':contact['distance_m'],'penetration_m':max(0.,-contact['distance_m']),
                        'normal_force_n':contact.get('normal_force_n')})
        record['offline_contact_audit']={'evaluation_only':True,'robot_contact_counts':contact_counts,
            'non_permitted_robot_parts':violations,'not_sent_to_actor_or_planner':True,
            'note':'Finger-body distal/proximal locations require per-point geometry; names alone do not certify contact permissions.'}
        record.update(native_success_seen=any(r['evaluation_task_success'] for r in actions),
            final_task_success=bool(actions and actions[-1]['evaluation_task_success']),control_steps=service.state['control_steps'])
        record['retained_success_verified']=bool(record['settling_control_steps']==20 and record['final_task_success'])
    except Exception as error:
        record.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        write_json(out/'replay.json',record)
        if service.sim:service.finish_attempt('diagnostic_complete')
    print(json.dumps({k:v for k,v in record.items() if k not in {'skill_result','settling'}}),flush=True)


if __name__=='__main__':main()
