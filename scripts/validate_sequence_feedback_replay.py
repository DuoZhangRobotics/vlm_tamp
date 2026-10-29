"""Replay a recorded approach and test sequence feedback with public geometry.

The physical prefix and chosen approach are fixed. Native joints and contacts
are read only after execution for grading, never to construct a robot command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from serve_subscription_robot import RobotService
from replay_subscription_episode import source_commands,source_reference_mode,source_case
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.profiled_sim_client import execute_batch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--ground-event',type=int,default=22);parser.add_argument('--plan-event',type=int,default=25)
    parser.add_argument('--execution-event',type=int,default=33);parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--control-steps',type=int,default=1200)
    parser.add_argument('--calibration',choices=['enabled','disabled'],default='enabled')
    parser.add_argument('--feedback-degrees',type=float)
    args=parser.parse_args();source=Path(args.source).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    work=out/'workspace';work.mkdir();case=source_case(source,args.attempt)
    events={e['sequence']:e for e in map(json.loads,(source/'tool-events.jsonl').read_text().splitlines())}
    grounding=events[args.ground_event];planned=events[args.plan_event];execution=events[args.execution_event]
    if not all(e['response'].get('ok') for e in [grounding,planned,execution]):raise ValueError('Select completed source operations')
    checkpoint=grounding['response']['result']['scene_reference_control_step']
    source_plan_id=planned['response']['result']['plan_id'];commands=source_commands(source,args.attempt)
    selection=source/f'attempt{args.attempt:02d}/{args.plan_event:03d}-grasp/selection'
    selected=read_json(selection/'selected.json');path=np.load(selection/'approach.npy')
    old_actions=read_json(source/f'attempt{args.attempt:02d}/initial/actions.json')
    service=RobotService(out/'service',work,case['state_index'],1,task=case['task'],
        surface_contact_tools=True,part_motion_tools=True,landmark_evidence=True,stop_on_success=True,
        maximum_control_steps=args.control_steps,reference_mode=source_reference_mode(source,args.attempt))
    report={'source':str(source),'source_events':[args.ground_event,args.plan_event,args.execution_event],
        'recorded_public_selections_and_approach':True,'fresh_luna_evaluation':False,
        'intermediate_arc_plans_regenerated_not_replayed':True,
        'native_state_used_for_planning':False,'physics_state_assignment_used':False,
        'checkpoint':checkpoint,'tested_controller':service.reference_mode,'started_at':time.time()}
    report['automatic_calibration']=args.calibration=='enabled'
    report['feedback_degrees_override']=args.feedback_degrees
    sources=[*ROOT.glob('src/vlm_tamp/**/*.py'),*ROOT.glob('scripts/*.py'),ROOT/'configs/articulated_libero.json']
    write_json(out/'source-hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    try:
        service.dispatch('reset',{})
        for command in commands:
            if service.state['control_steps']==checkpoint:break
            if service.state['control_steps']+len(command['references'])>checkpoint:raise ValueError('Grounding must follow a complete recorded stage')
            service.state,_=execute_batch(service.sim,service.state,command['references'],gripper=command['gripper'])
            if command['gripper']!=0:service.last_gripper_command=command['gripper']
        error=float(np.max(abs(np.asarray(service.state['metadata']['joint_positions'])-old_actions[checkpoint-1]['joint_positions'])))
        report['prefix_joint_error_rad']=error
        if service.state['control_steps']!=checkpoint or error>1e-6:raise RuntimeError('Recorded physical prefix does not match')
        service.calls+=1;service.capture()
        service.dispatch('ground',grounding['request']['arguments'])
        request=planned['request']['arguments'];options=selected['budget_timing_assumptions']
        service.calls+=1
        plan=service.keep_plan(path,{'status':'recorded_observed_approach'},'sequence',{
            'goal_degrees':float(request['degrees']),'grasp_constraint':request.get('grasp_constraint','jaw_axis'),
            'grasp_joints':selected['joints'],'approach_speed':options['speed'],
            'contact_depth_offset_m':selected.get('contact_depth_offset_m',0.),
            'opening_required':planned['response']['result'].get('opening_required',False),
            'selection_options':{'max_candidates':request.get('max_candidates',24),'seed':request.get('seed',0),'budget_seconds':request.get('budget_seconds',30)},
            'collision_policy':request.get('collision_policy','advisory')})
        args_execute={**execution['request']['arguments'],'plan_id':plan['plan_id'],'calibrate_joint':args.calibration=='enabled'}
        if args.feedback_degrees is not None:args_execute['feedback_degrees']=args.feedback_degrees
        reply=service.dispatch('execute_sequence',args_execute)
        report['execution_response']=reply
        report['execution']=reply;report['control_steps']=service.state['control_steps']
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        if service.sim:service.finish_attempt('recorded_sequence_feedback_diagnostic')
        initial=out/'service/attempt01/initial'
        actions=read_json(initial/'actions.json') if (initial/'actions.json').exists() or (initial/'actions.json.gz').exists() else []
        successful=[a['step'] for a in actions if a['evaluation_task_success']]
        step=checkpoint;manipulations=0
        for command in commands:
            if f'-issued-{source_plan_id}-' not in Path(command['path']).name:continue
            step+=len(command['references'])
            if Path(command['path']).stem.endswith('-closure') and len(actions)>=step:
                report['matched_approach_and_closure_control_step']=step
                report['approach_and_closure_joint_error_rad']=float(np.max(abs(np.asarray(actions[step-1]['joint_positions'])-old_actions[step-1]['joint_positions'])))
            if Path(command['path']).stem.endswith('-manipulation'):manipulations+=1
            if manipulations==2:break
        if args.calibration=='enabled' and manipulations==2 and len(actions)>=step:
            report['matched_through_second_probe_control_step']=step
            report['second_probe_joint_error_rad']=float(np.max(abs(np.asarray(actions[step-1]['joint_positions'])-old_actions[step-1]['joint_positions'])))
            report['intermediate_probes_match_source_at_sampled_step']=report['second_probe_joint_error_rad']<=1e-6
        report.update(native_success_seen=bool(successful),first_native_success_step=min(successful) if successful else None,
            final_native_success=bool(actions and actions[-1]['evaluation_task_success']),elapsed_seconds=time.time()-report['started_at'])
        diag=read_json(initial/'evaluation_contact_diagnostics.json') if actions else {}
        if diag.get('steps'):
            native=[j for j in diag['steps'][-1]['evaluation_only_articulation_state'] if j['type']=='hinge' and not j['joint_name'].startswith('gripper')]
            report['offline_only_final_articulation']=native
            if len(native)==1:
                origin=np.array(native[0]['anchor_world_m']);axis=np.array(native[0]['axis_world'])
                report['offline_only_joint_revision_axis_errors_m']=[]
                for revision in report.get('execution',{}).get('joint_feedback',[]):
                    if not revision.get('applied'):continue
                    delta=np.asarray(revision['updated_origin_world_m'])-origin
                    report['offline_only_joint_revision_axis_errors_m'].append(float(np.linalg.norm(delta-axis*(delta@axis))))
        write_json(out/'replay.json',report)
    print(json.dumps({k:report.get(k) for k in ['native_success_seen','control_steps','prefix_joint_error_rad','second_probe_joint_error_rad','offline_only_joint_revision_axis_errors_m']}))


if __name__=='__main__':main()
