"""Frozen evaluation of one VLM interaction interface and generic depth fitters."""
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
from vlm_tamp.coupled_storage import read_json,write_json,compact_episode
from vlm_tamp.coupled_resources import api_snapshot
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.profiled_pipeline import compress_retained_observations
from vlm_tamp.profiled_sim_client import execute_batch
from vlm_tamp.interaction_resources import InteractionResources
from vlm_tamp.interaction_execution import execution_backend,InteractionSurfaceScene
from vlm_tamp import reliable_mechanisms,reliable_bowl_fallback
import reliable_microwave_final
from run_success_pipeline import SceneBoundaryClient
from run_all_prior_tasks import TASKS as PRIOR_TASKS,configuration

TASKS=[{k:v for k,v in t.items() if k in {'label','name','suite'}} for t in PRIOR_TASKS]+[
 {'label':'open_bottom_drawer','name':'KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet','suite':'libero_90'},
 {'label':'close_bottom_drawer','name':'KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet','suite':'libero_90'}]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def config():
    value=configuration()
    value.update(version='shared-interaction-1',grounding_interface='interaction-v1',
        maximum_candidate_grasps=100,maximum_total_ik_postures=256,maximum_grasp_search_seconds=240.,
        minimum_manipulation_steps_reserved=200,microwave_approach_speed_rad_s=.30)
    # Closing allowances are supplied by fitted geometry, not task-name config.
    value.pop('closing_seating_allowance_deg',None);value.pop('closing_seating_trigger_deg',None)
    return value


def prior_case_evidence(tasks,states):
    found=[];wanted={(t['label'],i) for t in tasks for i in states}
    for base,dirs,files in os.walk(ROOT/'results'):
        dirs[:]=[d for d in dirs if d not in {'source-snapshot','source-before','frames','temporary-proposal-cache','.git'}]
        if 'summary.json' not in files:continue
        p=Path(base)/'summary.json'
        try:v=read_json(p)
        except (ValueError,OSError):continue
        if not isinstance(v,dict) or not isinstance(v.get('task'),str) or not isinstance(v.get('initial_state'),int):continue
        if (v.get('task'),v.get('initial_state')) in wanted:
            found.append({'task':v['task'],'state':v['initial_state'],'source':str(p.relative_to(ROOT))})
    return found


def episode(task,index,folder,resources,timeline,*,perception_only=False):
    folder.mkdir(parents=True,exist_ok=False);start=time.perf_counter();before=resources.new_api_calls
    sim=SceneBoundaryClient(folder/'simulator.log',fast=True,timeline=timeline);stage='initial_observation';engine_owns=False
    try:
        state=sim.reset(task['name'],index,folder/'initial',suite_name=task['suite'],seed=0)
        state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,output_dir=folder/'opened')
        stage='vlm_selection_and_depth_fitting'
        grounded=resources.ground_interaction(folder/'opened',state['metadata'],folder/'grounding')
        backend=execution_backend(grounded)
        write_json(folder/'routing.json',{'backend':backend,'source':'predicted interaction and measured geometry',
            'task_name_used_for_dispatch':False,'geometry_backend':grounded.evidence['backend'],
            'full_pose_fallback_for_non_upright_hinges':True,
            'partial_pose_tracker_capability':'upright bar and hinge, within 3 degrees'})
        if perception_only:
            result={'status':'perception_only','official_task_success':False,'final_task_success':False,
                'physical_execution_attempted':False,'fitted_geometry':True}
        else:
            stage='physical_execution';prepared={'sim':sim,'state':state,'grounded':grounded}
            if backend=='placement':engine=reliable_bowl_fallback;fn=engine.run_episode
            elif backend=='upright_bar_hinge':engine=reliable_microwave_final;fn=engine.episode
            else:
                engine=reliable_mechanisms;fn=engine.run_episode
                if grounded.spec['joint_family']=='revolute':
                    grounded.spec['target_progress']+=grounded.spec['completion_allowance']
            engine.execute=execute_batch
            if engine is reliable_mechanisms:engine.SurfaceScene=InteractionSurfaceScene
            engine_owns=True
            result=fn(task,index,folder,config(),resources,prepared=prepared)
            result['physical_execution_attempted']=True
            result['selected_backend']=backend
    except Exception as error:
        (folder/'interface-error.txt').write_text(traceback.format_exc())
        result={'status':'failed','official_task_success':False,'final_task_success':False,
                'failure_stage':stage,'failure':str(error),'physical_execution_attempted':False}
    finally:
        if not engine_owns:sim.close()
    actions=read_json(folder/'initial/actions.json') if (folder/'initial/actions.json').exists() else []
    result.update(task=task['label'],official_task_name=task['name'],suite=task['suite'],initial_state=index,seed=0,
        new_api_calls=resources.new_api_calls-before,total_wall_seconds=time.perf_counter()-start,
        control_steps=len(actions),official_task_success=any(a['evaluation_task_success'] for a in actions),
        final_task_success=bool(actions and actions[-1]['evaluation_task_success']),
        grounding_interface='interaction-v1',execution_veto_checks=result.get('execution_veto_checks',0),
        simulator_scene_geometry_used_for_policy=False,all_failures_in_denominator=True)
    write_json(folder/'summary.json',result)
    if not (folder/'tracking.json').exists():write_json(folder/'tracking.json',[])
    resources.cleanup_proposals()
    prior_cleanup=result.get('storage',{})
    result['storage']=compact_episode(folder)
    result['storage']['bytes_reclaimed_this_pass']+=prior_cleanup.get('bytes_reclaimed_this_pass',0)
    write_json(folder/'summary.json',result)
    compress_retained_observations(folder)
    return read_json(folder/'summary.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--tasks',nargs='+',choices=[t['label'] for t in TASKS])
    parser.add_argument('--states',nargs='+',type=int,default=[2]);parser.add_argument('--development',action='store_true')
    parser.add_argument('--perception-only',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--input-observation',help='Development only: fit a retained observation without simulation')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Output must be inside project results')
    tasks=[t for t in TASKS if not args.tasks or t['label'] in args.tasks]
    if len(set(args.states))!=len(args.states) or any(not 0<=i<50 for i in args.states):raise ValueError('Unique official indices 0..49 required')
    if not args.development and any(i in {0,1,11,21,31,41} for i in args.states):raise ValueError('Previously inspected regression states require --development')
    if args.input_observation and not args.development:raise ValueError('Retained observation fitting is development only')
    planned=[{'task':t['label'],'state':i} for i in args.states for t in tasks]
    if args.resume:
        protocol=read_json(out/'protocol.json');sources=protocol['sources_sha256']
        if read_json(out/'manifest.json')['episodes']!=planned:raise ValueError('Resume manifest differs')
    else:
        prior=prior_case_evidence(tasks,args.states) if not args.development else []
        if prior:raise ValueError('These task/state pairs have prior results; choose unused indices or development mode: '+str(prior))
        out.mkdir(parents=True,exist_ok=False)
        files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[Path(__file__),ROOT/'scripts/reliable_microwave_final.py',ROOT/'scripts/run_success_pipeline.py',ROOT/'scripts/run_all_prior_tasks.py',ROOT/'configs/prompts/interaction_choices_v1.txt']
        sources={str(p.relative_to(ROOT)):sha(p) for p in files}
        for name in sources:
            target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        write_json(out/'manifest.json',{'episodes':planned,'every_planned_case_in_denominator':True,
            'seed':0,'settling_actions':5,'maximum_control_steps':600})
        write_json(out/'protocol.json',{'sources_sha256':sources,'configuration':config(),
            'development':args.development,'perception_only':args.perception_only,
            'prior_case_search':prior,'held_out_from_success_first_tuning':not args.development,
            'fresh_schema_for_all_cases':True,'single_prompt_for_all_tasks':True,'task_name_dispatch':False,
            'fixed_before_first_trial':True,'execution_time_vetoes':False,'simulator_object_truth_for_policy':False,
            'model':'gpt-6-luna','reasoning_effort':'high','output_tokens':12000,'automatic_retries':0,
            'api_before':api_snapshot(),'remaining_assumptions':['selected primitive library','upright partial-pose executor; general full-pose fallback',
                'quarter-turn opening goal','robot-hand-sized sliding opening','point-cloud scene coverage and existing contact models']})
    if not all(sha(ROOT/name)==digest for name,digest in sources.items()):raise ValueError('Frozen sources changed')
    timeline=Timeline();resources=InteractionResources(out,timeline,fresh=True)
    if args.input_observation:
        source=Path(args.input_observation);metadata=read_json(source/'metadata.json')
        result=resources.ground_interaction(source,metadata,out/'grounding')
        write_json(out/'development-result.json',{'source':str(source),'backend':execution_backend(result),
            'instruction':metadata['instruction'],'spec':result.spec,'api':api_snapshot()})
        timeline.save(out/'timing.json');return
    resources.warm_models();rows=[]
    for entry in planned:
        if not all(sha(ROOT/name)==digest for name,digest in sources.items()):raise ValueError('Frozen sources changed during cohort')
        folder=out/f'{entry["task"]}-state{entry["state"]:03d}';timeline.context=entry
        if (folder/'summary.json').exists():result=read_json(folder/'summary.json')
        else:
            if folder.exists():raise ValueError('Interrupted case must be diagnosed, not silently rerun')
            print(json.dumps({'event':'started',**entry}),flush=True)
            task=next(t for t in tasks if t['label']==entry['task'])
            result=episode(task,entry['state'],folder,resources,timeline,perception_only=args.perception_only)
        rows.append(result);timeline.save(out/'timing.json')
        write_json(out/'summary.json',{'planned':len(planned),'completed':len(rows),
            'native_successes':sum(r['official_task_success'] for r in rows),'episodes':rows,'api':api_snapshot()})
        print(json.dumps({'event':'completed',**entry,'success':result['official_task_success'],
            'failure':result.get('failure'),'steps':result['control_steps']}),flush=True)
    resources.cleanup_proposals()


if __name__=='__main__':main()
