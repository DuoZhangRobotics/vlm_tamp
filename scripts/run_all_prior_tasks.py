"""Frozen end-to-end regression of every previously physically tested task."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from vlm_tamp.coupled_storage import read_json,write_json,POLICY
from vlm_tamp.coupled_resources import api_snapshot
from vlm_tamp.cohort_resources import CohortResources
from vlm_tamp.cohort_execution import run_episode
from run_mcvamp_microwave import episode as microwave_episode

TASKS=[
    {'label':'bowl_on_plate','name':'put_the_bowl_on_the_plate','suite':'libero_goal','legacy_index':8,'adapter':'point_goal'},
    {'label':'open_middle_drawer','name':'open_the_middle_drawer_of_the_cabinet','suite':'libero_goal','legacy_index':0,'adapter':'line'},
    {'label':'open_top_drawer','name':'KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet','suite':'libero_90','adapter':'line'},
    {'label':'close_top_drawer','name':'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet','suite':'libero_90','adapter':'line'},
    {'label':'open_microwave','name':'KITCHEN_SCENE7_open_the_microwave','suite':'libero_90','adapter':'MC-VAMP hinge'},
    {'label':'close_microwave','name':'KITCHEN_SCENE6_close_the_microwave','suite':'libero_90','adapter':'MC-VAMP hinge'},
    {'label':'turn_on_stove','name':'KITCHEN_SCENE3_turn_on_the_stove','suite':'libero_90','adapter':'control rotation'}]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(tasks,states):
    if len(set(states))!=len(states) or any(i not in {1,11,21,31,41} for i in states):
        raise ValueError('Choose unique states from the frozen five-state set')
    return [{'task':t['label'],'state':i,'suite':t['suite'],'name':t['name'],'adapter':t['adapter']}
        for i in states for t in tasks]


def configuration():
    config=read_json(ROOT/'configs/articulated_libero.json')
    config.update(version='all-prior-tasks-1.2',maximum_candidate_grasps=24,maximum_total_ik_postures=96,
        maximum_grasp_search_seconds=120.,maximum_grasp_poses=24,maximum_contact_plan_seconds=8.,
        maximum_ik_postures_per_grasp=8,complete_planar_panel=False,scene_voxel_spacing_m=.003,
        full_task_only=False,planning_backend='partial_pose_arc',motion_policy='short',hinge_estimator='hybrid',
        reconcile_handle_roles=True,handle_association_m=.008,replay_grasp_source=None,
        shared_cohort_grounding=True,arc_search_seconds=20.,arc_goal_proposals=32,scene_mode='strict',
        max_output_tokens=12000,execution_vetoes=False)
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--tasks',nargs='+',choices=[t['label'] for t in TASKS])
    parser.add_argument('--states',nargs='+',type=int,default=[1,11,21,31,41])
    parser.add_argument('--resume',action='store_true',help='Continue an interrupted frozen cohort; never rerun a completed case')
    args=parser.parse_args();tasks=[t for t in TASKS if not args.tasks or t['label'] in args.tasks]
    planned=manifest(tasks,args.states);output=(ROOT/args.output).resolve();config=configuration()
    protected=read_json(ROOT/'results/coupled-planning-resumed-20261015/frozen-evaluation/protocol.json')['sources_sha256']
    if not all(sha(ROOT/n)==h for n,h in protected.items()):raise ValueError('Protected historical baseline source changed')
    if args.resume:
        protocol=read_json(output/'protocol.json')
        if read_json(output/'manifest.json')['planned_episodes']!=planned:raise ValueError('Resume manifest differs')
        sources=protocol['sources_sha256']
        if any(sha(ROOT/n)!=h for n,h in sources.items()):raise ValueError('Resume requires the exact frozen implementation')
    else:
        output.mkdir(parents=True,exist_ok=False)
        files=list((ROOT/'src/vlm_tamp').rglob('*.py'))
        files += [ROOT/n for n in ['scripts/run_mcvamp_microwave.py','scripts/run_all_prior_tasks.py',
            'scripts/try_rekep_bowl_v3.py','scripts/run_coupled_door.py','src/vlm_tamp/native/mcvamp_microwave.cc',
            'configs/articulated_libero.json','configs/physical_tasks_curobo.json']]
        sources={str(p.relative_to(ROOT)):sha(p) for p in files}
        for n in sources:
            target=output/'source-snapshot'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target)
        write_json(output/'manifest.json',{'planned_episodes':planned,'all_failures_in_denominator':True,
            'seed':0,'settling_actions':5,'maximum_control_steps':600})
        ledger=api_snapshot();write_json(output/'api-ledger-before.json',ledger)
        write_json(output/'protocol.json',{'sources_sha256':sources,'configuration':config,
            'protected_baseline_sources_sha256':protected,'frozen_before_first_episode':True,
            'all_states_previously_inspected':True,'held_out_generalization_claim':False,
            'fixed_grasp_replay':False,'candidate_cache_allowed_only_after_current_geometry_agreement':True,
            'fresh_grasp_selection_per_episode':True,'same_native_success_predicate':True,
            'execution':'full timed joint references; no runtime collision, residual, retention or progress vetoes',
            'sampling_motion_planner_tasks':['open_microwave','close_microwave'],
            'other_task_planner':'cuRobo multi-IK continuation and sampled observed robot/payload collision admission',
            'initial_state_indices':args.states,'native_build':read_json(ROOT/'.runtime/mcvamp-microwave/build.json'),
            'simulator_object_truth_policy_input':False,'physical_grasp_only':True,
            'task_success':'native LIBERO predicate only; every manifest case counts',
            'model':'gpt-6-luna','reasoning_effort':'high','max_output_tokens':12000,
            'api_automatic_retries':0,'api_budget_usd':2.,'api_spend_before_usd':ledger['cumulative_conservative_spend_usd'],
            'maximum_api_scene_records':65,'scene_count_limit_extension_only_for_requested_cohort':True,
            'storage_policy':POLICY,'thread_environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}})
        write_json(output/'integration-repair-provenance.json',{
            'superseded_integration_cohorts':['results/all-prior-tasks-frozen-20261015','results/all-prior-tasks-frozen-v2-20261015'],
            'repairs':['Preserve synchronized robot-only depth arrays in scene/tracking observations',
                'Keep physical grasp retention separate from storage retention.json'],
            'offline_regression':'results/all-prior-tasks-integration-20261015/depth-loader-repair',
            'grasp_search_or_task_parameters_tuned_after_integration':False,
            'all_completed_integration_attempts_retained':True})
    resources=CohortResources(output);summaries=[];began=time.perf_counter()
    for entry in planned:
        folder=output/f'{entry["task"]}-state{entry["state"]:03d}'
        if any(sha(ROOT/n)!=h for n,h in sources.items()):raise ValueError('Frozen cohort source changed')
        if (folder/'summary.json').exists():
            if not args.resume:raise ValueError('Refusing to overwrite existing episode')
            result=read_json(folder/'summary.json')
        else:
            if folder.exists():raise ValueError('An interrupted episode requires explicit diagnosis before continuation')
            task=next(t for t in tasks if t['label']==entry['task']);index=entry['state'];calls=resources.new_api_calls
            print(json.dumps({'event':'episode_started','task':entry['task'],'state':index,'completed':len(summaries),'total':len(planned)}),flush=True)
            if task['adapter']=='MC-VAMP hinge':
                result=microwave_episode(task,index,folder,config,resources)
                result.update(new_api_calls=resources.new_api_calls-calls,fixed_grasp_replay=False)
                write_json(folder/'summary.json',result)
            else:result=run_episode(task,index,folder,config,resources)
        summaries.append(result);ledger=api_snapshot()
        write_json(output/'summary.json',{'planned':len(planned),'completed':len(summaries),'episodes':summaries,
            'native_successes':sum(r['official_task_success'] for r in summaries),
            'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd'],'remaining_api_budget_usd':ledger['remaining_usd'],
            'baseline_sources_unchanged':all(sha(ROOT/n)==h for n,h in protected.items()),
            'runner_elapsed_seconds':time.perf_counter()-began})
        print(json.dumps({'event':'episode_completed','task':entry['task'],'state':entry['state'],
            'success':result['official_task_success'],'failure':result.get('failure'),
            'control_steps':result.get('control_steps'),'reclaimed_bytes':result.get('storage',{}).get('bytes_reclaimed_this_pass')}),flush=True)


if __name__=='__main__':main()
