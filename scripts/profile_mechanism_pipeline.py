"""Full fresh-frontend timing with optional batched execution; same planner."""
import time
PROCESS_ENTRY=time.perf_counter()
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.profiled_pipeline import ProfiledResources,instrument_engine,compress_retained_observations
from vlm_tamp.coupled_resources import api_snapshot
from vlm_tamp.coupled_storage import read_json,write_json
from run_all_prior_tasks import TASKS,configuration


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--backend',choices=['baseline','fast'],default='fast')
    parser.add_argument('--grounding',choices=['cached','fresh'],default='fresh')
    parser.add_argument('--tasks',nargs='+',default=['open_top_drawer','turn_on_stove'])
    parser.add_argument('--states',nargs='+',type=int,default=[1]);args=parser.parse_args()
    out=(ROOT/args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    tasks=[t for t in TASKS if t['label'] in args.tasks]
    if not tasks or any(t['label'] not in {'open_middle_drawer','open_top_drawer','close_top_drawer','turn_on_stove'} for t in tasks):raise ValueError('Unsupported mechanism task')
    if args.grounding=='fresh' and any('legacy_index' in t for t in tasks):raise ValueError('Fresh API profiling uses the existing articulated prompt, not the legacy task program')
    prior=read_json(ROOT/'results/mcvamp-mechanisms-camera-retry-20261016/protocol.json')['sources_sha256']
    assert all(sha(ROOT/n)==h for n,h in prior.items()),'Baseline implementation changed'
    config=configuration();config.update(version='mcvamp-mechanisms-1.0')
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/vlm_tamp').rglob('*.py')}
    for p in [Path(__file__).resolve(),ROOT/'src/vlm_tamp/native/mcvamp_mechanism.cc']:sources[str(p.relative_to(ROOT))]=sha(p)
    for name in sources:
        dst=out/'source-snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
    planned=[{'task':t['label'],'state':i} for i in args.states for t in tasks]
    write_json(out/'manifest.json',{'episodes':planned,'maximum_control_steps':600,'settling_actions':5,'seed':0,'all_failures_in_denominator':True})
    write_json(out/'protocol.json',{'sources_sha256':sources,'unchanged_planning_sources_sha256':prior,'configuration':config,
        'backend':args.backend,'grounding':args.grounding,'same_planner_and_constraint_parameters':True,
        'same_trajectory_horizons':True,'asynchronous_replanning':False,'execution_vetoes':False,
        'planner_camera_depth_views_unchanged':True,'fixed_reference_replay':False,
        'api_before':api_snapshot(),'model':'gpt-6-luna','reasoning':'high','max_output_tokens':12000,
        'api_automatic_retries':0,'api_metadata_documentation':'https://developers.openai.com/api/reference/python/resources/responses/methods/create',
        'fresh_response_cache_bypass_changes_metadata_only':True,'server_prompt_caching_may_still_apply':True,
        'cold_model_initialization_recorded_separately':True,'single_simulation_worker':True})
    trace=Timeline();trace.origin=int(PROCESS_ENTRY*1e9);trace.mark('research_process_imports_completed')
    resources=ProfiledResources(out,trace,fresh=args.grounding=='fresh')
    with trace.span('initialization.models_total'):resources.warm_models()
    trace.save(out/'timing.json');results=[]
    for entry in planned:
        assert all(sha(ROOT/n)==h for n,h in sources.items()),'Frozen profiling source changed'
        trace.context={**entry,'backend':args.backend,'grounding_mode':args.grounding}
        folder=out/f'{entry["task"]}-state{entry["state"]:03d}';task=next(t for t in tasks if t['label']==entry['task'])
        print({'event':'started',**entry,'backend':args.backend,'grounding':args.grounding},flush=True)
        with trace.span('episode.total_including_finalization_and_cleanup') as episode:
            with instrument_engine(trace,fast=args.backend=='fast') as (engine,clients):
                result=engine.run_episode(task,entry['state'],folder,config,resources)
            with trace.span('storage.compress_retained_observations'):compress_retained_observations(folder)
        result=read_json(folder/'summary.json');result['profiling_backend']=args.backend
        result['grounding_cache_mode']=args.grounding;result['timing_start_s']=episode['start_s'];result['timing_end_s']=episode['end_s']
        case_events=[e for e in trace.events if e.get('task')==entry['task'] and e.get('state')==entry['state']]
        case_marks=[m for m in trace.marks if m.get('task')==entry['task'] and m.get('state')==entry['state']]
        start=next((e['start_s'] for e in case_events if e['name'] in {'grounding.total','grounding.cached_legacy'}),None)
        first=next((m['at_s'] for m in case_marks if m['name']=='first_planned_task_motion_requested'),None)
        result['fresh_observation_to_first_task_motion_seconds']=first-start if start is not None and first is not None else None
        result['episode_to_first_task_motion_seconds']=first-episode['start_s'] if first is not None else None
        result['full_episode_including_cleanup_seconds']=episode['seconds']
        write_json(folder/'summary.json',result)
        write_json(folder/'rpc-profile.json',[r for c in clients for r in c.rpc_events])
        write_json(folder/'timing.json',{'events':case_events,'marks':case_marks,'durations_inclusive':True})
        results.append(result);trace.save(out/'timing.json')
        write_json(out/'summary.json',{'planned':len(planned),'completed':len(results),'episodes':results,
            'native_successes':sum(r['official_task_success'] for r in results),'api_after':api_snapshot()})
        print({'event':'completed',**entry,'success':result['official_task_success'],'failure':result.get('failure'),
               'first_motion_s':result['fresh_observation_to_first_task_motion_seconds'],'episode_s':result['episode_wall_seconds']},flush=True)
    trace.context={};trace.mark('all_episodes_complete');trace.save(out/'timing.json')

if __name__=='__main__':main()
