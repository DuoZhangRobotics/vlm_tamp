"""Execute an independent subset of an existing frozen 35-case manifest.

This is a simulation worker, not another agent. It imports the exact pinned
policy functions and never modifies the parent manifest, configuration or
aggregate summary. Private proposal caches avoid concurrent cache writers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from run_all_prior_tasks import TASKS,sha
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.cohort_resources import CohortResources
from vlm_tamp.cohort_execution import run_episode
from run_mcvamp_microwave import episode as microwave_episode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--states',nargs='+',type=int,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();output=(ROOT/args.root).resolve()
    protocol=read_json(output/'protocol.json');config=protocol['configuration'];sources=protocol['sources_sha256']
    assert all(sha(ROOT/n)==h for n,h in sources.items())
    planned=[e for e in read_json(output/'manifest.json')['planned_episodes'] if e['state'] in args.states]
    if not planned or len(set(args.states))!=len(args.states):raise ValueError('Choose a nonempty independent state partition')
    name='states-'+'-'.join(map(str,args.states));cache=output/('proposal-cache-'+name)
    cache.mkdir(exist_ok=True);resources=CohortResources(output);resources.cache=cache
    write_json(output/('partition-'+name+'-protocol.json'),{'states':args.states,'planned_episodes':planned,
        'parent_configuration_unmodified':True,'policy_sources_verified_against_parent':True,
        'worker_source_sha256':sha(Path(__file__)),'candidate_cache_private_to_avoid_write_races':str(cache),
        'per_generation_seed':0,'api_ledger_shared_and_locked':True,'cumulative_dollar_cap':2.,
        'latency_measurements_taken_under_concurrent_load':True})
    summaries=[]
    for entry in planned:
        assert all(sha(ROOT/n)==h for n,h in sources.items())
        folder=output/f'{entry["task"]}-state{entry["state"]:03d}'
        if (folder/'summary.json').exists() and args.resume:result=read_json(folder/'summary.json')
        else:
            if folder.exists():raise ValueError('Partition overlaps an existing or active episode')
            print(json.dumps({'event':'partition_episode_started','task':entry['task'],'state':entry['state']}),flush=True)
            task=next(t for t in TASKS if t['label']==entry['task']);before=resources.new_api_calls
            if task['adapter']=='MC-VAMP hinge':
                result=microwave_episode(task,entry['state'],folder,config,resources)
                result.update(new_api_calls=resources.new_api_calls-before,fixed_grasp_replay=False)
                write_json(folder/'summary.json',result)
            else:result=run_episode(task,entry['state'],folder,config,resources)
        summaries.append(result)
        write_json(output/('partition-'+name+'-summary.json'),{'planned':len(planned),'completed':len(summaries),
            'native_successes':sum(r['official_task_success'] for r in summaries),'episodes':summaries})
        print(json.dumps({'event':'partition_episode_completed','task':entry['task'],'state':entry['state'],
            'success':result['official_task_success'],'failure':result.get('failure')}),flush=True)
    # Every episode holds its own exact inputs, candidates and provenance.
    # The extra runtime cache is redundant once this worker has finished.
    if cache.is_relative_to(output) and cache.name.startswith('proposal-cache-'):shutil.rmtree(cache)
    write_json(output/('partition-'+name+'-complete.json'),{'completed':len(summaries),
        'private_runtime_cache_removed':True,'all_episode_inputs_preserved':True})


if __name__=='__main__':main()
