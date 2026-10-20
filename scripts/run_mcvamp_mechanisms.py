"""Compare observed-surface MC-VAMP drawers/knobs against the frozen cohort."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from vlm_tamp.coupled_storage import read_json,write_json,POLICY
from vlm_tamp.coupled_resources import api_snapshot
from vlm_tamp.mechanism_surface import MechanismResources,BASELINE
from vlm_tamp.mcvamp_articulated import run_episode
from run_all_prior_tasks import TASKS,configuration,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--tasks',nargs='+',default=['open_top_drawer','close_top_drawer','turn_on_stove'])
    parser.add_argument('--states',nargs='+',type=int,default=[1,11,21,31,41])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--development',action='store_true')
    args=parser.parse_args();tasks=[t for t in TASKS if t['label'] in args.tasks]
    if not tasks or any(t['label'] not in {'open_middle_drawer','open_top_drawer','close_top_drawer','turn_on_stove'} for t in tasks):
        raise ValueError('This adapter supports only the four requested drawer/knob tasks')
    out=(ROOT/args.output).resolve();config=configuration();config.update(version='mcvamp-mechanisms-1.0')
    manifest=[{'task':t['label'],'state':i} for i in args.states for t in tasks]
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[ROOT/'src/vlm_tamp/native/mcvamp_mechanism.cc',Path(__file__).resolve()]
    protected=read_json(BASELINE/'protocol.json')['sources_sha256']
    unchanged=all(sha(ROOT/n)==h for n,h in protected.items())
    if not unchanged:raise ValueError('Historical cohort sources have changed')
    if args.resume:
        protocol=read_json(out/'protocol.json');sources=protocol['sources_sha256']
        if read_json(out/'manifest.json')['episodes']!=manifest:raise ValueError('Manifest mismatch')
    else:
        out.mkdir(parents=True,exist_ok=False);sources={str(p.relative_to(ROOT)):sha(p) for p in files}
        for n in sources:
            target=out/'source-snapshot'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target)
        write_json(out/'manifest.json',{'episodes':manifest,'all_failures_in_denominator':True,'maximum_control_steps':600,'seed':0})
        protocol={'sources_sha256':sources,'configuration':config,'development':args.development,
            'baseline':str(BASELINE),'baseline_sources_unchanged':unchanged,'frozen_before_first_episode':True,
            'native_build':read_json(ROOT/'.runtime/mcvamp-mechanism/build.json'),'api_before':api_snapshot(),
            'storage_policy':POLICY,'execution_vetoes':False,'all_issued_references_completed':True,
            'simulator_scene_truth_policy_input':False,'stove_direction':'bounded physical probes and RGB-D response',
            'scene_model':'current measured endpoints; no convex mechanism/environment admission',
            'comparison_is_multi_component_pipeline_change':True,'held_out_generalization':False}
        write_json(out/'protocol.json',protocol)
    resources=MechanismResources(out);episodes=[]
    for entry in manifest:
        if any(sha(ROOT/n)!=h for n,h in sources.items()):raise ValueError('Frozen experimental source changed')
        folder=out/f'{entry["task"]}-state{entry["state"]:03d}'
        if (folder/'summary.json').exists():
            if not args.resume:raise ValueError('Existing episode')
            result=read_json(folder/'summary.json')
        else:
            print({'event':'started',**entry},flush=True)
            task=next(t for t in tasks if t['label']==entry['task'])
            result=run_episode(task,entry['state'],folder,config,resources)
        episodes.append(result);ledger=api_snapshot()
        write_json(out/'summary.json',{'planned':len(manifest),'completed':len(episodes),'episodes':episodes,
            'native_successes':sum(e['official_task_success'] for e in episodes),
            'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd'],'remaining_api_budget_usd':ledger['remaining_usd']})
        print({'event':'completed',**entry,'success':result['official_task_success'],'failure':result.get('failure'),
            'control_steps':result['control_steps'],'seconds':result['episode_wall_seconds']},flush=True)

if __name__=='__main__':main()
