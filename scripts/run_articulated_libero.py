"""Official articulated tasks with RGB-D grounding and physical execution."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'.runtime/mc-vamp-deps')]

from try_rekep_bowl_v3 import save
from vlm_tamp.rekep_libero.articulated_grounding import ArticulatedLedger
from vlm_tamp.rekep_libero.mechanism_execution import run_episode
from vlm_tamp.rekep_libero.perception import Perception,GraspGen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--tasks',nargs='+')
    parser.add_argument('--states',nargs='+',type=int,default=[1])
    parser.add_argument('--scene-mode',choices=['strict','diagnostic'],default='strict')
    parser.add_argument('--reuse-run')
    parser.add_argument('--development',action='store_true')
    args=parser.parse_args()
    output=ROOT/args.output
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/articulated_libero.json').read_text())
    config['scene_mode']=args.scene_mode
    tasks=[row for row in config['tasks'] if not args.tasks or row['label'] in args.tasks]
    planned=[{'task':row['label'],'name':row['name'],'state':state} for row in tasks for state in args.states]
    if not planned:
        raise ValueError('No planned tasks')
    save(output/'manifest.json',{'planned_episodes':planned,'all_in_denominator':True,
        'seed':0,'maximum_control_steps':600,'development':args.development})
    save(output/'configuration.json',config)
    files=list((ROOT/'src/vlm_tamp/rekep_libero').glob('*.py'))
    files += [Path(__file__),ROOT/'configs/articulated_libero.json',ROOT/'scripts/try_rekep_bowl_v3.py',
        ROOT/'src/vlm_tamp/drawer_local_plane.py',ROOT/'src/vlm_tamp/pose_constraints.py',
        ROOT/'requirements-curobo-libero.txt',ROOT/'docs/articulated_libero.md',ROOT/'tests/test_mechanism_geometry.py']
    hashes={}
    for source in files:
        relative=source.relative_to(ROOT)
        destination=output/'source-snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    before=ArticulatedLedger(ROOT/'results/articulated-api-v1').snapshot()
    save(output/'protocol.json',{'sources_sha256':hashes,'development':args.development,
        'development_state':0,'evaluation_states':args.states,'scene_mode':args.scene_mode,
        'task_templates_developed_on_these_tasks':True,'held_out_task_generalization_claim':False,
        'simulator_scene_state_used':False,'physical_contacts_enabled':True,'assisted_grasping':False,
        'curobo_ik_used':True,'curobo_motiongen_used':False,'curobo_revision':config['curobo_revision'],
        'graspgen_revision':'2dd8852e1be60f5f9d277fafcc621835cdf59110',
        'new_api_ledger':'results/articulated-api-v1/ledger.json','cumulative_api_spend_before_usd':before['cumulative_conservative_spend_usd'],
        'api_budget_usd':2.,'api_retries':0,'all_planned_failures_counted':True,'collision_free_guarantee':False})
    perception,grasps=Perception(),GraspGen()
    summaries=[]
    for row in planned:
        folder=output/f'{row["task"]}-state{row["state"]:03d}'
        if args.reuse_run:
            upstream=ROOT/args.reuse_run/f'{row["task"]}-state{row["state"]:03d}'/'summary.json'
            deadline=time.monotonic()+360
            while not upstream.is_file():
                if time.monotonic()>deadline:
                    raise TimeoutError('Input-producing episode did not finish within the bounded wait')
                time.sleep(1)
        summaries.append(run_episode(row['task'],row['name'],row['state'],folder,config,perception,grasps,args.development,
                                     ROOT/args.reuse_run if args.reuse_run else None))
        ledger=ArticulatedLedger(ROOT/'results/articulated-api-v1').snapshot()
        save(output/'summary.json',{'planned_episodes':len(planned),'completed_episodes':len(summaries),
            'official_task_successes':sum(r['official_task_success'] for r in summaries),'episodes':summaries,
            'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd'],'remaining_api_budget_usd':ledger['remaining_usd']})


if __name__=='__main__':
    main()
