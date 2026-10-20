"""Frozen, matched VLM vs fixed strategy search on continuous physical episodes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
import numpy as np
from run_shared_contact_feedback import Resources as BaseResources,TASKS
from run_success_pipeline import task_configuration
from vlm_tamp.strategy_episode import run_episode
from vlm_tamp.strategy_decision import LunaStrategy,EnumeratedStrategy,HeuristicStrategy,PROMPT
from vlm_tamp.run_api_budget import ledger_for_output,api_run_scope,snapshot
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.profiled_pipeline import compress_retained_observations
from vlm_tamp.strategy_grounding_repair import GroundingRepairMixin


class Resources(GroundingRepairMixin,BaseResources):
    pass


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--api-run-id',required=True)
    parser.add_argument('--cases',nargs='+',required=True,help='task:official-state')
    parser.add_argument('--policies',nargs='+',choices=['enumerated','heuristic','vlm'],default=['heuristic','vlm'])
    parser.add_argument('--allow-paid',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--development',action='store_true')
    parser.add_argument('--reproducible-ik',action='store_true')
    parser.add_argument('--tool-version',choices=['v1','v2'],default='v1')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Output must be under results')
    cases=[]
    for token in args.cases:
        name,index=token.rsplit(':',1);index=int(index)
        if name not in TASKS or TASKS[name]['adapter']!='line' or 'legacy_index' in TASKS[name] or not 0<=index<50:
            raise ValueError('This pilot supports visually grounded drawer tasks and official states 0..49')
        cases.append((name,index))
    if len(set(cases))!=len(cases) or len(set(args.policies))!=len(args.policies):raise ValueError('Duplicate case or policy')
    planned=[{'task':name,'state':index,'policy':policy} for name,index in cases for policy in args.policies]
    if args.tool_version=='v2':
        from vlm_tamp.recovery_decision import PROMPT as RECOVERY_PROMPT,parse_decision as parse_recovery,RecoveryHeuristic
        if not args.reproducible_ik:raise ValueError('Recovery comparisons require reproducible IK')
        if 'enumerated' in args.policies:raise ValueError('Use heuristic or VLM for the recovery comparison')
    if args.resume:
        protocol=read_json(out/'protocol.json')
        if read_json(out/'manifest.json')['episodes']!=planned:raise ValueError('Resume requires the frozen manifest')
    else:
        out.mkdir(parents=True,exist_ok=False)
        files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[Path(__file__),PROMPT,
            ROOT/'scripts/run_shared_contact_feedback.py',ROOT/'scripts/run_success_pipeline.py',
            ROOT/'scripts/run_all_prior_tasks.py',ROOT/'configs/api_budget_policy.json',ROOT/'configs/strategy_feedback_pilot.json']
        if args.tool_version=='v2':files.append(RECOVERY_PROMPT)
        hashes={str(path.relative_to(ROOT)):digest(path) for path in files}
        for name in hashes:
            target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        protocol={'sources_sha256':hashes,'api_run_id':args.api_run_id,'api_budget_scope':'experiment_run','api_budget_usd':2.,
            'development':args.development,'maximum_decision_rounds':5,'maximum_control_steps':600,'seed':0,'settling_actions':5,
            'model':'gpt-6-luna','reasoning_effort':'high','decision_max_output_tokens':32000,'grounding_max_output_tokens':12000,
            'automatic_api_retries':0,'maximum_initial_grounding_validation_corrections':1,
            'execution_vetoes':False,'scene_resets_per_episode':1,
            'single_synthetic_tool_use_example':True,'example_contains_drawer_strategy':False,
            'controller':'fixed shared reactive reference correction in both arms',
            'comparison':'VLM-selected vs declared search control using the same tool library; heuristic retains the old clearance/motion mode ordering. Not the historical full-pipeline comparison.',
            'task_specific_strategy_rules':False,'numerical_geometry_and_contact_priors_retained':True,
            'geometry_model_revision_implemented':False,'cross_episode_rule_learning':False,
            'automatic_contact_mode_fallback':False,'original_reference_promoted_or_replaced':False,
            'maximum_segments_per_goal_call':4,'configuration':{name:task_configuration(TASKS[name])[1] for name,_ in cases},
            'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
        protocol['reproducible_ik'] = args.reproducible_ik
        protocol['strategy_tool_version'] = args.tool_version
        protocol['heuristic_v2_policy'] = 'Retained geometry mode ordering, 25 mm steps, automatic IK rank search; same available tools.'
        protocol['ik_correctness_mode'] = 'Immutable full seed pool; fixed robot-only warmup; fresh cuRobo optimizer per query; graph capture disabled.' if args.reproducible_ik else 'legacy'
        if args.reproducible_ik:
            for configuration in protocol['configuration'].values():
                configuration['curobo_query_reproducible'] = True
        for configuration in protocol['configuration'].values():configuration['strategy_tool_version']=args.tool_version
        write_json(out/'protocol.json',protocol);write_json(out/'manifest.json',{'episodes':planned,'all_failures_in_denominator':True})
    ledger=ledger_for_output(out,args.api_run_id)
    if args.allow_paid:
        from vlm_tamp.cli import load_credentials
        load_credentials('/home/duo/.codex/keys/vlm-tamp.env')
    timeline=Timeline();resources=Resources(out,timeline);resources.allow_paid=args.allow_paid;rows=[]
    with api_run_scope(ledger.run_id):
        resources.warm_models()
        for name,index in cases:
            for kind in args.policies:
                import torch
                random.seed(0);np.random.seed(0);torch.manual_seed(0)
                if torch.cuda.is_available():torch.cuda.manual_seed_all(0)
                if not all(digest(ROOT/n)==h for n,h in protocol['sources_sha256'].items()):raise ValueError('Frozen sources changed')
                folder=out/f'{name}-state{index:03d}-{kind}'
                if (folder/'summary.json').exists():result=read_json(folder/'summary.json')
                else:
                    if folder.exists():raise ValueError('Interrupted episode needs diagnosis; no silent restart')
                    print(json.dumps({'event':'started','task':name,'state':index,'policy':kind}),flush=True)
                    if args.tool_version=='v2':
                        policy=LunaStrategy(ledger,allow_paid=args.allow_paid,prompt_path=RECOVERY_PROMPT,parser=parse_recovery) if kind=='vlm' else RecoveryHeuristic()
                    else:
                        policy=LunaStrategy(ledger,allow_paid=args.allow_paid) if kind=='vlm' else (
                            HeuristicStrategy() if kind=='heuristic' else EnumeratedStrategy())
                    result=run_episode(TASKS[name],index,folder,protocol['configuration'][name],resources,policy)
                    compress_retained_observations(folder);result=read_json(folder/'summary.json')
                    result.update(strategy_policy=kind,development=args.development)
                    write_json(folder/'summary.json',result)
                rows.append(result);timeline.save(out/'timing.json')
                groups={kind:{'planned':sum(e['policy']==kind for e in planned),
                    'completed':sum(r['strategy_policy']==kind for r in rows),
                    'native_successes':sum(r['official_task_success'] for r in rows if r['strategy_policy']==kind)}
                    for kind in args.policies}
                write_json(out/'summary.json',{'planned':len(planned),'completed':len(rows),'by_policy':groups,'episodes':rows,'api':snapshot(ledger)})
                print(json.dumps({'event':'completed','task':name,'state':index,'policy':kind,
                    'success':result['official_task_success'],'control_steps':result['control_steps'],'failure':result.get('failure')}),flush=True)
            resources.cleanup_case()
        resources.cleanup_case();resources.case_cache.rmdir()


if __name__=='__main__':main()
