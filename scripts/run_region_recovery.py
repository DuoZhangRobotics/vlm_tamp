"""Matched Luna control versus collision evidence and RGB-region revision."""
import argparse
import hashlib
from pathlib import Path
import random
import shutil
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
import numpy as np
from run_strategy_feedback import Resources,TASKS
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.profiled_pipeline import compress_retained_observations
from vlm_tamp.strategy_episode import run_episode
from vlm_tamp.strategy_decision import LunaStrategy
from vlm_tamp.recovery_decision import PROMPT as BASE_PROMPT,parse_decision as parse_base
from vlm_tamp.region_decision import PROMPT as REGION_PROMPT,parse_decision as parse_region
from vlm_tamp.run_api_budget import api_run_scope,ledger_for_output,snapshot

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--api-run-id',required=True)
    parser.add_argument('--states',nargs='+',type=int,required=True)
    parser.add_argument('--policies',nargs='+',choices=['baseline','region','separate'],default=['baseline','region'])
    parser.add_argument('--development',action='store_true');parser.add_argument('--allow-paid',action='store_true')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Output must be in results')
    if len(set(args.states))!=len(args.states) or any(not 0<=s<50 for s in args.states):raise ValueError('Distinct official states 0..49 required')
    if len(set(args.policies))!=len(args.policies):raise ValueError('Duplicate policy')
    out.mkdir(exist_ok=False)
    config=read_json(ROOT/'results/vlm-recovery-frozen-20261018/protocol.json')['configuration']['close_bottom_drawer']
    config={**config,'collision_evidence_views':True}
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[Path(__file__),BASE_PROMPT,REGION_PROMPT,
        ROOT/'scripts/run_strategy_feedback.py',ROOT/'scripts/run_shared_contact_feedback.py',ROOT/'scripts/run_success_pipeline.py',
        ROOT/'src/vlm_tamp/native/mcvamp_mechanism.cc']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name in hashes:
        dest=out/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    planned=[{'task':'close_bottom_drawer','state':s,'policy':p} for s in args.states for p in args.policies]
    write_json(out/'manifest.json',{'episodes':planned,'all_failures_in_denominator':True})
    write_json(out/'protocol.json',{'sources_sha256':hashes,'configuration':config,'development':args.development,
        'model':'gpt-6-luna','reasoning_effort':'high','max_output_tokens':32000,
        'maximum_decision_rounds':8,'maximum_control_steps':600,'seed':0,'settling_actions':5,
        'api_run_id':args.api_run_id,'api_budget_usd':2.,'automatic_retries':0,
        'comparison':'Luna v2 tools versus v3 evidence/region revision; both get eight decisions. Not isolated prompt or visualization ablation.',
        'baseline_prompt':str(BASE_PROMPT.relative_to(ROOT)),'region_prompt':str(REGION_PROMPT.relative_to(ROOT)),
        'collision_evidence_logged_in_both_conditions':True,'collision_evidence_sent_to_baseline':False,
        'image_input_mode':{p:'separate_cameras' if p=='separate' else 'composite' for p in args.policies},
        'motion_planner_controller_collision_policy_changed':False,'new_surface_contact_frame':'gravity-aligned local patch tangent; explicit surface revisions only',
        'human_replacement_coordinates_or_object_identity_hint_sent':False,'simulator_object_truth_used_for_control':False,
        'execution_vetoes':False,'historical_results_preserved':True,'previous_five_case_reference_promoted_or_replaced':False})
    ledger=ledger_for_output(out,args.api_run_id)
    if args.allow_paid:
        from vlm_tamp.cli import load_credentials
        load_credentials('/home/duo/.codex/keys/vlm-tamp.env')
    resources=Resources(out,Timeline());resources.allow_paid=args.allow_paid;rows=[]
    with api_run_scope(ledger.run_id):
        resources.warm_models()
        for state in args.states:
            for kind in args.policies:
                import torch
                random.seed(0);np.random.seed(0);torch.manual_seed(0)
                if torch.cuda.is_available():torch.cuda.manual_seed_all(0)
                if not all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()):
                    raise ValueError('Frozen sources changed')
                if kind=='separate':
                    from vlm_tamp.region_multiview import SeparateLunaStrategy
                    policy=SeparateLunaStrategy(ledger,allow_paid=args.allow_paid)
                else:
                    policy=LunaStrategy(ledger,allow_paid=args.allow_paid,prompt_path=REGION_PROMPT if kind=='region' else BASE_PROMPT,
                        parser=parse_region if kind=='region' else parse_base)
                local={**config,'strategy_tool_version':'v2' if kind=='baseline' else 'v3','separate_camera_inputs':kind=='separate'}
                folder=out/f'close_bottom_drawer-state{state:03d}-{kind}'
                print(json.dumps({'event':'started','state':state,'policy':kind}),flush=True)
                result=run_episode(TASKS['close_bottom_drawer'],state,folder,local,resources,policy,maximum_rounds=8)
                compress_retained_observations(folder);result=read_json(folder/'summary.json')
                result.update(strategy_policy=kind,development=args.development)
                write_json(folder/'summary.json',result);rows.append(result)
                write_json(out/'summary.json',{'planned':len(planned),'completed':len(rows),'episodes':rows,'api':snapshot(ledger)})
                print(json.dumps({'event':'completed','state':state,'policy':kind,'success':result['official_task_success'],
                    'control_steps':result['control_steps'],'failure':result.get('failure')}),flush=True)
            resources.cleanup_case()
        resources.cleanup_case();resources.case_cache.rmdir()

if __name__=='__main__':main()
