"""Rerun four transfer states with persistence and generic contact choices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.tool_event_audit import native_attempt_outcome
from replay_subscription_episode import compress_observations


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'results'):raise ValueError('Use a new results directory')
    profile_path=ROOT/'configs/subscription_persistent_transfer.json'
    profile=read_json(profile_path);states=[1,11,21,31]
    frozen={str(profile_path):sha(profile_path)}
    for key in ['prompt','common_guide','successful_method']:
        path=ROOT/profile[key];frozen[str(path)]=sha(path)
    sources=list((ROOT/'src/vlm_tamp').rglob('*.py'))+list((ROOT/'src/vlm_tamp/native').glob('*.cc'))
    sources+=list((ROOT/'scripts').glob('*.py'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    out.mkdir()
    for name in hashes:
        dest=out/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    write_json(out/'protocol.json',{'states':states,'attempts_each':1,'planned':4,
        'control_step_budget':600,'wall_budget_seconds':1800,'model':'gpt-6-luna','reasoning_effort':'high',
        'same_context_resumed_until_terminal':True,'voluntary_early_finish_allowed':False,
        'generic_contact_tools':True,'specific_contact_strategy_supplied':False,
        'prompt_path':profile['prompt'],'frozen_inputs':frozen,'sources_sha256':hashes,
        'native_outcome_authoritative':True,'native_success_ends_motion_immediately':True,
        'execution_time_collision_vetoes':False,'no_across_case_memory_updates':True,
        'comparison_source':'results/shared-tools-transfer-20261020/verified-summary.json',
        'comparison_scope':'Persistence, generic contact capability and general prompt change together; not an isolated ablation',
        'seed':0,'separately_billed_api_requests_allowed':False,
        'thread_environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}})
    rows=[]
    for state in states:
        assert all(sha(p)==h for p,h in frozen.items()),'Inputs changed during evaluation'
        assert all(sha(ROOT/p)==h for p,h in hashes.items()),'Tools changed during evaluation'
        folder=out/f'state{state:03d}';began=time.perf_counter()
        print(json.dumps({'event':'started','state':state}),flush=True)
        with (out/f'state{state:03d}.log').open('w') as log:
            process=subprocess.run([sys.executable,str(ROOT/'scripts/run_subscription_remembered_agent.py'),
                '--output',str(folder),'--state',str(state),'--memory-profile',str(profile_path),
                '--evaluation-case','--stop-on-success','--persistent-episode','--surface-contact-tools',
                '--agent-seconds','1800'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        outcome=native_attempt_outcome(folder)
        episode=read_json(folder/'evaluation-private.json')['attempts'][0] if outcome['native_evidence_available'] else {}
        completion=read_json(folder/'agent-exit.json') if (folder/'agent-exit.json').exists() else {}
        compression=compress_observations(folder) if folder.exists() else {}
        row={'state':state,**outcome,'control_steps':episode.get('control_steps',0),
            'returncode':process.returncode,'agent_turns':completion.get('turns',1),
            'termination_reason':completion.get('termination_reason',episode.get('reason','runner_failure')),
            'episode_end_reason':episode.get('reason'),'premature_end':episode.get('premature_end',False),
            'physical_attempts':len(list(folder.glob('attempt[0-9][0-9]'))),
            'wall_seconds':time.perf_counter()-began,'cleanup':compression}
        assert row['physical_attempts']<=1
        rows.append(row)
        write_json(out/'summary.json',{'planned':4,'completed':len(rows),
            'native_successes':sum(r['native_success'] for r in rows),'cases':rows,
            'previous_results_preserved':True,'all_planned_cases_in_denominator':True})
        print(json.dumps({'event':'completed',**{k:v for k,v in row.items() if k!='cleanup'}}),flush=True)
    assert all(sha(p)==h for p,h in frozen.items())
    assert all(sha(ROOT/p)==h for p,h in hashes.items())


if __name__=='__main__':
    main()
