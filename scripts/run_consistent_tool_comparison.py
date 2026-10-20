"""Compare the scripted baseline and frozen Luna method using shared tools.

This is a newly authorized comparison, separate from the earlier twenty-attempt
practice budget. Four one-attempt agent cases are run, with no memory updates.
"""
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
from replay_subscription_episode import compress_observations
from vlm_tamp.tool_event_audit import native_attempt_outcome


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--reference-transfer',default='results/frozen-prepared-method-transfer-20261020')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'results'):
        raise ValueError('Use a new results directory')
    reference=(ROOT/args.reference_transfer).resolve()
    previous=read_json(reference/'protocol.json')
    states=previous['states']
    if states!=[1,11,21,31]:raise ValueError('This comparison preserves the original four cases')
    frozen=previous['frozen_method_inputs']
    assert all(sha(p)==h for p,h in frozen.items()),'Original transfer memory changed'
    profile_path=next(Path(p) for p in frozen if Path(p).name=='profile.json')
    source_files=list((ROOT/'src/vlm_tamp').rglob('*.py'))
    source_files+=list((ROOT/'src/vlm_tamp/native').glob('*.cc'))
    source_files+=list((ROOT/'scripts').glob('*.py'))
    source_files+=[ROOT/'configs/articulated_libero.json']
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    out.mkdir()
    for name in sources:
        dest=out/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    protocol={'states':states,'scripted_cases':4,'agent_cases':4,'agent_attempts_each':1,
        'new_comparison_agent_attempt_cap':4,'previous_practice_attempts_unchanged':20,
        'reference_transfer':str(reference),'frozen_method_inputs':frozen,
        'sources_sha256':sources,'memory_matches_original_transfer_byte_for_byte':True,
        'no_across_case_memory_updates':True,'native_success_termination':True,
        'maximum_control_steps':600,'seed':0,'requested_model':'gpt-6-luna',
        'reasoning_effort':'high','separately_billed_api_requests_allowed':False,
        'scope':'Same task and asset, same four previously evaluated initial states',
        'comparison':'Shared low-level tools; caller grounding and orchestration remain different',
        'native_build':read_json(ROOT/'.runtime/mcvamp-contact-numerical/build.json'),
        'thread_environment':{k:os.environ.get(k) for k in
                              ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}}
    write_json(out/'protocol.json',protocol)

    def unchanged():
        assert all(sha(p)==h for p,h in frozen.items()),'Frozen method changed'
        assert all(sha(ROOT/p)==h for p,h in sources.items()),'Tools changed during comparison'

    unchanged()
    print(json.dumps({'event':'scripted_started','planned':4}),flush=True)
    with (out/'scripted.log').open('w') as log:
        baseline=subprocess.run([sys.executable,str(ROOT/'scripts/run_preplanned_sequence_regression.py'),
            '--output',str(out/'scripted'),'--tasks','open_microwave',
            '--states',*[str(s) for s in states],'--horizon','remaining','--adaptive-horizon',
            '--payload-boundary','--grasp-seed','1',
            '--proposal-source','results/prepared-final-all-tasks-20261020'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    baseline_summary=read_json(out/'scripted/summary.json') if (out/'scripted/summary.json').exists() else {}
    write_json(out/'scripted-result.json',{'returncode':baseline.returncode,'summary':baseline_summary})
    print(json.dumps({'event':'scripted_completed','returncode':baseline.returncode,
                     'successes':baseline_summary.get('native_successes',0)}),flush=True)
    rows=[]
    for state in states:
        unchanged();folder=out/'luna'/f'state{state:03d}';folder.parent.mkdir(exist_ok=True)
        began=time.perf_counter()
        print(json.dumps({'event':'luna_started','state':state}),flush=True)
        with (out/f'luna-state{state:03d}.log').open('w') as log:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/run_subscription_remembered_agent.py'),
                '--output',str(folder),'--state',str(state),'--memory-profile',str(profile_path),
                '--stop-on-success','--evaluation-case','--agent-seconds','1800'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        path=folder/'remembered-agent-result.json'
        record=read_json(path) if path.exists() else {}
        cleanup=compress_observations(folder) if folder.exists() else {}
        row={'state':state,**native_attempt_outcome(folder),'collection_report_available':path.exists(),
            'returncode':result.returncode,'attempts_started':len(list(folder.glob('attempt[0-9][0-9]'))),
            'wall_seconds':time.perf_counter()-began,'result':str(path),'cleanup':cleanup}
        assert row['attempts_started']<=1,'Agent exceeded the declared one-attempt budget'
        rows.append(row)
        write_json(out/'summary.json',{'states':states,'planned_agent_cases':4,
            'completed_agent_cases':len(rows),'agent_native_successes':sum(r['native_success'] for r in rows),
            'scripted_planned_cases':4,'scripted_completed_cases':baseline_summary.get('completed',0),
            'scripted_native_successes':baseline_summary.get('native_successes',0),'agent_cases':rows,
            'agent_attempts_started':sum(r['attempts_started'] for r in rows),
            'old_results_unchanged':True})
        print(json.dumps({'event':'luna_completed',**{k:v for k,v in row.items() if k!='cleanup'}}),flush=True)
    unchanged()
    contracts=[read_json(out/'scripted'/f'open_microwave-state{s:03d}'/'shared-tool-contract.json')
               for s in states if (out/'scripted'/f'open_microwave-state{s:03d}'/'shared-tool-contract.json').exists()]
    contracts += [read_json(out/'luna'/f'state{s:03d}'/'shared-tool-contract.json')
                  for s in states if (out/'luna'/f'state{s:03d}'/'shared-tool-contract.json').exists()]
    assert len(contracts)==8 and all(c==contracts[0] for c in contracts),'Tool contracts diverged'
    write_json(out/'tool-parity.json',{'identical_contracts':8,'contract':contracts[0],
        'same_implementations':True,'same_higher_level_policy':False})
    print(json.dumps({'event':'finished','tool_contracts_identical':8}),flush=True)


if __name__=='__main__':
    main()
