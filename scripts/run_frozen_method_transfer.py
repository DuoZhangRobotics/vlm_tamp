"""One-attempt frozen-method transfer using only the remaining practice budget."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--profile',required=True)
    parser.add_argument('--states',nargs='+',type=int,default=[1,11,21,31])
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'results'):raise ValueError('Use a new results directory')
    if len(set(args.states))!=len(args.states) or any(not 0<=s<50 for s in args.states):
        raise ValueError('Use unique official initial-state indices')
    prior=[ROOT/'results'/n for n in ['subscription-scene-sync-practice-20261020',
                                    'subscription-prepared-sequence-practice-20261020']]
    used=sum(sum(p.is_dir() for p in folder.glob('attempt[0-9][0-9]')) for folder in prior)
    if used+len(args.states)>20:raise ValueError('Transfer would exceed the combined twenty-attempt budget')
    profile_path=ROOT/args.profile;profile=json.loads(profile_path.read_text())
    frozen={str(profile_path):sha(profile_path)}
    for key in ['successful_method','common_guide','prompt']:
        path=ROOT/profile[key];frozen[str(path)]=sha(path)
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+list((ROOT/'src/vlm_tamp/native').glob('*.cc'))
    files += [ROOT/'scripts'/n for n in ['serve_subscription_robot.py','prepare_subscription_pilot.py',
        'run_subscription_agent_pilot.py','run_subscription_remembered_agent.py']]
    sources={str(p.relative_to(ROOT)):sha(p) for p in files}
    out.mkdir()
    protocol={'states':args.states,'attempts_each':1,'earlier_started_attempts':used,
        'combined_attempt_cap':20,'frozen_method_inputs':frozen,'tool_source_sha256':sources,
        'no_across_case_memory_updates':True,'native_success_termination':True,
        'scope':'Different initial states of the same microwave task; previously inspected benchmark cases, not novel assets',
        'api_key_authentication':False,'requested_model':'gpt-6-luna','reasoning_effort':'high'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    for state in args.states:
        assert all(sha(Path(p))==h for p,h in frozen.items()),'Frozen method input changed'
        assert all(sha(ROOT/p)==h for p,h in sources.items()),'Tool implementation changed'
        folder=out/f'state{state:03d}';before=time.perf_counter()
        print(json.dumps({'event':'started','state':state}),flush=True)
        with (out/f'state{state:03d}.log').open('w') as log:
            completed=subprocess.run([sys.executable,str(ROOT/'scripts/run_subscription_remembered_agent.py'),
                '--output',str(folder),'--state',str(state),'--memory-profile',str(profile_path),
                '--stop-on-success','--evaluation-case','--agent-seconds','1800'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        result_path=folder/'remembered-agent-result.json'
        record=json.loads(result_path.read_text()) if result_path.exists() else {}
        row={'state':state,'returncode':completed.returncode,'native_success':bool(record.get('first_attempt_success')),
            'attempts_started':sum(p.is_dir() for p in folder.glob('attempt[0-9][0-9]')),
            'wall_seconds':time.perf_counter()-before,'result':str(result_path)}
        rows.append(row)
        (out/'summary.json').write_text(json.dumps({'planned':len(args.states),'completed':len(rows),
            'native_successes':sum(r['native_success'] for r in rows),'cases':rows,
            'combined_started_attempts':used+sum(r['attempts_started'] for r in rows)},indent=2)+'\n')
        print(json.dumps({'event':'completed',**row}),flush=True)
    assert all(sha(Path(p))==h for p,h in frozen.items())


if __name__=='__main__':main()
