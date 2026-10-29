"""Fixed-tool practice followed by paired frozen-memory transfer."""
import argparse
from collections import Counter
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
from vlm_tamp.recovery_memory import freeze_lessons
from vlm_tamp.tool_event_audit import native_attempt_outcome,operation_name
from replay_subscription_episode import compress_observations
from run_subscription_learning_practice import summarize
from run_subscription_dino_study import retained_inputs
from vlm_tamp.storage_budget import InsufficientStorage,require_storage_headroom


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnostic(source):
    errors=Counter();physical=[]
    for line in (source/'tool-events.jsonl').read_text().splitlines():
        event=json.loads(line);response=event['response'];result=response.get('result',{})
        operation=operation_name(event)
        if not response.get('ok') or result.get('planned') is False:
            reason=str(response.get('error',result.get('reason',result.get('failure_reason','planned=false'))))
            errors[(operation,reason)]+=1
        if operation in {'execute','execute_sequence','gripper'}:
            physical.append({'sequence':event['sequence'],'operation':operation,
                             'request':event['request'],'response':response})
    return {'source':str(source.relative_to(ROOT)),'actor_receives_this_diagnostic':False,
            'failure_counts':[{'operation':op,'reason':why,'count':n} for (op,why),n in errors.most_common()],
            'physical_calls':physical,
            'interpretation':'Distinguish rejected plans from ineffective or poorly tracked physical motion. No change to the numerical tools is made from this audit.'}


def base_command(folder,state,prompt,seconds,*,dino_tools=True):
    return [sys.executable,str(ROOT/'scripts/run_subscription_agent_pilot.py'),
        '--output',str(folder),'--state',str(state),'--scene-variant','nominal',
        '--prompt-file',str(prompt),'--agent-seconds',str(seconds),
        '--fit-diagnostics',*(['--dino-tools'] if dino_tools else []),'--abstract-experience',
        '--stop-on-success','--persistent-episode','--surface-contact-tools']


def run_logged(command,path):
    try:
        require_storage_headroom(ROOT)
    except InsufficientStorage as error:
        status_path=path.parent/'status.json'
        status=read_json(status_path) if status_path.exists() else {}
        write_json(status_path,{**status,'phase':'storage_blocked','active':False,
            'reason':str(error),'next_run_log':path.name})
        raise
    began=time.perf_counter()
    with path.open('w') as log:
        completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    return completed.returncode,time.perf_counter()-began


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--transfer-states',type=int,nargs='+',default=[2,12,32,44])
    parser.add_argument('--practice-session-seconds',type=int,default=36000)
    parser.add_argument('--disable-dino',action='store_true',help='Declare the same reduced optional tool set for every condition')
    args=parser.parse_args()
    if (not args.transfer_states or len(set(args.transfer_states))!=len(args.transfer_states)
            or any(not 0<=s<50 or s==1 for s in args.transfer_states)):
        raise ValueError('Transfer needs distinct official states outside the practice state')
    if args.practice_session_seconds<1800:
        raise ValueError('Practice session must permit at least one complete episode budget')
    planned=2*len(args.transfer_states)
    out=(ROOT/args.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'results'):raise ValueError('Use a new results directory')
    require_storage_headroom(ROOT)
    out.mkdir()
    inputs=out/'inputs';inputs.mkdir()
    profile=read_json(ROOT/'configs/subscription_persistent_transfer.json')
    for key in ['common_guide','successful_method']:
        source=ROOT/profile[key]
        assert sha(source)==profile[key+'_sha256']
        shutil.copyfile(source,inputs/(key+'.json'))
    for stage in ['practice','transfer']:
        shutil.copyfile(ROOT/f'prompts/recovery_learning_{stage}.txt',inputs/f'{stage}-prompt.txt')
    sources=list((ROOT/'src/vlm_tamp').rglob('*.py'))+list((ROOT/'src/vlm_tamp/native').glob('*.cc'))
    sources+=list((ROOT/'scripts').glob('*.py'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs.iterdir()}
    def verify():
        assert all(sha(ROOT/p)==h for p,h in hashes.items()),'Study tools changed'
        assert all(sha(ROOT/p)==h for p,h in input_hashes.items()),'Frozen study inputs changed'
    protocol={'development_state':1,'initial_practice_attempts':5,'maximum_practice_attempts':20,
        'practice_session_seconds':args.practice_session_seconds,'seconds_per_episode':1800,'control_steps_per_episode':600,
        'transfer_states':args.transfer_states,'conditions':['without_learned_memory','with_learned_memory'],
        'transfer_attempts_each':1,'planned_transfer_episodes':planned,'seed':0,
        'optional_dino_tools_available':not args.disable_dino,
        'source_sha256':hashes,'input_sha256':input_hashes,
        'model':'gpt-6-luna','reasoning_effort':'high','authentication':'ChatGPT subscription',
        'paid_api_requests_allowed':False,'numerical_algorithms_unchanged':True,
        'shared_scaffold':'Identical tool documentation, abstract worked example, prompts, model and limits in both transfer conditions',
        'memory_difference':'Only with_learned_memory receives the frozen agent-authored practice synthesis; neither transfer condition receives the earlier episodic library or successful-method file',
        'held_out_scope':'Fresh relative to scanned microwave-agent protocol manifests and this practice session; same task and asset, not new objects',
        'no_cross_evaluation_case_updates':True,'native_success_authoritative':True,
        'execution_time_vetoes':False,'cleanup_after_each_attempt':True,
        'thread_environment':{key:os.environ.get(key) for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}
    write_json(out/'protocol.json',protocol)
    write_json(out/'diagnostic-private.json',diagnostic(ROOT/'results/persistent-contact-transfer-v2-20261020/state001'))
    write_json(out/'status.json',{'phase':'practice','active':True})
    practice=out/'practice'
    command=base_command(practice,1,inputs/'practice-prompt.txt',args.practice_session_seconds,
        dino_tools=not args.disable_dino)+[
        '--attempts','20','--initial-attempts','5','--learning-practice','--episode-seconds','1800',
        '--guidelines-source',str(inputs/'common_guide.json'),
        '--successful-method-file',str(inputs/'successful_method.json'),
        '--memory-store',str(ROOT/profile['public_library'])]
    verify()
    print(json.dumps({'event':'practice_started','output':str(practice)}),flush=True)
    code,elapsed=run_logged(command,out/'practice.log')
    summary=summarize(practice,20,code,elapsed)
    summary['raw_input_retention']=retained_inputs(practice)
    summary['lossless_compression']=compress_observations(practice)
    write_json(practice/'learning-summary.json',summary)
    verify()
    print(json.dumps({'event':'practice_completed','returncode':code,'attempts':summary['attempts_used'],
        'successes':summary['native_goal_reached_attempts']}),flush=True)
    termination=read_json(practice/'agent-exit.json') if (practice/'agent-exit.json').exists() else {}
    normal_budget_exit=(termination.get('normal_terminal') and
        termination.get('termination_reason')=='practice_wall_budget_exhausted')
    if code and not normal_budget_exit:
        write_json(out/'status.json',{'phase':'practice_infrastructure_failure','active':False,'returncode':code,
            'transfer_planned':planned,'transfer_started':0})
        raise SystemExit(code)
    frozen=freeze_lessons(practice)
    memory=inputs/'learned-guidelines.json'
    write_json(memory,frozen)
    input_hashes[str(memory.relative_to(ROOT))]=sha(memory)
    write_json(out/'freeze.json',{'memory_sha256':sha(memory),'source':str(practice),
        'practice_native_successes':summary['native_goal_reached_attempts'],'tools_unchanged':True,
        'scope':'Conditional agent-authored lessons, including failed or unconfirmed hypotheses; not certified universal rules'})
    rows=[]
    for i,state in enumerate(protocol['transfer_states']):
        conditions=protocol['conditions'] if i%2==0 else list(reversed(protocol['conditions']))
        for condition in conditions:
            verify()
            folder=out/'transfer'/f'state{state:03d}-{condition}'
            folder.parent.mkdir(exist_ok=True)
            write_json(out/'status.json',{'phase':'transfer','active':True,'state':state,
                'condition':condition,'completed':len(rows),'planned':planned})
            command=base_command(folder,state,inputs/'transfer-prompt.txt',1800,
                dino_tools=not args.disable_dino)+['--attempts','1','--evaluation-case']
            if condition=='with_learned_memory':command+=['--guidelines-source',str(memory)]
            print(json.dumps({'event':'transfer_started','state':state,'condition':condition}),flush=True)
            code,elapsed=run_logged(command,out/f'state{state:03d}-{condition}.log')
            outcome=native_attempt_outcome(folder)
            episodes=read_json(folder/'evaluation-private.json')['attempts'] if (folder/'evaluation-private.json').exists() else []
            cleanup={
                'raw_input_retention':retained_inputs(folder) if (folder/'agent-workspace').exists() else {},
                'lossless_compression':compress_observations(folder) if folder.exists() else [],
            }
            row={'state':state,'condition':condition,'returncode':code,'wall_seconds':elapsed,
                **outcome,'episodes':episodes,'physical_attempts':len(list(folder.glob('attempt[0-9][0-9]'))),
                'cleanup':cleanup}
            assert row['physical_attempts']<=1
            rows.append(row)
            write_json(out/'transfer-summary.json',{'planned':planned,'completed':len(rows),'cases':rows,
                'scores':{c:{'native_successes':sum(r['native_success'] for r in rows if r['condition']==c),
                             'planned':len(protocol['transfer_states']),'completed':sum(r['condition']==c for r in rows)} for c in protocol['conditions']},
                'all_planned_cases_in_denominator':True})
            print(json.dumps({'event':'transfer_completed','state':state,'condition':condition,
                'native_success':outcome['native_success'],'steps':episodes[0]['control_steps'] if episodes else None,
                'returncode':code}),flush=True)
    verify()
    write_json(out/'status.json',{'phase':'complete','active':False,'completed_transfer':len(rows),
        'source_and_inputs_unchanged':True,'cleanup_after_each_attempt':True})


if __name__=='__main__':
    main()
