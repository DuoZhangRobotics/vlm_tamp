"""Persistent Luna practice across declared cases, then frozen-memory evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from run_restriction_study import source_hashes, validate_terminal
from run_contact_skill_study import freeze_memory
from report_restriction_fixes import render
from vlm_tamp.execution_limits import checked_control_steps


def actor_command(out,arguments,*,control_steps,episode_seconds):
    checked_control_steps(control_steps)
    return [sys.executable,str(ROOT/'scripts/run_subscription_agent_pilot.py'),'--output',str(out),
        '--model','gpt-6-luna','--reasoning','high','--reference-mode','slew_limited','--control-steps',str(control_steps),
        '--episode-seconds',str(episode_seconds),'--stop-on-success','--persistent-episode','--surface-contact-tools',
        '--part-motion-tools','--fit-diagnostics','--landmark-evidence','--compact-interface',
        '--visual-inspection','--completion-checks','--no-example',*arguments]


def memory_arguments(memory):
    return ['--guidelines-source',str(memory)] if memory is not None else ['--clean-start']


def validate_run_exit(code,agent,evaluation,*,expected_attempts,interrupted=False):
    attempts=evaluation.get('attempts',[])
    if len(attempts)!=expected_attempts:
        raise RuntimeError('Run did not complete its declared episode count')
    if expected_attempts==1:
        validate_terminal(code,agent,evaluation,interrupted)
        return attempts
    for attempt in attempts:
        validate_terminal(0,agent,{'attempts':[attempt]},interrupted)
    if code and not (code==143 and agent.get('returncode')==-15):
        raise RuntimeError('Unexpected infrastructure exit after practice')
    return attempts


def audit_memory_input(out,expected_hash):
    protocol=read_json(out/'protocol.json')
    actual=out/'agent-workspace/experience/guidelines.json'
    # For clean actors the protocol records the pre-launch absence check.
    # A new notebook written by that actor later is not inherited memory.
    observed=hashlib.sha256(actual.read_bytes()).hexdigest() if expected_hash is not None and actual.exists() else None
    if protocol.get('frozen_agent_guidelines_sha256')!=expected_hash or observed!=expected_hash:
        raise RuntimeError('Delivered memory differs from the declared condition')
    if protocol.get('example_supplied'):
        raise RuntimeError('This study supplies no worked example')
    if expected_hash is None and not protocol.get('clean_start'):
        raise RuntimeError('A no-memory condition must use explicit clean-start validation')
    result={'expected_memory_sha256':expected_hash,'delivered_memory_sha256':observed,
        'passed':True,'worked_example_supplied':False,'evaluation_notebooks_fed_forward':False,
        'experience_file_created_during_clean_run':expected_hash is None and actual.exists()}
    write_json(out/'memory-input-audit.json',result)
    return result


def run_actor(study,name,arguments,*,control_steps,episode_seconds,expected_attempts,expected_memory_hash):
    out=study/'trials'/name
    if out.exists():raise FileExistsError('Preserving existing run '+str(out))
    if os.environ.get('VLM_TAMP_CAPACITY_STATE'):
        from supervise_luna_dashboard_study import wait_for_capacity
        wait_for_capacity(study,name)
    command=actor_command(out,arguments,control_steps=control_steps,episode_seconds=episode_seconds)
    with (study/(name+'.log')).open('x') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT)
        record={'pid':process.pid,'run':name,'status':'running','command':command,'started_at':time.time()}
        write_json(study/'active-process.json',record)
        try:
            while process.poll() is None:render(study);time.sleep(15)
        except BaseException:
            process.send_signal(2)
            try:process.wait(timeout=60)
            except subprocess.TimeoutExpired:process.terminate()
            raise
        record.update(status='completed',returncode=process.returncode,finished_at=time.time())
        write_json(study/'active-process.json',record)
    render(study)
    agent=read_json(out/'agent-exit.json');evaluation=read_json(out/'evaluation-private.json')
    attempts=validate_run_exit(process.returncode,agent,evaluation,expected_attempts=expected_attempts,
        interrupted=(out/'operator-interruption.json').exists())
    if any(a.get('maximum_control_steps')!=control_steps or a['control_steps']>control_steps for a in attempts):
        raise RuntimeError('Observed control budgets do not match the declaration')
    audit_memory_input(out,expected_memory_hash)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',default=str(ROOT/'results/restriction-fixes-20261029'))
    p.add_argument('--tag',default='repaired-v1');p.add_argument('--plan')
    p.add_argument('--prior-memory',help='Optional declared prior; omit for clean practice')
    phases=p.add_mutually_exclusive_group()
    phases.add_argument('--practice-only',action='store_true',help='Freeze practice notes and stop for scope review before fresh evaluation')
    phases.add_argument('--evaluate-only',action='store_true',help='Continue an already completed practice manifest without retraining')
    a=p.parse_args();study=Path(a.study).resolve();study.mkdir(parents=True,exist_ok=True);(study/'trials').mkdir(exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise ValueError('GPU 3 only')
    plan=read_json(Path(a.plan) if a.plan else study/'varied-practice-plan.json')
    controls=checked_control_steps(plan['control_steps']);seconds=int(plan['episode_wall_seconds'])
    if seconds<=0:raise ValueError('Use a positive episode wall budget')
    conditions=plan.get('evaluation_memory_conditions',['frozen'])
    if not conditions or len(set(conditions))!=len(conditions) or any(x not in ['none','frozen'] for x in conditions):
        raise ValueError('Use distinct none/frozen evaluation memory conditions')
    path=study/(a.tag+'-practice-manifest.json')
    if a.evaluate_only:
        manifest=read_json(path)
        if manifest.get('completed') or not manifest.get('practice_completed'):
            raise RuntimeError('Evaluation continuation requires completed practice and pending evaluations')
        if manifest['plan']!=plan:raise RuntimeError('The declared plan changed')
        frozen=manifest['source_sha256'];memory=Path(manifest['frozen_memory'])
    else:
        if path.exists():raise FileExistsError('Choose a new tag')
        frozen=source_hashes();prior=Path(a.prior_memory).resolve() if a.prior_memory else None
        prior_hash=hashlib.sha256(prior.read_bytes()).hexdigest() if prior else None
        if prior_hash!=plan.get('starting_memory_sha256'):
            raise ValueError('Initial memory differs from the declared prior; clean practice requires no prior')
        manifest={'source_sha256':frozen,'prior_memory_sha256':prior_hash,'clean_start':prior is None,
            'plan':plan,'started_at':time.time(),'evaluations':[]}
        write_json(path,manifest)
        attempts=int(plan['initial_and_maximum_attempts']);curriculum=study/plan['practice_cases']
        practice=run_actor(study,a.tag+'-practice',['--task','close_microwave','--state','17',
            '--learning-practice','--practice-cases',str(curriculum),'--attempts',str(attempts),
            '--initial-attempts',str(attempts),'--agent-seconds',str(seconds*attempts),
            '--prompt-file',str(ROOT/'prompts/contact_skills_practice.txt'),*memory_arguments(prior)],
            control_steps=controls,episode_seconds=seconds,expected_attempts=attempts,expected_memory_hash=prior_hash)
        if source_hashes()!=frozen:raise RuntimeError('Runtime source changed during practice')
        write_json(practice/'source-integrity.json',{'sources_unchanged_during_run':True,'checked_files':len(frozen)})
        practice_results=read_json(practice/'evaluation-private.json')['attempts']
        memory=freeze_memory(study,practice,a.tag)
        manifest.update(practice_run=str(practice),practice_attempts=practice_results,practice_completed=True,
            frozen_memory=str(memory),frozen_memory_sha256=hashlib.sha256(memory.read_bytes()).hexdigest())
        write_json(path,manifest)
    if a.practice_only:
        manifest.update(practice_completed=True,evaluations_pending=True,completed=False)
        write_json(path,manifest);render(study);return
    for index,case in enumerate(plan['later_evaluation_cases']):
        order=conditions if index%2==0 else list(reversed(conditions))
        for condition in order:
            if source_hashes()!=frozen:raise RuntimeError('Source changed before frozen-memory evaluation')
            if hashlib.sha256(memory.read_bytes()).hexdigest()!=manifest['frozen_memory_sha256']:
                raise RuntimeError('Frozen learner memory changed')
            name=f'{a.tag}-{condition}-{case["task"]}-state{case["state_index"]:03d}-{controls}'
            if any(row['run']==name for row in manifest['evaluations']):continue
            supplied=memory if condition=='frozen' else None
            out=run_actor(study,name,['--task',case['task'],'--state',str(case['state_index']),
                '--evaluation-case','--attempts','1','--agent-seconds',str(seconds),*memory_arguments(supplied)],
                control_steps=controls,episode_seconds=seconds,expected_attempts=1,
                expected_memory_hash=manifest['frozen_memory_sha256'] if supplied else None)
            if source_hashes()!=frozen:raise RuntimeError('Runtime source changed during evaluation')
            if hashlib.sha256(memory.read_bytes()).hexdigest()!=manifest['frozen_memory_sha256']:
                raise RuntimeError('Frozen learner memory changed')
            write_json(out/'source-integrity.json',{'sources_unchanged_during_run':True,'checked_files':len(frozen)})
            result=read_json(out/'evaluation-private.json')['attempts'][0];agent=read_json(out/'agent-exit.json')
            manifest['evaluations'].append({'run':name,'memory_condition':condition,**result,
                'actor_wall_seconds':agent['wall_seconds']});write_json(path,manifest)
            print(json.dumps({'run':name,'native_success':result['official_task_success'],
                'controls':result['control_steps']}),flush=True)
    manifest.update(completed=True,evaluations_pending=False,finished_at=time.time());write_json(path,manifest);render(study)


if __name__=='__main__':main()
