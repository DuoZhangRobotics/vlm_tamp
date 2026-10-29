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
from run_restriction_study import source_hashes
from run_contact_skill_study import freeze_memory
from report_restriction_fixes import render


def run_actor(study,name,arguments):
    out=study/'trials'/name
    command=[sys.executable,str(ROOT/'scripts/run_subscription_agent_pilot.py'),'--output',str(out),
        '--model','gpt-6-luna','--reasoning','high','--reference-mode','slew_limited','--control-steps','1200',
        '--episode-seconds','1800','--stop-on-success','--persistent-episode','--surface-contact-tools',
        '--part-motion-tools','--fit-diagnostics','--landmark-evidence','--compact-interface',
        '--visual-inspection','--completion-checks','--no-example',*arguments]
    with (study/(name+'.log')).open('w') as stream:
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
    if process.returncode:raise RuntimeError('Actor runtime failed; preserve and inspect '+str(out))
    if not read_json(out/'agent-exit.json').get('normal_terminal'):raise RuntimeError('Actor did not end at a normal terminal condition')
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',default=str(ROOT/'results/restriction-fixes-20261029'))
    p.add_argument('--tag',default='repaired-v1');p.add_argument('--plan')
    p.add_argument('--prior-memory',default=str(ROOT/'prompts/learned/contact-practice-20261028/guidelines.json'))
    p.add_argument('--practice-only',action='store_true',help='Freeze practice notes and stop so their transfer scope can be inspected before fresh evaluation')
    a=p.parse_args();study=Path(a.study).resolve();study.mkdir(parents=True,exist_ok=True);(study/'trials').mkdir(exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise ValueError('GPU 3 only')
    plan=read_json(Path(a.plan) if a.plan else study/'varied-practice-plan.json')
    if plan['control_steps']!=1200 or plan['episode_wall_seconds']!=1800:raise ValueError('This study fixes the recovery budgets')
    path=study/(a.tag+'-practice-manifest.json')
    if path.exists():raise FileExistsError('Choose a new tag')
    frozen=source_hashes();prior=Path(a.prior_memory).resolve();prior_hash=hashlib.sha256(prior.read_bytes()).hexdigest()
    if prior_hash!=plan['starting_memory_sha256']:raise ValueError('Initial memory differs from the declared prior')
    manifest={'source_sha256':frozen,'prior_memory_sha256':prior_hash,'plan':plan,'started_at':time.time(),'evaluations':[]}
    write_json(path,manifest)
    attempts=int(plan['initial_and_maximum_attempts']);curriculum=study/plan['practice_cases']
    practice=run_actor(study,a.tag+'-practice',['--task','close_microwave','--state','17',
        '--learning-practice','--practice-cases',str(curriculum),'--attempts',str(attempts),
        '--initial-attempts',str(attempts),'--agent-seconds',str(1800*attempts),
        '--prompt-file',str(ROOT/'prompts/contact_skills_practice.txt'),'--guidelines-source',str(prior)])
    if source_hashes()!=frozen:raise RuntimeError('Runtime source changed during practice')
    practice_results=read_json(practice/'evaluation-private.json')['attempts']
    if len(practice_results)!=attempts:raise RuntimeError('Practice did not complete its declared initial grant')
    memory=freeze_memory(study,practice,a.tag)
    manifest.update(practice_run=str(practice),practice_attempts=practice_results,
        frozen_memory=str(memory),frozen_memory_sha256=hashlib.sha256(memory.read_bytes()).hexdigest())
    write_json(path,manifest)
    if a.practice_only:
        manifest.update(practice_completed=True,evaluations_pending=True,completed=False)
        write_json(path,manifest);render(study);return
    for case in plan['later_evaluation_cases']:
        if source_hashes()!=frozen:raise RuntimeError('Source changed before frozen-memory evaluation')
        name=f'{a.tag}-transfer-{case["task"]}-state{case["state_index"]:03d}'
        out=run_actor(study,name,['--task',case['task'],'--state',str(case['state_index']),
            '--evaluation-case','--attempts','1','--agent-seconds','1800','--guidelines-source',str(memory)])
        if source_hashes()!=frozen:raise RuntimeError('Runtime source changed during evaluation')
        if hashlib.sha256(memory.read_bytes()).hexdigest()!=manifest['frozen_memory_sha256']:raise RuntimeError('Frozen learner memory changed')
        result=read_json(out/'evaluation-private.json')['attempts']
        if len(result)!=1:raise RuntimeError('Evaluation must contain exactly one physical attempt')
        manifest['evaluations'].append({'run':name,**result[0]});write_json(path,manifest)
    manifest.update(completed=True,finished_at=time.time());write_json(path,manifest);render(study)


if __name__=='__main__':main()
