"""Frozen regression, clean persistent practice, then held-out memory evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

from vlm_tamp.coupled_storage import read_json,write_json
from run_restriction_study import source_hashes
import run_varied_contact_practice as runner
from report_luna_pipeline_validation import render

ROOT=Path(__file__).resolve().parents[1]


def plan():
    tasks=[('open_microwave',37),('open_microwave',13),('close_microwave',17),('close_microwave',49)]
    order=[tasks,list(reversed(tasks)),[tasks[i] for i in [1,3,0,2]]]
    baseline=[{'task':task,'state_index':state,'repeat':r+1,
        'run':f'baseline-r{r+1}-{task}-state{state:03d}-600'} for r,cases in enumerate(order) for task,state in cases]
    held=random.Random(20261030).sample([i for i in range(50) if i not in {13,17,37,49}],4)
    transfer=[]
    for i,state in enumerate(held):
        task='open_microwave' if i<2 else 'close_microwave'
        for condition in (['none','frozen'] if i%2==0 else ['frozen','none']):
            transfer.append({'task':task,'state_index':state,'memory_condition':condition,
                'run':f'transfer-{condition}-{task}-state{state:03d}-600'})
    return {'model':'gpt-6-luna','reasoning':'high','control_steps':600,'episode_seconds':1800,
        'baseline':baseline,'practice_attempts':6,
        'practice_cases':[{'task':task,'state_index':state} for task,state in
            [('close_microwave',17),('close_microwave',49),('open_microwave',13)]],
        'transfer':transfer,'holdout_selection_seed':20261030,
        'holdout_scope':'Excluded from this study practice and baseline; not claimed globally unused',
        'evaluation_notebooks_fed_forward':False,'teacher_or_reflection_model':False}


def binary_pins():
    paths=[ROOT/'.runtime/mcvamp-contact-numerical/libmcvamp_contact_numerical.so',
        ROOT/'.runtime/mcvamp-contact-numerical/build.json']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def require_frozen(manifest):
    if source_hashes()!=manifest['source_sha256'] or binary_pins()!=manifest['native_binary_sha256']:
        raise RuntimeError('Frozen source or native binary changed; preserve the study and review before proceeding')
    study=Path(manifest['study_directory'])
    if read_json(study/'plan.json')!=manifest['plan'] or read_json(study/'practice-cases.json')!=manifest['plan']['practice_cases']:
        raise RuntimeError('Frozen case schedule changed')


def collect(study,entry,stage,out):
    agent=read_json(out/'agent-exit.json');attempts=read_json(out/'evaluation-private.json')['attempts']
    result={'stage':stage,'run':out.name,'actor_wall_seconds':agent['wall_seconds'],
        'attempts':attempts,'memory_condition':entry.get('memory_condition','none')}
    if len(attempts)==1:result.update(attempts[0])
    return result


def run_entry(study,manifest,entry,stage,arguments,expected_attempts,memory_hash=None):
    require_frozen(manifest);out=study/'trials'/entry['run']
    manifest.update(phase=stage,active_run=entry['run']);manifest.pop('problem',None)
    write_json(study/'study-manifest.json',manifest)
    print(json.dumps({'event':'started','stage':stage,'run':entry['run']}),flush=True)
    if out.exists():
        if not (out/'agent-exit.json').exists():raise RuntimeError('Existing unfinished run requires investigation: '+str(out))
        agent=read_json(out/'agent-exit.json');evaluation=read_json(out/'evaluation-private.json')
        code=agent.get('returncode',0)
        runner.validate_run_exit(128-code if code<0 else code,agent,evaluation,expected_attempts=expected_attempts,
            interrupted=(out/'operator-interruption.json').exists())
        runner.audit_memory_input(out,memory_hash)
    else:
        out=runner.run_actor(study,entry['run'],arguments,control_steps=600,episode_seconds=1800,
            expected_attempts=expected_attempts,expected_memory_hash=memory_hash)
    require_frozen(manifest)
    write_json(out/'source-integrity.json',{'sources_unchanged_during_run':True,
        'checked_files':len(manifest['source_sha256']),'native_binary_unchanged':True})
    row=collect(study,entry,stage,out)
    expected=(manifest['plan']['practice_cases']*2 if stage=='practice' else [entry])
    if [(a['task'],a['state_index']) for a in row['attempts']] != [(a['task'],a['state_index']) for a in expected]:
        raise RuntimeError('Completed episode labels differ from the declared cases')
    manifest['completed_runs'].append(row)
    manifest['active_run']=None;write_json(study/'study-manifest.json',manifest);render(study)
    print(json.dumps({'event':'completed','stage':stage,'run':entry['run'],
        'outcomes':[(r['official_task_success'],r['control_steps']) for r in row['attempts']]}),flush=True)
    return out


def freeze_reviewed_memory(study,manifest):
    source=study/'trials/clean-practice/agent-workspace/guidelines.json'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    review=read_json(study/'memory-scope-review.json')
    if review.get('source_sha256')!=digest or review.get('transfer_scope_passed') is not True:
        raise RuntimeError('Learner memory has not passed scope review for this exact file')
    data=read_json(source)
    if not isinstance(data,dict) or not data.get('principles'):raise ValueError('Learner notes lack principles')
    folder=study/'frozen-memory/clean-practice';folder.mkdir(parents=True,exist_ok=True)
    target=folder/'guidelines.json'
    if target.exists() and target.read_bytes()!=source.read_bytes():raise RuntimeError('Frozen notes already differ')
    if not target.exists():shutil.copyfile(source,target)
    write_json(folder/'provenance.json',{'source':str(source),'sha256':digest,'verbatim_learner_authored':True,
        'teacher_or_reflection_model':False,'evaluation_notebooks_fed_forward':False,
        'bytes':source.stat().st_size,'whitespace_words':len(source.read_text().split())})
    manifest.update(frozen_memory=str(target),frozen_memory_sha256=digest)
    write_json(study/'study-manifest.json',manifest)
    return target,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--stage',choices=['baseline','practice','transfer'],required=True)
    p.add_argument('--limit',type=int,help='Complete at most this many additional runs in this invocation')
    args=p.parse_args();study=args.study.resolve()
    if not study.is_relative_to(ROOT/'results'):raise ValueError('Use a results directory')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or os.environ.get('MUJOCO_EGL_DEVICE_ID')!='3':
        raise ValueError('CUDA and EGL must both use GPU 3')
    study.mkdir(exist_ok=True);(study/'trials').mkdir(exist_ok=True)
    runner.render=render
    path=study/'study-manifest.json'
    if not path.exists():
        declared=plan();write_json(study/'plan.json',declared);write_json(study/'practice-cases.json',declared['practice_cases'])
        manifest={'study_directory':str(study),'plan':declared,'source_sha256':source_hashes(),'native_binary_sha256':binary_pins(),
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'started_at':time.time(),'phase':'preflight','completed_runs':[],'completed':False}
        write_json(path,manifest)
        write_json(study/'implementation-status.json',{'title':'Luna full-task validation — frozen tools, 600 controls',
            'steps':[{'id':'tool_contracts_and_memory_isolation','status':'validated before study'}]})
    manifest=read_json(path);require_frozen(manifest);declared=manifest['plan']
    if declared!=plan() or read_json(study/'plan.json')!=declared or read_json(study/'practice-cases.json')!=declared['practice_cases']:
        raise RuntimeError('The declared case schedule changed')
    done={r['run'] for r in manifest['completed_runs']}
    active=read_json(study/'active-process.json') if (study/'active-process.json').exists() else {}
    if active.get('status')=='running' and Path(f'/proc/{active["pid"]}/cmdline').exists():
        command=Path(f'/proc/{active["pid"]}/cmdline').read_bytes()
        if b'run_subscription_agent_pilot.py' in command:raise RuntimeError('An actor is still running; monitor it instead of starting a duplicate')
    try:
        if args.stage=='baseline':
            pending=[e for e in declared['baseline'] if e['run'] not in done]
            for entry in pending[:args.limit] if args.limit is not None else pending:
                run_entry(study,manifest,entry,'baseline',['--task',entry['task'],'--state',str(entry['state_index']),
                    '--attempts','1','--agent-seconds','1800','--evaluation-case','--clean-start'],1)
            manifest['baseline_completed']=all(e['run'] in {r['run'] for r in manifest['completed_runs']} for e in declared['baseline'])
        elif args.stage=='practice':
            if not manifest.get('baseline_completed'):raise RuntimeError('Complete the frozen regression first')
            if 'clean-practice' not in done:
                run_entry(study,manifest,{'run':'clean-practice'},'practice',
                    ['--task','close_microwave','--state','17','--attempts','6','--initial-attempts','6',
                     '--agent-seconds','10800','--learning-practice','--clean-start',
                     '--practice-cases',str(study/'practice-cases.json'),
                     '--prompt-file',str(ROOT/'prompts/contact_skills_practice.txt')],6)
            manifest.update(practice_completed=True,phase='memory_scope_review')
        else:
            if not manifest.get('practice_completed'):raise RuntimeError('Complete clean practice first')
            memory,digest=freeze_reviewed_memory(study,manifest)
            for entry in declared['transfer']:
                if entry['run'] in done:continue
                if hashlib.sha256(memory.read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen memory changed')
                supplied=entry['memory_condition']=='frozen'
                run_entry(study,manifest,entry,'transfer',['--task',entry['task'],'--state',str(entry['state_index']),
                    '--attempts','1','--agent-seconds','1800','--evaluation-case',
                    *(['--guidelines-source',str(memory)] if supplied else ['--clean-start'])],1,digest if supplied else None)
            manifest.update(completed=True,phase='complete',finished_at=time.time())
    except BaseException as error:
        manifest['problem']={'type':type(error).__name__,'reason':str(error),'recorded_at':time.time(),
            'existing_attempts_preserved':True}
        write_json(path,manifest);render(study);raise
    write_json(path,manifest);render(study)


if __name__=='__main__':main()
