"""Run matched physical qualification, Luna practice, frozen-memory evaluation."""
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
from vlm_tamp.contact_skill_service import qualification_sources
from report_contact_skills import render


def run(command,log,study):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        write_json(study/'active-process.json',{'pid':process.pid,'command':command,'log':str(log),'started':time.time()})
        try:
            while process.poll() is None:
                render(study,ROOT/'results/luna-live-hinge-comparison-20261027/watch.html')
                time.sleep(15)
        except BaseException:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    render(study,ROOT/'results/luna-live-hinge-comparison-20261027/watch.html')
    return process.returncode


def qualify(study,tag):
    source=ROOT/'results/perception-contract-repair-20261028/trials/close_microwave-state017-perception-repaired'
    events={e['sequence']:e for e in (json.loads(l) for l in (source/'tool-events.jsonl').read_text().splitlines())}
    selection=events[39]['request']['arguments'];panel=events[49]['request']['arguments']['regions']
    push={'camera':selection['camera'],'region_xyxy':selection['region_xyxy'],
        'target_region_xyxy':panel['moving_panel_box_xyxy'],'distance_m':.12,'budget_seconds':120}
    handle=events[5]['request']['arguments']['regions']
    hook={'camera':handle['camera'],'region_xyxy':handle['contact_box_xyxy'],
        'target_region_xyxy':handle['moving_panel_box_xyxy'],'distance_m':.03,'budget_seconds':120,
        '_recorded_ground_regions':handle,'_derive_joint_pull':True}
    write_json(study/f'{tag}-push-request.json',push);write_json(study/f'{tag}-hook-request.json',hook)
    write_json(study/f'{tag}-impulse-request.json',{**push,'distance_m':.03})
    conditions=[('controlled',.03,'push',459)]+[('controlled',.03,'hook_pull',60)]+[
        ('push_and_release',speed,'push',459) for speed in [.03,.06,.12]]
    rows=[];sources=qualification_sources()
    for mode,speed,kind,checkpoint in conditions:
        for repeat in range(1,4):
            name=f'{tag}-{kind}-{mode}-{round(speed*100):02d}cms-r{repeat}'
            out=study/name
            if out.exists():raise FileExistsError(f'Choose a new tag; preserving {out}')
            command=[sys.executable,str(ROOT/'scripts/validate_contact_skill_checkpoint.py'),
                '--output',str(out),'--replay','--kind',kind,'--checkpoint',str(checkpoint),
                '--mode',mode,'--speed',str(speed),'--request-file',str(study/f'{tag}-{"impulse" if mode=="push_and_release" else "push" if kind=="push" else "hook"}-request.json')]
            code=run(command,study/'logs'/f'{name}.log',study)
            result=read_json(out/'replay.json') if (out/'replay.json').exists() else {'runtime_error':True}
            skill=result.get('skill_result',{});stages=skill.get('stages',[])
            moved=any(s.get('observed_contact_progress_m') is not None and s['observed_contact_progress_m']>.001 for s in stages)
            complete=(skill.get('status')=='push_and_release_complete' if mode=='push_and_release'
                else skill.get('status') in {'requested_progress_reached','task_success'})
            audit=result.get('offline_contact_audit',{})
            # Geom-name-only evidence cannot certify a finger-body tip contact;
            # preserve it as unqualified rather than silently accepting it.
            ambiguous=any('finger' in n and not n.endswith('_pad_collision') for n in audit.get('robot_contact_counts',{}))
            row={'name':name,'kind':kind,'mode':mode,'speed_m_s':speed,'repeat':repeat,
                'process_exit':code,'native_success':result.get('native_success_seen',False),
                'commanded_impulse_distance_m':.03 if mode=='push_and_release' else None,
                'retained_success':result.get('retained_success_verified',False),
                'measured_engagement':moved,'completed_skill':complete,
                'stop_reason':skill.get('stop_reason',result.get('error')),
                'execution_qualified':bool(code==0 and complete and moved and audit and not ambiguous
                    and not audit['non_permitted_robot_parts'] and result.get('settling_control_steps')==20)}
            rows.append(row);write_json(study/f'{tag}-qualification-runs.json',rows)
            print(json.dumps(row),flush=True)
            if code:raise RuntimeError('Qualification runtime error; physical failure and infrastructure failure are separate')
    momentum=[r for r in rows if r['mode']=='push_and_release']
    report={'execution_qualified':len(momentum)==9 and all(r['execution_qualified'] for r in momentum),
        'sources_sha256':sources,'sources_unchanged':sources==qualification_sources(),
        'rows':rows,'scope':'historical Luna regions in matched physical replays; no fresh actor generalization',
        'native_and_retained_success_are_separate':True}
    report['execution_qualified'] &= report['sources_unchanged']
    write_json(study/'momentum-qualification.json',report)


def actor(study,name,task,state,*,practice=False,guidelines=None,practice_attempts=6,initial_attempts=3,control_steps=600):
    out=study/'trials'/name;out.parent.mkdir(exist_ok=True)
    command=[sys.executable,str(ROOT/'scripts/run_subscription_agent_pilot.py'),'--output',str(out),
        '--task',task,'--state',str(state),'--model','gpt-6-luna','--reasoning','high','--control-steps',str(control_steps),
        '--agent-seconds',str(1800*practice_attempts) if practice else '1800','--episode-seconds','1800',
        '--attempts',str(practice_attempts) if practice else '1','--stop-on-success','--persistent-episode',
        '--surface-contact-tools','--part-motion-tools','--fit-diagnostics','--landmark-evidence',
        '--compact-interface','--visual-inspection','--completion-checks','--no-example']
    if practice:command+=['--learning-practice','--initial-attempts',str(initial_attempts),'--prompt-file',str(ROOT/'prompts/contact_skills_practice.txt')]
    else:command+=['--evaluation-case']
    if guidelines:command+=['--guidelines-source',str(guidelines)]
    code=run(command,study/'logs'/f'{name}.log',study)
    if code:raise RuntimeError(f'Luna runtime failed; inspect {study/"logs"/(name+".log")}')
    return out


def freeze_memory(study,practice,tag):
    source=practice/'agent-workspace/guidelines.json'
    data=read_json(source)
    if not isinstance(data,dict) or not data.get('principles'):raise ValueError('Luna did not leave usable principles')
    # Explicit structured coordinates are not transferable procedural memory.
    def inspect(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if any(term in key.lower() for term in ['pixels_xy','region_xyxy','pose_world','joint_positions','state_index']):
                    raise ValueError('Guidelines contain episode-specific fields; preserve without promoting')
                inspect(item)
        elif isinstance(value,list):
            if len(value)>=3 and all(isinstance(x,(int,float)) for x in value):
                raise ValueError('Guidelines contain an unexplained numeric vector; preserve without promoting')
            for item in value:inspect(item)
    inspect(data)
    folder=study/'frozen-memory'/tag;folder.mkdir(parents=True,exist_ok=False);target=folder/'guidelines.json';shutil.copyfile(source,target)
    evaluation=read_json(practice/'evaluation-private.json')
    outcomes=[bool(a.get('official_task_success')) for a in evaluation.get('attempts',[])]
    write_json(folder/'provenance.json',{'learner':'gpt-6-luna','source':str(source),
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'verbatim_learner_authored':True,
        'teacher_or_reflection_model':False,'held_out_updates':False,'native_attempt_outcomes':outcomes,
        'successes':sum(outcomes),'attempts':len(outcomes),'has_successful_attempt':any(outcomes),
        'consecutive_successes_observed':any(a and b for a,b in zip(outcomes,outcomes[1:]))})
    return target


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',default=str(ROOT/'results/contact-skills-20261028'))
    p.add_argument('--phase',choices=['qualify','practice','evaluate','all'],default='all');p.add_argument('--tag',default='validated-v1')
    p.add_argument('--practice-attempts',type=int,default=6);p.add_argument('--initial-attempts',type=int,default=3)
    p.add_argument('--prior-guidelines');p.add_argument('--memory-tag')
    p.add_argument('--control-steps',type=int,default=600,help='Legacy study budget; pass 1200 explicitly for extended recovery')
    a=p.parse_args();study=Path(a.study).resolve();study.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise ValueError('This study is assigned GPU 3 only')
    os.environ['VLM_TAMP_MOMENTUM_QUALIFICATION']=str(study/'momentum-qualification.json')
    write_json(study/f'{a.tag}-manifest.json',{'phase':a.phase,'sources_sha256':qualification_sources(),
        'actor':'gpt-6-luna','gpu':3,'limits':{'controls':a.control_steps,'episode_seconds':1800},
        'fresh_evaluation_cases':[{'task':'close_microwave','state':17},{'task':'close_microwave','state':49},{'task':'open_microwave','state':13}],
        'historically_used_cases_not_claimed_unseen':True,'all_failures_retained':True})
    if a.phase in {'qualify','all'}:qualify(study,a.tag)
    if a.phase in {'practice','all'}:
        practice=actor(study,f'{a.tag}-practice-close17','close_microwave',17,practice=True,
            guidelines=a.prior_guidelines,practice_attempts=a.practice_attempts,initial_attempts=a.initial_attempts,control_steps=a.control_steps)
        freeze_memory(study,practice,a.tag)
    if a.phase in {'evaluate','all'}:
        memory=study/'frozen-memory'/(a.memory_tag or a.tag)/'guidelines.json'
        if not memory.exists():raise FileNotFoundError('Freeze learner-authored practice memory before evaluation')
        for task,state in [('close_microwave',17),('close_microwave',49),('open_microwave',13)]:
            actor(study,f'{a.tag}-{task}-state{state:03d}',task,state,guidelines=memory,control_steps=a.control_steps)
    write_json(study/f'{a.tag}-completed.json',{'phase':a.phase,'completed':True,'time':time.time()})
    render(study,ROOT/'results/luna-live-hinge-comparison-20261027/watch.html')


if __name__=='__main__':main()
