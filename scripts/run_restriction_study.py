"""Compare fixed-memory Luna under the cleaned policy at declared control budgets."""
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
from report_restriction_fixes import render


def source_hashes():
    paths=[*ROOT.glob('src/vlm_tamp/**/*.py'),*ROOT.glob('src/vlm_tamp/native/*.cc'),
        *ROOT.glob('scripts/*.py'),ROOT/'configs/articulated_libero.json',*ROOT.glob('prompts/*.txt')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_terminal(returncode, agent, evaluation, interrupted=False):
    """Keep task outcomes when the CLI deadline follows a completed episode.

    The pilot preserves a SIGTERM exit as 143 even when the robot has already
    reached success or its episode budget. Require independent episode evidence
    before accepting that exit. An unverified or interrupted episode must still
    stop the study instead of being silently counted as a normal evaluation.
    """
    attempts = evaluation.get('attempts', [])
    if interrupted or not agent.get('normal_terminal') or len(attempts) != 1:
        raise RuntimeError('Evaluation lacks one uninterrupted normal terminal episode')
    attempt = attempts[0]
    reason = attempt.get('reason')
    if attempt.get('premature_end') or not attempt.get('all_trajectories_completed_or_feedback_terminated',
            attempt.get('all_trajectories_completed_or_success_terminated')):
        raise RuntimeError('Physical episode did not complete normally')
    controls = attempt.get('control_steps')
    limit = attempt.get('maximum_control_steps')
    remaining_wall = attempt.get('remaining_wall_seconds')
    control_exhausted = (reason == 'control_budget_exhausted'
        and isinstance(controls, int) and isinstance(limit, int)
        and limit > 0 and controls >= limit)
    wall_exhausted = (reason == 'wall_budget_exhausted'
        and isinstance(remaining_wall, (int, float)) and remaining_wall <= 0)
    success = reason == 'simulator_success' and attempt.get('official_task_success') is True
    if not (control_exhausted or wall_exhausted or success):
        raise RuntimeError('Terminal evidence does not establish success or an exhausted budget')
    if returncode and not (returncode == 143 and agent.get('returncode') == -15
            and agent.get('termination_reason') == reason):
        raise RuntimeError('Unexpected infrastructure exit after episode')
    return attempt


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',default=str(ROOT/'results/restriction-fixes-20261029'))
    p.add_argument('--tag',default='cleaned-v1');p.add_argument('--budgets',nargs='+',type=int,default=[600])
    p.add_argument('--cases',nargs='+',default=['close_microwave:17','close_microwave:49','open_microwave:13'])
    p.add_argument('--memory',default=str(ROOT/'results/contact-skills-20261028/frozen-memory/advisory-v4/guidelines.json'))
    a=p.parse_args();study=Path(a.study).resolve();study.mkdir(parents=True,exist_ok=True);(study/'trials').mkdir(exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise ValueError('This study uses GPU 3 only')
    frozen=source_hashes();memory=Path(a.memory).resolve();memory_hash=hashlib.sha256(memory.read_bytes()).hexdigest()
    manifest={'tag':a.tag,'budgets':a.budgets,'cases':a.cases,'actor':'gpt-6-luna','reasoning':'high',
        'reference_mode':'slew_limited','frozen_memory_sha256':memory_hash,'source_sha256':frozen,
        'historically_used_cases_not_claimed_unseen':True,'started_at':time.time(),'rows':[]}
    path=study/(a.tag+'-manifest.json')
    if path.exists():raise FileExistsError('Choose a fresh study tag')
    write_json(path,manifest)
    for controls in a.budgets:
        for case in a.cases:
            if source_hashes()!=frozen:raise RuntimeError('Runtime sources changed during the frozen comparison')
            task,state=case.rsplit(':',1);name=f'{a.tag}-{task}-state{int(state):03d}-{controls}'
            out=study/'trials'/name
            command=[sys.executable,str(ROOT/'scripts/run_subscription_agent_pilot.py'),'--output',str(out),
                '--task',task,'--state',state,'--control-steps',str(controls),'--reference-mode','slew_limited',
                '--model','gpt-6-luna','--reasoning','high','--agent-seconds','1800','--episode-seconds','1800',
                '--attempts','1','--stop-on-success','--persistent-episode','--surface-contact-tools','--part-motion-tools',
                '--fit-diagnostics','--landmark-evidence','--compact-interface','--visual-inspection','--completion-checks',
                '--no-example','--evaluation-case','--guidelines-source',str(memory)]
            with (study/(name+'.log')).open('w') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                active={'pid':process.pid,'run':name,'status':'running','command':command,'started_at':time.time()}
                write_json(study/'active-process.json',active)
                try:
                    while process.poll() is None:
                        render(study);time.sleep(15)
                except BaseException:
                    process.send_signal(2)
                    try:process.wait(timeout=60)
                    except subprocess.TimeoutExpired:process.terminate()
                    raise
                active.update(status='completed',returncode=process.returncode,finished_at=time.time());write_json(study/'active-process.json',active)
            render(study)
            try:
                evaluation=read_json(out/'evaluation-private.json');agent=read_json(out/'agent-exit.json')
            except (OSError,ValueError) as error:
                raise RuntimeError('Actor infrastructure failure: missing or invalid terminal records') from error
            attempt=validate_terminal(process.returncode,agent,evaluation,(out/'operator-interruption.json').exists())
            unchanged=source_hashes()==frozen
            write_json(out/'source-integrity.json',{'sources_unchanged_during_run':unchanged,'checked_files':len(frozen)})
            if not unchanged:raise RuntimeError('Sources changed before evaluation ended')
            if hashlib.sha256(memory.read_bytes()).hexdigest()!=memory_hash:raise RuntimeError('Frozen memory changed')
            row={'run':name,'budget':controls,**attempt,'actor_wall_seconds':agent['wall_seconds'],
                'runner_returncode':process.returncode,'actor_returncode':agent['returncode'],
                'normal_terminal_verified':True}
            manifest['rows'].append(row);write_json(path,manifest)
            print(json.dumps({k:row.get(k) for k in ['run','official_task_success','control_steps','success_by_600_controls','first_native_success_control_step']}),flush=True)
    manifest.update(completed=True,finished_at=time.time());write_json(path,manifest);render(study)


if __name__=='__main__':main()
