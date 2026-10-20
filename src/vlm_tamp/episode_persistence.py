"""Task-independent continuation policy for an already-started robot episode."""
import json
from pathlib import Path
import shutil
import subprocess
import time
import re

from .subscription_agent import codex_arguments,codex_resume_arguments


def continuation_decision(status,returncode,remaining_seconds):
    if status.get('task_success'):return 'simulator_success'
    if status.get('remaining_control_steps',600)<=0:return 'control_budget_exhausted'
    if remaining_seconds<=0:return 'wall_budget_exhausted'
    if returncode!=0:return 'agent_runtime_error'
    return 'continue_same_episode'


def continuation_prompt(status,remaining_seconds):
    steps=status.get('remaining_control_steps',600)
    return f"""The episode is still incomplete. {steps} control steps and approximately
{int(max(0,remaining_seconds))} seconds remain in this same episode. Continue
working from the current observations and tool feedback. Do not reset or end
the task while budget remains. Call episode_status to check the current budget.
Choose your own next useful observation, model revision or manipulation.
An unsuccessful plan or previous final response does not end the episode.
Stop on native simulator success or an actual budget limit. Preserve findings
and uncertainty; do not issue meaningless actions just to consume steps."""


def run_persistent_agent(work,out,env,readonly,initial_prompt,seconds):
    work,out=Path(work),Path(out)
    started=time.monotonic();deadline=started+seconds
    (out/'episode-deadline.json').write_text(json.dumps({'deadline_monotonic':deadline,
        'wall_budget_seconds':seconds,'scope':'one physical episode across all resumed agent turns'})+'\n')
    (work/'robot_data/episode-deadline.json').write_text(json.dumps({
        'deadline_monotonic':deadline,'wall_budget_seconds':seconds})+'\n')
    thread_id=None;turns=[];prompt=initial_prompt;code=0;reason=None
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:reason='wall_budget_exhausted';break
        number=len(turns)+1;last=out/f'agent-turn{number:03d}-last.txt'
        command=(codex_arguments(work,last,readonly_paths=readonly,ephemeral=False) if thread_id is None
                 else codex_resume_arguments(work,last,thread_id,readonly_paths=readonly))
        (out/f'agent-turn{number:03d}-command.json').write_text(json.dumps(command,indent=2)+'\n')
        (out/f'agent-turn{number:03d}-prompt.txt').write_text(prompt)
        if number==1:
            (out/'agent-command.json').write_text(json.dumps(command,indent=2)+'\n')
            (out/'agent-prompt.txt').write_text(prompt)
        offset=(out/'agent-events.jsonl').stat().st_size if (out/'agent-events.jsonl').exists() else 0
        begin=time.monotonic()
        with (out/'agent-events.jsonl').open('a') as log,(out/'agent-stderr.log').open('a') as err:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,text=True,env=env,
                cwd=work,stdout=log,stderr=err)
            (out/'agent-process.json').write_text(json.dumps({'pid':process.pid,
                'started':time.time(),'turn':number,'thread_id':thread_id})+'\n')
            try:
                process.stdin.write(prompt);process.stdin.close()
                try:code=process.wait(timeout=max(.1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:code=process.wait(timeout=20)
                    except subprocess.TimeoutExpired:process.kill();code=process.wait(timeout=10)
                    (out/'agent-timeout.json').write_text(json.dumps({'timeout_seconds':seconds})+'\n')
            finally:
                if process.poll() is None:
                    process.terminate();process.wait(timeout=30)
        with (out/'agent-events.jsonl').open() as stream:
            stream.seek(offset)
            for line in stream:
                try:event=json.loads(line)
                except ValueError:continue
                if event.get('type')=='thread.started':
                    observed=event['thread_id']
                    if thread_id is not None and observed!=thread_id:
                        raise RuntimeError('Continuation unexpectedly started a different agent thread')
                    thread_id=observed
        if last.exists():shutil.copyfile(last,out/'agent-last.txt')
        # A CLI turn may end while an already-submitted numerical request is
        # still running. Let that request finish before asking for more work.
        while code==0 and time.monotonic()<deadline:
            pending=[p for p in (work/'requests').glob('*.json')
                     if re.fullmatch(r'[0-9a-f]{32}\.json',p.name)
                     and not (work/'replies'/p.name).exists()]
            if not pending:break
            time.sleep(.2)
        status_path=out/'live-episode-status.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        reason=continuation_decision(status,code,deadline-time.monotonic())
        turns.append({'turn':number,'thread_id':thread_id,'returncode':code,
            'wall_seconds':time.monotonic()-begin,'status_at_turn_end':status,'decision':reason})
        (out/'continuation.json').write_text(json.dumps({'turns':turns,'wall_budget_seconds':seconds,
            'single_physical_episode':True,'automatic_messages':'Budget/status reminders only; no task strategies',
            'termination_reason':reason},indent=2)+'\n')
        if reason!='continue_same_episode':break
        if thread_id is None:raise RuntimeError('Agent session did not return a resumable thread ID')
        prompt=continuation_prompt(status,deadline-time.monotonic())
    (out/'stop-service-reason.json').write_text(json.dumps({'reason':reason})+'\n')
    (out/'agent-exit.json').write_text(json.dumps({'returncode':code,
        'wall_seconds':time.monotonic()-started,'turns':len(turns),'thread_id':thread_id,
        'termination_reason':reason,'normal_terminal':reason in {
            'simulator_success','control_budget_exhausted','wall_budget_exhausted'}},indent=2)+'\n')
    return code
