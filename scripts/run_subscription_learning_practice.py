"""Adaptive development practice within one isolated Luna/high session."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from run_subscription_remembered_agent import command as registered_command
from run_subscription_dino_study import read, write, retained_inputs
from replay_subscription_episode import compress_observations
from vlm_tamp.episodic_memory import store_hashes
from vlm_tamp.subscription_agent import validate_attempt_budget


def practice_prompt(registered_prompt, attempts, initial_attempts=None):
    initial_attempts = min(5, attempts) if initial_attempts is None else initial_attempts
    validate_attempt_budget(attempts, learning_practice=True, initial_attempts=initial_attempts)
    replacements = {
        '''There is ONE physical attempt, seed
and scene supplied by the experiment, and 600 recorded control steps including
reset's initial hold. Call reset once, inspect current images, and call
memory.recall with the returned instruction and robot Panda before movement.''':
        f'''There are initially {initial_attempts} granted attempts and a hard cap of
{attempts} physical attempts on the same development scene, with 600 recorded control steps per attempt including reset's initial hold.
This is one continuous learning session. Each reset restores the initial scene;
your reasoning context and saved notes persist. Inspect fresh images after each
reset. Before the first movement, call memory.recall with the returned instruction
and robot Panda.''',
        '''Never
reset for another attempt.''':
        '''Save your attempt findings before resetting for the next practice
attempt. Re-measure current geometry and obtain fresh plans after every reset.''',
        '''Completed requires a fresh supported measured check. Incomplete and uncertain
remain legitimate outcomes if you cannot establish the intended result.''':
        '''Completion measurements are advisory during practice. Only the simulator
success signal completes this benchmark episode. State evidence and uncertainty;
an unavailable measurement alone is not a reason to stop investigating.
Incomplete and uncertain remain legitimate judgments for unsuccessful attempts.''',
    }
    prompt = registered_prompt
    if 'Episode protocol: native-success-v1.' in prompt:
        old='Episode budget: one physical attempt with at most 600 control steps. Reset once; do not retry with another reset.'
        if prompt.count(old)!=1:raise ValueError('Native episode budget block changed')
        prompt=prompt.replace(old,f'''Episode budget: initially {initial_attempts} granted attempts,
with a hard maximum of {attempts}, and 600 recorded control steps per attempt.
This is one continuous learning session. Each reset restores the initial scene;
notes persist. Re-measure geometry and make fresh plans after every reset.
Completion measurements are advisory; native simulator success ends the episode.''')
    else:
        for old, new in replacements.items():
            if prompt.count(old) != 1:
                raise ValueError('Registered prompt changed; review practice adaptation')
            prompt = prompt.replace(old, new)
        first = prompt.index('Before moving, use declare_completion')
        last = prompt.index('\nUse fresh plans via robot.py.', first)
        prompt = prompt[:first]+'''Before moving, interpret the instruction and identify its intended end state.
Judge progress from fresh images and other available observations. Optional
measure_surface, compare_surfaces, DINO and check_completion tools can inform
that judgment. If you choose quantitative criteria, declare them before motion.
No particular matching algorithm or numerical completion check is mandatory.
Do not confuse unavailable measurement evidence with physical task failure;
inspect another view or investigate when useful. Planned trajectories alone do
not prove object motion. Record evidence and uncertainty without reading native
task predicates or changing the goal to fit partial progress.
'''+prompt[last:]
    prompt=prompt.replace('Every issued trajectory executes fully.',
        'Every issued trajectory executes until complete or the simulator reports success.')
    full = prompt + f'''
Practice protocol: start with {initial_attempts} attempts, extending up to {attempts}
only when further tests are informative. Use them to investigate failures,
test corrections and build a method you can execute again. An unsuccessful or
uncertain attempt is not by itself a reason to end the whole session. finish
ends only the current physical attempt; use reset for another while budget
remains. Never repeat a failed strategy without explaining what new evidence or
changed choice makes the next attempt informative. When feasible, vary one
relevant decision at a time to distinguish competing explanations.

Before each attempt, save practice/attemptNN/plan.json with the current hypothesis,
the intended change or repeatability check, and predicted observations that would
support or contradict it. After the attempt, save findings.txt, guidelines.json
and memory-use.json in that same directory, preserving earlier versions. Maintain
root-level versions as your current synthesis. Ground completion judgments in
observations and label uncertainty; never invent success to justify stopping.

After simulator-reported success, use a fresh reset to test whether the method works
again. Preserve which parts of the strategy are fixed and which quantities are
re-estimated. Two consecutive simulator-reported successes from fresh resets
establish a provisional working baseline; they are a transition to efficiency
practice, not by themselves a reason to end the session. Preserve the working
method and its evidence before trying alternatives. This is evidence on this
development scene, not proof of generalization or universal reliability.

Efficiency is a soft objective after a working baseline exists. Prefer methods
that retain observed task success while using fewer planning calls, fewer other
tool calls, fewer control steps, and less elapsed time. Record these quantities
separately; a longer trajectory is useful only if it improves successful execution
or avoids unnecessary replanning. Do not maximize trajectory length for its own
sake. Failed planning calls and failed physical attempts still count as costs.
The ideal with the current interface is one useful manipulation trajectory after
establishing contact; approach and gripper closure remain separate actions.

Use remaining practice attempts to test efficiency hypotheses that you choose,
such as a longer useful planned segment or fewer redundant observations. Keep
necessary feedback when it helps success. Compare candidates with the preserved
baseline; report regressions instead of replacing a working method with an
unreliable shortcut. Re-test a promising cheaper candidate after a fresh reset
before calling it the preferred method. Record baseline and candidate planning
calls, total robot-tool calls, executed trajectories, control steps and measured
elapsed time in practice/attemptNN/efficiency.json. Distinguish tool time from
whole-attempt time and mark unavailable measurements rather than inventing them.

You may stop early after the working method is repeatable and you have tested
an efficiency hypothesis, when no further informative feasible improvement is
justified. Explain the comparison and plateau; do not claim efficiency improved
merely because the task succeeded. If the hard budget prevents confirmation,
preserve the best supported method and label candidate improvements unconfirmed.
You may also stop when no informative feasible experiment remains or a resource
limit is reached; explain what prevents further learning and preserve uncertainty.
Learning an interesting fact alone need not end practice if the task is unsolved
and a useful test remains. Repeating this scene
tests repeatability, not generalization to other scenes or mechanisms.

At the end of a granted block, call finish for the active attempt, then review
the accumulated evidence. If another useful correction or repeatability check
remains, call request_attempts(learned=your findings and uncertainty,
next_experiments=a list of concrete tests). The service grants up to five more
attempts without exceeding {attempts}. Extensions may fund efficiency experiments
after success, not only failure recovery. An extension also makes sense after one
success that still needs a fresh-reset repeatability check. Use
practice_status to inspect the budget and completed-attempt success booleans.
Finish by writing practice-summary.json with every attempted episode, observations,
candidate lessons, unresolved failures, repeatability evidence, efficiency
comparisons, the best-supported method, remaining tradeoffs, stop_reason,
and unused budget. Cite observations for claimed success and testable reasons
for stopping without it; neither a budget extension nor a finish claim certifies
that the task was completed.
The official simulator success boolean is supplied after execution. The first
successful control step ends the episode immediately: no extra release, retreat
or return-home motion is needed or permitted. Remaining trajectory commands are
discarded on success. Use the returned task_success and episode_done flags as
the benchmark outcome; optional SIFT, DINO or geometric checks do not override
this signal. You may inspect the saved final image and write notes after success,
then reset for the next practice attempt. finish is unnecessary after automatic
successful termination. All other simulator internals stay private. No external API calls,
hidden code, prior coordinates or human-written task fixes are added.
'''
    first=full.index('After simulator-reported success, use a fresh reset')
    last=full.index('At the end of a granted block',first)
    stage='''Current phase: SUCCESS FIRST. If experience/successful-method.json is
provided, read that successful agent-authored method before any movement. It is
the starting point; older retrieved notes are background. Reproduce the same
qualitative procedure using freshly measured geometry and new plans. Preserve
the earlier method in your notes. Do not change it merely to reduce tool calls,
increase trajectory length or save time. If it fails, investigate reproduction
and seek task success before any optimization.

After a simulator success, reset and repeat the method. The service releases
the next phase's instructions only after two consecutive simulator-successful
episodes in this session. Read learning_phase in reset/execute replies or call
practice_status after an attempt. Follow only the current phase's objective.
Until the phase changes, do not run efficiency experiments. Record any departure
from the supplied method and why it was needed for successful execution.
You may stop if no feasible informative experiment remains or a resource limit
is reached, but explain the blocker and preserve every unsuccessful attempt.

'''
    full=full[:first]+stage+full[last:]
    full=full.replace('Extensions may fund efficiency experiments\nafter success, not only failure recovery.',
        'Extensions may fund experiments appropriate to the current learning phase.')
    return full


def relative_tool_choices(run,outcomes):
    """Extract only scalar/categorical parameters from the agent's own calls."""
    path=Path(run)/'tool-events.jsonl'
    if not path.exists():return []
    allowed={'plan_arc':{'degrees','refined','grasp_constraint'},
        'plan_grasp':{'probe_degrees','max_candidates','seed'},
        'plan_sequence':{'degrees','max_candidates','seed','grasp_constraint'},
        'execute':{'gripper','speed','hold'},
        'execute_sequence':{'close_steps','speed','hold','manipulation_gripper'},
        'gripper':{'close','steps'}}
    successes={r['attempt']:bool(r['official_task_success']) for r in outcomes}
    current=None;plans={};choices={}
    for line in path.read_text().splitlines():
        event=json.loads(line);request=event['request'];response=event['response']
        op=request['operation'];args=request.get('arguments',{});result=response.get('result',{})
        if op=='reset' and response['ok']:
            current=result.get('attempt');plans={}
        if current is None:continue
        if response['ok'] and op.startswith('plan_') and result.get('plan_id'):
            plans[result['plan_id']]=result.get('kind')
        if op not in allowed:continue
        def safe(k,v):
            return (v is None or (isinstance(v,(int,float,bool)) and -float('inf')<v<float('inf'))
                    or (k=='grasp_constraint' and isinstance(v,str) and v in {'point','rigid','jaw_axis'}))
        entry={'operation':op,'provided_parameters':{k:v for k,v in args.items()
            if k in allowed[op] and safe(k,v)}}
        if op=='execute':entry['plan_kind']=plans.get(args.get('plan_id'),'unknown')
        key=json.dumps(entry,sort_keys=True)
        record=choices.setdefault(key,{**entry,'attempts':[],'rpc_ok_count':0,'rpc_error_count':0})
        if current not in record['attempts']:record['attempts'].append(current)
        record['rpc_ok_count' if response['ok'] else 'rpc_error_count']+=1
    return [{**r,'episode_outcomes':{str(n):successes.get(n) for n in r['attempts']},
        'interpretation':'Observed choices, not isolated causal effects or universal prescriptions. Omitted parameters used the documented defaults.'} for r in choices.values()]


def successful_method_payload(run):
    """Copy agent-authored principles, omitting links to historical raw evidence."""
    run=Path(run).resolve()
    if not run.is_relative_to(ROOT/'results'):raise ValueError('Use a saved project result')
    outcomes=read(run/'evaluation-private.json')['attempts']
    successes=[r['attempt'] for r in outcomes if r['official_task_success'] and r['final_task_success']]
    if len(successes)<2:raise ValueError('The supplied method needs at least two recorded successes')
    source=run/'agent-workspace/guidelines.json'
    original=read(source)
    if not original.get('principles'):raise ValueError('No agent-authored principles found')
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k!='references'}
        if isinstance(value,list):
            if value and all(isinstance(x,(int,float)) for x in value):raise ValueError('Historical numeric arrays are not allowed')
            return [clean(v) for v in value]
        if isinstance(value,str):
            if re.search(r'\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d',value):raise ValueError('Historical coordinate array in method')
            if re.search(r'(?:robot_data/|replies/|requests/|/home/|/tmp/|\.np[yz])',value):raise ValueError('Raw historical evidence link in method')
        return value
    conclusion=read(run/'agent-workspace/practice-summary.json',{})
    return {'origin':'Prior acting agent, copied without new robotics strategies',
            'source_guidelines_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'observed_successful_attempts':successes,
            'observed_attempt_outcomes':[{'attempt':row['attempt'],'task_success':row['official_task_success'],
                'final_task_success':row['final_task_success']} for row in outcomes],
            'agent_conclusion':clean(conclusion.get('best_supported_method')),
            'agent_remaining_tradeoffs':clean(conclusion.get('remaining_tradeoffs')),
            'tested_relative_tool_choices':relative_tool_choices(run,outcomes),
            'scope':'Repeated success in the same development scene; not transfer evidence.',
            'representation':'Historical evidence links removed. Tested relative control choices retained. No historical image regions, world poses, trajectories or raw replies supplied.',
            'method':clean(original)}


def load_successful_method(profile, baseline_run=None):
    if baseline_run:return successful_method_payload(ROOT/baseline_run),baseline_run
    if not profile.get('successful_method'):return None,None
    path=ROOT/profile['successful_method']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=profile['successful_method_sha256']:
        raise RuntimeError('Registered successful method changed; do not silently fall back to older memory')
    return read(path),profile.get('successful_method_source_run',profile['successful_method'])


def build_command(args, prompt_path):
    initial = getattr(args, 'initial_attempts', min(5, args.attempts))
    validate_attempt_budget(args.attempts, learning_practice=True, initial_attempts=initial)
    if not 0 <= args.state < 50 or args.agent_seconds <= 0:
        raise ValueError('Use state 0..49 and a positive session time budget')
    command, output = registered_command(args)
    command[command.index('--attempts')+1] = str(args.attempts)
    command[command.index('--prompt-file')+1] = str(prompt_path)
    command.append('--learning-practice')
    if '--stop-on-success' not in command:command.append('--stop-on-success')
    command.extend(['--initial-attempts', str(initial)])
    if getattr(args,'successful_method_file',None):
        command.extend(['--successful-method-file',str(args.successful_method_file)])
    return command, output


def efficiency_metrics(events):
    """Measured tool costs; reasoning/waiting time is not inferred from RPC time."""
    rows={};current=None
    for event in events:
        operation=event['request']['operation'];response=event['response']
        result=response.get('result',{})
        if operation=='reset' and response['ok']:
            current=result.get('attempt')
        if current is None:continue
        row=rows.setdefault(current,{'attempt':current,'robot_tool_calls':0,
            'planning_calls':0,'failed_planning_calls':0,'executed_trajectories':0,
            'tool_seconds':0.,'attempt_wall_seconds':None})
        row['robot_tool_calls']+=1
        row['tool_seconds']+=event['wall_seconds']
        if operation.startswith('plan_'):
            row['planning_calls']+=1
            row['failed_planning_calls']+=int(not response['ok'] or result.get('planned') is False)
        if operation=='execute' and response['ok']:
            row['executed_trajectories']+=1
        if operation=='execute_sequence' and response['ok']:
            row['executed_trajectories']+=sum(s['stage']!='closure' for s in result.get('stages',[]))
    return list(rows.values())


def summarize(output, attempts, returncode, seconds):
    episodes = read(output/'evaluation-private.json', {'attempts': []})['attempts']
    by_id = {row['attempt']: row for row in episodes}
    slots = [{'attempt': i, 'status': 'attempted' if i in by_id else 'not_attempted',
              'outcome': by_id.get(i)} for i in range(1, attempts+1)]
    reached = [row['attempt'] for row in episodes if row['official_task_success']]
    retained = [row['attempt'] for row in episodes if row['final_task_success']]
    events_path=output/'tool-events.jsonl'
    metrics=efficiency_metrics([json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else [])
    for row in metrics:
        episode=by_id.get(row['attempt'],{})
        row.update(native_success=episode.get('official_task_success'),control_steps=episode.get('control_steps'),
                   attempt_wall_seconds=episode.get('attempt_wall_seconds'))
    return {'mode': 'development_practice', 'same_agent_session': True,
            'adaptive_budget': read(output/'practice-budget.json'),
            'agent_learning_summary': read(output/'agent-workspace/practice-summary.json'),
            'efficiency_metrics':metrics,
            'efficiency_metric_scope':'Tool calls include recorded RPCs and terminal bookkeeping; tool seconds exclude model reasoning and between-call time.',
            'maximum_attempts': attempts, 'attempts_used': len(episodes),
            'unused_attempt_budget': attempts-len(episodes), 'attempts': slots,
            'native_goal_reached_attempts': reached, 'native_goal_retained_attempts': retained,
            'first_native_success_attempt': min(reached) if reached else None,
            'execution_veto_checks': sum(row['execution_veto_checks'] for row in episodes),
            'all_issued_trajectories_completed': all(row['all_issued_trajectories_completed'] for row in episodes),
            'all_trajectories_completed_or_success_terminated':all(row.get('all_trajectories_completed_or_success_terminated',row['all_issued_trajectories_completed']) for row in episodes),
            'control_steps_after_first_success':sum(row.get('control_steps_after_first_success',0) for row in episodes),
            'returncode': returncode, 'session_wall_seconds': seconds,
            'held_out_evaluation': False, 'generalization_measured': False,
            'automatic_memory_promotion': False,
            'native_outcomes_sent_to_agent': True,
            'feedback_scope':'success boolean only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--state', type=int, default=47)
    parser.add_argument('--scene-variant', default='nominal')
    parser.add_argument('--attempts', '--max-attempts', dest='attempts', type=int, default=20)
    parser.add_argument('--initial-attempts', type=int, default=5)
    parser.add_argument('--agent-seconds', type=int, help='Total session limit; default 1800 seconds per allowed attempt')
    parser.add_argument('--memory-profile', default='configs/subscription_memory.json')
    parser.add_argument('--baseline-run',help='Prior successful agent methodology to reproduce before further learning')
    parser.add_argument('--study-note-file',help='Additional recorded experiment question; does not change phase gating')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.agent_seconds is None:
        args.agent_seconds = args.attempts*1800
    profile = read(ROOT/args.memory_profile)
    baseline,baseline_source=load_successful_method(profile,args.baseline_run)
    prompt = practice_prompt((ROOT/profile['prompt']).read_text(), args.attempts, args.initial_attempts)
    if args.study_note_file:
        prompt+='\nCurrent experiment note:\n'+Path(args.study_note_file).read_text()+'\n'
    if args.dry_run and baseline is not None:
        args.successful_method_file=(ROOT/profile['successful_method']) if not args.baseline_run else '<generated-successful-method>'
    command, output = build_command(args, '<generated-practice-prompt>')
    if args.dry_run:
        print(json.dumps({'command': command, 'maximum_attempts': args.attempts,
                          'initial_attempts': args.initial_attempts,
                          'completion_policy': 'advisory',
                          'successful_method_supplied':baseline is not None,
                          'same_agent_session': True, 'prompt': prompt}, indent=2))
        return
    if output.exists():
        raise ValueError('Use a new output directory')
    original_memory = store_hashes(ROOT/profile['public_library'])
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='vlm-practice-prompt-') as temporary:
        prompt_path = Path(temporary)/'prompt.txt'
        prompt_path.write_text(prompt)
        if baseline is not None:
            args.successful_method_file=Path(temporary)/'successful-method.json'
            args.successful_method_file.write_text(json.dumps(baseline,indent=2)+'\n')
        command, output = build_command(args, prompt_path)
        run = subprocess.run(command, cwd=ROOT)
    if not output.exists():
        raise RuntimeError('Practice failed before startup; no physical attempts recorded')
    summary = summarize(output, args.attempts, run.returncode, time.perf_counter()-started)
    summary['original_library_unchanged'] = original_memory == store_hashes(ROOT/profile['public_library'])
    summary['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
    summary['source_profile'] = profile
    summary['successful_method_source']=baseline_source
    summary['successful_method_sha256']=hashlib.sha256(json.dumps(baseline,indent=2).encode()+b'\n').hexdigest() if baseline else None
    if (output/'agent-workspace').exists():
        summary['retention'] = retained_inputs(output)
    summary['lossless_compression'] = compress_observations(output)
    write(output/'learning-summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['attempts_used', 'maximum_attempts',
        'native_goal_reached_attempts', 'native_goal_retained_attempts',
        'original_library_unchanged', 'returncode']}, indent=2))
    if run.returncode or not summary['original_library_unchanged']:
        raise SystemExit(run.returncode or 1)


if __name__ == '__main__':
    main()
