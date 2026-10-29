"""Reproduce the isolated Luna/ChatGPT subscription pilot, without model API keys."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.subscription_agent import clean_environment, codex_arguments, verify_isolation, validate_attempt_budget
from vlm_tamp.subscription_tasks import DEFAULT_TASK, TASKS, task_spec

PROMPT='''Run the autonomous robotics pilot described in tools.txt. Read the tool
documentation and the one approved example in example/. Open the microwave
through physical simulator interaction, investigate failures yourself, and save
evidence-backed findings and transferable guidelines. Begin with reset and
inspect the current images. Write/run analysis code whenever useful. You choose
regions, grasps, probes, model updates and replanning. There are at most two
physical attempts on the provided case, 600 control steps each. All reasoning
is through this Codex session, with no external model API calls. Work only in
this public directory. Do not obtain hidden implementation, old project results,
simulator ground truth or other agent history. Your only recovery guidance is
the explicit example. Do not ask the human to debug the task. Write findings.txt
and guidelines.json and call finish. Distinguish supplied ideas from discoveries.
'''


def ledger_hashes():
    files=[ROOT/'results/articulated-api-v1/ledger.json',*(ROOT/'results/api-runs').glob('*/ledger.json')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--model',choices=['gpt-6-luna','gpt-6-astra'],default='gpt-6-luna')
    p.add_argument('--reasoning',choices=['low','medium','high','xhigh','max'],default='high')
    p.add_argument('--agent-seconds',type=int,default=1800)
    p.add_argument('--control-steps',type=int,default=1200)
    p.add_argument('--reference-mode',choices=['measured_delta','slew_limited'],default='slew_limited')
    p.add_argument('--episode-seconds',type=int,default=1800,help='Per-attempt wall budget during persistent practice')
    p.add_argument('--prompt-file',help='Optional exact recorded initial prompt for a fresh reproduction')
    p.add_argument('--fit-diagnostics',action='store_true',help='Expose sensor-only fit quality separately from initial-model agreement')
    p.add_argument('--state',type=int,default=21);p.add_argument('--dino-tools',action='store_true')
    p.add_argument('--task',choices=sorted(TASKS),default=DEFAULT_TASK)
    p.add_argument('--landmark-evidence',action='store_true',help='Expose measured landmark displacement summaries; does not change pose fitting')
    p.add_argument('--multiview-grounding',action='store_true',help='Enable caller-selected calibrated multiview hinge/handle checking')
    p.add_argument('--part-motion-tools',action='store_true',help='Enable part masks, calibrated associations and independently initialized motion fits')
    p.add_argument('--guidelines-source');p.add_argument('--evaluation-case',action='store_true')
    p.add_argument('--scene-variant',default='nominal')
    p.add_argument('--attempts',type=int,choices=range(1,21),default=2)
    p.add_argument('--learning-practice',action='store_true')
    p.add_argument('--practice-cases',help='Private JSON curriculum; only reset instructions and observations are shown to the learner')
    p.add_argument('--initial-attempts',type=int)
    p.add_argument('--stop-on-success',action='store_true')
    p.add_argument('--persistent-episode',action='store_true')
    p.add_argument('--surface-contact-tools',action='store_true')
    p.add_argument('--checkpoint-source',help='Private recorded physical prefix for a declared checkpoint experiment; no historical annotations supplied to actor')
    p.add_argument('--checkpoint-controls',type=int,default=459)
    p.add_argument('--compact-interface',action='store_true')
    p.add_argument('--visual-inspection',action='store_true',help='Enable local public-observation comparison helper')
    p.add_argument('--numerical-backend',choices=['legacy','curobo_v2'],default='curobo_v2')
    p.add_argument('--successful-method-file')
    p.add_argument('--memory-store',help='Read-only public episodic library; no implementation or evaluation files')
    p.add_argument('--completion-checks',action='store_true')
    p.add_argument('--abstract-experience',action='store_true')
    p.add_argument('--no-example',action='store_true',help='Do not supply a worked example to the actor')
    a=p.parse_args();out=(ROOT/a.output).resolve()
    if a.practice_cases:
        if not a.learning_practice or a.evaluation_case or a.checkpoint_source:raise ValueError('Curriculum is practice-only and cannot be combined with a checkpoint')
        from serve_practice_curriculum import checked_cases
        cases=checked_cases(a.practice_cases)
        a.task=cases[0]['task'];a.state=int(cases[0]['state_index'])
    if a.agent_seconds<=0:raise ValueError('Use a positive episode time budget')
    if a.episode_seconds<=0:raise ValueError('Use a positive per-attempt time budget')
    if a.persistent_episode and (not a.stop_on_success or (a.attempts!=1 and not a.learning_practice)):
        raise ValueError('Persistence requires native-success termination and either one attempt or declared practice')
    validate_attempt_budget(a.attempts,learning_practice=a.learning_practice,evaluation_case=a.evaluation_case,initial_attempts=a.initial_attempts)
    if a.learning_practice and not a.prompt_file:
        raise ValueError('Learning practice requires an explicit practice prompt')
    prompt=Path(a.prompt_file).read_text() if a.prompt_file else PROMPT.replace('Open the microwave',task_spec(a.task)['instruction'].capitalize())
    if a.no_example and not a.prompt_file:
        prompt=prompt.replace('documentation and the one approved example in example/.','documentation.')
        prompt=prompt.replace('Your only recovery guidance is\nthe explicit example.','Recovery decisions belong to you.')
    if not 0<=a.state<50:raise ValueError('Official initial-state index must be 0..49')
    if not out.is_relative_to(ROOT/'results') or out==ROOT/'results':
        raise ValueError('Use a new results/<run> directory')
    from vlm_tamp.storage_budget import require_storage_headroom
    storage_check=require_storage_headroom(ROOT)
    status=subprocess.run(['codex','login','status'],capture_output=True,text=True,env=clean_environment(),check=True)
    if 'Logged in using ChatGPT' not in status.stdout+status.stderr:
        raise RuntimeError('ChatGPT login is required; API-key fallback is not permitted')
    out.mkdir(exist_ok=False)
    (out/'storage-preflight.json').write_text(json.dumps(storage_check,indent=2)+'\n')
    work=Path(tempfile.mkdtemp(prefix='vlm-luna-workspace-'))
    env=clean_environment()
    subprocess.run(['/usr/bin/python3',str(ROOT/'scripts/prepare_subscription_pilot.py'),
        '--workspace',str(work),'--output',str(out),'--state',str(a.state),'--task',a.task,
        '--control-steps',str(a.control_steps),'--reference-mode',a.reference_mode,
        *(['--landmark-evidence'] if a.landmark_evidence else []),
        *(['--multiview-grounding'] if a.multiview_grounding else []),
        *(['--part-motion-tools'] if a.part_motion_tools else []),
        *(['--dino-tools'] if a.dino_tools else []),
        *(['--completion-checks'] if a.completion_checks else []),
        *(['--abstract-experience'] if a.abstract_experience else []),
        *(['--no-example'] if a.no_example else []),
        *(['--learning-practice'] if a.learning_practice else []),
        *(['--practice-cases',str(Path(a.practice_cases).resolve())] if a.practice_cases else []),
        *(['--initial-attempts',str(a.initial_attempts)] if a.initial_attempts is not None else []),
        *(['--stop-on-success'] if a.stop_on_success else []),
        *(['--persistent-episode'] if a.persistent_episode else []),
        *(['--surface-contact-tools'] if a.surface_contact_tools else []),
        *(['--compact-interface'] if a.compact_interface else []),
        *(['--visual-inspection'] if a.visual_inspection else []),
        *(['--successful-method-file',str(Path(a.successful_method_file).resolve())] if a.successful_method_file else []),
        *(['--guidelines-source',str(Path(a.guidelines_source).resolve())] if a.guidelines_source else []),
        *(['--evaluation-case'] if a.evaluation_case else []),'--attempts',str(a.attempts),
        *(['--memory-store',str(Path(a.memory_store).resolve())] if a.memory_store else [])],check=True,cwd=ROOT,env=env)
    protocol=json.loads((out/'protocol.json').read_text())
    protocol['model']=a.model;protocol['reasoning_effort']=a.reasoning
    protocol['unconditioned_hinge_fit_diagnostics']=bool(a.fit_diagnostics)
    protocol['scene_variant']=a.scene_variant
    protocol['scene_variant_parameters_exposed_to_agent']=False
    protocol['episode_wall_budget_seconds']=a.episode_seconds if a.learning_practice else a.agent_seconds
    protocol['session_wall_budget_seconds']=a.agent_seconds
    protocol['persistent_episode']=a.persistent_episode
    protocol['surface_contact_tools']=a.surface_contact_tools
    protocol['controller_reference_mode']=a.reference_mode
    if a.checkpoint_source:
        protocol['physical_prefix_initialization']={'source':str(Path(a.checkpoint_source).resolve()),
            'control_step':a.checkpoint_controls,'historical_annotations_supplied':False}
    protocol['landmark_evidence']=bool(a.landmark_evidence)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    forbidden=[ROOT/'src/vlm_tamp/rigid_hinge_tracking.py',ROOT/'docs/state11_recovery.md',
               ROOT/'results/state11-observed-contact-recovery-20261015/summary.json',
               Path('/mnt/c/Users/duo/.codex/memories/MEMORY.md'),
               Path('/home/duo/.codex/keys/vlm-tamp.env')]
    if a.abstract_experience:
        forbidden.append(ROOT/'data/agent_memory/luna-microwave-v1/library/episode-0001/procedure.json')
    readonly=[work/'episodic_library'] if a.memory_store else []
    if a.successful_method_file:readonly.append(work/'experience/successful-method.json')
    if a.guidelines_source:readonly.append(work/'experience/guidelines.json')
    if a.visual_inspection:readonly.extend([work/'visual.py',work/'tools/visual_inspection.txt'])
    (out/'isolation.json').write_text(json.dumps(verify_isolation(work,forbidden,readonly_paths=readonly),indent=2))
    if a.memory_store:
        from vlm_tamp.episodic_memory import verify_memory_access
        (out/'memory-access-check.json').write_text(json.dumps(verify_memory_access(work),indent=2))
    (out/'api-ledger-checkpoint.json').write_text(json.dumps(ledger_hashes(),indent=2))
    version=subprocess.check_output(['codex','--version'],text=True).strip()
    (out/'authentication-check.json').write_text(json.dumps({'logged_in_using_chatgpt':True,
        'api_key_authentication':False,'client_version':version,'requested_model':a.model,
        'requested_reasoning':a.reasoning},indent=2))
    hashes={}
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+list((ROOT/'src/vlm_tamp/native').glob('*.cc'))+list((ROOT/'scripts').glob('*.py'))+[ROOT/'configs/articulated_libero.json']
    if a.part_motion_tools:files.append(ROOT/'prompts/part_motion_introduction.txt')
    if a.surface_contact_tools:files.append(ROOT/'prompts/contact_skills_introduction.txt')
    for path in files:
        name=path.relative_to(ROOT);hashes[str(name)]=hashlib.sha256(path.read_bytes()).hexdigest()
        dest=out/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(path,dest)
    (out/'source-hashes.json').write_text(json.dumps(hashes,indent=2))
    prompt=prompt.replace('600 control',f'{a.control_steps} control').replace('600-step',f'{a.control_steps}-step')
    backend=None;agent=None
    try:
        with (out/'service.log').open('w') as service_log:
            backend=subprocess.Popen([str(ROOT/'.runtime/envs/rekep-libero/bin/python'),
                str(ROOT/'scripts/serve_practice_curriculum.py' if a.practice_cases else ROOT/'scripts/serve_contact_checkpoint.py' if a.checkpoint_source else ROOT/'scripts/serve_subscription_robot.py'),
                *(['--curriculum-file',str(Path(a.practice_cases).resolve())] if a.practice_cases else []),
                *(['--prefix-source',str(Path(a.checkpoint_source).resolve()),'--prefix-controls',str(a.checkpoint_controls)] if a.checkpoint_source else []),
                '--output',str(out),
                '--workspace',str(work),'--state',str(a.state),'--task',a.task,'--attempts',str(a.attempts),'--scene-variant',a.scene_variant,
                '--control-steps',str(a.control_steps),'--reference-mode',a.reference_mode,
                *(['--landmark-evidence'] if a.landmark_evidence else []),
                *(['--multiview-grounding'] if a.multiview_grounding else []),
                *(['--part-motion-tools'] if a.part_motion_tools else []),
                '--numerical-backend',a.numerical_backend,
                *(['--fit-diagnostics'] if a.fit_diagnostics else []),
                *(['--completion-checks'] if a.completion_checks else []),
                *(['--learning-practice'] if a.learning_practice else []),
                *(['--initial-attempts',str(a.initial_attempts)] if a.initial_attempts is not None else []),
                *(['--stop-on-success'] if a.stop_on_success else []),
                *(['--persistent-episode'] if a.persistent_episode else []),
                *(['--surface-contact-tools'] if a.surface_contact_tools else []),
                *(['--dino-tools'] if a.dino_tools else [])],
                cwd=ROOT,env=env,stdout=service_log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+60
            while not (work/'requests').is_dir():
                if backend.poll() is not None:raise RuntimeError('Numerical service failed; inspect service.log')
                if time.monotonic()>deadline:raise TimeoutError('Numerical service startup timed out')
                time.sleep(.1)
            if a.persistent_episode:
                from vlm_tamp.episode_persistence import run_persistent_agent
                code=run_persistent_agent(work,out,env,readonly,prompt,a.agent_seconds,
                    learning_practice=a.learning_practice,episode_seconds=a.episode_seconds,
                    model=a.model,reasoning=a.reasoning)
            else:
                args=codex_arguments(work,out/'agent-last.txt',model=a.model,reasoning=a.reasoning,readonly_paths=readonly)
                (out/'agent-command.json').write_text(json.dumps(args,indent=2))
                (out/'agent-prompt.txt').write_text(prompt)
                started=time.time()
                with (out/'agent-events.jsonl').open('w') as log,(out/'agent-stderr.log').open('w') as err:
                    agent=subprocess.Popen(args,stdin=subprocess.PIPE,text=True,env=env,stdout=log,stderr=err)
                    agent.stdin.write(prompt);agent.stdin.close()
                    (out/'agent-process.json').write_text(json.dumps({'pid':agent.pid,'started':started}))
                    try:code=agent.wait(timeout=a.agent_seconds)
                    except subprocess.TimeoutExpired:
                        agent.terminate();code=agent.wait(timeout=30)
                        (out/'agent-timeout.json').write_text(json.dumps({'timeout_seconds':a.agent_seconds}))
                (out/'agent-exit.json').write_text(json.dumps({'returncode':code,'wall_seconds':time.time()-started}))
    finally:
        if agent is not None and agent.poll() is None:
            agent.terminate();agent.wait(timeout=30)
        if backend is not None and backend.poll() is None:
            (out/'stop-service').touch()
            # The service completes an already issued physical trajectory before
            # handling this shutdown flag; there is no mid-trajectory veto.
            backend.wait(timeout=900)
    if (out/'agent-exit.json').exists():
        subprocess.run([str(ROOT/'.runtime/envs/rekep-libero/bin/python'),str(ROOT/'scripts/report_subscription_pilot.py'),
            '--run',str(out),'--workspace',str(work)],cwd=ROOT,env=env,check=True)
        if (out/'agent-workspace').exists():
            shutil.rmtree(work)
    print(json.dumps({'run':str(out),'workspace':str(work),'completed':(out/'audit.json').exists()}))
    if code:raise SystemExit(128-code if code<0 else code)


if __name__=='__main__':main()
