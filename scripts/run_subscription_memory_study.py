"""One-attempt paired experiment: concrete episodic memory versus the same empty library."""
import argparse
import copy
import html
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from vlm_tamp.episodic_memory import store_hashes
from run_subscription_dino_study import read,write,sha,collect
from replay_subscription_episode import compress_observations

PROMPT='''Open the microwave through physical simulator interaction using tools.txt.
Read the documented tools, the approved example, and experience/guidelines.json.
There is exactly ONE physical attempt with 600 control steps. Begin with reset,
inspect the current images, and call memory.recall with the task instruction and
robot Panda BEFORE physical movement. The library may be empty or may contain
recorded experiences. If it returns an episode, inspect its concrete procedure
and relevant evidence. Reuse decisions that apply to the current observation;
check the scene rather than assuming old coordinates always apply. Obtain fresh
plans through robot.py. You may write and run code, use optional DINO, investigate
planning failures, and adapt between completed trajectories within this one
attempt. Do not reset for a second attempt. Execute complete issued trajectories.
Determine completion from observations; native evaluation is private.
Write findings.txt, guidelines.json, and memory-use.json documenting retrieved
ids, decisions reused/changed and evidence for applicability. Call finish.
All reasoning uses this Luna/high Codex session and ChatGPT subscription. No
external model API calls, hidden implementation, evaluator state, other project
results or agent history. The only prior experience available is explicitly
supplied in the example, fixed guide and read-only episodic library. Do not ask
the human to debug or supply strategies. Distinguish inherited and learned ideas.
'''


def freeze(study):
    if (study/'protocol.json').exists():raise ValueError('Study already frozen')
    author=study/'memory-author-audited';assert read(author/'completion.json')['completed']
    assert read(study/'exact-replay/summary.json')['completed']==5
    library=author/'library';(study/'empty-library').mkdir(exist_ok=True)
    prior=ROOT/'results/subscription-strategy-transfer-20261019'
    shutil.copyfile(prior/'original-guidelines.json',study/'guidelines.json')
    (study/'prompt.txt').write_text(PROMPT)
    selected=read(study/'state-reservation.json')['selected_states'][:3]
    scenarios=[{'scenario_id':f'same-{i:02d}','group':'same_case','state':37,'scene_variant':'nominal'} for i in range(1,6)]
    scenarios +=[{'scenario_id':f'transfer-{state:03d}-{variant}','group':'transfer','state':state,'scene_variant':variant}
        for state,variant in zip(selected,['nominal','translated','yaw_positive'])]
    cases=[]
    for i,s in enumerate(scenarios):
        for condition in (['without_memory','with_memory'] if i%2==0 else ['with_memory','without_memory']):
            cases.append({**s,'condition':condition,'case_id':s['scenario_id']+'-'+condition})
    write(study/'manifest.json',{'task':'KITCHEN_SCENE7_open_the_microwave','scenarios':scenarios,'cases':cases,
        'planned_cases':len(cases),'one_physical_attempt_each':True,'all_planned_cases_in_denominator':True})
    inherited=read(prior/'protocol.json')
    protocol={k:copy.deepcopy(inherited[k]) for k in ['model','reasoning_effort','authentication','dino',
        'dino_runtime','runtime_artifacts_sha256','simulator_seed','settling_actions','controller']}
    protocol.update(task='KITCHEN_SCENE7_open_the_microwave',maximum_attempts_per_case=1,
        maximum_control_steps_per_attempt=600,agent_seconds_per_case=1800,
        source_experience=str(prior/'runs/nominal-state037-original'),training_state=37,
        transfer_states=selected,primary_metric='native predicate true at the end of the one physical attempt',
        secondary_metric='native predicate true at any step of the one physical attempt',
        prompt_sha256=sha(study/'prompt.txt'),guidelines_sha256=sha(study/'guidelines.json'),
        library_hashes=store_hashes(library),empty_library_hashes={},
        separate_api_calls_allowed=False,execution_vetoes=0,
        memory_policy='Concrete public experience and agent-authored procedure are persisted once and retrieved in fresh sessions; freeze for matched evaluation, no cross-evaluation learning.',
        planning_grasp_ik_control_implementations_unchanged=True,
        physical_replay='Five identical-state seed-0 command replays; repeatability control, not learned generalization',
        limitations='One task and asset; five stochastic agent repetitions on the training state and three transfer scenes, one session per condition there. Inherited tool priors remain. No guarantees or cross-task claims.',
        infrastructure_policy='Record all failed launches in denominator; stop on subscription quota or capacity error, no automatic retry or model/API fallback.')
    old=read(prior/'frozen-source-hashes.json')
    allowed={'scripts/run_subscription_agent_pilot.py','scripts/prepare_subscription_pilot.py','src/vlm_tamp/subscription_agent.py','scripts/report_subscription_pilot.py'}
    drift=[f for f,h in old.items() if (ROOT/f).exists() and sha(ROOT/f)!=h]
    if set(drift)-allowed:raise RuntimeError('Unexpected numerical source change: '+str(set(drift)-allowed))
    protocol['prior_source_changes']=drift
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+list((ROOT/'src/vlm_tamp/native').glob('*.cc'))+list((ROOT/'scripts').glob('*.py'))+[ROOT/'configs/articulated_libero.json']
    write(study/'frozen-source-hashes.json',{str(f.relative_to(ROOT)):sha(f) for f in files})
    write(study/'protocol.json',protocol)


def report(study,manifest,records):
    scores={}
    for group in ['same_case','transfer']:
        scores[group]={}
        for condition in ['without_memory','with_memory']:
            cases=[c for c in manifest['cases'] if c['group']==group and c['condition']==condition]
            done=[records[c['case_id']] for c in cases if records.get(c['case_id'],{}).get('status') in {'completed','runner_failure'}]
            scores[group][condition]={'planned':len(cases),'completed':len(done),
                'final_success':sum(x.get('final_task_success',False) for x in done),
                'ever_success':sum(x.get('first_attempt_success',False) for x in done),
                'retrieved_experience':sum(x.get('retrieved_entries',0)>0 for x in done)}
    pairs={}
    for s in manifest['scenarios']:
        a=records.get(s['scenario_id']+'-without_memory',{}).get('initial_arrays',{})
        b=records.get(s['scenario_id']+'-with_memory',{}).get('initial_arrays',{})
        if a and b:pairs[s['scenario_id']]={'same_keys':a.keys()==b.keys(),'equal_arrays':a==b}
    replay=read(study/'exact-replay/summary.json',{})
    summary={'scores':scores,'records':records,'initial_sensor_pairs':pairs,'replay':replay,
        'completed_cases':sum(r.get('status') in {'completed','runner_failure'} for r in records.values()),
        'planned_cases':len(manifest['cases']),'separately_billed_api_calls':sum(r.get('api_calls',0) for r in records.values())}
    write(study/'summary.json',summary)
    score_rows=''.join(f'<tr><td>{g}</td><td>{c}</td><td>{s["completed"]}/{s["planned"]}</td><td>{s["final_success"]}/{s["planned"]}</td><td>{s["ever_success"]}/{s["planned"]}</td></tr>' for g,cs in scores.items() for c,s in cs.items())
    rows=[]
    for c in manifest['cases']:
        r=records.get(c['case_id'],{});name=c['case_id'];link=f'runs/{name}/report/index.html'
        label=f'<a href="{link}">{name}</a>' if (study/link).exists() else name
        rows.append(f'<tr><td>{label}</td><td>{r.get("status","planned")}</td><td>{r.get("final_task_success","—")}</td><td>{r.get("first_attempt_success","—")}</td><td>{r.get("retrieved_entries","—")}</td><td>{r.get("case_wall_seconds",0)/60:.1f}</td></tr>')
    procedure=read(study/'memory-author-audited/library/episode-0001/procedure.json',{})
    (study/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Can Luna remember a successful correction?</title>
<style>body{{font:17px system-ui;max-width:1250px;margin:30px auto;padding:20px;background:#f5f8fb;color:#183344}}article{{background:white;padding:20px;border:1px solid #ccd6df;border-radius:12px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #d1dae2}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}video{{width:100%;max-height:550px}}</style>
<h1>Can Luna remember a successful correction?</h1><p>{summary['completed_cases']}/{summary['planned_cases']} one-attempt sessions complete. Pending cases have no outcome yet.</p>
<article><h2>Physical repeatability control</h2><p>Exact replay: {replay.get('final_successes',0)}/{replay.get('planned',5)} ended successful. Commands, gripper settings, initialization and simulator seed reproduced from one successful retry. This tests physical repeatability, not an agent's ability to reuse or generalize a procedure.</p><video controls preload="metadata" src="exact-replay/replay01/initial/rollout.mp4"></video></article>
<article><h2>Memory comparison</h2><p>Same Luna/high subscription model, approved example, fixed guide, optional DINO and numerical tools. A fresh session gets a generic memory retrieval interface backed by either an empty library or the concrete past failed/successful experience plus Luna's own procedure. One physical attempt, 600 steps, no reset retry, full issued trajectories execute. Library frozen during comparison; new evaluation experience does not leak to later cases.</p><table><tr><th>Cases</th><th>Memory</th><th>Finished</th><th>Success at end</th><th>Success at any step</th></tr>{score_rows}</table><p>Five sessions per condition on training state 37; three matched transfer scenes from reserved states. Native task scores stay outside the agent.</p></article>
<article><h2>Runs and videos</h2><table><tr><th>Case</th><th>Status</th><th>At end</th><th>Any step</th><th>Retrieved entries</th><th>Minutes</th></tr>{''.join(rows)}</table></article>
<article><h2>Luna-authored remembered procedure</h2><pre>{html.escape(json.dumps(procedure,indent=2))}</pre></article>
<article><h2>Interpretation limits</h2><p>Inherited geometric, grasp and contact priors remain in the numerical tools. This study isolates concrete experience retrieval, not autonomous invention of the whole manipulation pipeline. Same-case repetition is not novel-scene transfer; three transfer scenes cannot establish universal reliability. Exact replay and agent planning are different conditions.</p></article>
<p><a href="protocol.json">Frozen protocol</a> · <a href="summary.json">Results and audits</a> · <a href="memory-author-audited/library/episode-0001/procedure.json">Persistent memory</a></p>''')


def initial_arrays(folder):
    import hashlib
    candidates=sorted((folder/'agent-workspace/robot_data/attempt01').glob('obs*/sensors.npz'))
    if not candidates:return {}
    with np.load(candidates[0]) as z:
        return {k:hashlib.sha256(str((z[k].dtype.str,z[k].shape)).encode()+z[k].tobytes()).hexdigest() for k in sorted(z.files)}


def run(study):
    manifest=read(study/'manifest.json');protocol=read(study/'protocol.json');records=read(study/'progress.json',{})
    (study/'runs').mkdir(exist_ok=True);report(study,manifest,records)
    write(study/'batch-process.json',{'pid':__import__('os').getpid(),'started_unix':time.time()})
    for c in manifest['cases']:
        if records.get(c['case_id'],{}).get('status') in {'completed','runner_failure'}:continue
        if (study/'stop-study').exists():break
        drift=[f for f,h in read(study/'frozen-source-hashes.json').items() if not (ROOT/f).exists() or sha(ROOT/f)!=h]
        if drift:raise RuntimeError('Frozen source drift: '+str(drift[:8]))
        if store_hashes(study/'memory-author-audited/library')!=protocol['library_hashes']:raise RuntimeError('Memory library drift')
        if store_hashes(study/'empty-library')!={}:raise RuntimeError('Control library is no longer empty')
        assert sha(study/'prompt.txt')==protocol['prompt_sha256'] and sha(study/'guidelines.json')==protocol['guidelines_sha256']
        assert all(sha(Path(f))==h for f,h in protocol['runtime_artifacts_sha256'].items())
        assert sha(Path(protocol['dino']['checkpoint']))==protocol['dino']['sha256']
        folder=study/'runs'/c['case_id']
        if folder.exists():raise RuntimeError('Interrupted case exists; preserve and audit before resuming')
        memory=study/('memory-author-audited/library' if c['condition']=='with_memory' else 'empty-library')
        records[c['case_id']]={**c,'status':'running'};write(study/'progress.json',records);report(study,manifest,records)
        command=['/usr/bin/python3',str(ROOT/'scripts/run_subscription_agent_pilot.py'),'--output',str(folder),
            '--state',str(c['state']),'--scene-variant',c['scene_variant'],'--prompt-file',str(study/'prompt.txt'),
            '--guidelines-source',str(study/'guidelines.json'),'--memory-store',str(memory),'--attempts','1',
            '--fit-diagnostics','--evaluation-case','--dino-tools','--agent-seconds',str(protocol['agent_seconds_per_case'])]
        start=time.perf_counter();print(json.dumps({'started':c}),flush=True)
        with (study/(c['case_id']+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:code=process.wait()
            except KeyboardInterrupt:process.send_signal(signal.SIGINT);code=process.wait(timeout=950);(study/'stop-study').touch()
        r=collect(c,folder,code,time.perf_counter()-start)
        r['final_task_success']=bool(r['attempts'] and r['attempts'][0]['final_task_success'])
        r['initial_arrays']=initial_arrays(folder)
        retrieval=folder/'agent-workspace/memory-retrieval.jsonl'
        queries=[json.loads(s) for s in retrieval.read_text().splitlines()] if retrieval.exists() else []
        r['memory_queries']=queries;r['retrieved_entries']=sum(len(q['results']) for q in queries)
        r['memory_use']=read(folder/'agent-workspace/memory-use.json')
        r['lossless_compression']=compress_observations(folder)
        r['retained_bytes']=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
        write(folder/'study-case-result.json',r);records[c['case_id']]=r
        write(study/'progress.json',records);report(study,manifest,records)
        print(json.dumps({'finished':c['case_id'],'final_success':r['final_task_success'],'ever_success':r['first_attempt_success'],'retrieved_entries':r['retrieved_entries']}),flush=True)
        capacity='Selected model is at capacity' in (folder/'agent-events.jsonl').read_text() if (folder/'agent-events.jsonl').exists() else False
        if r['subscription_limit_detected'] or capacity:
            write(study/'blocked.json',{'reason':'Subscription capacity or quota; no automatic retry or model/API fallback','case':c['case_id']});break
    write(study/'batch-exit.json',{'finished_unix':time.time(),'finished':sum(r.get('status') in {'completed','runner_failure'} for r in records.values()),'planned':len(manifest['cases'])})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--freeze',action='store_true');a=p.parse_args()
    study=Path(a.study).resolve()
    if a.freeze:freeze(study)
    run(study)
