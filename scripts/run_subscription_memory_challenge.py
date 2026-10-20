"""Frozen paired stress test of existing episodic memory; no numerical-policy changes."""
import argparse
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.episodic_memory import store_hashes
from run_subscription_dino_study import read,write,sha,collect
from run_subscription_memory_study import initial_arrays
from replay_subscription_episode import compress_observations

CONDITIONS=('without_memory','with_memory')
LABELS={'without_memory':'Fixed guide only','with_memory':'Guide + episode memory'}
GROUPS={'known_challenge':'Previously difficult cases','fresh_transfer':'Reserved transfer scenes'}
DONE={'completed','runner_failure'}


def scenarios(states):
    if len(states)!=3 or len(set(states))!=3 or any(type(s) is not int or not 0<=s<50 for s in states):
        raise ValueError('Three distinct official state indices required')
    if set(states)&{37,47}:raise ValueError('Reserved states must differ from the known challenge states')
    rows=[dict(scenario_id='known-nominal-047',group='known_challenge',state=47,scene_variant='nominal'),
          dict(scenario_id='known-yaw-shift-037',group='known_challenge',state=37,scene_variant='yaw_negative_shifted'),
          dict(scenario_id='known-clutter-047',group='known_challenge',state=47,scene_variant='clutter_shifted')]
    for state,variant in zip(states,['yaw_negative_shifted','clutter_shifted','frontview_shifted']):
        rows.append(dict(scenario_id=f'fresh-{variant}-{state:03d}',group='fresh_transfer',state=state,scene_variant=variant))
    return rows


def freeze(study):
    if (study/'protocol.json').exists():raise ValueError('Already frozen; never overwrite')
    prior=ROOT/'results/subscription-episodic-memory-v2-20261019'
    if not read(prior/'completion.json')['completed']:raise ValueError('Prior comparison incomplete')
    profile=read(ROOT/'configs/subscription_memory.json');library=(ROOT/profile['public_library']).resolve()
    hashes=store_hashes(library)
    if hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()!=profile['library_manifest_sha256']:
        raise ValueError('Registered memory changed')
    reservation=read(study/'state-reservation.json');states=reservation['selected_states'][:3]
    if set(states)&set(reservation['excluded_used_or_previously_planned_states']):raise ValueError('Reserved state was previously used')
    rows=scenarios(states);cases=[]
    for i,s in enumerate(rows):
        for condition in (CONDITIONS if i%2==0 else CONDITIONS[::-1]):
            cases.append({**s,'condition':condition,'case_id':s['scenario_id']+'-'+condition})
    manifest={'task':'KITCHEN_SCENE7_open_the_microwave','scenarios':rows,'cases':cases,
        'planned_cases':len(cases),'all_planned_cases_in_denominator':True,'one_physical_attempt_each':True}
    for key,dest in [('prompt','prompt.txt'),('common_guide','guidelines.json')]:
        source=ROOT/profile[key]
        if sha(source)!=profile[key+'_sha256']:raise ValueError('Registered input changed: '+key)
        shutil_copy(source,study/dest)
    (study/'empty-library').mkdir(exist_ok=True)
    inherited=read(prior/'protocol.json')
    protocol={k:copy.deepcopy(inherited[k]) for k in ['model','reasoning_effort','authentication','dino','dino_runtime',
        'runtime_artifacts_sha256','simulator_seed','settling_actions','controller','agent_seconds_per_case']}
    old_sources=read(prior/'frozen-source-hashes.json')
    drift=[f for f,h in old_sources.items() if not (ROOT/f).exists() or sha(ROOT/f)!=h]
    if drift:raise ValueError('Previously evaluated source changed: '+str(drift))
    old_analysis=read(ROOT/'results/subscription-strategy-transfer-20261019/supplemental-analysis.json')
    selected_names={'nominal-state047','yaw_negative_shifted-state037','clutter_shifted-state047'}
    selection_evidence=[x for x in old_analysis['paired_scenarios'] if x['scenario'] in selected_names]
    write(study/'known-case-selection.json',{'source':'Earlier two-attempt guide comparison, not this memory experiment',
        'selected_because_of_prior_difficulty':True,'records':selection_evidence,
        'scope':'Deliberately selected challenge cases, not held-out cases or an unbiased success-rate estimate.'})
    protocol.update(task=manifest['task'],maximum_attempts_per_case=1,maximum_control_steps_per_attempt=600,
        training_state=37,reserved_states=states,memory_store=str(library),library_hashes=hashes,
        prompt_sha256=sha(study/'prompt.txt'),guidelines_sha256=sha(study/'guidelines.json'),
        same_prompt_guide_example_tools_as_previous_study=True,previous_evaluated_source_drift=[],
        primary_metric='Native LIBERO goal becomes true during the single episode',
        secondary_metric='Native goal is still true at episode end',
        metric_note='Both measures match the preceding report. Its stricter end-state measure was listed first; neither measure controls execution.',
        groups=GROUPS,memory_updated_from_evaluation=False,new_task_specific_rules=False,
        numerical_planning_grasp_ik_contact_control_unchanged=True,separate_api_calls_allowed=False,
        execution_vetoes=0,source_and_private_evaluation_access=False,
        infrastructure_policy='Keep failed launches in the denominator; stop on subscription capacity/quota with no automatic retry, model switch, or paid API fallback.',
        interpretation='One task and asset. Known difficult cases and reserved scenes are separate strata. Historical guide results use a different prompt/attempt budget and are selection evidence only, not the paired baseline.',
        cleanup='Compact every episode, prune only redundant or unneeded raw inputs, losslessly compress retained private observations; protect active persistent memory.')
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+list((ROOT/'src/vlm_tamp/native').glob('*.cc'))+list((ROOT/'scripts').glob('*.py'))
    files +=[ROOT/'configs/articulated_libero.json',ROOT/'configs/subscription_memory.json',ROOT/'requirements-rekep-libero.txt',ROOT/'requirements-lock.txt']
    write(study/'frozen-source-hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
    write(study/'manifest.json',manifest);write(study/'protocol.json',protocol)
    print(json.dumps({'frozen_cases':len(cases),'reserved_states':states,'memory_unchanged':True}),flush=True)


def shutil_copy(source,target):
    import shutil
    if target.exists():raise ValueError('Refusing to overwrite input '+str(target))
    shutil.copyfile(source,target)


def diagnostic_counts(folder):
    path=folder/'tool-events.jsonl'
    events=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    return {'planning_failures':sum(e['request']['operation'].startswith('plan_') and not e['response']['ok'] for e in events),
            'grounding_calls':sum(e['request']['operation']=='ground' for e in events),
            'model_updates':sum(e['request']['operation'] in {'apply_tracking','set_model'} and e['response']['ok'] for e in events),
            'tool_errors':sum(not e['response']['ok'] for e in events),
            'tool_wall_seconds':sum(e['wall_seconds'] for e in events)}


def report(study,manifest,records):
    scores={}
    for group in GROUPS:
        scores[group]={}
        for condition in CONDITIONS:
            cases=[c for c in manifest['cases'] if c['group']==group and c['condition']==condition]
            done=[records[c['case_id']] for c in cases if records.get(c['case_id'],{}).get('status') in DONE]
            scores[group][condition]={'planned':len(cases),'completed':len(done),
                'native_goal_reached':sum(r.get('first_attempt_success',False) for r in done),
                'native_goal_at_end':sum(r.get('final_task_success',False) for r in done)}
    pairs={}
    for scenario in manifest['scenarios']:
        a=records.get(scenario['scenario_id']+'-without_memory',{}).get('initial_arrays',{})
        b=records.get(scenario['scenario_id']+'-with_memory',{}).get('initial_arrays',{})
        if a and b:pairs[scenario['scenario_id']]={'same_keys':a.keys()==b.keys(),'equal_arrays':a==b}
    summary={'scores':scores,'records':records,'initial_sensor_pairs':pairs,'all_planned_cases_in_denominator':True,
        'completed_cases':sum(r.get('status') in DONE for r in records.values()),'planned_cases':len(manifest['cases']),
        'separately_billed_api_calls':sum(r.get('api_calls',0) for r in records.values())}
    write(study/'summary.json',summary)
    table=[]
    for group,conditions in scores.items():
        for condition,s in conditions.items():
            table.append(f'<tr><td>{GROUPS[group]}</td><td>{LABELS[condition]}</td><td>{s["completed"]}/{s["planned"]}</td><td>{s["native_goal_reached"]}/{s["planned"]}</td><td>{s["native_goal_at_end"]}/{s["planned"]}</td></tr>')
    rows=[];failures=[];clips=[]
    for c in manifest['cases']:
        r=records.get(c['case_id'],{});folder=study/'runs'/c['case_id'];href=f'runs/{c["case_id"]}/report/index.html'
        label=f'<a href="{href}">{c["case_id"]}</a>' if (study/href).exists() else c['case_id']
        attempts=r.get('attempts',[]);steps=attempts[0]['control_steps'] if attempts else '—'
        rows.append(f'<tr><td>{label}</td><td>{r.get("status","planned")}</td><td>{r.get("first_attempt_success","—")}</td><td>{r.get("final_task_success","—")}</td><td>{steps}</td><td>{r.get("diagnostics",{}).get("planning_failures","—")}</td><td>{r.get("case_wall_seconds",0)/60:.1f}</td></tr>')
        if r.get('status') in DONE and not r.get('first_attempt_success',False):
            findings=folder/'agent-workspace/findings.txt'
            text=findings.read_text() if findings.exists() else 'No agent findings were produced. See launch logs and private evaluation.'
            failures.append(f'<details><summary>{label}: native goal not reached</summary><pre>{html.escape(text)}</pre></details>')
        if attempts:
            clips.append(f'<details><summary>{label} — goal reached: {r["first_attempt_success"]}; at end: {r["final_task_success"]}</summary><video controls preload="none" src="runs/{c["case_id"]}/attempt01/initial/rollout.mp4"></video></details>')
    completed=summary['completed_cases']==summary['planned_cases']
    phase='Completed' if completed else 'Running; pending cases have no outcome yet'
    (study/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Does memory help on harder microwave cases?</title>
<style>body{{font:17px system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#183344;background:#f5f8fb}}article{{background:white;border:1px solid #ccd6df;border-radius:12px;padding:20px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #d1dae2;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}video{{width:100%;max-height:550px}}details{{margin:14px 0}}</style>
<h1>Does saved experience help on harder cases?</h1><p>{phase}: {summary['completed_cases']}/{summary['planned_cases']} sessions.</p>
<article><p>The preceding small test passed in both conditions. This follow-up keeps Luna/high, the prompt, fixed guide, human example, optional DINO, grasp/IK/planning/contact tools and controller unchanged. The only treatment is access to the previously saved public episode and its agent-authored procedure. One physical attempt per session, 600 control steps; every issued trajectory executes fully.</p>
<table><tr><th>Case group</th><th>Experience</th><th>Finished</th><th>Native goal reached</th><th>Goal true at end</th></tr>{''.join(table)}</table>
<p>Previously difficult cases were selected using older outcomes and are not held-out tests. Reserved scenes use previously unused state indices from retained metadata. Results for these groups stay separate. The original task-success criterion is the native predicate becoming true; end-state retention is reported alongside it. Planning failures alone do not make an episode a task failure.</p></article>
<article><h2>Runs</h2><table><tr><th>Case</th><th>Status</th><th>Goal reached</th><th>At end</th><th>Steps</th><th>Planning failures</th><th>Minutes</th></tr>{''.join(rows)}</table></article>
<article><h2>Videos</h2><p>Movies show simulated motion, not the reasoning and planning waits. Session wall times are listed above.</p>{''.join(clips) or '<p>No completed physical episodes yet.</p>'}</article>
<article><h2>Failure evidence</h2>{''.join(failures) or '<p>No completed native-goal failures recorded so far.</p>'}<p>Agent findings are interpretations. Native outcomes are checked independently, and causal diagnoses require additional evidence.</p></article>
<article><h2>Limits</h2><p>One microwave task and asset, with inherited numerical and grasp priors. No broader-task or universal-reliability claim. The library is frozen; evaluation outcomes are not fed into later agents. Earlier two-attempt guide-study scores are selection evidence, not the matched baseline here. Source code, unrelated history and private simulator evaluation remain unavailable to the agent.</p></article>
<p><a href="protocol.json">Frozen protocol</a> · <a href="manifest.json">All planned cases</a> · <a href="known-case-selection.json">Challenge selection</a> · <a href="state-reservation.json">State reservation</a> · <a href="summary.json">Results</a> · <a href="../../data/agent_memory/luna-microwave-v1/library/episode-0001/procedure.json">Unchanged memory</a></p>''')


def run(study, *, extra_arguments=(), reporter=None, diagnostic_reader=None):
    render=reporter or report
    diagnose=diagnostic_reader or diagnostic_counts
    manifest=read(study/'manifest.json');protocol=read(study/'protocol.json');records=read(study/'progress.json',{})
    (study/'runs').mkdir(exist_ok=True);render(study,manifest,records)
    write(study/'batch-process.json',{'pid':os.getpid(),'started_unix':time.time()})
    for c in manifest['cases']:
        if records.get(c['case_id'],{}).get('status') in DONE:continue
        if (study/'stop-study').exists():break
        drift=[f for f,h in read(study/'frozen-source-hashes.json').items() if not (ROOT/f).exists() or sha(ROOT/f)!=h]
        if drift:raise RuntimeError('Frozen source changed: '+str(drift[:8]))
        assert store_hashes(Path(protocol['memory_store']))==protocol['library_hashes']
        assert store_hashes(study/'empty-library')=={}
        assert sha(study/'prompt.txt')==protocol['prompt_sha256'] and sha(study/'guidelines.json')==protocol['guidelines_sha256']
        assert all(sha(Path(f))==h for f,h in protocol['runtime_artifacts_sha256'].items())
        assert sha(Path(protocol['dino']['checkpoint']))==protocol['dino']['sha256']
        folder=study/'runs'/c['case_id']
        if folder.exists():raise RuntimeError('Interrupted case already exists; do not overwrite or silently retry')
        memory=Path(protocol['memory_store']) if c['condition']=='with_memory' else study/'empty-library'
        records[c['case_id']]={**c,'status':'running'};write(study/'progress.json',records);render(study,manifest,records)
        args=['/usr/bin/python3',str(ROOT/'scripts/run_subscription_agent_pilot.py'),'--output',str(folder),
            '--state',str(c['state']),'--scene-variant',c['scene_variant'],'--prompt-file',str(study/'prompt.txt'),
            '--guidelines-source',str(study/'guidelines.json'),'--memory-store',str(memory),'--attempts','1',
            '--fit-diagnostics','--evaluation-case','--dino-tools','--agent-seconds',str(protocol['agent_seconds_per_case']),*extra_arguments]
        before=time.perf_counter();print(json.dumps({'started':c}),flush=True)
        with (study/(c['case_id']+'.log')).open('w') as log:
            process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:code=process.wait()
            except KeyboardInterrupt:
                process.send_signal(signal.SIGINT);code=process.wait(timeout=950);(study/'stop-study').touch()
        r=collect(c,folder,code,time.perf_counter()-before)
        r['final_task_success']=bool(r['attempts'] and r['attempts'][0]['final_task_success'])
        r['initial_arrays']=initial_arrays(folder)
        retrieval=folder/'agent-workspace/memory-retrieval.jsonl'
        queries=[json.loads(x) for x in retrieval.read_text().splitlines()] if retrieval.exists() else []
        r['memory_queries']=queries;r['retrieved_entries']=sum(len(q['results']) for q in queries)
        r['memory_use']=read(folder/'agent-workspace/memory-use.json');r['diagnostics']=diagnose(folder)
        r['lossless_compression']=compress_observations(folder)
        r['retained_bytes']=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
        write(folder/'study-case-result.json',r);records[c['case_id']]=r;write(study/'progress.json',records);render(study,manifest,records)
        print(json.dumps({'finished':c['case_id'],'native_goal':r['first_attempt_success'],'at_end':r['final_task_success']}),flush=True)
        capacity=any('Selected model is at capacity' in p.read_text() for p in [folder/'agent-events.jsonl',folder/'agent-stderr.log'] if p.exists())
        if r['subscription_limit_detected'] or capacity:
            write(study/'blocked.json',{'case':c['case_id'],'reason':'Subscription capacity/quota; no automatic retry or fallback'});break
    write(study/'batch-exit.json',{'finished_unix':time.time(),'finished_cases':sum(r.get('status') in DONE for r in records.values()),'planned_cases':len(manifest['cases'])})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',required=True);parser.add_argument('--freeze',action='store_true');a=parser.parse_args()
    study=(ROOT/a.study).resolve()
    if study==ROOT/'results' or not study.is_relative_to(ROOT/'results'):raise ValueError('Use a results subdirectory')
    if a.freeze:freeze(study)
    run(study)
