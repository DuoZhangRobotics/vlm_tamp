"""Run all ten frozen agent cases sequentially, retaining every planned case."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def retained_inputs(run):
    archive=run/'agent-workspace';events=[]
    if (run/'tool-events.jsonl').exists():events=[json.loads(x) for x in (run/'tool-events.jsonl').read_text().splitlines()]
    keep=set();reasons={};latest=None
    def keep_obs(number,why):
        if number is None:return
        for p in (archive/'robot_data').glob(f'attempt*/obs{int(number):03d}/sensors.npz'):
            keep.add(p);reasons.setdefault(str(p.relative_to(run)),[]).append(why)
    for attempt in (archive/'robot_data').glob('attempt*'):
        paths=sorted(attempt.glob('obs*/sensors.npz'))
        keep.update(paths[:1]+paths[-1:])
    for event in events:
        result=event['response'].get('result',{});observation=result.get('observation',result)
        if 'observation_id' in observation:latest=observation['observation_id']
        if event['request']['operation'].startswith('dino_'):
            keep_obs(latest,'DINO input');keep_obs(event['request'].get('arguments',{}).get('reference_observation'),'DINO reference')
        if event['request']['operation']=='measure_surface':
            keep_obs(event['request'].get('arguments',{}).get('observation_id'),'Completion surface measurement')
    commands=[]
    if (run/'agent-events.jsonl').exists():
        for line in (run/'agent-events.jsonl').read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            item=row.get('item',{})
            if row.get('type')=='item.completed' and item.get('type')=='command_execution':commands.append(item.get('command',''))
    scripts=[p.read_text() for p in archive.glob('*.py') if p.name!='robot.py']
    uncertain=False
    for code in commands+scripts:
        if not any(x in code for x in ['np.load','numpy.load','sensors.npz','read_bytes(']):continue
        observed=re.findall(r'robot_data/attempt\d+/obs\d+',code)
        for name in observed:
            target=archive/name/'sensors.npz'
            if target.exists():keep.add(target)
        known_other=bool(re.search(r'(matches\d+\.npz|/p\d+\.npy|geometry\d+/points\.npz)',code))
        if not observed and not known_other and any(x in code for x in ['np.load','numpy.load','read_bytes(']):uncertain=True
    removed=[]
    if not uncertain:
        for path in (archive/'robot_data').glob('attempt*/obs*/sensors.npz'):
            if path in keep:continue
            removed.append({'path':str(path.relative_to(run)),'bytes':path.stat().st_size,'sha256':sha(path)});path.unlink()
    record={'raw_input_dependencies_uncertain':uncertain,'removed':removed,
        'bytes_reclaimed':sum(p['bytes'] for p in removed),'dino_input_retention':reasons,
        'policy':'Retain initial/final raw depth, DINO input/reference frames, completion measurement inputs, identifiable custom-analysis inputs, all RGB, landmarks, geometry, trajectories, videos and code; keep all raw depth when dependencies are uncertain.'}
    write(run/'study-retention.json',record)
    return record


def collect(case,folder,returncode,seconds):
    folder.mkdir(parents=True,exist_ok=True)
    attempts=read(folder/'evaluation-private.json',{'attempts':[]})['attempts']
    audit=read(folder/'audit.json',{})
    result={**case,'status':'completed' if returncode==0 and audit else 'runner_failure',
        'returncode':returncode,'case_wall_seconds':seconds,'attempts':attempts,
        'first_attempt_success':bool(attempts and attempts[0]['official_task_success']),
        'within_two_attempts_success':any(e['official_task_success'] for e in attempts),
        'physical_attempts_used':len(attempts),'agent_exit':read(folder/'agent-exit.json'),
        'dino_calls':audit.get('optional_dino_tool_calls',[]),'api_calls':audit.get('separately_billed_api_calls',0),
        'completion_claim':read(folder/'agent-workspace/completion.json'),
        'input_images':{name:sha(folder/f'attempt01/opened/{name}.png') for name in ['agentview','frontview','wrist'] if (folder/f'attempt01/opened/{name}.png').exists()},
        'trajectory_audit':audit.get('trajectory_audit',[]),
        'isolation_passed':audit.get('filesystem_isolation',{}).get('passed',False)}
    if (folder/'agent-workspace').exists():result['retention']=retained_inputs(folder)
    result['retained_bytes']=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
    text=''
    for name in ['agent-stderr.log','agent-events.jsonl']:
        if (folder/name).exists():text+=(folder/name).read_text()[-200000:]
    result['subscription_limit_detected']=bool(re.search(r"usage.limit|you.ve hit.*limit|quota.exceeded",text,re.I))
    write(folder/'study-case-result.json',result)
    return result


def report(study,manifest,records):
    summaries={}
    for condition in ['no_dino','optional_dino']:
        cases=[c for c in manifest['cases'] if c['condition']==condition]
        done=[records[c['case_id']] for c in cases if records.get(c['case_id'],{}).get('status') in {'completed','runner_failure'}]
        summaries[condition]={'planned_cases':len(cases),'finished_cases':len(done),
            'first_attempt_successes':sum(r.get('first_attempt_success',False) for r in done),
            'within_two_attempts_successes':sum(r.get('within_two_attempts_success',False) for r in done),
            'physical_attempts':sum(r.get('physical_attempts_used',0) for r in done),
            'dino_calls':sum(len(r.get('dino_calls',[])) for r in done),
            'cases_invoking_dino':sum(bool(r.get('dino_calls')) for r in done),
            'case_wall_seconds':sum(r.get('case_wall_seconds',0) for r in done)}
    paired={}
    for state in read(study/'protocol.json')['states']:
        a=records.get(f'state{state:03d}-no_dino',{}).get('input_images',{})
        b=records.get(f'state{state:03d}-optional_dino',{}).get('input_images',{})
        paired[str(state)]={k:a[k]==b[k] for k in a if k in b}
    summary={'conditions':summaries,'records':records,'initial_rgb_pair_checks':paired,
        'all_planned_cases_in_denominator':True,'completed_cases':sum(r.get('status') in {'completed','runner_failure'} for r in records.values()),
        'planned_cases':10,'separate_model_api_calls':sum(r.get('api_calls',0) for r in records.values()),
        'interpretation':'Initial-state transfer on one microwave task/asset; optional tool availability is the treatment. No cross-task generalization or encoder ranking is established.'}
    write(study/'summary.json',summary)
    rows=[]
    for case in manifest['cases']:
        r=records.get(case['case_id'],{});folder=study/'runs'/case['case_id']
        link=f'<a href="runs/{case["case_id"]}/report/index.html">{case["case_id"]}</a>' if (folder/'report/index.html').exists() else case['case_id']
        rows.append(f'<tr><td>{link}</td><td>{html.escape(r.get("status","planned"))}</td>'
            f'<td>{r.get("first_attempt_success","—")}</td><td>{r.get("within_two_attempts_success","—")}</td>'
            f'<td>{r.get("physical_attempts_used","—")}</td><td>{len(r.get("dino_calls",[]))}</td>'
            f'<td>{r.get("case_wall_seconds",0)/60:.1f}</td></tr>')
    cards=[]
    for name,s in summaries.items():
        cards.append(f'<article><h2>{"DINO available on demand" if name=="optional_dino" else "No DINO tool"}</h2>'
            f'<p>First attempt: <b>{s["first_attempt_successes"]}/5</b><br>Within two attempts: <b>{s["within_two_attempts_successes"]}/5</b></p>'
            f'<p>Finished cases: {s["finished_cases"]}/5. DINO calls: {s["dino_calls"]}.</p></article>')
    page=f'''<!doctype html><meta charset="utf-8"><title>Optional DINO agent study</title>
<style>body{{font:17px system-ui;max-width:1200px;margin:35px auto;padding:0 24px;background:#f5f8fb;color:#183344}}article{{padding:18px;background:white;border:1px solid #cad6df;border-radius:12px;margin:15px 0}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:10px;border-bottom:1px solid #ccd6df;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>Can Luna use DINO when it needs it?</h1><p>{summary['completed_cases']}/10 planned agent cases finished. Results are preliminary until all cases are accounted for.</p>
<p>Five reserved initial states; Luna/high through ChatGPT; the same example and frozen agent-written hypotheses; two physical attempts and 600 control steps per attempt. Native LIBERO success is evaluated privately. All planned cases remain in the denominator.</p>
<div class="cards">{''.join(cards)}</div><table><tr><th>Case</th><th>Status</th><th>First attempt</th><th>Within two</th><th>Attempts</th><th>DINO calls</th><th>Total min</th></tr>{''.join(rows)}</table>
<article><h2>What is controlled</h2><p>The robot, grasp/IK/planning tools, measured-depth geometry and physical execution are shared. DINO availability and its tool descriptions differ. Luna may ignore it. Source code, prior project results and hidden object state remain inaccessible. Guidelines are frozen input hypotheses, not proven general rules.</p><p>Cold DINO loading, inference and communication are included in agent time. Full case time also includes setup and cleanup. A case using DINO is not automatically evidence that DINO caused its outcome.</p></article>
<article><h2>Paired initial RGB checks</h2><pre>{html.escape(json.dumps(paired,indent=2))}</pre></article>
<p><a href="protocol.json">Frozen protocol</a> · <a href="guidelines.json">Supplied agent-written hypotheses</a> · <a href="state-reservation.json">State reservation audit</a> · <a href="summary.json">Full results</a></p>'''
    (study/'index.html').write_text(page)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',required=True);args=parser.parse_args()
    study=(ROOT/args.study).resolve();manifest=read(study/'manifest.json');protocol=read(study/'protocol.json')
    hashes=read(study/'frozen-source-hashes.json');records=read(study/'progress.json',{})
    (study/'runs').mkdir(exist_ok=True);report(study,manifest,records)
    write(study/'batch-process.json',{'pid':os.getpid(),'started_unix':time.time()})
    for case in manifest['cases']:
        if records.get(case['case_id'],{}).get('status') in {'completed','runner_failure'}:continue
        if (study/'stop-study').exists():break
        changed=[name for name,h in hashes.items() if not (ROOT/name).exists() or sha(ROOT/name)!=h]
        if changed:raise RuntimeError('Frozen source changed: '+', '.join(changed[:8]))
        if sha(study/'prompt.txt')!=protocol['prompt_sha256'] or sha(study/'guidelines.json')!=protocol['guidelines_sha256']:
            raise RuntimeError('Frozen input changed')
        if sha(Path(protocol['dino']['checkpoint']))!=protocol['dino']['sha256']:raise RuntimeError('DINO checkpoint changed')
        if any(sha(Path(p))!=h for p,h in protocol['runtime_artifacts_sha256'].items()):raise RuntimeError('Frozen planner or grasp model artifact changed')
        folder=study/'runs'/case['case_id']
        if folder.exists():raise RuntimeError('An interrupted case exists; preserve it and review before resuming')
        records[case['case_id']]={**case,'status':'running'};write(study/'progress.json',records);report(study,manifest,records)
        command=['/usr/bin/python3',str(ROOT/'scripts/run_subscription_agent_pilot.py'),'--output',str(folder),
            '--state',str(case['state']),'--prompt-file',str(study/'prompt.txt'),
            '--guidelines-source',str(study/'guidelines.json'),'--fit-diagnostics','--evaluation-case',
            '--agent-seconds',str(protocol['agent_seconds_per_case']),
            *(['--dino-tools'] if case['condition']=='optional_dino' else [])]
        start=time.perf_counter();print(json.dumps({'event':'case_started',**case}),flush=True)
        with (study/(case['case_id']+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:code=process.wait()
            except KeyboardInterrupt:
                process.send_signal(signal.SIGINT);code=process.wait(timeout=950)
                (study/'stop-study').touch()
        result=collect(case,folder,code,time.perf_counter()-start)
        records[case['case_id']]=result;write(study/'progress.json',records);report(study,manifest,records)
        print(json.dumps({'event':'case_finished','case':case['case_id'],'first':result['first_attempt_success'],
            'within_two':result['within_two_attempts_success'],'attempts':result['physical_attempts_used'],'dino_calls':len(result['dino_calls'])}),flush=True)
        if result['subscription_limit_detected']:
            write(study/'blocked.json',{'reason':'ChatGPT subscription usage limit; no API-key fallback','case':case['case_id']})
            break
    write(study/'batch-exit.json',{'finished_unix':time.time(),'completed_cases':sum(r.get('status') in {'completed','runner_failure'} for r in records.values()),'planned_cases':10})


if __name__=='__main__':main()
