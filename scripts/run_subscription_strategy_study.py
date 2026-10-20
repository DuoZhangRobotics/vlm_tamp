"""Sequential frozen guideline-transfer evaluation with per-case cleanup."""
import argparse
import html
import json
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
from run_subscription_dino_study import read,write,sha,collect
ROOT=Path(__file__).resolve().parents[1]

def diagnostics(folder):
    p=folder/'tool-events.jsonl'
    rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    count=lambda op:sum(e['request']['operation']==op for e in rows)
    result={'tool_wall_seconds':sum(e['wall_seconds'] for e in rows),
        'planning_failures':sum(e['request']['operation'].startswith('plan_') and not e['response']['ok'] for e in rows),
        'ground_calls':count('ground'),'model_update_calls':count('set_model')+count('apply_tracking'),
        'physical_resets':count('reset'),'gripper_close_requests':sum(e['request']['operation']=='gripper' and e['request'].get('arguments',{}).get('close',True) for e in rows),
        'tracking_calls':count('measure_tracking'),
        'tracking_rejections':sum(e['request']['operation']=='measure_tracking' and not e['response'].get('result',{}).get('measurement',{}).get('accepted',False) for e in rows),
        'interpretation':'Request counts are behavior diagnostics, not proof of a repeated mistake or a causal failure diagnosis.'}
    first=next((e for e in rows if e['request']['operation']=='ground' and e['response']['ok']),None)
    native=[]
    for path in sorted((folder/'attempt01/initial').glob('evaluation_contact_diagnostics.json')):
        def walk(x):
            if isinstance(x,dict):
                if x.get('joint_name')=='microwave_1_microjoint':native.append(x)
                for v in x.values():walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        walk(read(path,{}))
    if first and native:
        estimated=first['response']['result']['model']['joint_origin_world_m']
        result['initial_hinge_xy_error_m_evaluation_only']=float(np.linalg.norm(np.asarray(estimated)[:2]-np.asarray(native[0]['anchor_world_m'])[:2]))
    return result

def report(study,manifest,records):
    summaries={}
    for group in ['all','nominal','changed_scene']:
        summaries[group]={}
        for condition in ['original','consolidated']:
            cases=[c for c in manifest['cases'] if c['condition']==condition and (group=='all' or c['group']==group)]
            done=[records[c['case_id']] for c in cases if records.get(c['case_id'],{}).get('status') in {'completed','runner_failure'}]
            summaries[group][condition]={'planned':len(cases),'finished':len(done),
                'first_success':sum(r.get('first_attempt_success',False) for r in done),
                'within_two_success':sum(r.get('within_two_attempts_success',False) for r in done),
                'physical_attempts':sum(r.get('physical_attempts_used',0) for r in done),
                'dino_calls':sum(len(r.get('dino_calls',[])) for r in done),
                'case_wall_seconds':sum(r.get('case_wall_seconds',0) for r in done)}
    pairs={}
    for s in manifest['scenarios']:
        a=records.get(s['scenario_id']+'-original',{}).get('input_images',{})
        b=records.get(s['scenario_id']+'-consolidated',{}).get('input_images',{})
        pairs[s['scenario_id']]={k:a[k]==b[k] for k in a if k in b}
    summary={'conditions':summaries,'records':records,'initial_rgb_pair_checks':pairs,
        'completed_cases':sum(r.get('status') in {'completed','runner_failure'} for r in records.values()),
        'planned_cases':20,'all_planned_cases_in_denominator':True,
        'separate_model_api_calls':sum(r.get('api_calls',0) for r in records.values())}
    write(study/'summary.json',summary)
    rows=[]
    for c in manifest['cases']:
        r=records.get(c['case_id'],{});url=f'runs/{c["case_id"]}/report/index.html'
        label=f'<a href="{url}">{c["case_id"]}</a>' if (study/url).exists() else c['case_id']
        rows.append(f'<tr><td>{label}</td><td>{r.get("status","planned")}</td><td>{r.get("first_attempt_success","—")}</td><td>{r.get("within_two_attempts_success","—")}</td><td>{r.get("physical_attempts_used","—")}</td><td>{r.get("case_wall_seconds",0)/60:.1f}</td></tr>')
    scores=[]
    for group,conditions in summaries.items():
        for condition,x in conditions.items():scores.append(f'<tr><td>{group}</td><td>{condition}</td><td>{x["finished"]}/{x["planned"]}</td><td>{x["first_success"]}/{x["planned"]}</td><td>{x["within_two_success"]}/{x["planned"]}</td></tr>')
    note='Final case accounting complete; see the independent audit when available.' if summary['completed_cases']==20 else 'Running: pending cases have no outcome yet; final denominators remain fixed.'
    (study/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Does agent experience transfer?</title><style>body{{font:17px system-ui;max-width:1300px;margin:30px auto;padding:20px;background:#f5f8fb;color:#183344}}article{{background:white;border:1px solid #cbd5de;border-radius:12px;padding:20px;margin:20px 0}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #d1dae2}}img{{max-width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>Does Luna's consolidated experience transfer?</h1><p>{summary['completed_cases']}/20 agent sessions complete. {note}</p>
<article><p>Compare the original guide with Luna's independently consolidated and audited guide. Five fresh initial states plus five scene variants; two physical attempts per session, 600 control steps per attempt. Luna/high uses ChatGPT throughout; DINOv3 remains optional in both conditions. Same prompt, human example and numerical tools. Guides are frozen before evaluation; no cross-case learning or human recovery hints during the test.</p>
<table><tr><th>Scene group</th><th>Guide</th><th>Finished</th><th>First-attempt native success</th><th>Within-two native success</th></tr>{''.join(scores)}</table></article>
<article><h2>Case evidence and videos</h2><table><tr><th>Case</th><th>Status</th><th>First</th><th>Within two</th><th>Attempts</th><th>Total minutes</th></tr>{''.join(rows)}</table></article>
<article><h2>Predeclared scene changes</h2><p>This preview uses a previously used development state, not an evaluation state. Variants change placement, yaw, the frontal camera, or plate clutter. The bowl is transported with the microwave to preserve its support. Task success uses LIBERO's unchanged native predicate. These variants are custom scene perturbations, not official benchmark episodes.</p><img src="development-scene-sheet.png" alt="Validated nominal and changed microwave scenes"></article>
<article><h2>Controls and limits</h2><p>The actual physics, complete-trajectory executor, contact handling, grasp/IK and MC-VAMP implementations are shared. Changes occur at scene initialization. Agent inputs contain observations and calibration; simulator identities and true articulation remain private evaluation data. All planned cases stay in the denominator, including failures before movement. No execution-time vetoes or separately billed API calls.</p><p>One task and asset; one sampled session per guide/scenario. This tests transfer under these scene changes, not new mechanisms or universal reliability. Tools include inherited geometric/contact priors. Timing includes reasoning, communication, perception, planning, execution, recording and cleanup; nested tool times must not be added to total time.</p></article>
<p><a href="original-guidelines.json">Original guide</a> · <a href="consolidated-guidelines.json">Luna consolidated guide</a> · <a href="consolidation-audit.json">Luna evidence audit</a> · <a href="protocol.json">Frozen protocol</a> · <a href="summary.json">Full results</a></p>''')

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args()
    study=(ROOT/a.study).resolve();manifest=read(study/'manifest.json');protocol=read(study/'protocol.json')
    hashes=read(study/'frozen-source-hashes.json');records=read(study/'progress.json',{})
    (study/'runs').mkdir(exist_ok=True);report(study,manifest,records)
    write(study/'batch-process.json',{'pid':__import__('os').getpid(),'started_unix':time.time()})
    for c in manifest['cases']:
        if records.get(c['case_id'],{}).get('status') in {'completed','runner_failure'}:continue
        if (study/'stop-study').exists():break
        drift=[f for f,h in hashes.items() if not (ROOT/f).exists() or sha(ROOT/f)!=h]
        if drift:raise RuntimeError('Frozen source changed: '+str(drift[:8]))
        assert sha(study/'prompt.txt')==protocol['prompt_sha256']
        assert all(sha(study/f'{k}-guidelines.json')==v for k,v in protocol['guidelines_sha256'].items())
        assert sha(Path(protocol['dino']['checkpoint']))==protocol['dino']['sha256']
        assert all(sha(Path(f))==h for f,h in protocol['runtime_artifacts_sha256'].items())
        folder=study/'runs'/c['case_id']
        if folder.exists():raise RuntimeError('Interrupted case exists; preserve and review before resuming')
        records[c['case_id']]={**c,'status':'running'};write(study/'progress.json',records);report(study,manifest,records)
        command=['/usr/bin/python3',str(ROOT/'scripts/run_subscription_agent_pilot.py'),'--output',str(folder),
            '--state',str(c['state']),'--scene-variant',c['scene_variant'],'--prompt-file',str(study/'prompt.txt'),
            '--guidelines-source',str(study/f'{c["condition"]}-guidelines.json'),'--fit-diagnostics','--evaluation-case',
            '--dino-tools','--agent-seconds',str(protocol['agent_seconds_per_case'])]
        start=time.perf_counter();print(json.dumps({'event':'case_started',**c}),flush=True)
        with (study/(c['case_id']+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:code=process.wait()
            except KeyboardInterrupt:
                process.send_signal(signal.SIGINT);code=process.wait(timeout=950);(study/'stop-study').touch()
        r=collect(c,folder,code,time.perf_counter()-start);r['diagnostics']=diagnostics(folder)
        write(folder/'study-case-result.json',r);records[c['case_id']]=r
        write(study/'progress.json',records);report(study,manifest,records)
        print(json.dumps({'event':'case_finished','case':c['case_id'],'first':r['first_attempt_success'],
            'within_two':r['within_two_attempts_success'],'attempts':r['physical_attempts_used']}),flush=True)
        if r['subscription_limit_detected']:
            write(study/'blocked.json',{'reason':'ChatGPT subscription limit; no API fallback','case':c['case_id']});break
    write(study/'batch-exit.json',{'finished_unix':time.time(),'completed_cases':sum(r.get('status') in {'completed','runner_failure'} for r in records.values()),'planned_cases':20})
if __name__=='__main__':main()
