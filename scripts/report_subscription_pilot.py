"""Audit a completed subscription-agent pilot and render its compact report."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.tool_event_audit import operation_name


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def archive_workspace(work,archive):
    """Move finished-run evidence instead of temporarily doubling disk usage."""
    work,archive=Path(work).resolve(),Path(archive).resolve()
    archive.mkdir(exist_ok=True)
    retained=[]
    for f in work.rglob('*'):
        relative=f.relative_to(work)
        if relative.parts[0] in {'__pycache__','episodic_library'} or '__pycache__' in relative.parts:
            continue
        if not f.is_file() or f.is_symlink() or not f.resolve().is_relative_to(work):
            continue
        dest=archive/relative
        if work!=archive:
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and hashlib.sha256(f.read_bytes()).digest()!=hashlib.sha256(dest.read_bytes()).digest():
                raise RuntimeError(f'Archive conflict; preserving both copies: {relative}')
            shutil.move(str(f),str(dest))
        if relative.parts[0]!='robot_data':retained.append(str(relative))
    return retained


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--workspace',required=True)
    p.add_argument('--report')
    a=p.parse_args();run=Path(a.run).resolve();work=Path(a.workspace).resolve()
    if not (run/'agent-exit.json').exists():raise ValueError('Agent is still running')
    rows=[json.loads(x) for x in (run/'tool-events.jsonl').read_text().splitlines()] if (run/'tool-events.jsonl').exists() else []
    evaluation=read(run/'evaluation-private.json',{'attempts':[]})
    protocol=read(run/'protocol.json',{})
    evaluation_case=protocol.get('held_out_initial_state_evaluation',False)
    scope=('one reserved case in a frozen comparison; inherited numerical tools; not cross-task generalization'
           if evaluation_case else 'one development case; inherited numerical tools; not a transfer evaluation')
    if 'episodic_memory' in protocol:
        scope='One physical attempt in the episodic-memory study; see the study manifest for same-case versus transfer membership. Explicit public memory is allowed; implementation and private evaluation remain blocked.'
    if protocol.get('learning_practice'):
        scope='Adaptive development practice in one agent session on one scene. Outcome feedback is limited to the official success boolean when enabled. These are learning attempts, not held-out transfer results.'
    episodes=evaluation['attempts']
    archive=run/'agent-workspace'
    retained=archive_workspace(work,archive)
    issued_audit=[]
    for episode in episodes:
        folder=run/f'attempt{episode["attempt"]:02d}'
        tracking=read(folder/'tracking.json',[])
        files=sorted(folder.glob('*-issued-*.npy'),key=lambda p:int(p.name.split('-')[0]))
        paths=[np.load(f) for f in files]
        expected=np.vstack(paths) if paths else np.empty((0,7))
        actual=np.asarray([e['reference_joints'] for e in tracking]).reshape(-1,7)
        issued_audit.append({'attempt':episode['attempt'],
            'issued_references':len(expected),'recorded_references':len(actual),
            'references_match':bool(np.array_equal(expected,actual)),
            'all_commands_marked_issued':all(e.get('command_issued',False) for e in tracking),
            'execution_veto_checks':sum(e.get('execution_veto_checks',0) for e in tracking)})
        issued_audit[-1]['control_steps_after_first_success']=episode.get('control_steps_after_first_success',0)
        issued_audit[-1]['success_terminated_trajectories']=episode.get('success_terminated_trajectories',0)
    before=read(run/'api-ledger-checkpoint.json',{})
    ledger_unchanged=all(Path(f).exists() and hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in before.items())
    tool_text=json.dumps([r['response'] for r in rows])
    leak_keys=['evaluation_task_success','native_hinge','native_joint','object_inventory','evaluation_contact']
    usage=[];agent_commands=[];agent_images=[]
    for line in (run/'agent-events.jsonl').read_text().splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError:continue
        if row.get('type')=='turn.completed':usage.append(row.get('usage',{}))
        item=row.get('item',{})
        if row.get('type')=='item.completed' and item.get('type')=='command_execution':
            agent_commands.append(item.get('command',''))
        if item.get('type') in {'image_view','view_image'}:agent_images.append(item)
    audit={'authentication':read(run/'authentication-check.json'),
        'filesystem_isolation':read(run/'isolation.json'), 'trajectory_audit':issued_audit,
        'api_ledgers_unchanged_since_checkpoint':ledger_unchanged,
        'separately_billed_api_calls':0,
        'forbidden_evaluation_fields_in_feedback':[k for k in leak_keys if k in tool_text],
        'agent_shell_commands':len(agent_commands),'agent_usage':usage,
        'retained_agent_authored_files':retained,'human_task_hints_during_run':0,
        'optional_dino_tool_calls':[{'event':r['sequence'],'operation':r['request']['operation'],
            'ok':r['response']['ok'],'wall_seconds':r['wall_seconds']} for r in rows
            if operation_name(r).startswith('dino_')],
        'agent_exit':read(run/'agent-exit.json'),
        'scope':scope}
    (run/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    report=Path(a.report).resolve() if a.report else run/'report'
    report.mkdir(parents=True,exist_ok=True)
    cards=[]
    for ep in episodes:
        movie=run/f'attempt{ep["attempt"]:02d}/initial/rollout.mp4'
        import os
        href=os.path.relpath(movie,report)
        cards.append(f'<article><h2>Attempt {ep["attempt"]}: {"success" if ep["official_task_success"] else "failure"}</h2>'
            f'<p>Native LIBERO success: <strong>{str(ep["official_task_success"]).lower()}</strong>. '
            f'{ep["control_steps"]} control steps; {ep["completed_trajectories"]} complete trajectories; '
            f'{ep.get("success_terminated_trajectories",0)} terminated on simulator success. '
            f'{ep.get("control_steps_after_first_success",0)} control steps after first success.</p>'
            f'<video controls preload="metadata" src="{html.escape(href)}"></video></article>')
    trace=[]
    for row in rows:
        request=row['request'];response=row['response'];result=response.get('result',{})
        trace.append(f'<tr><td>{row["sequence"]}</td><td>{html.escape(operation_name(row))}</td>'
            f'<td>{"ok" if response["ok"] else html.escape(response.get("error","error"))}</td>'
            f'<td>{row["wall_seconds"]:.2f}s</td><td><details><summary>Inputs / output</summary><pre>'
            f'{html.escape(json.dumps({"arguments":request.get("arguments",{}),"result":result},indent=2))}</pre></details></td></tr>')
    findings=(archive/'findings.txt').read_text() if (archive/'findings.txt').exists() else 'No findings file was produced.'
    guidelines=(archive/'guidelines.json').read_text() if (archive/'guidelines.json').exists() else 'No guidelines file was produced.'
    successful=sum(ep['official_task_success'] for ep in episodes)
    body=f'''<!doctype html><meta charset="utf-8"><title>Subscription Luna robotics pilot</title>
<style>body{{font:17px system-ui;margin:36px auto;max-width:1100px;padding:0 24px;color:#183344;background:#f5f8fb}}article{{background:white;border:1px solid #d1dbe4;border-radius:12px;padding:20px;margin:18px 0}}video{{width:100%;max-height:600px;background:#152535}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;max-height:500px;overflow:auto}}table{{width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid #ccd6df;padding:9px;text-align:left}}.note{{padding:16px;background:#e8eff8;border-radius:10px}}</style>
<h1>Luna through ChatGPT: autonomous robot-tool pilot</h1>
<p>{successful}/{len(episodes)} physical attempts succeeded on one microwave opening case.</p>
<p class="note">Luna high reasoning used the ChatGPT login. The agent had a coding workspace,
live RGB-D observations and numerical robot tools. Implementation source, private evaluation,
and credentials were blocked. The protocol records the explicitly supplied memory and examples.
Existing geometry, grasp and planning implementations remain inherited tools.</p>
<p>{html.escape(scope)}. All attempts remain in the denominator.
The native task predicate was evaluated outside the agent. Separate API calls: 0.</p>
{''.join(cards)}
<article><h2>Agent-authored findings</h2><pre>{html.escape(findings)}</pre></article>
<article><h2>Agent-authored guidelines (claims, not independently established rules)</h2><pre>{html.escape(guidelines)}</pre></article>
<article><h2>Verified boundaries and execution</h2><pre>{html.escape(json.dumps(audit,indent=2))}</pre></article>
<article><h2>Tool trace</h2><table><thead><tr><th>#</th><th>Tool</th><th>Outcome</th><th>Time</th><th>Evidence</th></tr></thead><tbody>{''.join(trace)}</tbody></table></article>
'''
    (report/'index.html').write_text(body)
    (report/'summary.json').write_text(json.dumps({'successful_attempts':successful,'attempts':episodes,'audit':audit},indent=2))
    print(json.dumps({'report':str(report/'index.html'),'successes':successful,'attempts':len(episodes),'audit':audit},indent=2))


if __name__=='__main__':main()
