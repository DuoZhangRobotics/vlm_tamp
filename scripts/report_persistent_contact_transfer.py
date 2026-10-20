"""Explain the persistent-episode comparison and retain every planned case."""
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import html
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.tool_event_audit import operation_name

RUN=ROOT/'results/persistent-contact-transfer-v2-20261020'
OUT=ROOT/'results/persistent-contact-transfer-report-20261020'


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def link(path,text):
    return f'<a href="{html.escape(os.path.relpath(path,OUT),quote=True)}">{html.escape(str(text))}</a>'


def main():
    OUT.mkdir(exist_ok=True)
    summary=read(RUN/'summary.json',{});rows={r['state']:r for r in summary.get('cases',[])}
    old={r['state']:r for r in read(ROOT/'results/shared-tools-transfer-20261020/verified-summary.json',{}).get('agent_cases',[])}
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Persistent Luna transfer: success or budget exhaustion</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 22px;color:#243743;background:#f7f9fc}
h1,h2{line-height:1.2}h2{margin-top:30px}a{color:#126a95}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}
th,td{padding:10px;text-align:left;vertical-align:top;border-bottom:1px solid #dce4ec}th{background:#edf2f6}
.note{padding:15px 20px;border-left:4px solid #367c9e;background:#e8f3f9}.scope{background:#fff3e2;border-left-color:#b27e32}
.small{font-size:14px;color:#526677}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}article{background:white;padding:14px}video{width:100%}</style><body>
<h1>Luna continues until success or an actual budget limit</h1>'''
    stamp=datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M %Z')
    page+=f'<p class="small">Updated {stamp}.</p>'
    page+='''<p class="note">Each case has one physical episode, 600 control steps and the same 30-minute agent/tool budget. Incomplete finish requests are rejected. If the agent ends a reasoning turn early, the supervisor resumes the same thread and physical state with a budget reminder. There is no reset retry or memory transfer between cases.</p>
<p>At the wall-time limit, no new requests are started. An already-issued request may finish, so recorded wall time can include that completion and cleanup. Simulator success stops motion immediately. Runtime/account errors remain separate infrastructure outcomes; they are not called budget exhaustion.</p>
<h2>General manipulation choices</h2>
<p>The prompt explains that manipulation may use multiple stages and contacts: grasp, release, regrasp, push, pull or slide. It supplies no specific surface, direction, pose or recovery sequence. The earlier agent-authored method is unchanged and is a hypothesis rather than a mandatory recipe.</p>
<p>An initial setup attempt exposed a transport problem: status queries and duplicate plans could queue behind a long request. It was interrupted at 60 setup steps before any manipulation. Its evidence and original four-slot manifest remain in the archived initial run. The comparison below uses a fresh frozen cohort after adding single-request admission, nonblocking status snapshots and retrieval of existing replies.</p>
<p>A new generic surface-contact tool selects a bounded patch from current RGB-D. Configured fingers may contact that patch; arm, palm and other observed surfaces remain checked. A pose planner exposes IK and free-motion planning without mandatory closure or a handle/cylinder fit. Luna chooses the geometry and actions. The scene must be refreshed after physical motion; no object attachment or motion prediction is assumed in this mode.</p>
<p class="scope note">Persistence, the general prompt and generic contact capability change together. This is a bundled rerun, not an isolated ablation. Existing perception, robot and contact priors remain. Same microwave asset and initial states; no new-asset generalization claim.</p>
<h2>Same four transfer states</h2><table><tr><th>State</th><th>Previous result</th><th>Current native result</th><th>Steps</th><th>Agent turns</th><th>Termination</th></tr>'''
    audits=[]
    for state in [1,11,21,31]:
        row=rows.get(state);folder=RUN/f'state{state:03d}'
        previous='Success' if old.get(state,{}).get('native_success') else 'Failure'
        if row:
            page+=f'<tr><td>{link(folder/"remembered-agent-result.json",state)}</td><td>{previous}</td><td>{"Success" if row["native_success"] else "Failure"}</td><td>{row["control_steps"]}</td><td>{row["agent_turns"]}</td><td>{html.escape(str(row["termination_reason"]))}</td></tr>'
        else:
            live=read(folder/'live-episode-status.json',{})
            page+=f'<tr><td>{state}</td><td>{previous}</td><td>Running / pending</td><td>{live.get("control_steps","—")}</td><td>—</td><td>—</td></tr>'
        events_path=folder/'tool-events.jsonl'
        events=[json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        counts=Counter(operation_name(event) for event in events)
        denied=sum(operation_name(e)=='finish' and e['response'].get('result',{}).get('must_continue',False) for e in events)
        audits.append({'state':state,'tool_counts':dict(counts),'early_finish_requests_rejected':denied,
            'continuation':read(folder/'continuation.json',{}),
            'surface_contact_selections':counts['select_surface_contact'],'pose_plan_requests':counts['plan_pose']})
    page+='</table>'
    page+=f'<p><strong>{summary.get("native_successes",0)}/4 native successes; {summary.get("completed",0)}/4 cases completed.</strong> Every planned case remains in the denominator.</p>'
    page+='<p>'+link(ROOT/'prompts/persistent_transfer.txt','Exact general task prompt')+' · '+link(RUN/'protocol.json','Frozen experiment protocol')+' · '+link(OUT/'transport-smoke/result.json','Same-thread continuation smoke test')+'</p>'
    page+='<p>'+link(ROOT/'results/persistent-contact-transfer-20261020/transport-interruption.json','Preserved transport-interruption record')+'</p>'
    page+='<h2>Continuation and contact-tool use</h2><table><tr><th>State</th><th>Early finish requests rejected</th><th>Surface selections</th><th>Pose-plan requests</th></tr>'
    for r in audits:
        page+=f'<tr><td>{r["state"]}</td><td>{r["early_finish_requests_rejected"]}</td><td>{r["surface_contact_selections"]}</td><td>{r["pose_plan_requests"]}</td></tr>'
    page+='</table><p>Tool invocation is not evidence of physical contact or task success. See the recorded native outcome and command trace.</p>'
    final=read(OUT/'final-audit.json',{})
    for text in final.get('findings',[]):page+='<p>'+html.escape(text)+'</p>'
    page+='<h2>Videos</h2><div class="gallery">'
    for state in [1,11,21,31]:
        movie=OUT/'media'/f'state{state:03d}.mp4'
        if movie.exists():page+=f'<article><h3>State {state}</h3><video controls preload="metadata" src="{os.path.relpath(movie,OUT)}"></video></article>'
    page+='</div><p class="small">Display videos use 20 Hz control time; reasoning and planning pauses are excluded. Originals remain in the run folders.</p>'
    if final:page+='<p>'+link(OUT/'final-audit.json','Outcome, persistence, cleanup and validation audit')+'</p>'
    page+='<p>'+link(ROOT/'results/shared-tools-transfer-report-20261020/index.html','Previous shared-tool comparison and full experiment map')+'</p></body></html>'
    (OUT/'index.html').write_text(page)
    (OUT/'tool-use-audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    print(json.dumps({'report':str(OUT/'index.html'),'completed':summary.get('completed',0),
                      'native_successes':summary.get('native_successes',0)}))


if __name__=='__main__':
    main()
