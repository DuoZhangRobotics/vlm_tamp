"""Plain-language experiment map and the matched shared-tool transfer rerun."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import html
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/shared-tools-transfer-report-20261020'
RUN=ROOT/'results/shared-tools-transfer-20261020'

# Keep old evidence paths stable. These labels explain the actual question.
CATALOG=[
('M1','Which contact constraint permits a long motion?','grasp-horizon-physical-20261020',
 'One microwave state; six conditions: old short baseline, synchronized short/long point contact, rigid short/long, and finger-axis long.',
 'The conditions differ; the aggregate is an inventory, not one policy score.'),
('M2','Do the short and long motions repeat?','grasp-horizon-repeatability-20261020',
 'Two additional repeats each of synchronized 15° and 90° point-contact motions.',
 'With M1, each condition succeeded 3/3 on the same development state.'),
('B1','How does the scripted baseline perform?','success-first-family-baseline-20261020',
 'One initial state each for bowl placement, middle/top drawer opening, top drawer closing, and stove activation.',
 'Middle-drawer opening failed.'),
('B2','Can it always plan the full remaining motion?','success-first-family-long-20261020',
 'Four matched articulated cases with full remaining-motion planning.',
 'Longer requests alone did not solve the problem.'),
('B3','Does screening grasps for the full motion help?','success-first-full-grasp-screen-20261020',
 'Two difficult cases using full-motion grasp admission with the original candidate sample.',
 'Both failed; these results remain in the record.'),
('B4','Does a different grasp sample change the drawer result?','middle-drawer-grasp-seed1-20261020',
 'Middle-drawer development case with another recorded grasp seed and full-motion screening.',
 'A sampling diagnostic, not generalization.'),
('B5','Does full-motion preference with a short fallback work?','adaptive-family-seed1-20261020',
 'Five task-family cases with the shared adaptive horizon policy.',
 'Bowl placement remained blocked by the held-object clearance reserve.'),
('B6','Is the extra held-object clearance reserve the blocker?','adaptive-family-payload-boundary-20261020',
 'Matched five-case rerun allowing nonpenetration at the held-object boundary.',
 'True penetration and unrelated-obstacle checks remain.'),
('B7','Does that policy cover seven tasks and five states?','adaptive-all-tasks-five-state-20261020',
 'Seven tasks × official initial states 1, 11, 21, 31 and 41.',
 'Bowl states 21/31 and middle-drawer state 21 failed.'),
('B8','Can an already-admitted plan recover two failures?','preplanned-state21-comparison-20261020',
 'Bowl and middle-drawer state 21; execute the manipulation admitted during grasp selection.',
 'Initial arrays and all approach/closure commands match the failed counterparts.'),
('B9','Does prepared execution repeat on states 1/11/21?','preplanned-three-state-confirmation-20261020',
 'Five families × three initial states.',
 'Development confirmation; repeated cases.'),
('B10','Does it also cover states 31/41?','preplanned-remaining-states-20261020',
 'The same five families × the remaining two states.',
 'Development confirmation; repeated cases.'),
('B11','Final scripted-baseline regression','prepared-final-all-tasks-20261020',
 'Seven tasks × five states with one frozen configuration.',
 '35/35 applies to scripted orchestration with established family adapters and matched predictor inputs. Microwave cases still use multiple short planned segments.'),
]


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def href(path,label):
    return f'<a href="{html.escape(os.path.relpath(path,OUT),quote=True)}">{html.escape(str(label))}</a>'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-videos',action='store_true')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    current=read(RUN/'verified-summary.json',read(RUN/'summary.json',{}))
    old=ROOT/'results/frozen-prepared-method-transfer-20261020'
    old_rows={r['state']:r for r in read(old/'summary.json',{}).get('cases',[])}
    baseline={r['initial_state']:r for r in read(RUN/'scripted/summary.json',{}).get('episodes',[])}
    actor={r['state']:r for r in current.get('agent_cases',[])}
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shared tools: scripted baseline and Luna transfer</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 22px;color:#213743;background:#f6f8fb}
h1,h2{line-height:1.2}h2{margin-top:34px}a{color:#126d96}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}
th,td{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #dce4ed}th{background:#e9f0f6}
.note{padding:15px 20px;background:#e8f4fa;border-left:4px solid #337c9e}.scope{background:#fff3e2;border-left-color:#b58038}
.small{font-size:14px;color:#526876}details{background:white;border:1px solid #dce4ed;padding:14px;margin:16px 0}summary{cursor:pointer;font-weight:600}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}article{background:white;padding:14px}video{width:100%}
code{background:#e8edf4;padding:2px 4px}</style><body><h1>Shared tools, two ways to choose actions</h1>'''
    stamp=datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M %Z')
    page+=f'<p class="small">Updated {stamp}. Old folder names remain unchanged to preserve links and evidence.</p>'
    page+='''<p class="note"><strong>“Numerical pipeline” means the scripted baseline.</strong> Human-written code routes the task to a family adapter, uses previously generated visual grounding when the observed input matches, and chooses the next planning/execution call. Luna is the alternative orchestrator: it views observations, chooses regions, calls tools and decides what to do after feedback. Both ultimately move the same simulated Panda using numerical planners and the same low-level servo.</p>
<p>The earlier 35/35 result measures the scripted baseline, including its multiple short microwave segments. It is not a 35-case single-trajectory result. The earlier 1/4 transfer result measures Luna using its learned method with fresh grounding. They also had different tool implementations, so the gap could not be attributed solely to the caller.</p>
<h2>What was inconsistent, and what is shared now?</h2>
<p>The scripted runner and agent service had separate imports and copied code. Improvements to the numerical runner were not propagated to the older agent service. Consolidation replaces those parallel implementations with one shared module.</p>
<table><tr><th>Component</th><th>Earlier difference</th><th>Shared implementation now</th></tr>
<tr><td>Observed handle</td><td>Agent used the older cylinder fit and height quantiles.</td><td>Refined observed bar and finite configured finger-pad support.</td></tr>
<tr><td>IK search</td><td>Agent requested six solutions and inspected three; scripted runner inspected eight.</td><td>Same cuRobo solver, isolated seven-joint kinematics, and eight-solution search.</td></tr>
<tr><td>Arm collision</td><td>Agent used native arm spheres; scripted runner used calibrated convex arm links.</td><td>Same calibrated arm and gripper geometry against observed point clouds.</td></tr>
<tr><td>Contact permissions</td><td>Local mount association and existing shallow-contact handling differed.</td><td>Existing scripted-baseline permissions, including bounded submillimetre existing palm/contact tolerance. No wholesale palm or obstacle exemption.</td></tr>
<tr><td>Search limits</td><td>Agent had a smaller grasp-search budget.</td><td>Same tool defaults and 600-second grasp-search setting; checked between candidate groups, so an in-progress group can overrun it. Identical arc search limits.</td></tr>
<tr><td>Motion planner</td><td>Separate wrappers selected different native collision backends.</td><td>One MC-VAMP backend and continuous partial-pose arc routine, with point, rigid-pose and finger-axis choices.</td></tr>
<tr><td>Execution</td><td>Caller-selected speeds, holds and gripper commands differ.</td><td>Same trajectory scheduler and joint-position execution primitives. These explicit caller choices remain visible in logs.</td></tr></table>
<p class="note scope">No new task/state-specific constants, remembered coordinates, preferred grasp poses, or recovery strategy were added. Existing upright-panel/vertical-handle assumptions and analytic contact proposals remain inherited priors. Sharing tools does not make those priors general to every articulated object.</p>
<p>Inputs and policy choices still differ: the scripted regression preserves its input-validated cached grounding and seed-1 GraspGen proposals; Luna selects current regions and requests its own grasp samples (default seed 0). Both use the same hybrid grasp generator, whose analytic observed-handle candidates can be selected ahead of GraspGen. This comparison tests shared implementations under two callers, not identical complete requests or a causal estimate of reasoning quality.</p>
<h2>Current four-state comparison</h2>
<p>Same microwave task and states 1, 11, 21, 31. Luna uses the exact registered task/strategy prompt, abstract findings and relative parameters from the original transfer study. The tool documentation and feedback describe the current implementation, including the read-only scene diagnostics added after that study. One attempt per state, no retries and no findings shared across cases. This is a new four-attempt comparison, separate from the earlier twenty attempts. Native success stops each episode immediately; there is no post-success retreat or execution-time collision veto.</p>'''
    page+='<table><tr><th>State</th><th>Earlier Luna transfer</th><th>Shared-tool scripted baseline</th><th>Shared-tool Luna transfer</th></tr>'
    for state in [1,11,21,31]:
        cells=[]
        for row,field in [(old_rows.get(state),'native_success'),(baseline.get(state),'official_task_success'),(actor.get(state),'native_success')]:
            cells.append('Pending' if row is None else ('Success' if row[field] else 'Failure'))
        page+=f'<tr><td>{state}</td>'+''.join(f'<td>{c}</td>' for c in cells)+'</tr>'
    page+='</table>'
    if current:
        page+=f'<p><strong>Scripted baseline: {current.get("scripted_native_successes",0)}/4. Luna: {current.get("agent_native_successes",0)}/4</strong> ({current.get("completed_agent_cases",0)} agent cases completed). Pending cases remain in the planned denominator.</p>'
    if current.get('verification',{}).get('corrections'):
        page+='<p class="note">State 31 reached native success at 335 steps. A malformed rejected tool request crashed postprocessing, so the original runner aggregate incorrectly marked its missing report as failure. The verified aggregate uses the agreeing native action log and evaluation record. The original aggregate and error logs remain unchanged; no physical episode was rerun.</p>'
        page+='<p>'+href(RUN/'verified-summary.json','Verified native outcomes')+' · '+href(RUN/'summary.json','Original runner aggregate with reporting defect')+'</p>'
    page+='<p>'+href(RUN/'protocol.json','Frozen protocol')+' · '+href(OUT/'preflight/parity.json','Identical-input geometry check')+'</p>'
    if (RUN/'tool-parity.json').exists():page+='<p>'+href(RUN/'tool-parity.json','All eight run contracts match')+'</p>'
    diagnosis=read(OUT/'diagnosis.json',{})
    if diagnosis:
        page+='<h2>Rerun findings</h2>'
        for sentence in diagnosis.get('findings',[]):page+='<p>'+html.escape(sentence)+'</p>'
        audit=read(OUT/'transfer-evidence-audit.json',{}).get('cases',[])
        page+='<table><tr><th>Luna state</th><th>Native result</th><th>Steps</th><th>Tool calls</th><th>Final door opening (evaluation only)</th></tr>'
        for row in audit:
            angle=row['evaluation_only_final_door_angle_deg']
            page+=f'<tr><td>{row["state"]}</td><td>{"Success" if row["native_success"] else "Failure"}</td><td>{row["control_steps"]}</td><td>{sum(row["tool_call_counts"].values())}</td><td>{angle:.1f}°</td></tr>'
        page+='</table><p>'+href(OUT/'transfer-evidence-audit.json','Recorded commands, physical progress and failed queries')+'</p>'
    page+='''<h2>How Luna practiced</h2>
<p>Luna ran as a subscription-backed agent with high reasoning in an isolated workspace. It could inspect RGB-D, proprioception and tool output, call DINO optionally, request grasps and plans, execute them, and write notes. It could not read the implementation, private simulator object state or our diagnostic solutions. The tools themselves retained the stated geometry/contact priors.</p>
<p>Practice began with five granted attempts and could extend adaptively. The agent had to establish two consecutive native successes before optimizing calls and time. The first session completed seven attempts (5/7 successes); its last two had a neutral-gripper scene-update bug. After that tool bug was fixed, the second session completed nine attempts (8/9), testing 70°, 90°, 80° and 85° prepared motions. The 90° attempt failed. The final 85° recipe succeeded twice locally, reducing planning calls from 7–8 to 1 while increasing control steps from 250 to 283.</p>
<p>The remaining four original attempts tested frozen transfer (1/4). Practice changed notes and relative tool choices, not neural-network weights. Memory stores principles, observed outcomes, limitations and relative parameters; it excludes past world coordinates, image regions and trajectories.</p>
<h2>All earlier physical experiments</h2>
<p>M = motion-component study; B = scripted-baseline development; L = Luna study. Rows overlap in scenes and settings. Do not combine their success rates into one benchmark score.</p>
<table><tr><th>ID / question</th><th>What ran</th><th>Native outcomes</th><th>Interpretation</th></tr>'''
    ledger=[]
    for code,title,name,question,scope in CATALOG:
        folder=ROOT/'results'/name;data=read(folder/'summary.json',{});episodes=data.get('episodes',[])
        successes=sum(bool(r.get('official_task_success')) for r in episodes)
        page+=f'<tr><td><strong>{code}</strong> {href(folder/"summary.json",title)}</td><td>{html.escape(question)}</td><td>{successes}/{len(episodes)}</td><td>{html.escape(scope)}</td></tr>'
        ledger.append({'id':code,'label':title,'folder':name,'successes':successes,'episodes':len(episodes),'scope':scope})
    for code,title,name,successes,total,question in [
        ('S1','Prepared-sequence tool smoke test','robot-sequence-api-smoke-20261020',1,1,'Matched RGB input; fresh grasp and trajectory through the public API.'),
        ('L1','Initial Luna practice','subscription-scene-sync-practice-20261020',5,7,'Agent exploration; neutral-hold update bug confounds the last two efficiency results.'),
        ('L2','Corrected Luna practice','subscription-prepared-sequence-practice-20261020',8,9,'Success first, then fewer calls; final local recipe repeated twice.'),
        ('L3','Frozen Luna transfer','frozen-prepared-method-transfer-20261020',1,4,'Fresh grounding on four other initial states; one attempt each.')]:
        folder=ROOT/'results'/name
        page+=f'<tr><td><strong>{code}</strong> {href(folder,title)}</td><td>{html.escape(question)}</td><td>{successes}/{total}</td><td>Separate development or transfer cohort.</td></tr>'
        ledger.append({'id':code,'label':title,'folder':name,'successes':successes,'episodes':total})
    page+='''</table><p>These are 130 scripted/component episodes and 20 agent attempts, including repeated scenes and failed attempts. Additional work used saved evidence only: physical contact/orientation audits, paired-input checks, model-versus-depth checks, tool-parity audits and unit tests. Those are not additional physical successes.</p>
<h2>Rerun videos</h2><p class="small">20 Hz control-time playback when exported; planning and model pauses are excluded. Original evidence remains available.</p><div class="gallery">'''
    for state in [1,11,21,31]:
        for label,path in [
            ('Scripted baseline',RUN/'scripted'/f'open_microwave-state{state:03d}'/'initial/rollout.mp4'),
            ('Luna',RUN/'luna'/f'state{state:03d}'/'attempt01/initial/rollout.mp4')]:
            if not path.exists():continue
            relative=path.relative_to(ROOT/'results')
            display=OUT/'media'/'--'.join(relative.parts)
            if args.export_videos and not display.exists():
                import export_control_time_videos as exporter
                exporter.OUTPUT=OUT/'media';exporter.OUTPUT.mkdir(exist_ok=True)
                exporter.export(str(relative))
            if not display.exists():continue
            page+=f'<article><h3>{label} · state {state}</h3><video controls preload="metadata" src="{html.escape(os.path.relpath(display,OUT))}"></video><p>{href(path,"Original")}</p></article>'
    page+='</div>'
    validation=read(OUT/'validation.json',{})
    if validation:
        page+='<h2>Validation and cleanup</h2>'
        page+='<p>'+html.escape(validation['summary'])+'</p>'
        if 'storage_summary' in validation:page+='<p>'+html.escape(validation['storage_summary'])+'</p>'
    page+='<p>'+href(ROOT/'results/overnight-long-horizon-report-20261020/index.html','Earlier detailed report')+' · '+href(ROOT/'docs/shared_tools_comparison_20261020.md','Implementation and comparison notes')+'</p></body></html>'
    (OUT/'index.html').write_text(page)
    (OUT/'experiment-catalog.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps({'report':str(OUT/'index.html'),'earlier_physical_cohorts':len(ledger)}))


if __name__=='__main__':
    main()
