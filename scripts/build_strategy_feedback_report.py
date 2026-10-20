"""Compact evidence report; preserve development failures and every planned case."""
import argparse
import html
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.run_api_budget import RunLedger,snapshot
from validate_strategy_feedback import verify_run
from build_bottom_drawer_investigation_report import CSS


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--development',nargs='*',default=[]);p.add_argument('--output',required=True)
    p.add_argument('--followup',nargs='*',default=[])
    p.add_argument('--api-run-id',required=True);a=p.parse_args();out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    rows=[];development=[];followup=[];checks=[];planned=[]
    for kind,runs in [('evaluation',a.runs),('development',a.development),('followup',a.followup)]:
        for run in runs:
            verified=verify_run(ROOT/run);checks.append({'run':run,'kind':kind,**verified})
            for row in verified['episodes']:
                (rows if kind=='evaluation' else followup if kind=='followup' else development).append({**row,'study_group':kind})
            planned.extend({'kind':kind,**entry} for entry in read_json(ROOT/run/'manifest.json')['episodes'])
    budget=snapshot(RunLedger(a.api_run_id))
    counts={policy:{'planned':sum(r['policy']==policy and r['kind']=='evaluation' for r in planned),
        'completed':sum(r['policy']==policy for r in rows),
        'successes':sum(r['success'] for r in rows if r['policy']==policy)} for policy in {r['policy'] for r in planned if r['kind']=='evaluation'}}
    write_json(out/'evaluation.json',{'by_policy':counts,'episodes':rows,'development':development,'followup':followup,'api':budget,
        'checks':checks,'all_planned_episodes':planned,'candidate_promoted':False,'historical_35_case_comparison_run':False})
    def link(path):return html.escape(os.path.relpath(path,out),quote=True)
    def table(items):
        content=[]
        for row in items:
            folder=Path(row['folder'])
            content.append(f'<tr><td>{html.escape(row["task"])} · {row["state"]}</td><td>{row["policy"]}</td>'
                f'<td>{"Success" if row["success"] else "Failure"}</td><td>{row["control_steps"]}</td><td>{len(row["decisions"])}</td>'
                f'<td>{row["seconds"]:.1f}</td><td>{html.escape(row.get("failure") or "")}</td>'
                f'<td><a href="{link(folder/"initial/rollout.mp4")}">Video</a> · <a href="{link(folder/"summary.json")}">Evidence</a></td></tr>')
        return '<div class="scroll"><table><thead><tr><th>Task/state</th><th>Decision policy</th><th>Native outcome</th><th>Control steps</th><th>Decisions</th><th>Seconds</th><th>Remaining failure</th><th>Evidence</th></tr></thead><tbody>'+''.join(content)+'</tbody></table></div>'
    cards=[]
    for row in rows+followup+development:
        if row['policy']!='vlm':continue
        rounds=[]
        for d in row['decisions']:
            choice=d['choice'];folder=Path(d['folder']);result=d['result'] or {}
            rounds.append(f'<details><summary>{d["round"]}: {choice["operation"]} / {choice["contact_mode"]} / {choice["approach_tilt_deg"]}° → {result.get("status","interrupted")}</summary>'
                f'<p><b>Model evidence:</b> {html.escape(choice["evidence"])}</p><p><b>Expected:</b> {html.escape(choice["expected_observation"])}</p>'
                f'<img loading="lazy" src="{link(folder/"feedback-sheet.png")}" alt="The exact visual feedback sent to Luna">'
                f'<p><a href="{link(folder/"feedback.json")}">Input feedback</a> · <a href="{link(folder/"prompt.txt")}">Exact prompt</a> · '
                f'<a href="{link(folder/"response.json")}">Raw response and usage</a> · <a href="{link(folder/"tool-result.json")}">Tool result</a></p></details>')
        cards.append(f'<section><h3>{html.escape(row["task"])} · state {row["state"]} · {row["study_group"]}</h3>'+''.join(rounds)+'</section>')
    repairs=[]
    for row in followup:
        folder=Path(row['folder'])/'grounding/validation-repair'
        if not (folder/'verification.json').exists():continue
        evidence=read_json(folder/'verification.json')
        repairs.append(f'<p>{html.escape(row["task"])} · {row["state"]} · {row["policy"]}: '
            f'{html.escape(evidence["previous_validation_error"])}. The corrected response passed schema and depth-support validation. '
            f'<a href="{link(folder/"prompt.txt")}">Exact validation feedback</a> · '
            f'<a href="{link(folder/"response.txt")}">Model correction</a> · '
            f'<a href="{link(folder/"verification.json")}">Provenance</a>.</p>')
    stats=''.join(f'<div class="stat"><strong>{v["successes"]}/{v["planned"]}</strong>{name}; {v["completed"]} completed</div>' for name,v in sorted(counts.items()))
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VLM strategy feedback pilot</title><style>{CSS}</style></head><body><main>
<div class="eyebrow">Experimental pilot · 18 October 2026</div><h1>Can Luna select and revise contact strategies?</h1>
<p class="lead">Luna receives RGB observations, sensor-derived geometry, planner diagnostics and observed movement. It chooses from a fixed tool menu. Every decision and its exact evidence are retained below.</p>
<p class="notice">This is a drawer strategy-selection pilot. The original successful task-family pipeline remains the accepted reference. This experiment does not rerun or replace the 35-case reference, establish broad generalization, or demonstrate discovery of new physical primitives.</p>
<div class="stats">{stats}</div>
<h2>What the VLM controls</h2><p>One contact mode (pinch, press-and-drag or push), an optional part-relative approach tilt, a probe or goal motion, reobservation, or finishing. The selected contact tool cannot silently switch mode. Five decision rounds share one 600-step physical episode; there are no simulator resets between attempts.</p>
<h2>What remains engineered</h2><p>The common backend retains depth fitting, an upright-panel prior, GraspGen proposals, contact templates, estimated travel/completion rules and fixed reactive control. Both policies use the same planner, collision/contact permissions and motion budgets. The heuristic control retains the prior clearance/motion mode ordering; optional enumerated search uses a fixed sequence. The VLM sees no benchmark state indices, native object geometry, true joint axes or success flags.</p>
<p>Luna high reasoning, 32,000 generated-token ceiling for strategy calls and 12,000 for new grounding calls. One synthetic tool-use example teaches observation after an IK failure; it contains no drawer strategy. API failures and incomplete responses are not automatically retried. All issued trajectories complete; planning failures return feedback instead of ending the episode immediately.</p>
<p><b>Shared experiment spending:</b> ${budget['run_conservative_spend_usd']:.6f} of $2, including development and all comparison policies. Uncertain requests retain their reservation. Runtime includes API waiting; input caches are explicitly recorded.</p>
<p>Per-episode times exclude the shared model-loading phase. Initial semantic grounding is reused only after RGB/depth validation; previously uncached states run fresh vision and grounding. The simulator pauses between control requests, so these timings do not establish real-time operation in a changing physical scene.</p>
<h2>Frozen evaluation</h2>{table(rows)}
<p>In the known opening case, both policies selected press-and-drag at 45 degrees but obtained different redundant-arm IK solutions and grouped motion differently. These are whole-pipeline outcomes; the small comparison does not isolate VLM reasoning from numerical search sensitivity. <a href="comparison-notes.json">Detailed comparison notes</a>.</p>
<h2>Separate grounding-validation follow-up</h2><p>The original ten-episode comparison remains above. Its invalid initial grounding response never reached the strategy loop. The subsequent follow-up permits one model correction using the original image and validator error, without a manual pixel transformation. It is evaluated separately under the same shared API budget.</p>{''.join(repairs)}{table(followup)}
<h2>VLM decisions and exact feedback</h2>{''.join(cards)}
<h2>Development history</h2><p>Initialization errors and failed pilots remain visible. No task-specific contact rule is added after observing a frozen evaluation result.</p>{table(development)}
<h2>Verification and retention</h2><p>Native LIBERO success is read after execution. The verifier compares every issued joint command with the action log, checks the feedback boundary and selected mode, verifies the movies and frozen source snapshots, and audits paired inputs. Temporary frames and intermediate observation archives are removed after each episode; necessary feedback images, trajectories and outcomes remain.</p>
<p><a href="evaluation.json">Complete results, budgets and input-pairing checks</a> · <a href="verification-summary.json">Final verification</a> · <a href="storage-summary.json">Storage cleanup</a> · <a href="api-usage.json">Actual API usage</a> · <a href="../../configs/prompts/strategy_feedback_v1.txt">Shared prompt</a> · <a href="../../scripts/run_strategy_feedback.py">Runner</a> · <a href="../shared-contact-feedback-report/index.html">Earlier controller comparison</a> · <a href="../success-first-report/index.html">Original successful reference</a></p>
</main></body></html>'''
    (out/'index.html').write_text(text)
    print({'report':str(out/'index.html'),'counts':counts,'run_spend_usd':budget['run_conservative_spend_usd']})


if __name__=='__main__':main()
