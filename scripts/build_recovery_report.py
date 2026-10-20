"""Evidence report for reproducible IK and the frozen recovery tool comparison."""
import html
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from validate_strategy_feedback import verify_run
from build_bottom_drawer_investigation_report import CSS
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.run_api_budget import RunLedger,snapshot

RUN=ROOT/'results/vlm-recovery-frozen-20261018'
PILOT=ROOT/'results/vlm-recovery-ik-only-20261018'
OUT=ROOT/'results/vlm-recovery-tools-report'

def main():
    OUT.mkdir(exist_ok=True)
    current=verify_run(RUN);pilot=verify_run(PILOT)
    smoke_path=ROOT/'results/vlm-recovery-tool-smoke-20261018'
    smoke=verify_run(smoke_path) if smoke_path.exists() else None
    if current['pending'] or pilot['pending']:raise ValueError('A final report requires every planned episode')
    if not current['all_pairs_identical'] or not pilot['all_pairs_identical']:raise ValueError('Paired input mismatch')
    audit=read_json(ROOT/'results/vlm-recovery-ik-audit-v6-20261018/audit.json')
    assert audit['passed']
    budget=snapshot(RunLedger('vlm-recovery-tools-20261018'))
    counts={p:{'successes':sum(r['success'] for r in current['episodes'] if r['policy']==p),
               'planned':sum(r['policy']==p for r in read_json(RUN/'manifest.json')['episodes'])}
            for p in ['heuristic','vlm']}
    rows=current['episodes'];old=read_json(ROOT/'results/vlm-strategy-frozen-20261018/summary.json')
    evidence={'by_policy':counts,'ik_only':pilot,'frozen':current,'ik_audit':audit,
              'api':budget,'promoted':False,'historical_reference_rerun':False,
              'original_strategy_results':old['by_policy'],'scripted_tool_smoke':smoke}
    write_json(OUT/'evaluation.json',evidence)
    def link(path):return html.escape(os.path.relpath(path,OUT),quote=True)
    def table(items):
        lines=[]
        for r in items:
            end=next((d['result'] for d in reversed(r['decisions']) if d['result'] and d['result'].get('status')=='planning_failed'),{})
            explanation='' if r['success'] else str(end.get('failure_loci') or end.get('reason') or r['failure'])
            f=Path(r['folder'])
            lines.append(f'<tr><td>{html.escape(r["task"])} · {r["state"]}</td><td>{r["policy"]}</td>'
                f'<td>{"Success" if r["success"] else "Failure"}</td><td>{r["control_steps"]}</td>'
                f'<td>{r["seconds"]:.1f}</td><td>{html.escape(explanation)}</td>'
                f'<td><a href="{link(f/"initial/rollout.mp4")}">Video</a> · <a href="{link(f/"summary.json")}">Record</a></td></tr>')
        return '<div class="scroll"><table><tr><th>Task / official state</th><th>Policy</th><th>Native result</th><th>Steps</th><th>Seconds</th><th>Last planning failure</th><th>Evidence</th></tr>'+''.join(lines)+'</table></div>'
    cards=[]
    for r in rows:
        if r['policy']!='vlm':continue
        rounds=[]
        for d in r['decisions']:
            c=d['choice'];f=Path(d['folder']);res=d['result'] or {}
            rounds.append(f'<details><summary>{d["round"]}: {html.escape(c["operation"])}; {html.escape(str(c["contact_mode"]))}; '
                f'{c.get("step_mm")} mm; IK rank {c.get("posture_rank")} → {html.escape(res.get("status","interrupted"))}</summary>'
                f'<p>{html.escape(c["evidence"])}</p><p>Expected: {html.escape(c["expected_observation"])}</p>'
                f'<img loading="lazy" src="{link(f/"feedback-sheet.png")}" alt="Actual four-panel RGB image sent to Luna">'
                f'<pre>{html.escape(json.dumps(res,indent=2))}</pre><p>'
                f'<a href="{link(f/"prompt.txt")}">Exact prompt and history</a> · '
                f'<a href="{link(f/"response.json")}">API response and tokens</a> · '
                f'<a href="{link(f/"feedback.json")}">Structured input</a></p></details>')
        cards.append(f'<section><h3>{html.escape(r["task"])} · {r["state"]}</h3>'
            f'<video controls preload="metadata" src="{link(Path(r["folder"])/"initial/rollout.mp4")}"></video>'+''.join(rounds)+'</section>')
    commands='''PYTHONPATH=src:scripts:.runtime/mc-vamp-deps MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .runtime/envs/rekep-libero/bin/python scripts/run_strategy_feedback.py --output results/NEW_OUTPUT --api-run-id YOUR_EXPERIMENT_ID --cases open_bottom_drawer:15 close_bottom_drawer:15 close_top_drawer:1 open_bottom_drawer:17 close_bottom_drawer:17 --policies heuristic vlm --reproducible-ik --tool-version v2 --allow-paid'''
    stats=''.join(f'<div class="stat"><strong>{v["successes"]}/{v["planned"]}</strong>{p}</div>' for p,v in counts.items())
    smoke_html=''
    if smoke:
        item=smoke['episodes'][0];history=item['decisions'];folder=Path(item['folder'])
        assert history[0]['result']['status']=='motion_completed'
        assert history[1]['result']['release_executed'] and history[1]['result']['retreat_executed']
        assert history[2]['result']['status']=='motion_completed'
        assert history[2]['result']['selected_strategy']['actual_ik_rank']==1
        smoke_html=f'''<h2>Separate physical tool interface check: passed</h2><p>A predeclared scripted sequence established contact, released, completed a planned 20 mm retreat and recontacted with IK rank 1. All eight issued trajectories completed; no API calls were needed. The two probes were only 5 mm each and the script then finished, so it did not attempt a full opening and is excluded from policy success rates. <a href="{link(folder/'initial/rollout.mp4')}">Video</a> · <a href="{link(folder/'strategy-history.json')}">Exact tool results</a>.</p>'''
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VLM recovery tools</title><style>{CSS}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:16px;border-radius:8px}}video{{width:100%;max-height:560px}}img{{max-width:100%}}</style></head><body><main>
<div class="eyebrow">Frozen experiment · 18 October 2026</div><h1>Reproducible IK and VLM recovery tools</h1>
<p class="lead">RGB observations and numerical feedback go to Luna; Luna chooses a bounded tool; local planning and control execute it. The original successful reference remains preserved.</p>
<div class="stats">{stats}</div><p>Every planned case counts, including failures before motion. Success means LIBERO's native predicate became true; final success is also retained. Five cases per policy are a diagnostic comparison, not a reliability or generalization estimate.</p>
<h2>What changed</h2><p>IK now uses all 64 explicitly specified seeds, a fixed robot-only warmup and a fresh cuRobo optimizer per query. CUDA graph capture is disabled in this optional correctness mode. Stable candidate ordering precedes duplicate removal. The legacy mode remains available. Repeated identical requests, interleaved requests and fresh processes produced identical candidate arrays in the GPU audit.</p>
<p>The v2 interface adds actual 5/10/25 mm travel choices, optional IK candidate ranks 0–5, and a release-and-retreat tool with 20/40 mm choices. The feedback identifies invalid starts, missing goal IK, terminal projection, goal collisions and path-search failures. These are generic tool choices; no new task-specific strategy or collision permission was added. The existing geometry and contact priors remain engineered.</p>
<p>The heuristic retains its original mode ordering, 25 mm step and automatic IK rank search. Both arms have the same expanded tools, geometry, controller and budgets; the heuristic does not acquire a new recovery policy. This is a comparison against that fixed policy, not proof that VLM search beats all non-VLM search strategies.</p>
<h2>First isolate the IK change</h2>{table(pilot['episodes'])}
<p>The old interface and prompt were retained for this pair. Both policies selected exactly the same contact posture after the fix, and both succeeded. The historical VLM run had failed. The identical first VLM request reused its cached response; later responses used updated observations. This single reproduction does not establish general reliability or isolate every source of the old failure.</p>
<h2>Frozen five-case comparison with v2</h2>{table(rows)}
<h2>Communication</h2><ol><li>Initial grounding uses RGB images and local DINO/depth processing to select the part and estimate geometry.</li>
<li>Each strategy request includes one 1024×1024 PNG: previous/current agentview and frontview. It also includes instruction, robot state, estimated geometry, prior choices/results and remaining budget.</li>
<li>Luna replies with one JSON tool choice. GraspGen, cuRobo, MC-VAMP and the fixed feedback controller run locally. Every issued trajectory completes.</li>
<li>After a tool completes or planning fails, new observations and diagnostics enter the next request. Up to five decisions share one 600-step physical episode.</li></ol>
<p>Each request explicitly resends this episode's history; there is no persistent server conversation or learned strategy memory shared across cases. The response is ordinary JSON validated locally before dispatch. No raw depth arrays, videos, simulator object identities, true articulated-object joints or native success flags are sent. The initial grounding call is separate from the strategy loop. Grounding can be reused after RGB/depth validation. <a href="communication-audit.json">Actual API image/text payload verification</a>.</p>
<p>Luna high reasoning; strategy ceiling 32,000 output tokens, initial grounding 12,000; no automatic network or incomplete-response retries. One validation correction is permitted for a completed invalid grounding response. Shared experiment spending, including development: <b>${budget['run_conservative_spend_usd']:.6f} of $2</b>. Times include API waiting but exclude shared model loading. These paused-simulator runs do not establish real-time deployment.</p>
<h2>Inspect what Luna actually saw and chose</h2>{''.join(cards)}
{smoke_html}
<h2>Verification, storage and reproduction</h2><p>All paired initial inputs, source snapshots, action logs, videos and feedback boundaries were verified. Cleanup runs after each episode; initial/final RGB-D, geometry, trajectories, exact API evidence and videos remain. No previous results were overwritten.</p>
<p><a href="evaluation.json">Complete verified results</a> · <a href="storage-summary.json">Cleanup and retained size</a> · <a href="verification-summary.json">Validation</a> · <a href="{link(ROOT/'results/vlm-recovery-ik-audit-v6-20261018/audit.json')}">GPU reproducibility audit</a> · <a href="{link(RUN/'protocol.json')}">Frozen protocol</a> · <a href="{link(RUN/'source-snapshot/configs/prompts/strategy_feedback_v2.txt')}">Frozen prompt</a> · <a href="../vlm-strategy-feedback-report/index.html">Prior pilot</a> · <a href="../shared-contact-feedback-report/index.html">Original reference</a></p>
<p>After evaluation, a regression caught a label error when explicitly requesting a pinch IK rank: the selected joints were correct but the reported rank used the filtered list index. The label was corrected; no frozen decision exercised that pinch option. Frozen sources and results are preserved. <a href="post-evaluation-label-fix.json">Scope audit</a> · <a href="post-evaluation-label-fix.patch">Exact one-line correction</a>.</p>
<pre>{html.escape(commands)}</pre></main></body></html>'''
    (OUT/'index.html').write_text(text)
    print(json.dumps({'report':str(OUT/'index.html'),'counts':counts,'spend':budget['run_conservative_spend_usd']}))

if __name__=='__main__':main()
