"""Verified comparison of existing Luna tools and visual region recovery."""
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from validate_strategy_feedback import verify_run,check_feedback
from build_bottom_drawer_investigation_report import CSS
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.run_api_budget import RunLedger,snapshot
from vlm_tamp.rekep_libero.api import request_arguments,_canonical

RUN=ROOT/'results/region-recovery-frozen-v2-20261018'
DEV=ROOT/'results/region-recovery-development-20261018'
INTERRUPTED=ROOT/'results/region-recovery-frozen-20261018'
SEPARATE=ROOT/'results/region-recovery-separate-20261018'
CONTINUED=ROOT/'results/region-recovery-continuation-v2-20261018'
STOPPED=ROOT/'results/region-recovery-continuation-20261018'
OUT=ROOT/'results/region-recovery-report'

def audit_episode(row,ledger):
    folder=Path(row['folder']);requests=[];captures=[];revisions=[]
    for item in row['decisions']:
        f=Path(item['folder']);choice=item['choice']
        if (f/'api-accounting.json').exists():
            account=read_json(f/'api-accounting.json')
            if account.get('image_input_mode')=='separate_cameras':
                from vlm_tamp.region_multiview import request_arguments as multi_arguments
                arguments,image_files=multi_arguments(f,(f/'prompt.txt').read_text())
            else:
                arguments=request_arguments(f/'feedback-sheet.png',(f/'prompt.txt').read_text(),32000)
                image_files=['feedback-sheet.png']
            digest=hashlib.sha256(_canonical(arguments).encode()).hexdigest()
            path=ledger.directory/f'request-{digest}.json'
            if not path.exists():
                hit=next(x for x in ledger.snapshot().get('cache_hits',{}).values() if x['request_sha256']==digest)
                path=Path(hit['source_ledger']).parent/f'request-{digest}.json'
            actual=read_json(path);assert arguments==actual
            pictures=[i for i in actual['input'][0]['content'] if i['type']=='input_image']
            assert len(pictures)==len(image_files)
            for picture,name in zip(pictures,image_files):
                assert base64.b64decode(picture['image_url'].split(',',1)[1])==(f/name).read_bytes()
            requests.append({'round':f.name,'request_sha256':digest,'exact_image_and_prompt_verified':True,'image_files':image_files})
        if (f/'collision-evidence/summary.json').exists():
            evidence=read_json(f/'collision-evidence/summary.json');captures.append(evidence)
        if choice['operation']=='revise_region':
            rec={'round':f.name,'request':choice['region'],'result':item['result'],'folder':str(f)}
            if (f/'region-revision/selection.json').exists():
                selected=read_json(f/'region-revision/selection.json')
                assert selected['request']==choice['region'] and selected['human_supplied_replacement_coordinates'] is False
                assert selected['collision_policy_changed'] is False and selected['simulator_object_truth_used'] is False
                rec['selection']=selected
            revisions.append(rec)
    return {'requests':requests,'captures':captures,'revisions':revisions}

def main():
    OUT.mkdir(exist_ok=True)
    frozen=verify_run(RUN);development=verify_run(DEV);interrupted=verify_run(INTERRUPTED)
    separate=verify_run(SEPARATE);continued=verify_run(CONTINUED);stopped=verify_run(STOPPED)
    assert not frozen['pending'] and frozen['all_pairs_identical']
    assert not separate['pending'] and not continued['pending']
    ledger=RunLedger('vlm-region-recovery-20261018');budget=snapshot(ledger)
    assert budget['run_conservative_spend_usd']<=2
    audits=[{'folder':r['folder'],**audit_episode(r,ledger)} for r in frozen['episodes']+separate['episodes']+continued['episodes']]
    counts={p:{'planned':2,'successes':sum(r['success'] for r in frozen['episodes'] if r['policy']==p)} for p in ('baseline','region')}
    evidence={'by_policy':counts,'frozen':frozen,'development':development,'interrupted':interrupted,
        'interruption':read_json(INTERRUPTED/'interruption.json'),'audits':audits,'api':budget,
        'separate_camera_followup':separate,'conditional_continuation':continued,'stopped_replay_checks':stopped,
        'all_failures_in_final_manifest_denominator':True,'promoted':False,
        'study':'two known failures, matched eight-decision conditions; diagnostic evidence, not broad generalization'}
    write_json(OUT/'evaluation.json',evidence)
    def link(path):return html.escape(os.path.relpath(path,OUT),quote=True)
    def table(rows):
        body=[]
        for r in rows:
            revisions=sum(d['choice']['operation']=='revise_region' for d in r['decisions'])
            last=next((d['result'] for d in reversed(r['decisions']) if d['result'] and d['result'].get('status') not in ('agent_finished','observation_updated')), {})
            desc='' if r['success'] else str(last.get('failure_loci') or last.get('reason') or r['failure'])
            f=Path(r['folder'])
            raw=read_json(f/'summary.json')
            outcome='Interrupted' if raw.get('interrupted_for_display_fix') else 'Success' if r['success'] else 'Failure'
            body.append(f'<tr><td>{r["state"]}</td><td>{r["policy"]}</td><td>{outcome}</td>'
                f'<td>{r["control_steps"]}</td><td>{len(r["decisions"])}</td><td>{revisions}</td><td>{html.escape(desc)}</td>'
                f'<td><a href="{link(f/"initial/rollout.mp4")}">Video</a> · <a href="{link(f/"summary.json")}">Record</a></td></tr>')
        return '<div class="scroll"><table><tr><th>State</th><th>Policy</th><th>Native outcome</th><th>Steps</th><th>Decisions</th><th>Region requests</th><th>Last failure</th><th>Evidence</th></tr>'+''.join(body)+'</table></div>'
    cards=[]
    for r in frozen['episodes']+separate['episodes']+continued['episodes']:
        if r['policy']=='baseline':continue
        rounds=[]
        for item in r['decisions']:
            d=item['choice'];f=Path(item['folder']);result=item['result'] or {}
            api_links=(f'<p><a href="{link(f/"prompt.txt")}">Exact prompt</a> · <a href="{link(f/"response.json")}">Raw response</a></p>') if (f/'response.json').exists() else '<p>Replayed prior model choice; no new API call.</p>'
            rounds.append(f'<details><summary>{f.name}: {d["operation"]} / {d["contact_mode"]} → {html.escape(result.get("status","interrupted"))}</summary>'
                f'<p><b>Luna:</b> {html.escape(d["evidence"])}</p><p><b>Expected:</b> {html.escape(d["expected_observation"])}</p>'
                f'<img loading="lazy" src="{link(f/"feedback-sheet.png")}" alt="Exact image supplied to Luna">'
                f'<pre>{html.escape(json.dumps(result,indent=2))}</pre>'
                +api_links+f'<p><a href="{link(f/"decision.json")}">Returned choice and coordinates</a> · <a href="{link(f/"feedback.json")}">Input feedback</a></p></details>')
        cards.append(f'<section><h2>State {r["state"]}: {r["policy"]}</h2>'
            f'<video controls preload="metadata" src="{link(Path(r["folder"])/"initial/rollout.mp4")}"></video>'+''.join(rounds)+'</section>')
    stats=''.join(f'<div class="stat"><strong>{v["successes"]}/{v["planned"]}</strong>{p}</div>' for p,v in counts.items())
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visual region recovery</title><style>{CSS}video,img{{max-width:100%;width:100%}}video{{max-height:560px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:12px}}</style></head><body><main>
<div class="eyebrow">Luna · collision evidence · RGB-region revision · 18 October 2026</div><h1>Can the model revise the part it is trying to contact?</h1>
<p class="lead">The new tool lets Luna revise both the contact patch and the moving panel from RGB. Numerical tools validate depth support and refit the existing geometry model. Rejected poses are shown with robot geometry and a measured-surface overlap witness.</p>
<div class="stats">{stats}</div><p>Native LIBERO success determines these results. All four episodes in the final frozen manifest remain in the denominator. These are two previously failed cases, not unseen tasks or a generalization benchmark.</p>
<h2>What changed</h2><p>New camera-local normalized region choices; point-prompted MobileSAM and measured depth; an explicit gravity-aligned tangent for flat contact patches; candidate-search coverage; and visualizations of native rejected robot poses. Purple is the hypothetical robot, red is the reported rejected part and reconstructed overlap. These are planning predictions, not measured simulator contacts.</p>
<p>The native collision implementation, arm/palm checks, bounded functional-finger contact policy, MC-VAMP planner, cuRobo IK and feedback controller are unchanged. No simulator object identities or axes are used. Collision snapshots are captured in both conditions; only the new policy receives the previews and revision tool. Expensive reconstruction/rendering runs after tool completion, outside planner search budgets.</p>
<p>Current region outlines are projected from tracked 3-D geometry at each observation. They are estimates and may be occluded. Original selection coordinates are retained separately with their selection time. A region revision begins a new progress reference and requires released contact; progress from potentially different parts is not added together.</p>
<h2>Frozen comparison</h2><p>Both policies use Luna high, a 32,000-token response ceiling, eight decisions and 600 control steps. Earlier reference runs had five decisions. Thus this compares two matched eight-decision conditions, not an isolated visualization ablation or a direct rerun of the earlier five-case headline.</p>{table(frozen['episodes'])}
<h2>Why the region tool matters</h2><p>A visual audit found that the initial state-17 grounding selected the wooden stand instead of the metal cabinet's drawer. The new model prompt receives the original task instruction, RGB and numerical evidence; no human replacement coordinates or object-location hint is supplied. A depth-valid region can still be semantically wrong. The following images and native outcomes distinguish region revision from actual task completion.</p>
<h2>Separate-camera follow-up</h2><p>Some composite-image selections were consistent with whole-sheet horizontal normalization. This follow-up supplied each annotated 512×512 camera image independently, with collision previews as a separate diagnostic image. No returned coordinate was manually corrected. Tools, model, physical budget and eight-decision limit were retained. Initial RGB and geometry match the earlier runs, but freshly sampled GraspGen proposals differ across processes; this is an exploratory follow-up, not a pure presentation ablation.</p>{table(separate['episodes'])}
<h2>Conditional continuation</h2><p>Both separate-image episodes ended with completed motion and positive observed progress while control budget remained. Their prior eight model choices were replayed, the endpoint checked, and up to four new model decisions allowed. The 600-step limit and shared API cap remained fixed. These are conditional continuations, not fresh twelve-decision policy evaluations; the earlier scores are unchanged.</p>{table(continued['episodes'])}
<p>An initial replay check stopped before any new API call because it compared an extra source-only observation taken after the original decision loop. The corrected check preserves exact robot/image checks and compares the policy's tracking prefix, excluding that one bookkeeping frame. <a href="{link(STOPPED/'replay-check-diagnosis.json')}">Preserved stopped-check diagnosis</a>.</p>
<h2>Evaluation-only geometry and tracking audit</h2><p>In separate-image state 15, the final inferred motion axis was about 88° from the requested drawer's actual joint axis. The four manipulation segments reported about 39.8 mm of projected progress, while the native bottom-drawer joint moved about 7.7 mm during those segments. These simulator measurements were read only after execution and were never supplied to the model or controller. A passing depth fit and apparently strong tracking can therefore still describe the wrong motion. <a href="mechanism-evaluation.json">Per-episode joint/axis audit</a>.</p>
<p>In state 17, the continuation closed the native drawer joint by about 127.9 mm but still did not reach the native success predicate. Its final four segments each commanded roughly 5 mm forward motion, while the hand moved about 0.2 mm backward per segment. The final evaluation-only contact log includes palm–cabinet contact at about 28 N. This supports a physical contact/control stall; the commands were not vetoed or cancelled. The specific remedy has not been established by this experiment.</p>
{''.join(cards)}
<h2>Development and interrupted attempt</h2><p>The integration pilot is preserved below. A later comparison attempt was stopped after discovering that its outline display retained selection-time image coordinates after object motion. One baseline completed, one region episode was interrupted, and two episodes were not run in that attempt. The corrected comparison was frozen under a new output directory; the interrupted request's reservation remains in the shared budget.</p>
{table(development['episodes'])}{table(interrupted['episodes'])}
<p><a href="{link(INTERRUPTED/'interruption.json')}">All four entries in the interrupted manifest</a>. These infrastructure/development records are separate from the final four-episode comparison and are not hidden or overwritten.</p>
<h2>Validation and retention</h2><p>Tests verify native collision-decision parity, a reconstructed witness for a real saved rejection, normalized coordinates, transactional revision failure, search coverage and moving overlays. The episode verifier checks native outcomes, issued commands, initial paired inputs, videos and source snapshots. Every issued trajectory completes; no execution-time veto is introduced.</p>
<p>Shared experiment accounting, including the pilot, follow-ups and interrupted attempt: <b>${budget['run_conservative_spend_usd']:.6f} of $2</b>. Uncertain calls retain their full reservation and are not automatically retried. Reservations count every image separately. Temporary frames, meshes and intermediate depth archives are removed after each run; necessary pictures, geometry, trajectories, videos and API evidence remain. <a href="https://developers.openai.com/api/docs/models/gpt-6-luna">Official Luna capabilities and pricing</a>.</p>
<p><a href="evaluation.json">Complete outcomes and audits</a> · <a href="verification-summary.json">Final verification</a> · <a href="storage-summary.json">Cleanup</a> · <a href="{link(RUN/'protocol.json')}">Frozen protocol</a> · <a href="{link(RUN/'source-snapshot/configs/prompts/strategy_feedback_v3.txt')}">Frozen region prompt</a> · <a href="../astra-recovery-diagnostic-report/index.html">Prior Astra diagnostic</a></p>
</main></body></html>'''
    (OUT/'index.html').write_text(text)
    for source,name in [('/tmp/region-recovery-tests-v2-20261018.log','tests.log'),
        ('/tmp/region-recovery-development-20261018.log','development.log'),
        ('/tmp/region-recovery-frozen-20261018.log','interrupted.log'),
        ('/tmp/region-recovery-frozen-v2-20261018.log','frozen.log'),
        ('/tmp/region-recovery-separate-20261018.log','separate.log'),
        ('/tmp/region-recovery-continuation-v2-20261018.log','continuation.log')]:shutil.copyfile(source,OUT/name)
    print(json.dumps({'report':str(OUT/'index.html'),'counts':counts,'conservative_spend_usd':budget['run_conservative_spend_usd']}))

if __name__=='__main__':main()
