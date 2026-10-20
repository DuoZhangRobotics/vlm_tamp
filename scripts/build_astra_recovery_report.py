"""Verify and report the matched model diagnostic and shared physical replays."""
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.model_diagnostic_budget import DiagnosticLedger,usage_cost
from vlm_tamp.recovery_decision import parse_decision
from vlm_tamp.rekep_libero.articulated_grounding import response_text
from vlm_tamp.rekep_libero.api import _canonical
from validate_strategy_feedback import verify_run,check_feedback
from replay_astra_recovery_choices import signature
from build_bottom_drawer_investigation_report import CSS

RUN=ROOT/'results/astra-recovery-diagnostic-20261018'
PHYSICAL=ROOT/'results/astra-recovery-physical-20261018'
OUT=ROOT/'results/astra-recovery-diagnostic-report'

def main():
    OUT.mkdir(exist_ok=True)
    summary=read_json(RUN/'summary.json');protocol=read_json(RUN/'protocol.json')
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((RUN/'source-snapshot'/name).read_bytes()).hexdigest()==digest
    ledger=DiagnosticLedger(RUN/'api').snapshot();cases=[];verified_requests=[]
    for case,source_episode,source_round in protocol['cases']:
        request_rows=[];model_rows=[]
        for model in protocol['models']:
            entry=ledger['requests'][case+'-'+model];digest=entry['request_sha256']
            request=read_json(RUN/'api'/f'request-{digest}.json')
            assert hashlib.sha256(_canonical(request).encode()).hexdigest()==digest
            assert request['model']==model and request['max_output_tokens']==12000 and request['reasoning']=={'effort':'high'}
            content=request['input'][0]['content'];source=RUN/'inputs'/case
            assert content[0]['text']==(source/'prompt.txt').read_text()
            assert base64.b64decode(content[1]['image_url'].split(',',1)[1])==(source/'feedback-sheet.png').read_bytes()
            check_feedback(read_json(source/'feedback.json'))
            raw=read_json(RUN/'api'/f'response-{digest}.json');assert raw['status']=='completed'
            decision=parse_decision(response_text(raw));folder=RUN/'decisions'/case/model
            assert decision==read_json(folder/'decision.json')
            cost,known=usage_cost(raw,model,entry['reservation_usd']);assert known and abs(cost-entry['cost_usd'])<1e-12
            record=read_json(folder/'summary.json')
            model_rows.append({'model':model,'actual_response_model':raw['model'],'decision':decision,'usage':raw['usage'],
                'seconds':record['round_trip_seconds'],'conservative_cost_usd':cost,'folder':str(folder)})
            request_rows.append({k:v for k,v in request.items() if k!='model'})
            verified_requests.append({'case':case,'model':model,'request_sha256':digest,'text_and_image_verified':True})
        assert request_rows[0]==request_rows[1]
        cases.append({'case':case,'source_episode':source_episode,'source_round':source_round,
            'models':model_rows,'same_command':signature(model_rows[0]['decision'])==signature(model_rows[1]['decision']),
            'original_Luna_choice':read_json(RUN/'inputs'/case/'decision.json')})
    physical=verify_run(PHYSICAL);assert not physical['pending']
    for case in cases:
        row=next(r for r in physical['episodes'] if r['state']==(15 if case['case']=='closing15_late' else 17))
        folder=Path(row['folder']);match=read_json(folder/'decision-point-match.json');assert match['accepted']
        branch=folder/'decisions'/case['source_round'];result=read_json(branch/'tool-result.json')
        actual=read_json(branch/'decision.json')
        assert all(signature(actual)==signature(m['decision']) for m in case['models'])
        case.update(physical=row,reconstructed_state=match,new_choice_result=result,
                    original_choice_result=read_json(RUN/'inputs'/case['case']/'tool-result.json'))
    assert ledger['conservative_spend_usd']<=2.
    data={'cases':cases,'same_commands':sum(c['same_command'] for c in cases),'paired_decision_points':len(cases),
          'physical_unique_replays':len(physical['episodes']),'independent_model_episode_success_rates_measured':False,
          'api':ledger,'request_audit':verified_requests,'physical_verification':physical}
    write_json(OUT/'evaluation.json',data)
    def link(path):return html.escape(os.path.relpath(path,OUT),quote=True)
    def choice(d):
        if d['operation']=='try_contact':return f"{d['contact_mode']}, tilt {d['approach_tilt_deg']}, {d['step_mm']} mm {d['motion_extent']}, IK {d['posture_rank']}"
        return f"{d['operation']}, {d['step_mm']} mm {d['motion_extent']}, IK {d['posture_rank']}"
    rows=[];cards=[]
    for c in cases:
        r=c['new_choice_result'];f=Path(c['physical']['folder']);progress=r.get('object_progress_delta_m')
        description=(f"{r.get('completed_segments',0)} completed segment(s); "
            +(f"{progress*1000:.1f} mm additional observed progress; " if progress is not None else '')
            +str(r.get('failure_loci',{})))
        rows.append(f'<tr><td>{html.escape(c["case"])}</td><td>{html.escape(choice(c["models"][0]["decision"]))}</td>'
            f'<td>{html.escape(choice(c["models"][1]["decision"]))}</td><td>{html.escape(description)}</td>'
            f'<td>{"Reached" if c["physical"]["success"] else "Not reached"}</td></tr>')
        text=''.join(f'<h4>{m["model"]}</h4><p>{html.escape(m["decision"]["evidence"])}</p>'
            f'<p>Expected: {html.escape(m["decision"]["expected_observation"])}</p>'
            f'<p>{m["usage"]["input_tokens"]} input / {m["usage"]["output_tokens"]} output tokens; '
            f'{m["seconds"]:.1f} seconds; estimated ${m["conservative_cost_usd"]:.6f}. '
            f'<a href="{link(Path(m["folder"])/"response.json")}">Raw response</a></p>' for m in c['models'])
        cards.append(f'<section><h2>{html.escape(c["case"])}</h2><p>Earlier Luna choice at this point: '
            f'{html.escape(choice(c["original_Luna_choice"]))}. Both new calls had a matched 12,000-token ceiling.</p>'
            f'<details><summary>Exact image and prompt supplied to both models</summary>'
            f'<img src="{link(RUN/"inputs"/c["case"]/"feedback-sheet.png")}" alt="Identical saved RGB feedback image">'
            f'<p><a href="{link(RUN/"inputs"/c["case"]/"prompt.txt")}">Exact prompt and history</a></p></details>{text}'
            f'<h3>Physical replay of their shared choice</h3><video controls preload="metadata" src="{link(f/"initial/rollout.mp4")}"></video>'
            f'<p>The video includes the original prefix before the new recovery choice. '
            f'<a href="{link(f/"decision-point-match.json")}">Decision-point matching</a> · '
            f'<a href="{link(f/"decisions"/c["source_round"]/"tool-result.json")}">New tool result</a> · '
            f'<a href="{link(f/"summary.json")}">Episode record</a></p></section>')
    document=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Astra versus Luna: recovery diagnostic</title><style>{CSS}video,img{{max-width:100%;width:100%}}video{{max-height:560px}}table{{width:100%}}</style></head><body><main>
<div class="eyebrow">Matched saved-observation diagnostic · 18 October 2026</div><h1>Did Astra choose better recoveries?</h1>
<p class="lead">Both models selected the same command at both tested failure points. The two shared choices were then tested physically with the existing tools.</p>
<p>This is a two-point diagnostic conditioned on prior Luna actions. It does not measure complete Astra episode success, establish model equivalence, or evaluate a new agent interface. The historical five-case comparison remains unchanged.</p>
<div class="stats"><div class="stat"><strong>{sum(c['same_command'] for c in cases)}/{len(cases)}</strong>identical command choices</div>
<div class="stat"><strong>4/4</strong>completed, valid model replies</div><div class="stat"><strong>${ledger['conservative_spend_usd']:.3f}</strong>estimated spend of $2 cap</div></div>
<h2>Matched conditions</h2><p>Identical prompt, 1024×1024 RGB image, history, existing tool menu, high reasoning, 12,000-token ceiling and default processing tier. Only the model field differs between each new pair. The earlier Luna run used 32,000 tokens; new paired Luna calls control for that ceiling change. No additional hints, simulator object truth, native tool-calling integration or persistent conversation state were introduced.</p>
<p>The original action prefix was replayed from the official initial state. Robot joints, TCP, fingers, RGB feedback, estimated geometry and progress were checked before executing the new choice. Simulator state was not injected. Identical model commands share one physical replay per point and are not counted as independent model trials. Every issued trajectory completed; no execution-time veto was added.</p>
<div class="scroll"><table><tr><th>Failure point</th><th>New Luna choice</th><th>Astra choice</th><th>Physical result</th><th>Native task predicate</th></tr>{''.join(rows)}</table></div>
<h2>Interpretation</h2><p>No better recovery action was observed from Astra on these two inputs. Both fresh models chose alternatives to the earlier Luna response, so those changes cannot be attributed specifically to Astra. The physical results below distinguish feasible motion from an explanation that merely sounds plausible.</p>
<p>The next useful investigation is the evidence and action interface: show which candidates were already tested, localize predicted collisions, and allow sensor-grounded contact-region revision. A different model cannot select a contact region that the tool API does not expose. No such interface change was made in this experiment.</p>
{''.join(cards)}
<h2>Reproducibility and accounting</h2><p>The shared ledger reserved the full output allowance before each request. All four worst-case reservations summed to ${read_json(RUN/'budget-preflight.json')['maximum_total_usd']:.6f}, below $2. No automatic retries were enabled, and physical replays made no API calls. Prices include a conservative input cache-write ceiling; timing observations are single samples and are not a latency benchmark.</p>
<p><a href="evaluation.json">Full results and request audit</a> · <a href="verification-summary.json">Final checks</a> · <a href="storage-summary.json">Cleanup</a> ·
<a href="{link(RUN/'protocol.json')}">Model diagnostic protocol</a> · <a href="{link(PHYSICAL/'protocol.json')}">Physical replay protocol</a> ·
<a href="{link(RUN/'api/ledger.json')}">Cost ledger</a> · <a href="https://developers.openai.com/api/docs/models/gpt-6-astra">Official Astra pricing and capabilities</a> ·
<a href="../vlm-recovery-tools-report/index.html">Prior five-case evaluation</a></p>
</main></body></html>'''
    (OUT/'index.html').write_text(document)
    for source,name in [('/tmp/astra-recovery-tests-20261018.log','tests.log'),('/tmp/astra-replay-tests-20261018.log','replay-tests.log'),
                        ('/tmp/astra-recovery-diagnostic-20261018.log','model-run.log'),('/tmp/astra-recovery-physical-20261018.log','physical-run.log')]:
        shutil.copyfile(source,OUT/name)
    write_json(OUT/'verification-summary.json',{'model_and_budget_tests_passed':52,'replay_guard_tests_passed':5,
        'model_requests_verified':len(verified_requests),'matched_input_pairs':len(cases),
        'physical_decision_points_matched':sum(c['reconstructed_state']['accepted'] for c in cases),
        'all_issued_trajectories_verified':True,'frozen_snapshots_verified':True,'budget_cap_respected':True})
    print(json.dumps({'report':str(OUT/'index.html'),'same_choices':data['same_commands'],'spend_usd':ledger['conservative_spend_usd'],
        'physical_outcomes':[{k:r[k] for k in ('task','state','success','control_steps')} for r in physical['episodes']]}))

if __name__=='__main__':main()
