"""Render the overnight experiment ledger without altering measured results."""
from datetime import datetime,timezone
import html
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/overnight-long-horizon-report-20261020'
RUNS=[
    ('Matched microwave constraints','grasp-horizon-physical-20261020'),
    ('Microwave repeatability','grasp-horizon-repeatability-20261020'),
    ('Family baseline, state 1','success-first-family-baseline-20261020'),
    ('Paired full remaining motions','success-first-family-long-20261020'),
    ('Paired full-motion grasp screening','success-first-full-grasp-screen-20261020'),
    ('Middle drawer, new sample and full screening','middle-drawer-grasp-seed1-20261020'),
    ('Shared adaptive policy, seed 1','adaptive-family-seed1-20261020'),
    ('Paired held-object boundary policy','adaptive-family-payload-boundary-20261020'),
    ('Seven tasks × five regression states','adaptive-all-tasks-five-state-20261020'),
    ('Prepared sequence, matched state-21 failures','preplanned-state21-comparison-20261020'),
    ('Prepared sequence, states 1 / 11 / 21','preplanned-three-state-confirmation-20261020'),
    ('Prepared sequence, states 31 / 41','preplanned-remaining-states-20261020'),
    ('Final shared policy, seven tasks × five states','prepared-final-all-tasks-20261020'),
]


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def relative(path):
    return html.escape(os.path.relpath(path,OUT),quote=True)


def link(path,label):
    return f'<a href="{relative(path)}">{html.escape(str(label))}</a>'


def video(path,title):
    if not path.exists():return ''
    timed=OUT/'media'/'--'.join(path.relative_to(ROOT/'results').parts)
    display=timed if timed.exists() else path
    note='20 Hz control-time playback; planning and model pauses excluded.' if timed.exists() else 'Original retained video; playback rate may differ from control timing.'
    return f'<article><h3>{html.escape(title)}</h3><video controls preload="metadata" src="{relative(display)}"></video><p class="small">{note} {link(path,"Original")}</p></article>'


def main():
    OUT.mkdir(exist_ok=True)
    rows=[];details=[];ledger=[]
    for title,name in RUNS:
        folder=ROOT/'results'/name
        if not folder.exists():continue
        summary=read(folder/'summary.json',{})
        episodes=summary.get('episodes',[])
        protocol=read(folder/'protocol.json',{})
        planned=summary.get('planned',len(protocol.get('manifest',[])))
        successes=sum(bool(r.get('official_task_success')) for r in episodes)
        rows.append(f'<tr><td>{link(folder/"protocol.json",title)}</td><td>{successes}/{len(episodes)}</td><td>{planned}</td><td>{"Complete" if planned and len(episodes)==planned else "Running / pending"}</td></tr>')
        case_rows=[]
        for r in episodes:
            label=r.get('condition',r.get('task','unknown'))
            index=r.get('initial_state',r.get('state_index'))
            path=(folder/f'{label}-r{r["repetition"]}/result.json' if 'condition' in r
                  else folder/f'{label}-state{index:03d}/summary.json')
            outcome='Success' if r.get('official_task_success') else 'Failure'
            failure=r.get('error') or r.get('failure') or ''
            if failure=='simulator_success':failure=''
            case_rows.append(f'<tr><td>{link(path,label)}</td><td>{index}</td><td class="{outcome.lower()}">{outcome}</td><td>{r.get("control_steps","—")}</td><td>{html.escape(str(failure))}</td></tr>')
            ledger.append({'run':name,'case':label,'state':index,'native_success':bool(r.get('official_task_success')),
                'steps':r.get('control_steps'),'failure':failure,'result_path':str(path)})
        details.append(f'<details><summary>{html.escape(title)} — {successes}/{len(episodes)} completed episodes successful</summary><table><tr><th>Condition/task</th><th>State</th><th>Native result</th><th>Steps</th><th>Failure</th></tr>{"".join(case_rows)}</table></details>')
    actor_rows=[]
    outcomes=[];actor=None;actor_status='Not started';phase='not yet reported'
    actor_runs=['subscription-scene-sync-practice-20261020','subscription-prepared-sequence-practice-20261020']
    for stage,name in enumerate(actor_runs,1):
        folder=ROOT/'results'/name
        if not folder.exists():continue
        actor=folder
        local=read(folder/'evaluation-private.json',{}).get('attempts',[])
        exit_record=read(folder/'agent-exit.json')
        budget=read(folder/'practice-budget.json',{})
        phase=budget.get('learning_phase',{}).get('phase','not yet reported')
        actor_status='Finished' if exit_record else 'Running' if (folder/'agent-process.json').exists() else 'Not started'
        for r in local:
            path=folder/f'attempt{r["attempt"]:02d}/summary.json'
            actor_rows.append(f'<tr><td>{link(path,str(stage)+" / "+str(r["attempt"]))}</td><td>{"Success" if r["official_task_success"] else "Failure"}</td><td>{r["control_steps"]}</td><td>{r.get("requested_trajectories","—")}</td><td>{html.escape(r.get("reason",""))}</td></tr>')
            outcomes.append({**r,'practice_stage':stage,'run':name})
    actor_rows=''.join(actor_rows)
    videos=[
        video(ROOT/'results/subscription-prepared-sequence-practice-20261020/attempt08/initial/rollout.mp4','Luna: learned 85° sequence, 283 steps'),
        video(ROOT/'results/frozen-prepared-method-transfer-20261020/state011/attempt01/initial/rollout.mp4','Frozen agent transfer, state 11: native failure'),
        video(ROOT/'results/grasp-horizon-physical-20261020/synced-point-90-r1/attempt01/initial/rollout.mp4','Microwave: one long manipulation plan'),
        video(ROOT/'results/grasp-horizon-physical-20261020/synced-point-15-r1/attempt01/initial/rollout.mp4','Microwave: synchronized short motions'),
        video(ROOT/'results/adaptive-family-payload-boundary-20261020/bowl_on_plate-state001/initial/rollout.mp4','Bowl placement with held-object boundary clearance'),
        video(ROOT/'results/adaptive-family-payload-boundary-20261020/open_middle_drawer-state001/initial/rollout.mp4','Middle drawer with full-motion grasp screening'),
        video(ROOT/'results/adaptive-family-payload-boundary-20261020/turn_on_stove-state001/initial/rollout.mp4','Stove activation with the shared adaptive policy'),
    ]
    for r in outcomes:
        if r['official_task_success']:
            videos.append(video(ROOT/'results'/r['run']/f'attempt{r["attempt"]:02d}/initial/rollout.mp4','Luna: first successful practice attempt'))
            break
    updated=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    final_cohort=read(ROOT/'results/prepared-final-all-tasks-20261020/summary.json',{})
    transfer_root=ROOT/'results/frozen-prepared-method-transfer-20261020'
    transfer=read(transfer_root/'summary.json',{})
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Long motion, contact constraints and adaptive practice</title>
<style>body{font:16px/1.55 system-ui,sans-serif;margin:32px auto;padding:0 22px;max-width:1200px;color:#233441;background:#f8fafc}h1,h2{line-height:1.2}h2{margin-top:34px}a{color:#176b91}table{border-collapse:collapse;width:100%;margin:16px 0;background:white}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #dce4ec;vertical-align:top}th{background:#edf3f7}td:last-child{overflow-wrap:anywhere}.success{color:#16714c}.failure{color:#a13d31}.note{padding:14px 20px;border-left:4px solid #387c99;background:#eaf4f8}.scope{background:#fff6e9;border-left-color:#b77b29}.small{font-size:14px;color:#506676}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}article{padding:14px;background:white;border:1px solid #dae3eb;border-radius:8px}video,img{max-width:100%;height:auto}details{margin:14px 0;padding:12px;background:white;border:1px solid #dce4ec}summary{cursor:pointer;font-weight:600}code{background:#edf1f5;padding:2px 4px}</style><body>'''
    page+=f'<h1>Long motions and contact constraints</h1><p class="small">Updated {updated}. This page is generated from retained results. Refresh after the report is regenerated.</p>'
    if final_cohort:
        page+=f'<p class="note"><strong>Final numerical regression: {final_cohort.get("native_successes",0)}/{final_cohort.get("completed",0)} completed cases successful, {final_cohort.get("planned",0)} planned.</strong> This tests the planning/execution pipeline with matched perception/proposal inputs and inherited family priors.</p>'
    if transfer:
        page+=f'<p class="note scope"><strong>Frozen learned-agent transfer: {transfer.get("native_successes",0)}/{transfer.get("completed",0)} completed cases successful.</strong> Fresh agent grounding and the learned sequence recipe are not yet generally reliable. These results must remain separate from the numerical pipeline score.</p>'
    page+='''<p class="note"><strong>A long microwave motion works in the matched development scene.</strong> The synchronized 90° condition achieved 3/3 native successes with one manipulation plan and 330 control steps. The synchronized 15° condition achieved 3/3 with six plans and 339 steps. These are repeatability results from one verified post-closure robot state and scene, not generalization or verification of a grasp attachment.</p>
<p>The first component runs took about 56 s for the long condition versus 161 s for short motions, including setup and execution but excluding actor reasoning. Later development cohorts may overlap Luna practice; their wall times are not strict performance comparisons.</p>
<h2>What changed</h2>
<ul><li>Moving clouds are synchronized with the next planning frame. RGB-D rigid tracking is preferred; hand-motion prediction is labeled explicitly when tracking is insufficient.</li>
<li>The tool exposes point, rigid-pose and finger-axis contact constraints. Rigid or finger-axis assumptions are optional; they failed the initial long-motion searches and are not silently promoted over the successful point mode.</li>
<li>Planning failures expose current-query projection, collision and search diagnostics. Grasp sampling has an explicit seed included in cache identity.</li>
<li>The family experiment can prefer grasps supporting the full task motion, retain a short-probe fallback, and fall back to shorter planning when necessary.</li>
<li>For held objects, an optional nonpenetration boundary replaces an extra positive clearance reserve. Unrelated obstacle margins and true penetration checks remain. The paired bowl retry succeeded after a positive 0.87 mm gap had previously failed a 1 mm requirement.</li></ul>
<p>The prepared-sequence comparison executes the full manipulation admitted during grasp selection, after physical closure. It retains the existing direction probe for ambiguous rotary controls and the short-plan fallback when no full plan is available. The bowl sequence ends at placement and release, with no retreat. On the two state-21 counterfactuals, initial sensor arrays and all approach/closure commands match exactly; both previously failed cases succeeded.</p>
<p>Execution uses the existing 20 Hz joint-position servo. All motion ends at the first native simulator success. No extra retreat or execution-time collision veto is added.</p>
<h2>Numerical experiment ledger</h2><p>Different rows use different declared settings and may repeat the same scenes. Do not combine them into one benchmark score. Failures before motion remain included. A native success overrides the legacy engine’s terminal-exception status label; the original files are preserved and the separate audit explains that bookkeeping.</p>'''
    page+='<table><tr><th>Experiment</th><th>Successes / completed</th><th>Planned</th><th>Status</th></tr>'+''.join(rows)+'</table>'
    page+=''.join(details)
    page+=f'<h2>Luna practice</h2><p><strong>{actor_status}</strong> · {sum(r["official_task_success"] for r in outcomes)}/{len(outcomes)} completed attempts successful · phase: {html.escape(str(phase))}.</p>'
    page+='''<p>ChatGPT subscription, requested model <code>gpt-6-luna</code>, high reasoning. Stage 1 completed seven attempts (5/7 successes); stage 2 completed nine (8/9). The four remaining attempts were reserved for frozen transfer, reaching twenty total. Practice starts with five granted attempts and extends adaptively. Success-first instructions remain until two consecutive native successes in the current session. Source code, human diagnostic trajectories and simulator object state are hidden.</p>
<p class="note scope">Stage-1 attempts 6 and 7 used neutral gripper commands that bypassed scene synchronization because of a trigger bug. Their efficiency comparison is confounded. Stage 2 corrects that trigger, adds the prepared-sequence tool and supplies the agent’s own abstract findings and whitelisted relative tool parameters. Historical image coordinates, world poses and trajectories are excluded. Earlier attempts remain in the ledger.</p>'''
    page+='<table><tr><th>Stage / attempt</th><th>Native result</th><th>Steps</th><th>Execution segments</th><th>Termination</th></tr>'+actor_rows+'</table>'
    page+='''<p>Luna's final two local attempts used a prepared 85° motion, eight closure steps and neutral gripper hold, succeeding twice at 283 control steps. One planning RPC and one sequence execution RPC cover three physical stages. Planning RPCs fell from 7–8 to 1, and total robot RPCs from 17 to 4–5. The baseline used 250 control steps, so the change reduces calls but increases physical steps. Tool time was approximately 55–56 s versus 173–203 s, excluding Luna reasoning; overlapping runs prevent a controlled end-to-end speed claim. A tested 90° recipe failed and remains in the results.</p>'''
    if actor is not None and (actor/'learning-summary.json').exists():page+='<p>'+link(actor/'learning-summary.json','Latest practice summary and measured efficiency')+'</p>'
    if transfer:
        page+='<h2>Frozen-memory transfer</h2><p>One attempt each on different initial states of the same microwave task and asset, not previously unseen appliances or a strict held-out set. The method and tools were frozen, with no findings passed between cases. This uses the four remaining agent attempts: 16 practice attempts plus four transfers, twenty total.</p><table><tr><th>State</th><th>Native result</th><th>Control steps</th><th>Physical segments</th></tr>'
        for r in transfer.get('cases',[]):
            case=transfer_root/f'state{r["state"]:03d}'
            ep=read(case/'evaluation-private.json',{}).get('attempts',[{}])
            ep=ep[0] if ep else {}
            page+=f'<tr><td>{link(case/"remembered-agent-result.json",r["state"])}</td><td>{"Success" if r["native_success"] else "Failure"}</td><td>{ep.get("control_steps","—")}</td><td>{ep.get("requested_trajectories","—")}</td></tr>'
        page+='</table>'
    support=read(OUT/'scene-support/audit.json',[])
    if support:
        page+='''<h2>Why transfer still fails</h2><p>Initial model points had depth support in every audited case. At the ends of the three failed transfers, 77–83% of the predicted contact points conflicted with observed free space in multiple views. The scene model had been transported from hand motion after tracking failed. This supports a state-estimation/contact-recovery problem; it does not establish that every collision rejection was erroneous.</p>
<p>A new read-only <code>inspect_scene</code> tool exposes these depth-support statistics and coloured overlays, and failure feedback names blocking geometry/robot parts. It adds no action rule, model correction or execution veto. These diagnostics were added after the frozen transfer study and have not been credited with any new agent success.</p>'''
        page+='''<p>The agent tool and numerical runner also differ in handle fitting and contact-point selection. The numerical runner accounts for finite finger-pad support; the older agent tool chose TCPs 13–16 mm higher in these four cases. This may affect contact, but the experiments do not isolate its causal effect. The performance gap cannot be attributed solely to agent reasoning.</p>'''
        example=OUT/'scene-support/state011-model-overlay.png'
        if example.exists():page+=f'<img style="max-width:620px" src="{relative(example)}" alt="Predicted moving geometry conflicts with current measured depth">'
        page+='<p>'+link(OUT/'scene-support/audit.json','Sensor-only model/depth audit')+'</p>'
    page+='''<h2>What physical success does and does not establish</h2>
<p class="note scope">The successful microwave interaction is not a verified rigid pinch. At the end of the shared closure prefix, the finger pads touch each other. Later finger and palm contact opens the door. The long case has 27 manipulation steps with palm/microwave contact, and no recorded unrelated-object contact during that manipulation. These native contact logs were inspected only after execution.</p>
<p>The native opening predicate ends the episode near 75° of actual door opening, before the entire requested 90° reference is sent. The point-cloud model still omits unobserved space and mechanism-versus-environment collision. Transporting a cloud is a state estimate, not fresh reconstruction of every surface or proof of retained contact.</p>'''
    image=ROOT/'results/grasp-horizon-physical-20261020/long-motion-contact-frames.png'
    if image.exists():page+=f'<img src="{relative(image)}" alt="Frames of the successful long microwave interaction">'
    page+='''<h2>Videos</h2><div class="gallery">'''+''.join(videos)+'</div>'
    image=OUT/'drawer-palm-collision.png'
    if image.exists():
        page+=f'<h2>Drawer collision evidence</h2><p>The rejected next palm pose contains 19 observed points labeled stationary. This overlay uses sensor calibration and robot geometry; it does not use simulator object identities. Red points may be occluded in the displayed camera while visible in another view.</p><img style="max-width:650px" src="{relative(image)}" alt="Observed points inside the rejected palm pose">'
    page+='''<h2>Scope and reproducibility</h2><p>The family cohorts use the established numerical adapters and inherited geometry/contact priors with RGB-validated grounding reuse. They are not evidence that the microwave acting agent has learned or transferred all those tasks. The microwave proposals selected in the matched test and early agent attempts came from the inherited <em>RGB-D frontal handle refinement</em> generator, which is part of the hybrid grasp tool alongside GraspGen. This contribution is not an agent discovery. Model weights are unchanged. Candidate arrays can differ across inference runs despite identical recorded seeds, clouds and checkpoint hashes; paired comparisons therefore reuse validated identical proposal arrays.</p>
<p>The API cap is $2 per experiment. Subscription practice uses no separately billed API requests. Initial/final observations, numerical plans, issued commands, tracking, videos, outcomes and source provenance are retained; temporary proposal caches and redundant per-step artifacts are cleaned after episodes.</p>'''
    page+='''<h2>Validation and code checkpoint</h2><p>The broad non-integration test run recorded 1,162 passes, 13 failures, three skips and six deselections. Two missing-dependency cases passed in the original LIBERO environment. Ten failures need previously pruned historical fixtures; one frozen-baseline hash guard already differed before this work. New and relevant regression tests passed; the full suite is not presented as green.</p>
<p>The pre-work and first implementation checkpoints were pushed. Automatic approval review blocked a later update to the public repository. The final code is retained locally; the repository link below refers to the earlier published checkpoint.</p>'''
    storage=read(OUT/'storage-and-links-audit.json',{})
    if storage:
        page+=f'<p>Completed-day results occupy approximately {storage["allocated_bytes_unique_inodes"]/1e9:.1f} GB after sharing byte-identical evidence. Deduplication shared {storage["shared_logical_duplicate_bytes"]/1e9:.1f} GB of logical duplicate data while preserving every evidence path and its contents.</p>'
    page+='<p>'+link(OUT/'baseline-terminal-label-audit.json','Terminal-label audit')+' · '+link(OUT/'contact-tool-parity-audit.json','Contact-tool parity audit')+' · '+link(OUT/'validation/summary.json','Validation summary')+' · '+link(OUT/'storage-and-links-audit.json','Storage audit')+' · '+link(ROOT/'docs/long_horizon_experiment_20261020.md','Technical notes')+' · '+link(ROOT/'data/agent_memory/luna-microwave-v4/successful-method.json','Candidate memory and transfer limits')+' · <a href="https://github.com/DuoZhangRobotics/vlm_tamp/tree/codex/grasp-consistent-long-horizon">Earlier published code checkpoint</a></p></body></html>'
    (OUT/'index.html').write_text(page)
    (OUT/'episode-ledger.json').write_text(json.dumps({'updated_utc':updated,'numerical_episodes':ledger,
        'actor_attempts':outcomes,'actor_status':actor_status,'frozen_transfer':transfer},indent=2)+'\n')
    print(json.dumps({'report':str(OUT/'index.html'),'numerical_completed':len(ledger),
        'actor_attempts':len(outcomes),'actor_status':actor_status}))


if __name__=='__main__':main()
