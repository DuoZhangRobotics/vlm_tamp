"""Compact report for paired shared-contact / execution-controller evaluations."""
import argparse
import html
import json
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from validate_shared_feedback import verify_episode
from build_bottom_drawer_investigation_report import CSS

LABELS={'nominal':'Standard servo (new pipeline)',
        'reanchor':'One-time correction (new pipeline)',
        'reactive':'Continuous correction (new pipeline)'}


def comparison_context(rows,core_keys):
    original=read_json(ROOT/'results/success-first-report/evaluation.json')
    diagnostic=read_json(ROOT/'results/bottom-drawer-feedback-20261016/summary.json')
    transfer=read_json(ROOT/'results/family-transfer-frozen-20261016/summary.json')
    original_tasks={'bowl_on_plate','open_middle_drawer','open_top_drawer','close_top_drawer',
                    'open_microwave','close_microwave','turn_on_stove'}
    bottom_cases=[r for r in transfer['episodes'] if r['task'] in {'open_bottom_drawer','close_bottom_drawer'}]
    repeated={key for key in core_keys if key[0] in original_tasks}
    repeated_counts={v:{'planned':len(repeated),'successes':sum(r['native_success'] for r in rows
        if r['variant']==v and (r['task'],r['state']) in repeated)} for v in LABELS}
    return {'variant_labels':LABELS,'nominal_definition':
        'New geometry/contact-selection/task-loop pipeline with the added reference-correction layer disabled; ordinary joint servo feedback and between-segment perception/replanning remain active.',
        'historical_baseline_replayed_in_full':False,'repeated_original_core_cases':repeated_counts,
        'original_regression':{'source':'results/success-first-report/evaluation.json',
            'planned':original['planned'],'successes':original['native_successes'],
            'tasks':sorted(original_tasks),'states':[1,11,21,31,41],
            'scope':'Aggregated frozen task-family regressions on developed cases; excludes bottom drawers'},
        'family_transfer':{'source':'results/family-transfer-frozen-20261016/summary.json',
            'groups':transfer['groups'],'bottom_drawer_successes':sum(r['official_task_success'] for r in bottom_cases),
            'bottom_drawer_completed':len(bottom_cases)},
        'bottom_opening_diagnostic':{'source':'results/bottom-drawer-feedback-20261016/summary.json',
            'planned':diagnostic['planned'],'successes':diagnostic['native_successes'],
            'planning_scene_collision_admission':False,'scope':'Physics-only development diagnosis'},
        'budget_policy':read_json(ROOT/'configs/api_budget_policy.json'),
        'promotion_policy':read_json(ROOT/'configs/execution_comparison_policy.json')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--development',nargs='*',default=[]);p.add_argument('--output',required=True);a=p.parse_args()
    out=ROOT/a.output;out.mkdir(exist_ok=True,parents=True);rows=[];development=[];manifests=[]
    def url(path):return html.escape(os.path.relpath(path,out),quote=True)
    for group,names in [('evaluation',a.runs),('development',a.development)]:
        for name in names:
            run=ROOT/name;manifest=read_json(run/'manifest.json')['episodes'];manifests.append({'run':name,'group':group,'planned':len(manifest)})
            for entry in manifest:
                folder=run/f'{entry["task"]}-state{entry["state"]:03d}-{entry["variant"]}'
                if not (folder/'summary.json').exists() or 'controller_variant' not in read_json(folder/'summary.json'):
                    reason=('Superseded before execution (development); retained in planned denominator'
                            if group=='development' and (run/'superseded.json').exists()
                            else 'Not executed / pending; included in planned denominator')
                    row={**entry,'folder':str(folder),'complete':False,'native_success':False,'failure':reason}
                else:
                    row={**verify_episode(folder),'folder':str(folder),'complete':True}
                    row['summary']=read_json(folder/'summary.json')
                (rows if group=='evaluation' else development).append(row)
    core_keys={key for key in {(r['task'],r['state']) for r in rows}
               if {r['variant'] for r in rows if (r['task'],r['state'])==key}=={'nominal','reanchor','reactive'}}
    counts={}
    for v in ['nominal','reanchor','reactive']:
        chosen=[r for r in rows if r['variant']==v and (r['task'],r['state']) in core_keys]
        counts[v]={'planned':len(chosen),'completed':sum(r['complete'] for r in chosen),
                   'successes':sum(r['native_success'] for r in chosen),
                   'final_successes':sum(r.get('final_native_success',False) for r in chosen)}
    context=comparison_context(rows,core_keys)
    write_json(out/'evaluation.json',{'matched_three_way_comparison':counts,'core_case_keys':sorted(core_keys),
        'episodes':rows,'development':development,'manifests':manifests,
        'comparison_context':context,
        'unequal_task_mixes_not_aggregated_for_controller_comparison':True})
    def table(records):
        content=[]
        for r in records:
            folder=Path(r['folder']);complete=r['complete']
            result=('Success (held)' if r.get('final_native_success') else 'Reached, not held') if r['native_success'] else 'Failure' if complete else 'Not run'
            mode=r.get('contact_mode') or ('pass-through' if r['task']=='bowl_on_plate' else 'not reached')
            force=f'{r["peak_single_contact_normal_force_n"]:.1f}' if complete else '—'
            reason=r.get('failure') or ('' if r['native_success'] else 'Task predicate not reached')
            links=f'<a href="{url(folder/"initial/rollout.mp4")}">Video</a> · <a href="{url(folder/"summary.json")}">Log</a>' if complete else '—'
            content.append(f'<tr><td>{html.escape(r["task"])} · {r["state"]}</td><td>{LABELS[r["variant"]]}</td><td>{mode}</td><td>{result}</td><td>{r.get("control_steps","—")}</td><td>{force}</td><td>{html.escape(reason)}</td><td>{links}</td></tr>')
        return '<div class="scroll"><table><thead><tr><th>Task / state</th><th>Controller</th><th>Contact mode</th><th>Native outcome</th><th>Steps</th><th>Peak contact N</th><th>Remaining failure</th><th>Evidence</th></tr></thead><tbody>'+''.join(content)+'</tbody></table></div>'
    stats=''.join(f'<div class="stat"><strong>{x["successes"]}/{x["planned"]}</strong>{LABELS[v]}; {x["completed"]} completed</div>' for v,x in counts.items())
    diagnostic=context['bottom_opening_diagnostic'];transfer=context['family_transfer']
    originals=transfer['groups']['original_tasks']
    repeated='; '.join(f'{LABELS[v]}: {c["successes"]}/{c["planned"]}'
                      for v,c in context['repeated_original_core_cases'].items())
    history_rows=''.join(f'<tr><td>{label}</td><td>{score}</td><td>{scope}</td></tr>' for label,score,scope in [
        ('Original seven-task regression',f'{context["original_regression"]["successes"]}/{context["original_regression"]["planned"]}',
         'Frozen task-family policies, five developed states each. No bottom-drawer tasks.'),
        ('Family transfer: original tasks, new states',f'{originals["native_successes"]}/{originals["planned"]}','Top-drawer opening/closing and stove; new initial states.'),
        ('Family transfer: bottom-drawer tasks',f'{transfer["bottom_drawer_successes"]}/{transfer["bottom_drawer_completed"]}','Both bottom-drawer opening and closing were new tasks.'),
        ('Bottom-drawer opening diagnosis',f'{diagnostic["successes"]}/{diagnostic["planned"]}',
         'Physics-only development test with planning-time scene collision admission bypassed.'),
        ('Previously successful cases repeated in this core comparison',repeated,
         'Top-drawer closing, microwave opening and microwave closing, state 1. The entire 35-case set was not rerun.')])
    vids=[]
    for task in ['open_bottom_drawer','open_microwave','close_microwave','close_top_drawer']:
        choices=[r for r in rows if r['complete'] and r['task']==task and r['variant']=='reactive']
        if not choices:continue
        r=choices[0];folder=Path(r['folder'])
        vids.append(f'<figure><video controls preload="metadata" src="{url(folder/"initial/rollout.mp4")}"></video><figcaption>{html.escape(task)} · state {r["state"]} · {"native success" if r["native_success"] else "native failure"}. Slowed playback; no frames omitted from the retained indexed movie.</figcaption></figure>')
    paired=[]
    for task,state in dict.fromkeys((r['task'],r['state']) for r in rows):
        group=[r for r in rows if r['task']==task and r['state']==state]
        if len(group)>1 and all(r['complete'] for r in group):paired.append((task,state,group))
    if paired:
        fig,axes=plt.subplots(len(paired),1,figsize=(10,max(3,len(paired)*2.3)),squeeze=False,constrained_layout=True)
        for ax,(task,state,group) in zip(axes[:,0],paired):
            names=[r['variant'] for r in group];forces=[r['peak_single_contact_normal_force_n'] for r in group]
            ax.bar([LABELS[n].removesuffix(' (new pipeline)') for n in names],forces,color=[{'nominal':'#bc6c25','reanchor':'#7389ae','reactive':'#087f8c'}[n] for n in names])
            ax.set_title(f'{task}, state {state}');ax.set_ylabel('Peak contact (N)')
            ax.set_ylim(0,max(5.,max(forces)*1.25))
            for i,r in enumerate(group):
                outcome='success' if r['native_success'] else 'failure before motion' if r['issued_joint_commands']==0 else 'failure'
                ax.text(i,forces[i],f'{forces[i]:.1f} N\n{outcome}',ha='center',va='bottom',fontsize=9)
        fig.savefig(out/'paired-contact-forces.png',dpi=140);plt.close(fig)
    forceplot='<img src="paired-contact-forces.png" alt="Paired peak native contact forces">' if paired else ''
    motion_rows=[]
    for path in (out/'object-motion').rglob('object-motion.json'):
        audit=read_json(path)
        entry=next((r for r in rows if r['folder']==audit['source']),None)
        if entry is None:continue
        objects=[o for o in audit['objects'] if not (entry['task']=='bowl_on_plate' and 'bowl' in o['name_evaluation_only'])]
        maximum=max((o['peak_translation_m'] for o in objects),default=0.)*1000
        rotation=max((o.get('final_rotation_deg',0.) for o in objects),default=0.)
        motion_rows.append(f'<tr><td>{html.escape(entry["task"])} · {entry["state"]}</td><td>{LABELS[entry["variant"]]}</td><td>{maximum:.3f}</td><td>{rotation:.3f}</td><td>{"Exact replay verified" if audit["accepted_as_original_episode_measurement"] else "Replay mismatch; do not attribute measurement"}</td><td><a href="{url(path)}">Audit</a></td></tr>')
    motion_table='<div class="scroll"><table><thead><tr><th>Task / state</th><th>Controller</th><th>Largest other-object translation (mm)</th><th>Largest final rotation (deg)</th><th>Replay validation</th><th>Evidence</th></tr></thead><tbody>'+''.join(motion_rows)+'</tbody></table></div>'
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shared contact and feedback evaluation</title><style>{CSS}</style></head><body><main>
<div class="eyebrow">VLM + motion planning · 18 October 2026</div><h1>Shared contact selection and reactive execution</h1>
<p class="notice"><b>Experimental candidate; not accepted as a replacement.</b> The original task-family pipeline remains the accepted reference with 35/35 observed successes. This report compares controller variants inside a changed pipeline. A full matched comparison with the exact original versions has not yet run. Promotion requires preserving the original successes and demonstrating a benefit on the same harder cases. <a href="../../configs/execution_comparison_policy.json">Comparison and promotion requirements</a>.</p>
<p class="lead">Three execution variants within the new geometry/contact-selection pipeline. Native LIBERO predicates determine outcomes. The earlier 35/35 result remains a separate, preserved experiment.</p>
<h2>How this relates to the earlier successes</h2><div class="scroll"><table><thead><tr><th>Experiment</th><th>Result</th><th>Scope and settings</th></tr></thead><tbody>{history_rows}</tbody></table></div>
<p>The new 3/5 standard-servo result adds two bottom-drawer cases to three previously successful cases. Both failures are bottom-drawer tasks. The old 35-case experiment excluded bottom drawers, and the later 4/4 bottom-opening diagnosis used different planning checks. These counts do not establish a regression of the old 35/35 result.</p>
<h2>Comparison within the new pipeline</h2>
<div class="stats">{stats}</div><p>These headline counts use only cases evaluated with all three controllers. Additional two-controller transfer cases and single-controller regression checks appear below; their different task mixes are not pooled into the headline comparison.</p>
<p class="notice">These are bounded development/regression comparisons, not a claim of 100% success or new-object generalization. New initial-state tests use the same LIBERO asset families. No RL training or demonstration was used. Missing, grounding, planning and execution cases remain in their planned denominators.</p>
<h2>Measured findings</h2><p>Continuous correction opened bottom-drawer states 15 and 7 where the new pipeline with standard servo alone failed. Top-drawer closing and microwave opening/closing passed their matched tests; the stove and bowl regression checks also passed. Bottom-drawer closing still failed before manipulation because no contact candidate had an admitted approach and probe.</p>
<p>Improved success did not consistently reduce force: the fresh bottom-drawer state reached about 292 N peak single-contact normal force with feedback. This controller accommodates position error; it does not regulate measured force. The intended bowl movement is excluded from the other-object translation audit below.</p>
<h2>What changed</h2><ol>
<li><b>Contact selection:</b> pinch, press/drag and push candidates use observed handle axes, local clearance and configured finger/palm dimensions. The same finite angle menu is used across cases. Press/drag uses a rear-bar contact envelope derived from the observed cross-section and pad dimensions; push uses a depth-grid-sized contact allowance. Contact is allowed only for selected fingers at the observed target region; arm and palm still undergo planning-time collision checks.</li>
<li><b>Reference correction:</b> at each 20 Hz simulator control tick, a proprioceptive outer loop adjusts small transverse pose offsets while preserving the nominal progress direction. This frequency is simulated control time, not a claim of end-to-end real-time performance. Press/drag retains a 2 mm preload; RGB-D registration can add bounded transverse corrections between complete motions. All nominal samples advance and are logged beside their issued commands.</li>
<li><b>Measured completion:</b> drawer replanning uses observed handle travel rather than summing requested travel. An occluded update uses a recorded robot-motion fallback; it does not veto the trajectory. Closing keeps one bounded seating motion near the observed cabinet face.</li>
<li><b>Numerical geometry:</b> the existing upright-drawer prior is applied during plane fitting, avoiding arbitrary axes from almost-horizontal table normals. Closing looks for cabinet support above or below the drawer. The RGB-selected handle seed removes disconnected background components.</li></ol>
<p><b>Controller scope:</b> this is bounded reference accommodation, not a force-sensor admittance controller or learned residual policy. Native contact forces are evaluation-only. Corrections may leave the nominal planned path; the experiment provides no collision-free execution guarantee. All gains are shared across the evaluated cases. Bowl execution is a pass-through regression because it has no articulated-contact context.</p>
<h2>Comparison protocol</h2><p>Within each paired case, all variants share the new visual-grounding inputs, contact menu, geometry, observed-progress task loop, grasp proposals and planner configuration. <b>Standard servo</b> (raw result key “nominal”) executes planned joint references with the existing joint servo feedback; between-segment perception and replanning still operate. Only the added outer reference-correction layer is disabled. <b>One-time correction</b> (“reanchor”) adds one reference adjustment per manipulation segment. <b>Continuous correction</b> (“reactive”) also adjusts transverse references during execution. The old full pipeline is preserved separately and was not replayed across all 35 cases here.</p>
<p>Seed 0, five settling actions, maximum 600 control steps. Every issued trajectory completes without execution-time collision, progress, retention or constraint vetoes. Paid perception uses Luna high reasoning, a 12k output ceiling and no automatic retries. <b>Budget clarification: $2 per experiment run</b>, shared by its episodes, variants, parallel workers and resumptions. A new experiment gets a new allowance; project spending history is informational. This accounting/report correction made no paid API calls and did not rerun or alter the recorded episodes.</p>
<h2>Evaluation outcomes</h2><p>The original success criterion is that the native task predicate becomes true. The table distinguishes success retained at the final step from transient success; final-predicate counts are also saved in the result data.</p>{table(rows)}
<h2>Videos</h2><div class="grid">{''.join(vids)}</div>
<h2>Contact forces</h2><p>Largest single native robot-contact normal force over each episode, including approach. This is not net wrist force. Compare success and loads together; reduced force after losing contact is not an improvement.</p>{forceplot}
<h2>Other-object movement</h2><p>Recorded low-level controls are replayed in a separate evaluation process to measure free-object translation and final orientation change. Measurements are attributed to the original run only when the full joint trace matches within 0.00001 rad and every native success flag matches. The intended moving bowl is excluded for bowl-on-plate; all other free objects are included. These labels and positions never enter the policy. Peak translation and final rotation are reported; transient rotation and articulated-joint disturbance are outside this metric.</p>{motion_table}
<h2>Development history</h2><p>Earlier versions are retained to expose the failed contact and control assumptions. V1 passed above the handle. V2 established front-side contact but slipped, and its feedback decayed preload. V3 maintained a rear contact but the old task loop counted commanded travel. The current loop uses measured displacement. Superseded, unexecuted entries are explicitly retained.</p>{table(development)}
<h2>Reproducibility and limits</h2><p>Each run retains source/configuration snapshots, selected contacts, predicted constraints, planned and issued trajectories, RGB-D tracking records, native outcomes and indexed videos. Temporary frames and intermediate observations are pruned after each episode. No simulator object IDs, joint axes, object poses or native success values enter contact selection or the feedback controller.</p>
<p>Manual assumptions remain explicit: an upright panel prior for these drawers, a finite contact-mode/tilt menu, depth-grid contact tolerances, and shared feedback gains. Different object designs, uncertain friction and wrist-force sensing require separate evaluation. Failures before contact require grounding or planning repair; local feedback cannot solve them by itself.</p>
<p><a href="evaluation.json">Machine-readable results</a> · <a href="verification-summary.json">Final verification and API accounting</a> · <a href="storage-summary.json">Cleanup accounting</a> · <a href="runtime-provenance.json">Environment and checkpoint versions</a> · <a href="input-pairing-audit.json">Paired-input audit</a> · <a href="../../scripts/run_shared_contact_feedback.py">Reproducible runner</a> · <a href="../success-first-report/index.html">Original 35/35 regression</a> · <a href="../family-transfer-report/index.html">Preserved frozen family benchmark</a> · <a href="../bottom-drawer-investigation-report/index.html">Earlier physics-only drawer diagnosis</a> · <a href="../../configs/api_budget_policy.json">Per-run API budget policy</a></p>
</main></body></html>'''
    (out/'index.html').write_text(doc)
    print(json.dumps({'report':str(out/'index.html'),'controllers':counts,'development_planned':len(development)}))

if __name__=='__main__':main()
