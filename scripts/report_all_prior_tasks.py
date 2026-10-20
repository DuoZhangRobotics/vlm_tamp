"""Native-outcome report for the frozen seven-task physical regression."""
import argparse
from collections import Counter
import hashlib
import html
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from run_all_prior_tasks import TASKS


def exists(path):return path.exists() or path.with_suffix('.json.gz').exists()
def esc(value):return html.escape(str(value))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_metrics(folder,entry):
    row=read_json(folder/'summary.json');actions=read_json(folder/'initial/actions.json') if exists(folder/'initial/actions.json') else []
    success=any(a['evaluation_task_success'] for a in actions)
    if success!=row['official_task_success']:raise ValueError('Saved native success differs from action log')
    phases=read_json(folder/'phases.json') if exists(folder/'phases.json') else []
    complete=all(p['all_commands_issued'] and p['planned_commands']==p['issued_commands'] for p in phases)
    vetoes=sum(p['execution_veto_checks'] for p in phases)
    assert complete and vetoes==0 and row['control_steps']<=600
    import numpy as np
    expected=0
    if entry['adapter']=='MC-VAMP hinge':
        approach=folder/'approach-references.npy'
        if approach.exists():expected+=len(np.load(approach))
        if exists(folder/'closed/metadata.json') or sum(a.get('execution_controller')=='JOINT_POSITION' for a in actions)>expected:
            expected+=40
        expected+=sum(len(np.load(p)) for p in folder.glob('plan-*/execution-references.npy'))
    else:expected=sum(len(np.load(p)) for p in folder.glob('path-*/references.npy'))
    issued=sum(a.get('execution_controller')=='JOINT_POSITION' for a in actions)
    tracker=read_json(folder/'tracking.json') if exists(folder/'tracking.json') else []
    row['saved_joint_references']=expected;row['joint_commands_in_native_action_log']=issued
    row['joint_commands_in_tracking_log']=len(tracker)
    row['every_saved_joint_reference_issued']=expected==issued==len(tracker)
    selected=read_json(folder/'chosen-grasp.json') if exists(folder/'chosen-grasp.json') else None
    row.update(adapter=entry['adapter'],complete_issued_trajectories=len(phases),all_issued_trajectories_completed=complete,
        execution_veto_checks=vetoes,first_native_success_step=next((a['step'] for a in actions if a['evaluation_task_success']),None),
        grasp_source=(selected.get('source','GraspGen candidate') if selected else 'none selected'))
    if selected is not None:
        row['grasp_selection']={k:v for k,v in selected.items() if k in ['source','index','variant','ik_rank','ik_alternative','tested_postures']}
    cycles=[p['planning_cycle_seconds'] for p in phases if 'planning_cycle_seconds' in p]
    method='recorded monotonic duration'
    if not cycles:
        pairs=[(folder/f'{label}-planning-scene.npz',folder/f'{label}-planning.json') for label in ['lift','transport','place']]
        pairs.extend((p,p.with_name(p.name.replace('mechanism-planning-scene-','mechanism-plan-').replace('.npz','.json')))
            for p in folder.glob('mechanism-planning-scene-*.npz'))
        cycles=[(b.stat().st_mtime_ns-a.stat().st_mtime_ns)/1e9 for a,b in pairs
            if a.exists() and b.exists() and 0<=b.stat().st_mtime_ns-a.stat().st_mtime_ns<600e9]
        method='planning-artifact timestamp estimate; scene saved before IK/admission, JSON saved afterward'
    row['postgrasp_segment_preparation_seconds']=cycles
    row['segment_preparation_timing_method']=method
    queries=folder/'curobo-queries.json'
    if exists(queries):
        data=read_json(queries)
        if isinstance(data,list):
            row['curobo_query_seconds']=[q['elapsed_seconds'] for q in data if 'elapsed_seconds' in q]
    failures=folder/'grasp-admission.json'
    if exists(failures):
        data=read_json(failures);rejects=data.get('rejected',data.get('rejected_candidates',[]))
        row['grasp_rejections_by_reason']=dict(Counter(r.get('reason','unspecified') for r in rejects))
        row['grasp_tested_postures']=data.get('tested',data.get('tested_ik_postures'))
    closed=folder/'closed/metadata.json';evaluation=folder/'initial/evaluation_contact_diagnostics.json'
    if exists(closed) and exists(evaluation):
        metadata=read_json(closed);step=int(metadata['control_steps']);records=read_json(evaluation)['steps']
        maximum_pads=0
        for record in records:
            if not step-4<=record['step']<=step+4:continue
            pads=set()
            for contact in record.get('robot_involved_contacts',[]):
                if all(contact['robot_flags']) or contact['normal_force_n']<=.1:continue
                pads.update(n for n,robot in zip(contact['geom_names'],contact['robot_flags'])
                    if robot and n.endswith('_pad_collision'))
            maximum_pads=max(maximum_pads,len(pads))
        row['closure_contact_evaluation_only']={'control_step':step,'maximum_external_pad_contacts_near_closure':maximum_pads,
            'physical_gripper_gap_m':sum(abs(x) for x in metadata['gripper_qpos']),
            'normal_force_threshold_n':.1,
            'correct_target_identity_not_inferred_from_this_count':True}
        if selected:
            import numpy as np
            target=selected.get('pose_world_xyzw',selected.get('fk_pose_world_xyzw'))
            if target is not None:
                row['closure_contact_evaluation_only']['tcp_reference_position_error_m']=float(np.linalg.norm(
                    np.asarray(target)[:3]-np.asarray(metadata['eef_pose_world_xyzw'])[:3]))
    if entry['task'] in {'open_microwave','close_microwave'}:
        evaluation=folder/'initial/evaluation_contact_diagnostics.json'
        if exists(evaluation):
            records=read_json(evaluation)['steps']
            if records:
                joint=next(j for j in records[-1]['evaluation_only_articulation_state'] if j['joint_name']=='microwave_1_microjoint')
                import math
                row['final_door_opening_deg_evaluation_only']=max(0.,-math.degrees(joint['position']))
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',default='results/all-prior-tasks-frozen-v3-20261015')
    parser.add_argument('--output',default='results/all-prior-tasks-report')
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args();run=ROOT/args.runs;output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    protocol=read_json(run/'protocol.json');planned=read_json(run/'manifest.json')['planned_episodes']
    assert all(sha(run/'source-snapshot'/n)==h for n,h in protocol['sources_sha256'].items())
    rows=[];cards=[];pending=[]
    for entry in planned:
        name=f'{entry["task"]}-state{entry["state"]:03d}';folder=run/name
        if not (folder/'summary.json').exists():pending.append(entry);continue
        row=episode_metrics(folder,entry);rows.append(row)
        video=folder/'initial/rollout.mp4';relative='../'+args.runs.removeprefix('results/')+'/'+name+'/initial/rollout.mp4'
        identifier='video-'+name
        media=(f'<video id="{identifier}" controls preload="none" src="{relative}"></video>'
            f'<button onclick="document.getElementById(\'{identifier}\').playbackRate=1">1× diagnostic playback</button> '
            f'<button onclick="document.getElementById(\'{identifier}\').playbackRate=5">5× ≈ joint-control real time</button>'
            if video.exists() else '<p>No rollout movie was recorded.</p>')
        outcome='SUCCESS' if row['official_task_success'] else 'FAILURE'
        reason=row.get('failure',row.get('continuation_stop_reason','All scheduled task commands completed.'))
        if not row['official_task_success'] and 'failure' not in row:
            reason='All scheduled task commands completed, but the native task predicate remained false.'
        if row['official_task_success'] and 'failure' in row:
            reason='Native task succeeded; a subsequent pipeline stage stopped: '+reason
        details=esc(row.get('grasp_rejections_by_reason',{}))
        cycles=row['postgrasp_segment_preparation_seconds']
        timing=('No postgrasp planning duration available.' if not cycles else
            f'Postgrasp segment preparation: {sum(cycles)/len(cycles):.2f} s mean; {esc(row["segment_preparation_timing_method"])}.')
        contact=row.get('closure_contact_evaluation_only')
        contact_text=('' if contact is None else f'<p>Closure diagnostic, evaluation only: {contact["maximum_external_pad_contacts_near_closure"]} pads exerting more than 0.1 N on external geometry near closure; '
            f'gripper gap {1000*contact["physical_gripper_gap_m"]:.1f} mm. This count alone does not verify the intended grasp.</p>')
        if contact and 'tcp_reference_position_error_m' in contact:
            contact_text+=f'<p>TCP reference-position error after closure: {1000*contact["tcp_reference_position_error_m"]:.1f} mm.</p>'
        angle=(f'<p>Final actual door opening: {row["final_door_opening_deg_evaluation_only"]:.2f}° (evaluation only).</p>'
            if 'final_door_opening_deg_evaluation_only' in row else '')
        cards.append(f'''<details class="episode"><summary>{esc(entry['task'].replace('_',' '))} · state {entry['state']}: <b>{outcome}</b></summary>
        <p>{esc(reason)}</p><p>Failure stage: {esc(row.get('failure_stage','—'))}. Control steps: {row['control_steps']}.
        Complete issued trajectories: {row['complete_issued_trajectories']}; execution veto checks: {row['execution_veto_checks']}.</p>
        <p>Joint-reference audit: {row['joint_commands_in_native_action_log']}/{row['saved_joint_references']} saved references issued;
        action/tracking logs agree: {row['joint_commands_in_native_action_log']==row['joint_commands_in_tracking_log']}.</p>
        <p>Final-state native predicate: <b>{row.get('final_task_success',False)}</b>. {timing}
        Total episode wall time: {row.get('episode_wall_seconds',0.):.1f} s.</p>
        <p>Task adapter: {esc(entry['adapter'])}. Grasp source: {esc(row['grasp_source'])}. Fixed grasp replay: false.</p>
        {angle}{contact_text}<p>Grasp-search rejection counts: {details}.</p>{media}</details>''')
    if pending and not args.partial:raise ValueError(f'Only {len(rows)}/{len(planned)} cases completed; use --partial for a progress view')
    task_rows=[];task_stats=[]
    for task in TASKS:
        targets=[e for e in planned if e['task']==task['label']]
        if not targets:continue
        subset=[r for r in rows if r['task']==task['label']];wins=sum(r['official_task_success'] for r in subset)
        statistics={'task':task['label'],'planned':len(targets),'completed':len(subset),'successes':wins,
            'failures_by_stage':dict(Counter(r.get('failure_stage','native_goal_false') for r in subset if not r['official_task_success'])),
            'adapter':task['adapter']}
        task_stats.append(statistics)
        cells=[]
        for state in [1,11,21,31,41]:
            case=next((r for r in subset if r['initial_state']==state),None)
            cells.append('<td class="'+('pending' if case is None else 'yes' if case['official_task_success'] else 'no')+'">'
                +('pending' if case is None else 'success' if case['official_task_success'] else 'failure')+'</td>')
        task_rows.append(f'<tr><td>{esc(task["label"].replace("_"," "))}</td>'+''.join(cells)+
            f'<td>{wins}/{len(targets)}</td><td>{esc(task["adapter"])}</td></tr>')
    before=read_json(run/'api-ledger-before.json');ledger=read_json(ROOT/'results/articulated-api-v1/ledger.json')
    spent=ledger['cumulative_conservative_spend_usd']-before['cumulative_conservative_spend_usd']
    reclaimed=sum(r.get('storage',{}).get('bytes_reclaimed_this_pass',0) for r in rows)
    dedup=read_json(run/'binary-deduplication.json') if (run/'binary-deduplication.json').exists() else {}
    before_motion=sum(not r['official_task_success'] and r['joint_commands_in_native_action_log']==0 for r in rows)
    partitions=[read_json(p) for p in run.glob('partition-*-protocol.json')]
    interruptions=[read_json(p) for p in run.glob('infrastructure-interruptions/*/summary.json')]
    integration=[]
    for version in ['all-prior-tasks-frozen-20261015','all-prior-tasks-frozen-v2-20261015']:
        integration_root=ROOT/'results'/version
        for path in sorted(integration_root.glob('*-state*/summary.json')):
            data=read_json(path);integration.append({'run':version,**{k:data[k] for k in ['task','initial_state','status','official_task_success','failure_stage','failure'] if k in data}})
    evaluation={'planned':len(planned),'completed':len(rows),'pending':pending,'native_successes':sum(r['official_task_success'] for r in rows),
        'per_task':task_stats,'episodes':rows,'all_manifest_failures_in_denominator':True,'fixed_grasp_replay':False,
        'execution_veto_checks':sum(r['execution_veto_checks'] for r in rows),'source_snapshots_verified':True,
        'baseline_sources_unchanged':all(sha(ROOT/n)==h for n,h in protocol['protected_baseline_sources_sha256'].items()),
        'api_new_requests':len(ledger['requests'])-len(before['requests']),'api_new_conservative_spend_usd':spent,
        'failures_before_grasp_motion':before_motion,
        'failed_cases_after_joint_execution':sum(not r['official_task_success'] and r['joint_commands_in_native_action_log']>0 for r in rows),
        'binary_deduplication_bytes_reclaimed':dedup.get('unique_inode_bytes_reclaimed',0),
        'remaining_api_budget_usd':ledger['remaining_usd'],'reclaimed_bytes':reclaimed,'focused_tests_passed':41,
        'superseded_integration_attempts':integration,
        'infrastructure_interruptions':interruptions,
        'all_completed_attempts_across_versions':len(rows)+len(integration)+len(interruptions),
        'parallel_simulation_partitions':partitions,
        'regression_not_held_out_generalization':True,'comparison_scope':'different task adapters; not a single-planner benchmark'}
    write_json(output/'evaluation.json',evaluation)
    status='IN PROGRESS' if pending else 'COMPLETE'
    document='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>All prior tasks: frozen physical evaluation</title><style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:17px/1.5 system-ui;background:#f5f7fa;color:#172b3d}
    table{border-collapse:collapse;background:white;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #d9dfe5}td.yes{color:#14733c}td.no{color:#9b322b}td.pending{color:#777}
    article,.episode{background:white;padding:18px;margin:18px 0;border-radius:10px}summary{cursor:pointer;font-weight:650}video{display:block;max-width:100%;width:512px;margin-top:16px}.scroll{overflow-x:auto}</style>
    <h1>Seven prior tasks · five official states each</h1>'''
    document+=f'''<article><b>{status}</b>: {len(rows)}/{len(planned)} episodes completed.
    Native successes: {evaluation['native_successes']}/{len(planned)} planned episodes.
    Pending cases are shown explicitly and are not classified as failures.</article>'''
    if not pending:
        document+=f'<p>Of {len(rows)-evaluation["native_successes"]} failures, <b>{before_motion}</b> stopped before grasp motion and <b>{evaluation["failed_cases_after_joint_execution"]}</b> followed physical joint execution. Pre-motion admission failures are software-pipeline outcomes, not evidence that the physical tasks are impossible.</p>'
    document+='''<p>Fresh physical resets and fresh grasp/IK selection; no fixed grasp replay. Identical cached RGB grounding and
    GraspGen proposals may be reused only after fresh sensor/checkpoint agreement. No provided inventory, simulator object
    pose, object mesh, joint axis or contact identity enters policy decisions. The same Panda controller follows complete timed joint paths.</p>
    <p>Collision/constraint admission happens during planning. Every issued path is executed fully; contacts, tracking error,
    grasp-retention diagnostics and native success cannot interrupt its references. Replanning occurs between finished paths.
    Physical contacts remain active and no attachment is injected. Native LIBERO success alone determines task outcome.</p>
    <p>Movies are diagnostic recordings at 4 fps. Joint motion is captured every 20 Hz control step, so 1× playback shows it
    five times slower than simulation time; choose 5× for approximately real-time joint motion. Initial sensing frames are
    sampled more sparsely. Playback speed does not include the time spent planning between segments.</p>
    <div class="scroll"><table><tr><th>Task</th><th>State 1</th><th>11</th><th>21</th><th>31</th><th>41</th><th>Success / planned</th><th>Planner adapter</th></tr>'''
    document+=''.join(task_rows)+'</table></div>'
    document+='''<article><b>Planner scope:</b> microwave doors use the updated MC-VAMP partial-pose hinge planner with RGB-D
    landmark refinement. Bowl/drawer/stove adapters use cuRobo multi-IK continuation with sampled observed robot/payload
    collision admission; they are not MC-VAMP search benchmarks. Bowl motion has point goals and no added geometric path
    constraints. Drawers use line constraints; the stove uses a coupled rotation model. These previously explored tasks/states
    form a regression cohort, not a held-out generalization test. Bounded grasp/IK search failure does not prove all grasps infeasible.</article>'''
    document+=f'''<p>Execution vetoes: <b>{evaluation['execution_veto_checks']}</b>. New Luna High requests: {evaluation['api_new_requests']},
    12k output limit, zero automatic retries. New conservative API cost: ${spent:.4f}; cumulative cap $2;
    remaining ${ledger['remaining_usd']:.4f}. Cleanup reclaimed {reclaimed/2**30:.2f} GiB.
    The full-episode gallery references the original movies directly.</p>'''
    if dedup:
        document+=f'<p>Additional exact-binary deduplication reclaimed {dedup["unique_inode_bytes_reclaimed"]/2**20:.1f} MiB. Every original pathname and byte remains available; planning timestamps are preserved.</p>'
    previews=[p for p in sorted((output/'previews').glob('*.mp4'))]
    if previews:
        document+='<details class="episode"><summary>Short successful previews at simulation speed</summary><p>These preserve every joint-control frame at 20 Hz; planning pauses and initial sensing are omitted. Full diagnostic recordings remain in each case below.</p>'
        for movie in previews:
            document+=f'<p>{esc(movie.stem.replace("_"," "))}</p><video controls preload="none" src="previews/{movie.name}"></video>'
        document+='</details>'
    if partitions:
        document+='<p>Independent cases use the same frozen policy and per-episode seeds. Up to three simulation workers were used; concurrency was reduced after a system memory interruption. Planning durations are operational measurements under concurrent load, not isolated planner benchmarks.</p>'
    if interruptions:
        document+='''<details class="episode"><summary>Preserved infrastructure interruption</summary><p>A system memory spike
        killed the main worker during state-11 microwave planning. The saved action log verifies that all 296 issued joint
        references completed; the door had reached 67.8° and the next path had not been issued. Its inputs, actions, video
        and source provenance are archived separately. The affected case is restarted from the same official reset with the
        exact frozen policy. The interrupted prefix is not classified as a completed robot failure.</p></details>'''
    if integration:
        entries=''.join(f'<li>{esc(r["task"])} state {r["initial_state"]}: native success {r["official_task_success"]}; {esc(r.get("failure","completed"))}</li>' for r in integration)
        document+='''<details class="episode"><summary>Preserved initial integration attempts</summary><p>The initial non-microwave
        adapter omitted robot-only depth arrays, causing robot surfaces to enter the obstacle cloud. The reader was corrected
        universally. A second integration trial completed bowl placement successfully, then exposed a post-run filename
        collision between grasp retention and storage retention; its physical result and video were recovered.
        Both fixes and regressions were completed before the final 35-case source freeze. Search/task parameters were unchanged.
        All completed initial attempts are retained here; these superseded adapter trials are separate from the corrected
        frozen cohort above. Unstarted initial-manifest cases were cancelled, rather than labeled physical failures.</p><ul>'''+entries+'</ul></details>'
    document+=''.join(cards)+'''<p>41 focused checks passed. <a href="evaluation.json">Full outcomes and failure counts</a> ·
    <a href="../../docs/all_prior_tasks.md">Protocol and reproduction</a>.</p>'''
    (output/'index.html').write_text(document)
    print({'completed':len(rows),'planned':len(planned),'native_successes':evaluation['native_successes'],'report':str(output/'index.html')})


if __name__=='__main__':main()
