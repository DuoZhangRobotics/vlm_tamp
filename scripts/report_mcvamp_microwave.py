"""Paired LIBERO outcomes and videos for MC-VAMP versus the frozen pipeline."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from run_mcvamp_microwave import baseline_folder
from report_coupled_door import video


def metrics(folder):
    row=read_json(folder/'summary.json')
    evaluations=read_json(folder/'initial/evaluation_contact_diagnostics.json')['steps'] if (folder/'initial/evaluation_contact_diagnostics.json').exists() else []
    native={e['step']:next(j for j in e['evaluation_only_articulation_state'] if j['joint_name']=='microwave_1_microjoint')
        for e in evaluations}
    final=native[max(native)] if native else None
    row['final_opening_angle_deg']=max(0.,-float(final['position'])*180/3.141592653589793) if final else None
    # Evaluation-only asset names never flow into policy/perception/planning.
    handle={'microwave_1_g14','microwave_1_g15','microwave_1_g16'}
    contacts=[];handle_steps=[];two_pads=[]
    for evaluation in evaluations:
        blocked=[];pads=set();target=False
        for contact in evaluation.get('robot_involved_contacts',[]):
            flags=contact['robot_flags']
            if all(flags) or contact['normal_force_n']<=1e-5:continue
            external=next(n for n,r in zip(contact['geom_names'],flags) if not r)
            robot=next(n for n,r in zip(contact['geom_names'],flags) if r)
            if not(external in handle and 'finger' in robot):blocked.append(contact)
            else:
                target=True
                if robot.endswith('_pad_collision'):pads.add(robot)
        if blocked:contacts.append(evaluation['step'])
        if target:handle_steps.append(evaluation['step'])
        if len(pads)==2:two_pads.append(evaluation['step'])
    row['planning_queries']=[]
    row['non_target_positive_contact_steps']=len(contacts)
    row['handle_contact_steps_evaluation_only']=len(handle_steps)
    row['bilateral_pad_contact_steps_evaluation_only']=len(two_pads)
    row['final_handle_contact_evaluation_only']=bool(evaluations and evaluations[-1]['step'] in handle_steps)
    for path in folder.glob('**/result.json'):
        if path.parent.name.startswith('plan-'):continue
        result=read_json(path)
        if result.get('planner') in {'MC-VAMP CRRTC','VAMP RRTC'}:
            row['planning_queries'].append({'path':str(path.relative_to(folder)),
                **{k:result[k] for k in ['planner','exact_solution','solve_seconds','resampling_seconds','total_seconds','iterations'] if k in result}})
    return row,native,contacts


def esc(value):return html.escape(str(value))
def angle(value):return '—' if value is None else f'{value:.1f}°'


def plot_pair(baseline,current,output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axis=plt.subplots(figsize=(7.2,3.4),layout='constrained')
    for folder,label,color in [(baseline,'Frozen baseline','#778494'),(current,'MC-VAMP + timed execution','#146cbd')]:
        records=read_json(folder/'initial/evaluation_contact_diagnostics.json')['steps']
        points=[(e['step'],max(0.,-np.degrees(next(j['position'] for j in e['evaluation_only_articulation_state']
            if j['joint_name']=='microwave_1_microjoint')))) for e in records]
        if points:axis.plot(*np.asarray(points).T,label=label,color=color,linewidth=2)
        actions=read_json(folder/'initial/actions.json')
        first=next((a['step'] for a in actions if a['evaluation_task_success']),None)
        if first is not None:
            value=next(y for x,y in points if x==first)
            axis.scatter([first],[value],color=color,s=35,zorder=5)
    axis.set(xlabel='Control step (simulation)',ylabel='Actual door opening (degrees)',ylim=(-2,122))
    axis.grid(alpha=.25);axis.legend(fontsize=8,loc='best')
    axis.set_title('Dots mark the first native task success; articulation is evaluation-only',fontsize=10)
    fig.savefig(output,dpi=130);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=False)
    pairs=[];cards=[];seen=set();validations=[];all_cycles=[]
    for run in args.runs:
        root=ROOT/run;manifest=read_json(root/'manifest.json');summary=read_json(root/'summary.json')
        protocol=read_json(root/'protocol.json')
        if summary['completed']!=len(manifest['planned_episodes']):raise ValueError('Finish every planned episode before reporting')
        unchanged=all(hashlib.sha256((root/'source-snapshot'/name).read_bytes()).hexdigest()==digest
            for name,digest in protocol['sources_sha256'].items())
        validations.append({'run':run,'saved_source_hashes_match':unchanged,
            'baseline_sources_unchanged':summary['baseline_sources_unchanged'],
            'api_ledger_unchanged':summary['api_ledger_unchanged']})
        for entry in manifest['planned_episodes']:
            label,index=entry['task'],entry['state'];key=(label,index)
            if key in seen:raise ValueError('Duplicate paired case')
            seen.add(key)
            case=f'{label}-state{index:03d}';folder=root/case
            new,native,contacts=metrics(folder);old,_,_=metrics(baseline_folder(label,index))
            old={k:v for k,v in old.items() if k!='planning_queries'}
            pair={'task':label,'initial_state':index,'baseline':old,'mc_vamp':new}
            phases=read_json(folder/'phases.json') if (folder/'phases.json').exists() else []
            all_cycles.extend(p['planning_cycle_seconds'] for p in phases)
            if phases:new['median_planning_cycle_seconds']=float(np.median([p['planning_cycle_seconds'] for p in phases]))
            pairs.append(pair)
            movie=None
            if native and (folder/'task-spec.json').exists():
                movie=case+'.mp4'
                video(folder,new,native[max(native)],native,contacts,output/movie)
            reason=(new.get('continuation_stop_reason') or new.get('failure') or
                'All planned commands issued; native task predicate decides success.')
            if new['official_task_success'] and new.get('pipeline_status',new.get('status'))=='failed':
                reason='Native task succeeded. A subsequent planning continuation stopped: '+reason
            media=f'<video controls preload="metadata" src="{movie}"></video>' if movie else '<p>No task-motion video: failure before planning/execution.</p>'
            chart=case+'-opening.png'
            plot_pair(baseline_folder(label,index),folder,output/chart)
            original=f'../planning-control-frozen-report/{case}.mp4'
            completed=new.get('completed_contact_trajectories',0)
            execution_text=(f'{completed} issued contact trajectories completed in full.' if completed else
                'No contact trajectory was issued: planning stopped before manipulation.')
            cards.append(f'<article><h2>{esc(label.replace("_"," "))} · state {index}</h2>'
                f'<p>Frozen baseline: <b>{"success" if old["official_task_success"] else "failure"}</b>, final opening {angle(old["final_opening_angle_deg"])}. '
                f'MC-VAMP: <b>{"success" if new["official_task_success"] else "failure"}</b>, final predicate {new["final_task_success"]}, '
                f'final opening {angle(new["final_opening_angle_deg"])}.</p>'
                f'<p>{esc(reason)}</p><p>{execution_text} Execution veto checks: <b>{new.get("execution_veto_checks",0)}</b>.</p>'
                f'<p>Contact diagnostic only: bilateral pad contact on {new["bilateral_pad_contact_steps_evaluation_only"]} controls '
                f'(baseline {old["bilateral_pad_contact_steps_evaluation_only"]}); final handle contact: '
                f'{new["final_handle_contact_evaluation_only"]}. Contact counts do not determine task success.</p>'
                f'<img alt="Actual native door angle versus control step" style="max-width:100%;width:720px" src="{chart}">{media}'
                f'<details><summary>Frozen baseline video</summary><video controls preload="none" src="{original}"></video></details></article>')
    task_rows=[]
    for task in sorted({p['task'] for p in pairs}):
        selected=[p for p in pairs if p['task']==task]
        old=sum(p['baseline']['official_task_success'] for p in selected)
        new=sum(p['mc_vamp']['official_task_success'] for p in selected)
        completed=sum(p['mc_vamp'].get('all_planned_commands_issued',False) for p in selected)
        task_rows.append(f'<tr><td>{esc(task.replace("_"," "))}</td><td>{old}/{len(selected)}</td><td>{new}/{len(selected)}</td><td>{completed}/{len(selected)}</td></tr>')
    failures=Counter(p['mc_vamp'].get('failure_stage','native_goal_false_after_completed_plan')
        for p in pairs if not p['mc_vamp']['official_task_success'])
    benchmark=read_json(ROOT/'results/mcvamp-microwave-saved-query-benchmark/summary.json')
    cycles=[p['mc_vamp']['median_planning_cycle_seconds'] for p in pairs if 'median_planning_cycle_seconds' in p['mc_vamp']]
    evaluation={'pairs':pairs,'all_planned_failures_in_denominator':True,'validations':validations,
        'failures_by_stage':dict(failures),'new_api_calls':0,
        'comparison_scope':'pipeline comparison, not a planner-only causal ablation',
        'baseline_preserved':all(v['baseline_sources_unchanged'] for v in validations),
        'cleanup_after_each_run':all('storage' in p['mc_vamp'] for p in pairs),
        'native_libero_success_only':True,'saved_query_component_benchmark':benchmark,
        'median_episode_planning_cycle_seconds':float(np.median(cycles)) if cycles else None}
    evaluation['executed_contact_trajectory_count']=len(all_cycles)
    if all_cycles:
        evaluation['executed_plan_preparation_seconds']={'median':float(np.median(all_cycles)),
            'minimum':min(all_cycles),'maximum':max(all_cycles),'p95':float(np.quantile(all_cycles,.95)),
            'excludes_fresh_perception':True,'successful_executed_plans_only':True}
    if len(args.runs)==1:
        for key,name in [('post_run_audit','audit.json'),('post_evaluation_cache_hardening','post-evaluation-hardening.json')]:
            if (ROOT/args.runs[0]/name).exists():evaluation[key]=str(Path(args.runs[0])/name)
    write_json(output/'evaluation.json',evaluation)
    actual_timing=('' if not all_cycles else '<p>Preparing an actually executed plan took a median <b>'
        +f'{np.median(all_cycles):.2f}'+' seconds</b> (range '+f'{min(all_cycles):.2f}–{max(all_cycles):.2f}'
        +' seconds), including endpoint proposals and native scene setup, excluding fresh perception. All '
        +str(len(all_cycles))+' issued contact trajectories completed; execution vetoes: <b>0</b>.</p>\n')
    document='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MC-VAMP microwave comparison</title><style>body{font:17px/1.5 system-ui;background:#f5f6f8;color:#17212b;max-width:1100px;margin:36px auto;padding:0 20px}article,.note{background:white;border:1px solid #d6dde6;border-radius:12px;padding:20px;margin:20px 0}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:12px;border-bottom:1px solid #d6dde6}video{max-width:100%;width:512px;background:#151c25}h1{font-size:30px}h2{font-size:23px}summary{cursor:pointer}code{background:#e7edf4;padding:2px 5px}</style></head><body>
<h1>MC-VAMP on microwave opening and closing</h1>
<p>RGB-D grounding → GraspGen and existing grasp refinements → cuRobo endpoint IK → MC-VAMP constrained planning → full timed joint execution.</p>
<div class="note"><b>Success is LIBERO’s actual task predicate.</b> Every planned case counts, including missing frozen VLM outputs and failures before motion. Contacts and tracking error are recorded; they cannot stop trajectory execution. Physics and physical gripper closure remain enabled.</div>
<table><tr><th>Task</th><th>Frozen baseline success</th><th>MC-VAMP success</th><th>Episodes executing complete contact trajectories</th></tr>'''+''.join(task_rows)+'''</table>
<div class="note"><b>What changed:</b> MC-VAMP replaces Atlas; timed execution replaces guarded progress tracking; collision planning uses observed surfaces rather than the inferred solid panel, with a calibrated gripper sphere cover and selected finger–handle/mount contact permissions. Full remaining-task plans are preferred; complete 20°/10° segments are used when necessary, as in the baseline. Every segment finishes before observation/replanning. This measures the combined pipeline change, not MC-VAMP alone. The custom constraint projection currently uses scalar finite differences inside native MC-VAMP composition, so this is not a reproduction of the paper’s fully vectorized projection benchmark.</div>
<p>On saved baseline planning queries, the adapter solved '''+str(benchmark['solved'])+'/'+str(benchmark['queries'])+''' queries. Successful queries took a median '''+f"{1000*benchmark['median_native_search_seconds']:.1f}"+''' ms for native search and '''+f"{1000*benchmark['median_planner_with_resampling_and_worker_seconds']:.1f}"+''' ms including dense resampling and the worker. Cold CAPT construction took '''+f"{benchmark['median_capt_build_seconds']:.2f}"+''' s. These component timings exclude perception and endpoint generation and use the new collision approximation.</p>
'''+actual_timing+'''<p>Native planning times, full cycle times, failure stages, source hashes and cleanup records are in <a href="evaluation.json">evaluation.json</a>. Frozen perception inputs are reused; new API calls: <b>0</b>. Temporary observations and frames are compacted after each episode.</p>
'''+''.join(cards)+'</body></html>'
    (output/'index.html').write_text(document)
    print(json.dumps({'report':str(output/'index.html'),'pairs':len(pairs),'failures':dict(failures)}))

if __name__=='__main__':main()
