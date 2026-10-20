"""Report the paired five-task articulated pilot, including failed stages."""

import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.rekep_libero.geometry import interpolate_path
from vlm_tamp.rekep_libero.mechanism_geometry import Mechanism,project


def read(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def constraint_figure(folder,output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from PIL import Image
    spec=read(folder/'task-spec.json')
    if spec is None:
        return None
    mechanism=Mechanism(spec)
    camera=read(folder/'opened/metadata.json')['cameras'][spec['camera']]
    chosen=read(folder/'chosen-grasp.json',{})
    pose=chosen.get('fk_pose_world_xyzw',spec['source_anchor_world_m']+[0,0,0,1])
    path=mechanism.path(pose)
    straight=interpolate_path([pose,path[-1]],position_step=.005,rotation_step=.03)
    curved_uv=project(np.asarray(path)[:,:3],camera)
    straight_uv=project(straight[:,:3],camera)
    figure,axes=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
    axes[0].imshow(Image.open(folder/'opened'/f'{spec["camera"]}.png'))
    axes[0].plot(curved_uv[:,0],curved_uv[:,1],color='#1b9e77',linewidth=2,label='Estimated mechanism path')
    axes[0].plot(straight_uv[:,0],straight_uv[:,1],'--',color='#d95f02',linewidth=1.5,label='Endpoint-only chord')
    axes[0].scatter(curved_uv[0,0],curved_uv[0,1],s=25,c='#efbd35',label='Contact anchor')
    axes[0].set(xlim=(0,512),ylim=(512,0),title='Estimated path overlay — not an execution trace')
    axes[0].axis('off')
    axes[0].legend(fontsize=7,loc='lower left')
    progress=np.linspace(0,1,len(straight))
    residual=np.array([mechanism.residual(pose,p)['position_residual_m'] for p in straight])*1000
    axes[1].plot(progress,residual,color='#d95f02',label='Straight chord residual')
    axes[1].axhline(15,color='#999',linestyle=':',label='15 mm model tolerance')
    axes[1].set(xlabel='Fraction of endpoint-only path',ylabel='Estimated manifold residual (mm)',
                title='Endpoint agreement does not enforce the interior path')
    axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
    figure.savefig(output,dpi=140)
    plt.close(figure)
    return {'maximum_endpoint_only_position_residual_m':float(residual.max()/1000),
            'constraint_family':spec['constraint']['type'],'executed_ablation':False}


def geometry_evaluation(folder):
    """Native joint axes are read only here, after all policy execution."""
    spec=read(folder/'task-spec.json')
    diagnostics=read(folder/'initial/evaluation_contact_diagnostics.json',{})
    steps=diagnostics.get('steps',[])
    if not spec or not steps:
        return None
    label=read(folder/'summary.json')['task']
    joints=steps[0].get('evaluation_only_articulation_state',[])
    category='microwave' if 'microwave' in label else 'stove' if 'stove' in label else 'wooden_cabinet'
    candidates=[j for j in joints if category in j['joint_name']]
    if not candidates:
        return {'available':False,'evaluation_only':True}
    point=np.asarray(spec['source_anchor_world_m'])
    actual=min(candidates,key=lambda j:np.linalg.norm(np.cross(point-np.asarray(j['anchor_world_m']),j['axis_world'])))
    axis=np.asarray(actual['axis_world']);axis/=np.linalg.norm(axis)
    predicted=np.asarray(spec['joint_axis_world']);predicted/=np.linalg.norm(predicted)
    parallel_angle=float(np.degrees(np.arccos(np.clip(abs(axis@predicted),0,1))))
    pivot=np.asarray(spec['joint_origin_world_m'])
    final=next((j for j in steps[-1].get('evaluation_only_articulation_state',[]) if j['joint_name']==actual['joint_name']),actual)
    return {'available':True,'evaluation_only':True,'excluded_from_policy_input':True,
        'native_joint_name':actual['joint_name'],'native_type':actual['type'],
        'predicted_type':spec['joint_family'],'type_matches':(actual['type']=='slide')==(spec['joint_family']=='prismatic'),
        'unsigned_axis_error_deg':parallel_angle,
        'predicted_pivot_distance_to_native_hinge_line_m':(float(np.linalg.norm(np.cross(pivot-np.asarray(actual['anchor_world_m']),axis)))
                                                         if actual['type']=='hinge' and spec['joint_family']=='revolute' else None),
        'native_initial_position':actual['position'],'native_final_position':final['position'],
        'native_joint_motion':float(final['position']-actual['position'])}


def physical_contacts(folder,label):
    diagnostics=read(folder/'initial/evaluation_contact_diagnostics.json',{})
    if diagnostics and (not diagnostics.get('evaluation_only') or not diagnostics.get('excluded_from_policy_input')):
        raise ValueError('Contact labels must be evaluation-only')
    category='microwave' if 'microwave' in label else 'stove' if 'stove' in label else 'wooden_cabinet'
    pairs,other,peaks=Counter(),Counter(),{}
    for step in diagnostics.get('steps',[]):
        for item in step['robot_involved_contacts']:
            if item['normal_force_n']<=1e-6:
                continue
            pair=' / '.join(item['geom_names'])
            pairs[pair]+=1
            peaks[pair]=max(peaks.get(pair,0.),item['normal_force_n'])
            if sum(item['robot_flags'])==1:
                robot=next(n for n,b in zip(item['geom_names'],item['robot_flags']) if b)
                target=next(n for n,b in zip(item['geom_names'],item['robot_flags']) if not b)
                if 'finger' not in robot or category not in target:
                    other[pair]+=1
    return {'evaluation_only':True,'positive_force_contact_entries':sum(pairs.values()),
        'non_finger_or_other_object_entries':dict(other),'peak_normal_force_n':max(peaks.values(),default=0.),
        'pairs':dict(pairs),'object_labels_do_not_certify_selected_local_region':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict',default='results/articulated-strict-final')
    parser.add_argument('--diagnostic',default='results/articulated-physics-pilot')
    parser.add_argument('--output',default='results/articulated-libero-report')
    args=parser.parse_args()
    output=ROOT/args.output
    output.mkdir(parents=True,exist_ok=False)
    roots={'strict':ROOT/args.strict,'diagnostic':ROOT/args.diagnostic}
    protocols={mode:read(base/'protocol.json') for mode,base in roots.items()}
    if protocols['strict']['sources_sha256']!=protocols['diagnostic']['sources_sha256']:
        raise ValueError('Paired runs must use the same frozen source files')
    all_rows,table,cards=[],[],[]
    def link(path,label):
        return f'<a href="{html.escape(os.path.relpath(path,output),quote=True)}">{html.escape(label)}</a>'
    pairs={}
    for mode,base in roots.items():
        manifest=read(base/'manifest.json')
        for planned in manifest['planned_episodes']:
            label,state=planned['task'],planned['state']
            folder=base/f'{label}-state{state:03d}'
            summary=read(folder/'summary.json')
            if summary is None:
                raise ValueError('Every planned episode must finish before report generation')
            actions=read(folder/'initial/actions.json',[])
            native=any(a['evaluation_task_success'] for a in actions)
            if native!=summary['official_task_success'] or len(actions)!=summary['control_steps'] or len(actions)>600:
                raise ValueError('Native predicate or control-budget evidence disagrees with summary')
            admission=read(folder/'grasp-admission.json',{})
            row={**summary,'folder':str(folder.relative_to(ROOT)),
                 'native_success_recomputed':native,'reached_physical_closure':(folder/'closed').is_dir(),
                 'rejection_counts':dict(Counter(r['reason'] for r in admission.get('rejected_candidates',[]))),
                 'moving_panel_probe':read(folder/'moving-panel-probe.json'),
                 'native_geometry_evaluation':geometry_evaluation(folder),
                 'physical_contact_evaluation':physical_contacts(folder,label)}
            native_geometry=row['native_geometry_evaluation']
            row['retention_flag_discrepancy']=bool(summary['grasp_retention_verified'] and native_geometry
                and native_geometry.get('available') and abs(native_geometry['native_joint_motion'])<
                (.008 if summary.get('joint_family')=='prismatic' else .05))
            all_rows.append(row)
            pairs.setdefault((label,state),{})[mode]=row
            assets=output/f'{mode}-{label}-state{state:03d}'
            assets.mkdir()
            figure=assets/'estimated-constraint.png'
            comparison=constraint_figure(folder,figure)
            row['endpoint_only_geometry_comparison']=comparison
            video=folder/'initial/rollout.mp4'
            reason=summary.get('failure') or ('Native goal reached.' if native else 'Commands completed without native task success.')
            retention_note=('<p><b>Retention audit:</b> an earlier feedback window set the internal retention flag, '
                'but post-run native joint motion was negligible. This flag is not reliable terminal grasp evidence; '
                'the native task outcome remains failed.</p>' if row['retention_flag_discrepancy'] else '')
            details=[link(folder/'summary.json','Outcome'),link(folder/'command-log.json','Commands'),
                     link(folder/'initial/actions.json','Native success log')]
            for filename,caption in [('task-spec.json','Measured specification'),('grounding/response.txt','Luna choices'),
                ('grounding/prompt.txt','Exact prompt'),('grasp-admission.json','Grasp/IK admission'),
                ('moving-panel-probe.json','Moving geometry admission'),('mechanism-feedback.json','Observed mechanism motion'),
                ('initial/evaluation_contact_diagnostics.json','Evaluation-only physical contacts'),
                ('collision-prediction-events.json','Predicted collision witnesses')]:
                if (folder/filename).is_file():details.append(link(folder/filename,caption))
            image=f'<img class="curve" src="{assets.name}/{figure.name}">' if comparison else ''
            video_html=f'<video controls preload="metadata" src="{html.escape(os.path.relpath(video,output))}"></video>' if video.is_file() else ''
            cards.append(f'<article><h2>{html.escape(label.replace("_"," "))} · {mode} · state {state}: '
                f'<span class="{"success" if native else "failed"}">{"SUCCESS" if native else "FAILED"}</span></h2>'
                f'<p>{html.escape(reason)}</p><p>Stop stage: {html.escape(summary.get("failure_stage") or "completed commands")}; '
                f'{summary["control_steps"]}/600 controls. Physical closure reached: {row["reached_physical_closure"]}.</p>'
                f'{retention_note}{video_html}{image}<p>{" · ".join(details)}</p>'
                f'<details><summary>Post-execution numeric geometry evaluation</summary><pre>{html.escape(json.dumps(row["native_geometry_evaluation"],indent=2))}</pre></details></article>')
    for (label,state),modes in pairs.items():
        strict,diagnostic=modes['strict'],modes['diagnostic']
        table.append(f'<tr><td>{html.escape(label.replace("_"," "))}</td><td>{state}</td>'
            f'<td>{"success" if strict["official_task_success"] else "failed"}</td>'
            f'<td>{"success" if diagnostic["official_task_success"] else "failed"}</td>'
            f'<td>{html.escape(diagnostic.get("failure_stage") or "completed commands")}</td></tr>')
    counts={mode:{'planned':sum(r['scene_mode']==mode for r in all_rows),
                  'native_successes':sum(r['scene_mode']==mode and r['official_task_success'] for r in all_rows),
                  'physical_closures':sum(r['scene_mode']==mode and r['reached_physical_closure'] for r in all_rows)} for mode in roots}
    ledger=read(ROOT/'results/articulated-api-v1/ledger.json')
    data={'task_state_pairs':len(pairs),'episodes':all_rows,'modes':counts,
        'one_initial_state_per_task':True,'held_out_task_generalization_claim':False,
        'new_model_requests_in_paired_diagnostic':0,'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd'],
        'remaining_api_budget_usd':ledger['remaining_usd'],'collision_free_guarantee':False,
        'report_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'evaluation.json').write_text(json.dumps(data,indent=2)+'\n')
    text=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Articulated LIBERO pilot</title><style>body{{font-family:system-ui,sans-serif;background:#f2f5f8;color:#18232f;max-width:1140px;margin:auto;padding:28px;line-height:1.55}}article,.box{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:22px;margin:22px 0}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dce3eb}}video{{width:100%;max-height:510px;background:#121820}}img.curve{{width:100%}}.success{{color:#087c50}}.failed{{color:#b43c4b}}a{{color:#1b5d9c}}pre{{overflow:auto;font-size:.85rem}}</style></head><body>
<h1>Five new articulated LIBERO tasks</h1><p>Native success: strict admission <b>{counts['strict']['native_successes']}/{counts['strict']['planned']}</b>; paired physics diagnostic <b>{counts['diagnostic']['native_successes']}/{counts['diagnostic']['planned']}</b>.</p>
<div class="box"><p>This is an exploratory pilot on one official initial state per task, with development on state 0 and evaluation on state 1. Every planned episode counts, including failures before grasping. The earlier 10/10 bowl/middle-drawer result is a different task set.</p>
<p>Luna high fills bounded RGB role and mechanism choices using two raw/numbered views. MobileSAM and metric depth supply contact/panel regions and numeric estimates. Prismatic tasks use a line; revolute tasks use a circle with coupled gripper orientation. GraspGen and multiple cuRobo IK postures supply grasps. Short verified probes are replanned from observed RGB-D feedback.</p>
<p>DINO points assist the role decision. Execution refines the VLM's selected RGB contact pixel rather than automatically snapping it onto a nearby mounting stem, then fuses observed protruding handle surfaces inside its predicted region.</p>
<p><b>Strict admission:</b> predicted scene overlaps veto motion. <b>Physics diagnostic:</b> the same scene predictions remain logged but do not veto commands. Both modes retain robot self checks, joint limits, estimated manifold checks, a 600-step budget, and native simulator contact physics. The diagnostic is not a collision-safe planner.</p>
<p>The paired runs reuse verified RGB grounding and GraspGen candidates. Simulator object/joint data are excluded from policy inputs; native axes are read only for the post-execution error analysis below. Constraint overlays and chord comparisons are estimates, not executed physical ablations.</p>
<p>Kitchen scenes exposed unnamed-geometry gaps in the wrapper's name list. Resolving known robot names to actual IDs fixes their collision/self-depth geometry. The older bowl/middle-drawer scenes have no such mismatch; their frozen sources and results remain preserved.</p></div>
<table><tr><th>Task</th><th>State</th><th>Strict</th><th>Physics diagnostic</th><th>Diagnostic stop stage</th></tr>{''.join(table)}</table>
<p>Cumulative conservative model API spend: ${ledger['cumulative_conservative_spend_usd']:.6f}; remaining ${ledger['remaining_usd']:.6f} under the $2 cap. No automatic API retries.</p>
<p>{link(output/'evaluation.json','Measurements')} · {link(ROOT/'docs/articulated_libero.md','Protocol and limitations')} · <a href="https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master/libero/libero/bddl_files/libero_90">Official LIBERO task definitions</a></p>
{''.join(cards)}</body></html>'''
    (output/'index.html').write_text(text)
    print(json.dumps({'modes':counts,'episodes':len(all_rows),'cumulative_api_spend_usd':data['cumulative_api_spend_usd']}))


if __name__=='__main__':
    main()
