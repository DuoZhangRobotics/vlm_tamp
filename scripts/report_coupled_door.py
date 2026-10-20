"""Separate whole-motion planning, tracking, and native task outcomes."""

import argparse
from collections import Counter
import copy
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import subprocess
from itertools import chain

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from visualize_microwave_hinge_audit import annotation,FONT,COLORS
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.coupled_storage import read_json,video_frames


def read(path):return read_json(path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def video(folder,row,native,native_steps,contacts,output):
    actions={a['step']:a for a in read(folder/'initial/actions.json')}
    initial=read(folder/'initial/metadata.json')
    spec=read(folder/'task-spec.json')
    data={'native':native,'native_steps':native_steps,'spec':spec,'initial':initial,
          'actions':actions,'target':initial['eef_pose_world_xyzw'][:3]}
    tracked={t['control_step'] for t in read(folder/'tracking.json')} if (folder/'tracking.json').exists() else set()
    closure=read(folder/'closure-authorization.json')['control_step'] if (folder/'closure-authorization.json').exists() else None
    observations={o['control_step']:o for o in read(folder/'observations.json')} if (folder/'observations.json').exists() else {}
    updates=[]
    for index,r in enumerate(read(folder/'replans.json') if (folder/'replans.json').exists() else []):
        request=folder/f'plan-{index+1:02d}'/'request.json'
        origin=r.get('estimated_hinge_world_m')
        if origin is None and (request.exists() or request.with_suffix('.json.gz').exists()):origin=read(request)['task_spec']['joint_origin_world_m']
        if origin is None:origin=observations.get(r['control_step'],{}).get('hinge_fit',{}).get('hinge_world_m')
        if origin is not None:updates.append((r['control_step'],origin))
    indexed=folder/'initial/video-frames.json'
    steps=(read(indexed)['control_steps'] if indexed.exists() else
           [int(p.stem[4:]) for p in sorted((folder/'initial/frames').glob('*.jpg'))])
    last=steps[-1] if steps else -1
    tail=[]
    for metadata in folder.glob('*/metadata.json'):
        candidate=metadata.parent/'agentview.png'
        step=read(metadata)['control_steps']
        if candidate.exists() and step==row['control_steps'] and step>last:
            with Image.open(candidate) as image:tail.append((step,image.convert('RGB').copy()))
            last=step
    encoder=subprocess.Popen(['ffmpeg','-nostdin','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','512x644','-framerate','5','-i','pipe:0','-an','-c:v','libx264','-threads','2',
        '-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    try:
        previous=-1
        for step,image in chain(video_frames(folder),tail):
            action=actions.get(step)
            data['target']=(action['target_pose_world_xyzw'][:3] if action else initial['eef_pose_world_xyzw'][:3])
            joint=bool(action and action.get('execution_controller')=='JOINT_POSITION')
            data['spec']=copy.deepcopy(spec)
            for update_step,origin in updates:
                if update_step<=step:data['spec']['joint_origin_world_m']=origin
            phase=('Joint path tracking' if tracked and step>=min(tracked) else
                   'Physical closure' if closure is not None and step>closure else
                   'Planned joint approach' if joint else 'Initial sensing')
            panel=annotation(image,data,initial['cameras']['agentview'],step,
                row['task'].replace('_',' ')+f' state {row["initial_state"]} | '+phase)
            draw=ImageDraw.Draw(panel)
            draw.rectangle((215,26,511,47),fill='#10151e')
            draw.text((215,28),'Native hinge (eval)',font=FONT,fill=COLORS['native'])
            if any(previous<s<=step for s in contacts):
                draw.rectangle((0,0,511,25),fill='#8e2727');draw.text((10,3),'Non-target contact: evaluation only',font=FONT,fill='white')
            draw.rectangle((0,584,511,609),fill='#10151e')
            opening=max(0.,-np.degrees(native_steps.get(step,native)['position']))
            draw.text((10,589),f'Step {step} | Door opening {opening:.1f} deg',font=FONT,fill='white')
            actual=np.asarray(action['eef_pose_world_xyzw'][:3] if action else initial['eef_pose_world_xyzw'][:3])
            error=1000*np.linalg.norm(actual-data['target'])
            draw.rectangle((0,610,511,643),fill='#10151e')
            draw.text((10,614),f'TCP reference error {error:.1f} mm',font=FONT,fill='white')
            encoder.stdin.write(panel.tobytes())
            previous=step
        if previous>=0:
            for _ in range(5):encoder.stdin.write(panel.tobytes())
        encoder.stdin.close()
        if encoder.wait(timeout=60)!=0:raise RuntimeError('Video encoding failed')
    finally:
        if encoder.poll() is None:encoder.kill();encoder.wait()


def figure(folder,row,output):
    planned=folder/'pregrasp-complete-path.npy'
    if not planned.exists():return None
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    chain=RobotChain(read(folder/'robot-chain.json'))
    paths=[planned,*sorted(folder.glob('plan-*/path.npy'))]
    trajectories=[np.array([chain.forward(x[:7])[:3] for x in np.load(path)]) for path in paths]
    xyz=trajectories[0]
    actions=read(folder/'initial/actions.json')
    tracking_steps={t['control_step'] for t in read(folder/'tracking.json')} if (folder/'tracking.json').exists() else set()
    actual=np.array([a['eef_pose_world_xyzw'][:3] for a in actions if a['step'] in tracking_steps])
    fig=plt.figure(figsize=(7.4,5.2),layout='constrained');ax=fig.add_subplot(111,projection='3d')
    for index,points in enumerate(trajectories):
        ax.plot(*points.T,label='Checked plans (including replaced tails)' if index==0 else None,color='#2679bc',alpha=.7)
    if len(actual):ax.plot(*actual.T,label='Executed manipulation TCP',color='#d36037')
    spec=read(folder/'task-spec.json');hinge=np.asarray(spec['joint_origin_world_m'])
    axis=np.asarray(spec['joint_axis_world']);segment=hinge+np.array([-.12,.12])[:,None]*axis
    ax.plot(*segment.T,'--',color='#68717c',label='Initial estimated hinge')
    ax.scatter(*xyz[0],s=35,color='#2679bc');ax.scatter(*xyz[-1],s=45,marker='*',color='#2679bc')
    ax.set(xlabel='World x (m)',ylabel='World y (m)',zlabel='World z (m)',title='Prepared motion and executed TCP')
    ax.legend(fontsize=8);fig.savefig(output,dpi=150);plt.close(fig)
    return output.name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    ledger=read(ROOT/'results/articulated-api-v1/ledger.json')
    budget_ok=ledger['budget_usd']==2. and ledger['cumulative_conservative_spend_usd']<=2.+1e-12
    rows=[];cards=[];validations=[];protocols=[];keys=set()
    # These pinned asset identifiers label evaluation contacts only. They are
    # never passed to a policy, a grasp selector or a collision checker.
    handle_geoms={'microwave_1_g14','microwave_1_g15','microwave_1_g16'}
    for run_name in args.runs:
        run=ROOT/run_name;manifest=read(run/'manifest.json');summary=read(run/'summary.json');protocol=read(run/'protocol.json')
        protocols.append(protocol)
        if summary['completed']!=len(manifest['planned_episodes']):raise ValueError('A planned episode is incomplete')
        hashes=all(digest(run/'source-snapshot'/p)==h for p,h in protocol['sources_sha256'].items())
        unchanged=digest(ROOT/'results/articulated-api-v1/ledger.json')==protocol['api_ledger_sha256_before']
        api_ok=budget_ok and (protocol.get('paid_grounding_allowed',False) or unchanged or summary.get('api_ledger_unchanged') is True)
        validations.append({'run':run_name,'frozen_sources_match':hashes,'api_ledger_unchanged':unchanged,'api_cost_policy_satisfied':api_ok})
        for entry in manifest['planned_episodes']:
            task,index=entry['task'],entry['state'];key=(task,index)
            if key in keys:raise ValueError('Duplicate task/initial state in the report')
            keys.add(key);folder=run/f'{task}-state{index:03d}'
            row=read(folder/'summary.json');row.update(initial_state=index,split='development' if index in {0,1} else 'held_out')
            actions=read(folder/'initial/actions.json') if (folder/'initial/actions.json').exists() else []
            evaluations=read(folder/'initial/evaluation_contact_diagnostics.json')['steps'] if (folder/'initial/evaluation_contact_diagnostics.json').exists() else []
            native_steps={e['step']:next(j for j in e['evaluation_only_articulation_state'] if j['joint_name']=='microwave_1_microjoint') for e in evaluations}
            flags=[bool(a['evaluation_task_success']) for a in actions]
            replay=None
            if actions and all(a['step'] in native_steps for a in actions):
                replay=[native_steps[a['step']]['position']<-1.3 if task=='open_microwave' else native_steps[a['step']]['position']>-.005 for a in actions]
                if replay!=flags:raise ValueError('Native predicate replay differs')
            contacts=set();pairs=Counter();target_pairs=Counter();self_steps=set();palm_peak=0.;two_pads=0
            for e in evaluations:
                pads=set();palm_force=0.
                for c in e['robot_involved_contacts']:
                    if c['normal_force_n']<=.1:continue
                    if sum(c['robot_flags'])==2:self_steps.add(e['step']);continue
                    if sum(c['robot_flags'])!=1:continue
                    external=next(n for n,r in zip(c['geom_names'],c['robot_flags']) if not r)
                    robot_name=next(n for n,r in zip(c['geom_names'],c['robot_flags']) if r)
                    intended=external in handle_geoms and 'finger' in robot_name
                    if not intended:contacts.add(e['step']);pairs[' / '.join(c['geom_names'])]+=1
                    else:
                        target_pairs[' / '.join(c['geom_names'])]+=1
                        if robot_name.endswith('_pad_collision'):pads.add(robot_name)
                    if robot_name.endswith('hand_collision') and external.startswith('microwave_1_'):palm_force+=c['normal_force_n']
                palm_peak=max(palm_peak,palm_force);two_pads+=len(pads)==2
            tracks=read(folder/'tracking.json') if (folder/'tracking.json').exists() else []
            grasps=read(folder/'grasp-search.json') if (folder/'grasp-search.json').exists() else {}
            plans=[read(p) for p in folder.glob('**/solution/result.json')]
            final_native=native_steps.get(actions[-1]['step']) if actions else None
            row.update(episode=str(folder.relative_to(ROOT)),native_success_ever=any(flags),native_success_final=bool(flags and flags[-1]),
                first_native_success_step=next((a['step'] for a in actions if a['evaluation_task_success']),None),
                native_final_joint_position_rad=final_native['position'] if final_native else None,
                native_final_open_angle_deg=max(0.,-float(np.degrees(final_native['position']))) if final_native else None,
                native_predicate_replay_matches=(replay==flags) if replay is not None else None,
                non_target_positive_contact_steps=len(contacts),non_target_contact_pairs=dict(pairs),
                intended_handle_region_contact_pairs=dict(target_pairs),native_self_contact_steps=len(self_steps),
                maximum_sampled_palm_target_normal_force_n=palm_peak,two_pad_positive_contact_steps=int(two_pads),
                initial_scene_model=read(folder/'initial-scene.json') if (folder/'initial-scene.json').exists() else None,
                joint_controller_controls=sum(a.get('execution_controller')=='JOINT_POSITION' for a in actions),
                coupled_tracking_controls=len(tracks),constrained_solver_attempts=len(plans),
                exact_complete_plans=sum(bool(p.get('exact_solution')) for p in plans),
                tracking_maximum_tube_error_rad=max((t['path_tube_error_rad'] for t in tracks),default=None),
                tracking_maximum_point_residual_m=max((t['model_errors']['position_error_m'] for t in tracks),default=None),
                tracking_maximum_jaw_error_rad=max((t['model_errors']['jaw_axis_error_rad'] for t in tracks),default=None),
                tracking_maximum_jaw_tilt_rad=max((t['model_errors'].get('measured_jaw_tilt_rad',t['model_errors']['jaw_axis_error_rad']) for t in tracks),default=None),
                planning_buffer_violation_steps=sum(not t.get('planning_buffer_accepted',True) for t in tracks),
                grasp_admission_gate_counts=grasps.get('gate_counts',{}),grasp_failure_classification=grasps.get('failure_classification'))
            stem=f'{task}-state{index:03d}';movie=stem+'.mp4'
            if actions and native_steps and (folder/'task-spec.json').exists():
                video(folder,row,native_steps[min(native_steps)],native_steps,contacts,out/movie)
            plot=figure(folder,row,out/(stem+'-plan.png')) if (folder/'robot-chain.json').exists() else None
            rows.append(row)
            cards.append(f'<details {"open" if index==1 else ""}><summary>{html.escape(task)} · state {index} · {row["split"]} · native {"success" if any(flags) else "failure"}</summary>'
                f'<p>{html.escape(row.get("failure") or row.get("status",""))}</p>'
                + (f'<video controls preload="metadata" src="{movie}"></video>' if (out/movie).exists() else '')
                + (f'<img src="{plot}">' if plot else '')+'</details>')
    same=all(p['sources_sha256']==protocols[0]['sources_sha256'] for p in protocols)
    aggregates=[]
    for task in sorted({r['task'] for r in rows}):
        for split in ['development','held_out']:
            group=[r for r in rows if r['task']==task and r['split']==split]
            if group:aggregates.append({'task':task,'split':split,'planned':len(group),
                'native_successes':sum(r['native_success_ever'] for r in group),
                'final_native_successes':sum(r['native_success_final'] for r in group),
                'completed_planned_trajectories':sum(r['status']=='completed_planned_trajectory' for r in group)})
    result={'development_only':all(r['split']=='development' for r in rows),'held_out_task_generalization_claim':False,
        'planned':len(rows),'native_successes':sum(r['native_success_ever'] for r in rows),'episodes':rows,
        'aggregates':aggregates,'same_frozen_pipeline_sources':same,'integrity_checks':validations,
        'new_api_calls':sum(r.get('new_api_calls',0) for r in rows),
        'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd'],'remaining_api_budget_usd':ledger['remaining_usd'],
        'report_source_sha256':digest(__file__),
        'report_renderer_updated_after_data_collection':any(p['sources_sha256'].get('scripts/report_coupled_door.py')!=digest(__file__) for p in protocols),
        'native_contact_evaluation_only':{'intended_finger_contact_geoms':sorted(handle_geoms),
            'source':'pinned LIBERO microwave handle/bar mounting region; never supplied to policy'},
        'limitations':['State 1 is development; states 11, 21, 31 and 41 are unseen initial states of the same tasks',
            'Aligned contact uses four equalities; bounded-tilt contact uses three position equalities and a jaw-tilt inequality. Capture does not certify force closure',
            'Complete bounded contact segments are planned before execution; a full task path is not required before grasp',
            'Observed surfaces and selected-panel completion do not certify hidden geometry or unknown space',
            'Twelve calibrated RGB-D cameras include additional arm views',
            'Moving door versus environment swept-volume collision is not certified',
            'Physics contacts are sampled at 20Hz, not every physics substep',
            'Simulation pauses during planning; this is not a real-time hardware evaluation']}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    summary_table=''.join(f'<tr><td>{a["task"]}</td><td>{a["split"]}</td><td>{a["native_successes"]}/{a["planned"]}</td><td>{a["final_native_successes"]}/{a["planned"]}</td><td>{a["completed_planned_trajectories"]}/{a["planned"]}</td></tr>' for a in aggregates)
    episode_table=''.join(f'<tr><td>{r["task"]}</td><td>{r["initial_state"]}</td><td>{r["split"]}</td><td>{r["native_success_ever"]} / {r["native_success_final"]}</td><td>{r["control_steps"]}</td><td>{r["non_target_positive_contact_steps"]}</td><td>{html.escape(r.get("failure_stage") or r["status"])}</td></tr>' for r in rows)
    spend=ledger['cumulative_conservative_spend_usd']
    (out/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>RGB-D planning and feedback evaluation</title>'
        '<style>body{font:17px system-ui;max-width:1150px;margin:24px auto;padding:20px;line-height:1.5}video,img{max-width:100%;width:780px;display:block;margin:20px 0}td,th{padding:9px;text-align:left;border-bottom:1px solid #ccd}summary{cursor:pointer;font-weight:650;padding:12px}details{border:1px solid #ddd;margin:16px 0;padding:10px}</style>'
        '<h1>RGB-D grounding → constrained planning → physical control</h1>'
        '<p>Luna-high supplies RGB role and hinge-edge choices, with observation-validated caching. DINOv2/MobileSAM and measured depth provide perception; GraspGen proposals use each current predicted handle cloud and are combined with calibrated RGB-D frontal grasp templates. No provided simulator object inventory is sent to the VLM.</p>'
        '<p>OMPL Atlas/RRTConnect plans seven arm joints plus door angle. The grasp point follows the observed hinge arc. Aligned-jaw plans are preferred; a bounded-tilt contact mode is available when needed, with capture and retention checks. The planner uses a menu of heights, orientations and IK postures, including partial-constraint goal IK. Joint-position control executes the checked paths through physical dynamics.</p>'
        '<p>The runner prefers a complete task path, and uses checked bounded segments when the current model cannot support one. Feedback replanning can reduce the segment from 20° to 10° when needed. It observes motion and replans. The overall opening target stays 90°. The initial short segment supplies motion for hinge refinement. Numerical constraint checks use 10 micrometres; physical tracking has a separately recorded allowance. Hierarchical VLM decisions are deferred.</p>'
        '<p>Panel completion is bounded by selected RGB-D plane pixels; handle/mount observations cannot enlarge that inferred footprint. Its shape then moves rigidly. All measured static and moving surfaces remain included. Planning margins are 3mm for arms and 2mm for tools; tracking uses a separate 1mm guard. Intended finger contact is confined to the selected observed handle region. An observed-target-in-pad-channel check validates grasp binding; continued co-motion is required.</p>'
        '<p>Task success is LIBERO’s actual predicate. Success at any step, success in the final recorded state and completing the planned trajectory are reported separately. Every planned case remains in the denominator. State 1 is development and is separated from the four unseen initial states; these are the same task types.</p>'
        f'<p>Shared cumulative API accounting: ${spend:.6f} of $2.00. New calls in these runs: {result["new_api_calls"]}. No automatic API retries.</p>'
        '<h2>Outcomes</h2><table><tr><th>Task</th><th>Split</th><th>Native ever</th><th>Native final</th><th>Planned trajectory completed</th></tr>'+summary_table+'</table>'
        '<table><tr><th>Task</th><th>State</th><th>Split</th><th>Native ever/final</th><th>Controls</th><th>Non-handle contact steps</th><th>Stop phase</th></tr>'+episode_table+'</table>'
        '<p>Contact labels are evaluation-only. The native asset’s handle/bar mounting region is distinguished from the cabinet, plate, table and other objects. Force threshold: 0.1N at control ticks. Native hinge overlays in cyan are evaluation-only; magenta is the estimate used for planning.</p>'
        +''.join(cards)+'<h2>Measurements and reproduction</h2><p><a href="evaluation.json">Full measurements, source validation and limitations</a></p>'
        '<p>run_coupled_door.py supports task and official initial-state selections. evaluate_coupled_states.py freezes tested sources, reuses the marked development measurements and runs the unseen states without per-state retuning. Models, inputs, constraints, proposals, trajectories and failure reasons are retained in each run.</p>'
        '<p>Observed surfaces and inferred panels do not certify hidden geometry, continuous collisions or the moving door’s collision with the environment. Simulation advances on actions and pauses for planning.</p></html>')
    if not same or not all(v['frozen_sources_match'] and v['api_cost_policy_satisfied'] for v in validations):
        raise ValueError('Report integrity check failed')
    print(json.dumps({'planned':len(rows),'native_successes':result['native_successes'],'aggregates':aggregates}))


if __name__=='__main__':main()
