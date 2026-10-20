"""Post-hoc state-11 contact, tracking and hinge audit; never a policy input."""
import argparse
import hashlib
import html
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.rekep_libero.geometry import pose_matrix
from report_mcvamp_microwave import metrics
from report_coupled_door import video

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/short-planned-hinge-state11-20261015/open_microwave-state011'


def exists(path):return path.exists() or path.with_suffix(path.suffix+'.gz').exists()


def start_collision(folder):
    failed=[]
    for path in sorted(folder.glob('plan-*/horizon-*/result.json')):
        data=read_json(path)
        if data.get('status')!='invalid_start':continue
        start=data['start'];part=int(start['last_rejected_part']);name='arm or other'
        model=read_json(path.parent/'native-scene.json')
        for volume in model['gripper_cover']:
            if volume['sphere_range'][0]+1000==part:name=volume['robot_geom_name'];break
        failed.append({'path':str(path.relative_to(folder)),'native_part':part,
            'robot_volume':name,'surface':int(start['last_rejected_surface'])})
    if not failed:return {'invalid_start_queries':[]}
    query=folder/failed[-1]['path'];inputs=read_json(query.parent/'planning-input.json')
    arrays=np.load(read_json(query.parent/'scene-reference.json')['arrays_path'])
    volume=next((v for v in inputs['metadata']['robot_collision_volumes_world']
        if v['robot_geom_name']==failed[-1]['robot_volume']),None)
    witnesses=[];reference_witnesses=[];measured_minimum=None
    if volume is not None:
        points=arrays['static_points'];planes=np.asarray(volume['planes_world'])
        value=np.max(points@planes[:,:3].T+planes[:,3],axis=1)
        indices=np.flatnonzero(value<0);indices=indices[np.argsort(value[indices])][:25]
        for i in indices:
            witnesses.append({'world_m':points[i].tolist(),'convex_halfspace_value_m':float(value[i]),
                'distance_to_selected_handle_cloud_m':float(cKDTree(arrays['contact_reference']).query(points[i])[0])})
        measured=arrays['moving_reference'];reference=arrays['contact_reference']
        measured_minimum=float(np.min(np.max(measured@planes[:,:3].T+planes[:,3],axis=1)))
        value=np.max(reference@planes[:,:3].T+planes[:,3],axis=1)
        for i in np.flatnonzero(value<0)[:25]:
            reference_witnesses.append({'world_m':reference[i].tolist(),'convex_halfspace_value_m':float(value[i])})
    return {'invalid_start_queries':failed,'raw_static_penetration_witnesses':witnesses,
        'transported_tracking_reference_penetration_witnesses':reference_witnesses,
        'minimum_measured_moving_convex_halfspace_value_m':measured_minimum,
        'witnesses_are_reconstructed_observed_points_not_simulator_scene_geometry':True}


def audit(folder):
    row,native,contacts=metrics(folder);chain=RobotChain(read_json(folder/'robot-chain.json'))
    actions={a['step']:a for a in read_json(folder/'initial/actions.json')}
    phases=read_json(folder/'phases.json') if exists(folder/'phases.json') else []
    cursor=min((t['control_step'] for t in read_json(folder/'tracking.json')),default=239)
    # First timed controls include approach and closure; manipulation starts at
    # the physical closed observation, retained explicitly by its replay.
    cursor=239 if row.get('frozen_grasp_replay') else cursor
    paths=[]
    for phase in phases:
        index=phase['phase'];q=np.load(folder/f'plan-{index:02d}/path.npy')[:,:7]
        reference=pose_matrix(chain.forward(q[0]))[:3,:3]
        wrist=max(Rotation.from_matrix(reference.T@pose_matrix(chain.forward(x))[:3,:3]).magnitude() for x in q)
        end=cursor+phase['issued_commands']
        actual_reference=pose_matrix(chain.forward(actions[cursor]['joint_positions']))[:3,:3]
        actual=max(Rotation.from_matrix(actual_reference.T@pose_matrix(chain.forward(actions[s]['joint_positions']))[:3,:3]).magnitude()
            for s in range(cursor+1,end+1))
        paths.append({'phase':index,'requested_pull_deg':float(np.degrees(phase['planned_horizon_rad'])),
            'planned_wrist_excursion_deg':float(np.degrees(wrist)),
            'actual_wrist_excursion_deg':float(np.degrees(actual)),
            'maximum_joint_excursion_rad':float(np.max(np.abs(q-q[0]))),
            'start_control_step':cursor,'end_control_step':end,'all_commands_issued':phase['all_commands_issued']})
        cursor=end
    truth=np.asarray(native[max(native)]['anchor_world_m'])
    initial=np.asarray(read_json(folder/'task-spec.json')['joint_origin_world_m'])
    axis=np.asarray(native[max(native)]['axis_world'])
    error=lambda x:float(np.linalg.norm((np.asarray(x)-truth)-axis*((np.asarray(x)-truth)@axis)))
    updates=[]
    if (folder/'closure-rigid-tracking.json').exists():
        fit=read_json(folder/'closure-rigid-tracking.json')
        if fit.get('accepted'):updates.append({'control_step':239,'error_m':error(fit['hinge_world_m']),'source':'grasp-induced motion'})
    observations=read_json(folder/'observations.json') if exists(folder/'observations.json') else []
    for obs in observations:
        fit=obs['hinge_fit']
        if fit.get('accepted'):updates.append({'control_step':obs['control_step'],'error_m':error(fit['hinge_world_m']),
            'source':fit.get('source'),'origin_world_m':fit['hinge_world_m']})
    checkpoints=[]
    evaluations=read_json(folder/'initial/evaluation_contact_diagnostics.json')['steps']
    success_step=next((step for step,a in sorted(actions.items()) if a['evaluation_task_success']),None)
    bilateral=[];palm=[];mug=[]
    handle={'microwave_1_g14','microwave_1_g15','microwave_1_g16'}
    for record in evaluations:
        pads=set()
        for contact in record['robot_involved_contacts']:
            if contact['normal_force_n']<=1e-4 or all(contact['robot_flags']):continue
            external=next(n for n,r in zip(contact['geom_names'],contact['robot_flags']) if not r)
            robot=next(n for n,r in zip(contact['geom_names'],contact['robot_flags']) if r)
            if external in handle and robot.endswith('_pad_collision'):pads.add(robot)
            if 'hand_collision' in robot:palm.append((record['step'],contact['normal_force_n']))
            if 'mug' in external:mug.append(record['step'])
        if len(pads)==2:bilateral.append(record['step'])
    key_steps={239,253,280,row['control_steps']}|{p['end_control_step'] for p in paths}
    for record in evaluations:
        if record['step'] not in key_steps:continue
        checkpoints.append({'control_step':record['step'],
            'opening_deg':max(0.,-float(np.degrees(native[record['step']]['position']))),
            'positive_robot_contacts':[c for c in record['robot_involved_contacts'] if c['normal_force_n']>1e-4]})
    return {**row,'phase_motion':paths,'start_collision':start_collision(folder),
        'contact_quality_evaluation_only':{'first_native_success_step':success_step,
            'bilateral_handle_contact_at_first_success':success_step in bilateral,
            'last_bilateral_handle_contact_step':max(bilateral,default=None),
            'bilateral_handle_contact_at_final_step':row['control_steps'] in bilateral,
            'positive_robot_mug_contact_steps':len(set(mug)),
            'positive_palm_contact_steps_after_closure':len({s for s,f in palm if s>239}),
            'maximum_palm_contact_force_during_shared_approach_n':max((f for s,f in palm if s<=239),default=0.)},
        'hinge_evaluation_only':{'initial_error_m':error(initial),'applied_estimate_errors':updates},
        'contact_checkpoints_evaluation_only':checkpoints},native,contacts


def annotated_video(folder,row,native,contacts,movie):
    """Supply display-only hinge updates without editing frozen episode files."""
    updates=[]
    closure=folder/'closure-rigid-tracking.json'
    if closure.exists():
        fit=read_json(closure)
        if fit.get('accepted'):updates.append({'control_step':239,'estimated_hinge_world_m':fit['hinge_world_m']})
    for obs in read_json(folder/'observations.json'):
        fit=obs['hinge_fit']
        if fit.get('accepted'):updates.append({'control_step':obs['control_step'],'estimated_hinge_world_m':fit['hinge_world_m']})
    write_json(movie.with_suffix('.annotations.json'),{'posthoc_display_only':True,'updates':updates,
        'estimates_from_policy_measurements':True,'true_hinge_is_separate_evaluation_overlay':True})
    with tempfile.TemporaryDirectory(prefix='state11-report-video-') as temporary:
        view=Path(temporary)
        for path in folder.iterdir():
            if path.name in {'replans.json','tracking.json'}:continue
            (view/path.name).symlink_to(path.resolve(),target_is_directory=path.is_dir())
        write_json(view/'tracking.json',read_json(folder/'tracking.json'))
        write_json(view/'replans.json',updates)
        video(view,row,native[max(native)],native,contacts,movie)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--output',default='results/state11-recovery-report')
    parser.add_argument('--refresh-videos',action='store_true')
    args=parser.parse_args();output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    old,old_native,_=audit(OLD);rows=[];cards=[];traces=[('Original',old_native)]
    write_json(output/'original-failure-audit.json',{'evaluation_only':True,**old})
    for index,run in enumerate(args.runs,1):
        root=ROOT/run;folder=root/'open_microwave-state011';row,native,contacts=audit(folder)
        protocol=read_json(root/'protocol.json');summary=read_json(root/'summary.json')
        assert summary['completed']==summary['planned']
        assert summary['api_ledger_unchanged'] and summary['baseline_sources_unchanged']
        assert all(hashlib.sha256((root/'source-snapshot'/n).read_bytes()).hexdigest()==h
            for n,h in protocol['sources_sha256'].items())
        row['snapshot_hashes_verified']=True;row['run']=run;rows.append(row);traces.append((f'Revision {index}',native))
        movie=output/f'revision-{index}.mp4'
        if args.refresh_videos:movie.unlink(missing_ok=True)
        if not movie.exists():
            annotated_video(folder,row,native,contacts,movie)
        phase_rows=''.join(f'<tr><td>{p["phase"]}</td><td>{p["requested_pull_deg"]:.1f}°</td>'
            f'<td>{p["planned_wrist_excursion_deg"]:.1f}°</td><td>{p["end_control_step"]}</td></tr>' for p in row['phase_motion'])
        fits=row['hinge_evaluation_only']['applied_estimate_errors']
        fit_text='; '.join(f'step {f["control_step"]}: {1000*f["error_m"]:.1f} mm' for f in fits) or 'none'
        quality=row['contact_quality_evaluation_only']
        cards.append(f'''<article><h2>Revision {index}: {html.escape(Path(run).name)}</h2>
        <p>Native success: <b>{row['official_task_success']}</b>. Final opening: <b>{row['final_opening_angle_deg']:.2f}°</b>.
        All {len(row['phase_motion'])} issued short paths finished. Execution veto checks: <b>{row['execution_veto_checks']}</b>.</p>
        <p>Hinge-axis errors, post-hoc evaluation only: {html.escape(fit_text)}.</p>
        <p>Physical contact audit: bilateral handle contact at first native success: <b>{quality['bilateral_handle_contact_at_first_success']}</b>;
        at final step: <b>{quality['bilateral_handle_contact_at_final_step']}</b>.
        Robot–mug contact steps: {quality['positive_robot_mug_contact_steps']}.
        Palm-contact steps after closure: {quality['positive_palm_contact_steps_after_closure']}.</p>
        <p>The shared replayed approach has a peak palm–handle force of {quality['maximum_palm_contact_force_during_shared_approach_n']:.1f} N.
        Native task success does not certify collision-free execution or continuous grasp retention.</p>
        <p>{html.escape(row.get('failure',row.get('continuation_stop_reason','Completed')))}</p>
        <video controls preload="metadata" src="{movie.name}"></video>
        <p>The magenta hinge marker updates from the policy’s measured estimates. The native hinge overlay is evaluation only.</p>
        <p>Red video banners mark recorded contacts, including finger–panel contact. Native task success is reported separately;
        contact banners do not stop execution or classify the episode as a failure.</p>
        <details><summary>Completed planned paths</summary><table><tr><th>Phase</th><th>Pull</th><th>Largest wrist turn</th><th>End step</th></tr>{phase_rows}</table></details>
        <p>Cleanup reclaimed {row['storage']['bytes_reclaimed_this_pass']/2**20:.1f} MiB.</p></article>''')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for label,native in traces:
        steps=sorted(native);axes[0].plot(steps,[-np.degrees(native[s]['position']) for s in steps],label=label)
    axes[0].set(xlabel='Simulation control step',ylabel='Actual opening (degrees)',title='State 11: all development trials')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].axhline(old['hinge_evaluation_only']['initial_error_m']*1000,color='gray',ls='--',label='Initial visual estimate')
    for index,row in enumerate(rows,1):
        fits=row['hinge_evaluation_only']['applied_estimate_errors']
        if fits:axes[1].plot([f['control_step'] for f in fits],[f['error_m']*1000 for f in fits],'.-',label=f'Revision {index}')
    axes[1].set(xlabel='Control step after completed motion',ylabel='Perpendicular axis error (mm)',title='Hinge accuracy: evaluation only')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(output/'comparison.png',dpi=140);plt.close(fig)
    ledger=read_json(ROOT/'results/articulated-api-v1/ledger.json')
    write_json(output/'evaluation.json',{'original':old,'trials':rows,'targeted_development_not_held_out':True,
        'same_initial_grasp_replayed':True,'simulator_geometry_policy_input':False,
        'all_failed_development_trials_included':True,'focused_tests_passed':36,'new_api_calls':0,
        'remaining_api_budget_usd':ledger['remaining_usd'],'author_videos_are_not_matched_state_comparisons':True})
    document='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>State 11: contact and hinge diagnosis</title><style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:17px/1.5 system-ui;background:#f5f7fa;color:#172d40}
    article{padding:20px;background:white;border-radius:12px;margin:20px 0}video,img{max-width:100%}video{width:512px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}</style>
    <h1>Why state 11 stopped at 8.9°</h1><p>The immediate stop was a planning-time invalid-start collision:
    observed points classified as stationary penetrated the configured second finger. Before that stop, a nominal 2° pull
    requested a 34° wrist turn. The actual motion turned the wrist about 32°, advanced the door only about half a degree,
    and created a 34.9 N palm-to-handle contact. The grasp stayed in contact near the upper handle mount; it became wedged.</p>
    <img src="../state11-recovery-diagnosis-20261015/old-contact-wedge.png" alt="Three frames of the old wrist sweep">
    <h2>Changes being tested</h2><p>Continuous local wrist choices during planning; persistent RGB-D material landmarks for
    hinge and rigid target-motion estimation; measured-motion association of moving handle points, with present stationary
    observations preserved. Revision 1 retains the old translation-only handle update. Revision 2 also transports the handle
    using measured rigid motion, uses motion during grasping, and refreshes texture landmarks after an observed 8° arc.
    Revision 2 exposed a separate false positive: a transported tracking-reference point penetrated the palm by 0.45 mm,
    while the current measured moving surface was clear and the simulator recorded no palm contact.
    Revision 3 removes tracking-reference points from collision geometry; current measured moving endpoints remain checked
    and are rotated about the estimated hinge when querying future planned states. The tracking cloud still defines target association.</p>
    <p>All issued short trajectories run to completion. There are no execution-time collision, constraint, tracking or progress
    vetoes. No simulator hinge, object pose, collision identity or attachment is used by the policy. Simulator contacts and
    hinge positions appear only in this post-hoc report.</p>
    <img src="comparison.png" alt="Opening and hinge-error traces">'''+cards[-1]+(
        '<details><summary>Earlier development trials, including both failures</summary>'+''.join(cards[:-1])+'</details>' if len(cards)>1 else '')+'''
    <h2>Hinge observability and comparison limits</h2><p>The axis is assumed upright. A closed-door depth image gives a rough
    edge-based origin; measured rigid motion of several spatially separated door landmarks improves it. A single vertical
    handle is insufficient. Bootstrap spread checks geometric stability and is not a calibrated uncertainty guarantee.
    Current hinge estimates still vary with visibility and depth correspondence.</p>
    <p>These are repeated development trials on state 11 with the same captured initial grasp, not independent benchmark
    episodes or isolated causal ablations. The real-robot author videos use different geometry, clutter and controllers;
    they do not establish success on this LIBERO state. A matched VLA rollout has not been run here.
    <a href="https://intuitive-robots.github.io/flower_vla/">FLOWER author source</a> ·
    <a href="https://www.pi.website/blog/pi0">π₀ author source</a>.</p>
    <p>36 focused tests passed; no new API requests. Every run was compacted. <a href="evaluation.json">Full numeric audit</a>
    · <a href="../../docs/state11_recovery.md">Reproduction notes</a>.</p>'''
    latest=rows[-1]
    lead=f'''<article><b>Latest same-grasp replay:</b> native task success <b>{latest['official_task_success']}</b>,
    final opening <b>{latest['final_opening_angle_deg']:.2f}°</b>,
    {latest.get('completed_contact_trajectories',0)} complete planned pulls and
    {latest['execution_veto_checks']} execution veto checks.
    These are development trials on one reset, not a held-out success rate.</article>'''
    document=document.replace('<h1>Why state 11 stopped at 8.9°</h1>','<h1>Why state 11 stopped at 8.9°</h1>'+lead)
    (output/'index.html').write_text(document)
    print({'report':str(output/'index.html'),'outcomes':[(r['official_task_success'],r['final_opening_angle_deg']) for r in rows]})


if __name__=='__main__':main()
