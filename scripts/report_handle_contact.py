"""Native outcomes and calibrated visualizations for handle-contact repairs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from visualize_microwave_hinge_audit import annotation, FONT
from try_rekep_bowl_v3 import read_observation_v3
from vlm_tamp.handle_contact import translated_surface


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_video(data,path,minimum_step=0):
    encoder=subprocess.Popen(['ffmpeg','-nostdin','-loglevel','error','-f','rawvideo',
        '-pixel_format','rgb24','-video_size','512x644','-framerate','4','-i','pipe:0',
        '-an','-c:v','libx264','-preset','veryfast','-threads','2','-crf','19',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    try:
        for frame in data['frames']:
            step=int(frame.stem[4:])
            if step<minimum_step:
                continue
            target=(data['actions'][step]['target_pose_world_xyzw'][:3] if step else data['initial']['eef_pose_world_xyzw'][:3])
            current={**data,'target':target}
            with Image.open(frame) as rgb:
                panel=annotation(rgb,current,data['initial']['cameras']['agentview'],step,
                                 'RGB-D feedback | '+data['summary']['task'].replace('_',' '))
            actual=data['actions'][step] if step else data['initial']
            error=1000*float(np.linalg.norm(np.asarray(actual['eef_pose_world_xyzw'][:3])-target))
            gap=1000*sum(abs(v) for v in actual['gripper_qpos'])
            draw=ImageDraw.Draw(panel)
            draw.rectangle((0,610,511,643),fill='#10151e')
            draw.text((10,614),f'TCP command error {error:.1f} mm | Finger gap {gap:.1f} mm',font=FONT,fill='white')
            if step in data.get('non_target_contact_steps',set()):
                draw.rectangle((0,0,511,25),fill='#8e2727')
                draw.text((10,3),'Non-target contact (evaluation audit)',font=FONT,fill='white')
            encoder.stdin.write(panel.tobytes())
        encoder.stdin.close()
        if encoder.wait(timeout=60)!=0:
            raise RuntimeError('Video encoding failed')
    finally:
        if encoder.poll() is None:
            encoder.kill();encoder.wait()


def hinge_plot(folder, feedback, origin, initial, output):
    if not feedback or not (folder/'closed/metadata.json').exists():
        return None
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    cloud=np.load(folder/'handle-cloud.npy')
    cameras,metadata=read_observation_v3(folder/'closed',folder/'closed/metadata.json')
    closure=translated_surface(cloud,cameras,metadata['robot_collision_volumes_world'],np.zeros(3))
    centre=np.median(cloud,axis=0)+np.asarray(closure['translation_world_m'])
    centres=[centre.copy()]
    for item in feedback:
        centre=centre+np.asarray(item['translation_world_m']);centres.append(centre.copy())
    centres=np.asarray(centres)
    figure,axes=plt.subplots(figsize=(5.8,4.6),constrained_layout=True)
    axes.plot(centres[:,0],centres[:,1],'.-',label='Observed RGB-D handle motion',color='#2563a5')
    axes.scatter(initial[0],initial[1],marker='x',s=80,color='#bd3bc5',label='Initial visual hinge')
    axes.scatter(origin[0],origin[1],marker='*',s=100,color='#222',label='Native hinge (evaluation only)')
    fitted=next((f['online_hinge_fit'] for f in reversed(feedback) if f['online_hinge_fit']['accepted']),None)
    if fitted:
        pivot=np.asarray(fitted['hinge_world_m'])
        angle=np.unwrap(np.arctan2(centres[:,1]-pivot[1],centres[:,0]-pivot[0]))
        sample=np.linspace(angle.min(),angle.max(),100)
        axes.plot(pivot[0]+fitted['radius_m']*np.cos(sample),pivot[1]+fitted['radius_m']*np.sin(sample),
                  '--',color='#128b5b',label='Circle fitted from observed motion')
        axes.scatter(pivot[0],pivot[1],s=45,color='#128b5b',label='Final fitted hinge')
    axes.set(xlabel='World x (m)',ylabel='World y (m)',title='Hinge refinement from handle motion')
    axes.axis('equal');axes.grid(alpha=.2);axes.legend(fontsize=7)
    figure.savefig(output,dpi=155);plt.close(figure)
    return str(output.name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',default='results/handle-contact-fix-report')
    args=parser.parse_args()
    base,output=ROOT/args.run,ROOT/args.output
    manifest,protocol=read(base/'manifest.json'),read(base/'protocol.json')
    planned=manifest['planned_episodes']
    if read(base/'summary.json')['completed']!=len(planned):
        raise ValueError('All planned development trials must finish before reporting')
    output.mkdir(parents=True,exist_ok=False)
    rows,cards,stills=[],[],[]
    for task in planned:
        folder=base/f'{task["task"]}-state001'
        summary=read(folder/'summary.json')
        actions=read(folder/'initial/actions.json')
        diagnostics=read(folder/'initial/evaluation_contact_diagnostics.json')['steps']
        native_steps={s['step']:next(j for j in s['evaluation_only_articulation_state']
                     if j['joint_name']=='microwave_1_microjoint') for s in diagnostics}
        native=native_steps[1]
        values=[native_steps[a['step']]['position'] for a in actions]
        expected=[q<-1.3 if task['task']=='open_microwave' else q>-.005 for q in values]
        if expected!=[a['evaluation_task_success'] for a in actions]:
            raise ValueError('Native microwave outcome replay disagrees')
        spec=read(folder/'task-spec.json')
        origin,axis=np.asarray(native['anchor_world_m']),np.asarray(native['axis_world'])
        pivot=np.asarray(spec['joint_origin_world_m'])
        initial_error=float(np.linalg.norm(np.cross(pivot-origin,axis)))
        feedback=read(folder/'feedback.json') if (folder/'feedback.json').exists() else []
        fit_errors=[]
        filtered=pivot.copy()
        for item in feedback:
            fit=item['online_hinge_fit']
            if fit['accepted']:
                candidate=np.asarray(fit['hinge_world_m'])
                fit_errors.append({'step':item['control_steps'],
                    'raw_fit_pivot_line_error_m':float(np.linalg.norm(np.cross(candidate-origin,axis)))})
                if np.linalg.norm((candidate-filtered)[:2])<.15:
                    filtered+=.35*(candidate-filtered)
        counter,unrelated=Counter(),Counter()
        unrelated_steps=set()
        for step in diagnostics:
            for contact in step['robot_involved_contacts']:
                if contact['normal_force_n']>.1 and sum(contact['robot_flags'])==1:
                    robot=next(n for n,b in zip(contact['geom_names'],contact['robot_flags']) if b)
                    external=next(n for n,b in zip(contact['geom_names'],contact['robot_flags']) if not b)
                    if not external.startswith('microwave_1_'):
                        unrelated[' / '.join(contact['geom_names'])]+=1
                        unrelated_steps.add(step['step'])
                    if 'finger' not in robot:
                        counter[' / '.join(contact['geom_names'])]+=1
        authorizations=read(folder/'closure-authorizations.json') if (folder/'closure-authorizations.json').exists() else []
        row={**summary,'episode':str(folder.relative_to(ROOT)),
            'sensor_views_used_for_geometry':list(read(folder/'opened/metadata.json')['cameras']),
            'native_final_joint_position_rad':float(values[-1]),
            'native_final_open_angle_deg':float(max(0.,-np.degrees(values[-1]))),
            'native_predicate_replay_agrees':True,'native_success_ever':any(expected),
            'native_success_final':expected[-1], 'closure_authorizations':authorizations,
            'initial_hinge_line_error_m_evaluation_only':initial_error,
            'final_filtered_hinge_line_error_m_evaluation_only':float(np.linalg.norm(np.cross(filtered-origin,axis))),
            'online_fit_errors_evaluation_only':fit_errors,
            'non_target_positive_force_steps_evaluation_only':len(unrelated_steps),
            'non_target_positive_force_contacts_evaluation_only':dict(unrelated),
            'nonfinger_robot_positive_force_entries':dict(counter)}
        if (folder/'handle-geometry.json').exists():
            geometry=read(folder/'handle-geometry.json')
            chosen=read(folder/'chosen-grasp.json') if (folder/'chosen-grasp.json').exists() else {}
            anchor_index=chosen.get('anchor_index')
            anchors=geometry.get('anchors_world_m',[])
            row['selected_handle_height_m']=(anchors[anchor_index][2]
                if anchor_index is not None and anchor_index<len(anchors) else None)
            row['observed_handle_height_bounds_m']=geometry.get('continuous_cylinder_z_bounds_m')
            row['grasp_source']=chosen.get('source')
        refreshes=(read(folder/'clearance-refreshes.json')
                   if (folder/'clearance-refreshes.json').exists() else [])
        row['actual_collision_depth_rechecks']=refreshes
        row['old_scene_rejections_cleared_by_current_depth']=sum(
            r.get('synchronized_scene_check',{}).get('accepted',False) for r in refreshes)
        wrist,panel_bands=[],[]
        for record in sorted(folder.glob('continuation-*.json')):
            query=read(record)
            selected=query.get('selected')
            if query.get('candidates'):
                check=query['candidates'][0]['check']
                if check.get('tool_panel_contact'):
                    panel_bands.append({'continuation':record.name,
                        'half_band_m':check['panel_surface_half_band_m'],
                        'measurement':check.get('panel_band_evidence')})
            if selected and selected.get('contact_orientation_offset_rad'):
                wrist.append({'continuation':record.name,'axis':selected['contact_orientation_axis'],
                              'offset_rad':selected['contact_orientation_offset_rad']})
        row['selected_wrist_alternatives']=wrist
        row['closing_panel_band_observations']=panel_bands
        row['observed_hinge_fit_figure']=hinge_plot(folder,feedback,origin,pivot,output/f'{task["task"]}-hinge-fit.png')
        rows.append(row)
        grounded={'joint_family':spec['joint_family'],'joint_axis_world':spec['joint_axis_world'],
                  'joint_origin_world_m':filtered.tolist(),'hinge_axis_prior':'upright door',
                  'source':'initial RGB-D fit plus logged observed-handle circle-fit updates',
                  'simulator_geometry_used':False,'execution_mode':'local tangent guide and measured contact feedback'}
        (output/f'{task["task"]}-grounded-model.json').write_text(json.dumps(grounded,indent=2)+'\n')
        if (folder/'chosen-grasp.json').exists():
            chosen=read(folder/'chosen-grasp.json')
            data={'folder':folder,'spec':spec,'initial':read(folder/'initial/metadata.json'),
                'native':native,'native_steps':native_steps,'actions':{a['step']:a for a in actions},
                'target':chosen['pose_world_xyzw'][:3],'summary':summary,
                'non_target_contact_steps':unrelated_steps,
                'frames':sorted((folder/'initial/frames').glob('*.jpg'))}
            filename=f'{task["task"]}.mp4';render_video(data,output/filename)
            if (folder/'at-grasp/metadata.json').exists():
                meta=read(folder/'at-grasp/metadata.json');camera='frontview' if task['task']=='open_microwave' else 'agentview'
                render_video(data,output/f'{task["task"]}-contact.mp4',minimum_step=max(0,meta['control_steps']-12))
                with Image.open(folder/'at-grasp'/f'{camera}.png') as image:
                    still=annotation(image,data,meta['cameras'][camera],meta['control_steps'],
                        task['task'].replace('_',' ')+' | Verified arrival')
                stills.append(still)
            cards.append(f'<h2>{task["task"].replace("_"," ")}: '
                         f'{"SUCCESS" if any(expected) else "FAILED"}</h2><video controls src="{filename}"></video>'
                         +(f'<img src="{row["observed_hinge_fit_figure"]}">' if row['observed_hinge_fit_figure'] else ''))
    if stills:
        combined=Image.new('RGB',(512*len(stills),644))
        for i,still in enumerate(stills):combined.paste(still,(512*i,0))
        combined.save(output/'verified-grasp-arrivals.png')
    snapshots=all(digest(base/'source-snapshot'/p)==h for p,h in protocol['source_sha256'].items())
    original=read(ROOT/'results/articulated-physics-pilot/protocol.json')['sources_sha256']
    original_preserved=all(digest(ROOT/p)==h for p,h in original.items())
    budget_unchanged=digest(ROOT/'results/articulated-api-v1/ledger.json')==protocol['api_ledger_sha256_before']
    result={'development_only':True,'same_previously_inspected_initial_state':True,
        'held_out_generalization_claim':False,'planned':len(planned),
        'native_successes':sum(r['native_success_ever'] for r in rows),'episodes':rows,
        'snapshot_hashes_match':snapshots,'original_frozen_sources_preserved':original_preserved,
        'api_ledger_unchanged':budget_unchanged,'new_api_calls':0,
        'simulator_geometry_used_for_policy':False,
        'hinge_error_and_contact_labels_evaluation_only':True,
        'visual_overlay_hinge_is_initial_estimate':True,
        'limitations':['One development state per task; not a new benchmark sweep',
            'Frontal RGB-D handle templates supplement cached GraspGen proposals',
            'Extra calibrated camera views are part of the adaptation',
            'The circle in task-spec is a geometry model, not a globally enforced execution manifold',
            'After verified capture, local hand-handle contact is allowed; closing also permits hand-sized bracing on the measured selected panel',
            'Measured robot surfaces are checked; hidden geometry and swept door geometry are not certified']}
    history=[]
    for prior in sorted(base.parent.glob('pilot-v*')):
        if (prior/'manifest.json').is_file():
            declared=read(prior/'manifest.json')['planned_episodes']
            summary=read(prior/'summary.json') if (prior/'summary.json').is_file() else {'episodes':[]}
            interrupted=(read(prior/'infrastructure-abort.json')
                         if (prior/'infrastructure-abort.json').exists() else None)
            history.append({'run':str(prior.relative_to(ROOT)),'planned':len(declared),
                            'completed':len(summary['episodes']),
                            'native_successes':sum(r['official_task_success'] for r in summary['episodes']),
                            'status':('infrastructure_aborted' if interrupted else
                                      'completed' if len(summary['episodes'])==len(declared) else 'incomplete'),
                            'infrastructure_abort':interrupted,
                            'episodes':summary['episodes']})
    result['all_development_passes']=history
    result['development_passes_are_different_algorithm_versions']=True
    authorized=all(all(a['position_error_m']<=.006 and a['orientation_error_rad']<=.08
                      for a in r['closure_authorizations']) for r in rows)
    no_early_close=True
    for row in rows:
        first=min((a['control_step'] for a in row['closure_authorizations']),default=float('inf'))
        actions=read(ROOT/row['episode']/'initial/actions.json')
        no_early_close &= all(a['step']>first for a in actions if a['control'][-1]>0)
    validation={'all_planned_latest_episodes_reported':len(rows)==len(planned),
        'all_native_predicates_replayed':all(r['native_predicate_replay_agrees'] for r in rows),
        'all_episode_control_budgets_respected':all(r['control_steps']<=600 for r in rows),
        'all_closures_have_verified_arrival':authorized,'no_closure_command_before_authorization':no_early_close,
        'source_snapshots_match':snapshots,'original_frozen_sources_preserved':original_preserved,
        'api_ledger_unchanged':budget_unchanged,'all_development_passes_included':len(history)}
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    table=''.join(f'<tr><td>{r["task"]}</td><td>{r["native_success_ever"]}</td>'
        f'<td>{r["native_final_open_angle_deg"]:.2f}</td><td>{r.get("failure","")}</td></tr>' for r in rows)
    (output/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Handle contact repair</title>'
        '<style>body{font:17px system-ui;max-width:1100px;margin:30px auto;padding:20px;line-height:1.5}'
        'img,video{width:100%}td,th{padding:10px;text-align:left}table{border-collapse:collapse}</style>'
        '<h1>Handle contact repair — development pilot</h1><p>RGB-D upright door and handle fits, '
        'robot-clear grasp approaches, measured arrival before closure, and short feedback pulls. '
        'No simulator hinge geometry enters planning. The hinge is refined only when observed handle motion '
        'supports a circle fit. The fitted circle guides local motion rather than enforcing a fixed global path. '
        'Finger width is not a retention veto. After capture, local hand-handle and finger-panel contact is allowed. '
        'Closing additionally permits palm bracing within the configured hand reach around the tracked handle '
        'and the measured panel slab. Arm contacts and unrelated surfaces stay checked.</p>'
        '<p>Native LIBERO predicates determine task success. These are the same previously inspected '
        'initial states, so this is development evidence rather than a held-out benchmark.</p>'
        '<p>The opening/closing sign is checked against adjacent stationary depth geometry using an upright-door prior. '
        'This is a task-template heuristic, not a learned general articulation estimator.</p>'
        '<p>The handle cylinder is extended only along continuous support in observed depth. '
        'Higher grasp positions aim to keep the hand clear of nearby objects. '
        'Grasp heights, source proposals and all rejected candidates are retained in the measurements.</p>'
        '<p>Cyan axes are native geometry shown only for evaluation. Magenta axes show the initial '
        'RGB-D estimate. Yellow circles show the current commanded TCP; red crosses show the measured TCP.</p>'
        '<table><tr><th>Task</th><th>Native success</th><th>Final door opening (degrees)</th><th>Stop reason</th></tr>'
        +table+'</table><p><a href="evaluation.json">Full measurements</a></p>'
        +('<img src="verified-grasp-arrivals.png">' if stills else '')+''.join(cards)
        +'<h2>All development passes</h2><p>Each pass changes the algorithm; these counts are development history, '
        'not independent held-out trials. Failures before motion remain included.</p><ul>'
        +''.join(f'<li>{h["run"]}: {h["native_successes"]}/{h["planned"]} native successes; '
                  f'{h["completed"]} episodes completed; {h["status"]}.</li>' for h in history)+'</ul></html>')
    if not(snapshots and original_preserved and budget_unchanged and authorized and no_early_close):
        raise ValueError('Integrity check failed; results preserved')
    print(json.dumps({'native_successes':result['native_successes'],'planned':len(planned),
        'episodes':[{k:r.get(k) for k in ['task','native_final_open_angle_deg','official_task_success','failure','closure_authorizations']} for r in rows]}))


if __name__=='__main__':main()
