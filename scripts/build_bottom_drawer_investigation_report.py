"""Build the evaluation-only bottom-drawer diagnostic report from saved trials."""
from pathlib import Path
import hashlib
import html
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from vlm_tamp.coupled_storage import read_json, write_json
from vlm_tamp.rekep_libero.geometry import pose_matrix, transform_points

OUT = ROOT / 'results/bottom-drawer-investigation-report'
MODES = ROOT / 'results/bottom-drawer-contact-modes-20261016'
FEEDBACK = ROOT / 'results/bottom-drawer-feedback-20261016'
AUDIT = ROOT / 'results/bottom-drawer-native-audit-v2-20261016'
CORRECTION = ROOT / 'results/bottom-drawer-closure-aperture-audit-20261016'
LABELS = {
    'detector_full_open': 'Detector pose + closure',
    'frontal_full_open': 'Frontal pinch',
    'tilted_partial_pinch': 'Tilted pinch, partial opening',
    'tilted_parallel_hook': 'Fingers along handle + nominal pull',
    'tilted_parallel_feedback': 'Fingers along handle + measured-hand update',
}

CSS = """
:root{color-scheme:light;--ink:#162a3a;--muted:#516476;--line:#d5e0e6;--teal:#087f8c;--amber:#a64b00}
*{box-sizing:border-box}body{margin:0;background:#f4f7f9;color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:1160px;margin:auto;padding:36px 24px 70px}h1{font-size:38px;line-height:1.2;max-width:900px}
h2{font-size:26px;margin-top:40px}h3{font-size:19px}a{color:#006b84}p{max-width:1000px}
.eyebrow{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.lead{font-size:21px;max-width:960px}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:22px;margin:16px 0}
.notice{border-left:5px solid var(--amber);background:#fff7ed;padding:16px 20px;border-radius:4px}
.good{border-left-color:var(--teal);background:#edf8f7}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.stat{background:white;border:1px solid var(--line);padding:18px;border-radius:12px}.stat strong{display:block;font-size:32px;color:var(--teal)}
figure{margin:0}img,video{width:100%;height:auto;border-radius:8px;background:#e3eaee}figcaption{font-size:14px;color:var(--muted);padding:10px 0}
table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;padding:11px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#eaf0f4}small,.muted{color:var(--muted)}
.scroll{overflow-x:auto}code{font-size:.9em;background:#eaf0f4;padding:2px 4px}nav{display:flex;gap:18px;flex-wrap:wrap}.pill{font-size:13px;border-radius:20px;padding:4px 10px;background:#e6eff4;display:inline-block}
@media(max-width:760px){.grid,.stats{grid-template-columns:1fr}h1{font-size:30px}main{padding:24px 16px}}
"""


def href(path):
    return html.escape(os.path.relpath(path, OUT), quote=True)


def episode(folder):
    summary = read_json(folder / 'summary.json')
    actions = read_json(folder / 'initial/actions.json')
    diagnostics = read_json(folder / 'initial/evaluation_contact_diagnostics.json')
    steps, opening, forces = [], [], []
    for record in diagnostics['steps']:
        joint = next(x for x in record['evaluation_only_articulation_state']
                     if x['joint_name'].endswith('_bottom_level'))
        contacts = [c for c in record['robot_involved_contacts']
                    if any('cabinet' in name for name in c['geom_names'])]
        steps.append(record['step'])
        opening.append(-1000 * joint['position'])
        forces.append(max((c['normal_force_n'] for c in contacts), default=0.))
    start = 61
    phases = []
    for phase in read_json(folder / 'phases.json'):
        end = start + phase['issued_commands'] - 1
        selected = [f for s, f in zip(steps, forces) if start <= s <= end]
        phases.append({'label': phase['label'], 'start_step': start, 'end_step': end,
                       'peak_single_cabinet_contact_normal_force_n': max(selected, default=0.)})
        start = end + 1
    return {
        'folder': str(folder), 'mode': summary['mode'], 'state': summary['initial_state'],
        'official_task_success': summary['official_task_success'],
        'final_task_success': summary['final_task_success'], 'steps': summary['control_steps'],
        'max_opening_mm': max(opening), 'final_opening_mm': opening[-1],
        'peak_single_cabinet_contact_normal_force_n': max(forces),
        'phases': phases, 'series': {'control_steps': steps, 'opening_mm': opening,
                                    'max_single_cabinet_contact_normal_force_n': forces},
        'final_tracking_error_mm': 1000 * np.asarray(actions[-1]['position_error_m']),
        'measured_mean_finger_joint_aperture_m': summary['measured_half_aperture_before_approach_m'],
        'all_issued_trajectories_completed': summary['all_issued_trajectories_completed'],
    }


def corrected_audits():
    rows = []
    replacement = read_json(CORRECTION / 'close_bottom_drawer-state015/native-audit.json')
    corrected = {(r['detector_index'], r['ik_rank'], r['planner_reason']): r
                 for r in replacement['poses']}
    for folder in sorted(AUDIT.glob('*-state*')):
        data = read_json(folder / 'native-audit.json')
        if folder.name == 'close_bottom_drawer-state015':
            for i, pose in enumerate(data['poses']):
                key = (pose['detector_index'], pose['ik_rank'], pose['planner_reason'])
                if key in corrected:
                    data['poses'][i] = corrected[key]
            data['aperture_correction_applied'] = True
        data['poses_without_native_penetration'] = sum(p['native_penetrations_over_0_1mm'] == 0 for p in data['poses'])
        data['poses_with_other_native_penetration'] = sum(p['native_penetrations_outside_local_finger_contact'] > 0 for p in data['poses'])
        data['palm_overlap_poses'] = sum(any('hand_collision' in c['robot_part'] and c['signed_distance_m'] < -.0001
                                          for c in p['native_contacts']) for p in data['poses'])
        rows.append({k: v for k, v in data.items() if k not in {'poses', 'renders'}})
    assert sum(r['audited_poses'] for r in rows if r['case'].startswith('open_')) == 1024
    assert sum(r['poses_with_other_native_penetration'] for r in rows if r['case'].startswith('open_')) == 1024
    return rows


def geometry():
    source = ROOT / 'results/family-transfer-frozen-20261016/open_bottom_drawer-state005'
    meta = read_json(source / 'opened/metadata.json')
    inverse = np.linalg.inv(pose_matrix(meta['eef_pose_world_xyzw']))
    shapes = {v['robot_geom_name']: transform_points(v['vertices_world'], inverse)
              for v in meta['robot_collision_volumes_world'] if 'gripper' in v['robot_geom_name']}
    hand = shapes['gripper0_hand_collision']
    fingers = np.concatenate([shapes[f'gripper0_finger{i}_collision'] for i in [1, 2]])
    cloud = np.load(source / 'grasp-cloud.npy')
    return {'configured_palm_span_jaw_axis_mm': float(np.ptp(hand[:, 0]) * 1000),
            'configured_open_outer_finger_span_mm': float(np.ptp(fingers[:, 0]) * 1000),
            'predicted_handle_center_height_above_table_mm': float((read_json(source / 'task-spec.json')['source_anchor_world_m'][2] - .9) * 1000),
            'observed_handle_low_surface_height_above_table_mm': float((np.quantile(cloud[:, 2], .01) - .9) * 1000),
            'table_height_m_evaluation_only': .9,
            'note': 'Robot dimensions from configured collision meshes. Table height is native evaluation geometry, never used to generate contact proposals.'}


def plot(paired, rows):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#bc6c25', '#087f8c']
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for row, color, label in zip(paired, colors, ['Nominal pull from intended contact pose', 'Pull updated from measured hand pose']):
        series = row['series']
        axes[0].plot(series['control_steps'], series['opening_mm'], color=color, label=label, linewidth=2)
        axes[1].plot(series['control_steps'], series['max_single_cabinet_contact_normal_force_n'], color=color, linewidth=1.8)
    axes[0].axhline(140, color='#4b5563', ls='--', lw=1.2, label='LIBERO opening threshold: > 140 mm')
    axes[0].set_ylabel('Actual drawer opening (mm)')
    axes[1].set_ylabel('Largest cabinet contact\nnormal force (N)')
    axes[1].set_xlabel('Control step (20 Hz simulation control; not wall-clock timing)')
    for ax in axes:
        ax.axvline(260, color='#9ca3af', ls=':'); ax.grid(alpha=.15)
    axes[0].set_title('State 5: same first 200 joint references; different post-contact reference')
    axes[0].legend(fontsize=9, loc='upper left')
    axes[1].annotate('Reference updated after contact hold', xy=(260, 25), xytext=(120, 145),
                     arrowprops={'arrowstyle': '->', 'color': '#516476'}, color='#516476')
    fig.savefig(OUT / 'paired-reference-correction.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    labels = ['Detector / s5', 'Frontal pinch / s5', 'Tilted pinch / s5', 'Parallel nominal / s5']
    labels.extend(f'Measured update / s{x["state"]}' for x in rows[4:])
    ax.barh(labels[::-1], [x['final_opening_mm'] for x in rows][::-1], color=(['#bc6c25'] * 4 + ['#087f8c'] * 4)[::-1])
    ax.axvline(140, ls='--', color='#4b5563'); ax.set_xlim(0, 162)
    ax.set_xlabel('Final actual drawer opening (mm)'); ax.set_title('Eight development diagnostics; native physics, no scene admission')
    ax.grid(axis='x', alpha=.15)
    for i, value in enumerate([x['final_opening_mm'] for x in rows][::-1]):
        ax.text(max(0, value) + 1.5, i, f'{value:.1f}', va='center', fontsize=10)
    fig.savefig(OUT / 'opening-outcomes.png', dpi=160); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    rows = [episode(MODES / x) for x in list(LABELS)[:4]]
    rows.extend(episode(FEEDBACK / f'state{s:03d}') for s in [5, 15, 25, 45])
    a = read_json(MODES / 'tilted_parallel_hook/initial/actions.json')
    b = read_json(FEEDBACK / 'state005/initial/actions.json')
    qa = np.asarray([x['target_joints'] for x in a if 'target_joints' in x])
    qb = np.asarray([x['target_joints'] for x in b if 'target_joints' in x])
    assert np.array_equal(qa[:200], qb[:200])
    audits = corrected_audits(); dims = geometry()
    plot([rows[3], rows[4]], rows)
    write_json(OUT / 'measurements.json', {'episodes': rows, 'native_pose_audits_corrected': audits,
               'geometry': dims, 'paired_first_200_joint_references_identical': True,
               'native_success_threshold_mm': 140, 'scope': 'Development diagnostics, not a new frozen benchmark.'})
    table = ''.join(f'<tr><td>{html.escape(LABELS[r["mode"]])}</td><td>{r["state"]}</td><td>{r["final_opening_mm"]:.2f} mm</td><td>{"Yes" if r["final_task_success"] else "No"}</td><td>{r["steps"]}</td><td><a href="{href(Path(r["folder"])/"initial/rollout.mp4")}">Video</a> · <a href="{href(Path(r["folder"])/"summary.json")}">Log</a></td></tr>' for r in rows)
    audit_table = ''.join(f'<tr><td>{r["case"]}</td><td>{html.escape(str(r["contact_pixel_ray_evaluation"]["geom_name_evaluation_only"]))}</td><td>{r["fitted_plane_tilt_from_vertical_deg"]:.1f}°</td><td>{r["poses_with_other_native_penetration"]} / {r["audited_poses"]}</td></tr>' for r in audits)
    first = AUDIT / 'open_bottom_drawer-state005/first_proposed-front.png'
    frontal = AUDIT / 'open_bottom_drawer-state015/gripper_model_rejection-front.png'
    before = MODES / 'tilted_parallel_hook/initial/rollout.mp4'
    after = FEEDBACK / 'state005/initial/rollout.mp4'
    result = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bottom drawer: contact mode and reference correction</title><style>{CSS}</style></head><body><main>
<div class="eyebrow">VLM + motion planning · Physical diagnosis · 16 October 2026</div>
<h1>The bottom drawer can open with fingers along the handle</h1>
<p class="lead">The offered pinch poses collide with real geometry. A different contact orientation, followed by a pull referenced to the achieved hand pose, opened the drawer in four diagnostic cases.</p>
<nav><a href="#geometry">Why the grasps fail</a><a href="#physics">Physical trials</a><a href="#feedback">What changed</a><a href="#limits">Scope and next step</a><a href="../rl-execution-literature-review/index.html">RL execution literature review</a></nav>
<div class="stats"><div class="stat"><strong>1,024 / 1,024</strong>offered opening postures had unintended native overlap</div><div class="stat"><strong>4 / 4</strong>measured-hand diagnostic pulls ended in native success</div><div class="stat"><strong>0 API calls</strong>RGB-D, known robot geometry and proprioception reused</div></div>
<p class="notice">These are development diagnostics with planning-scene collision admission bypassed. They establish physical feasibility. The frozen bottom-drawer benchmark remains 0/10; this behavior has not been admitted by the collision-aware planner or evaluated on new scenes.</p>
<h2 id="geometry">The gripper orientation matters more than simply closing it</h2>
<p>In our configured simulator model, the palm spans approximately <b>{dims['configured_palm_span_jaw_axis_mm']:.0f} mm</b> along the jaw axis, and the open fingers span {dims['configured_open_outer_finger_span_mm']:.0f} mm at their outer surfaces. The selected handle center is only <b>{dims['predicted_handle_center_height_above_table_mm']:.0f} mm above the table</b>; its observed lower surface has roughly {dims['observed_handle_low_surface_height_above_table_mm']:.0f} mm of clearance. Turning the jaw axis vertically puts the bulky palm into that small gap. These dimensions describe this simulation model, not a manufacturer specification.</p>
<div class="grid"><figure><img src="{href(first)}" alt="Static reconstructed detector grasp overlapping the cabinet"><figcaption>Opening state 5: a rejected detector posture reconstructed in a private evaluation state. The palm overlaps the cabinet. This is a hypothetical pose visualization, not an executed grasp.</figcaption></figure><figure><img src="{href(frontal)}" alt="Static vertically oriented gripper close to the table"><figcaption>Opening state 15: the vertically oriented jaw/palm arrangement has inadequate table clearance. Native signed contact distances confirm overlap; images alone can hide penetration.</figcaption></figure></div>
<p>The audit reconstructed every retained opening IK posture for states 5, 15, 25 and 45: 256 per state. All 1,024 had native overlap outside the permitted local finger-contact region. Thus these rejections are not explained solely by point-cloud conservatism. This finite set does not establish that every possible grasp is impossible.</p>
<h3>Closing also has a perception problem</h3><p>Selected pixels in closing states 5 and 45 hit the table; state 25 hits the wine rack. State 15 selects the handle, but the inferred panel plane is nearly horizontal. An alternative gripper orientation cannot fix a target selected on the wrong object. Native object names below are an evaluation-only check and never entered proposal generation.</p>
<div class="scroll"><table><thead><tr><th>Case</th><th>Selected pixel hits (native audit)</th><th>Panel tilt from vertical</th><th>Unintended overlap / tested IK poses</th></tr></thead><tbody>{audit_table}</tbody></table></div>
<p class="muted">State 35 is outside this pose audit: opening failed before reliable panel fitting, and closing had invalid/mixed image coordinates. It remains a failure in the original denominator. Eight closing-state-15 checks originally reconstructed the open aperture; their corrected closure-aperture audit is applied here. The original audit is preserved for provenance.</p>
<h2 id="physics">Letting physical contact decide</h2>
<p>Eight physical episodes were executed. Robot-valid IK and joint limits remained; planning-scene collision admission was disabled for this diagnostic. Every issued trajectory was sent in full, with no execution collision, retention or progress veto. The simulator supplied contact dynamics; there were no assisted attachments, object teleports or friction changes.</p>
<img src="opening-outcomes.png" alt="Final drawer opening across eight physical diagnostics">
<div class="scroll"><table><thead><tr><th>Contact / reference rule</th><th>State</th><th>Final opening</th><th>Native success at end</th><th>Control steps</th><th>Evidence</th></tr></thead><tbody>{table}</tbody></table></div>
<p>LIBERO's wooden-cabinet opening predicate requires the slide joint below −0.14 m: more than 140 mm of opening. The parallel-finger nominal pull moved it 115 mm, so it really opened partially but did not satisfy the task. The measured-hand variants ended at 140.8–147.7 mm and satisfied the actual predicate.</p>
<div class="grid"><figure><video controls preload="metadata" src="{href(before)}"></video><figcaption>State 5 · Fingers along the handle, nominal pull · 115 mm, partial opening.</figcaption></figure><figure><video controls preload="metadata" src="{href(after)}"></video><figcaption>State 5 · Same approach, measured-hand reference update · 147 mm, native success. Video playback is slowed; timing uses recorded control steps.</figcaption></figure></div>
<h2 id="feedback">What the reference update actually does</h2>
<p>The contact proposal uses a declared 45° tilt and places the jaw axis along the observed handle. The gripper is partially closed before approach and then holds its commanded opening. The measured mean finger-joint magnitude before approach is about 16 mm. During the pull, contact logs show pressing/sliding on the handle, often through one finger; they do not establish a secure two-finger pinch or force closure. The internal mode name “hook” describes the proposed contact strategy, not a proven contact topology.</p>
<p>In state 5 the nominal final hand position stayed about 26 mm above its intended position because of physical contact. After the approach and 30-step contact hold complete, the feedback variant reads the actual hand pose from robot proprioception, places the reference 3 mm below it, and solves a new 150 mm pull along the existing RGB-D-derived drawer direction. The 3 mm downward bias is the existing depth-grid spacing; the joint servo remains unchanged at gain 300. This is one reference update between complete motions, <b>not an RL controller or continuous adaptive hinge estimator</b>.</p>
<img src="paired-reference-correction.png" alt="Paired native drawer motion and cabinet contact forces with and without reference correction">
<p>The first 200 joint references, through control step 260, are exactly identical in the paired state-5 runs. The changed pull raises final opening from 115 to 147 mm. The largest single cabinet-contact normal force during the pull falls from about 192 N to 14 N. These force measurements are native evaluation diagnostics, not policy inputs.</p>
<p class="notice">The approach still produces large contact loads: peak single-contact normal forces across the four successful episodes are about 214–250 N. Success in this simulator does not demonstrate gentle contact, hardware suitability or collision-free execution. The force plot shows why compliant contact control is a meaningful next baseline.</p>
<h2 id="limits">What is established, and what remains</h2>
<div class="card"><p><b>Established:</b> the previous grasp set was physically unsuitable; partial non-pinching contact can move the drawer; using the achieved hand pose improves a completed pull without an execution veto. Four previously inspected opening states succeed under one frozen diagnostic rule.</p>
<p><b>Still unresolved:</b> the family controller does not yet generate/admit this contact mode, closing can target the wrong object, and state 35 lacks usable grounding. The 45° orientation and downward preload are explicit manual contact assumptions. They are not evidence of general contact selection or a zero-shot learned skill.</p>
<p><b>Recommended next experiment:</b> add pinch, press/drag and push as general contact-mode candidates; let observed clearance and semantic grounding select the mode. Keep intentional finger contact local while checking the rest of the arm/palm. First compare a measured-reference/admittance baseline with the existing controller, then a bounded residual policy. Evaluate on new cabinets and handle heights, retaining failures and native success as the task criterion.</p></div>
<p><a href="verification.json">Independent episode verification</a> · <a href="measurements.json">Measurements and corrected audit</a> · <a href="{href(FEEDBACK/'manifest.json')}">Frozen diagnostic manifest</a> · <a href="{href(FEEDBACK/'frozen-runner.py')}">Exact diagnostic runner</a> · <a href="../family-transfer-report/index.html">Unchanged frozen transfer benchmark</a></p>
<p class="muted">All eight movies and reference/action traces verified. Frozen family-policy source hashes unchanged. Cumulative API spend remains $0.343700125 of $2. Raw temporary frames and redundant observations are pruned after runs; compact evidence is retained.</p>
</main></body></html>'''
    (OUT / 'index.html').write_text(result)
    print(json.dumps({'report': str(OUT/'index.html'), 'episodes': len(rows), 'feedback_successes': sum(r['final_task_success'] for r in rows[4:]), 'geometry': dims}))


if __name__ == '__main__':
    main()
