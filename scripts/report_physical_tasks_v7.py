"""Report full-arm/contact/moving-part admission separately from physical success."""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from try_rekep_bowl_v3 import save


def read(path):
    return json.loads(Path(path).read_text())


def collision_counts(admission):
    reasons, pairs = Counter(), Counter()
    for row in admission.get('rejected_candidates', []):
        reasons[row['reason']] += 1
        check = row.get('clearance', row.get('path', {}).get('clearance', {}))
        pairs.update(f"{item['robot_geom_name']} / {item['scene_part']}" for item in check.get('collisions', []))
    return dict(reasons), dict(pairs)


def collect(base):
    protocol, manifest = read(base / 'protocol.json'), read(base / 'manifest.json')
    rows = []
    for task, state in manifest['planned_episodes']:
        folder = base / f'goal{task:02d}-state{state:03d}'
        if not (folder / 'summary.json').is_file():
            rows.append({'task_index': task, 'initial_state': state, 'status': 'pending', 'official_task_success': False})
            continue
        row = read(folder / 'summary.json')
        actions = read(folder / 'initial/actions.json')
        success = any(action['evaluation_task_success'] for action in actions)
        if success != row['official_task_success']:
            raise ValueError('Summary disagrees with the actual LIBERO predicate')
        row.update(folder=str(folder.relative_to(ROOT)), control_budget_passed=len(actions) <= 600,
                   evaluation_recomputed_from_actions=True, grasp_admitted=(folder / 'chosen-grasp.json').is_file(),
                   reached_closure=(folder / 'closed').is_dir(), grounding_completed=(folder / 'task-spec.json').is_file())
        if (folder / 'initial-actual-clearance.json').is_file():
            row['initial_full_arm_clearance_passed'] = read(folder / 'initial-actual-clearance.json')['accepted']
        if (folder / 'grasp-admission.json').is_file():
            row['candidate_rejection_counts'], row['collision_pair_counts'] = collision_counts(read(folder / 'grasp-admission.json'))
        rows.append(row)
    freeze = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
              for name, digest in protocol['sources_sha256'].items()}
    return {'planned_episodes': len(manifest['planned_episodes']), 'all_failures_in_denominator': True,
            'pipeline_version': read(base / 'configuration.json')['version'],
            'completed_episodes': sum(row['status'] != 'pending' for row in rows),
            'successes_by_task': {name: sum(row['official_task_success'] for row in rows if row['task_index'] == task)
                                  for task, name in [(8, 'bowl'), (0, 'drawer')]},
            'planned_by_task': {name: sum(row['task_index'] == task for row in rows) for task, name in [(8, 'bowl'), (0, 'drawer')]},
            'grasp_admitted': sum(row.get('grasp_admitted', False) for row in rows),
            'reached_closure': sum(row.get('reached_closure', False) for row in rows),
            'initial_clearance_passed': sum(row.get('initial_full_arm_clearance_passed', False) for row in rows),
            'source_freeze_passed': all(freeze.values()), 'source_freeze_checks': freeze,
            'control_budgets_passed': all(row.get('control_budget_passed', True) for row in rows),
            'api_ledger_unchanged': read(base / 'summary.json')['api_ledger_unchanged'],
            'failures_by_stage': dict(Counter(row.get('failure_stage') or 'goal_not_achieved' for row in rows if not row['official_task_success'])),
            'episodes': rows}


def scene_plot(folder, destination, task):
    if not (folder / 'initial-scene.npz').is_file():
        return False
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with np.load(folder / 'initial-scene.npz') as arrays:
        static, moving, contacts = [arrays[key] for key in ['static_points', 'moving_reference', 'contact_reference']]
    near = static[(static[:, 2] > .89) & (static[:, 2] < 1.24)
                  & (static[:, 0] > -.32) & (static[:, 0] < .32)
                  & (static[:, 1] > -.4) & (static[:, 1] < .28)]
    near = near[::max(1, len(near) // 14000)]
    moving = moving[::max(1, len(moving) // 5000)]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.3), constrained_layout=True)
    for axis, dims, labels in zip(axes, [(0, 2), (1, 2)], [('World x (m)', 'Height (m)'), ('World y (m)', 'Height (m)')]):
        axis.scatter(near[:, dims[0]], near[:, dims[1]], s=.6, c='#9ba8b7', alpha=.35, label='Stationary observed scene')
        axis.scatter(moving[:, dims[0]], moving[:, dims[1]], s=1, c='#15956f', label='Estimated moving source')
        if task == 0:
            axis.scatter(contacts[:, dims[0]], contacts[:, dims[1]], s=1.5, c='#c27217', label='Selected handle contact region')
        axis.set(xlabel=labels[0], ylabel=labels[1])
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=.15)
    axes[0].legend(fontsize=7, markerscale=4, loc='lower left')
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return True


def render(data, output):
    cards, rows = [], []
    for row in data['episodes']:
        task, state = row['task_index'], row['initial_state']
        label = f"{'Bowl onto plate' if task == 8 else 'Middle drawer'} · state {state}"
        status = 'Success' if row['official_task_success'] else 'Pending' if row['status'] == 'pending' else 'Failed'
        failure = row.get('failure', '')
        rows.append(f'<tr><td>{label}</td><td>{status}</td><td>{html.escape(row.get("failure_stage") or "—")}</td>'
                    f'<td>{row.get("control_steps", "—")}/600</td><td>{html.escape(failure)}</td></tr>')
        if row['status'] == 'pending':
            continue
        folder = ROOT / row['folder']
        assets = output / f'goal{task:02d}-state{state:03d}'
        assets.mkdir()
        media = ''
        if (folder / 'initial/rollout.mp4').is_file():
            shutil.copyfile(folder / 'initial/rollout.mp4', assets / 'rollout.mp4')
            media += f'<video controls preload="metadata" src="{assets.name}/rollout.mp4"></video>'
        if scene_plot(folder, assets / 'scene.png', task):
            media += f'<img class="scene" src="{assets.name}/scene.png" alt="Observed stationary scene, moving source and contact region">'
        evidence = []
        for name in ['summary.json', 'task-spec.json', 'contact-policy.json', 'observed-clearance.json',
                     'initial-actual-clearance.json', 'grasp-admission.json', 'chosen-grasp.json',
                     'collision-checks.json', 'scene-updates.json', 'drawer-feedback.json', 'retention.json',
                     'command-log.json', 'grasp-cache-evidence.json', 'final-scene-scope.json']:
            if (folder / name).is_file():
                shutil.copyfile(folder / name, assets / name)
                evidence.append(f'<a href="{assets.name}/{name}">{name}</a>')
        reject_rows = ''.join(f'<tr><td>{html.escape(reason)}</td><td>{count}</td></tr>'
                              for reason, count in row.get('candidate_rejection_counts', {}).items())
        pair_rows = ''.join(f'<tr><td>{html.escape(pair)}</td><td>{count}</td></tr>'
                            for pair, count in sorted(row.get('collision_pair_counts', {}).items(), key=lambda item: -item[1]))
        cards.append(f'<article><h2>{label}: {status}</h2><p>{html.escape(failure)}</p>{media}'
                     f'<details><summary>Candidate rejection reasons and collision pairs</summary><div class="tables">'
                     f'<table>{reject_rows}</table><table>{pair_rows}</table></div></details>'
                     f'<details><summary>Recorded evidence</summary><p>{" · ".join(evidence)}</p></details></article>')
    total = data['planned_episodes']
    validation_links = []
    for source, name in [
        (ROOT / 'results/physical-tasks-v7-development/synchronized-camera-smoke/result.json', 'camera-and-kinematics-validation.json'),
        (ROOT / 'results/physical-tasks-v7-development/moving-drawer-replay/summary.json', 'moving-drawer-observation-replay.json'),
        (ROOT / 'results/physical-tasks-v7-eval/summary.json', 'first-frozen-v7-cohort.json'),
        (ROOT / 'results/physical-tasks-v7.1-eval/summary.json', 'synchronized-v7.1-cohort.json')]:
        if source.is_file():
            shutil.copyfile(source, output / name)
            validation_links.append(f'<a href="{name}">{name}</a>')
    comparison = []
    for name, task in [('bowl', 8), ('drawer', 0)]:
        successes = 0
        for state in [1, 11, 21, 31, 41]:
            old = ROOT / f'results/physical-tasks-v6-eval/{name}/goal{task:02d}-state{state:03d}/summary.json'
            successes += read(old)['official_task_success']
        comparison.append(f'<tr><td>{name.capitalize()}</td><td>{successes}/5</td>'
                          f'<td>{data["successes_by_task"][name]}/{data["planned_by_task"][name]}</td></tr>')
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full-arm collision and moving-object evaluation</title><style>
body{{margin:0;background:#f5f7fa;color:#1c2c3c;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1150px;margin:auto;padding:28px}}
h1{{font-size:32px;line-height:1.2}}h2{{font-size:21px}}article,.note,.metrics{{background:white;border:1px solid #dce4eb;border-radius:12px;padding:20px;margin:22px 0}}
.metrics{{display:flex;gap:30px;flex-wrap:wrap}}.metric strong{{display:block;font-size:29px}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border-bottom:1px solid #e2e7ed;padding:9px;text-align:left;vertical-align:top}}video{{width:100%;max-height:420px;background:#111}}
.scene{{width:100%;margin-top:14px}}.tables{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}a{{color:#176784}}summary{{cursor:pointer;margin-top:14px}}
</style><main><h1>Full-arm collision checking and moving-object geometry</h1>
<p>Cached Luna roles and the same GraspGen pose proposals as v6, validated against fresh RGB-D with recorded tight tolerances. Admission checks every moving arm link, the palm,
finger meshes, local pad contacts, and an estimated moving-source envelope. LIBERO contact physics stays enabled.</p>
<div class="metrics"><div class="metric"><strong>{data['completed_episodes']}/{total}</strong>Episodes recorded</div>
<div class="metric"><strong>{data['initial_clearance_passed']}/{total}</strong>Initial full-arm checks passed</div>
<div class="metric"><strong>{data['grasp_admitted']}/{total}</strong>Grasps admitted</div>
<div class="metric"><strong>{data['reached_closure']}/{total}</strong>Episodes reached closure</div></div>
<div class="note"><strong>Scope:</strong> Green points are sensor-estimated moving geometry; the drawer model covers the observed front panel and handle.
Its hidden body remains unmodelled. Contacts are restricted by region and stage. Known pad regions overlapping a finger mesh receive the same local allowance;
outer finger surfaces receive no general exemption. Edges are sampled every 0.025 rad; this is not continuous collision detection or a certificate for unseen space.
The near-pad clearance shell exception applies only to points outside the actual finger mesh, and does not allow shoulder penetration.
The self-collision backend remains VAMP’s Panda sphere model. Atlas, pRRTC, ST-pRRTC, RNDF and ReKep’s native optimizers are not used for execution.</div>
<h2>Actual LIBERO success</h2><table><tr><th>Task</th><th>v6 palm-only admission</th><th>{html.escape(data['pipeline_version'])} full-arm/contact/moving-envelope admission</th></tr>{''.join(comparison)}</table>
<p>These are previously inspected states, not a held-out generalization test. Every planned episode remains in the denominator, including failures before robot motion.
A more restrictive checker can reduce success by exposing invalid grasps or by making conservative geometry estimates; these results alone do not identify which.</p>
<p>The first frozen v7 run is preserved. It had four initial-frame failures which motivated synchronized RGB-D, calibration and robot-prior capture in v7.1.
The moving-drawer replay is evidence for scene updates on recorded v6 observations, not a new physical success. Validation records: {' · '.join(validation_links)}.</p>
<p>v7.2 validates cache reuse within 0.2 mm selected-depth agreement, mean RGB channel difference ≤0.25 and maximum channel difference ≤16.
Nonidentical captures are recorded as such. These tolerances do not change contact allowances or collision geometry; all cached grasp poses are rechecked against current observations.</p>
<h2>Episodes</h2><table><tr><th>Task/state</th><th>Success</th><th>Failure stage</th><th>Controls</th><th>Reason</th></tr>{''.join(rows)}</table>
<p>Source freeze: {data['source_freeze_passed']}. Control budgets: {data['control_budgets_passed']}. API ledger unchanged: {data['api_ledger_unchanged']}.</p>
<p><a href="evaluation.json">Evaluation data</a> · <a href="protocol.json">Frozen protocol</a> · <a href="configuration.json">Configuration</a>
 · <a href="https://arxiv.org/pdf/2309.16085">RNDF paper suggested for a later geometry backend</a></p>{''.join(cards)}</main></html>'''
    (output / 'index.html').write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/physical-tasks-v7.2-eval')
    parser.add_argument('--output', default='results/task-grounding-collision-v7-report')
    args = parser.parse_args()
    base, output = ROOT / args.input, ROOT / args.output
    if output.exists():
        raise FileExistsError('Preserve existing reports; choose a fresh output path')
    output.mkdir(parents=True)
    data = collect(base)
    save(output / 'evaluation.json', data)
    for name in ['protocol.json', 'configuration.json']:
        shutil.copyfile(base / name, output / name)
    render(data, output)
    print(json.dumps({key: value for key, value in data.items() if key not in ['episodes', 'source_freeze_checks']}))


if __name__ == '__main__':
    main()
