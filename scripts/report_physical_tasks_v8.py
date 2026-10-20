"""Report contact semantics independently from actual LIBERO task success."""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import shutil

from report_physical_tasks_v7 import collect, read, scene_plot
from try_rekep_bowl_v3 import save

ROOT = Path(__file__).resolve().parents[1]


def render(data, output):
    cards, rows = [], []
    permissions = Counter()
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
            media += f'<img class="scene" src="{assets.name}/scene.png" alt="Measured stationary scene, moving source and selected contact region">'
        evidence = []
        for name in ['summary.json', 'task-spec.json', 'contact-policy.json', 'observed-clearance.json',
                     'initial-actual-clearance.json', 'grasp-admission.json', 'chosen-grasp.json',
                     'collision-checks.json', 'scene-updates.json', 'drawer-feedback.json', 'retention.json',
                     'command-log.json', 'grasp-cache-evidence.json', 'final-scene-scope.json']:
            if (folder / name).is_file():
                shutil.copyfile(folder / name, assets / name)
                evidence.append(f'<a href="{assets.name}/{name}">{name}</a>')
        if (folder / 'collision-checks.json').is_file():
            for check in read(folder / 'collision-checks.json'):
                for item in check['result'].get('contact_permissions_applied', []):
                    permissions[item['rule']] += item['points']
        reject_rows = ''.join(f'<tr><td>{html.escape(reason)}</td><td>{count}</td></tr>'
                              for reason, count in row.get('candidate_rejection_counts', {}).items())
        pair_rows = ''.join(f'<tr><td>{html.escape(pair)}</td><td>{count}</td></tr>'
                            for pair, count in sorted(row.get('collision_pair_counts', {}).items(), key=lambda item: -item[1]))
        cards.append(f'<article><h2>{label}: {status}</h2><p>{html.escape(failure)}</p>{media}'
                     f'<details><summary>Candidate rejections</summary><div class="tables">'
                     f'<table>{reject_rows}</table><table>{pair_rows}</table></div></details>'
                     f'<details><summary>Recorded evidence</summary><p>{" · ".join(evidence)}</p></details></article>')
    comparison = []
    for name, task in [('bowl', 8), ('drawer', 0)]:
        old_successes = sum(read(ROOT / f'results/physical-tasks-v6-eval/{name}/goal{task:02d}-state{state:03d}/summary.json')['official_task_success']
                            for state in [1, 11, 21, 31, 41])
        comparison.append(f'<tr><td>{name.capitalize()}</td><td>{old_successes}/5</td><td>0/5</td>'
                          f'<td>{data["successes_by_task"][name]}/{data["planned_by_task"][name]}</td></tr>')
    audit_links = []
    audit = ROOT / 'results/contact-policy-v8-development/successful-contact-audit-03'
    for name in ['summary.json', 'bowl-audit.json', 'drawer-audit.json']:
        if (audit / name).is_file():
            destination = f'contact-{name}'
            shutil.copyfile(audit / name, output / destination)
            audit_links.append(f'<a href="{destination}">{destination}</a>')
    total = data['planned_episodes']
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Correcting task contact permissions</title><style>
body{{margin:0;background:#f5f7fa;color:#1c2c3c;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1150px;margin:auto;padding:28px}}
h1{{font-size:32px;line-height:1.2}}h2{{font-size:21px}}article,.note,.metrics{{background:white;border:1px solid #dce4eb;border-radius:12px;padding:20px;margin:22px 0}}
.metrics{{display:flex;gap:30px;flex-wrap:wrap}}.metric strong{{display:block;font-size:29px}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border-bottom:1px solid #e2e7ed;padding:9px;text-align:left;vertical-align:top}}video{{width:100%;max-height:420px;background:#111}}
.scene{{width:100%;margin-top:14px}}.tables{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}a{{color:#176784}}summary{{cursor:pointer;margin-top:14px}}
</style><main><h1>Correcting task contact permissions</h1>
<p>v7.2 stopped all ten episodes at software grasp admission, before physical closure. Its pad-only contact rule falsely rejected a bowl grasp
that succeeded in v6: a finger mesh overlapped the selected bowl by 0.66 mm. The successful episode's simulator contact log confirms finger-to-bowl contact.
Drawer fingers also brace against the nearby cabinet front; those contacts were blocked by the old rule.</p>
<div class="metrics"><div class="metric"><strong>{data['completed_episodes']}/{total}</strong>Episodes recorded</div>
<div class="metric"><strong>{data['initial_clearance_passed']}/{total}</strong>Initial checks passed</div>
<div class="metric"><strong>{data['grasp_admitted']}/{total}</strong>Grasps admitted</div>
<div class="metric"><strong>{data['reached_closure']}/{total}</strong>Episodes reached closure</div></div>
<h2>Contacts allowed by the corrected policy</h2><table><tr><th>Pair</th><th>Where and when</th></tr>
<tr><td>Finger links ↔ selected bowl or handle</td><td>Functional finger surfaces, during grasping/holding/releasing. Permission is restricted to the selected RGB-D contact region.</td></tr>
<tr><td>Fingers ↔ local drawer front</td><td>Observed flat panel within 5 cm of the selected handle, with a 4 mm association band, during grasp/pull/release. Other protruding handles stay checked.</td></tr>
<tr><td>Bowl ↔ initial support / selected plate</td><td>Initial support while lifting; selected plate while placing/releasing. Other obstacles stay checked.</td></tr>
<tr><td>Moving drawer front ↔ cabinet seam</td><td>Observed boundary and shallow guide contact as the drawer moves; hidden rails/body are not certified.</td></tr>
<tr><td>Arm/palm ↔ scene; fingers ↔ other objects</td><td>Arm/palm keep the 1 mm clearance margin. Unrelated finger penetrations are blocked; positive finger clearance is not called a collision.</td></tr></table>
<p>These are automatic task/robot contact rules; GPT does not generate a global ignore list. Contact permission affects software admission.
MuJoCo still computes forces, gripper closure and object motion. No grasp attachment or object snapping is introduced.
<a href="https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html">MoveIt's pick-and-place tutorial</a>
also permits hand-link contact with the grasped object.</p>
<h2>Actual LIBERO success</h2><table><tr><th>Task</th><th>v6 palm-only admission</th><th>v7.2 restrictive contacts</th><th>{html.escape(data['pipeline_version'])}</th></tr>{''.join(comparison)}</table>
<p>Every planned episode remains in the denominator. These states were previously inspected; this is a regression experiment, not a held-out generalization test.
Plausible poses, admitted contacts and successful planning are intermediate measurements; the success count comes from the native LIBERO predicate.</p>
<div class="note"><strong>Limits:</strong> Full-arm collision checks use measured surfaces and a convex moving-object envelope, sampled at 0.025 rad.
Partial geometry and filled-in convex regions can still be conservative. Hidden bodies, unknown space and continuous edges are not certified.
The grasp selector still tries limited IK/path alternatives. VAMP checks robot self collision; ReKep's native optimizer and Atlas/pRRTC are not used for execution.</div>
<p>Same cached Luna grounding and GraspGen proposals as v6/v7.2, rechecked against fresh observations. No new API or GraspGen inference calls.
Earlier results are preserved. Offline contact audit (simulator identities are explanation-only and excluded from policy): {' · '.join(audit_links)}.</p>
<h2>Episodes</h2><table><tr><th>Task/state</th><th>Success</th><th>Failure stage</th><th>Controls</th><th>Reason</th></tr>{''.join(rows)}</table>
<p>Source freeze: {data['source_freeze_passed']}. Control budgets: {data['control_budgets_passed']}. API ledger unchanged: {data['api_ledger_unchanged']}.</p>
<p><a href="evaluation.json">Evaluation data</a> · <a href="protocol.json">Frozen protocol</a> · <a href="configuration.json">Configuration</a></p>{''.join(cards)}</main></html>'''
    (output / 'index.html').write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/physical-tasks-v8-eval')
    parser.add_argument('--output', default='results/task-grounding-contact-v8-report')
    args = parser.parse_args()
    base, output = ROOT / args.input, ROOT / args.output
    if output.exists():
        raise FileExistsError('Preserve reports')
    output.mkdir(parents=True)
    data = collect(base)
    save(output / 'evaluation.json', data)
    for name in ['protocol.json', 'configuration.json']:
        shutil.copyfile(base / name, output / name)
    render(data, output)
    print(json.dumps({key: value for key, value in data.items() if key not in ['episodes', 'source_freeze_checks']}))


if __name__ == '__main__':
    main()
