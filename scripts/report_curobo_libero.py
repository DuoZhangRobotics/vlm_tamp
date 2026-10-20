"""Native-success comparison with videos, IK evidence, and explicit scope."""

import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/physical-tasks-curobo-eval')
    parser.add_argument('--output', default='results/task-grounding-curobo-report')
    args = parser.parse_args()
    base, output = ROOT/args.input, ROOT/args.output
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(base/'manifest.json')
    config = read(base/'configuration.json')
    rows, cards = [], []

    def link(path, label):
        return '<a href="'+html.escape(os.path.relpath(path, output), quote=True)+'">'+html.escape(label)+'</a>'

    for task, state in manifest['planned_episodes']:
        folder = base/f'goal{task:02d}-state{state:03d}'
        summary = read(folder/'summary.json')
        baseline = read(ROOT/f'results/physical-tasks-v11-eval/goal{task:02d}-state{state:03d}/summary.json')
        actions = read(folder/'initial/actions.json', [])
        native = any(row['evaluation_task_success'] for row in actions)
        if summary['official_task_success'] != native:
            raise ValueError('Native success disagrees with action logs')
        admission = read(folder/'grasp-admission.json', {})
        chosen = read(folder/'chosen-grasp.json', {})
        queries = read(folder/'curobo-queries.json', [])
        frame = read(folder/'curobo-frame-validation.json', {})
        failure_reasons = dict(Counter(row['reason'] for row in admission.get('rejected_candidates', [])))
        margin = chosen.get('minimum_grasp_joint_reserve_rad')
        row = {**summary, 'baseline_v11_success': baseline['official_task_success'],
            'folder': str(folder.relative_to(ROOT)), 'native_success_recomputed': native,
            'ik_queries': len(queries), 'tested_ik_postures': admission.get('tested_ik_postures', 0),
            'grasp_joint_reserve_rad': margin, 'ik_alternative': chosen.get('ik_alternative'),
            'grasp_rejection_reasons': failure_reasons, 'frame_validation_passed': frame.get('accepted', False),
            'peak_curobo_gpu_allocated_bytes': max((q['peak_gpu_allocated_bytes'] for q in queries), default=0)}
        rows.append(row)
        name = 'Bowl onto plate' if task == 8 else 'Middle drawer'
        outcome = 'SUCCESS' if native else 'FAILED'
        details = [link(folder/'summary.json', 'Episode outcome'), link(folder/'command-log.json', 'Executed commands')]
        for filename, label in [('chosen-grasp.json','Selected grasp'), ('grasp-admission.json','Rejected postures'),
            ('curobo-queries.json','GPU IK queries'), ('curobo-frame-validation.json','Frame calibration'),
            ('collision-checks.json','Contact and collision checks'), ('drawer-feedback.json','Observed motion')]:
            if (folder/filename).is_file():
                details.append(link(folder/filename, label))
        reason = html.escape(summary.get('failure') or 'Native success predicate became true during physical execution.')
        video = folder/'initial/rollout.mp4'
        video_html = (f'<video controls preload="metadata" src="{html.escape(os.path.relpath(video, output))}"></video>'
                      if video.is_file() else '<p>No video was produced.</p>')
        images = []
        camera = 'frontview' if task == 0 else 'agentview'
        for stage in ['opened', 'at-grasp', 'closed', 'final']:
            image = folder/stage/f'{camera}.png'
            if image.is_file():
                images.append(f'<figure><img src="{html.escape(os.path.relpath(image, output))}" alt="{stage}">'
                              f'<figcaption>{stage}</figcaption></figure>')
        selected_text = (f"Grasp {chosen['index']} / {html.escape(chosen['variant'])}; "
                         f"IK alternative {chosen['ik_alternative']+1} of {chosen['ik_alternatives_available']}; "
                         f"endpoint joint reserve {margin:.3f} rad." if chosen else 'No grasp admitted.')
        cards.append(f'<article><h2>{name} · state {state}: <span class="{outcome.lower()}">{outcome}</span></h2>'
            f'<p>{reason}</p><p>{selected_text} Tested {row["tested_ik_postures"]} IK postures; '
            f'{len(queries)} GPU queries; {summary["control_steps"]}/600 control steps.</p>{video_html}'
            f'<div class="images">{"".join(images)}</div><p>{" · ".join(details)}</p>'
            f'<details><summary>Preflight rejection counts</summary><pre>{html.escape(json.dumps(failure_reasons, indent=2))}</pre></details></article>')
    successes = sum(r['native_success_recomputed'] for r in rows)
    baseline_successes = sum(r['baseline_v11_success'] for r in rows)
    table = []
    for row in rows:
        table.append(f'<tr><td>{"Bowl" if row["task_index"]==8 else "Drawer"}</td><td>{row["initial_state"]}</td>'
            f'<td>{"✓" if row["baseline_v11_success"] else "—"}</td><td>{"✓" if row["native_success_recomputed"] else "—"}</td>'
            f'<td>{html.escape(row.get("failure_stage") or "complete")}</td><td>{row["tested_ik_postures"]}</td>'
            f'<td>{row["control_steps"]}</td></tr>')
    groups = []
    for task, label in [(8, 'Bowl'), (0, 'Drawer')]:
        cohort = [row for row in rows if row['task_index']==task]
        groups.append(f'{label}: {sum(row["native_success_recomputed"] for row in cohort)}/{len(cohort)}')
    ledger = read(ROOT/'results/rekep-libero-api/ledger.json')
    evaluation = {'planned_episodes': len(rows), 'official_task_successes': successes,
        'baseline_v11_successes': baseline_successes, 'episodes': rows,
        'curobo_used_for': 'multiple grasp and pose IK proposals', 'curobo_motiongen_used': False,
        'contact_policy_changed_from_v11': False, 'command_joint_reserve_rad': config['command_joint_reserve_rad'],
        'collision_free_guarantee': False, 'new_api_calls': 0,
        'cumulative_api_spend_usd': ledger['cumulative_conservative_spend_usd'],
        'remaining_api_budget_usd': ledger['remaining_usd']}
    pilots = []
    for name, task, state in [('drawer11', 0, 11), ('bowl21', 8, 21),
                              ('drawer11-v2', 0, 11), ('bowl21-v2', 8, 21)]:
        pilot_path = ROOT/f'results/curobo-development/{name}/goal{task:02d}-state{state:03d}/summary.json'
        pilot = read(pilot_path)
        if pilot is not None:
            pilots.append({**pilot, 'summary_path': str(pilot_path.relative_to(ROOT)), 'development_run': name})
    evaluation['preserved_development_episodes'] = pilots
    pilot_table = ''.join(f'<tr><td>{html.escape(p["development_run"])}</td>'
        f'<td>{"success" if p["official_task_success"] else "failure"}</td>'
        f'<td>{html.escape(p.get("failure_stage") or "complete")}</td>'
        f'<td>{link(ROOT/p["summary_path"], "Preserved outcome")}</td></tr>' for p in pilots)
    (output/'evaluation.json').write_text(json.dumps(evaluation, indent=2)+'\n')
    for name in ['configuration.json','protocol.json','manifest.json']:
        shutil.copyfile(base/name, output/name)
    document = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>cuRobo IK on LIBERO</title><style>
body{{font-family:system-ui,sans-serif;background:#f3f5f8;color:#172331;max-width:1160px;margin:auto;padding:28px;line-height:1.55}}
h1{{font-size:2rem;margin-bottom:.3em}} .lead{{font-size:1.15rem}} article,.box{{background:white;padding:22px;border:1px solid #dce3eb;border-radius:12px;margin:22px 0}}
video{{width:100%;max-height:540px;background:#101820;border-radius:8px}} .images{{display:flex;gap:10px;flex-wrap:wrap}} figure{{margin:12px 0;flex:1;min-width:170px}} img{{width:100%;border-radius:6px}} figcaption{{font-size:.85rem;color:#5a6675}}
table{{width:100%;border-collapse:collapse;background:white}} td,th{{text-align:left;padding:9px;border-bottom:1px solid #dce3eb}}
.success{{color:#08784d}} .failed{{color:#af3345}} a{{color:#1c60a0}} pre{{overflow:auto;font-size:.85rem}}
</style></head><body><h1>cuRobo IK on contact-rich LIBERO tasks</h1>
<p class="lead">Physical success: <b>{successes}/{len(rows)}</b> · {' · '.join(groups)}. The matched v11 cohort achieved {baseline_successes}/{len(rows)}.</p>
<div class="box"><b>What changed</b><p>cuRobo optimizes 64 IK seeds per pose and returns up to 32 solutions. Up to 16 distinct robot-valid postures are tried for each GraspGen pose. Joint commands retain a 0.025 rad reserve from the configured limits. One rejected posture no longer discards its grasp. The final approach is solved backward from the selected grasp posture, so closure and manipulation use the same joint branch that was admitted.</p>
<b>What the result measures</b><p>Success means the native LIBERO predicate became true during physical execution. All {len(rows)} planned episodes count, including pre-motion failures. States were previously inspected; this is a matched development comparison, not held-out generalization.</p>
<p>The comparison measures the adapter as a whole: multiple IK alternatives, joint reserve, and consistent approach endpoints change together. It does not isolate GPU versus CPU IK or establish that cuRobo alone caused a success-rate change.</p>
<b>Contact and planning scope</b><p>Finger contact with the selected handle/object and local panel, and bowl/support contact, retain v11's explicit permissions. VAMP checks robot self collision; full robot collision meshes are checked against measured RGB-D surfaces with moving-object updates. cuRobo's internal scene collision checker and MotionGen are not used in this adapter. The approach and manipulation templates remain, with bounded CPU point-goal refinement for bowl placement.</p>
<p>These are contact-rich physical trials. Partial observed geometry, sampled checks, and controller tracking do not establish a collision-free guarantee. Simulator physics remains enabled; no grasp attachment is injected.</p></div>
<table><thead><tr><th>Task</th><th>State</th><th>v11</th><th>cuRobo IK</th><th>Stop stage</th><th>Postures tried</th><th>Controls</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<details class="box"><summary>Preserved development trials before freezing this comparison</summary>
<p>The initial bowl pilot failed during approach. Correcting approach endpoint consistency and local branch continuation preceded the final freeze. The corrected bowl and drawer pilots were rerun. These explicitly marked development trials are preserved separately from the fixed ten-episode comparison.</p>
<table><tr><th>Pilot</th><th>Native outcome</th><th>Stop stage</th><th>Evidence</th></tr>{pilot_table}</table></details>
<p>No new model API requests. Cumulative conservative spend: ${ledger['cumulative_conservative_spend_usd']:.6f}; remaining cap: ${ledger['remaining_usd']:.6f}.</p>
<p>{link(output/'evaluation.json','Measurements')} · {link(output/'configuration.json','Frozen configuration')} · {link(output/'protocol.json','Source hashes and scope')} · {link(ROOT/'docs/curobo_libero.md','Reproduction notes')} · <a href="https://github.com/NVlabs/curobo/tree/v0.7.8">Pinned upstream cuRobo</a></p>
{''.join(cards)}</body></html>'''
    (output/'index.html').write_text(document)
    print(json.dumps({k:v for k,v in evaluation.items() if k!='episodes'}))


if __name__ == '__main__':
    main()
