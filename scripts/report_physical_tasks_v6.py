"""Compare preserved frozen cohorts and report the latest physical evidence."""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from try_rekep_bowl_v3 import save


def read(path):
    return json.loads(Path(path).read_text())


def collect(version):
    rows, hashes = [], {}
    for kind, task in [('bowl', 8), ('drawer', 0)]:
        base = ROOT / f'results/physical-tasks-v{version}-eval/{kind}'
        protocol = read(base / 'protocol.json')
        hashes[kind] = all(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest
                           for filename, digest in protocol['sources_sha256'].items())
        for state in [1, 11, 21, 31, 41]:
            folder = base / f'goal{task:02d}-state{state:03d}'
            if not (folder / 'summary.json').is_file():
                rows.append({'kind': kind, 'initial_state': state, 'status': 'pending', 'official_task_success': False})
                continue
            row = read(folder / 'summary.json')
            actions = read(folder / 'initial/actions.json')
            success = any(action['evaluation_task_success'] for action in actions)
            if success != row['official_task_success']:
                raise ValueError('Success summary disagrees with the logged LIBERO predicate')
            row.update(kind=kind, folder=str(folder.relative_to(ROOT)),
                       evaluation_recomputed_from_actions=True, control_budget_passed=len(actions) <= 600,
                       typed_grounding_completed=(folder / 'task-spec.json').is_file(),
                       failure_category=row.get('failure_stage') or ('none' if success else 'task_goal_not_achieved'))
            if (folder / 'grasp-admission.json').is_file():
                admission = read(folder / 'grasp-admission.json')
                row['grasp_rejection_counts'] = dict(Counter(item['reason'] for item in admission['rejected_candidates']))
            rows.append(row)
    return {'version': version, 'planned_episodes': 10, 'planned_per_task': 5,
            'completed_episodes': sum(row['status'] != 'pending' for row in rows),
            'successes_by_task': {kind: sum(row['official_task_success'] for row in rows if row['kind'] == kind)
                                  for kind in ['bowl', 'drawer']},
            'grounding_completed': sum(row.get('typed_grounding_completed', False) for row in rows),
            'source_freeze_checks': hashes,
            'all_completed_control_budgets_passed': all(row.get('control_budget_passed', True) for row in rows),
            'failure_categories': dict(Counter(row.get('failure_category', 'pending') for row in rows if not row['official_task_success'])),
            'episodes': rows}


def draw_feedback(folder, destination):
    data = read(folder / 'drawer-feedback.json')
    if len(data) < 2 or not all('observed_progress_m' in row for row in data):
        return False
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    x = [row['control_steps'] for row in data]
    figure, axes = plt.subplots(figsize=(6.6, 3.2), constrained_layout=True)
    axes.plot(x, [100 * row['expected_hand_progress_m'] for row in data], '--', label='Measured hand movement', color='#5b7289')
    axes.plot(x, [100 * row['observed_progress_m'] for row in data], 'o-', label='Observed handle movement', color='#12845a')
    axes.axhline(15, color='#bd762a', linestyle=':', label='Configured observed goal')
    axes.set(xlabel='Control steps', ylabel='Distance from closed state (cm)', title='Pull feedback from RGB-D')
    axes.grid(alpha=.2)
    axes.legend(fontsize=8)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return True


def render(cohorts, output):
    latest = cohorts[-1]
    rows, cards = [], []
    comparison = ''.join(f'<tr><td>v{item["version"]}</td><td>{item["successes_by_task"]["bowl"]}/5</td>'
                         f'<td>{item["successes_by_task"]["drawer"]}/5</td><td>{item["grounding_completed"]}/10</td>'
                         f'<td>{item["completed_episodes"]}/10</td></tr>' for item in cohorts)
    for row in latest['episodes']:
        kind, state = row['kind'], row['initial_state']
        label = f'{"Bowl onto plate" if kind == "bowl" else "Middle drawer"} · state {state}'
        status = 'Success' if row['official_task_success'] else ('Pending' if row['status'] == 'pending' else 'Failed')
        detail = row.get('failure', '') or ('Commands completed; LIBERO goal remains false' if status == 'Failed' else '')
        rows.append(f'<tr><td>{label}</td><td class="{status.lower()}">{status}</td>'
                    f'<td>{row.get("control_steps", "—")}/600</td><td>{html.escape(row.get("failure_category", "pending"))}</td>'
                    f'<td>{html.escape(detail)}</td></tr>')
        if status == 'Pending':
            continue
        folder = ROOT / row['folder']
        assets = output / f'{kind}-state{state:03d}'
        assets.mkdir(parents=True, exist_ok=True)
        video = folder / 'initial/rollout.mp4'
        video_html = ''
        if video.is_file():
            shutil.copyfile(video, assets / 'rollout.mp4')
            video_html = f'<video controls preload="metadata" src="{assets.name}/rollout.mp4"></video>'
        images = []
        for name in ['numbered.png', 'source-mask-overlay.png', 'target-mask-overlay.png', 'final/agentview.png']:
            if (folder / name).is_file():
                destination = assets / name.replace('/', '-')
                shutil.copyfile(folder / name, destination)
                images.append(f'<img loading="lazy" src="{assets.name}/{destination.name}" alt="{html.escape(name)}">')
        plot = ''
        if kind == 'drawer' and draw_feedback(folder, assets / 'feedback.png'):
            plot = f'<img class="plot" src="{assets.name}/feedback.png" alt="Observed handle travel and measured hand motion">'
        evidence = []
        for filename in ['summary.json', 'task-spec.json', 'chosen-grasp.json', 'grasp-admission.json',
                         'bound-goal.json', 'bound-path-constraint.json', 'drawer-feedback.json',
                         'placement-feedback.json', 'retention.json', 'observed-clearance.json', 'command-log.json']:
            if (folder / filename).is_file():
                shutil.copyfile(folder / filename, assets / filename)
                evidence.append(f'<a href="{assets.name}/{filename}">{filename}</a>')
        correction = ('<p>The original cached landmark was at another handle height. Execution used the observed middle bar; this correction is recorded separately from Luna.</p>'
                      if row.get('automatic_ordinal_role_correction') else '')
        cards.append(f'<article><h3>{label}: <span class="{status.lower()}">{status}</span></h3><p>{html.escape(detail)}</p>'
                     f'{correction}{video_html}{plot}<details><summary>Grounding images and recorded evidence</summary>'
                     f'<div class="images">{"".join(images)}</div><p>{" · ".join(evidence)}</p></details></article>')
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grounding and execution corrections</title><style>
body{{font:17px/1.55 system-ui,sans-serif;background:#f4f6f8;color:#182531;margin:0}}main{{max-width:1120px;margin:auto;padding:30px 24px}}h1{{font-size:34px;line-height:1.2}}h2{{font-size:24px}}h3{{font-size:20px}}
.metrics{{display:flex;gap:16px;flex-wrap:wrap}}.metric,article,.note{{background:white;border:1px solid #dce3e8;border-radius:12px;padding:20px;margin:20px 0}}.metric{{flex:1;min-width:180px}}.metric strong{{display:block;font-size:36px}}
table{{width:100%;border-collapse:collapse;background:white;font-size:15px}}td,th{{text-align:left;vertical-align:top;padding:11px;border-bottom:1px solid #dce3e8}}.success{{color:#067a53}}.failed{{color:#a03b32}}.pending{{color:#88610c}}video{{max-width:100%;width:530px;border-radius:8px}}.images{{display:flex;flex-wrap:wrap;gap:12px}}img{{width:260px;max-width:100%;border-radius:8px}}img.plot{{width:590px;display:block;margin:16px 0}}a{{color:#17619a}}details{{margin-top:16px}}.small{{font-size:14px;color:#536676}}
</style><main><h1>Grounding and execution corrections</h1><p>Task regions now compile from all ten cached RGB-D scenes. Grasp admission checks subsequent manipulation, bowl placement uses a point goal, and drawer pulling checks observed handle progress after each short command.</p>
<div class="metrics"><div class="metric">Bowl onto plate<strong>{latest['successes_by_task']['bowl']}/5</strong></div><div class="metric">Middle drawer<strong>{latest['successes_by_task']['drawer']}/5</strong></div><div class="metric">New API spend<strong>$0</strong></div></div>
<p>{latest['completed_episodes']}/10 latest episodes completed. States 1, 11, 21, 31, 41; seed 0; five settling actions; at most 600 control steps. Success is LIBERO's native predicate, recomputed from the action log. Every planned state is in the denominator.</p>
<h2>Preserved experiment comparison</h2><table><thead><tr><th>Frozen version</th><th>Bowl success</th><th>Drawer success</th><th>Typed grounding exported</th><th>Episodes finished</th></tr></thead><tbody>{comparison}</tbody></table>
<p>v4 is the previous baseline. v5 fixed mask/ordinal grounding and added complete-task IK and observed pulling. v6 added sampled palm-penetration admission and removed hard wrist orientation from bowl lifting/placement. Each row is a separate ten-episode cohort; failed episodes and development attempts are preserved.</p>
<div class="note"><strong>What changed in grounding</strong><p>A broad overlapping mask no longer deletes valid plate roles. For drawers, the adapter checks three observed protruding bars and chooses their middle height, then fits the adjacent front panel. Three original Luna handle choices required this geometry correction. This is a documented task-template adaptation, not a claim of improved raw VLM predictions.</p></div>
<div class="note"><strong>Planning scope</strong><p>Normal simulator contact physics stays enabled, with no object snapping or assisted attachment. Native ReKep and Atlas optimizers are not used during these episodes. Robot self-collision is checked, and sampled palm intersections with observed surfaces can reject a grasp. Other arm/finger intersections remain diagnostics; unknown space, continuous clearance and the carried object's complete shape are not validated.</p></div>
<h2>Latest episode outcomes</h2><table><thead><tr><th>Task</th><th>LIBERO result</th><th>Controls</th><th>Stage</th><th>Recorded reason</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Videos and observed feedback</h2>{''.join(cards)}
<p><a href="implementation.md">Implementation and reproduction</a> · <a href="evaluation.json">Full comparison data</a></p>
<p class="small">All states were inspected previously; this is a development comparison rather than a held-out generalization estimate. Changes were tested together, so success differences do not isolate a single component's effect. Original strict results are preserved.</p></main></html>'''
    (output / 'index.html').write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/task-grounding-execution-v6-report')
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    cohorts = [collect(version) for version in [4, 5, 6]]
    ledger = read(ROOT / 'results/rekep-libero-api/ledger.json')
    result = {'cohorts': cohorts, 'new_api_calls': 0,
              'api_cumulative_conservative_spend_usd': ledger['cumulative_conservative_spend_usd'],
              'api_cap_usd': ledger['budget_usd'], 'api_remaining_usd': ledger['remaining_usd']}
    save(output / 'evaluation.json', result)
    shutil.copyfile(ROOT / 'docs/physical_tasks_v6.md', output / 'implementation.md')
    render(cohorts, output)
    print(json.dumps([{key: value for key, value in item.items() if key != 'episodes'} for item in cohorts]), flush=True)


if __name__ == '__main__':
    main()
