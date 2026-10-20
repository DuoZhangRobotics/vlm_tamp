"""Report frozen physical episodes separately from planner compatibility tests."""

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
from vlm_tamp.rekep_libero.geometry import pose_matrix, transform_points
from vlm_tamp.rekep_libero.object_motion_v3 import observed_object_cloud, verify_observed_transport
from vlm_tamp.rekep_libero.perception import _read_observation


def read(path):
    return json.loads(Path(path).read_text())


def drawer_observation(episode):
    if not (episode / 'final/metadata.json').is_file():
        return None
    reference = np.load(episode / 'grasp-cloud.npy', allow_pickle=False)
    closed = read(episode / 'closed/metadata.json')
    registration = read(episode / 'closure-surface-registration.json')
    cameras, final = _read_observation(episode / 'final/obs.npz', episode / 'final/metadata.json')
    closed_pose, final_pose = pose_matrix(closed['eef_pose_world_xyzw']), pose_matrix(final['eef_pose_world_xyzw'])
    motion = final_pose @ np.linalg.inv(closed_pose) @ registration['transform_initial_to_closed_world']
    observed = observed_object_cloud(cameras, reference, motion, final['robot_collision_volumes_world'])
    retention = verify_observed_transport(reference, observed, motion, final['gripper_qpos'], minimum_motion=.015)
    spec = read(episode / 'task-spec.json')
    center = reference.mean(axis=0)
    object_motion = transform_points([center], retention['transform_world'])[0] - center if 'transform_world' in retention else None
    return {'posthoc_observation_only': True, 'used_by_policy': False,
            'hand_progress_m': float((final_pose[:3, 3] - closed_pose[:3, 3]) @ spec['pull_axis_world']),
            'observed_surface_progress_m': float(object_motion @ spec['pull_axis_world']) if object_motion is not None else None,
            'requested_travel_m': spec['requested_travel_m'], 'retention': retention}


def export_regions(episode):
    """Diagnose mask-role association without replacing the frozen outcome."""
    if not (episode / 'opened/metadata.json').is_file():
        return None
    info = read(episode / 'summary.json')
    if info['task_index'] != 8 or 'separate target region' not in info.get('failure', ''):
        return None
    from vlm_tamp.rekep_libero.physical_tasks_v4 import cached_scene
    from vlm_tamp.rekep_libero.task_geometry_v4 import referenced_points
    source = ROOT / f'results/rekep-libero-v2/libero_goal-08-state{info["initial_state"]:03d}-seed0'
    bundle, program, _, selected = cached_scene(source, episode / 'opened')
    source_index = program.grasp_keypoints[0]
    candidates = []
    for index in referenced_points(program.stages[1]):
        pixel = tuple(bundle['pixels_yx'][index])
        overlapping = [int(mask.sum()) for mask in bundle['masks']
                       if mask[tuple(bundle['pixels_yx'][source_index])] and mask[pixel]]
        candidates.append({'landmark': index, 'inside_selected_source_mask': bool(selected[pixel]),
                           'overlapping_candidate_mask_areas': overlapping})
    return {'diagnostic_only': True, 'frozen_result_replaced': False, 'source_keypoint': source_index,
            'stage2_role_candidates': candidates,
            'cause': 'Adapter excludes target anchors covered by any overlapping source-containing SAM proposal, including broad masks',
            'recommended_fix': 'Use the admitted source mask and consistent mask groups to exclude source anchors; preserve ambiguous cases explicitly'}


def collect(output):
    rows, frozen_checks = [], {}
    planned = [(kind, task, state) for kind, task in [('bowl', 8), ('drawer', 0)] for state in [1, 11, 21, 31, 41]]
    for kind in ['bowl', 'drawer']:
        protocol = read(ROOT / f'results/physical-tasks-v4-eval/{kind}/protocol.json')
        frozen_checks[kind] = all(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
                                  for relative, digest in protocol['sources_sha256'].items())
    for kind, task, state in planned:
        episode = ROOT / f'results/physical-tasks-v4-eval/{kind}/goal{task:02d}-state{state:03d}'
        if not (episode / 'summary.json').is_file():
            rows.append({'kind': kind, 'task_index': task, 'initial_state': state, 'status': 'pending',
                         'official_task_success': False})
            continue
        row = read(episode / 'summary.json')
        actions = read(episode / 'initial/actions.json')
        verified_success = any(action['evaluation_task_success'] for action in actions)
        if verified_success != row['official_task_success']:
            raise ValueError('Driver success summary disagrees with the logged LIBERO predicate')
        row.update(kind=kind, relative_episode=str(episode.relative_to(ROOT)),
                   evaluation_success_recomputed_from_actions=verified_success,
                   final_task_success=bool(actions and actions[-1]['evaluation_task_success']),
                   control_budget_passed=len(actions) <= 600,
                   typed_geometry_exported=(episode / 'task-spec.json').is_file(),
                   grasp_candidate_selected=(episode / 'chosen-grasp.json').is_file(),
                   failure_category=row['failure_stage'] or ('none' if verified_success else 'task_goal_not_achieved'))
        if kind == 'drawer':
            diagnostic = drawer_observation(episode)
            if diagnostic:
                save(output / f'diagnostics/drawer-state{state:03d}.json', diagnostic)
                row['posthoc_drawer_observation'] = diagnostic
        region = export_regions(episode)
        if region:
            save(output / f'diagnostics/bowl-state{state:03d}-mask-association.json', region)
            row['posthoc_mask_association'] = region
        rows.append(row)
    counts = {kind: sum(row['official_task_success'] for row in rows if row['kind'] == kind) for kind in ['bowl', 'drawer']}
    ledger = read(ROOT / 'results/rekep-libero-api/ledger.json')
    result = {'planned_episodes': len(planned), 'completed_episodes': sum(row['status'] != 'pending' for row in rows),
              'official_task_successes': sum(counts.values()), 'successes_by_task': counts, 'denominator_by_task': 5,
              'failure_categories': dict(Counter(row.get('failure_category', 'pending') for row in rows
                                                if not row['official_task_success'])),
              'source_freeze_checks_passed': frozen_checks,
              'all_completed_control_budgets_passed': all(row.get('control_budget_passed', True) for row in rows),
              'new_api_calls': 0, 'api_cumulative_conservative_spend_usd': ledger['cumulative_conservative_spend_usd'],
              'api_cap_usd': ledger['budget_usd'], 'api_remaining_usd': ledger['remaining_usd'],
              'task_success_definition': 'LIBERO native predicate becomes true during ordinary physical execution',
              'native_rekep_optimizer_used': False, 'software_collision_checks': False,
              'physical_episode_planner': 'constraint-guided IK/OSC baseline',
              'initial_state_1_used_in_development': True, 'episodes': rows}
    save(output / 'evaluation.json', result)
    return result


def link(path, text):
    return f'<a href="{html.escape(path, quote=True)}">{html.escape(text)}</a>'


def render(result, output):
    cards, table = [], []
    for row in result['episodes']:
        title = f'{"Bowl onto plate" if row["kind"] == "bowl" else "Middle drawer"} · state {row["initial_state"]}'
        outcome = 'Success' if row['official_task_success'] else ('Pending' if row['status'] == 'pending' else 'Failed')
        reason = row.get('failure', '') or ('Commands completed; LIBERO goal remains false' if outcome == 'Failed' else '')
        stage = row.get('failure_category', row['status']).replace('_', ' ')
        table.append(f'<tr><td>{title}</td><td class="{outcome.lower()}">{outcome}</td><td>{html.escape(stage)}</td>'
                     f'<td>{row.get("control_steps", "—")}/600</td><td>{html.escape(reason)}</td></tr>')
        if row['status'] == 'pending':
            continue
        episode = ROOT / row['relative_episode']
        assets = output / f'{row["kind"]}-state{row["initial_state"]:03d}'
        assets.mkdir(parents=True, exist_ok=True)
        video = episode / 'initial/rollout.mp4'
        preview = ''
        if video.is_file():
            shutil.copyfile(video, assets / 'rollout.mp4')
            preview = f'<video controls preload="metadata" src="{assets.name}/rollout.mp4"></video>'
        figures = []
        for name in ['numbered.png', 'final/agentview.png', 'probe-held/frontview.png', 'lifted-held/agentview.png']:
            source = episode / name
            if source.is_file():
                destination = assets / name.replace('/', '-')
                shutil.copyfile(source, destination)
                figures.append(f'<img loading="lazy" src="{assets.name}/{destination.name}" alt="{html.escape(name)}">')
        files = []
        for filename in ['summary.json', 'task-spec.json', 'bound-goal.json', 'bound-path-constraint.json',
                         'chosen-grasp.json', 'closure-surface-registration.json', 'retention.json',
                         'command-log.json', 'cached-program/output_raw.txt']:
            source = episode / filename
            if source.is_file():
                destination = assets / filename.replace('/', '-')
                shutil.copyfile(source, destination)
                files.append(link(f'{assets.name}/{destination.name}', filename))
        extra = ''
        if row.get('posthoc_drawer_observation'):
            diagnostic = row['posthoc_drawer_observation']
            progress = diagnostic['observed_surface_progress_m']
            extra = (f'<p>Observed handle-surface progress: {progress * 100:.1f} cm; requested: 14 cm. '
                     'LIBERO requires the cabinet joint to pass its 14 cm opening threshold. '
                     'The native predicate stays false; observed progress alone does not count as success.</p>')
        if row.get('posthoc_mask_association'):
            extra += '<p>The cached program contains target references. The adapter rejected them because a broad overlapping mask covers both bowl and plate.</p>'
        cards.append(f'<article><h3>{title}: <span class="{outcome.lower()}">{outcome}</span></h3>'
                     f'<p>{html.escape(reason)}</p>{extra}{preview}<details><summary>Images and saved evidence</summary>'
                     f'<div class="images">{"".join(figures)}</div><p>{" · ".join(files)}</p></details></article>')
    fixture_rows = []
    for kind in ['projected', 'atlas', 'mc-vamp']:
        folder = ROOT / f'results/planner-feasibility-v4/{kind}-02'
        info = read(folder / 'summary.json')
        metrics = info.get('path_metrics', {})
        if metrics:
            residual = f'{metrics["maximum_line_error_m"] * 1000:.3f} mm'
        else:
            residual = '—'
        shutil.copytree(folder, output / f'planner-{kind}', dirs_exist_ok=True)
        fixture_rows.append(f'<tr><td>{html.escape(info["planner"])}</td><td>{"Solved" if info["exact_solution"] else "Timeout"}</td>'
                            f'<td>{metrics.get("states", "—")}</td><td>{residual}</td>'
                            f'<td>{link(f"planner-{kind}/summary.json", "Details")}</td></tr>')
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Physical tasks and constrained planner feasibility</title><style>
body{{font:17px/1.55 system-ui,sans-serif;background:#f4f6f8;color:#182531;margin:0}}main{{max-width:1100px;margin:auto;padding:32px 24px}}h1{{font-size:34px;line-height:1.2}}h2{{font-size:24px}}h3{{font-size:20px}}
.metrics{{display:flex;gap:16px;flex-wrap:wrap}}.metric,article,.note{{background:white;border:1px solid #dce3e8;border-radius:12px;padding:20px;margin:20px 0}}.metric{{flex:1;min-width:180px}}.metric strong{{display:block;font-size:36px}}
table{{width:100%;border-collapse:collapse;background:white;font-size:15px}}td,th{{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #dce3e8}}.success{{color:#067a53}}.failed{{color:#a03b32}}.pending{{color:#88610c}}video{{max-width:100%;width:560px;border-radius:8px}}.images{{display:flex;flex-wrap:wrap;gap:12px}}img{{width:280px;max-width:100%;border-radius:8px}}a{{color:#17619a}}details{{margin-top:18px}}.small{{font-size:14px;color:#536676}}
</style><main><h1>Physical tasks and constrained planner feasibility</h1>
<p>Frozen automatic pipeline: RGB-selected landmarks and depth → GraspGen → typed task goals → IK and pose control. Success is LIBERO's actual task check.</p>
<div class="metrics"><div class="metric">Bowl onto plate<strong>{result['successes_by_task']['bowl']}/5</strong></div><div class="metric">Open middle drawer<strong>{result['successes_by_task']['drawer']}/5</strong></div><div class="metric">New API spend<strong>$0</strong><span class="small">Cumulative ledger: ${result['api_cumulative_conservative_spend_usd']:.4f} / $2</span></div></div>
<p>{result['completed_episodes']}/10 planned episodes finished. States 1, 11, 21, 31, 41; simulation seed 0; five settling actions; maximum 600 controls each. Every planned episode is counted, including failures before motion.</p>
<div class="note"><strong>What this tests</strong><p>Normal physical contacts remain enabled. Software collision admission is bypassed. No object is snapped or attached in the simulator. State 1 was used for development. The physical episodes use an IK/OSC baseline; native ReKep optimization and the planner fixtures below are separate measurements.</p></div>
<h2>Physical episode outcomes</h2><table><thead><tr><th>Task and state</th><th>LIBERO result</th><th>Stage</th><th>Controls</th><th>Recorded reason</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<h2>Why we can export numerical constraints</h2><p>The VLM still selects landmark roles from its cached image request. New software fills two predefined templates from metric depth: a terminal placement point and a drawer pull line. Observed object motion and the robot TCP bind the placement point after a retained lift. This does not certify the estimates or compile arbitrary Python constraints. Mask association, grasp retention and collision geometry still fail in some states.</p>
<h2>Planner compatibility tests</h2><p>These tests use a 10 cm segment of the observed drawer-line proposal, fixed grasp orientation, calibrated seven-joint Panda kinematics and native VAMP self-collision checks. The selected grasp did not yield an accurate 14 cm IK goal. The environment is empty during planning; no physical drawer task was executed by these planners.</p>
<table><thead><tr><th>Planner</th><th>Result</th><th>Sampled states</th><th>Maximum line error</th><th>Evidence</th></tr></thead><tbody>{''.join(fixture_rows)}</tbody></table>
<p>MC-VAMP's sparse endpoints require constraint-aware resampling. The planners use different projection tolerances, and this is one compatibility run per setting; timing is not a fair speed benchmark. Native VAMP accepts observed depth points, but the unadapted collision model rejected both fixture endpoints with no intended-contact exceptions.</p>
<p>Recommended order: use Atlas as a general reference and MC-VAMP for the compiled line/orientation template. Validate scene clearance and grasp-to-manipulation reachability before physical integration. nvblox can provide an ESDF, but it does not replace robot geometry or contact handling.</p>
<p>{link('planner-feasibility.md', 'Detailed feasibility analysis and reproduction')} · {link('evaluation.json', 'Full evaluation data')}</p>
<h2>Videos and grounding evidence</h2>{''.join(cards)}
<p class="small">Original strict ReKep results are preserved. Cached Luna high-reasoning requests were reused; no new inventory, API retry or paid request was supplied. Source freeze checks: {html.escape(str(result['source_freeze_checks_passed']))}.</p></main></html>'''
    (output / 'index.html').write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/task-planner-v4-report')
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'docs/planner_feasibility_v4.md', output / 'planner-feasibility.md')
    result = collect(output)
    render(result, output)
    print(json.dumps({key: value for key, value in result.items() if key != 'episodes'}), flush=True)


if __name__ == '__main__':
    main()
