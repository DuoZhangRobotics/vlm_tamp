"""Combine completed opening/closing trials that used identical frozen sources."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shlex
import shutil


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opening-report', required=True)
    parser.add_argument('--closing-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    source_reports = [ROOT / args.opening_report, ROOT / args.closing_report]
    rows, protocols, checks = [], [], []
    for folder, task in zip(source_reports, ['open_microwave', 'close_microwave']):
        result, validation = read(folder / 'evaluation.json'), read(folder / 'validation.json')
        matches = [row for row in result['episodes'] if row['task'] == task]
        if len(matches) != 1 or result['planned'] != 1:
            raise ValueError('Expected one completed trial in each source report')
        row = matches[0]
        run = (ROOT / row['episode']).parent
        protocol = read(run / 'protocol.json')
        if read(run / 'summary.json')['completed'] != 1:
            raise ValueError('An episode is incomplete')
        protocols.append(protocol)
        checks.append(validation)
        rows.append({**row, 'source_report': str(folder.relative_to(ROOT)),
                     'run': str(run.relative_to(ROOT))})
    same_sources = protocols[0]['source_sha256'] == protocols[1]['source_sha256']
    if not same_sources or not all(all(value is True for key, value in check.items()
                                      if key != 'all_development_passes_included') for check in checks):
        raise ValueError('Source identity or trial validation failed')
    output.mkdir(parents=True, exist_ok=False)
    cards = []
    for row, folder in zip(rows, source_reports):
        task = row['task']
        shutil.copyfile(folder / f'{task}.mp4', output / f'{task}.mp4')
        if (folder / f'{task}-contact.mp4').exists():
            shutil.copyfile(folder / f'{task}-contact.mp4', output / f'{task}-contact.mp4')
        shutil.copyfile(folder / f'{task}-grounded-model.json', output / f'{task}-grounded-model.json')
        picture = folder / f'{task}-hinge-fit.png'
        if picture.exists():
            shutil.copyfile(picture, output / picture.name)
        cards.append(f'<section><h2>{task.replace("_", " ").capitalize()}</h2>'
                     f'<video controls preload="metadata" src="{task}.mp4"></video>'
                     + (f'<img src="{picture.name}" alt="Observed handle path and fitted hinge">'
                        if picture.exists() else '') + '</section>')
    ledger = read(ROOT / 'results/articulated-api-v1/ledger.json')
    history = []
    development = ROOT / 'results/handle-contact-development'
    def number(path):
        match = re.fullmatch(r'pilot-v(\d+)', path.name)
        return int(match[1]) if match else -1
    for folder in sorted((p for p in development.glob('pilot-v*') if p.is_dir()), key=number):
        if not (folder / 'manifest.json').exists():
            continue
        manifest = read(folder / 'manifest.json')
        summary = read(folder / 'summary.json') if (folder / 'summary.json').exists() else {'episodes': []}
        interrupted = read(folder / 'infrastructure-abort.json') if (folder / 'infrastructure-abort.json').exists() else None
        history.append({'run': str(folder.relative_to(ROOT)),
                        'planned': len(manifest['planned_episodes']),
                        'completed': len(summary['episodes']),
                        'native_successes': sum(r['official_task_success'] for r in summary['episodes']),
                        'status': 'infrastructure_aborted' if interrupted else
                                  'completed' if len(summary['episodes']) == len(manifest['planned_episodes']) else 'incomplete',
                        'infrastructure_abort': interrupted})
    comparison = []
    for version in dict.fromkeys(['pilot-v6', 'pilot-v8', 'pilot-v9', 'pilot-v11', 'pilot-v13', 'pilot-v15', 'pilot-v17', rows[1]['run'].split('/')[-1]]):
        folder = development / version / 'close_microwave-state001'
        if not (folder / 'summary.json').exists():
            continue
        episode = read(folder / 'summary.json')
        steps = read(folder / 'initial/evaluation_contact_diagnostics.json')['steps']
        joint = next(j for j in steps[-1]['evaluation_only_articulation_state']
                     if j['joint_name'] == 'microwave_1_microjoint')
        import math
        comparison.append({'version': version, 'final_open_angle_deg': max(0.,-math.degrees(joint['position'])),
                           'native_success': episode['official_task_success'],
                           'failure': episode.get('failure'), 'control_steps': episode['control_steps']})
    result = {'development_only': True, 'same_previously_inspected_state': 1,
              'held_out_generalization_claim': False, 'planned': 2, 'completed': 2,
              'native_successes_ever': sum(r['native_success_ever'] for r in rows),
              'native_successes_final': sum(r['native_success_final'] for r in rows),
              'identical_frozen_pipeline_sources': same_sources,
              'episodes': rows, 'closing_development_comparison': comparison,
              'all_development_passes': history, 'new_api_calls': 0,
              'api_spend_usd': ledger['cumulative_conservative_spend_usd'],
              'remaining_api_budget_usd': ledger['remaining_usd'],
              'targeted_test_record': 'results/handle-contact-development/resume-targeted-tests-20261015.json',
              'source_report_validations': checks,
              'report_source_sha256': digest(__file__),
              'limitations': ['One previously inspected official state per task',
                              'Upright door and vertical handle priors',
                              'Cached GraspGen proposals supplemented by RGB-D handle templates',
                              'Panel contact ownership uses local spatial association to the tracked handle and panel estimate',
                              'Observed-space collision checks; no hidden-space or swept-door certificate',
                              'A small wrist turn or tilt may slide at the contact; subsequent RGB-D motion is checked']}
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'pipeline-source-hashes.json').write_text(json.dumps(protocols[0]['source_sha256'], indent=2) + '\n')
    tests = read(ROOT / result['targeted_test_record'])
    if any(tests['source_sha256'].get(name) != protocols[0]['source_sha256'][name]
           for name in ['src/vlm_tamp/handle_contact.py', 'scripts/run_handle_contact.py']):
        raise ValueError('Targeted checks do not match the frozen pipeline')
    (output / 'targeted-tests.json').write_text(json.dumps(tests, indent=2) + '\n')
    runner='.runtime/envs/rekep-libero/bin/python'
    reproduction=['#!/usr/bin/env bash','set -euo pipefail',f'cd {shlex.quote(str(ROOT))}',
                  'out_root="${1:?Supply a new output directory, for example results/handle-contact-fresh}"',
                  'if [ -e "$out_root" ]; then echo "Output directory already exists" >&2; exit 1; fi',
                  'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src',
                  f'{runner} scripts/run_handle_contact.py --output "$out_root/runs/open" --tasks open_microwave',
                  f'{runner} scripts/run_handle_contact.py --output "$out_root/runs/close" --tasks close_microwave',
                  f'{runner} scripts/report_handle_contact.py --run "$out_root/runs/open" --output "$out_root/reports/open"',
                  f'{runner} scripts/report_handle_contact.py --run "$out_root/runs/close" --output "$out_root/reports/close"',
                  f'{runner} scripts/report_handle_contact_pair.py --opening-report "$out_root/reports/open" --closing-report "$out_root/reports/close" --output "$out_root/report"']
    (output/'reproduce.sh').write_text('\n'.join(reproduction)+'\n')
    table = ''.join(f'<tr><td>{row["task"].replace("_", " ")}</td>'
                    f'<td>{"Passed" if row["native_success_final"] else "Failed"}</td>'
                    f'<td>{row["native_final_open_angle_deg"]:.3f}°</td>'
                    f'<td>{row["control_steps"]}</td>'
                    f'<td>{row["non_target_positive_force_steps_evaluation_only"]}</td>'
                    f'<td>{html.escape(row.get("failure", "Completed feedback motion"))}</td></tr>' for row in rows)
    prior_rows = ''.join(f'<tr><td>{row["version"]}</td><td>{row["final_open_angle_deg"]:.2f}°</td>'
                         f'<td>{row["native_success"]}</td><td>{html.escape(row.get("failure") or "Completed")}</td></tr>'
                         for row in comparison)
    history_rows = ''.join(f'<tr><td>{row["run"].split("/")[-1]}</td><td>{row["planned"]}</td>'
                           f'<td>{row["completed"]}</td><td>{row["native_successes"]}</td>'
                           f'<td>{row["status"]}</td></tr>' for row in history)
    (output / 'index.html').write_text('<!doctype html><html><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Handle contact repair: resumed experiment</title>'
        '<style>body{font:17px system-ui;color:#182534;max-width:1100px;margin:24px auto;padding:20px;line-height:1.55}'
        'video,img{max-width:100%;display:block;margin:18px 0}video{width:760px}img{width:760px}'
        'table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:9px;text-align:left;border-bottom:1px solid #ccd5df}'
        '.note{background:#edf3f8;padding:15px;border-radius:9px}h2{margin-top:32px}</style>'
        '<h1>Handle contact repair: resumed experiment</h1>'
        f'<p class="note">Final native LIBERO success: <strong>{result["native_successes_final"]}/2</strong> '
        'in one development state per task. Both trials used identical frozen pipeline sources. '
        'Broader evaluation remains.</p>'
        + ('<p class="note"><strong>Goal success includes unintended contact.</strong> '
           'The closing trial contacted the mug. The native contact audit below is separate from task success; '
           'these results do not establish collision-free execution.</p>'
           if any(r['non_target_positive_force_steps_evaluation_only'] for r in rows) else '') +
        '<h2>What changed</h2><p>Continuous RGB-D support extends the initially partial handle detection. '
        'The robot grasps higher on the observed bar to raise its hand above the mug. If the next pull has '
        'no clear posture, it tries bounded wrist turns and tilts, then checks every candidate '
        'and connecting edge with the same collision rules. Intended finger and local hand contact remains allowed; '
        'closing also permits palm bracing on the measured selected panel within the known hand reach. '
        'The local panel band is remeasured from each current depth observation, with a 4 cm bound '
        'and a height gate around the observed handle. '
        'After capture, gripper fingers and palm reject non-target penetration without a positive-gap margin; '
        'arm links retain a 1 mm margin. '
        'The mug and arm links remain obstacles. When a measured contact motion is rejected against the previous '
        'depth map at points associated with the selected moving panel, a synchronized RGB-D observation rechecks '
        'the actual robot before stopping. Unrelated-obstacle rejections do not use this refresh exception. '
        'The fresh check retains the same contact permissions and collision margins.</p>'
        '<p>The hinge starts from the observed upright panel and selected edge. Short contact motions provide '
        'RGB-D handle positions for a circle fit; poorly conditioned short arcs are rejected. '
        'The estimated hinge guides subsequent local motions. No simulator hinge, object pose, segmentation '
        'or assisted attachment enters execution.</p>'
        f'<p>This adaptation fuses {len(rows[0]["sensor_views_used_for_geometry"])} calibrated simulator RGB-D '
        'views, including additional arm views. Camera names are recorded in the measurements.</p>'
        '<table><tr><th>Task</th><th>Final native predicate</th><th>Final opening</th><th>Controls</th>'
        '<th>Non-target contact steps</th><th>Controller stop</th></tr>'
        + table + '</table><p>The controller stop reason and the native task outcome are recorded separately. '
        'A motion can reach the physical closed stop before its requested travel is complete.</p>'
        '<h2>Closing development comparison</h2><table><tr><th>Version</th><th>Final opening</th>'
        '<th>Native success</th><th>Stop reason</th></tr>' + prior_rows + '</table>'
        '<p>These are repeated development trials on the same state. Each algorithm version is retained separately.</p>'
        '<h2>Videos and measured hinge motion</h2><p>Magenta: initial estimated hinge. Cyan: native hinge, '
        'shown only for evaluation. Yellow: commanded TCP. Red: measured TCP. '
        'TCP markers indicate the controller reference point, not fingertip contact.</p>'
        + ''.join(cards) + '<h2>Validation and budget</h2>'
        f'<p>{tests["tests"]} targeted checks passed. Native predicates were replayed from every logged control. '
        'Each closure had verified arrival; each episode stayed within 600 controls. Frozen sources and '
        f'API ledger hashes agree. New API cost: $0; cumulative API spend ${result["api_spend_usd"]:.4f}, '
        f'budget remaining ${result["remaining_api_budget_usd"]:.4f}.</p>'
        f'<p>Evaluation-only native contact audit: '
        f'{sum(r["non_target_positive_force_steps_evaluation_only"] for r in rows)} control steps with '
        'robot contacts above 0.1 N against non-target surfaces. Full pair counts are retained in the measurements.</p>'
        '<p><a href="evaluation.json">Full outcomes and provenance</a> · '
        '<a href="pipeline-source-hashes.json">Frozen pipeline hashes</a> · '
        '<a href="targeted-tests.json">Targeted checks</a> · '
        '<a href="reproduce.sh">Reproduce both trials</a></p>'
        '<h2>All development passes</h2><p>The interrupted filesystem-failure pass retains both planned '
        'episodes as incomplete. These development counts are not pooled into a benchmark success rate.</p>'
        '<table><tr><th>Pass</th><th>Planned</th><th>Completed</th><th>Native successes</th><th>Status</th></tr>'
        + history_rows + '</table></html>')
    print(json.dumps({key: result[key] for key in ['planned', 'completed', 'native_successes_final',
                                                  'identical_frozen_pipeline_sources', 'new_api_calls']}))


if __name__ == '__main__':
    main()
