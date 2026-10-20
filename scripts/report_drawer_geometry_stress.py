"""Separate report for four actual RGB view and drawer-geometry interventions."""

import argparse
import html
import json
from collections import Counter

import numpy as np
from PIL import Image

from report_drawer_plane_ablation import support_cloud_data, support_picture
from report_task_grounding_scope import (
    _json_details, _links, copy_asset, local_path, points_picture, read_rows, response_context,
)
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_landmark_trial import file_sha
from vlm_tamp.rgbd_numericals import json_text


DEFAULT_CONFIG = "configs/drawer_geometry_stress.json"
DEFAULT_TRIAL = "results/drawer-geometry-stress-luna"
DEFAULT_OUTPUT = "results/drawer-geometry-stress-report"
BASELINE_REPORT = "results/drawer-plane-ablation-report/summary.json"


def aggregate(tasks):
    scored = [task for task in tasks if task['response_context']['selection_scored']]
    return {
        'planned_cases': len(tasks), 'scored_selection_cases': len(scored),
        'status_counts': dict(Counter(task['status'] for task in tasks)),
        'response_status_counts': dict(Counter(task['response_context']['status'] for task in tasks)),
        'unscored_status_counts': dict(Counter(task['status'] for task in tasks
                                              if not task['response_context']['selection_scored'])),
        'geometry_status_counts': dict(Counter(
            (task['raw_row'].get('numericals') or {}).get('drawer_front_plane', {}).get('status', 'unavailable')
            for task in tasks)),
        'scored_geometry_unresolved_cases': sum(
            (task['raw_row'].get('numericals') or {}).get('drawer_front_plane', {}).get('status') == 'unresolved'
            for task in scored),
        'boolean_metric_counts': {
            key: sum(task['metrics'].get(key) is True for task in scored)
            for key in sorted({key for task in tasks for key, value in task['metrics'].items()
                               if isinstance(value, bool)})
        },
        'api_attempts': sum(task['raw_row'].get('attempts', 0) for task in tasks),
        'estimated_usage_cost_usd': sum(task['raw_row'].get('cost_usd', 0) for task in tasks),
        'denominator_policy': 'All four planned cases remain visible. Completed valid forms are scored even if local geometry is unresolved; incomplete and invalid forms are unscored selections.',
    }


def _value(value):
    if value is None:
        return 'unresolved'
    if isinstance(value, float):
        return format(value, '.6g')
    return str(value)


def render(summary):
    from report_rgbd_numericals import VIEWER

    esc = html.escape
    cfg, ledger, counts = summary['config'], summary['ledger'], summary['aggregate']
    score_counts = counts['boolean_metric_counts']
    baseline = summary['baseline_summary']
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Drawer grounding across changed views and cabinet poses</title>',
             '<style>body{margin:0;background:#f2f5fa;color:#182638;font:16px/1.5 system-ui}'
             'main{max-width:1240px;margin:auto;padding:25px}h1{font-size:29px}h2{font-size:23px}'
             '.panel,article{background:white;border:1px solid #d4dde9;border-radius:12px;padding:20px;margin:18px 0}'
             '.note{border-left:5px solid #bd751c;background:#fff9e9;padding:14px}.grid{display:grid;'
             'grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}.grid img{width:100%;height:auto}'
             'figure{margin:0}figcaption{font-size:14px}table{border-collapse:collapse;width:100%}'
             'th,td{text-align:left;padding:9px;border-bottom:1px solid #dbe1eb;vertical-align:top}'
             '.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6fa;'
             'padding:12px;font-size:13px}summary{cursor:pointer}a{color:#1465a4}'
             'canvas.cloud{width:100%;max-width:800px;height:auto;border-radius:8px;touch-action:none}'
             '</style></head><body><main><h1>Drawer grounding across changed views</h1>',
             '<p>Four new <strong>Luna-high</strong> requests use only the original full RGB image and '
             'five blanks: source, handle, joint type, motion and pulling constraint. There are no '
             'model-selected face points. The same fixed local-depth plane policy runs after each answer.</p>',
             '<div class="note"><strong>Separate development stress test:</strong> Camera changes and '
             'cabinet yaw/partial-opening interventions are deliberately constructed simulator captures. '
             'They are not an official LIBERO benchmark or physical robot trials. Correct drawer/handle '
             'selection, geometry availability, axis accuracy and middle-body support purity remain separate checks. '
             'A valid JSON answer can still leave the geometry unresolved.</div>',
             '<div class="panel"><h2>Results across all four planned cases</h2><div class="scroll"><table>',
             '<tr><th>Metric</th><th>Count</th></tr>']
    for key in ('role_binding_correct', 'front_plane_estimated', 'line_constraint_compiled',
                'axis_within_tolerance', 'two_point_geometry_criterion',
                'support_purity_above_diagnostic_threshold', 'two_point_geometry_and_reference_pure_support'):
        parts.append('<tr><td>' + esc(key.replace('_', ' ')) + '</td><td>' +
                     str(score_counts.get(key, 0)) + '/' + str(counts['planned_cases']) + '</td></tr>')
    parts += ['</table></div><p>Scored completed selections: ' + str(counts['scored_selection_cases']) +
              '/' + str(counts['planned_cases']) + '. Geometry statuses: ' + esc(json.dumps(counts['geometry_status_counts'])) +
              '. Valid selections with unresolved geometry: ' + str(counts['scored_geometry_unresolved_cases']) +
              '. Incomplete or unrequested cases remain unscored and keep their status.</p>',
              '<p>Actual calls: ' + str(ledger['api_attempts']) + '; new usage estimate: <strong>$' +
              format(ledger['estimated_usage_cost_usd'], '.6f') + '</strong>. Cumulative estimate: <strong>$' +
              format(cfg['prior_usage_cost_usd'] + ledger['estimated_usage_cost_usd'], '.6f') +
              '</strong> of the authorized <strong>$2.00</strong>. Input is charged conservatively at '
              'the larger uncached/cache-write rate; reasoning is already included in output tokens once.</p>',
              '<p><a href="summary.json">Full report data</a> · <a href="verification.json">Replay verification</a> · '
              '<a href="assets/config.json">Configuration</a> · <a href="assets/protocol.json">Frozen protocol</a> · '
              '<a href="assets/prompt.txt">Exact prompt</a> · <a href="assets/manifest.json">Run manifest</a></p>',
              _json_details('Usage ledger and stopping status', {'ledger': ledger, 'stopped': summary.get('stopped')}),
              _json_details('Aggregate metrics', counts), '</div>',
              '<div class="panel"><h2>Preserved frontal-view reference</h2><p>The earlier offline comparison '
              'reused four Luna-high answers with correct drawer and handle binding in all four cases. '
              'Its local-plane axes were within five degrees in all four, but only about 51–54% of the '
              'inliers belonged to the middle drawer body. This stress test changes the images and poses, '
              'so it is not a matched geometry ablation or proof of general robustness.</p>',
              '<p><a href="assets/baseline-summary.json">Preserved offline baseline data</a></p>',
              _json_details('Baseline aggregate', baseline['aggregate']), '</div>']
    for task in summary['tasks']:
        numeric = task['raw_row'].get('numericals') or {}
        front = numeric.get('drawer_front_plane', {})
        parts += ['<article class="drawer"><h2>' + esc(task['case_id']) + '</h2><p>' +
                  esc(task['instruction']) + '</p><p>Saved answer status: <strong>' + esc(task['status']) +
                  '</strong>. Local geometry: <strong>' + esc(front.get('status', 'unavailable')) +
                  '</strong>.</p>', _json_details('Capture intervention', task['capture_intervention']),
                  '<div class="grid">']
        for key, caption in (("original_rgb", "Actual RGB supplied to Luna"),
                             ("selected_points", "Returned source and handle pixels"),
                             ("local_support", "Observed green inliers, orange rejected candidates, magenta handle exclusion, yellow depth edges"),
                             ("depth", "Measured depth supplied only to the local geometry estimator")):
            parts.append('<figure><img src="' + esc(task['artifacts'][key]['asset'], quote=True) +
                         '" alt="' + esc(caption, quote=True) + '"><figcaption>' + esc(caption) +
                         '</figcaption></figure>')
        parts += ['</div><p>Observed 3D surfaces and local RGB-colored support. Green is the candidate '
                  'pull line when available. Drag to rotate; scroll to zoom.</p>',
                  '<canvas class="cloud" data-scene="scene-' + esc(task['case_id'], quote=True) +
                  '" aria-label="Interactive measured drawer support and candidate pull line"></canvas>',
                  '<script type="application/json" id="scene-' + esc(task['case_id'], quote=True) + '">' +
                  json_text(task['cloud_data']) + '</script>',
                  '<div class="scroll"><table><tr><th>Offline metric</th><th>Result</th></tr>']
        if not task['response_context']['selection_scored']:
            parts.append('<tr><td>Selection scoring</td><td>unscored: ' + esc(task['status']) + '</td></tr>')
        else:
            for key in ('middle_drawer_correct', 'middle_handle_correct', 'role_binding_correct',
                        'constraint_template_correct', 'line_constraint_compiled', 'axis_error_degrees',
                        'axis_within_tolerance', 'two_point_geometry_criterion', 'middle_panel_support_purity',
                        'two_point_geometry_and_reference_pure_support'):
                parts.append('<tr><td>' + esc(key.replace('_', ' ')) + '</td><td>' +
                             esc(_value(task['metrics'].get(key))) + '</td></tr>')
        for key in ('status', 'reason', 'plane_rms_m', 'sample_count', 'plane_extent_10_90_m'):
            parts.append('<tr><td>Plane ' + esc(key.replace('_', ' ')) + '</td><td>' +
                         esc(_value(front.get(key))) + '</td></tr>')
        parts += ['</table></div><p>' + _links(task) + '</p>',
                  _json_details('Returned form and normalized pixels', {
                      'choices': task['raw_row'].get('choices'), 'blanks': task['raw_row'].get('blanks')}),
                  _json_details('Response completion and text', task['response_context']),
                  _json_details('Held-back support quality; never used to fit or tune', task['support_audit']),
                  _json_details('Local numerical candidate', task['raw_row'].get('numericals')), '</article>']
    parts += ['<div class="panel"><h2>Scope and provenance</h2><p>Reference part masks, actual drawer '
              'joint axes, cabinet poses and joint values are held back for capture auditing and evaluation. '
              'They do not enter the API payload or local geometry. The fixed policy is not retuned from '
              'these results. Coplanar neighboring drawer fronts can still contaminate an accurate plane.</p>',
              '<p>Purity measures visible middle drawer body, excluding the handle; it does not certify '
              'front-face-only segmentation. The 90% threshold is an offline diagnostic and never a fit gate. '
              'The pulling line is only an abstract handle-tracking estimate. Gripper orientation, TCP '
              'attachment, opening distance, joint-space binding, IK, collisions and execution remain unresolved.</p>',
              _json_details('Provenance', summary['provenance']), '</div></main><script>' + VIEWER +
              '</script></body></html>']
    return '\n'.join(parts)


def build_report(config_path=DEFAULT_CONFIG, trial_path=DEFAULT_TRIAL, output_path=DEFAULT_OUTPUT):
    from report_rgbd_numericals import depth_picture
    from vlm_tamp.drawer_geometry_stress import check_protocol

    cfg, protocol, records = check_protocol(local_path(config_path))
    trial, report = local_path(trial_path), local_path(output_path)
    if report.resolve() != (ROOT / DEFAULT_OUTPUT).resolve():
        raise ValueError('Use the separate drawer-geometry-stress report directory')
    report.mkdir(parents=True, exist_ok=True)
    assets = report / 'assets'
    assets.mkdir(exist_ok=True)
    manifest = json.loads((trial / 'manifest.json').read_text())
    rows = read_rows(trial / 'predictions.jsonl')
    tasks = []
    for case in cfg['cases']:
        identifier = case['id']
        row, record = rows[identifier], records[case['example_id']]
        folder = trial / identifier
        capture = ROOT / cfg['dataset'] / case['example_id']
        rgb_source = capture / 'rgb.png'
        numeric, blanks = row.get('numericals') or {}, row.get('blanks') or {}
        artifacts = {'original_rgb': copy_asset(rgb_source, assets, identifier + '-rgb.png')}
        for name, label in (("request.json", "request"), ("response.json", "response"),
                            ("payload.json", "payload"), ("numericals.json", "numerical_specification")):
            if (folder / name).exists():
                artifacts[label] = copy_asset(folder / name, assets, identifier + '-' + name)
        selected_name = identifier + '-points.png'
        points_picture(rgb_source, blanks, assets / selected_name)
        artifacts['selected_points'] = {'asset': 'assets/' + selected_name}
        support_name = identifier + '-support.png'
        support_picture(rgb_source, numeric.get('local_plane_support', {}), blanks, assets / support_name)
        artifacts['local_support'] = {'asset': 'assets/' + support_name}
        depth = np.load(capture / 'depth_meters.npy', allow_pickle=False)
        depth_name = identifier + '-depth.png'
        depth_picture(depth, assets / depth_name)
        artifacts['depth'] = {'asset': 'assets/' + depth_name}
        for source, label in ((capture / 'calibration.json', 'calibration'),
                              (capture / 'evaluation/reference.json', 'offline_reference'),
                              (capture / 'intervention.json', 'capture_intervention')):
            artifacts[label] = copy_asset(source, assets, identifier + '-' + label + '.json')
        with Image.open(rgb_source) as image:
            rgb = np.asarray(image.convert('RGB'))
        cloud = np.load(capture / 'world_points.npy', allow_pickle=False)
        task = {**case, 'case_id': identifier, 'instruction': record['model_inputs']['instruction'],
                'raw_row': row, 'status': row['status'], 'metrics': row.get('metrics') or {},
                'support_audit': row.get('support_audit') or {}, 'artifacts': artifacts,
                'capture_intervention': json.loads((capture / 'intervention.json').read_text()),
                'response_context': response_context(folder, row),
                'cloud_data': support_cloud_data(rgb, cloud, numeric, {})}
        (assets / (identifier + '-result.json')).write_text(json_text(task) + '\n')
        task['artifacts']['result'] = {'asset': 'assets/' + identifier + '-result.json'}
        tasks.append(task)
    for source, name in ((local_path(config_path), 'config.json'),
                          (ROOT / cfg['frozen_protocol'], 'protocol.json'),
                          (trial / 'manifest.json', 'manifest.json'),
                          (ROOT / BASELINE_REPORT, 'baseline-summary.json')):
        copy_asset(source, assets, name)
    (assets / 'prompt.txt').write_text(manifest['prompt_text'])
    summary = {
        'config': cfg, 'frozen_protocol_sha256': file_sha(ROOT / cfg['frozen_protocol']),
        'tasks': tasks, 'aggregate': aggregate(tasks), 'ledger': manifest['ledger'],
        'stopped': manifest.get('stopped'), 'prompt_text': manifest['prompt_text'],
        'baseline_summary': json.loads((ROOT / BASELINE_REPORT).read_text()),
        'planner_ready': False, 'robot_execution_tested': False, 'official_benchmark': False,
        'provenance': {
            'real_api_requests': True, 'model': 'gpt-6-luna', 'reasoning_effort': 'high',
            'full_rgb_only_sent_to_vlm': True, 'depth_sent_to_vlm': False,
            'provided_inventory': False, 'reference_answers_sent_to_vlm': False,
            'dino_candidate_input_used': False, 'rgb_crop_used': False,
            'face_point_blanks_used': False, 'same_frozen_local_depth_policy': True,
            'camera_and_pose_interventions': True, 'official_benchmark': False,
            'held_back_part_labels_or_joint_axis_used_in_fit': False,
            'support_pixels_determined_before_reference_scoring': True,
            'coplanar_drawer_identity_resolved': False, 'planner_execution_tested': False,
            'browser_rendering_tested': False,
            'preserved_artifacts': len(protocol.get('preserved_artifacts', {})),
        },
    }
    (report / 'summary.json').write_text(json_text(summary) + '\n')
    (report / 'index.html').write_text(render(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--trial', default=DEFAULT_TRIAL)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build_report(args.config, args.trial, args.output)
    print(json_text({'aggregate': summary['aggregate'], 'ledger': summary['ledger']}))


if __name__ == '__main__':
    main()
