"""Offline paired comparison of three-point and local-depth drawer geometry."""

import argparse
import html
import json
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

from report_task_grounding_scope import (
    _json_details, _links, copy_asset, drawer_cloud_data, local_path, points_picture, read_rows,
)
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_landmark_trial import file_sha
from vlm_tamp.rgbd_numericals import json_text


DEFAULT_CONFIG = "configs/drawer_plane_ablation.json"
DEFAULT_TRIAL = "results/drawer-plane-ablation"
DEFAULT_OUTPUT = "results/drawer-plane-ablation-report"


def support_picture(source, support, blanks, destination):
    """Observed-only overlay: never use held-back drawer part masks."""
    with Image.open(source) as image:
        original = image.convert("RGB")
    base = original.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    colors = (
        ("rejected_candidate_pixel_xy", (250, 126, 25, 170)),
        ("inlier_pixel_xy", (25, 220, 180, 150)),
        ("excluded_handle_pixel_xy", (215, 55, 190, 125)),
        ("depth_edge_pixel_xy", (255, 225, 40, 170)),
    )
    for key, color in colors:
        for point in support.get(key, []):
            x, y = map(int, point)
            if 0 <= x < base.width and 0 <= y < base.height:
                draw.point((x, y), fill=color)
    result = Image.alpha_composite(base, layer).convert("RGB")
    draw = ImageDraw.Draw(result)
    for key, color in (("source_uv", (255, 70, 60)), ("handle_uv", (40, 235, 130))):
        uv = blanks.get(key)
        if uv is not None:
            from vlm_tamp.rgbd_numericals import pixel_index
            x, y = pixel_index(uv, (result.height, result.width))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="black", width=2)
            draw.text((min(x + 8, result.width - 65), max(0, y - 12)), key.removesuffix("_uv"),
                      fill="white", stroke_fill="black", stroke_width=2)
    result.save(destination)


def support_cloud_data(rgb, cloud, numeric, old_numeric):
    """RGB-colored local samples with observed fit diagnostics, no part labels."""
    scene = drawer_cloud_data(rgb, cloud, numeric)
    support = numeric.get("local_plane_support", {})
    points, palette = [], []
    for key, override in (("candidate_pixel_xy", None),
                          ("rejected_candidate_pixel_xy", [250, 126, 25])):
        selected = support.get(key, [])
        step = max(1, len(selected) // 3500)
        for x, y in selected[::step]:
            point = cloud[int(y), int(x)]
            if np.isfinite(point).all():
                points.append(np.round(point, 6).tolist())
                palette.append(rgb[int(y), int(x)].tolist() if override is None else override)
    scene["points"].extend(points)
    scene["colors"].extend(palette)
    for relation in old_numeric.get("relations", []):
        if relation["type"] == "point_on_line":
            origin = np.asarray(relation["line_point_world_m"])
            axis = np.asarray(relation["axis_world"])
            scene["lines"].append({"a": origin.tolist(), "b": (origin + .12 * axis).tolist(),
                                   "color": "#ffaa30"})
    return scene


def aggregate(tasks):
    """Retain all paired development cases, including unresolved geometry."""
    return {
        "paired_cases": len(tasks),
        "status_counts": dict(Counter(task["status"] for task in tasks)),
        "new_api_calls": 0,
        "new_usage_cost_usd": 0,
        "metrics": {
            key: sum(task["metrics"].get(key) is True for task in tasks)
            for key in sorted({key for task in tasks for key, value in task["metrics"].items()
                               if isinstance(value, bool)})
        },
        "old_metrics": {
            key: sum(task["old_metrics"].get(key) is True for task in tasks)
            for key in sorted({key for task in tasks for key, value in task["old_metrics"].items()
                               if isinstance(value, bool)})
        },
        "count_scope": "Matched cached Luna-high selections; four observations of one cabinet pose, not an independent generalization benchmark",
    }


def _value(value):
    if value is None:
        return "unresolved"
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def render(summary):
    from report_rgbd_numericals import VIEWER

    esc = html.escape
    aggregate_data = summary["aggregate"]
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Drawer geometry: local depth versus three selected points</title>',
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
             '</style></head><body><main><h1>Drawer geometry from local depth</h1>',
             '<p>Reuse the exact same four saved <strong>Luna-high</strong> source and handle predictions. '
             'Compare the preserved three-face-point geometry with a robust plane fit around the unchanged '
             'selected drawer pixel. <strong>No new API calls; additional cost $0.</strong></p>',
             '<div class="note"><strong>Two different geometry contracts:</strong> The old strict criterion '
             'requires all five selected landmarks to belong to the middle drawer and handle. The new '
             'criterion uses two selected points plus locally measured surface support. Removing the three '
             'face-point blanks changes the contract; a new estimate is not a repair of the old five-point answer. '
             'Report axis accuracy, part binding, actual support purity and unresolved fits separately.</div>',
             '<div class="panel"><h2>Paired offline results</h2><div class="scroll"><table>',
             '<tr><th>Metric</th><th>Three selected face points</th><th>Local depth plane</th></tr>']
    for key in sorted(set(aggregate_data["metrics"]) | set(aggregate_data["old_metrics"])):
        parts.append('<tr><td>' + esc(key.replace('_', ' ')) + '</td><td>' +
                     esc(_value(aggregate_data['old_metrics'].get(key))) + '</td><td>' +
                     esc(_value(aggregate_data['metrics'].get(key))) + '</td></tr>')
    parts += ['</table></div><p>All counts retain ' + str(aggregate_data['paired_cases']) +
              ' paired development cases. The selected source and handle are unchanged. '
              'Depth fitting uses no held-back part labels, joint axis, object inventory or simulator poses. '
              'Held-back references enter only after observed support and fitting are frozen.</p>',
              '<p>Cumulative prior API usage estimate: <strong>$' +
              format(summary['config']['prior_usage_cost_usd'], '.6f') +
              '</strong> of the authorized <strong>$2.00</strong>. Additional estimate: $0.000000.</p>',
              '<p><a href="summary.json">Full report data</a> · <a href="verification.json">Replay verification</a> · '
              '<a href="assets/config.json">Configuration</a> · <a href="assets/protocol.json">Frozen protocol</a> · '
              '<a href="assets/manifest.json">Offline run manifest</a></p>',
              _json_details('Aggregate', aggregate_data), '</div>']
    for task in summary['tasks']:
        parts += ['<article class="drawer"><h2>' + esc(task['case_id']) + '</h2><p>' +
                  esc(task['instruction']) + '</p><div class="grid">']
        for key, caption in (("original_rgb", "Original RGB from the saved Luna request"),
                             ("old_selected_points", "Preserved five selected landmarks"),
                             ("local_support", "Unchanged source and handle; green plane inliers, orange rejected candidates, magenta handle exclusion, yellow depth edges"),
                             ("depth", "Aligned measured depth used locally")):
            parts.append('<figure><img src="' + esc(task['artifacts'][key]['asset'], quote=True) +
                         '" alt="' + esc(caption, quote=True) + '"><figcaption>' + esc(caption) +
                         '</figcaption></figure>')
        parts += ['</div><p>Observed 3D surfaces and RGB-colored local samples. The green pull line is the '
                  'local-depth estimate; the orange line is the old three-point estimate when available. '
                  'Drag to rotate; scroll to zoom.</p>',
                  '<canvas class="cloud" data-scene="scene-' + esc(task['case_id'], quote=True) +
                  '" aria-label="Interactive measured drawer support and pull lines"></canvas>',
                  '<script type="application/json" id="scene-' + esc(task['case_id'], quote=True) + '">' +
                  json_text(task['cloud_data']) + '</script>',
                  '<div class="scroll"><table><tr><th>Offline metric</th><th>Three-point</th><th>Local-depth</th></tr>']
        for key in sorted(set(task['old_metrics']) | set(task['metrics'])):
            if key == 'anchors' or isinstance(task['metrics'].get(key, task['old_metrics'].get(key)), (list, dict)):
                continue
            parts.append('<tr><td>' + esc(key.replace('_', ' ')) + '</td><td>' +
                         esc(_value(task['old_metrics'].get(key))) + '</td><td>' +
                         esc(_value(task['metrics'].get(key))) + '</td></tr>')
        parts += ['</table></div><h3>Measured support quality</h3><div class="scroll"><table>'
                  '<tr><th>Metric</th><th>Three-point support</th><th>Local plane inliers</th></tr>']
        for key in ('support_sample_count', 'middle_panel_purity', 'middle_handle_fraction',
                    'other_cabinet_samples', 'non_cabinet_samples'):
            parts.append('<tr><td>' + esc(key.replace('_', ' ')) + '</td><td>' +
                         esc(_value(task['support_audit']['three_point'].get(key))) + '</td><td>' +
                         esc(_value(task['support_audit']['local_depth'].get(key))) + '</td></tr>')
        old_plane = task['old_numericals']['drawer_front_plane']
        local_plane = task['numericals']['drawer_front_plane']
        for key in ('status', 'plane_rms_m', 'sample_count', 'reason'):
            parts.append('<tr><td>Plane ' + esc(key.replace('_', ' ')) + '</td><td>' +
                         esc(_value(old_plane.get(key))) + '</td><td>' +
                         esc(_value(local_plane.get(key))) + '</td></tr>')
        parts += ['<tr><td>Observed span in meters</td><td>' +
                  esc(_value(old_plane.get('anchor_pair_distances_m'))) + '</td><td>' +
                  esc(_value(local_plane.get('plane_extent_10_90_m'))) + '</td></tr>',
                  '</table></div><p>Old span lists the three selected anchor pair distances; local span '
                  'lists the fitted plane\'s two central 80% extents. They are different diagnostics. '
                  'Purity counts only visible middle drawer body pixels, excluding the handle; it does '
                  'not certify front-face-only segmentation. The 90% purity threshold is an offline '
                  'diagnostic and never a fit gate.</p><p>' + _links(task) + '</p>',
                  _json_details('Held-back support quality audit; never used to select samples', task['support_audit']),
                  _json_details('Preserved source and handle plus reduced two-point form', task['blanks']),
                  _json_details('Local observed geometry and fit diagnostics', task['numericals']),
                  _json_details('Preserved three-point specification', task['old_numericals']), '</article>']
    parts += ['<div class="panel"><h2>What this comparison establishes</h2><p>Local plane support can '
              'include neighboring coplanar drawer fronts. Low plane error or an accurate axis therefore '
              'does not establish correct part segmentation or handle attachment. Correct drawer and '
              'handle membership is scored independently, and middle-panel support purity remains visible.</p>',
              '<p>The line constraint applies only during pulling. Robot TCP conversion, gripper orientation, '
              'rigid attachment, numerical opening distance, IK, collisions and physical execution remain '
              'unresolved. This reuses four frontal observations with the same cabinet pose and joint axis; '
              'new views and drawer configurations have not been tested.</p>',
              _json_details('Provenance', summary['provenance']), '</div></main><script>' + VIEWER +
              '</script></body></html>']
    return '\n'.join(parts)


def build_report(config_path=DEFAULT_CONFIG, trial_path=DEFAULT_TRIAL, output_path=DEFAULT_OUTPUT):
    from report_rgbd_numericals import depth_picture
    from vlm_tamp.drawer_plane_trial import check_protocol

    config, protocol, records = check_protocol(local_path(config_path))
    trial, output = local_path(trial_path), local_path(output_path)
    if output.resolve() != (ROOT / DEFAULT_OUTPUT).resolve():
        raise ValueError('Use the separate drawer-plane-ablation report directory')
    output.mkdir(parents=True, exist_ok=True)
    assets = output / 'assets'
    assets.mkdir(exist_ok=True)
    manifest = json.loads((trial / 'manifest.json').read_text())
    rows = read_rows(trial / 'predictions.jsonl')
    tasks = []
    for case in config['cases']:
        identifier = case['id']
        row, record = rows[identifier], records[case['example_id']]
        capture = ROOT / config['dataset'] / case['example_id']
        rgb_source = capture / 'rgb.png'
        old = row['old_numericals']
        numeric = row['numericals']
        artifacts = {'original_rgb': copy_asset(rgb_source, assets, identifier + '-rgb.png')}
        old_selected_name = identifier + '-old-points.png'
        points_picture(rgb_source, row['source_blanks'], assets / old_selected_name)
        artifacts['old_selected_points'] = {'asset': 'assets/' + old_selected_name}
        support_name = identifier + '-support.png'
        support_picture(rgb_source, numeric.get('local_plane_support', {}), row['blanks'], assets / support_name)
        artifacts['local_support'] = {'asset': 'assets/' + support_name}
        depth = np.load(capture / 'depth_meters.npy', allow_pickle=False)
        depth_name = identifier + '-depth.png'
        depth_picture(depth, assets / depth_name)
        artifacts['depth'] = {'asset': 'assets/' + depth_name}
        for name, label in (("numericals.json", "numerical_specification"),
                            ("result.json", "offline_result")):
            if (trial / identifier / name).exists():
                artifacts[label] = copy_asset(trial / identifier / name, assets, identifier + '-offline-' + name)
        for source, label in ((capture / 'calibration.json', 'calibration'),
                              (capture / 'evaluation/reference.json', 'offline_reference')):
            artifacts[label] = copy_asset(source, assets, identifier + '-' + label + '.json')
        source_folder = ROOT / config['source_trial'] / row['source_case_id']
        for name, label in (("request.json", "saved_request"), ("response.json", "saved_response")):
            artifacts[label] = copy_asset(source_folder / name, assets, identifier + '-' + name)
        with Image.open(rgb_source) as image:
            rgb = np.asarray(image.convert('RGB'))
        cloud = np.load(capture / 'world_points.npy', allow_pickle=False)
        task = {**case, 'case_id': identifier, 'instruction': record['model_inputs']['instruction'],
                'raw_row': row, 'status': row['status'], 'old_metrics': row['old_metrics'],
                'metrics': row['metrics'], 'support_audit': row['support_audit'], 'blanks': row['blanks'],
                'old_numericals': old, 'numericals': numeric, 'artifacts': artifacts,
                'cloud_data': support_cloud_data(rgb, cloud, numeric, old)}
        (assets / (identifier + '-result.json')).write_text(json_text(task) + '\n')
        task['artifacts']['result'] = {'asset': 'assets/' + identifier + '-result.json'}
        tasks.append(task)
    for source, name in ((local_path(config_path), 'config.json'),
                          (ROOT / config['frozen_protocol'], 'protocol.json'),
                          (trial / 'manifest.json', 'manifest.json')):
        copy_asset(source, assets, name)
    summary = {
        'config': config, 'frozen_protocol_sha256': file_sha(ROOT / config['frozen_protocol']),
        'tasks': tasks, 'aggregate': aggregate(tasks), 'manifest': manifest,
        'new_api_calls': 0, 'new_usage_cost_usd': 0,
        'planner_ready': False, 'robot_execution_tested': False, 'new_views_tested': False,
        'provenance': {
            'saved_luna_high_source_and_handle_unchanged': True, 'new_api_calls': 0,
            'same_four_frontal_observations': True, 'held_back_labels_used_in_fit': False,
            'support_pixels_frozen_before_reference_scoring': True,
            'different_old_five_point_and_new_two_point_geometry_contracts': True,
            'old_five_anchor_results_repaired': False, 'coplanar_panel_semantics_verified': False,
            'geometry_axis_hypothesis': 'Camera-facing visible front normal approximates slider opening direction',
            'reference_normal_or_joint_axis_used_in_fit': False,
            'new_views_tested': False, 'robot_execution_tested': False, 'browser_rendering_tested': False,
            'preserved_artifacts': len(protocol.get('preserved_artifacts', {})),
        },
    }
    (output / 'summary.json').write_text(json_text(summary) + '\n')
    (output / 'index.html').write_text(render(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--trial', default=DEFAULT_TRIAL)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build_report(args.config, args.trial, args.output)
    print(json_text(summary['aggregate']))


if __name__ == '__main__':
    main()
