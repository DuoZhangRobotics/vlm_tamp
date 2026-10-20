"""Matched Luna/Astra high-reasoning drawer report from saved API replies only."""

import argparse
import html
import json
from collections import Counter

import numpy as np
from PIL import Image

from report_task_grounding_scope import (
    _json_details, _links, copy_asset, drawer_cloud_data, local_path,
    points_picture, read_rows, response_context,
)
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_landmark_trial import file_sha
from vlm_tamp.rgbd_numericals import json_text


DEFAULT_CONFIG = "configs/drawer_models_high.json"
DEFAULT_TRIAL = "results/drawer-models-high"
DEFAULT_OUTPUT = "results/drawer-models-high-report"
HISTORICAL_REPORT = "results/task-grounding-frontview-report/summary.json"


def audit_metrics(metrics, scored):
    """Add strict binding diagnostics without changing the frozen base grader."""
    all_points = scored and metrics.get("role_binding_correct", False) and metrics.get(
        "front_body_landmarks_correct", False)
    return {
        "selection_scored": bool(scored),
        "all_five_landmarks_correct": bool(all_points),
        "numerical_binding_correct": bool(all_points and metrics.get("constraint_template_correct", False)
                                          and metrics.get("axis_within_tolerance", False)
                                          and metrics.get("line_constraint_compiled", False)),
        "unbound_axis_within_tolerance": bool(scored and metrics.get("axis_within_tolerance", False)
                                             and not all_points),
        "scope": "Strict offline part binding plus compiled line and axis tolerance; no grasp or execution certification",
    }


def aggregate_model(tasks):
    scored = [task for task in tasks if task["response_context"]["selection_scored"]]
    names = ("middle_drawer_correct", "middle_handle_correct", "role_binding_correct",
             "front_body_landmarks_correct", "constraint_template_correct",
             "line_constraint_compiled", "axis_within_tolerance", "numerical_candidate_correct")
    counts = {name: sum(task["metrics"].get(name) is True for task in scored) for name in names}
    counts.update({name: sum(task["binding_audit"].get(name) is True for task in scored)
                   for name in ("all_five_landmarks_correct", "numerical_binding_correct",
                                "unbound_axis_within_tolerance")})
    return {
        "planned_cases": len(tasks), "scored_selection_cases": len(scored),
        "status_counts": dict(Counter(task["status"] for task in tasks)),
        "response_status_counts": dict(Counter(task["response_context"]["status"] for task in tasks)),
        "unscored_status_counts": dict(Counter(task["status"] for task in tasks
                                               if not task["response_context"]["selection_scored"])),
        "boolean_metric_counts": counts,
        "correct_anchors": sum(task["metrics"].get("correct_anchors", 0) for task in scored),
        "planned_anchor_denominator": len(tasks) * 5,
        "scored_anchor_denominator": len(scored) * 5,
        "api_attempts": sum(task["raw_row"].get("attempts", 0) for task in tasks),
        "estimated_usage_cost_usd": sum(task["raw_row"].get("cost_usd", 0.) for task in tasks),
        "count_denominator_policy": "Report all planned cases and the scored subset; incomplete or unrequested cases are unscored",
    }


def model_key(case):
    return case.get("model_key", case.get("model", "unknown"))


def historical_comparison():
    history = json.loads((ROOT / HISTORICAL_REPORT).read_text())
    tasks = []
    for task in history["drawer_tasks"]:
        if task["condition"] != "direct_rgb":
            continue
        task = dict(task)
        task["binding_audit"] = audit_metrics(task["metrics"], task["response_context"]["selection_scored"])
        tasks.append(task)
    return {
        "source_summary": HISTORICAL_REPORT,
        "source_summary_sha256": file_sha(ROOT / HISTORICAL_REPORT),
        "model": history["model"], "reasoning_effort": history["config"]["reasoning_effort"],
        "max_output_tokens": history["config"]["max_output_tokens"],
        "aggregate": aggregate_model(tasks),
        "tasks": [{"case_id": task["case_id"], "example_id": task["example_id"],
                   "metrics": task["metrics"], "binding_audit": task["binding_audit"],
                   "response_context": task["response_context"]} for task in tasks],
        "comparison_scope": "Historical development reference: reasoning effort and output cap also changed; not a model-only ablation",
    }


def render(summary):
    from report_rgbd_numericals import VIEWER

    esc = html.escape
    config, ledger = summary["config"], summary["ledger"]
    prior = config["prior_usage_cost_usd"]
    total = prior + ledger["estimated_usage_cost_usd"]
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Luna and Astra: high-reasoning drawer grounding</title>',
             '<style>body{margin:0;background:#f2f5fa;color:#182638;font:16px/1.5 system-ui}'
             'main{max-width:1240px;margin:auto;padding:25px}h1{font-size:30px}h2{font-size:23px}'
             '.panel,article{background:white;border:1px solid #d4dde9;border-radius:12px;padding:20px;margin:18px 0}'
             '.note{border-left:5px solid #bd751c;background:#fff9e9;padding:14px}.grid{display:grid;'
             'grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.grid img{width:100%;height:auto}'
             'figure{margin:0}figcaption{font-size:14px}table{border-collapse:collapse;width:100%}'
             'th,td{text-align:left;padding:9px;border-bottom:1px solid #dbe1eb;vertical-align:top}'
             '.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6fa;'
             'padding:12px;font-size:13px}summary{cursor:pointer}a{color:#1465a4}'
             '.pill{display:inline-block;border-radius:20px;background:#e8eef8;padding:3px 9px;margin:2px}'
             'canvas.cloud{width:100%;max-width:700px;height:auto;border-radius:8px;touch-action:none}'
             '</style></head><body><main><h1>Luna and Astra: high-reasoning drawer grounding</h1>',
             '<p>Both models use <strong>high reasoning</strong>, the same four original frontal RGB images, '
             'the same drawer blanks and prompt, and a common output cap of ' + str(config["max_output_tokens"]) +
             ' tokens. No DINO landmarks, crops, depth, inventory or reference coordinates enter these API requests.</p>',
             '<div class="note"><strong>Main criterion:</strong> Select the correct middle drawer and its handle. '
             'The stricter numerical criterion also requires all three selected front-face points on that drawer, '
             'the correct pulling template, a compiled line and an axis within five degrees. '
             'A small axis error with incorrect part binding is an unbound geometric estimate.</div>',
             '<div class="panel"><h2>Matched model comparison</h2><div class="scroll"><table>',
             '<tr><th>Model</th><th>Planned / scored</th><th>Drawer + handle correct</th>'
             '<th>All face points correct</th><th>Strict numerical binding correct</th><th>Forms</th>'
             '<th>Actual calls</th><th>Usage estimate</th></tr>']
    for key, group in summary["model_summaries"].items():
        counts, n, scored = group["boolean_metric_counts"], group["planned_cases"], group["scored_selection_cases"]
        parts.append('<tr><td>' + esc(key) + '</td><td>' + str(n) + ' / ' + str(scored) + '</td><td>' +
                     str(counts["role_binding_correct"]) + '/' + str(n) + '</td><td>' +
                     str(counts["front_body_landmarks_correct"]) + '/' + str(n) + '</td><td>' +
                     str(counts["numerical_binding_correct"]) + '/' + str(n) + '</td><td>' +
                     esc(json.dumps(group["status_counts"])) + '</td><td>' + str(group["api_attempts"]) +
                     '</td><td>$' + format(group["estimated_usage_cost_usd"], '.6f') + '</td></tr>')
    parts += ['</table></div><p>Counts retain four planned cases per model. Unrequested requests, '
              'truncated responses and invalid forms are unscored selections; their status remains visible. '
              'This is a development diagnostic using one cabinet with the same drawer pose and joint axis, '
              'not a general model ranking.</p>',
              '<p>New usage estimate: <strong>$' + format(ledger["estimated_usage_cost_usd"], '.6f') +
              '</strong>. Prior usage included: $' + format(prior, '.6f') + '. Cumulative estimate: <strong>$' +
              format(total, '.6f') + '</strong> of the authorized <strong>$2.00</strong>. '
              'All input is conservatively charged at the larger uncached or cache-write rate. '
              'Output includes reasoning tokens once. Provider invoices determine the final charge.</p>',
              '<p><a href="summary.json">Full report data</a> · <a href="verification.json">Replay verification</a> · '
              '<a href="assets/config.json">Configuration</a> · <a href="assets/protocol.json">Frozen protocol</a> · '
              '<a href="assets/prompt.txt">Exact prompt</a> · <a href="assets/manifest.json">Run manifest</a></p>',
              _json_details('Usage ledger', ledger), _json_details('Stopping condition', summary.get("stopped")), '</div>']
    old = summary["historical_luna_none"]
    counts, n = old["aggregate"]["boolean_metric_counts"], old["aggregate"]["planned_cases"]
    parts += ['<div class="panel"><h2>Earlier Luna with reasoning disabled</h2><p>The preserved direct-RGB '
              'run had ' + str(counts["role_binding_correct"]) + '/' + str(n) + ' correct drawer–handle pairs and ' +
              str(counts["numerical_binding_correct"]) + '/' + str(n) + ' strict numerical bindings. '
              'It used reasoning none and a ' + str(old["max_output_tokens"]) + '-token output cap. '
              'Reasoning effort and output allowance both changed; this historical reference is not a '
              'model-only ablation.</p><p><a href="assets/historical-frontview-summary.json">Preserved historical summary</a></p></div>']
    for task in summary["drawer_tasks"]:
        parts += ['<article class="drawer"><h3>' + esc(task["case_id"]) + '</h3><p>' + esc(task["instruction"]) +
                  '</p><span class="pill">' + esc(task["model"]) + '</span><span class="pill">high reasoning</span>' +
                  '<span class="pill">' + esc(task["status"]) + '</span><div class="grid">']
        for key, caption in (("original_rgb", "Original RGB supplied to the model"),
                             ("selected_points", "Returned source, handle and front-face points"),
                             ("depth", "Measured depth used locally after selection")):
            if key in task["artifacts"]:
                if task["raw_row"].get("attempts", 0) == 0 and key == "original_rgb":
                    caption = "Prepared original RGB; no API request"
                parts.append('<figure><img src="' + esc(task["artifacts"][key]["asset"], quote=True) +
                             '" alt="' + esc(caption, quote=True) + '"><figcaption>' + esc(caption) + '</figcaption></figure>')
        parts.append('</div>')
        if task.get("cloud_data") is not None:
            scene = 'scene-' + task["case_id"]
            parts += ['<p>Visible depth surfaces and the estimated pull line, when available. '
                      'Drag to rotate; scroll to zoom.</p><canvas class="cloud" data-scene="' + esc(scene, quote=True) + '"></canvas>',
                      '<script type="application/json" id="' + esc(scene, quote=True) + '">' +
                      json_text(task["cloud_data"]) + '</script>']
        parts += ['<p>' + _links(task) + '</p><div class="scroll"><table><tr><th>Offline metric</th><th>Result</th></tr>']
        if not task["response_context"]["selection_scored"]:
            parts.append('<tr><td>Selection evaluation</td><td>unscored: ' + esc(task["status"]) + '</td></tr>')
        else:
            for key in ("middle_drawer_correct", "middle_handle_correct", "role_binding_correct",
                        "front_body_landmarks_correct", "constraint_template_correct", "line_constraint_compiled",
                        "axis_error_degrees", "handle_surface_distance_m"):
                value = task["metrics"].get(key)
                shown = 'unresolved' if value is None else str(value)
                parts.append('<tr><td>' + esc(key.replace('_', ' ')) + '</td><td>' + esc(shown) + '</td></tr>')
            for key in ("all_five_landmarks_correct", "numerical_binding_correct", "unbound_axis_within_tolerance"):
                parts.append('<tr><td>' + esc(key.replace('_', ' ')) + '</td><td>' +
                             str(task["binding_audit"][key]) + '</td></tr>')
        parts += ['</table></div>', _json_details('Returned form and normalized pixels', {
            "choices": task["raw_row"].get("choices"), "blanks": task["raw_row"].get("blanks")}),
            _json_details('Response completion and text', task["response_context"]),
            _json_details('Numerical candidate specification', task["raw_row"].get("numericals")),
            _json_details('Frozen base grader metrics', task["metrics"]),
            _json_details('Additional strict binding audit', task["binding_audit"]), '</article>']
    parts += ['<div class="panel"><h2>Scope and provenance</h2><p>The pulling line applies only after handle '
              'attachment, during the pull stage. Approach and retreat do not acquire this line constraint. '
              'Ordinary pick-and-place retains no added task-specific path constraint. No new pick-and-place '
              'requests are made here.</p><p>Simulator part masks, handle geometry and joint axis are used only '
              'for offline scoring. They do not enter the API, replace a selected pixel or repair the geometry. '
              'The compiler assumes the selected front normal approximates the sliding direction. Parallel '
              'drawer faces can explain accurate axes despite incorrect binding; this is a geometric hypothesis, '
              'not evidence that the correct drawer was selected.</p><p>A valid numerical candidate remains an '
              'abstract handle-tracking estimate. Gripper orientation, attachment transform, requested opening '
              'distance, joint-space binding, IK, collision checks and execution remain unresolved. Formula '
              'checks verify residual and Jacobian self-consistency only.</p>',
              _json_details('Provenance', summary["provenance"]), '</div></main><script>' + VIEWER + '</script></body></html>']
    return '\n'.join(parts)


def build_report(config_path=DEFAULT_CONFIG, trial_path=DEFAULT_TRIAL, output_path=DEFAULT_OUTPUT):
    from report_rgbd_numericals import depth_picture
    from vlm_tamp.drawer_model_comparison import check_protocol, grade_drawer, model_config

    config, protocol, records = check_protocol(local_path(config_path))
    trial, output = local_path(trial_path), local_path(output_path)
    if output.resolve() != (ROOT / DEFAULT_OUTPUT).resolve():
        if 'drawer-models-high' not in output.name:
            raise ValueError('Use a separate high-reasoning report directory')
    output.mkdir(parents=True, exist_ok=True)
    assets = output / 'assets'
    assets.mkdir(exist_ok=True)
    manifest = json.loads((trial / 'manifest.json').read_text())
    rows = read_rows(trial / 'predictions.jsonl')
    tasks = []
    for case in config['cases']:
        identifier = case['id']
        row, folder = rows[identifier], trial / identifier
        record = records[case['example_id']]
        metrics = grade_drawer(record, ROOT, case, row)
        context = response_context(folder, row)
        artifacts = {}
        for name, label in (('original.png', 'original_rgb'), ('request.json', 'request'),
                            ('response.json', 'response'), ('payload.json', 'payload'),
                            ('numericals.json', 'numerical_specification')):
            if (folder / name).exists():
                artifacts[label] = copy_asset(folder / name, assets, identifier + '-' + name)
        if 'original_rgb' in artifacts:
            name = identifier + '-points.png'
            points_picture(folder / 'original.png', row.get('blanks'), assets / name)
            artifacts['selected_points'] = {'asset': 'assets/' + name}
        capture = ROOT / config['dataset'] / case['example_id']
        depth = np.load(capture / 'depth_meters.npy', allow_pickle=False)
        name = identifier + '-depth.png'
        depth_picture(depth, assets / name)
        artifacts['depth'] = {'asset': 'assets/' + name}
        for source, label in ((capture / 'calibration.json', 'calibration'),
                              (capture / 'evaluation/reference.json', 'offline_reference')):
            artifacts[label] = copy_asset(source, assets, identifier + '-' + label + '.json')
        image_source = folder / 'original.png' if (folder / 'original.png').exists() else capture / 'rgb.png'
        with Image.open(image_source) as image:
            rgb = np.asarray(image.convert('RGB'))
        cloud_data = drawer_cloud_data(rgb, np.load(capture / 'world_points.npy', allow_pickle=False),
                                       row.get('numericals'))
        task = {**case, 'case_id': identifier, 'model_key': model_key(case),
                'model': model_config(config, case)['model'],
                'instruction': record['model_inputs']['instruction'], 'raw_row': row, 'status': row['status'],
                'metrics': metrics, 'binding_audit': audit_metrics(metrics, context['selection_scored']),
                'response_context': context, 'artifacts': artifacts, 'cloud_data': cloud_data}
        (assets / (identifier + '-result.json')).write_text(json_text(task) + '\n')
        task['artifacts']['result'] = {'asset': 'assets/' + identifier + '-result.json'}
        tasks.append(task)
    for source, name in ((local_path(config_path), 'config.json'),
                          (ROOT / config['frozen_protocol'], 'protocol.json'),
                          (trial / 'manifest.json', 'manifest.json'),
                          (ROOT / HISTORICAL_REPORT, 'historical-frontview-summary.json')):
        copy_asset(source, assets, name)
    (assets / 'prompt.txt').write_text(manifest['prompt_text'])
    summary = {
        'config': config, 'frozen_protocol_sha256': file_sha(ROOT / config['frozen_protocol']),
        'reasoning_effort': 'high', 'planned_drawer_cases': len(config['cases']),
        'drawer_tasks': tasks,
        'model_summaries': {key: aggregate_model([task for task in tasks if task['model_key'] == key])
                            for key in dict.fromkeys(model_key(case) for case in config['cases'])},
        'ledger': manifest['ledger'], 'stopped': manifest.get('stopped'),
        'prompt_text': manifest['prompt_text'], 'historical_luna_none': historical_comparison(),
        'robot_execution_tested': False, 'joint_space_compiled': False, 'model_ranking_established': False,
        'provenance': {
            'matched_four_frontal_rgb_images': True, 'same_prompt_schema_and_geometry': True,
            'common_high_reasoning_and_output_cap': True, 'provided_inventory': False,
            'reference_answers_in_inference': False, 'depth_sent_to_vlm': False,
            'dino_candidate_input_used': False, 'rgb_crop_used': False,
            'drawer_reference_data_used_only_for_evaluation': True,
            'base_grader_changed': False, 'strict_binding_audit_added': True,
            'opening_distance_unresolved': True, 'tcp_attachment_unresolved': True,
            'ik_tested': False, 'grasp_stability_verified': False, 'robot_execution_tested': False,
            'drawer_observations_share_pose_and_joint_axis': True, 'browser_rendering_tested': False,
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
    print(json_text({key: summary[key] for key in ('planned_drawer_cases', 'model_summaries', 'ledger')}))


if __name__ == '__main__':
    main()
