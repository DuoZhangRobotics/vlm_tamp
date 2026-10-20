"""Inspectable matched DINOv2/v3 landmark-selection report; no inference calls."""

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from report_rgbd_numericals import LABELS, VIEWER, cloud_data, depth_picture, overlay
from vlm_tamp.dataset import read_records
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_candidate_evaluation import grade_choices
from vlm_tamp.rgbd_landmark_trial import candidate_folder, check_protocol, declared_conditions
from vlm_tamp.rgbd_numeric_evaluation import capture_checks
from vlm_tamp.rgbd_numericals import json_text


DEFAULT_CONFIG = 'configs/rgbd_dino_comparison.json'
DEFAULT_TRIAL = 'results/rgbd-dino-comparison-mini'
DEFAULT_OUTPUT = 'results/rgbd-dino-comparison-report'
DIRECT_BASELINE = ROOT / 'results/rgbd-numerical-report/summary.json'


def local_path(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_rows(trial):
    path = trial / 'predictions.jsonl'
    rows = list(map(json.loads, path.read_text().splitlines())) if path.exists() else []
    by_id = {row['case_id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate saved case IDs')
    return by_id


def read_response_context(trial, identifier, row):
    path = trial / identifier / 'response.json'
    if not path.exists():
        return {'status': 'not_run', 'incomplete_details': None, 'output_text': None,
                'raw_choice_context': 'not_requested', 'selection_diagnostic_applicable': False,
                'unscored_selection_reason': 'not_run'}
    raw = json.loads(path.read_text())
    text = ''.join(item['text'] for block in raw.get('output', []) if block['type'] == 'message'
                   for item in block['content'] if item['type'] == 'output_text')
    details = raw.get('incomplete_details')
    reason = details.get('reason') if isinstance(details, dict) else None
    applicable = raw['status'] == 'completed' and row['status'] == 'valid'
    unscored = ('response_truncated' if raw['status'] == 'incomplete' and reason == 'max_output_tokens'
                else 'response_incomplete' if raw['status'] == 'incomplete' else 'invalid_response')
    return {'status': raw['status'], 'incomplete_details': details, 'output_text': text,
            'raw_choice_context': 'complete_accepted_form' if applicable else
            'partial_text_unusable' if text else 'no_output_text',
            'selection_diagnostic_applicable': applicable,
            'unscored_selection_reason': None if applicable else unscored}


def aggregate(tasks):
    diagnostics = Counter(anchor['diagnostic'] for task in tasks for anchor in task['metrics']['anchors'])
    selection = Counter(row['diagnostic'] for task in tasks
                        if task['response_context']['selection_diagnostic_applicable']
                        for row in task['metrics']['selection_diagnostics'])
    unscored = Counter(task['response_context']['unscored_selection_reason'] for task in tasks
                       if not task['response_context']['selection_diagnostic_applicable'])
    completion = Counter(task['response_context']['status'] for task in tasks)
    incomplete_reasons = Counter((task['response_context']['incomplete_details'] or {}).get('reason', 'unspecified')
                                 for task in tasks if task['response_context']['status'] == 'incomplete')
    checks = [check for task in tasks for check in task['metrics']['relation_checks']]
    return {
        'cases': len(tasks), 'anchor_denominator': 7*len(tasks),
        'valid_forms': sum(task['metrics']['choices_valid'] for task in tasks),
        'compiled_cases': sum(task['status'] == 'valid' for task in tasks),
        'status_counts': dict(Counter(task['status'] for task in tasks)),
        'response_completion_counts': dict(completion), 'response_incomplete_reason_counts': dict(incomplete_reasons),
        'completed_responses': completion['completed'], 'incomplete_responses': completion['incomplete'],
        'unscored_selection_cases': dict(unscored),
        'unscored_selection_anchor_counts': {key: count*7 for key, count in unscored.items()},
        'semantic_complete_goals': sum(task['metrics']['semantic_complete_goal'] for task in tasks),
        'anchor_diagnostics': dict(diagnostics), 'selection_diagnostics': dict(selection),
        'correct_anchors': diagnostics['correct_instance'],
        'depth_lifted_landmarks': sum(anchor['depth_status'] == 'measured'
                                     for task in tasks for anchor in task['metrics']['anchors']),
        'measured_landmarks_on_correct_instance': sum(anchor['depth_status'] == 'measured'
                                                     and anchor['diagnostic'] == 'correct_instance'
                                                     for task in tasks for anchor in task['metrics']['anchors']),
        'grasp_candidates': sum(task['metrics']['grasp_candidate'] for task in tasks),
        'correctly_bound_grasp_candidates': sum(task['metrics']['candidate_landmarks_on_correct_object']
                                               for task in tasks),
        'correct_path_choices': sum(task['metrics']['path_choice_correct'] for task in tasks),
        'relation_self_checks': len(checks), 'passing_relation_self_checks': sum(check['passed'] for check in checks),
        'maximum_jacobian_error': max((check['maximum_jacobian_error'] for check in checks), default=None),
        'api_attempts': sum(task['raw_row'].get('attempts', 0) for task in tasks),
        'estimated_usage_cost_usd': sum(task['raw_row'].get('cost_usd', 0.) for task in tasks),
    }


def build_report(config_path=DEFAULT_CONFIG, trial_path=DEFAULT_TRIAL, output_path=DEFAULT_OUTPUT):
    config, protocol, records = check_protocol(local_path(config_path))
    _, source = read_records(ROOT / config['dataset'])
    trial, output = local_path(trial_path), local_path(output_path)
    manifest = json.loads((trial / 'manifest.json').read_text())
    saved = read_rows(trial)
    output.mkdir(parents=True, exist_ok=True)
    assets = output / 'assets'
    assets.mkdir(exist_ok=True)
    tasks, viewers, captures, metadata, coverage = [], {}, {}, {}, {}
    for case in config['cases']:
        record = records[case['example_id']]
        folder = candidate_folder(config, case)
        candidates = json.loads((folder / 'candidates.json').read_text())
        info = json.loads((folder / 'metadata.json').read_text())
        row = saved.get(case['id'], {
            'case_id': case['id'], 'status': 'not_run', 'choices': None, 'blanks': None, 'numericals': None,
            'attempts': 0, 'cost_usd': 0., 'reason': 'No saved result; retained in configured denominator',
        })
        response_context = read_response_context(trial, case['id'], row)
        metric = grade_choices(record, source, case, row.get('choices'), candidates, row.get('numericals'))
        key = case['condition']+'/'+case['example_id']
        if metric['proposal_coverage'] != protocol['offline_proposal_coverage'][key]:
            raise ValueError('Frozen offline proposal coverage changed: '+key)
        coverage[key], metadata[key] = metric['proposal_coverage'], info
        capture = ROOT / config['depth_dataset'] / case['example_id']
        depth, cloud = [np.load(capture / name, allow_pickle=False)
                        for name in ('depth_meters.npy', 'world_points.npy')]
        calibration = json.loads((capture / 'calibration.json').read_text())
        captures[case['example_id']] = capture_checks(depth, cloud, calibration)
        identifier = case['id']
        for name, suffix in (('original.png', '-original.png'), ('numbered.png', '-numbered.png'),
                             ('candidates.json', '-candidates.json'), ('metadata.json', '-metadata.json')):
            (assets / (identifier+suffix)).write_bytes((folder / name).read_bytes())
        overlay(folder / 'original.png', metric['resolved_blanks'], metric, assets / (identifier+'-points.png'))
        depth_picture(depth, assets / (identifier+'-depth.png'))
        with Image.open(folder / 'original.png') as image:
            rgb = np.asarray(image.convert('RGB'))
        viewers[identifier] = cloud_data(rgb, cloud, row.get('numericals') if metric['choices_valid'] else None)
        frozen_request = next(item for item in protocol['requests'] if item['case_id'] == identifier)
        task = {**case, 'instruction': frozen_request['payload']['instruction'], 'status': row['status'],
                'choices': row.get('choices'), 'blanks': row.get('blanks'), 'resolved_blanks': metric['resolved_blanks'],
                'numericals': row.get('numericals'), 'metrics': metric, 'candidate_metadata': info, 'raw_row': row,
                'response_context': response_context,
                'visual_selection_diagnostics': metric['selection_diagnostics'] if
                response_context['selection_diagnostic_applicable'] else None}
        (assets / (identifier+'-result.json')).write_text(json_text(task)+'\n')
        for name in ('request.json', 'response.json', 'payload.json'):
            path = trial / identifier / name
            if path.exists():
                (assets / (identifier+'-'+name)).write_bytes(path.read_bytes())
        tasks.append(task)
    baseline = json.loads(DIRECT_BASELINE.read_text())
    conditions = {condition: aggregate([task for task in tasks if task['condition'] == condition])
                  for condition in declared_conditions(config)}
    baseline_counts = {key: baseline[key] for key in (
        'cases', 'valid_forms', 'semantic_complete_goals', 'anchor_diagnostics', 'depth_lifted_landmarks',
        'measured_landmarks_on_correct_instance', 'grasp_candidates', 'correctly_bound_grasp_candidates',
        'correct_path_choices', 'relation_self_checks', 'passing_relation_self_checks', 'ledger',
    )}
    paired = []
    for case in config['cases']:
        if case['condition'] != config['conditions'][0]:
            continue
        matching = [task for task in tasks if task['base_case_id'] == case['base_case_id']]
        outcomes = {task['condition']: task['metrics']['semantic_complete_goal'] for task in matching}
        paired.append({'base_case_id': case['base_case_id'], 'outcomes': outcomes})
    summary = {
        'model': config['model'], 'planned_cases': len(config['cases']), 'cases': len(tasks),
        'unique_scenes': len(captures), 'condition_summaries': conditions,
        'baseline_direct_xy': baseline_counts, 'baseline_source': str(DIRECT_BASELINE),
        'paired_goal_outcomes': paired, 'ledger': manifest['ledger'], 'capture_checks': captures,
        'stopped': manifest.get('stopped'),
        'incomplete_comparison': any(task['response_context']['status'] != 'completed'
                                     or task['status'] != 'valid' for task in tasks),
        'encoder_ranking_established': False,
        'proposal_coverage_before_selection': coverage, 'candidate_metadata': metadata, 'tasks': tasks,
        'prompt_text': manifest['prompt_text'], 'config': config,
        'frozen_protocol_sha256': manifest['frozen_protocol_sha256'],
        'comparison_scope': 'four development scenes; six cases per encoder; no population model ranking',
        'direct_xy_comparison_changes': ['candidate proposer', 'second RGB image with numbered points', 'ID selection form'],
        'encoder_pair_controls': ['same RGB/depth', 'same RGB-predicted masks', 'same proposal settings',
                                  'same prompt', 'same Mini model', 'same numerical compiler and grader'],
        'whole_object_coverage_only': True, 'part_coverage_verified': False, 'contact_coverage_verified': False,
        'axis_coverage_verified': False, 'grasp_stability_verified': False, 'ik_tested': False,
        'robot_execution_tested': False, 'joint_space_compiled': False, 'reference_answers_in_inference': False,
        'preserved_artifacts': len(protocol['preserved_artifacts']),
    }
    (output / 'summary.json').write_text(json_text(summary)+'\n')
    for path, name in ((local_path(config_path), 'config.json'),
                       (ROOT / config['frozen_protocol'], 'protocol.json'), (trial / 'manifest.json', 'manifest.json')):
        (assets / name).write_bytes(path.read_bytes())
    (assets / 'prompt.txt').write_text(manifest['prompt_text'])
    (assets / 'direct-xy-baseline.json').write_bytes(DIRECT_BASELINE.read_bytes())
    esc = html.escape
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>DINO landmark comparison</title>',
        '<style>body{font:16px system-ui;line-height:1.55;max-width:1500px;margin:auto;padding:24px;background:#111826;color:#edf2ff}.panel,article{background:#1b2638;padding:22px;margin:22px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:18px}img,canvas{width:100%;height:auto}canvas{touch-action:none}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #405169;text-align:left}a{color:#79d5ff}.scroll{overflow:auto}h2{font-size:22px}</style></head><body>',
        '<h1>Numbered landmarks with depth: DINOv2 versus DINOv3</h1>',
        '<p>Mini fills landmark-ID and method blanks. Local RGB features, RGB-predicted MobileSAM regions, and calibrated depth supply fixed candidates. The same geometry compiler supplies numerical estimates and equations.</p>',
        '<div class="panel"><strong>Matched inputs, incomplete comparison:</strong> DINOv2 and DINOv3 use the same four scenes, six instructions each, masks, proposal settings, prompt, Mini snapshot, and grader. All planned cases count, including missing, not-run and invalid replies. Output-token truncation and budget admission prevented complete answers for every case. The scores include those operational failures; they do not establish an encoder winner or population ranking.</div>',
        '<div class="scroll"><table><tr><th>Condition</th><th>Valid forms / planned</th><th>Complete goals / planned</th><th>Correct anchors</th><th>Depth accepted / correct</th><th>Grasp proposals / correctly bound</th><th>Path choices</th><th>Formula checks</th></tr>',
    ]
    comparison_rows = [('Direct XY, historical', baseline_counts)] + list(conditions.items())
    for condition, counts in comparison_rows:
        correct = counts.get('correct_anchors', counts.get('anchor_diagnostics', {}).get('correct_instance', 0))
        parts.append(f'<tr><td>{esc(condition)}</td><td>{counts["valid_forms"]}/{counts["cases"]}</td><td>{counts["semantic_complete_goals"]}/{counts["cases"]}</td>'
                     f'<td>{correct}/{7*counts["cases"]}</td><td>{counts["depth_lifted_landmarks"]} / {counts["measured_landmarks_on_correct_instance"]}</td>'
                     f'<td>{counts["grasp_candidates"]} / {counts["correctly_bound_grasp_candidates"]}</td>'
                     f'<td>{counts["correct_path_choices"]}/{counts["cases"]}</td>'
                     f'<td>{counts["passing_relation_self_checks"]}/{counts["relation_self_checks"]}</td></tr>')
    usage = manifest['ledger']
    parts += ['</table></div>',
              '<p>The historical direct-XY baseline has 2/6 complete goals and six complete replies. Comparing it with these partial runs changes the proposer, numbered-image input, answer format, and inference workload simultaneously; that comparison does not isolate the feature backbone. DINOv2 versus DINOv3 has matched inputs and settings, but unequal answer completion and unrequested cases prevent an encoder-quality conclusion.</p>',
              f'<p>One shared API ledger: {usage["api_attempts"]}/12 attempted calls, estimated ${usage["estimated_usage_cost_usd"]:.5f}; admission cap ${usage["budget_usd"]:.2f}. Unknown-cost reservation: ${usage["unknown_usage_reserved_usd"]:.5f}. Local feature inference adds no API calls.</p>',
              f'<p>Stop record: {esc(json.dumps(manifest.get("stopped")))}</p>',
              '<div class="panel"><strong>Meaning of the scores:</strong> Whole-object mask membership grades instance binding only. It cannot verify a handle, finger contact, physical object axis, support affordance, stable grasp, IK, collisions, or execution. Passing equation/Jacobian checks verifies formula self-consistency, not task correctness. A candidate grasp frame is a proposal. A point target is not a verified whole-object placement.</div>',
              '<h2>Proposal coverage before Mini selection</h2><p>These held-back labels are used only after local proposals are frozen. None are in the model payload. “Available” means a point uniquely hits the expected whole object; shared hits are listed separately.</p>',
              '<div class="scroll"><table><tr><th>Condition / scene</th><th>Candidates</th><th>Source IDs</th><th>Target IDs</th><th>Required roles</th><th>Background / wrong instance / shared</th></tr>']
    for key, item in coverage.items():
        parts.append(f'<tr><td>{esc(key)}</td><td>{item["candidate_count"]}</td>'
                     f'<td>{esc(str(item["roles"]["source"]["available_candidate_ids"]))}</td>'
                     f'<td>{esc(str(item["roles"]["target"]["available_candidate_ids"]))}</td>'
                     f'<td>{item["required_roles_covered"]}/2</td>'
                     f'<td>{len(item["background_candidate_ids"])} / {len(item["wrong_instance_candidate_ids"])} / {len(item["shared_candidate_ids"])}</td></tr>')
    parts += ['</table></div>',
              '<p>Missing selection diagnostics: <code>proposal_missing</code> means no candidate hits the expected whole object; <code>selection_abstention</code> means a hit exists but Mini supplied null or no usable form. Wrong or invalid non-null selections are <code>selection_error</code>. These labels do not prove the required part or contact was available.</p>',
              '<p>Visual-selection diagnostics are applied only to complete accepted forms. Truncated replies are reported as <code>response_truncated</code>; unrequested cases as <code>not_run</code>. They are not scored as visual mistakes or intentional abstentions. The raw unchanged grader treats absent coordinates as unfilled; its anchor counts therefore include operationally unavailable coordinates.</p>',
              '<p>Selected anchor labels: S source, G grasp center, A/B closing edges, L/H lower/upper object-axis points, T target. “!” marks a point that did not uniquely hit the expected whole object. Labels are drawn after inference.</p>']
    for condition, counts in conditions.items():
        parts.append(f'<div class="panel"><strong>{esc(condition)} response completion</strong><pre>{esc(json.dumps({"response_status": counts["response_completion_counts"], "incomplete_reasons": counts["response_incomplete_reason_counts"], "unscored_selection_cases": counts["unscored_selection_cases"]}, indent=2))}</pre></div>')
    for task in tasks:
        identifier, metric = task['id'], task['metrics']
        encoded_viewer = json_text(viewers[identifier]).replace('<', '\\u003c')
        parts += [f'<article id="{esc(identifier)}"><h2>{esc(task["condition"])} · {esc(task["instruction"])}</h2>',
                  f'<p>Status: {esc(task["status"])} · complete goal: {metric["semantic_complete_goal"]} · path choice: {metric["path_choice_correct"]} · grasp proposed: {metric["grasp_candidate"]} · correctly bound grasp anchors: {metric["candidate_landmarks_on_correct_object"]}</p>',
                  f'<p>API response: {esc(task["response_context"]["status"])} · incomplete details: {esc(str(task["response_context"]["incomplete_details"]))} · raw choice context: {esc(task["response_context"]["raw_choice_context"])}. Visual selection diagnostics applicable: {task["response_context"]["selection_diagnostic_applicable"]}.</p>',
                  '<div class="grid">',
                  f'<div><h3>Original RGB sent to Mini</h3><img src="assets/{identifier}-original.png" alt="Original frozen RGB scene"></div>',
                  f'<div><h3>Numbered RGB sent to Mini</h3><img src="assets/{identifier}-numbered.png" alt="Exact numbered candidate image sent to Mini"></div>',
                  f'<div><h3>Selected unchanged pixels</h3><img src="assets/{identifier}-points.png" alt="Selected landmarks with offline error marks"></div>',
                  f'<div><h3>Aligned metric depth</h3><img src="assets/{identifier}-depth.png" alt="Metric depth, blue to yellow"><small>Blue → yellow: 0.6 → 3.1 meters.</small></div>',
                  f'<div><h3>Observed 3D geometry</h3><canvas class="cloud" data-scene="scene-{identifier}" aria-label="Interactive visible depth and selected landmarks"></canvas><small>Drag to rotate; scroll to zoom. The view shows visible surfaces and proposals.</small></div></div>',
                  f'<script type="application/json" id="scene-{identifier}">{encoded_viewer}</script>',
                  '<div class="scroll"><table><tr><th>Anchor / ID</th><th>Selection diagnostic</th><th>Whole-object binding</th><th>Depth status / world position, m</th></tr>']
        for anchor, selected in zip(metric['anchors'], metric['selection_diagnostics']):
            point = task['numericals']['landmarks'][anchor['field']] if task['numericals'] else {}
            value = ', '.join(f'{number:.4f}' for number in point['position_m']) if 'position_m' in point else point.get('reason', 'unresolved')
            selection_label = (selected['diagnostic'] if task['response_context']['selection_diagnostic_applicable']
                               else task['response_context']['unscored_selection_reason'])
            binding_label = anchor['diagnostic'] if task['response_context']['selection_diagnostic_applicable'] else 'unscored'
            parts.append(f'<tr><td>{LABELS[anchor["field"]]} / {esc(str(selected["selected_id"]))}</td>'
                         f'<td>{esc(selection_label)}</td><td>{esc(binding_label)}</td>'
                         f'<td>{esc(anchor["depth_status"])} / {esc(value)}</td></tr>')
        parts += ['</table></div>',
                  f'<details><summary>Response completion and raw partial text</summary><pre>{esc(json.dumps(task["response_context"], indent=2))}</pre></details>',
                  f'<details><summary>Raw ID choices and resolved blanks</summary><pre>{esc(json.dumps({"choices": task["choices"], "resolved": task["resolved_blanks"]}, indent=2))}</pre></details>',
                  f'<details><summary>Numerical estimates and unresolved values</summary><pre>{esc(json.dumps(task["numericals"], indent=2))}</pre></details>',
                  f'<details><summary>Equation checks and offline metrics</summary><pre>{esc(json.dumps(metric, indent=2))}</pre></details>',
                  f'<details><summary>Feature and candidate provenance</summary><pre>{esc(json.dumps(task["candidate_metadata"], indent=2))}</pre></details>',
                  f'<p><a href="assets/{identifier}-result.json">Case JSON</a> · <a href="assets/{identifier}-candidates.json">Fixed candidate pixels</a> · <a href="assets/{identifier}-metadata.json">Feature metadata</a>']
        for name in ('request.json', 'response.json', 'payload.json'):
            if (assets / (identifier+'-'+name)).exists():
                parts.append(f' · <a href="assets/{identifier}-{name}">{esc(name)}</a>')
        parts.append('</p></article>')
    parts += [f'<details class="panel"><summary>Exact common prompt</summary><pre>{esc(manifest["prompt_text"])}</pre></details>',
              '<p><a href="summary.json">Complete summary JSON</a> · <a href="assets/config.json">Configuration</a> · <a href="assets/protocol.json">Frozen protocol</a> · <a href="assets/manifest.json">Actual API ledger</a> · <a href="assets/prompt.txt">Prompt text</a> · <a href="assets/direct-xy-baseline.json">Historical direct-XY results</a></p>',
              '<script>'+VIEWER+'</script></body></html>']
    (output / 'index.html').write_text('\n'.join(parts))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--trial', default=DEFAULT_TRIAL)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build_report(args.config, args.trial, args.output)
    print(json_text({key: summary[key] for key in ('cases', 'unique_scenes', 'condition_summaries', 'ledger')}))


if __name__ == '__main__':
    main()
