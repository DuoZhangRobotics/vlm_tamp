"""Report the frozen Mini repair and separately unqueried initial-state validation."""

import json
import shutil

from report_rgb_choices import escaped, evaluation_overlay, pretty
from vlm_tamp.dataset import read_records
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_choice_v2_trial import file_sha
from vlm_tamp.rgb_evaluation import evaluate_rgb, load_references
from vlm_tamp.rgb_grounding import RGBPrediction


LABELS = {
    "inside": "Inside",
    "on_top": "On top",
    "in_front_of": "In front of",
    "open": "Open",
    "closed": "Closed",
    "powered_on": "Powered on",
    "powered_off": "Powered off",
    "unknown": "Unknown",
}


def main():
    dataset = ROOT / "data/rgb-choices-v2"
    records, root = read_records(dataset)
    by_id = {record["example_id"]: record for record in records}
    references = load_references(dataset, ROOT / "configs/rgb_choices_v2_references.json")
    preflight = json.loads((ROOT / "results/rgb-choice-v2-preflight.json").read_text())
    coverage = {task["example_id"]: task for task in preflight["tasks"]}
    output = ROOT / "results/rgb-choice-v2-report"
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    partitions, tasks = {}, []
    calls, cost = 0, 0.0
    for partition in ("dev", "validation"):
        trial = ROOT / f"results/rgb-choice-v2-mini-{partition}"
        manifest = json.loads((trial / "manifest.json").read_text())
        assert not manifest["offline_mock"] and not manifest["model_input_inventory"]
        protocol = json.loads((ROOT / manifest["config"]["frozen_protocol"]).read_text())
        assert protocol["previous_mini_manifest_sha256"] == file_sha(
            ROOT / "results/rgb-choice-mini/manifest.json"
        )
        assert protocol["previous_astra_manifest_sha256"] == file_sha(
            ROOT / "results/rgb-choice-astra/manifest.json"
        )
        rows = [json.loads(line) for line in (trial / "predictions.jsonl").read_text().splitlines()]
        summary = json.loads((trial / "summary.json").read_text())
        partitions[partition] = summary
        calls += manifest["ledger"]["api_attempts"]
        cost += manifest["ledger"]["estimated_usage_cost_usd"]
        for index, row in enumerate(rows):
            record = by_id[row["example_id"]]
            prediction = RGBPrediction.model_validate(row["prediction"]) if row["prediction"] else None
            metric = evaluate_rgb(record, prediction, references[row["example_id"]], root)
            prefix = f"{partition}-{index:02d}"
            for name in ("original", "scene", "cards"):
                shutil.copyfile(trial / f"input-{index:02d}/{name}.png", assets / f"{prefix}-{name}.png")
            evaluation_overlay(record, prediction, root, assets / f"{prefix}-evaluation.png")
            tasks.append(
                {
                    "example_id": row["example_id"],
                    "partition": partition,
                    "instruction": record["model_inputs"]["instruction"],
                    "status": row["status"],
                    "choices": row["choices"],
                    "prediction": row["prediction"],
                    "metric": metric,
                    "coverage": coverage[row["example_id"]],
                    "asset_prefix": prefix,
                }
            )
    assert cost <= 0.50 + 1e-12
    summary = {
        "model": "gpt-5.4-mini-2026-03-17",
        "partitions": partitions,
        "tasks": tasks,
        "new_api_attempts": calls,
        "new_estimated_usage_usd": cost,
        "cumulative_api_attempts": 204 + calls,
        "cumulative_estimated_usage_usd": 3.4300996 + cost,
        "candidate_coverage_tasks": sum(task["all_required_owners_covered"] for task in coverage.values()),
        "candidate_owners_covered": sum(task["covered_owners"] for task in coverage.values()),
        "candidate_owners_required": sum(task["required_owners"] for task in coverage.values()),
        "simulator_inventory_supplied": False,
        "task_held_out_validation": False,
        "validation_scope": "Unqueried initial states of the same ten task groups; both partitions frozen before either inference run",
        "grader_unchanged": True,
        "image_resolution_changed": [256, 512],
        "constraints_verified": False,
        "part_pixel_localization_verified": False,
        "robot_execution_tested": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pieces = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT-5.4 Mini: repaired RGB choices</title><style>body{font:16px system-ui;line-height:1.5;max-width:1500px;margin:auto;padding:24px;background:#111826;color:#edf2ff}.panel,article{background:#1b2638;padding:20px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:20px}img{width:100%;height:auto;max-width:768px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #405169;text-align:left}.ok{color:#74dfae}.bad{color:#ffa39d}details{margin:12px 0}</style></head><body>',
        "<h1>GPT-5.4 Mini: repaired RGB choices</h1><p>Fixed overlapping final-state choices, preserved smaller contrasting image regions, and used genuine 512-pixel RGB camera observations. The model chooses five blanks; code constructs final goals.</p>",
        '<div class="panel"><strong>RGB input only:</strong> choices are ranked from image contrast, component shape and spatial diversity. No simulator inventory, object category labels, reference masks, task answers, geometry or prior response history enter the API. The numbered scene and crop sheet below are the actual inputs.</div>',
        "<table><tr><th>Configuration</th><th>Complete final goals</th><th>Estimated usage</th></tr><tr><td>Previous Mini choices, 256 RGB — saved original score</td><td>8/10</td><td>$0.0839</td></tr>",
    ]
    for partition, label in (
        ("dev", "Repaired Mini, same initial states at 512 RGB"),
        ("validation", "Repaired Mini, new initial states at 512 RGB"),
    ):
        part = partitions[partition]
        pieces.append(
            f"<tr><td>{label}</td><td>{part['semantic_complete_goals']}/{part['planned_tasks']}</td><td>${part['ledger']['estimated_usage_cost_usd']:.4f}</td></tr>"
        )
    pieces.extend(
        [
            "</table>",
            f"<p>Candidate coverage: {summary['candidate_coverage_tasks']}/20 scenes and {summary['candidate_owners_covered']}/{summary['candidate_owners_required']} required owner instances. This was measured offline after freezing and is not model success. New usage totals ${cost:.6f}.</p>",
            '<div class="panel"><strong>Scope:</strong> the same frozen semantic final-goal grader is used. Resolution, candidate ranking, crop presentation and form vocabulary change together. Validation uses new poses of previously studied tasks; it is not an unseen-task benchmark. Numerical geometry, physical part location, grasp/path constraints and robot execution remain unverified.</div>',
        ]
    )
    for task in tasks:
        prefix = task["asset_prefix"]
        matched = task["metric"]["semantic_goal"]["exact_match"]
        pieces.append(
            f'<article><h2>{escaped(task["partition"])} · {escaped(task["instruction"])}</h2><p class="{"ok" if matched else "bad"}">{"Complete semantic final-goal match" if matched else "Mismatch / incomplete final goals"}</p><div class="grid"><section><h3>Original 512 RGB</h3><img src="assets/{prefix}-original.png" alt="Original higher-resolution camera image"></section><section><h3>Numbered image choices seen by Mini</h3><img src="assets/{prefix}-scene.png" alt="RGB-derived numbered choices"></section></div><h3>Mini-filled form</h3><table><tr><th>Final state</th><th>Object number</th><th>Object part</th><th>Destination number</th><th>Destination part</th></tr>'
        )
        for goal in (task["choices"] or {}).get("goals", []):
            pieces.append(
                f"<tr><td>{LABELS[goal['state']]}</td><td>{goal['object_choice'] or 'Not found'}</td><td>{escaped(goal['object_part'].replace('_', ' '))}</td><td>{goal['target_choice'] or 'None / not found'}</td><td>{escaped(goal['target_part'].replace('_', ' '))}</td></tr>"
            )
        pieces.extend(
            [
                "</table>",
                f'<details><summary>Actual context crops sent to Mini</summary><img src="assets/{prefix}-cards.png" alt="RGB context crops"></details>',
                f'<details><summary>Independent evaluation overlay — never sent to the model</summary><p>Green: independently rendered reference objects. Blue: selected image pixels.</p><img src="assets/{prefix}-evaluation.png" alt="Offline evaluation overlay"></details>',
                f"<details><summary>Image identity and goal checks</summary><pre>{pretty(task['metric'])}</pre></details>",
                f"<details><summary>Code-assembled declaration</summary><pre>{pretty(task['prediction'])}</pre></details>",
                f"<p>Output status: {escaped(task['status'])}. Scene: {escaped(task['example_id'])}.</p></article>",
            ]
        )
    pieces.append("</body></html>")
    (output / "index.html").write_text("\n".join(pieces))
    print(
        json.dumps(
            {
                "partition_scores": {
                    key: part["semantic_complete_goals"] for key, part in partitions.items()
                },
                "new_api_attempts": calls,
                "new_estimated_usage_usd": cost,
                "candidate_coverage_tasks": summary["candidate_coverage_tasks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
