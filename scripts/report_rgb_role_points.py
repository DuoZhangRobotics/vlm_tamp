"""Compare actual direct-RGB points with saved MobileSAM-assisted role matching."""

import json
import shutil

from PIL import Image, ImageDraw

from report_rgb_choices import escaped, evaluation_overlay, pretty
from vlm_tamp.dataset import read_records
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_evaluation import evaluate_rgb, load_references
from vlm_tamp.rgb_grounding import RGBPrediction
from vlm_tamp.rgb_role_choices import instruction_template
from vlm_tamp.rgb_role_point_trial import load_point_config


def point_overlay(source, points, path):
    image = Image.open(source).convert("RGB").resize((768, 768))
    draw = ImageDraw.Draw(image)
    for number, point in enumerate((points or {}).values(), 1):
        if point is None:
            continue
        x, y = [coordinate * 768 for coordinate in point]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline="#4ec9ff", width=3)
        draw.line((x - 10, y, x + 10, y), fill="#4ec9ff", width=1)
        draw.line((x, y - 10, x, y + 10), fill="#4ec9ff", width=1)
        draw.text((x + 10, y + 4), str(number), fill="#4ec9ff", stroke_width=2, stroke_fill="black")
    image.save(path)


def main():
    config, protocol, prompt = load_point_config(ROOT / "configs/rgb_role_points_trial.json")
    dataset = ROOT / protocol["dataset"]
    records, root = read_records(dataset)
    by_id = {record["example_id"]: record for record in records}
    references = load_references(dataset, ROOT / config["references"])
    trial = ROOT / "results/rgb-role-points-mini"
    baseline = ROOT / protocol["baseline"]
    manifest = json.loads((trial / "manifest.json").read_text())
    assert not manifest["offline_mock"] and not manifest["model_input_inventory"]
    assert not manifest["segmentation_used"] and not manifest["crops_used"]
    rows = [json.loads(line) for line in (trial / "predictions.jsonl").read_text().splitlines()]
    old_rows = {
        row["example_id"]: row
        for row in (json.loads(line) for line in (baseline / "predictions.jsonl").read_text().splitlines())
    }
    source_summary = json.loads((trial / "summary.json").read_text())
    old_summary = json.loads((baseline / "summary.json").read_text())
    output = ROOT / "results/rgb-role-points-report"
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    tasks = []
    for index, row in enumerate(rows):
        record = by_id[row["example_id"]]
        prediction = RGBPrediction.model_validate(row["prediction"]) if row["prediction"] else None
        old_payload = old_rows[row["example_id"]]["prediction"]
        old_prediction = RGBPrediction.model_validate(old_payload) if old_payload else None
        metric = evaluate_rgb(record, prediction, references[row["example_id"]], root)
        old_metric = evaluate_rgb(record, old_prediction, references[row["example_id"]], root)
        prefix = f"task-{index:02d}"
        source = trial / f"input-{index:02d}/original.png"
        shutil.copyfile(source, assets / f"{prefix}-original.png")
        point_overlay(source, row["points"], assets / f"{prefix}-points.png")
        evaluation_overlay(record, prediction, root, assets / f"{prefix}-evaluation.png")
        unresolved = [item["entity"] for item in metric["identity_details"] if item["resolved"] is None]
        tasks.append({
            "example_id": row["example_id"], "instruction": record["model_inputs"]["instruction"],
            "status": row["status"], "points": row["points"], "prediction": row["prediction"],
            "roles": instruction_template(record["model_inputs"]["instruction"])["roles"],
            "metric": metric, "baseline_metric": old_metric,
            "unresolved_referents": unresolved, "asset_prefix": prefix,
        })
    transitions = {
        "both_correct": 0, "direct_rgb_only_correct": 0,
        "segmentation_only_correct": 0, "both_incorrect": 0,
    }
    for task in tasks:
        direct = task["metric"]["semantic_goal"]["exact_match"]
        old = task["baseline_metric"]["semantic_goal"]["exact_match"]
        key = ("both_correct" if direct and old else "direct_rgb_only_correct" if direct
               else "segmentation_only_correct" if old else "both_incorrect")
        transitions[key] += 1
    cost = manifest["ledger"]["estimated_usage_cost_usd"]
    summary = {
        **source_summary, "model": manifest["model"], "tasks": tasks,
        "baseline_semantic_complete_goals": sum(
            task["baseline_metric"]["semantic_goal"]["exact_match"] for task in tasks),
        "baseline_ledger": old_summary["ledger"], "paired_transitions": transitions,
        "source_image_count": 1, "image_resolution": 512,
        "vlm_job": "Match the same instruction-derived roles to model-selected normalized image points",
        "goal_compiler_unchanged": True, "grader_unchanged": True,
        "comparison_scope": protocol["comparison_scope"], "images_previously_queried": True,
        "task_held_out_validation": False,
        "cumulative_api_attempts": protocol["prior_api_attempts"] + manifest["ledger"]["api_attempts"],
        "cumulative_estimated_usage_usd": protocol["prior_estimated_usage_usd"] + cost,
        "constraints_verified": False, "part_pixel_localization_verified": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pieces = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mini grounding directly from RGB</title><style>body{font:16px system-ui;line-height:1.5;max-width:1400px;margin:auto;padding:24px;background:#111826;color:#edf2ff}article,.panel{background:#1b2638;padding:20px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:20px}img{width:100%;height:auto;max-width:768px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #405169;text-align:left}.ok{color:#74dfae}.bad{color:#ffa39d}details{margin:12px 0}a{color:#79d5ff}</style></head><body>',
        '<h1>Mini grounding directly from RGB</h1><p>GPT-5.4 Mini receives one untouched 512 × 512 RGB image and the same instruction-derived roles. It returns its own image points. Segmentation, numbered candidates and crops are removed. The goal compiler and offline semantic grader are unchanged.</p>',
        '<table><tr><th>Pipeline on the same ten scenes</th><th>Complete goal matches</th><th>Valid JSON</th><th>Estimated usage</th></tr>',
        f'<tr><td>Saved MobileSAM + numbered scene + crops</td><td>{old_summary["semantic_complete_goals"]}/10</td><td>{old_summary["valid_forms"]}/10</td><td>${old_summary["ledger"]["estimated_usage_cost_usd"]:.5f}</td></tr>',
        f'<tr><td>Original RGB + direct image points</td><td>{source_summary["semantic_complete_goals"]}/10</td><td>{source_summary["valid_forms"]}/10</td><td>${cost:.5f}</td></tr></table>',
        '<div class="panel"><strong>What this measures:</strong> correct object instances and complete symbolic final goals. Each returned point is tested against offline reference masks, with no coordinate adjustment or nearest-object rescue. These are repeated scenes from the same ten task groups, not a new generalization benchmark. Removing segmentation also changes the answer interface from candidate numbers to coordinates; this is not a crops-only comparison. Motion, constraints and physical part geometry remain untested.</div>',
        '<details><summary>Actual model prompt</summary><pre>' + escaped(prompt) + '</pre></details>',
    ]
    for task in tasks:
        matched = task["metric"]["semantic_goal"]["exact_match"]
        old_matched = task["baseline_metric"]["semantic_goal"]["exact_match"]
        prefix = task["asset_prefix"]
        pieces.append(
            f'<article><h2>{escaped(task["instruction"])}</h2><p class="{"ok" if matched else "bad"}">'
            f'Direct RGB: {"complete goal match" if matched else "goal mismatch"}. '
            f'Saved segmentation pipeline: {"match" if old_matched else "mismatch"}.</p>'
            f'<div class="grid"><section><h3>Actual RGB sent to Mini</h3><img src="assets/{prefix}-original.png" alt="Untouched original RGB input"></section>'
            f'<section><h3>Mini’s returned points</h3><p>Overlay added after inference for inspection.</p><img src="assets/{prefix}-points.png" alt="Model-selected points overlaid after inference"></section></div>'
            '<table><tr><th>Point label</th><th>Instruction role</th><th>Returned [x,y]</th></tr>'
        )
        for number, (key, phrase) in enumerate(task["roles"].items(), 1):
            point = (task["points"] or {}).get(key)
            pieces.append(f'<tr><td>{number}</td><td>{escaped(phrase)}</td><td>{escaped(point)}</td></tr>')
        pieces.extend([
            '</table>',
            f'<details><summary>Offline reference overlay — withheld from Mini</summary><p>Green boxes show reference identities; blue points are model predictions.</p><img src="assets/{prefix}-evaluation.png" alt="Offline evaluation overlay"></details>',
            f'<details><summary>Returned JSON</summary><pre>{pretty(task["points"])}</pre></details>',
            f'<details><summary>Code-assembled goals</summary><pre>{pretty(task["prediction"])}</pre></details>',
            f'<details><summary>Exact identity and goal comparison</summary><pre>{pretty(task["metric"])}</pre></details>',
            f'<p>Output status: {escaped(task["status"])}. Scene: {escaped(task["example_id"])}.</p></article>',
        ])
    pieces.append('</body></html>')
    (output / "index.html").write_text("\n".join(pieces))
    print(json.dumps({"direct_rgb_complete_goals": source_summary["semantic_complete_goals"],
                      "saved_segmentation_complete_goals": summary["baseline_semantic_complete_goals"],
                      "paired_transitions": transitions, "new_estimated_usage_usd": cost,
                      "report": str(output / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
