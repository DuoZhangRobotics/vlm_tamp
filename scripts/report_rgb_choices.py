"""Report actual finite-choice results against the saved RGB-only Mini baseline."""

import argparse
import html
import json
import shutil

from PIL import Image, ImageDraw

from vlm_tamp.dataset import read_records, safe_path
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_choice_trial import file_sha
from vlm_tamp.rgb_evaluation import evaluate_rgb, load_references
from vlm_tamp.rgb_grounding import RGBPrediction


LABELS = {
    "inside": "Inside",
    "supported_on": "Supported on",
    "in_front_of": "In front of",
    "open": "Open",
    "closed": "Closed",
    "turned_on": "Turned on",
    "turned_off": "Turned off",
    "stacked_on": "Stacked on",
    "unknown": "Unknown",
}


def escaped(value):
    return html.escape(str(value))


def pretty(value):
    return escaped(json.dumps(value, indent=2))


def evaluation_overlay(record, prediction, root, path):
    image = Image.open(safe_path(root, record["model_inputs"]["image"])).convert("RGB").resize((768, 768))
    draw = ImageDraw.Draw(image)
    for identifier, visual in record["reference"]["visual"].items():
        box = [coordinate * 768 for coordinate in visual["bbox"]]
        draw.rectangle(box, outline="#66da87", width=1)
        draw.text((box[0], box[1]), identifier, fill="#66da87", stroke_width=1, stroke_fill="black")
    if prediction:
        for entity in prediction.entities:
            if entity.point is None:
                continue
            x, y = [coordinate * 768 for coordinate in entity.point]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#4ec9ff", outline="black")
            draw.text((x + 7, y), entity.id, fill="#4ec9ff", stroke_width=1, stroke_fill="black")
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description="Report RGB choices against a saved RGB-only baseline")
    parser.add_argument("--experiment", default="results/rgb-choice-mini")
    parser.add_argument("--baseline", default="results/rgb-repair-mini")
    parser.add_argument("--output", default="results/rgb-choice-report")
    parser.add_argument("--label", default="New finite-choice form")
    parser.add_argument("--baseline-label", default="Saved free-form RGB baseline")
    args = parser.parse_args()
    dataset = ROOT / "data/expanded-grounding"
    records, root = read_records(dataset)
    references = load_references(dataset, ROOT / "configs/rgb_semantic_references_v1.json")
    trial = ROOT / args.experiment
    baseline_path = ROOT / args.baseline
    manifest = json.loads((trial / "manifest.json").read_text())
    protocol = json.loads((ROOT / manifest["config"]["frozen_protocol"]).read_text())
    assert not manifest["model_input_inventory"] and not manifest["offline_mock"]
    assert protocol["baseline_manifest_sha256"] == file_sha(baseline_path / "manifest.json")
    saved_manifest = json.loads((baseline_path / "manifest.json").read_text())
    assert not saved_manifest["model_input_inventory"] and not saved_manifest["offline_mock"]
    rows = [json.loads(line) for line in (trial / "predictions.jsonl").read_text().splitlines() if line]
    baseline = {
        row["example_id"]: row
        for row in (
            json.loads(line) for line in (baseline_path / "predictions.jsonl").read_text().splitlines()
        )
    }
    assert all(row["condition"] in {"rgb", "rgb_choices"} for row in baseline.values())
    preflight = json.loads((ROOT / "results/rgb-choice-preflight.json").read_text())
    coverage = {c["example_id"]: c for c in preflight["tasks"]}
    by_id = {r["example_id"]: r for r in records}
    output = ROOT / args.output
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    tasks = []
    for index, row in enumerate(rows):
        record = by_id[row["example_id"]]
        new = RGBPrediction.model_validate(row["prediction"]) if row["prediction"] else None
        old_payload = baseline[row["example_id"]]["prediction"]
        old = RGBPrediction.model_validate(old_payload) if old_payload else None
        metric = evaluate_rgb(record, new, references[row["example_id"]], root)
        old_metric = evaluate_rgb(record, old, references[row["example_id"]], root)
        if metric["semantic_goal"]["exact_match"]:
            inspection = "Selected image referents and final goals match."
        elif row["status"] == "abstained":
            inspection = "The model selected not found / unknown. The form remains valid, but a complete goal declaration cannot be assembled."
        else:
            predicted_arguments = sorted(
                json.dumps(goal[1]) for goal in metric.get("predicted_goal_signatures", [])
            )
            expected_arguments = sorted(
                json.dumps(goal[1]) for goal in references[row["example_id"]]["goals"]
            )
            inspection = (
                "The selected image referents and part arguments are correct. The final-state relation name differs from the frozen reference."
                if predicted_arguments == expected_arguments
                else "At least one image referent or part argument is wrong or unresolved."
            )
        for name in ("original", "scene", "cards"):
            shutil.copyfile(trial / f"input-{index:02d}/{name}.png", assets / f"{name}-{index:02d}.png")
        evaluation_overlay(record, new, root, assets / f"evaluation-{index:02d}.png")
        tasks.append(
            {
                "example_id": row["example_id"],
                "instruction": record["model_inputs"]["instruction"],
                "status": row["status"],
                "choices": row["choices"],
                "compiled_prediction": row["prediction"],
                "metric": metric,
                "baseline_metric": old_metric,
                "candidate_coverage": coverage[row["example_id"]],
                "latency_seconds": row["latency_seconds"],
                "post_run_inspection": inspection,
            }
        )
    source_summary = json.loads((trial / "summary.json").read_text())
    summary = {
        **source_summary,
        "tasks": tasks,
        "baseline_semantic_complete_goals": sum(
            t["baseline_metric"]["semantic_goal"]["exact_match"] for t in tasks
        ),
        "full_candidate_coverage_tasks": preflight["tasks_with_full_candidate_coverage"],
        "required_owners_covered": sum(c["covered_owners"] for c in coverage.values()),
        "required_owners": sum(c["required_owners"] for c in coverage.values()),
        "baseline_estimated_cost_usd": saved_manifest["ledger"]["estimated_usage_cost_usd"],
        "baseline_output_tokens": saved_manifest["ledger"]["output_tokens"],
        "baseline_input_tokens": saved_manifest["ledger"]["input_tokens"],
        "baseline_reasoning_tokens": saved_manifest["ledger"]["reasoning_tokens"],
        "prior_and_cumulative_usage": {
            "cumulative_api_attempts": protocol["prior_api_attempts"] + manifest["ledger"]["api_attempts"],
            "cumulative_estimated_usage_usd": protocol["prior_estimated_usage_usd"]
            + manifest["ledger"]["estimated_usage_cost_usd"],
        },
        "comparison_limit": protocol["comparison_scope"],
        "part_pixel_localization_verified": False,
        "constraints_verified": False,
        "current_model": manifest["model"],
        "baseline_model": saved_manifest["model"],
        "current_label": args.label,
        "baseline_label": args.baseline_label,
        "output_token_limits": {
            "current": manifest["config"]["max_output_tokens"],
            "baseline": saved_manifest["config"]["max_output_tokens"],
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pieces = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RGB multiple-choice grounding</title><style>body{font:16px system-ui;line-height:1.5;max-width:1500px;margin:auto;padding:24px;background:#111826;color:#edf2ff}.panel,article{background:#1b2638;padding:20px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:20px}img{width:100%;height:auto;max-width:768px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #405169;text-align:left}.ok{color:#74dfae}.bad{color:#ffa39d}.note{color:#b4c3dd}details{margin:12px 0}</style></head><body>',
        "<h1>RGB grounding by filling blanks</h1>",
        "<p>The model selects a desired state, object number, destination number and part selectors. Code constructs the semantic goals. It generates no object descriptions, coordinates, stages or constraints.</p>",
        '<div class="panel"><strong>Where the choices come from:</strong> fixed color-region processing of the original RGB. All scene regions compete, including robot/background distractors. Numbers mark image pixels, not simulator objects. No inventory, instance masks, category labels, poses or reference answers enter an API request. The original scene, numbered scene and crop sheet are all shown to the model.</div>',
        f"<table><tr><th>Model / form, medium reasoning</th><th>Complete semantic goals</th><th>Estimated usage</th></tr><tr><td>{escaped(args.baseline_label)}</td><td>{summary['baseline_semantic_complete_goals']}/10</td><td>${summary['baseline_estimated_cost_usd']:.4f}</td></tr><tr><td>{escaped(args.label)}</td><td>{summary['semantic_complete_goals']}/10</td><td>${manifest['ledger']['estimated_usage_cost_usd']:.4f}</td></tr></table>",
        f"<p>Candidate coverage: {summary['full_candidate_coverage_tasks']}/10 tasks; {summary['required_owners_covered']}/{summary['required_owners']} required object owners have a selectable image point. Coverage is measured offline after freezing and is not VLM success.</p>",
        f"<p>Valid forms: {summary['valid_forms']}/10. Compiled: {summary['compiled']}/10. Abstained: {summary['abstained']}. Relation types matched: {summary['relation_types_match']}/10.</p>",
        f'<div class="panel"><strong>Comparison limits:</strong> {escaped(protocol["comparison_scope"])} Whole-object image identity and semantic part selectors are checked; physical part geometry, required movement method, contact/path constraints and robot execution are untested.</div>',
    ]
    for index, task in enumerate(tasks):
        matched = task["metric"]["semantic_goal"]["exact_match"]
        pieces.extend(
            [
                f'<article><h2>{index + 1}. {escaped(task["instruction"])}</h2><p class="{"ok" if matched else "bad"}">{"Complete semantic goal match" if matched else "Mismatch / incomplete final goals"}</p><p>{escaped(task["post_run_inspection"])}</p><div class="grid"><section><h3>Original RGB</h3><img src="assets/original-{index:02d}.png" alt="Original scene"></section><section><h3>Numbered image choices seen by the model</h3><img src="assets/scene-{index:02d}.png" alt="RGB-only numbered regions"></section></div>',
                "<h3>Model-filled form</h3><table><tr><th>Desired state</th><th>Object choice</th><th>Object part</th><th>Destination choice</th><th>Destination part</th></tr>",
            ]
        )
        for goal in (task["choices"] or {}).get("goals", []):
            pieces.append(
                f"<tr><td>{escaped(LABELS[goal['state']])}</td><td>{goal['object_choice'] or 'Not found'}</td><td>{escaped(goal['object_part'].replace('_', ' '))}</td><td>{goal['target_choice'] or 'None / not found'}</td><td>{escaped(goal['target_part'].replace('_', ' '))}</td></tr>"
            )
        pieces.extend(
            [
                "</table>",
                f'<details><summary>Numbered crop sheet also seen by the model</summary><img src="assets/cards-{index:02d}.png" alt="Numbered context crops with exact selected pixels"></details>',
                f'<details><summary>Offline evaluation overlay — never sent to the model</summary><p>Green: independent simulator reference boxes. Blue: pixels selected through the form.</p><img src="assets/evaluation-{index:02d}.png" alt="Independent offline evaluation"></details>',
                f"<details><summary>Image identity and final-goal checks</summary><pre>{pretty(task['metric'])}</pre></details>",
                f"<details><summary>Code-assembled semantic declaration</summary><pre>{pretty(task['compiled_prediction'])}</pre></details>",
                f'<p class="note">{escaped(args.baseline_label)}: {"matched" if task["baseline_metric"]["semantic_goal"]["exact_match"] else "mismatched"}. Output status: {escaped(task["status"])}.</p></article>',
            ]
        )
    pieces.append("</body></html>")
    (output / "index.html").write_text("\n".join(pieces))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "semantic_complete_goals",
                    "baseline_semantic_complete_goals",
                    "full_candidate_coverage_tasks",
                    "required_owners_covered",
                    "required_owners",
                    "ledger",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
