"""Report prospective fixed-prompt RGB grounding on new task groups."""

import json
import shutil
from collections import Counter

from report_rgb_choices import escaped, evaluation_overlay, pretty
from report_rgb_role_points import point_overlay
from vlm_tamp.dataset import read_records
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_evaluation import evaluate_rgb, load_references
from vlm_tamp.rgb_grounding import RGBPrediction
from vlm_tamp.rgb_role_choices import instruction_template
from vlm_tamp.rgb_role_diagnostics import diagnose_role_points
from vlm_tamp.rgb_role_generalization_trial import file_sha, load_generalization_config


def main():
    config, protocol, prompt = load_generalization_config(ROOT / "configs/rgb_role_generalization_trial.json")
    records, root = read_records(ROOT / protocol["dataset"])
    by_id = {record["example_id"]: record for record in records}
    references = load_references(root, ROOT / config["references"])
    offline_roles = json.loads((ROOT / protocol["offline_roles_path"]).read_text())["roles"]
    trial = ROOT / "results/rgb-role-generalization-mini"
    manifest = json.loads((trial / "manifest.json").read_text())
    assert not manifest["offline_mock"] and not manifest["model_input_inventory"]
    rows = [json.loads(line) for line in (trial / "predictions.jsonl").read_text().splitlines()]
    source_summary = json.loads((trial / "summary.json").read_text())
    output = ROOT / "results/rgb-role-generalization-report"
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    tasks, suite_scores, diagnostics = [], {}, Counter()
    image_hashes = set()
    for index, row in enumerate(rows):
        record = by_id[row["example_id"]]
        prediction = RGBPrediction.model_validate(row["prediction"]) if row["prediction"] else None
        metric = evaluate_rgb(record, prediction, references[row["example_id"]], root)
        role_details = diagnose_role_points(record, row["points"], offline_roles[row["example_id"]], root)
        diagnostics.update(detail["diagnostic"] for detail in role_details)
        suite = record["metadata"]["suite"]
        score = suite_scores.setdefault(suite, {"scenes": 0, "complete_goals": 0,
                                               "roles": 0, "correct_roles": 0})
        score["scenes"] += 1
        score["complete_goals"] += metric["semantic_goal"]["exact_match"]
        score["roles"] += len(role_details)
        score["correct_roles"] += sum(detail["diagnostic"] == "correct_instance" for detail in role_details)
        prefix = f"task-{index:02d}"
        source = trial / f"input-{index:02d}/original.png"
        image_hashes.add(file_sha(source))
        shutil.copyfile(source, assets / f"{prefix}-original.png")
        point_overlay(source, row["points"], assets / f"{prefix}-points.png")
        evaluation_overlay(record, prediction, root, assets / f"{prefix}-evaluation.png")
        tasks.append({"example_id": row["example_id"], "instruction": record["model_inputs"]["instruction"],
                      "suite": suite, "status": row["status"], "points": row["points"],
                      "roles": instruction_template(record["model_inputs"]["instruction"])["roles"],
                      "prediction": row["prediction"], "metric": metric,
                      "offline_role_diagnostics": role_details, "asset_prefix": prefix})
    cost = manifest["ledger"]["estimated_usage_cost_usd"]
    summary = {**source_summary, "model": manifest["model"], "tasks": tasks,
               "suite_scores": suite_scores, "role_diagnostics": dict(diagnostics),
               "correct_roles": diagnostics["correct_instance"], "evaluated_roles": sum(diagnostics.values()),
               "catalogue_coverage": protocol["coverage"], "image_resolution": 512, "source_images_per_request": 1,
               "unique_original_rgb_images": len(image_hashes),
               "task_held_out_from_project_queries": True, "prompt_and_compiler_unchanged": True,
               "grader_unchanged": True, "comparison_scope": protocol["comparison_scope"],
               "cumulative_api_attempts": protocol["prior_api_attempts"] + manifest["ledger"]["api_attempts"],
               "cumulative_estimated_usage_usd": protocol["prior_estimated_usage_usd"] + cost,
               "constraints_verified": False, "physical_part_localization_verified": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    coverage = protocol["coverage"]
    pieces = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mini RGB grounding on new tasks</title><style>body{font:16px system-ui;line-height:1.5;max-width:1450px;margin:auto;padding:24px;background:#111826;color:#edf2ff}article,.panel{background:#1b2638;padding:20px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:20px}img{width:100%;height:auto;max-width:768px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #405169;text-align:left}.ok{color:#74dfae}.bad{color:#ffa39d}details{margin:12px 0}a{color:#79d5ff}</style></head><body>',
        '<h1>Mini RGB grounding on new tasks</h1><p>The same frozen prompt, raw RGB interface, role parser and semantic grader are evaluated on twenty new task groups. Mini sees one original 512 × 512 RGB image and instruction-derived roles, then returns normalized image points.</p>',
        f'<div class="panel"><strong>Complete symbolic final goals: {source_summary["semantic_complete_goals"]}/20.</strong> Valid JSON: {source_summary["valid_forms"]}/{source_summary["attempted_calls"]}. Correct individual roles: {summary["correct_roles"]}/{summary["evaluated_roles"]}. New API attempts: {manifest["ledger"]["api_attempts"]}; estimated usage ${cost:.5f}, within $0.10.</div>',
        '<table><tr><th>Suite</th><th>Complete goals</th><th>Correct image roles</th></tr>',
    ]
    for suite, score in suite_scores.items():
        pieces.append(f'<tr><td>{escaped(suite)}</td><td>{score["complete_goals"]}/{score["scenes"]}</td><td>{score["correct_roles"]}/{score["roles"]}</td></tr>')
    pieces.extend([
        '</table>',
        f'<div class="panel"><strong>Coverage:</strong> {coverage["supported_structures"]}/{coverage["catalogue_tasks"]} catalogue tasks pass the current instruction/reference goal-structure audit; {coverage["new_supported_structures"]} are new to project queries. These twenty tasks were selected from compatible contracts before rendering or model responses. Unsupported contracts are recorded separately. This score is for the selected supported interface, not whole-LIBERO accuracy.</div>',
        '<div class="panel"><strong>Scope:</strong> task groups are disjoint from earlier paid calls by suite/index, task name and BDDL hash. One new initial arrangement is used per task. No segmentation, crops, scene inventory, reference masks or earlier responses enter inference. Points are scored unchanged against offline masks. A wrong-instance point and a background/occluder point are different observations; a point-only output cannot prove whether an error came from recognition or coordinate precision. Motion, physical parts and constraints remain unverified.</div>',
        '<details><summary>Unchanged model prompt</summary><pre>' + escaped(prompt) + '</pre></details>',
        '<details><summary>Catalogue coverage audit</summary><pre>' + pretty(coverage) + '</pre></details>',
    ])
    for task in tasks:
        prefix = task["asset_prefix"]
        matched = task["metric"]["semantic_goal"]["exact_match"]
        pieces.append(f'<article><h2>{escaped(task["instruction"])}</h2><p class="{"ok" if matched else "bad"}">{"Complete goal match" if matched else "Goal mismatch or abstention"}</p><div class="grid"><section><h3>Actual RGB input</h3><img src="assets/{prefix}-original.png" alt="Original RGB supplied to Mini"></section><section><h3>Returned points</h3><p>Overlay added after inference.</p><img src="assets/{prefix}-points.png" alt="Mini image points shown after inference"></section></div><table><tr><th>Label</th><th>Instruction role</th><th>Returned point</th><th>Offline diagnostic</th></tr>')
        detail_by_slot = {detail["slot"]: detail for detail in task["offline_role_diagnostics"]}
        for number, (slot, phrase) in enumerate(task["roles"].items(), 1):
            detail = detail_by_slot[slot]
            pieces.append(f'<tr><td>{number}</td><td>{escaped(phrase)}</td><td>{escaped(detail["point"])}</td><td>{escaped(detail["diagnostic"])}</td></tr>')
        pieces.extend([
            '</table>',
            f'<details><summary>Offline reference overlay — withheld from Mini</summary><img src="assets/{prefix}-evaluation.png" alt="Independent reference boxes and predicted points"></details>',
            f'<details><summary>Returned JSON and assembled goals</summary><pre>{pretty(task["points"])}</pre><pre>{pretty(task["prediction"])}</pre></details>',
            f'<details><summary>Exact final-goal comparison</summary><pre>{pretty(task["metric"])}</pre></details>',
            f'<details><summary>Offline point diagnostics</summary><pre>{pretty(task["offline_role_diagnostics"])}</pre></details>',
            f'<p>Scene: {escaped(task["example_id"])}. Status: {escaped(task["status"])}.</p></article>',
        ])
    pieces.append('</body></html>')
    (output / "index.html").write_text("\n".join(pieces))
    print(json.dumps({"complete_goals": source_summary["semantic_complete_goals"],
                      "planned_tasks": source_summary["planned_tasks"], "suite_scores": suite_scores,
                      "role_diagnostics": dict(diagnostics), "estimated_usage_usd": cost,
                      "report": str(output / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
