"""RGB-only report: fresh compact-contract outputs and separately adapted saved RGB baselines."""

import hashlib
import json
import shutil

from PIL import Image, ImageDraw

from report_expanded_grounding import escaped, pretty
from vlm_tamp.dataset import read_records, safe_path
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_evaluation import evaluate_rgb, load_references
from vlm_tamp.rgb_grounding import RGBPrediction, from_legacy


def overlay(record, prediction, root, path):
    image = Image.open(safe_path(root, record["model_inputs"]["image"])).convert("RGB").resize((512, 512))
    draw = ImageDraw.Draw(image)
    for identifier, reference in record["reference"]["visual"].items():
        box = [x * 512 for x in reference["bbox"]]
        draw.rectangle(box, outline="#25c783", width=1)
        draw.text((box[0], box[1]), identifier, fill="#25c783", stroke_width=1, stroke_fill="black")
    if prediction:
        for entity in prediction.entities:
            if entity.bbox:
                box = [x * 512 for x in entity.bbox]
                draw.rectangle(box, outline="#f669d2", width=2)
            if entity.point:
                x, y = [p * 512 for p in entity.point]
                draw.line((x - 5, y, x + 5, y), fill="#65b9ff", width=3)
                draw.line((x, y - 5, x, y + 5), fill="#65b9ff", width=3)
                draw.text((x + 6, y), entity.id, fill="#65b9ff", stroke_width=1, stroke_fill="black")
    image.save(path)


def main():
    dataset = ROOT / "data/expanded-grounding"
    records, root = read_records(dataset)
    references = load_references(dataset, ROOT / "configs/rgb_semantic_references_v1.json")
    trial = ROOT / "results/rgb-repair-mini"
    rows = [json.loads(line) for line in (trial / "predictions.jsonl").read_text().splitlines() if line]
    manifest = json.loads((trial / "manifest.json").read_text())
    assert manifest["model_input_inventory"] is False and not manifest["offline_mock"]
    by_id = {r["example_id"]: r for r in records}
    runs = {}
    for key, label, path in [
        ("old-mini", "Saved Mini, old prompt · adapted goals", ROOT / "results/reasoning-mini-medium-raw"),
        ("old-astra", "Saved Astra, old prompt · adapted goals", ROOT / "results/reasoning-astra-medium-raw"),
    ]:
        predictions = {}
        for row in (
            json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines() if line
        ):
            if row["condition"] != "rgb":
                continue
            assert row.get("input_provenance") is None
            try:
                parsed = from_legacy(row.get("prediction") or row.get("unvalidated_prediction"))
                error = None
            except ValueError as exc:
                parsed, error = None, str(exc)
            predictions[row["example_id"]] = {
                "prediction": parsed,
                "source_status": row["status"],
                "adapter_error": error,
                "metric": evaluate_rgb(by_id[row["example_id"]], parsed, references[row["example_id"]], root),
            }
        assert len(predictions) == 10
        runs[key] = {
            "label": label,
            "examples": predictions,
            "additional_model_calls": 0,
            "same_model_output_reinterpreted": True,
        }
    fresh = {}
    for row in rows:
        parsed = RGBPrediction.model_validate(row["prediction"]) if row["prediction"] else None
        fresh[row["example_id"]] = {
            "prediction": parsed,
            "source_status": row["status"],
            "adapter_error": None,
            "metric": evaluate_rgb(by_id[row["example_id"]], parsed, references[row["example_id"]], root),
        }
    runs["new-mini"] = {
        "label": "Fresh Mini · 190-word RGB contract",
        "examples": fresh,
        "additional_model_calls": manifest["ledger"]["api_attempts"],
        "same_model_output_reinterpreted": False,
    }
    output = ROOT / "results/rgb-repair-report"
    output.mkdir(exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    summary = {
        "metric": "RGB semantic complete-goal match, not legacy BDDL-ID equality or robot execution",
        "tasks": 10,
        "inventory_provided_to_model": False,
        "input_fields": ["instruction", "image"],
        "new_usage": manifest["ledger"],
        "runs": {
            key: {
                "label": run["label"],
                "semantic_complete_goals": sum(
                    example["metric"]["semantic_goal"]["exact_match"] for example in run["examples"].values()
                ),
                "additional_model_calls": run["additional_model_calls"],
                "adapted_saved_outputs": run["same_model_output_reinterpreted"],
            }
            for key, run in runs.items()
        },
        "legacy_scores_unchanged": True,
        "reference_source": "Official task language and labeled instances, normalized before inference; synthetic coverage checks are not model results.",
        "part_pixel_localization_verified": False,
        "robot_execution_tested": False,
    }
    details = []
    pieces = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RGB-only grounding retry</title><style>body{font:16px system-ui;line-height:1.5;max-width:1650px;margin:auto;padding:24px;background:#111826;color:#edf2ff}.panel,article{background:#1b2638;padding:20px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:20px}img{width:100%;max-width:512px;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #405169;text-align:left}.ok{color:#74dfae}.bad{color:#ffa39d}.note{color:#b4c3dd}details{margin:12px 0}</style></head><body>',
        "<h1>RGB-only grounding: repaired prompt and evaluation</h1>",
        "<p>The fresh run receives only the original RGB and task instruction. No inventory, masks, scene IDs, poses or reference answers are supplied. Simulator labels are used offline to check actual image identity and intended relationships.</p>",
        '<div class="panel"><strong>Scoring distinction:</strong> This is a new semantic goal metric. It checks image-identified objects, natural-language relations and owned-part selectors. The previous raw BDDL-ID scores are unchanged. Old RGB outputs are explicitly adapted and rescored with this same evaluator; those rows are not new inference or a pure prompt ablation. Unknown part/region image geometry, grasp/path correctness and robot execution remain unverified.</div>',
        "<table><tr><th>RGB-only configuration</th><th>Semantic complete goals</th><th>New calls</th></tr>",
    ]
    for run in summary["runs"].values():
        pieces.append(
            f"<tr><td>{escaped(run['label'])}</td><td>{run['semantic_complete_goals']}/10</td><td>{run['additional_model_calls']}</td></tr>"
        )
    pieces.append(
        f'</table><p>New estimated usage ${manifest["ledger"]["estimated_usage_cost_usd"]:.6f}. The 190-word prompt replaces the previous 772-word bookkeeping prompt. All ten reference goals are scorable under this representation; synthetic controls establish coverage only.</p><p class="note">Green boxes: independent reference objects. Magenta boxes and blue points: model predictions. The model saw unannotated images.</p>'
    )
    for i, record in enumerate(records):
        image_name = f"original-{i:02d}.png"
        shutil.copyfile(safe_path(root, record["model_inputs"]["image"]), assets / image_name)
        pieces.append(
            f'<article><h2>{i + 1}. {escaped(record["model_inputs"]["instruction"])}</h2><div class="grid"><section><h3>Original RGB</h3><img src="assets/{image_name}" alt="Original unannotated scene"><details><summary>Frozen semantic reference, evaluation only</summary><pre>{pretty(references[record["example_id"]])}</pre></details></section>'
        )
        task = {
            "example_id": record["example_id"],
            "instruction": record["model_inputs"]["instruction"],
            "results": {},
        }
        for key, run in runs.items():
            example = run["examples"].get(
                record["example_id"],
                {
                    "prediction": None,
                    "source_status": "not_attempted",
                    "metric": evaluate_rgb(record, None, references[record["example_id"]], root),
                },
            )
            filename = f"prediction-{i:02d}-{key}.png"
            overlay(record, example["prediction"], root, assets / filename)
            metric = example["metric"]
            matched = metric["semantic_goal"]["exact_match"]
            task["results"][key] = {
                "semantic_goal_match": matched,
                "metric": metric,
                "source_status": example["source_status"],
                "prediction": example["prediction"].model_dump(mode="json")
                if example["prediction"]
                else None,
            }
            pieces.append(
                f'<section><h3>{escaped(run["label"])}</h3><img src="assets/{filename}" alt="Model image locations with independent object references"><p class="{"ok" if matched else "bad"}">{"Semantic goal match" if matched else "Mismatch / unresolved image identity"}</p><p class="note">Original output status: {escaped(example["source_status"])}</p>'
            )
            for title, data in [
                ("Model declaration", task["results"][key]["prediction"]),
                ("Independent identity and goal checks", metric),
            ]:
                pieces.append(f"<details><summary>{title}</summary><pre>{pretty(data)}</pre></details>")
            pieces.append("</section>")
        pieces.append("</div></article>")
        details.append(task)
    pieces.append("</body></html>")
    (output / "index.html").write_text("\n".join(pieces))
    summary["tasks_detail"] = details
    summary["dataset_sha256"] = hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest()
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("tasks_detail",)}, indent=2))


if __name__ == "__main__":
    main()
