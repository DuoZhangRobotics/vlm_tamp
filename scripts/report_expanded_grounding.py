"""Build an offline paired report from preserved raw and linked grounding outputs."""

import hashlib
import html
import json
import shutil
from collections import Counter

from PIL import Image, ImageDraw, ImageFont

from vlm_tamp.dataset import read_records, safe_path
from vlm_tamp.libero import ROOT
from vlm_tamp.visualization import contact_sheet, overlay


def read_rows(path):
    return {
        (r["example_id"], r["condition"]): r
        for r in (json.loads(line) for line in path.read_text().splitlines() if line)
    }


def escaped(value):
    return html.escape(str(value))


def pretty(value):
    return escaped(json.dumps(value, indent=2))


def matches(metric, key="symbolic"):
    return bool((metric.get(key) or {}).get("exact_match"))


def prompt_audit_notice(dataset):
    path = ROOT / "results/prompt-audit-2026-10-12/audit.json"
    if not path.exists():
        return ""
    audit = json.loads(path.read_text())
    if audit.get("dataset_sha256") != hashlib.sha256((dataset / "records.jsonl").read_bytes()).hexdigest():
        return ""
    if audit.get("evaluator_sha256") != hashlib.sha256(
        (ROOT / "src/vlm_tamp/evaluation.py").read_bytes()
    ).hexdigest():
        return ""
    return (
        '<div class="panel"><strong>Prompt/evaluation audit, 2026-10-12:</strong> '
        "The historical counts are unchanged. Five of ten RGB-only tasks require region IDs that the "
        "prompt forbids inventing and the current evaluator cannot recover from available visual "
        "references. Strict RGB complete-goal agreement is therefore structurally capped at 5/10. "
        "It is not a pure task-understanding accuracy measure. The cheese task also has an In/On domain "
        "convention mismatch, and the output schema permits goal-name strings rejected by backend validation. "
        '<a href="../prompt-audit-2026-10-12/audit.json">Inspect the audit</a>; '
        '<a href="../prompt-audit-2026-10-12/instructions.txt">read the exact prompt</a>.</div>'
    )


def preview(records, assets, output):
    figure = Image.new("RGB", (1050, 525), "#f4f7fb")
    draw = ImageDraw.Draw(figure)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 25)
        text_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        title_font = text_font = ImageFont.load_default()
    draw.text((18, 16), "Selected outputs: RGB + simulator inventory", fill="#142338", font=title_font)
    cases = [
        (
            "libero_10-03-state000-seed0",
            "Bowl in drawer, then close",
            "Final goals match",
            "Grasp and motion remain unverified",
            "#087d57",
        ),
        (
            "libero_10-04-state000-seed0",
            "Two mugs, two plates",
            "Wrong mug assignment",
            "Same mug assigned to both plates",
            "#ae2530",
        ),
        (
            "libero_10-09-state000-seed0",
            "Mug in microwave, then close",
            "Extra mug + wrong region",
            "One multicolored mug became two",
            "#ae2530",
        ),
    ]
    for column, (example, title, label, detail, color) in enumerate(cases):
        index = next(i for i, record in enumerate(records) if record["example_id"] == example)
        x = column * 350 + 18
        draw.text((x, 63), title, fill="#142338", font=text_font)
        with Image.open(assets / f"prediction-{index:02d}-candidates.png") as image:
            figure.paste(image.resize((314, 314), Image.Resampling.LANCZOS), (x, 96))
        draw.text((x, 426), label, fill=color, font=text_font)
        draw.text((x, 455), detail, fill="#40516b", font=text_font)
    draw.text(
        (18, 496),
        "Green boxes: measured references. Blue points: model predictions. Goal matches do not certify geometry.",
        fill="#40516b",
        font=text_font,
    )
    figure.save(output / "preview.png")


def main():
    dataset = ROOT / "data/expanded-grounding"
    raw_root = ROOT / "results/expanded-grounding-raw"
    linked_root = ROOT / "results/expanded-grounding-linked"
    output = ROOT / "results/expanded-grounding-review"
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    records, root = read_records(dataset)
    notes = json.loads((ROOT / "docs/expanded_grounding_notes.json").read_text())
    raw = read_rows(raw_root / "predictions.jsonl")
    linked = read_rows(linked_root / "predictions.jsonl")
    raw_metrics = read_rows(raw_root / "metrics.jsonl")
    linked_metrics = read_rows(linked_root / "metrics.jsonl")
    manifest = json.loads((raw_root / "manifest.json").read_text())
    frozen = json.loads((ROOT / "configs/frozen_expanded_grounding_protocol.json").read_text())
    assert (
        manifest["dataset_sha256"]
        == frozen["dataset_sha256"]
        == hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest()
    )
    assert (
        hashlib.sha256((ROOT / "src/vlm_tamp/evaluation.py").read_bytes()).hexdigest()
        == frozen["evaluator_sha256"]
    )
    assert manifest["offline_mock"] is False
    conditions = manifest["config"]["conditions"]
    expected = {(r["example_id"], c) for r in records for c in conditions}
    assert set(raw) <= expected and set(linked) == set(raw) == set(raw_metrics) == set(linked_metrics)
    assert all(not m["track_b"]["evaluable"] for m in linked_metrics.values())
    observed_models = sorted({r.get("response_model") for r in raw.values() if r.get("response_model")})

    groups = []
    for suite in ["all", *dict.fromkeys(r["metadata"]["suite"] for r in records)]:
        selected = [r for r in records if suite == "all" or r["metadata"]["suite"] == suite]
        for condition in conditions:
            keys = [(r["example_id"], condition) for r in selected]
            roles = [
                (linked_metrics.get(k, {}).get("visual") or {}).get("visible_whole_object_goal_roles") or {}
                for k in keys
            ]
            visuals = [(linked_metrics.get(k, {}).get("visual") or {}) for k in keys]
            groups.append(
                {
                    "suite": suite,
                    "condition": condition,
                    "planned_tasks": len(keys),
                    "evaluated": sum(k in linked_metrics for k in keys),
                    "raw_valid": sum(raw.get(k, {}).get("status") == "ok" for k in keys),
                    "linked_valid": sum(linked.get(k, {}).get("status") == "ok" for k in keys),
                    "raw_complete_goals": sum(matches(raw_metrics.get(k, {})) for k in keys),
                    "linked_complete_goals": sum(matches(linked_metrics.get(k, {})) for k in keys),
                    "linked_predicate_types": sum(
                        matches(linked_metrics.get(k, {}), "predicate_types") for k in keys
                    ),
                    "visible_object_roles_correct": sum(r.get("correct", 0) for r in roles),
                    "visible_object_roles_denominator": sum(r.get("denominator", 0) for r in roles),
                    "points_inside_masks": sum(
                        round(v.get("point_in_mask_accuracy", 0) * v.get("points_evaluable", 0))
                        for v in visuals
                        if v.get("point_in_mask_accuracy") is not None
                    ),
                    "points_evaluable": sum(v.get("points_evaluable", 0) for v in visuals),
                }
            )
    task_results = []
    for record in records:
        item = {
            "example_id": record["example_id"],
            "suite": record["metadata"]["suite"],
            "instruction": record["model_inputs"]["instruction"],
            "reference_goals": record["reference"]["goal_expression"],
            "inspection_note": notes["task_notes"].get(record["example_id"], ""),
            "conditions": {},
        }
        for condition in conditions:
            key = record["example_id"], condition
            row = linked.get(key, {})
            metric = linked_metrics.get(key, {})
            prediction = row.get("prediction") or row.get("unvalidated_prediction") or {}
            reasons = []
            if row.get("status") != "ok":
                reasons.append("Output failed schema or input-bound validation" if row else "Not attempted")
            elif matches(metric):
                reasons.append("Complete benchmark goal match; motion constraints remain unverified")
            else:
                if not matches(metric, "predicate_types"):
                    reasons.append("Different, missing or additional final predicates")
                else:
                    reasons.append("Ordered instance or region arguments differ from the benchmark")
                if not metric.get("terminal_intent_resolved", True):
                    reasons.append("Terminal intent unresolved")
                if metric.get("goal_logic") == "unknown":
                    reasons.append("Logical status unresolved")
            item["conditions"][condition] = {
                "raw_status": raw.get(key, {}).get("status", "not_attempted"),
                "linked_status": row.get("status", "not_attempted"),
                "complete_goal_match": matches(metric),
                "predicate_types_match": matches(metric, "predicate_types"),
                "visible_object_roles": (metric.get("visual") or {}).get("visible_whole_object_goal_roles"),
                "diagnostics": reasons,
                "goals": prediction.get("goals", []),
                "stages": prediction.get("stages", []),
                "constraints": prediction.get("constraints", []),
                "validation_errors": row.get("validation_errors", []),
                "link_events": row.get("link_events", []),
            }
        task_results.append(item)
    summary = {
        "task_count": len(records),
        "planned_predictions": len(expected),
        "completed_predictions": len(raw),
        "observed_response_models": observed_models,
        "groups": groups,
        "tasks": task_results,
        "raw_status_counts": dict(Counter(r["status"] for r in raw.values())),
        "linked_status_counts": dict(Counter(r["status"] for r in linked.values())),
        "new_ledger": manifest["ledger"],
        "cumulative_api_attempts": frozen["previous_api_attempts"] + manifest["ledger"]["api_attempts"],
        "cumulative_estimated_usage_usd": frozen["previous_estimated_usage_usd"]
        + manifest["ledger"]["estimated_usage_cost_usd"],
        "reviewed_constraint_annotations": 0,
        "robot_execution_tested": False,
        "scope": frozen["interpretation"],
        "inspection_notes_scope": notes["scope"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    contact_sheet(dataset, output / "contact-sheet.png", columns=5)

    pieces = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ten new grounding tasks</title>',
        "<style>body{font:16px system-ui;line-height:1.5;background:#111826;color:#edf2ff;max-width:1500px;margin:auto;padding:24px}article,.panel{background:#1b2638;padding:20px;margin:24px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:20px}img{width:100%;max-width:512px;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #42516a;text-align:left}.scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.ok{color:#74dfae}.bad{color:#ffa39d}.note{color:#b4c3dd}a{color:#8bcaff}details{margin:12px 0}h2{font-size:22px}</style></head><body>",
        "<h1>Grounding on ten new tasks</h1>",
        prompt_audit_notice(root),
        f"<p>{escaped(manifest['model'])} · One official initial state per task · {manifest['ledger']['api_attempts']} new calls · Estimated new usage ${manifest['ledger']['estimated_usage_cost_usd']:.6f}</p>",
        "<p>RGB alone is compared with the same RGB plus a simulator-provided object inventory. Inventory results are oracle-assisted. The same prompt, schema and generation settings were used as in the previous v5 trial. These task groups had not appeared in this project’s earlier paid requests.</p>",
        '<p class="note">Goal agreement is measured against official task specifications. Output validity, whole-object identity and complete goals are different measures. All model-generated grasp, contact and path declarations remain unverified; no robot trajectories were tested. Postprocessing only links declared symbols and preserves final goals. A benchmark mismatch does not automatically imply an infeasible physical plan.</p>',
        '<div class="panel scroll"><table><thead><tr><th>Input</th><th>Raw valid</th><th>Linked valid</th><th>Raw complete goals</th><th>Linked complete goals</th><th>Correct visible final-goal object roles</th></tr></thead><tbody>',
    ]
    for group in groups:
        if group["suite"] != "all":
            continue
        n = group["planned_tasks"]
        label = "RGB alone" if group["condition"] == "rgb" else "RGB + provided inventory"
        values = [
            label,
            f"{group['raw_valid']}/{n}",
            f"{group['linked_valid']}/{n}",
            f"{group['raw_complete_goals']}/{n}",
            f"{group['linked_complete_goals']}/{n}",
            f"{group['visible_object_roles_correct']}/{group['visible_object_roles_denominator']}",
        ]
        pieces.append("<tr>" + "".join(f"<td>{escaped(v)}</td>" for v in values) + "</tr>")
    pieces.append(
        '</tbody></table></div><p class="note">Object-role counts require the expected predicate name and ordered argument position; they do not measure identification of every object mentioned in the output. Prediction images below: green boxes are measured whole-object references; magenta boxes and blue points are predictions. The original image contains no annotations. Region masks and verified mechanism geometry were not supplied to either condition.</p>'
    )
    for group in groups:
        if group["suite"] == "all":
            label = "RGB alone" if group["condition"] == "rgb" else "RGB + provided inventory"
            pieces.append(
                f'<p class="note">{label}: {group["points_inside_masks"]}/{group["points_evaluable"]} evaluable predicted image points fall inside their corresponding measured object masks. This is conditional on a resolved identity and a supplied point; missing points are reflected separately in the detailed localization report.</p>'
            )
    for i, (record, result) in enumerate(zip(records, task_results)):
        source_name = f"original-{i:02d}.png"
        shutil.copyfile(safe_path(root, record["model_inputs"]["image"]), assets / source_name)
        pieces.append(
            f'<article id="task-{i}"><h2>{i + 1}. {escaped(result["instruction"])}</h2><p class="note">{escaped(result["example_id"])}</p><p><strong>Inspection note:</strong> {escaped(result["inspection_note"])}</p><div class="grid"><section><h3>Original scene</h3><img src="assets/{source_name}" alt="Original task scene"><details><summary>Official final goals, used only for evaluation</summary><pre>{pretty(result["reference_goals"])}</pre></details></section>'
        )
        for condition in conditions:
            row = linked.get((record["example_id"], condition), {"prediction": None})
            cell = result["conditions"][condition]
            prediction = row.get("prediction") or row.get("unvalidated_prediction") or {}
            filename = f"prediction-{i:02d}-{condition}.png"
            # Invalid outputs are displayed as text, never passed to the overlay renderer.
            overlay(record, {"prediction": row.get("prediction")}, root, assets / filename)
            label = "RGB alone" if condition == "rgb" else "RGB + provided inventory"
            score_label = (
                "Complete goal match" if cell["complete_goal_match"] else "Goal mismatch / unresolved"
            )
            if cell["linked_status"] != "ok":
                score_label = "Invalid output" if row else "Not attempted"
            color = "ok" if cell["complete_goal_match"] else "bad"
            pieces.append(
                f'<section><h3>{label}</h3><img src="assets/{filename}" alt="Scene with measured reference boxes and valid predicted locations"><p class="{color}">{score_label}</p><p>Raw validity: {escaped(cell["raw_status"])} · After linking: {escaped(cell["linked_status"])}</p><p>{escaped("; ".join(cell["diagnostics"]))}</p>'
            )
            entities = {e["id"]: e for e in prediction.get("entities", [])}
            readable = []
            for goal in prediction.get("goals", []):
                args = [
                    entities.get(a, {}).get("reference_id") or entities.get(a, {}).get("description") or a
                    for a in goal.get("arguments", [])
                ]
                readable.append(f"{goal['name']}({', '.join(args)}) [{goal.get('intent', 'unspecified')}]")
            pieces.append(
                f"<p><strong>Declared final goals:</strong><br>{'<br>'.join(escaped(g) for g in readable) or 'None'}</p>"
            )
            for title, value in [
                ("Entities and image coordinates", prediction.get("entities", [])),
                ("Proposed stages — unverified", cell["stages"]),
                ("Proposed constraints — unverified", cell["constraints"]),
                ("Output validation and linking", {k: cell[k] for k in ("validation_errors", "link_events")}),
                ("Exact structured output", prediction),
            ]:
                pieces.append(f"<details><summary>{title}</summary><pre>{pretty(value)}</pre></details>")
            pieces.append("</section>")
        pieces.append("</div></article>")
    pieces.append(
        "<p>Ten purposively selected tasks and one observation each do not establish general accuracy. Task-held-out means new to this project’s query history; model pretraining exposure is unknown.</p></body></html>"
    )
    (output / "index.html").write_text("\n".join(pieces))
    preview(records, assets, output)
    lines = [
        "# Expanded grounding: ten previously unqueried tasks",
        "",
        frozen["interpretation"],
        "",
        "| Suite | Input | Tasks | Raw valid | Linked valid | Raw complete goals | Linked complete goals | Correct visible object roles |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        n = g["planned_tasks"]
        lines.append(
            f"| {g['suite']} | {g['condition']} | {n} | {g['raw_valid']}/{n} | {g['linked_valid']}/{n} | {g['raw_complete_goals']}/{n} | {g['linked_complete_goals']}/{n} | {g['visible_object_roles_correct']}/{g['visible_object_roles_denominator']} |"
        )
    lines.extend(
        [
            "",
            f"New calls: {manifest['ledger']['api_attempts']}; estimated usage: ${manifest['ledger']['estimated_usage_cost_usd']:.8f}. Cumulative calls: {summary['cumulative_api_attempts']}; estimated usage: ${summary['cumulative_estimated_usage_usd']:.8f}.",
            "",
            "Grasp/contact/path correctness is N/A: zero reviewed constraint annotations and no robot execution. All raw failures remain in denominators. The separate linked report does not change final-goal declarations. See index.html for paired scene/prediction panels and summary.json for per-task diagnostics.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "task_count",
                    "completed_predictions",
                    "raw_status_counts",
                    "linked_status_counts",
                    "new_ledger",
                    "cumulative_estimated_usage_usd",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
