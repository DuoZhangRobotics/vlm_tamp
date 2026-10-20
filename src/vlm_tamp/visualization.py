"""Static, escaped HTML galleries with measured image overlays and reports."""

import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .annotations import load_annotations
from .dataset import CONDITIONS, read_records, safe_path
from .references import attach_reference, load_simulator_references


def contact_sheet(dataset, output, columns=4):
    records, root = read_records(dataset)
    tile_w, tile_h = 300, 350
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * tile_w, max(1, rows) * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(records):
        x, y = (i % columns) * tile_w, (i // columns) * tile_h
        image = Image.open(safe_path(root, record["model_inputs"]["image"])).convert("RGB")
        image.thumbnail((290, 290))
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + 294), record["example_id"], fill="black")
        instruction = record["model_inputs"]["instruction"]
        import textwrap

        draw.text((x + 4, y + 308), "\n".join(textwrap.wrap(instruction, 43)[:3]), fill="black")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def project_world(point, camera):
    rotation = np.asarray(camera["world_rotation"]).reshape(3, 3)
    p = rotation.T @ (np.asarray(point) - np.asarray(camera["world_position"]))
    if p[2] >= -1e-6:
        return None
    focal = camera["height"] / (2 * math.tan(math.radians(camera["fovy_degrees"]) / 2))
    return focal * p[0] / (-p[2]) + camera["width"] / 2, -focal * p[1] / (-p[2]) + camera["height"] / 2


def overlay(record, row, root, output):
    image = Image.open(safe_path(root, record["model_inputs"]["image"])).convert("RGB").resize((512, 512))
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for identifier, reference in record["reference"]["visual"].items():
        box = [
            reference["bbox"][0] * w,
            reference["bbox"][1] * h,
            reference["bbox"][2] * w,
            reference["bbox"][3] * h,
        ]
        color = "#ffd66b" if reference.get("kind") == "part" else "#27d171"
        prefix = "REF PART " if reference.get("kind") == "part" else "REF "
        draw.rectangle(box, outline=color, width=1)
        draw.text((box[0], box[1]), prefix + identifier, fill=color, stroke_width=1, stroke_fill="black")
    prediction = row.get("prediction") or {}
    for entity in prediction.get("entities", []):
        if entity.get("camera") not in {None, "agentview"}:
            continue
        if entity.get("bbox"):
            box = [entity["bbox"][0] * w, entity["bbox"][1] * h, entity["bbox"][2] * w, entity["bbox"][3] * h]
            draw.rectangle(box, outline="#ff68df", width=3)
            draw.text(
                (box[0], max(0, box[1] - 12)),
                "PRED " + entity["id"],
                fill="#ff68df",
                stroke_width=1,
                stroke_fill="black",
            )
        if entity.get("point"):
            x, y = entity["point"][0] * w, entity["point"][1] * h
            draw.line((x - 6, y, x + 6, y), fill="#57b8ff", width=3)
            draw.line((x, y - 6, x, y + 6), fill="#57b8ff", width=3)
    # Draw axes only when a world-frame anchor and axis were explicitly supplied.
    for constraint in prediction.get("constraints", []):
        params = {p["name"]: p for p in constraint.get("parameters", [])}
        a, origin = params.get("axis"), params.get("axis_point")
        if (
            a
            and origin
            and a.get("frame") == origin.get("frame") == "world"
            and isinstance(a.get("value"), list)
            and isinstance(origin.get("value"), list)
        ):
            if len(a["value"]) == len(origin["value"]) == 3:
                start = project_world(origin["value"], record["metadata"]["camera"])
                end = project_world(
                    np.asarray(origin["value"]) + 0.1 * np.asarray(a["value"]), record["metadata"]["camera"]
                )
                if start and end:
                    scale = w / record["metadata"]["camera"]["width"]
                    draw.line(tuple(x * scale for x in (*start, *end)), fill="#ffb347", width=4)
    image.save(output)


def write_markdown_report(summary, rows, manifest, output):
    def value(v):
        return "N/A" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    title = (
        "OFFLINE MOCK VALIDATION — not empirical VLM performance"
        if manifest["offline_mock"]
        else "Grounding pipeline with deterministic linking — raw model results preserved"
        if manifest.get("postprocessing")
        else "Small VLM grounding pilot"
    )
    lines = [
        f"# {title}",
        "",
        f"Provider: {manifest['provider']}; model: `{manifest['model']}`.",
        "",
        f"Evaluated {len(rows)} of {manifest['estimate']['planned_predictions']} planned predictions. "
        "Failures are retained in denominators; unsupported BDDL logic is excluded with coverage flags.",
        "",
        "## Table 1: Track A symbolic grounding",
        "",
        "| Model | Suite | Condition | N | Predicate P | R | F1 | Instance goal exact | Predicate type exact | Category goal exact | Object category accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if manifest.get("postprocessing"):
        lines[6:6] = [
            f"Derived from `{manifest['source_experiment']}` using the separately frozen structural linker. Final goal declarations and numerical geometry are unchanged. Conflicting identity claims become abstentions. This run makes no API calls; source inference cost was ${manifest['source_ledger']['estimated_usage_cost_usd']:.8f}.",
            "",
        ]
    for group in summary:
        keys = [
            "model",
            "suite",
            "condition",
            "examples",
            "predicate_precision",
            "predicate_recall",
            "predicate_f1",
            "goal_exact_match",
            "predicate_type_exact_match",
            "category_goal_exact_match",
            "object_category_accuracy",
        ]
        lines.append("| " + " | ".join(value(group[k]) for k in keys) + " |")
    lines += [
        "",
        "Instance goal exact preserves simulator identities and argument order. Category credit never resolves an instance. "
        "Standard inputs cannot name hidden simulator IDs; unique measured mask/box correspondence can resolve objects offline. "
        "High language-only predicate-type accuracy is not evidence of visual grounding.",
        "",
        "## Table 2: Visual grounding",
        "",
        "| Suite | Condition | Visible target denominator | Localized target recall | Box IoU | Boxes evaluable | Point-in-mask | Points evaluable | Correct whole-object goal arguments | Object argument denominator |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary:
        keys = [
            "suite",
            "condition",
            "visible_target_denominator",
            "localized_target_recall",
            "bbox_iou_mean",
            "bbox_evaluable",
            "point_in_mask_accuracy",
            "points_evaluable",
            "visible_whole_object_goal_roles_correct",
            "visible_whole_object_goal_roles_denominator",
        ]
        lines.append("| " + " | ".join(value(group[k]) for k in keys) + " |")
    lines += [
        "",
        "Boxes/masks cover visible whole objects, not drawer parts or arbitrary BDDL regions. "
        "Target recall counts missing localization as failure where a visible reference exists. "
        "Category/instance task-relevance accuracy and manipulable-part accuracy need reviewed entity mappings. "
        "Candidate, geometry and labelled-image conditions are oracle-assisted. Whole-object markers identify neither drawer parts nor arbitrary regions.",
        "",
        "## Table 3: Track B manipulation-constraint pilot",
        "",
        "| Suite | Condition | Reviewed/evaluable observations | Stage recall | Dependency recall | Constraint recall | Constraint precision |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in summary:
        examples = [r for r in rows if r["suite"] == group["suite"] and r["condition"] == group["condition"]]

        def mean_metric(category, metric):
            vals = [e["track_b"].get(category, {}).get(metric) for e in examples if e["track_b"]["evaluable"]]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        lines.append(
            "| "
            + " | ".join(
                value(v)
                for v in [
                    group["suite"],
                    group["condition"],
                    group["track_b_evaluable"],
                    mean_metric("stages", "recall"),
                    mean_metric("dependencies", "recall"),
                    mean_metric("constraints", "recall"),
                    mean_metric("constraints", "precision"),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Draft templates are excluded. Incomplete annotations permit recall and leave unlisted predictions unjudged; "
        "several reviewed alternative partial orders are supported. Simulator joint axes describe kinematics and do not "
        "establish required grasp/contact/path constraints. Expert stage, contact, and articulation annotations remain necessary.",
        "",
        "## Table 4: Cost and latency",
        "",
        "| Model | Suite | Condition | Input tokens | Output tokens | Usage cost USD | Cost/example USD | Mean wall latency s | Cache hits |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary:
        vals = [
            group[k] for k in ("model", "suite", "condition", "input_tokens", "output_tokens", "cost_usd")
        ]
        vals += [group["cost_usd"] / group["examples"], group["latency_seconds_mean"], group["cached"]]
        lines.append("| " + " | ".join(value(v) for v in vals) + " |")
    fact_lines = [
        "",
        "Verified simulator articulation facts (separate from expert task-plan labels):",
        "",
        "| Suite | Condition | Observations with facts | Goal parts | Identified goal parts | Known type declarations | Unknown type declarations | Matched joint-type claims | Correct joint-type claims |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary:
        fact_lines.append(
            "| "
            + " | ".join(
                value(group.get(k))
                for k in (
                    "suite",
                    "condition",
                    "simulator_fact_examples",
                    "reference_goal_parts",
                    "identified_goal_parts",
                    "explicit_known_joint_types",
                    "explicit_unknown_joint_types",
                    "matched_joint_type_claims",
                    "correct_joint_type_claims",
                )
            )
            + " |"
        )
    fact_lines += [
        "",
        "A named drawer-opening action is not a joint-type prediction. Exact declared prismatic/revolute aliases are scored; missing type claims remain missing. Part masks cover visible articulated bodies, not interior volumes. Joint types/axes were checked against official site/body mappings and Open-predicate perturbations. These facts do not verify required stages, grasp or contact strategies.",
        "",
    ]
    # Insert fact coverage into Table 3, before the cost table.
    cost_start = lines.index("## Table 4: Cost and latency")
    lines[cost_start:cost_start] = fact_lines
    ledger = manifest.get("ledger", {})
    lines += [
        "",
        f"API attempts: {ledger.get('api_attempts', 0)}. Estimated usage cost: "
        f"${ledger.get('estimated_usage_cost_usd', 0):.6f}. Unknown-usage reservation: "
        f"${ledger.get('unknown_usage_reserved_usd', 0):.6f}. Cached responses incur zero new API cost. "
        "Configured prices are applied to reported usage; provider invoice is authoritative.",
        "",
        "## Reliability and qualitative analysis",
        "",
    ]
    totals = {k: sum(r["reliability"][k] for r in rows) for k in rows[0]["reliability"]} if rows else {}
    lines += [
        "```json",
        json.dumps(totals, indent=2),
        "```",
        "",
        "Open `gallery.html` to inspect goals, entity localization, stages, constraints and errors for every prediction. "
        "Green boxes are simulator references, magenta boxes and blue points are predictions. "
        "Model confidence is not treated as calibrated probability.",
        "",
        "## Experimental limits",
        "",
        "This pilot is too small for conclusions about vision improvements or backbone rankings. "
        "Repeated initial states are not independent tasks. A larger study should aggregate by task and use paired comparisons "
        "with uncertainty intervals; oracle-assisted conditions must remain separate. Stage 1 produces semantic declarations "
        "for later verification; it neither compiles constraints nor integrates ST-pRRTC.",
    ]
    Path(output).write_text("\n".join(lines) + "\n")


def gallery(dataset, experiment, output=None, annotations=None, simulator_references=None):
    records, root = read_records(dataset)
    by_id = {r["example_id"]: r for r in records}
    experiment = Path(experiment).resolve()
    output = Path(output or experiment).resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions = [
        json.loads(line) for line in (experiment / "predictions.jsonl").read_text().splitlines() if line
    ]
    metrics = {
        (r["example_id"], r["condition"]): r
        for r in [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line]
    }
    summary = json.loads((output / "summary.json").read_text())["groups"]
    manifest = json.loads((experiment / "manifest.json").read_text())
    annotation_map = load_annotations(annotations)
    simulator_map = load_simulator_references(simulator_references)
    marker_samples = {}
    if "markers" in manifest["config"]["conditions"]:
        import hashlib
        from .markers import load_marker_bundle

        marker_path = Path(manifest["config"]["marker_bundle"])
        if hashlib.sha256(marker_path.read_bytes()).hexdigest() != manifest["config"]["marker_bundle_sha256"]:
            raise ValueError("Marker bundle changed since inference")
        selected = {r["example_id"] for r in predictions if r["condition"] == "markers"}
        marker_samples = load_marker_bundle(
            marker_path, dataset, [r for r in records if r["example_id"] in selected]
        )
    assets = output / "gallery-assets"
    assets.mkdir(exist_ok=True)
    heading = "OFFLINE MOCK — pipeline validation only" if manifest["offline_mock"] else "VLM grounding pilot"
    if manifest.get("postprocessing") and not manifest["offline_mock"]:
        heading = "Grounding pipeline with deterministic linking — raw model results preserved"
    pieces = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Grounding evaluation</title><style>body{font:16px system-ui;background:#111826;color:#edf2ff;max-width:1400px;margin:auto;padding:24px} "
        "article{background:#1b2638;padding:18px;margin:24px 0;border-radius:12px}img{width:min(100%,512px)} "
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px} "
        ".legend{color:#a7b8d8}a{color:#70b7ff}details{border-top:1px solid #394761;padding:12px 0}</style>",
        f'<body><h1>{heading}</h1><p>{html.escape(manifest["model"])}</p><p class="legend">'
        "Green: whole-object references · Gold: verified part references · Magenta: predicted boxes · Blue: predicted points · Orange: world-frame predicted axes</p>",
    ]
    for i, row in enumerate(predictions):
        record = by_id[row["example_id"]]
        if row["example_id"] in simulator_map:
            record = attach_reference(record, root, simulator_map[row["example_id"]])
        filename = f"example-{i:04d}.png"
        overlay(record, row, root, assets / filename)
        metric = metrics[(row["example_id"], row["condition"])]
        symbolic = metric["symbolic"]
        label = "GOAL MATCH" if symbolic and symbolic["exact_match"] else "INSPECT ERROR / UNRESOLVED"
        pieces.append(
            f"<article><h2>{html.escape(row['example_id'])} · {html.escape(CONDITIONS[row['condition']])}</h2>"
            f"<p>{html.escape(record['model_inputs']['instruction'])}</p><p>{label} · {html.escape(row['status'])}</p>"
            f'<div class="grid"><img src="gallery-assets/{filename}" alt="RGB observation with grounding overlays"><div>'
        )
        if row["condition"] == "markers":
            marker_name = f"model-input-{i:04d}.png"
            sample = marker_samples[row["example_id"]]
            if sample["image_sha256"] != row["image_sha256"]:
                raise ValueError("Gallery marker image differs from the actual model input")
            (assets / marker_name).write_bytes(sample["image"])
            pieces.append(
                f"<p>Actual model-visible oracle-labelled input (same image dimensions):</p>"
                f'<img src="gallery-assets/{marker_name}" alt="Actual simulator-assisted labelled input">'
            )
        displays = [
            ("Reference predicates", record["reference"]["goal_expression"]),
            ("Predicted predicates", (row.get("prediction") or {}).get("goals", [])),
            ("Predicted entities", (row.get("prediction") or {}).get("entities", [])),
            ("Separate geometric targets", (row.get("prediction") or {}).get("geometry_targets", [])),
            ("Predicted stages", (row.get("prediction") or {}).get("stages", [])),
            ("Predicted dependencies", (row.get("prediction") or {}).get("dependencies", [])),
            ("Predicted constraints", (row.get("prediction") or {}).get("constraints", [])),
            (
                "Verified simulator articulation facts",
                simulator_map[row["example_id"]].model_dump(mode="json")
                if row["example_id"] in simulator_map
                else "No verified simulator part reference",
            ),
            (
                "Pre-validation linked payload and errors"
                if row.get("derived")
                else "Unvalidated raw prediction and schema errors",
                {"prediction": row.get("unvalidated_prediction"), "errors": row.get("validation_errors")},
            ),
            (
                "Reference constraints",
                annotation_map[row["example_id"]].model_dump(mode="json")
                if row["example_id"] in annotation_map
                else "No reviewed constraint annotation",
            ),
            ("Evaluation", metric),
        ]
        if row.get("derived"):
            displays.insert(
                0,
                (
                    "Deterministic linking provenance",
                    {
                        "source_status": row["source_status"],
                        "link_events": row.get("link_events", []),
                        "source_raw_response": row.get("source_raw_response"),
                        "source_cost_usd": row["source_cost_usd"],
                        "final_goal_declarations_changed": False,
                    },
                ),
            )
        for j, (title, content) in enumerate(displays):
            pieces.append(
                f"<details {'open' if j < 2 else ''}><summary>{title}</summary><pre>{html.escape(json.dumps(content, indent=2))}</pre></details>"
            )
        pieces.append("</div></div></article>")
    pieces.append("</body></html>")
    (output / "gallery.html").write_text("\n".join(pieces))
    write_markdown_report(summary, list(metrics.values()), manifest, output / "report.md")
    return output / "gallery.html"
