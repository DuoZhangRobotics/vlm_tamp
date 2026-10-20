"""Paired descriptive comparisons on identical observations and scoring rules."""

import json
from pathlib import Path


def compare_experiments(baseline, updated, output, controlled_backbone=False):
    def load(path):
        root = Path(path).resolve()
        evaluation = json.loads((root / "evaluation_manifest.json").read_text())
        inference = json.loads((Path(evaluation["experiment"]) / "manifest.json").read_text())
        metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line]
        predictions = [
            json.loads(line)
            for line in (Path(evaluation["experiment"]) / "predictions.jsonl").read_text().splitlines()
            if line
        ]
        return (
            evaluation,
            inference,
            {(r["example_id"], r["condition"]): r for r in metrics},
            {(r["example_id"], r["condition"]): r for r in predictions},
        )

    be, bm, before, bp = load(baseline)
    ae, am, after, ap = load(updated)
    for field in (
        "evaluation_version",
        "evaluator_sha256",
        "dataset_sha256",
        "simulator_references_sha256",
        "expert_annotations_sha256",
    ):
        if be[field] != ae[field]:
            raise ValueError(f"Paired comparison requires identical {field}")
    if bm["offline_mock"] != am["offline_mock"]:
        raise ValueError("Do not compare mock results as empirical model performance")
    if controlled_backbone:
        if bm.get("postprocessing") != am.get("postprocessing"):
            raise ValueError("Controlled backbone comparison requires the same postprocessing protocol")
        if (
            bm["config"]["prompt_sha256"] != am["config"]["prompt_sha256"]
            or bm["prompt_text"] != am["prompt_text"]
        ):
            raise ValueError("Controlled backbone comparison requires the same frozen prompt")
        if bm.get("output_schema") != am.get("output_schema"):
            raise ValueError("Controlled backbone comparison requires the same output schema")
        for field, default in (
            ("temperature", None),
            ("image_detail", "high"),
            ("max_output_tokens", None),
            ("seed", 0),
        ):
            if bm["config"].get(field, default) != am["config"].get(field, default):
                raise ValueError(f"Controlled backbone comparison requires identical {field}")
        efforts = [m["config"].get("reasoning_effort") or "none" for m in (bm, am)]
        if efforts[0] != efforts[1]:
            raise ValueError("Controlled backbone comparison requires matched reasoning settings")
    common = sorted(set(before) & set(after))
    if not common:
        raise ValueError("No common example-condition pairs")
    for key in common:
        if bp[key]["image_sha256"] != ap[key]["image_sha256"]:
            raise ValueError("Matched observations have different image content")
        if (
            bp[key].get("payload_sha256")
            and ap[key].get("payload_sha256")
            and bp[key]["payload_sha256"] != ap[key]["payload_sha256"]
        ):
            raise ValueError("Matched observations have different model-visible payloads")
    if controlled_backbone and set(before) != set(after):
        raise ValueError(
            "Controlled backbone comparison requires complete matching example-condition coverage"
        )
    conditions = sorted({condition for _, condition in common})
    rows = []
    metrics = {
        "schema_valid": lambda r: r["status"] == "ok",
        "predicate_type_exact": lambda r: (
            r["predicate_types"]["exact_match"] if r["predicate_types"] else None
        ),
        "category_goal_exact": lambda r: r["category_goals"]["exact_match"] if r["category_goals"] else None,
        "instance_goal_exact": lambda r: r["symbolic"]["exact_match"] if r["symbolic"] else None,
        "predicate_content_exact": lambda r: (
            r.get("symbolic_content", {}).get("exact_match") if r.get("symbolic_content") else None
        ),
        "conjunctive_logic_declared": lambda r: r["status"] == "ok" and r.get("goal_logic") == "and",
    }
    same_generation = all(
        bm["config"].get(field) == am["config"].get(field)
        for field in ("temperature", "image_detail", "max_output_tokens", "seed", "reasoning_effort")
    )
    prompt_only = (
        bm["model"] == am["model"]
        and bm["provider"] == am["provider"]
        and bm.get("output_schema") == am.get("output_schema")
        and same_generation
        and bm["config"]["prompt_sha256"] != am["config"]["prompt_sha256"]
        and bm.get("postprocessing") == am.get("postprocessing")
    )
    changes = []
    for condition in conditions:
        keys = [key for key in common if key[1] == condition]
        entry = {"condition": condition, "pairs": len(keys)}
        for name, measure in metrics.items():
            pairs = [(measure(before[k]), measure(after[k])) for k in keys]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            entry[name] = {
                "evaluable_pairs": len(pairs),
                "baseline_correct": sum(x for x, _ in pairs),
                "updated_correct": sum(y for _, y in pairs),
                "improved": sum(not x and y for x, y in pairs),
                "regressed": sum(x and not y for x, y in pairs),
            }
        rows.append(entry)
        for key in keys:
            changes.append(
                {
                    "example_id": key[0],
                    "condition": condition,
                    "baseline_status": before[key]["status"],
                    "updated_status": after[key]["status"],
                    "baseline_goal_exact": before[key]["symbolic"]["exact_match"]
                    if before[key]["symbolic"]
                    else None,
                    "updated_goal_exact": after[key]["symbolic"]["exact_match"]
                    if after[key]["symbolic"]
                    else None,
                }
            )
    report = {
        "controlled_backbone": controlled_backbone,
        "offline_mock": bm["offline_mock"],
        "evaluation_version": be["evaluation_version"],
        "baseline_model": bm["model"],
        "updated_model": am["model"],
        "baseline_prompt_sha256": bm["config"]["prompt_sha256"],
        "updated_prompt_sha256": am["config"]["prompt_sha256"],
        "schema_changed": bm.get("output_schema") != am.get("output_schema"),
        "postprocessing_changed": bm.get("postprocessing") != am.get("postprocessing"),
        "prompt_only_configuration_change": prompt_only,
        "paired_predictions": len(common),
        "groups": rows,
        "changes": changes,
        "unpaired_baseline": len(set(before) - set(after)),
        "unpaired_updated": len(set(after) - set(before)),
        "baseline_usage_cost_usd": bm["ledger"]["estimated_usage_cost_usd"],
        "updated_usage_cost_usd": am["ledger"]["estimated_usage_cost_usd"],
        "limitation": (
            "Frozen prompt, schema, observations and generation settings; model-specific reasoning support is recorded. These descriptive counts on a small fixed task set do not establish a general model ranking. Oracle-assisted settings remain separate."
            if controlled_backbone
            else "Same model, output schema, observations and generation settings; the prompt wording changed. These are descriptive paired counts from one response per case, without replication. Oracle-assisted settings remain separate."
            if prompt_only
            else "Descriptive counts with changed structural postprocessing. Raw model outputs and linked pipeline outputs are distinct; this is not a prompt-only or backbone-only comparison. No new model usage is charged by deterministic linking."
            if bm.get("postprocessing") != am.get("postprocessing")
            else "Descriptive counts on a small fixed subset. Prompt and output-schema changes are coupled, so this is not a prompt-only causal ablation. Oracle-assisted settings remain separate."
        ),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# OFFLINE MOCK comparison"
        if bm["offline_mock"]
        else "# Controlled backbone comparison"
        if controlled_backbone
        else "# Paired grounding comparison",
        "",
        f"Baseline: `{bm['model']}`. Comparison: `{am['model']}`.",
        "",
        report["limitation"],
        "",
        "| Condition | Pairs | Schema valid baseline → comparison | Predicate type | Category goal | Strict instance goal | Predicate content (diagnostic) | Conjunctive logic declared |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cells = [row["condition"], str(row["pairs"])]
        for metric in metrics:
            entry = row[metric]
            cells.append(
                f"{entry['baseline_correct']}/{entry['evaluable_pairs']} → {entry['updated_correct']}/{entry['evaluable_pairs']}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    cost = report["baseline_usage_cost_usd"] + report["updated_usage_cost_usd"]
    lines += [
        "",
        f"Combined estimated usage cost: **${cost:.6f}**. Updated-only predictions: {report['unpaired_updated']} (excluded from paired counts).",
        "",
        "No confidence intervals or vision-benefit claim are warranted by four tasks. Failures remain in the paired denominators. See each run's gallery for localization and constraint details.",
        "Predicate-content agreement is an exploratory diagnostic introduced during error analysis. It ignores the model's goal_logic uncertainty flag while retaining predicate names and exact ordered instance arguments. Strict instance-goal scores remain unchanged and do not give credit for unresolved logical status.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


def compare_conditions(experiment, output):
    """Pair inventory-only RGB with the all-object marker oracle in one run."""
    root = Path(experiment).resolve()
    evaluation = json.loads((root / "evaluation_manifest.json").read_text())
    inference = Path(evaluation["experiment"])
    manifest = json.loads((inference / "manifest.json").read_text())
    metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line]
    predictions = {
        (r["example_id"], r["condition"]): r
        for r in (
            json.loads(line) for line in (inference / "predictions.jsonl").read_text().splitlines() if line
        )
    }
    before = {r["example_id"]: r for r in metrics if r["condition"] == "candidates"}
    after = {r["example_id"]: r for r in metrics if r["condition"] == "markers"}
    if not before or set(before) != set(after) or len(before) != manifest["estimate"]["planned_examples"]:
        raise ValueError("Marker comparison requires complete matching inventory/marker coverage")
    changes = []
    for eid in sorted(before):
        plain, marked = predictions[eid, "candidates"], predictions[eid, "markers"]
        if not plain.get("source_image_sha256") or plain["source_image_sha256"] != marked.get(
            "source_image_sha256"
        ):
            raise ValueError("Marker comparison requires identical original RGB observations")
        if not plain.get("candidate_inventory_sha256") or plain["candidate_inventory_sha256"] != marked.get(
            "candidate_inventory_sha256"
        ):
            raise ValueError("Marker comparison requires identical full candidate inventories")
        if not marked.get("input_provenance") or marked["input_provenance"].get(
            "marker_bundle_sha256"
        ) != manifest["config"].get("marker_bundle_sha256"):
            raise ValueError("Marker comparison requires recorded oracle input provenance")
        a, b = before[eid], after[eid]
        ar, br = (r["visual"]["visible_whole_object_goal_roles"] for r in (a, b))
        if ar["denominator"] != br["denominator"]:
            raise ValueError("Marker comparison object-role coverage differs")
        changes.append(
            {
                "example_id": eid,
                "baseline_status": a["status"],
                "assisted_status": b["status"],
                "baseline_goal_exact": a["symbolic"]["exact_match"] if a["symbolic"] else None,
                "assisted_goal_exact": b["symbolic"]["exact_match"] if b["symbolic"] else None,
                "baseline_object_roles_correct": ar["correct"],
                "assisted_object_roles_correct": br["correct"],
                "object_role_denominator": ar["denominator"],
            }
        )
    report = {
        "model": manifest["model"],
        "offline_mock": manifest["offline_mock"],
        "evaluation_version": evaluation["evaluation_version"],
        "pairs": len(changes),
        "baseline_condition": "candidates",
        "assisted_condition": "markers",
        "baseline_goal_exact": sum(c["baseline_goal_exact"] is True for c in changes),
        "assisted_goal_exact": sum(c["assisted_goal_exact"] is True for c in changes),
        "baseline_object_roles_correct": sum(c["baseline_object_roles_correct"] for c in changes),
        "assisted_object_roles_correct": sum(c["assisted_object_roles_correct"] for c in changes),
        "object_role_denominator": sum(c["object_role_denominator"] for c in changes),
        "changes": changes,
        "limitation": "Simulator masks supply all visible whole-object identity correspondences, including distractors. This measures the benefit of that assistance, not unaided RGB ability. Regions/parts are unmarked. Object-role identity is separate from logical status and strict complete-goal correctness. Four cases and one response per case cannot establish a general effect.",
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# OFFLINE MOCK marker comparison"
        if report["offline_mock"]
        else "# Oracle-assisted object-correspondence comparison",
        "",
        f"Model: `{report['model']}`. Paired observations: {report['pairs']}.",
        "",
        report["limitation"],
        "",
        "| Observation | Schema valid inventory / markers | Strict goal inventory / markers | Correct visible whole-object goal arguments inventory / markers |",
        "|---|---|---|---|",
    ]
    for c in changes:
        lines.append(
            f"| {c['example_id']} | {c['baseline_status']} / {c['assisted_status']} | "
            f"{c['baseline_goal_exact']} / {c['assisted_goal_exact']} | "
            f"{c['baseline_object_roles_correct']}/{c['object_role_denominator']} / "
            f"{c['assisted_object_roles_correct']}/{c['object_role_denominator']} |"
        )
    lines += [
        "",
        f"Strict complete goals: **{report['baseline_goal_exact']}/{report['pairs']} → {report['assisted_goal_exact']}/{report['pairs']}**. "
        f"Visible whole-object argument identity: **{report['baseline_object_roles_correct']}/{report['object_role_denominator']} → {report['assisted_object_roles_correct']}/{report['object_role_denominator']}**.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    return report
