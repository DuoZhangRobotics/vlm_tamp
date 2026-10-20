"""Compare saved Mini-none, Mini-medium and Astra-medium results without model calls."""

import hashlib
import json
import os
import shutil
from collections import Counter

from report_expanded_grounding import escaped, matches, pretty, prompt_audit_notice, read_rows
from vlm_tamp.dataset import read_records, safe_path
from vlm_tamp.libero import ROOT
from vlm_tamp.visualization import overlay


ARMS = [
    ("mini-none", "Mini · reasoning off", "expanded-grounding", False),
    ("mini-medium", "Mini · medium reasoning", "reasoning-mini-medium", True),
    ("astra-medium", "GPT-6 Astra · medium reasoning", "reasoning-astra-medium", True),
]


def explicit_final_ids(row):
    prediction = row.get("prediction") or {}
    entities = {e["id"]: e for e in prediction.get("entities", [])}
    return bool(prediction.get("goals")) and all(
        entities.get(argument, {}).get("reference_id") is not None
        for goal in prediction["goals"]
        for argument in goal["arguments"]
    )


def draw_chart(groups, output):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(10.8, 4.1), constrained_layout=True)
    for offset, condition, label, color in [
        (-0.18, "rgb", "RGB alone", "#5699dd"),
        (0.18, "candidates", "RGB + simulator inventory", "#31a883"),
    ]:
        counts = [
            next(
                g["linked_complete_goals"] for g in groups if g["arm"] == arm and g["condition"] == condition
            )
            for arm, _, _, _ in ARMS
        ]
        bars = ax.bar(x + offset, counts, 0.34, color=color, label=label)
        ax.bar_label(bars, labels=[f"{c}/10" for c in counts], padding=4)
    ax.set_xticks(x, ["Mini\nreasoning off", "Mini\nmedium reasoning", "GPT-6 Astra\nmedium reasoning"])
    ax.set_ylim(0, 11)
    ax.set_yticks(range(0, 11, 2))
    ax.set_ylabel("Exact complete-goal matches")
    ax.set_title("Same ten diagnostic tasks, one response per task and input condition")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    fig.savefig(output / "goal-comparison.png", dpi=150)
    plt.close(fig)


def main():
    dataset = ROOT / "data/expanded-grounding"
    records, root = read_records(dataset)
    protocol = json.loads((ROOT / "configs/frozen_reasoning_comparison_protocol.json").read_text())
    assert hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest() == protocol["dataset_sha256"]
    output = ROOT / "results/reasoning-comparison"
    output.mkdir(exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    runs = {}
    groups = []
    new_cost, new_calls = 0.0, 0
    conditions = ["rgb", "candidates"]
    baseline_metrics = read_rows(ROOT / "results/expanded-grounding-linked/metrics.jsonl")
    for arm, label, stem, is_new in ARMS:
        raw_root = ROOT / f"results/{stem}-raw"
        linked_root = ROOT / f"results/{stem}-linked"
        manifest = json.loads((raw_root / "manifest.json").read_text())
        raw = read_rows(raw_root / "predictions.jsonl")
        linked = read_rows(linked_root / "predictions.jsonl")
        metrics = read_rows(linked_root / "metrics.jsonl")
        raw_metrics = read_rows(raw_root / "metrics.jsonl")
        assert not manifest["offline_mock"] and manifest["dataset_sha256"] == protocol["dataset_sha256"]
        assert not any(m["track_b"]["evaluable"] for m in metrics.values())
        assert set(raw) == set(linked) == set(metrics) == set(raw_metrics)
        if is_new:
            new_cost += manifest["ledger"]["estimated_usage_cost_usd"]
            new_calls += manifest["ledger"]["api_attempts"]
        runs[arm] = {
            "label": label,
            "stem": stem,
            "manifest": manifest,
            "raw": raw,
            "linked": linked,
            "metrics": metrics,
        }
        for condition in conditions:
            keys = [(r["example_id"], condition) for r in records]
            selected = [metrics[k] for k in keys if k in metrics]
            raw_selected = [raw[k] for k in keys if k in raw]
            roles = [(m.get("visual") or {}).get("visible_whole_object_goal_roles") or {} for m in selected]
            visuals = [m.get("visual") or {} for m in selected]
            group = {
                "arm": arm,
                "label": label,
                "model": manifest["model"],
                "condition": condition,
                "planned_tasks": len(keys),
                "evaluated": len(selected),
                "api_attempts": sum(r["attempts"] for r in raw_selected),
                "raw_valid": sum(r["status"] == "ok" for r in raw_selected),
                "linked_valid": sum(m["status"] == "ok" for m in selected),
                "raw_complete_goals": sum(matches(raw_metrics.get(k, {})) for k in keys),
                "linked_complete_goals": sum(matches(m) for m in selected),
                "complete_goals_with_explicit_final_ids": sum(
                    matches(metrics.get(k, {})) and explicit_final_ids(linked.get(k, {})) for k in keys
                ),
                "predicate_types_match": sum(matches(m, "predicate_types") for m in selected),
                "visible_object_roles_correct": sum(r.get("correct", 0) for r in roles),
                "visible_object_roles_denominator": sum(
                    (
                        (baseline_metrics[k].get("visual") or {}).get("visible_whole_object_goal_roles") or {}
                    ).get("denominator", 0)
                    for k in keys
                ),
                "points_inside_masks": sum(
                    round(v["point_in_mask_accuracy"] * v["points_evaluable"])
                    for v in visuals
                    if v.get("point_in_mask_accuracy") is not None
                ),
                "points_evaluable": sum(v.get("points_evaluable", 0) for v in visuals),
                "cost_usd": sum(r["cost_usd"] for r in raw_selected),
                "mean_latency_seconds": sum(r["latency_seconds"] for r in raw_selected) / len(raw_selected)
                if raw_selected
                else None,
                "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in raw_selected),
            }
            groups.append(group)
    tasks = []
    for record in records:
        item = {
            "example_id": record["example_id"],
            "instruction": record["model_inputs"]["instruction"],
            "reference_goals": record["reference"]["goal_expression"],
            "conditions": {},
        }
        for condition in conditions:
            item["conditions"][condition] = {}
            key = record["example_id"], condition
            for arm, _, _, _ in ARMS:
                run = runs[arm]
                row = run["linked"].get(key, {})
                metric = run["metrics"].get(key, {})
                pred = row.get("prediction") or row.get("unvalidated_prediction") or {}
                item["conditions"][condition][arm] = {
                    "raw_status": run["raw"].get(key, {}).get("status", "not_attempted"),
                    "linked_status": row.get("status", "not_attempted"),
                    "goal_match": matches(metric),
                    "explicit_final_ids": explicit_final_ids(row),
                    "predicate_types_match": matches(metric, "predicate_types"),
                    "goals": pred.get("goals", []),
                    "entities": pred.get("entities", []),
                    "stages": pred.get("stages", []),
                    "constraints": pred.get("constraints", []),
                    "validation_errors": row.get("validation_errors", []),
                    "reasoning_tokens": run["raw"].get(key, {}).get("reasoning_tokens"),
                    "response_status": run["raw"].get(key, {}).get("response_status"),
                    "incomplete_details": run["raw"].get(key, {}).get("incomplete_details"),
                }
        tasks.append(item)
    summary = {
        "groups": groups,
        "tasks": tasks,
        "new_api_attempts": new_calls,
        "new_estimated_usage_usd": new_cost,
        "cumulative_api_attempts": protocol["previous_api_attempts"] + new_calls,
        "cumulative_estimated_usage_usd": protocol["previous_estimated_usage_usd"] + new_cost,
        "baseline_comparison_limit": protocol["baseline_comparison_limit"],
        "interpretation": protocol["interpretation"],
        "human_reviewed_constraints": 0,
        "robot_execution_tested": False,
        "arms": {
            arm: {
                "ledger": r["manifest"]["ledger"],
                "response_models": sorted(
                    {p["response_model"] for p in r["raw"].values() if p.get("response_model")}
                ),
                "raw_status_counts": dict(Counter(p["status"] for p in r["raw"].values())),
            }
            for arm, r in runs.items()
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    draw_chart(groups, output)
    pieces = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reasoning and model comparison</title>',
        "<style>body{font:16px system-ui;line-height:1.5;background:#111826;color:#edf2ff;max-width:1650px;margin:auto;padding:24px}article,.panel{background:#1b2638;padding:20px;margin:24px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:20px}img{width:100%;height:auto;max-width:512px}.chart{max-width:1100px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #42516a;text-align:left}.scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.ok{color:#74dfae}.bad{color:#ffa39d}.note{color:#b4c3dd}details{margin:12px 0}</style></head><body>",
        "<h1>Does reasoning or a stronger model improve grounding?</h1>",
        prompt_audit_notice(root),
        f"<p>Same ten diagnostic tasks, same observations, prompt, schema, structural linker and scoring. {new_calls} new calls; estimated new usage ${new_cost:.6f}. The saved Mini baseline is reused.</p>",
        f'<p class="note">{escaped(protocol["baseline_comparison_limit"])}</p>',
        '<p class="note">The Mini-medium versus Astra-medium comparison matches generation settings. Inventory is a simulator-provided list of scene object/region IDs, categories, kinds and ownership; it supplies no task goal answers or numerical geometry. These ten tasks have been inspected previously and are not a fresh held-out set. Exact benchmark-goal agreement is separate from physical executability; all grasp/contact/path declarations remain unverified.</p>',
        '<img class="chart" src="goal-comparison.png" alt="Exact complete-goal counts for three configurations on two input conditions">',
        '<div class="panel scroll"><table><thead><tr><th>Configuration</th><th>Input</th><th>Raw valid</th><th>Linked valid</th><th>Complete goals</th><th>Correct visible goal-object roles</th><th>Points inside masks</th><th>Estimated usage</th></tr></thead><tbody>',
    ]
    for g in groups:
        input_name = "RGB alone" if g["condition"] == "rgb" else "RGB + inventory"
        values = [
            g["label"],
            input_name,
            f"{g['raw_valid']}/10",
            f"{g['linked_valid']}/10",
            f"{g['linked_complete_goals']}/10",
            f"{g['visible_object_roles_correct']}/{g['visible_object_roles_denominator']}",
            f"{g['points_inside_masks']}/{g['points_evaluable']}",
            f"${g['cost_usd']:.6f}",
        ]
        pieces.append("<tr>" + "".join(f"<td>{escaped(v)}</td>" for v in values) + "</tr>")
    pieces.append(
        '</tbody></table></div><p class="note">Failures and unattempted cases remain in planned-goal denominators. Object-role identity requires the expected predicate and ordered argument; point accuracy is conditional on a resolved identity and a supplied image point. Green boxes: measured whole-object references. Magenta boxes and blue points: valid model predictions. For image-based outputs, the evaluator can independently resolve image points/boxes against simulator masks; it does not write those IDs into model outputs. A runtime perception/scene-binding layer is still required.</p>'
    )
    for g in groups:
        if g["condition"] == "candidates":
            pieces.append(
                f'<p class="note">{escaped(g["label"])}: {g["complete_goals_with_explicit_final_ids"]} of {g["linked_complete_goals"]} scored inventory-assisted complete goals have every final argument explicitly bound to a supplied ID in the output. Other successful scores depend on the independent image correspondence evaluator.</p>'
            )
    for i, (record, item) in enumerate(zip(records, tasks)):
        filename = f"original-{i:02d}.png"
        shutil.copyfile(safe_path(root, record["model_inputs"]["image"]), assets / filename)
        for condition in conditions:
            input_name = "RGB alone" if condition == "rgb" else "RGB + provided inventory"
            pieces.append(
                f'<article id="task-{i}-{condition}"><h2>{i + 1}. {escaped(item["instruction"])} · {input_name}</h2><div class="grid"><section><h3>Original scene</h3><img src="assets/{filename}" alt="Original robot task scene"><details><summary>Official final goals — evaluation only</summary><pre>{pretty(item["reference_goals"])}</pre></details></section>'
            )
            for arm, label, _, _ in ARMS:
                cell = item["conditions"][condition][arm]
                row = runs[arm]["linked"].get((record["example_id"], condition), {})
                image_name = f"prediction-{i:02d}-{condition}-{arm}.png"
                overlay(record, {"prediction": row.get("prediction")}, root, assets / image_name)
                status = "Complete goal match" if cell["goal_match"] else "Goal mismatch / unresolved"
                if cell["linked_status"] != "ok":
                    status = "Output not accepted: " + cell["linked_status"]
                color = "ok" if cell["goal_match"] else "bad"
                pieces.append(
                    f'<section><h3>{escaped(label)}</h3><img src="assets/{image_name}" alt="Measured reference boxes and valid model predicted locations"><p class="{color}">{escaped(status)}</p><p>Raw: {escaped(cell["raw_status"])} · Linked: {escaped(cell["linked_status"])}</p><details open><summary>Declared final goals</summary><pre>{pretty(cell["goals"])}</pre></details>'
                )
                for title, value in [
                    ("Entity IDs and image coordinates", cell["entities"]),
                    ("Proposed stages — unverified", cell["stages"]),
                    ("Proposed constraints — unverified", cell["constraints"]),
                    (
                        "Validation and response completion",
                        {
                            k: cell[k]
                            for k in (
                                "validation_errors",
                                "reasoning_tokens",
                                "response_status",
                                "incomplete_details",
                            )
                        },
                    ),
                ]:
                    pieces.append(f"<details><summary>{title}</summary><pre>{pretty(value)}</pre></details>")
                pieces.append("</section>")
            pieces.append("</div></article>")
    pieces.append("</body></html>")
    (output / "index.html").write_text("\n".join(pieces))
    lines = [
        "# Reasoning and stronger-model comparison",
        "",
        protocol["interpretation"],
        "",
        protocol["baseline_comparison_limit"],
        "",
        "| Configuration | Input | Raw valid | Linked valid | Complete goals | Correct goal-object roles | Points inside masks |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        lines.append(
            f"| {g['label']} | {g['condition']} | {g['raw_valid']}/10 | {g['linked_valid']}/10 | {g['linked_complete_goals']}/10 | {g['visible_object_roles_correct']}/{g['visible_object_roles_denominator']} | {g['points_inside_masks']}/{g['points_evaluable']} |"
        )
    lines += [
        "",
        f"New usage: {new_calls} calls / ${new_cost:.8f}. Cumulative usage: {summary['cumulative_api_attempts']} calls / ${summary['cumulative_estimated_usage_usd']:.8f}.",
        "",
        "All generated constraints remain unverified. No human-reviewed stage/contact/path labels or robot trajectories were added.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps({"new_calls": new_calls, "new_estimated_usage_usd": new_cost, "groups": groups}, indent=2)
    )


if __name__ == "__main__":
    main()
