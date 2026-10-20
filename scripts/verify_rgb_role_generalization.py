"""Verify prospective task disjointness, raw RGB requests, scores and costs."""

import base64
import inspect
import json
from collections import Counter

from verify_rgb_role_points import PageParser
from vlm_tamp.dataset import read_records, safe_path
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_evaluation import evaluate_rgb, load_references
from vlm_tamp.rgb_grounding import RGBPrediction
from vlm_tamp.rgb_role_diagnostics import diagnose_role_points, template_contract
from vlm_tamp.rgb_role_generalization_trial import audit_request, file_sha, load_generalization_config, request_arguments
from vlm_tamp.rgb_role_point_trial import request_arguments as previous_request_arguments
from vlm_tamp.rgb_role_points import build_point_request, compile_points, validate_points


def main():
    config, protocol, prompt = load_generalization_config(ROOT / "configs/rgb_role_generalization_trial.json")
    assert inspect.getsource(request_arguments) == inspect.getsource(previous_request_arguments)
    baseline = json.loads((ROOT / protocol["baseline"] / "manifest.json").read_text())
    assert prompt == baseline["prompt_text"]
    for key in ("model", "reasoning_effort", "image_detail", "max_output_tokens", "service_tier"):
        assert config[key] == baseline["config"][key]
    plan_path = ROOT / protocol["selection_plan_path"]
    assert file_sha(plan_path) == protocol["selection_plan_sha256"]
    plan = json.loads(plan_path.read_text())
    assert plan["selection_frozen_before_rendering_and_inference"]
    prior_names, prior_keys, prior_specs = set(), set(), set()
    for history in plan["prior_paid_history"]:
        path = ROOT / history["path"]
        assert file_sha(path) == history["sha256"]
        manifest = json.loads(path.read_text())
        if not manifest.get("dataset"):
            continue
        predictions = path.parent / "predictions.jsonl"
        rows = [json.loads(line) for line in predictions.read_text().splitlines()] if predictions.exists() else []
        ids = {row["example_id"] for row in rows if row.get("attempts", 1) > 0}
        prior_records, _ = read_records(manifest["dataset"])
        for record in prior_records:
            if record["example_id"] in ids:
                meta = record["metadata"]
                prior_names.add(meta["task_name"])
                prior_keys.add((meta["suite"], meta["task_index"]))
                prior_specs.add(meta["bddl_sha256"])
    records, root = read_records(ROOT / protocol["dataset"])
    assert file_sha(root / "records.jsonl") == protocol["dataset_sha256"]
    by_id = {record["example_id"]: record for record in records}
    assert len(records) == 20
    for record in records:
        meta = record["metadata"]
        assert meta["task_name"] not in prior_names
        assert (meta["suite"], meta["task_index"]) not in prior_keys
        assert meta["bddl_sha256"] not in prior_specs
        assert meta["state_index"] in {4, 5}
    roles_path = ROOT / protocol["offline_roles_path"]
    assert file_sha(roles_path) == protocol["offline_roles_sha256"]
    roles = json.loads(roles_path.read_text())["roles"]
    references = load_references(root, ROOT / config["references"])
    trial = ROOT / "results/rgb-role-generalization-mini"
    manifest = json.loads((trial / "manifest.json").read_text())
    rows = [json.loads(line) for line in (trial / "predictions.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (trial / "metrics.jsonl").read_text().splitlines()]
    assert not manifest["offline_mock"] and not manifest["dry_run"] and not manifest["model_input_inventory"]
    assert manifest["frozen_protocol_sha256"] == file_sha(ROOT / config["frozen_protocol"])
    assert len(rows) == len(metrics) == 20
    assert [row["example_id"] for row in rows] == config["example_ids"]
    input_tokens = output_tokens = reasoning_tokens = 0
    counts = {"raw_rgb_request_checks": 0, "raw_output_compilation_checks": 0,
              "recomputed_metric_checks": 0, "offline_role_diagnostic_checks": 0}
    diagnostic_counts = Counter()
    report = ROOT / "results/rgb-role-generalization-report"
    summary = json.loads((report / "summary.json").read_text())
    for index, (row, saved_metric, report_task) in enumerate(zip(rows, metrics, summary["tasks"])):
        record = by_id[row["example_id"]]
        contract = template_contract(record)
        assert contract["supported"] and contract["role_owners"] == roles[row["example_id"]]
        folder = trial / f"input-{index:02d}"
        args = json.loads((folder / "request.json").read_text())
        content = args["input"][0]["content"]
        assert [item["type"] for item in content] == ["input_text", "input_image"]
        payload = json.loads(content[0]["text"])
        assert set(payload) == {"instruction", "roles"}
        image_bytes = (folder / "original.png").read_bytes()
        assert base64.b64decode(content[1]["image_url"].split(",", 1)[1]) == image_bytes
        assert image_bytes == safe_path(root, record["model_inputs"]["image"]).read_bytes()
        request = build_point_request(payload["instruction"], image_bytes, prompt)
        assert args == request_arguments(config, request)
        assert audit_request(request) == protocol["requests"][row["example_id"]]
        assert not any(key in args for key in ("tools", "previous_response_id", "conversation"))
        counts["raw_rgb_request_checks"] += 1
        raw = json.loads((trial / row["raw_response"]).read_text())
        assert raw["status"] == "completed" and raw["incomplete_details"] is None
        assert raw["model"] == row["response_model"] == config["model"]
        assert raw["instructions"] == prompt and raw["reasoning"]["effort"] == "medium"
        assert raw["max_output_tokens"] == 3000
        fmt = raw["text"]["format"]
        assert fmt.get("schema", fmt.get("schema_")) == args["text"]["format"]["schema"] and fmt["strict"]
        texts = [item["text"] for output in raw["output"] if output["type"] == "message"
                 for item in output["content"] if item["type"] == "output_text"]
        points = validate_points(json.loads("".join(texts)), payload["instruction"])
        assert points == row["points"]
        try:
            compiled = compile_points(points, payload["instruction"]).model_dump(mode="json")
            assert row["status"] == "ok"
        except ValueError:
            compiled = None
            assert row["status"] == "abstained"
        assert compiled == row["prediction"]
        counts["raw_output_compilation_checks"] += 1
        prediction = RGBPrediction.model_validate(compiled) if compiled else None
        metric = evaluate_rgb(record, prediction, references[row["example_id"]], root)
        saved = {key: value for key, value in saved_metric.items() if key not in {"example_id", "status"}}
        assert json.loads(json.dumps(metric)) == saved == report_task["metric"]
        counts["recomputed_metric_checks"] += 1
        details = diagnose_role_points(record, points, roles[row["example_id"]], root)
        assert details == report_task["offline_role_diagnostics"]
        diagnostic_counts.update(detail["diagnostic"] for detail in details)
        counts["offline_role_diagnostic_checks"] += 1
        input_tokens += raw["usage"]["input_tokens"]
        output_tokens += raw["usage"]["output_tokens"]
        reasoning_tokens += raw["usage"]["output_tokens_details"]["reasoning_tokens"]
    ledger = manifest["ledger"]
    assert [input_tokens, output_tokens, reasoning_tokens] == [
        ledger["input_tokens"], ledger["output_tokens"], ledger["reasoning_tokens"]]
    cost = (input_tokens * config["pricing"]["input_per_million"]
            + output_tokens * config["pricing"]["output_per_million"]) / 1e6
    assert abs(cost - ledger["estimated_usage_cost_usd"]) < 1e-12
    assert cost <= 0.10 and ledger["api_attempts"] == 20 and ledger["unknown_usage_reserved_usd"] == 0
    assert summary["semantic_complete_goals"] == sum(metric["semantic_goal"]["exact_match"] for metric in metrics)
    assert diagnostic_counts == summary["role_diagnostics"]
    assert sum(diagnostic_counts.values()) == summary["evaluated_roles"] == 39
    assert summary["catalogue_coverage"] == protocol["coverage"]
    assert sum(score["complete_goals"] for score in summary["suite_scores"].values()) == summary["semantic_complete_goals"]
    page = PageParser()
    page.feed((report / "index.html").read_text())
    assert page.articles == 20 and len(page.images) == 60
    for name in page.images:
        assert (report / name).is_file()
    verification = {**counts, "new_task_disjointness_checks": 20, "new_api_attempts": 20,
                    "new_estimated_usage_usd": cost, "complete_goals": summary["semantic_complete_goals"],
                    "role_diagnostics": dict(diagnostic_counts), "catalogue_coverage": protocol["coverage"],
                    "prompt_transport_schema_compiler_unchanged": True, "grader_unchanged": True,
                    "selection_fixed_before_images_and_responses": True, "all_inputs_frozen_before_inference": True,
                    "earlier_results_preserved": True, "preserved_artifacts_checked": len(protocol["preserved_artifacts"]),
                    "source_images_per_request": 1, "inventory_provided": False,
                    "segmentation_used": False, "crops_used": False, "coordinate_adjustments_used": False,
                    "report_articles_checked": page.articles, "report_images_checked": len(page.images),
                    "browser_rendering_tested": False, "robot_execution_tested": False,
                    "constraints_verified": False, "checks_passed": True}
    (report / "final-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
