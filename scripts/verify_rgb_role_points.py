"""Recheck actual RGB-only requests, raw replies, compilation and paired scores."""

import base64
import json
from html.parser import HTMLParser

from vlm_tamp.dataset import read_records, safe_path
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_evaluation import evaluate_rgb, load_references
from vlm_tamp.rgb_grounding import RGBPrediction
from vlm_tamp.rgb_role_point_trial import audit_request, file_sha, load_point_config, request_arguments
from vlm_tamp.rgb_role_points import build_point_request, compile_points, validate_points


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.articles = 0
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.articles += 1
        if tag == "img":
            self.images.append(dict(attrs)["src"])


def main():
    config, protocol, prompt = load_point_config(ROOT / "configs/rgb_role_points_trial.json")
    records, root = read_records(ROOT / protocol["dataset"])
    assert file_sha(root / "records.jsonl") == protocol["dataset_sha256"]
    by_id = {record["example_id"]: record for record in records}
    references = load_references(root, ROOT / config["references"])
    trial = ROOT / "results/rgb-role-points-mini"
    baseline = ROOT / protocol["baseline"]
    manifest = json.loads((trial / "manifest.json").read_text())
    rows = [json.loads(line) for line in (trial / "predictions.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (trial / "metrics.jsonl").read_text().splitlines()]
    assert not manifest["offline_mock"] and not manifest["dry_run"]
    assert not manifest["model_input_inventory"] and not manifest["reference_answers_sent"]
    assert not manifest["segmentation_used"] and not manifest["crops_used"]
    assert manifest["frozen_protocol_sha256"] == file_sha(ROOT / config["frozen_protocol"])
    assert len(rows) == len(metrics) == 10
    assert [row["example_id"] for row in rows] == config["example_ids"]
    counts = {"raw_rgb_request_checks": 0, "paired_image_role_checks": 0,
              "raw_output_and_compilation_checks": 0, "recomputed_metric_checks": 0,
              "complete_responses": 0}
    input_tokens = output_tokens = reasoning_tokens = 0
    for index, (row, saved_metric) in enumerate(zip(rows, metrics)):
        record = by_id[row["example_id"]]
        folder = trial / f"input-{index:02d}"
        args = json.loads((folder / "request.json").read_text())
        content = args["input"][0]["content"]
        assert [item["type"] for item in content] == ["input_text", "input_image"]
        payload = json.loads(content[0]["text"])
        assert set(payload) == {"instruction", "roles"}
        image_bytes = (folder / "original.png").read_bytes()
        assert base64.b64decode(content[1]["image_url"].split(",", 1)[1]) == image_bytes
        assert image_bytes == safe_path(root, record["model_inputs"]["image"]).read_bytes()
        assert not any(key in args for key in ("tools", "previous_response_id", "conversation"))
        request = build_point_request(payload["instruction"], image_bytes, prompt)
        assert args == request_arguments(config, request)
        assert audit_request(request) == protocol["requests"][row["example_id"]]
        counts["raw_rgb_request_checks"] += 1
        old_args = json.loads((baseline / f"input-{index:02d}/request.json").read_text())
        old_payload = json.loads(old_args["input"][0]["content"][0]["text"])
        assert payload["instruction"] == old_payload["instruction"] and payload["roles"] == old_payload["roles"]
        assert image_bytes == (baseline / f"input-{index:02d}/original.png").read_bytes()
        for key in ("model", "reasoning", "service_tier", "max_output_tokens", "store"):
            assert args[key] == old_args[key]
        assert content[1]["detail"] == "high"
        counts["paired_image_role_checks"] += 1
        raw = json.loads((trial / row["raw_response"]).read_text())
        assert raw["status"] == row["response_status"] == "completed"
        assert raw["incomplete_details"] is None
        assert raw["model"] == row["response_model"] == config["model"]
        assert raw["instructions"] == prompt and raw["reasoning"]["effort"] == "medium"
        assert raw["max_output_tokens"] == 3000
        fmt = raw["text"]["format"]
        assert fmt.get("schema", fmt.get("schema_")) == args["text"]["format"]["schema"]
        assert fmt["strict"]
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
        counts["complete_responses"] += 1
        counts["raw_output_and_compilation_checks"] += 1
        prediction = RGBPrediction.model_validate(compiled) if compiled else None
        metric = evaluate_rgb(record, prediction, references[row["example_id"]], root)
        assert json.loads(json.dumps(metric)) == {key: value for key, value in saved_metric.items()
                                                 if key not in {"example_id", "status"}}
        counts["recomputed_metric_checks"] += 1
        input_tokens += raw["usage"]["input_tokens"]
        output_tokens += raw["usage"]["output_tokens"]
        reasoning_tokens += raw["usage"]["output_tokens_details"]["reasoning_tokens"]
    ledger = manifest["ledger"]
    assert [input_tokens, output_tokens, reasoning_tokens] == [
        ledger["input_tokens"], ledger["output_tokens"], ledger["reasoning_tokens"]]
    cost = (input_tokens * config["pricing"]["input_per_million"]
            + output_tokens * config["pricing"]["output_per_million"]) / 1e6
    assert abs(cost - ledger["estimated_usage_cost_usd"]) < 1e-12
    assert cost <= 0.10 and ledger["api_attempts"] == 10 and ledger["unknown_usage_reserved_usd"] == 0
    report = ROOT / "results/rgb-role-points-report"
    summary = json.loads((report / "summary.json").read_text())
    assert summary["semantic_complete_goals"] == sum(metric["semantic_goal"]["exact_match"] for metric in metrics)
    assert summary["baseline_semantic_complete_goals"] == protocol["baseline_complete_goal_matches"]
    assert len(summary["tasks"]) == 10 and sum(summary["paired_transitions"].values()) == 10
    for task, saved_metric in zip(summary["tasks"], metrics):
        assert json.loads(json.dumps(task["metric"])) == {key: value for key, value in saved_metric.items()
                                                         if key not in {"example_id", "status"}}
    page = PageParser()
    page.feed((report / "index.html").read_text())
    assert page.articles == 10 and len(page.images) == 30
    for name in page.images:
        assert (report / name).is_file()
    verification = {
        **counts, "new_api_attempts": ledger["api_attempts"], "new_estimated_usage_usd": cost,
        "direct_rgb_complete_goals": summary["semantic_complete_goals"],
        "saved_segmentation_complete_goals": summary["baseline_semantic_complete_goals"],
        "paired_transitions": summary["paired_transitions"],
        "source_images_per_request": 1, "segmentation_used": False, "crops_used": False,
        "inventory_provided": False, "coordinate_adjustments_used": False,
        "goal_compiler_and_grader_unchanged": True, "earlier_results_preserved": True,
        "preserved_artifacts_checked": len(protocol["preserved_artifacts"]),
        "frozen_before_inference": True, "images_previously_queried": True,
        "task_held_out_validation": False, "report_articles_checked": page.articles,
        "report_images_checked": len(page.images), "browser_rendering_tested": False,
        "robot_execution_tested": False, "constraints_verified": False, "checks_passed": True,
    }
    (report / "final-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
