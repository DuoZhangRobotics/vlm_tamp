"""Recheck the saved ten-call choice trial without loading credentials or calling APIs."""

import argparse
import base64
import json
from html.parser import HTMLParser

from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_choice_trial import audit_request, file_sha, load_choice_config, request_arguments
from vlm_tamp.rgb_choices import ChoicePrediction, build_choice_request, compile_choices


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.articles = 0
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.articles += 1
        if tag == "img":
            self.images.append(attrs["src"])


def main():
    parser = argparse.ArgumentParser(description="Verify saved RGB-only choice runs without API calls")
    parser.add_argument("--config", default="configs/rgb_choices_trial.json")
    parser.add_argument("--experiment", default="results/rgb-choice-mini")
    parser.add_argument("--baseline", default="results/rgb-repair-mini")
    parser.add_argument("--report", default="results/rgb-choice-report")
    args = parser.parse_args()
    config, protocol, prompt = load_choice_config(ROOT / args.config)
    trial = ROOT / args.experiment
    baseline = ROOT / args.baseline
    report = ROOT / args.report
    rows = [json.loads(line) for line in (trial / "predictions.jsonl").read_text().splitlines()]
    manifest = json.loads((trial / "manifest.json").read_text())
    assert len(rows) == manifest["ledger"]["api_attempts"] == len(config["example_ids"])
    counts = {
        key: 0
        for key in (
            "exact_saved_request_matches",
            "requests_contain_only_allowed_fields",
            "completed_responses",
            "response_model_matches",
            "frozen_instruction_echoes",
            "response_schema_echoes",
            "compiled_form_matches",
            "original_rgb_hash_matches",
        )
    }
    input_tokens = output_tokens = reasoning_tokens = 0
    for index, row in enumerate(rows):
        folder = trial / f"input-{index:02d}"
        args = json.loads((folder / "request.json").read_text())
        content = args["input"][0]["content"]
        payload = json.loads(content[0]["text"])
        assert set(payload) == {"instruction", "visual_choices"}
        assert all(isinstance(value, int) for value in payload["visual_choices"])
        assert len([item for item in content if item["type"] == "input_image"]) == 3
        assert all(key not in args for key in ("previous_response_id", "conversation", "tools"))
        counts["requests_contain_only_allowed_fields"] += 1
        request = build_choice_request(payload["instruction"], (folder / "original.png").read_bytes(), prompt)
        assert args == request_arguments(config, request)
        counts["exact_saved_request_matches"] += 1
        assert audit_request(request) == protocol["requests"][row["example_id"]]
        images = [item for item in content if item["type"] == "input_image"]
        for item, name in zip(images, ("original.png", "scene.png", "cards.png")):
            assert base64.b64decode(item["image_url"].split(",", 1)[1]) == (folder / name).read_bytes()
        counts["original_rgb_hash_matches"] += 1
        raw = json.loads((trial / row["raw_response"]).read_text())
        assert raw["status"] == row["response_status"] == "completed"
        counts["completed_responses"] += 1
        assert raw["model"] == row["response_model"] == config["model"]
        counts["response_model_matches"] += 1
        assert raw["instructions"] == prompt
        counts["frozen_instruction_echoes"] += 1
        assert raw["reasoning"]["effort"] == config["reasoning_effort"]
        assert raw["max_output_tokens"] == config["max_output_tokens"]
        format_echo = raw["text"]["format"]
        # SDK model_dump uses schema_ to avoid the BaseModel.schema name.
        assert format_echo.get("schema", format_echo.get("schema_")) == args["text"]["format"]["schema"]
        assert format_echo["strict"] is True
        counts["response_schema_echoes"] += 1
        parsed = ChoicePrediction.model_validate(row["choices"])
        try:
            compiled = compile_choices(parsed, request.regions).model_dump(mode="json")
            assert row["status"] == "ok"
        except ValueError:
            compiled = None
            assert row["status"] == "abstained"
        assert compiled == row["prediction"]
        counts["compiled_form_matches"] += 1
        assert "reference_id" not in json.dumps(row["prediction"])
        input_tokens += raw["usage"]["input_tokens"]
        output_tokens += raw["usage"]["output_tokens"]
        reasoning_tokens += raw["usage"]["output_tokens_details"]["reasoning_tokens"]
    ledger = manifest["ledger"]
    assert ledger["input_tokens"] == input_tokens and ledger["output_tokens"] == output_tokens
    assert ledger["reasoning_tokens"] == reasoning_tokens
    actual_estimate = (
        input_tokens * config["pricing"]["input_per_million"]
        + output_tokens * config["pricing"]["output_per_million"]
    ) / 1e6
    assert abs(actual_estimate - ledger["estimated_usage_cost_usd"]) < 1e-12
    assert ledger["unknown_usage_reserved_usd"] == 0
    assert ledger["estimated_usage_cost_usd"] <= config["max_budget_usd"]
    assert protocol["baseline_manifest_sha256"] == file_sha(baseline / "manifest.json")
    identical_inputs = 0
    if "public_request_comparison" in protocol:
        for index, _ in enumerate(rows):
            current = json.loads((trial / f"input-{index:02d}/request.json").read_text())
            old = json.loads((baseline / f"input-{index:02d}/request.json").read_text())
            assert {key for key in current if current[key] != old[key]} == {"model", "max_output_tokens"}
            identical_inputs += 1
    assert protocol["legacy_evaluator_sha256"] == file_sha(ROOT / "src/vlm_tamp/evaluation.py")
    summary = json.loads((report / "summary.json").read_text())
    assert summary["semantic_complete_goals"] == sum(
        t["metric"]["semantic_goal"]["exact_match"] for t in summary["tasks"]
    )
    assert summary["baseline_semantic_complete_goals"] == sum(
        t["baseline_metric"]["semantic_goal"]["exact_match"] for t in summary["tasks"]
    )
    page = PageParser()
    page.feed((report / "index.html").read_text())
    assert page.articles == len(rows) and len(page.images) == len(rows) * 4
    for relative in page.images:
        assert (report / relative).is_file()
    verification = {
        **counts,
        "full_candidate_coverage_tasks": summary["full_candidate_coverage_tasks"],
        "candidate_owners_covered": summary["required_owners_covered"],
        "candidate_owners_required": summary["required_owners"],
        "semantic_complete_goals": summary["semantic_complete_goals"],
        "baseline_semantic_complete_goals": summary["baseline_semantic_complete_goals"],
        "new_estimated_usage_usd": actual_estimate,
        **summary["prior_and_cumulative_usage"],
        "previous_run_and_legacy_evaluator_unchanged": True,
        "grader_unchanged": True,
        "simulator_inventory_supplied": False,
        "choice_generation_uses_rgb_only": True,
        "references_used_offline_only": True,
        "report_articles_checked": page.articles,
        "report_images_checked": len(page.images),
        "browser_rendering_tested": False,
        "constraints_verified": False,
        "part_pixel_localization_verified": False,
        "robot_execution_tested": False,
        "identical_backbone_input_comparisons": identical_inputs,
        "requested_model": config["model"],
        "compiled_specifications": sum(row["status"] == "ok" for row in rows),
        "abstentions": sum(row["status"] == "abstained" for row in rows),
        "checks_passed": True,
    }
    (report / "final-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
