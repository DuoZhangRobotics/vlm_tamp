"""Recheck Mini v2 request/response evidence and preserve earlier experiments."""

import base64
import json

from verify_rgb_choices import PageParser
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_choice_v2_trial import audit_request, file_sha, load_choice_config, request_arguments
from vlm_tamp.rgb_choices_v2 import ChoicePrediction, build_choice_request, compile_choices


def main():
    counts = {
        key: 0
        for key in (
            "exact_saved_request_matches",
            "completed_responses",
            "response_model_matches",
            "schema_instruction_echoes",
            "conversion_checks",
            "rgb_only_input_checks",
        )
    }
    total_calls, total_cost = 0, 0.0
    partitions = {}
    for partition in ("dev", "validation"):
        config, protocol, prompt = load_choice_config(ROOT / f"configs/rgb_choices_v2_{partition}.json")
        trial = ROOT / f"results/rgb-choice-v2-mini-{partition}"
        manifest = json.loads((trial / "manifest.json").read_text())
        rows = [json.loads(line) for line in (trial / "predictions.jsonl").read_text().splitlines()]
        assert len(rows) == manifest["ledger"]["api_attempts"] == 10
        assert not manifest["offline_mock"] and not manifest["model_input_inventory"]
        assert protocol["previous_mini_manifest_sha256"] == file_sha(
            ROOT / "results/rgb-choice-mini/manifest.json"
        )
        assert protocol["previous_astra_manifest_sha256"] == file_sha(
            ROOT / "results/rgb-choice-astra/manifest.json"
        )
        assert protocol["legacy_evaluator_sha256"] == file_sha(ROOT / "src/vlm_tamp/evaluation.py")
        assert protocol["frozen_before_either_partition_inference"]
        input_tokens = output_tokens = reasoning_tokens = 0
        for index, row in enumerate(rows):
            folder = trial / f"input-{index:02d}"
            args = json.loads((folder / "request.json").read_text())
            content = args["input"][0]["content"]
            payload = json.loads(content[0]["text"])
            assert set(payload) == {"instruction", "visual_choices"}
            assert all(isinstance(choice, int) for choice in payload["visual_choices"])
            assert not any(key in args for key in ("previous_response_id", "conversation", "tools"))
            images = [item for item in content if item["type"] == "input_image"]
            assert len(images) == 3
            for image, name in zip(images, ("original.png", "scene.png", "cards.png")):
                assert base64.b64decode(image["image_url"].split(",", 1)[1]) == (folder / name).read_bytes()
            counts["rgb_only_input_checks"] += 1
            request = build_choice_request(
                payload["instruction"], (folder / "original.png").read_bytes(), prompt
            )
            assert args == request_arguments(config, request)
            assert audit_request(request) == protocol["requests"][row["example_id"]]
            counts["exact_saved_request_matches"] += 1
            raw = json.loads((trial / row["raw_response"]).read_text())
            assert raw["status"] == "completed" and raw["incomplete_details"] is None
            counts["completed_responses"] += 1
            assert raw["model"] == row["response_model"] == config["model"]
            counts["response_model_matches"] += 1
            assert raw["instructions"] == prompt and raw["reasoning"]["effort"] == "medium"
            assert raw["max_output_tokens"] == 8000
            fmt = raw["text"]["format"]
            assert fmt.get("schema", fmt.get("schema_")) == args["text"]["format"]["schema"]
            assert fmt["strict"]
            counts["schema_instruction_echoes"] += 1
            choices = ChoicePrediction.model_validate(row["choices"])
            try:
                compiled = compile_choices(choices, request.regions).model_dump(mode="json")
                assert row["status"] == "ok"
            except ValueError:
                compiled = None
                assert row["status"] == "abstained"
            assert compiled == row["prediction"]
            assert "reference_id" not in json.dumps(compiled)
            counts["conversion_checks"] += 1
            input_tokens += raw["usage"]["input_tokens"]
            output_tokens += raw["usage"]["output_tokens"]
            reasoning_tokens += raw["usage"]["output_tokens_details"]["reasoning_tokens"]
        ledger = manifest["ledger"]
        assert [input_tokens, output_tokens, reasoning_tokens] == [
            ledger["input_tokens"],
            ledger["output_tokens"],
            ledger["reasoning_tokens"],
        ]
        cost = (input_tokens * 0.75 + output_tokens * 4.5) / 1e6
        assert abs(cost - ledger["estimated_usage_cost_usd"]) < 1e-12
        assert ledger["unknown_usage_reserved_usd"] == 0 and cost <= 0.25
        total_calls += ledger["api_attempts"]
        total_cost += cost
        partitions[partition] = json.loads((trial / "summary.json").read_text())
    assert total_calls == 20 and total_cost <= 0.50
    report = ROOT / "results/rgb-choice-v2-report"
    summary = json.loads((report / "summary.json").read_text())
    for partition, source in partitions.items():
        assert source["semantic_complete_goals"] == sum(
            task["metric"]["semantic_goal"]["exact_match"]
            for task in summary["tasks"]
            if task["partition"] == partition
        )
    page = PageParser()
    page.feed((report / "index.html").read_text())
    assert page.articles == 20 and len(page.images) == 80
    for image in page.images:
        assert (report / image).is_file()
    verification = {
        **counts,
        "new_api_attempts": total_calls,
        "new_estimated_usage_usd": total_cost,
        "partition_complete_goal_matches": {
            name: part["semantic_complete_goals"] for name, part in partitions.items()
        },
        "candidate_coverage_tasks": summary["candidate_coverage_tasks"],
        "candidate_owners_covered": summary["candidate_owners_covered"],
        "candidate_owners_required": summary["candidate_owners_required"],
        "cumulative_api_attempts": 204 + total_calls,
        "cumulative_estimated_usage_usd": 3.4300996 + total_cost,
        "previous_experiments_and_graders_unchanged": True,
        "both_partitions_frozen_before_inference": True,
        "validation_previously_unqueried_initial_states": True,
        "validation_is_task_held_out": False,
        "simulator_inventory_supplied": False,
        "report_articles_checked": 20,
        "report_images_checked": 80,
        "browser_rendering_tested": False,
        "constraints_verified": False,
        "part_pixel_localization_verified": False,
        "robot_execution_tested": False,
        "checks_passed": True,
    }
    (report / "final-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
