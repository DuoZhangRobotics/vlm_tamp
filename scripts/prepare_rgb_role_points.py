"""Freeze a paired comparison before any direct-RGB inference."""

import copy
import json

from vlm_tamp.dataset import read_records, safe_path
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_role_point_trial import audit_request, file_sha, request_arguments
from vlm_tamp.rgb_role_points import build_point_request, digest


def main():
    config_path = ROOT / "configs/rgb_role_points_trial.json"
    protocol_path = ROOT / "configs/frozen_rgb_role_points_trial.json"
    if config_path.exists() or protocol_path.exists():
        raise FileExistsError("Preserve the frozen direct-RGB trial")
    baseline = ROOT / "results/rgb-role-mini-validation"
    baseline_manifest = json.loads((baseline / "manifest.json").read_text())
    assert not baseline_manifest["offline_mock"] and not baseline_manifest["model_input_inventory"]
    config = copy.deepcopy(baseline_manifest["config"])
    config.update(
        prompt="configs/prompts/rgb_role_points_v1.txt",
        conditions=["rgb_role_points"],
        frozen_protocol="configs/frozen_rgb_role_points_trial.json",
        max_budget_usd=0.10,
        image_token_allowance=4096,
        pricing_checked="2026-10-12",
    )
    records, root = read_records(ROOT / "data/rgb-role-choices")
    by_id = {record["example_id"]: record for record in records}
    selected = [by_id[identifier] for identifier in config["example_ids"]]
    assert len(selected) == config["max_requests"] == 10
    prompt = (ROOT / config["prompt"]).read_text()
    requests = {}
    image_checks = []
    for index, record in enumerate(selected):
        image_bytes = safe_path(root, record["model_inputs"]["image"]).read_bytes()
        assert image_bytes == (baseline / f"input-{index:02d}/original.png").read_bytes()
        request = build_point_request(record["model_inputs"]["instruction"], image_bytes, prompt)
        baseline_arguments = json.loads((baseline / f"input-{index:02d}/request.json").read_text())
        baseline_payload = json.loads(baseline_arguments["input"][0]["content"][0]["text"])
        assert request.payload["roles"] == baseline_payload["roles"]
        assert request.payload["instruction"] == baseline_payload["instruction"]
        for key in ("model", "reasoning", "service_tier", "max_output_tokens", "store"):
            assert request_arguments(config, request)[key] == baseline_arguments[key]
        requests[record["example_id"]] = audit_request(request)
        image_checks.append(record["example_id"])
    modules = (
        "rgb_role_points.py", "rgb_role_point_trial.py", "rgb_role_choices.py", "rgb_object_choices.py",
        "rgb_choices_v2.py", "rgb_choices.py", "rgb_evaluation.py", "rgb_grounding.py",
        "dataset.py", "experiment.py", "providers.py", "grounding.py", "libero.py",
    )
    preserved = [
        ROOT / "src/vlm_tamp/evaluation.py",
        ROOT / "configs/frozen_rgb_role_choices_dev.json",
        ROOT / "configs/frozen_rgb_role_choices_validation.json",
        ROOT / "results/rgb-role-choice-report/summary.json",
        ROOT / "results/rgb-role-choice-report/index.html",
    ]
    # Freeze all earlier paid manifests/predictions/metrics/summaries. These
    # historical files are never rewritten by the new experiment or report.
    prior_calls, prior_cost = 0, 0.0
    for path in sorted((ROOT / "results").glob("*/manifest.json")):
        saved = json.loads(path.read_text())
        ledger = saved.get("ledger") or {}
        if saved.get("offline_mock") or saved.get("postprocessing") or not ledger.get("api_attempts"):
            continue
        prior_calls += ledger["api_attempts"]
        prior_cost += ledger["estimated_usage_cost_usd"]
        preserved.append(path)
        for name in ("predictions.jsonl", "metrics.jsonl", "summary.json"):
            if (path.parent / name).exists():
                preserved.append(path.parent / name)
    protocol = {
        "config": config,
        "dataset": "data/rgb-role-choices",
        "dataset_sha256": file_sha(root / "records.jsonl"),
        "prompt_sha256": digest(prompt.encode()),
        "references_sha256": file_sha(ROOT / config["references"]),
        "requests": requests,
        "implementation_hashes": {
            name: {"path": "src/vlm_tamp/" + name, "sha256": file_sha(ROOT / "src/vlm_tamp" / name)}
            for name in modules
        },
        "preserved_artifacts": {
            str(path.relative_to(ROOT)): {"path": str(path.relative_to(ROOT)), "sha256": file_sha(path)}
            for path in preserved
        },
        "baseline": "results/rgb-role-mini-validation",
        "paired_image_and_role_checks": image_checks,
        "baseline_complete_goal_matches": 9,
        "frozen_before_inference": True,
        "input_image_count": 1,
        "image_resolution": 512,
        "segmentation_used": False,
        "crops_used": False,
        "inventory_provided": False,
        "validation_is_task_held_out": False,
        "images_previously_queried": True,
        "comparison_scope": "Same ten state3 scenes, instructions, roles, pinned Mini, reasoning, output cap, goal compiler and semantic grader. Replace three candidate images/coordinates with one untouched RGB; Mini returns its own normalized points. This changes the localization interface as well as removing MobileSAM, so it is not a crops-only ablation.",
        "prior_api_attempts": prior_calls,
        "prior_estimated_usage_usd": prior_cost,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"paired_scenes": len(requests), "source_image_count": 1,
                      "segmentation_used": False, "crops_used": False,
                      "spending_cap_usd": config["max_budget_usd"],
                      "prompt_words": len(prompt.split()), "prior_api_attempts": prior_calls,
                      "prior_estimated_usage_usd": prior_cost}, indent=2))


if __name__ == "__main__":
    main()
