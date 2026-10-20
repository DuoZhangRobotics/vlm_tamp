"""Presentation-predicate repair; identical full frontal-view data and numerical replay."""

import argparse
import base64
import hashlib
import json
import re
from collections import Counter

import numpy as np
from PIL import Image

from report_task_grounding_frontview import (
    DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_TRIAL, PREVIOUS_REPORT, camera_comparison,
)
from report_task_grounding_scope import (
    HISTORICAL_REPORT, aggregate_drawer, local_path, read_rows, response_context, response_text,
)
from verify_task_grounding_scope import AssetReferences, _without_result_link, line_formula_check
from vlm_tamp.drawer_constraints import compile_drawer
from vlm_tamp.experiment import input_price_ceiling
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_landmark_trial import file_sha
from vlm_tamp.rgbd_numeric_evaluation import capture_checks
from vlm_tamp.rgbd_numericals import json_text


def verify(config_path=DEFAULT_CONFIG, trial_path=DEFAULT_TRIAL, report_path=DEFAULT_OUTPUT):
    from vlm_tamp.task_grounding_frontview import (
        check_protocol, derive_pick_place, grade_drawer, inputs as trial_inputs,
        request_arguments, request_cost_reservation, resolve_choices,
    )

    config, protocol, records = check_protocol(local_path(config_path))
    trial, report = local_path(trial_path), local_path(report_path)
    manifest = json.loads((trial / "manifest.json").read_text())
    summary = json.loads((report / "summary.json").read_text())
    rows = read_rows(trial / "predictions.jsonl")
    pick_place = json.loads((trial / "pick_place.json").read_text())
    assert pick_place == derive_pick_place()
    pick_rows = pick_place["tasks"] if isinstance(pick_place, dict) else pick_place
    assert len(pick_rows) == 8
    assert summary["ordinary_pick_place"]["cases"] == 8
    assert summary["ordinary_pick_place"]["new_api_calls"] == 0
    assert Counter(row["condition"] for row in pick_rows) == {"dinov2": 4, "dinov3": 4}
    pick_by_id = {task["case_id"]: task for task in summary["ordinary_pick_place"]["tasks"]}
    history = json.loads((ROOT / HISTORICAL_REPORT).read_text())
    historical = {task["id"]: task for task in history["tasks"]}
    for row in pick_rows:
        old, task = historical[row["case_id"]], pick_by_id[row["case_id"]]
        assert old["upright"] is False
        assert "path_constraint" not in row["choices"]
        assert not any(key.startswith("axis_") for key in row["choices"])
        assert not any(key.startswith("axis_") for key in row["blanks"])
        assert row["numericals"].get("object_axis") is None
        assert all(relation["scope"] == "goal" for relation in row["numericals"].get("relations", []))
        assert not row["numericals"].get("path_constraints", [])
        assert row["numericals"].get("robot_execution_tested") is False
        assert row["numericals"] == task["numericals"] and row["status"] == task["status"]
        assert task["metrics"]["semantic_complete_goal"] == old["metrics"]["semantic_complete_goal"]
        saved = json.loads((report / "assets" / (row["case_id"] + "-derived-result.json")).read_text())
        assert saved == _without_result_link(task)
        for artifact in task["artifacts"].values():
            if "source" in artifact:
                assert file_sha(local_path(artifact["source"])) == artifact["sha256"]
                assert file_sha(report / artifact["asset"]) == artifact["sha256"]
    for condition, counts in summary["ordinary_pick_place"]["condition_summaries"].items():
        group = [task for task in summary["ordinary_pick_place"]["tasks"] if task["condition"] == condition]
        assert counts == {"cases": len(group), "valid_forms": sum(task["status"] == "valid" for task in group),
                          "semantic_complete_goals": sum(task["metrics"]["semantic_complete_goal"] for task in group),
                          "path_relations": 0}
    assert summary["config"] == config and manifest["config"] == config
    assert summary["model"] == config["model"] == "gpt-6-luna"
    assert summary["frozen_protocol_sha256"] == file_sha(ROOT / config["frozen_protocol"])
    assert summary["planned_drawer_cases"] == len(config["cases"]) == 8
    assert set(rows) == {case["id"] for case in config["cases"]}
    tasks = {task["case_id"]: task for task in summary["drawer_tasks"]}
    assert set(tasks) == set(rows)
    assert Counter(case["condition"] for case in config["cases"]) == {"direct_rgb": 4, "dinov3_candidates": 4}
    assert summary["prompt_text"] == manifest["prompt_text"] == (ROOT / config["prompt"]).read_text()
    assert manifest["provider"] == "openai" and not manifest["dry_run"] and not manifest["offline_mock"]
    assert not manifest["model_input_inventory"] and not manifest["reference_answers_sent"]
    assert not manifest["depth_sent_to_vlm"] and manifest["pick_place_new_api_calls"] == 0
    assert manifest["offline_development_proposal_gate_used"] is True
    inputs, outputs, reasoning, cached, cache_written, responses, failures = 0, 0, 0, 0, 0, 0, 0
    formula_checks, captures, candidate_checks, checkpoint_checks = [], {}, {}, set()
    unknown_reservation = 0.
    for case in config["cases"]:
        identifier = case["id"]
        folder = trial / identifier
        row, task = rows[identifier], tasks[identifier]
        record = records[case["example_id"]]
        assert task["raw_row"] == row and task["status"] == row["status"]
        assert task["metrics"] == grade_drawer(record, ROOT, case, row)
        assert task["response_context"] == response_context(folder, row)
        saved = json.loads((report / "assets" / (identifier + "-result.json")).read_text())
        assert saved == _without_result_link(task)
        payload = json.loads((folder / "payload.json").read_text())
        request = json.loads((folder / "request.json").read_text())
        candidates, original, numbered, shape, expected_payload = trial_inputs(config, case, record)
        assert payload == expected_payload
        assert request == request_arguments(config, case, record)
        frozen = next(value for value in protocol["requests"] if value["case_id"] == identifier)
        assert frozen["request_sha256"] == hashlib.sha256(json_text(request).encode()).hexdigest()
        assert (folder / "original.png").read_bytes() == original
        if numbered is not None:
            assert (folder / "numbered.png").read_bytes() == numbered
        assert request["model"] == "gpt-6-luna" and request["reasoning"] == {"effort": "none"}
        assert request["max_output_tokens"] == 1000 and request["store"] is False
        assert request["instructions"] == manifest["prompt_text"]
        content = request["input"][0]["content"]
        assert len([item for item in content if item["type"] == "input_text"]) == 1
        assert json.loads(next(item["text"] for item in content if item["type"] == "input_text")) == payload
        assert not {"reference", "inventory", "object_poses", "joint_axis", "ground_truth", "evaluation"} & set(payload)
        images = [item for item in content if item["type"] == "input_image"]
        assert len(images) == (1 if case["condition"] == "direct_rgb" else 2)
        expected = [folder / "original.png"]
        if case["condition"] == "dinov3_candidates":
            expected.append(folder / "numbered.png")
        for image, path in zip(images, expected):
            assert base64.b64decode(image["image_url"].split(",", 1)[1], validate=True) == path.read_bytes()
            assert image["detail"] == config["image_detail"]
        capture = ROOT / config["dataset"] / case["example_id"]
        depth = np.load(capture / "depth_meters.npy", allow_pickle=False)
        cloud = np.load(capture / "world_points.npy", allow_pickle=False)
        calibration = json.loads((capture / "calibration.json").read_text())
        captures[case["example_id"]] = capture_checks(depth, cloud, calibration)
        assert captures[case["example_id"]]["passed"]
        assert (folder / "original.png").read_bytes() == (capture / "rgb.png").read_bytes()
        if case["condition"] == "dinov3_candidates":
            candidate_dir = ROOT / config["candidate_dataset"] / case["example_id"]
            metadata = json.loads((candidate_dir / "metadata.json").read_text())
            assert metadata["model_inference_executed"] and not metadata["offline_mock"]
            assert not metadata["provided_inventory"] and not metadata["reference_masks_used"]
            assert metadata["encoder"]["rgb_only"] and not metadata["encoder"]["depth_input"]
            assert metadata["encoder"]["model_id"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
            assert metadata["original_image_sha256"] == file_sha(capture / "rgb.png")
            assert metadata["depth_sha256"] == file_sha(capture / "depth_meters.npy")
            assert metadata["world_points_sha256"] == file_sha(capture / "world_points.npy")
            patches = np.load(candidate_dir / "patch_features.npy", allow_pickle=False)
            masks = np.load(candidate_dir / "predicted_masks.npy", allow_pickle=False)
            assert patches.shape == (32, 32, 384) and np.isfinite(patches).all()
            assert masks.dtype == bool and masks.shape[1:] == shape
            assert metadata["candidate_count"] == len(candidates)
            checkpoint = local_path(metadata["encoder"]["checkpoint_directory"]) / "model.safetensors"
            if str(checkpoint) not in checkpoint_checks:
                assert file_sha(checkpoint) == metadata["encoder"]["weights_sha256"]
                checkpoint_checks.add(str(checkpoint))
            for candidate in candidates:
                x, y = candidate["pixel_xy"]
                assert np.array_equal(np.asarray(candidate["position_world_m"]), cloud[y, x])
                region = candidate["predicted_region_id"]
                assert 1 <= region <= len(masks) and masks[region - 1, y, x]
            candidate_checks[case["example_id"]] = len(candidates)
            reference = json.loads((capture / "evaluation/reference.json").read_text())
            role_masks = {role: np.load(capture / reference["masks"][role]["npy"], allow_pickle=False)
                          for role in ("middle_drawer", "middle_handle", "middle_panel")}
            role_counts = {role: sum(bool(mask[candidate["pixel_xy"][1], candidate["pixel_xy"][0]])
                                     for candidate in candidates) for role, mask in role_masks.items()}
            coverage = protocol["offline_proposal_coverage"][case["example_id"]]
            assert role_counts == coverage["available_points_by_reference_role"]
            assert coverage["required_handle_covered"] == (role_counts["middle_handle"] > 0)
            assert coverage["reference_points_added"] is False
            assert task["offline_proposal_coverage"] == coverage
        response_path = folder / "response.json"
        if response_path.exists():
            raw = json.loads(response_path.read_text())
            assert raw.get("id") and raw["status"] in {"completed", "incomplete"}
            assert raw["model"] == "gpt-6-luna" or re.fullmatch(r"gpt-6-luna-\d{4}-\d{2}-\d{2}", raw["model"])
            assert row["response_model"] == raw["model"] and row["response_status"] == raw["status"]
            usage = raw["usage"]
            inputs += usage["input_tokens"]
            outputs += usage["output_tokens"]
            reasoning += (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            cached += (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
            cache_written += (usage.get("input_tokens_details") or {}).get("cache_write_tokens", 0) or 0
            responses += 1
            cost = (usage["input_tokens"] * input_price_ceiling(config) +
                    usage["output_tokens"] * config["pricing"]["output_per_million"]) / 1e6
            assert abs(cost - row["cost_usd"]) < 1e-12
            assert row["attempts"] == 1
            if row["status"] == "valid":
                assert raw["status"] == "completed"
                assert json.loads(response_text(raw)) == row["choices"]
                resolved = resolve_choices(row["choices"], case["condition"], candidates, shape)
                assert resolved == row["blanks"]
                numeric = compile_drawer(resolved, payload, depth, calibration, config["geometry"])
                assert numeric == row["numericals"]
                assert json.loads((folder / "numericals.json").read_text()) == numeric
                assert numeric["opening_goal"]["status"] == "unresolved"
                assert numeric["opening_goal"]["requested_extension_m"] is None
                assert numeric["planner_ready"] is False and numeric["robot_execution_tested"] is False
                for relation in numeric["relations"]:
                    assert relation["type"] == "point_on_line" and relation["scope"] == "path"
                    assert relation["stage"] == "pull" and relation["planner_ready"] is False
                    assert relation["maximum_travel_m"] is None
                    formula_checks.append(line_formula_check(relation))
                stages = {stage["id"]: stage for stage in numeric["stages"]}
                assert not stages["approach"]["path_constraint_ids"]
                assert not stages["grasp"]["path_constraint_ids"]
                assert not stages["release"]["path_constraint_ids"]
                assert stages["pull"]["path_constraint_ids"] == [relation["id"] for relation in numeric["relations"]]
        else:
            assert row["status"] in {"not_run", "api_failure", "proposal_unavailable"}
            if row["status"] == "proposal_unavailable":
                assert case["condition"] == "dinov3_candidates"
                assert not protocol["offline_proposal_coverage"][case["example_id"]]["required_handle_covered"]
                assert row["cost_usd"] == 0 and not task["response_context"]["selection_scored"]
            failures += row["status"] == "api_failure"
            if row["status"] == "api_failure":
                unknown_reservation += request_cost_reservation(config, request)
            assert row["attempts"] == (1 if row["status"] == "api_failure" else 0)
            assert row.get("choices") is None and row.get("numericals") is None
        for artifact in task["artifacts"].values():
            if "source" in artifact:
                assert file_sha(local_path(artifact["source"])) == artifact["sha256"]
                assert file_sha(report / artifact["asset"]) == artifact["sha256"]
    assert len(captures) == 4
    assert len(candidate_checks) == 4 and len(checkpoint_checks) == 1
    unavailable_cases = sum(row["status"] == "proposal_unavailable" for row in rows.values())
    assert unavailable_cases == sum(not item["required_handle_covered"]
                                    for item in protocol["offline_proposal_coverage"].values())
    assert all(check["passed"] for check in formula_checks)
    for condition, counts in summary["drawer_condition_summaries"].items():
        assert counts == aggregate_drawer([task for task in summary["drawer_tasks"] if task["condition"] == condition])
    ledger = manifest["ledger"]
    assert ledger == summary["ledger"]
    assert ledger["api_attempts"] == responses + failures <= 8
    assert ledger["input_tokens"] == inputs and ledger["output_tokens"] == outputs
    assert ledger["reasoning_tokens"] == reasoning == 0
    assert ledger["cached_input_tokens"] == cached and ledger["cache_write_tokens"] == cache_written
    assert ledger["accounting_input_rate_per_million"] == input_price_ceiling(config) == .125
    assert abs(unknown_reservation - ledger["unknown_usage_reserved_usd"]) < 1e-12
    cost = (inputs * input_price_ceiling(config) + outputs * config["pricing"]["output_per_million"]) / 1e6
    assert abs(cost - ledger["estimated_usage_cost_usd"]) < 1e-12
    assert cost + ledger["unknown_usage_reserved_usd"] <= config["max_budget_usd"]
    prior_cost = sum(json.loads((ROOT / folder / "manifest.json").read_text())["ledger"]["estimated_usage_cost_usd"]
                     for folder in config["prior_usage_trials"])
    assert abs(config["prior_usage_cost_usd"] - prior_cost) < 1e-12
    assert config["overall_budget_usd"] == 2
    assert config["prior_usage_cost_usd"] + cost + ledger["unknown_usage_reserved_usd"] <= 2
    assert abs(manifest["cumulative_luna_estimated_cost_usd"] - config["prior_usage_cost_usd"] - cost) < 1e-12
    for key in ("robot_execution_tested", "joint_space_compiled", "encoder_ranking_established"):
        assert summary[key] is False
    assert summary["provenance"]["reference_answers_in_inference"] is False
    parser = AssetReferences()
    page = (report / "index.html").read_text()
    parser.feed(page)
    assert parser.drawer_articles == parser.pick_articles == 8
    assert parser.canvases == sum(task.get("cloud_data") is not None for task in summary["drawer_tasks"])
    for value in parser.paths:
        path = (report / value).resolve()
        assert path.is_relative_to(report.resolve())
        if path == (report / "verification.json").resolve():
            continue
        assert path.is_file()
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
    for path, name in ((local_path(config_path), "config.json"),
                       (ROOT / config["frozen_protocol"], "protocol.json"),
                       (trial / "manifest.json", "manifest.json"), (trial / "pick_place.json", "pick_place.json")):
        assert (report / "assets" / name).read_bytes() == path.read_bytes()
    assert (report / "assets/prompt.txt").read_text() == manifest["prompt_text"]
    comparison = camera_comparison(summary)
    assert summary["camera_intervention_comparison"] == comparison
    assert json.loads((report / "assets/camera-comparison.json").read_text()) == comparison
    assert (report / "assets/previous-oblique-summary.json").read_bytes() == (ROOT / PREVIOUS_REPORT).read_bytes()
    assert "The camera intervention changes visible RGB, measured depth and landmark" in page
    assert "This retry uses an explicitly chosen frontal camera view" in page
    assert "offline development audit" in page and "Previous oblique view" in page
    assert "The four DINOv3 bundles contain no candidate" not in page
    verification = {"camera_intervention_not_model_or_encoder_ranking": True, "passed": True, "model": config["model"], "real_api_replies_replayed": responses,
                    "planned_drawer_cases_retained": 8, "ordinary_pick_place_cases_reused": 8,
                    "new_pick_place_api_calls": 0, "pick_place_path_constraints": 0,
                    "candidate_cases_unrequested_for_missing_handle_proposal": unavailable_cases,
                    "development_proposal_gate_uses_offline_reference_labels": True,
                    "drawer_line_relation_formula_checks": len(formula_checks),
                    "maximum_line_jacobian_error": max((value["maximum_jacobian_error"]
                                                        for value in formula_checks), default=None),
                    "depth_capture_inverse_consistency_checks": len(captures),
                    "real_dinov3_feature_captures_checked": len(candidate_checks),
                    "dinov3_checkpoint_hashes_checked": len(checkpoint_checks),
                    "candidate_counts": candidate_checks,
                    "estimated_new_usage_cost_usd": cost,
                    "estimated_total_luna_usage_cost_usd": config["prior_usage_cost_usd"] + cost,
                    "budget_authorized_usd": 2, "report_linked_assets_checked": len(set(parser.paths)),
                    "preserved_artifacts": len(protocol.get("preserved_artifacts", {})),
                    "planner_execution_or_grasp_stability_verified": False, "browser_rendering_tested": False}
    verification.update(
        verification_revision="frontview-verification-1.1",
        base_frozen_verifier="scripts/verify_task_grounding_frontview.py",
        base_frozen_verifier_sha256=file_sha(ROOT / "scripts/verify_task_grounding_frontview.py"),
        new_verifier="scripts/verify_task_grounding_frontview_v2.py",
        new_verifier_sha256=file_sha(ROOT / "scripts/verify_task_grounding_frontview_v2.py"),
        repair_scope="Presentation assertion only: recognize the actual camera-intervention and development-audit paragraphs",
        api_requests_changed=False, saved_responses_changed=False, numerical_compiler_changed=False,
        semantic_grading_changed=False, report_body_changed=False,
    )
    (report / "verification.json").write_text(json_text(verification) + "\n")
    return verification



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--trial", default=DEFAULT_TRIAL)
    parser.add_argument("--report", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json_text(verify(args.config, args.trial, args.report)))


if __name__ == "__main__":
    main()
