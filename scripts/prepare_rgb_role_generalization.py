"""Collect and freeze new task groups with the existing RGB prompt and compiler."""

import copy
import json
from collections import Counter
from pathlib import Path

from vlm_tamp import libero as collector
from vlm_tamp.bddl import parse_bddl
from vlm_tamp.dataset import read_records, safe_path, validate_dataset
from vlm_tamp.libero import ROOT, configure_libero
from vlm_tamp.rgb_evaluation import freeze_references
from vlm_tamp.rgb_role_diagnostics import template_contract
from vlm_tamp.rgb_role_generalization_trial import audit_request, file_sha
from vlm_tamp.rgb_role_points import build_point_request, digest


def main():
    selection_path = ROOT / "configs/rgb_role_generalization_selection.json"
    selection = json.loads(selection_path.read_text())
    dataset = ROOT / "data/rgb-role-generalization"
    config_path = ROOT / "configs/rgb_role_generalization_trial.json"
    protocol_path = ROOT / "configs/frozen_rgb_role_generalization_trial.json"
    refs_path = ROOT / "configs/rgb_role_generalization_references.json"
    roles_path = ROOT / "configs/rgb_role_generalization_roles.json"
    if any(path.exists() for path in (dataset, config_path, protocol_path, refs_path, roles_path)):
        raise FileExistsError("Preserve the prepared generalization experiment")
    configure_libero(gl="egl")
    from libero.libero import benchmark

    prior_names, prior_keys, prior_specs, preserved = set(), set(), set(), []
    history, prior_calls, prior_cost = [], 0, 0.0
    for path in sorted((ROOT / "results").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        ledger = manifest.get("ledger") or {}
        if manifest.get("offline_mock") or manifest.get("postprocessing") or not ledger.get("api_attempts"):
            continue
        history.append({"path": str(path.relative_to(ROOT)), "sha256": file_sha(path)})
        prior_calls += ledger["api_attempts"]
        prior_cost += ledger["estimated_usage_cost_usd"]
        preserved.append(path)
        for name in ("predictions.jsonl", "metrics.jsonl", "summary.json"):
            if (path.parent / name).exists():
                preserved.append(path.parent / name)
        if not manifest.get("dataset"):
            continue
        predictions_path = path.parent / "predictions.jsonl"
        rows = [json.loads(line) for line in predictions_path.read_text().splitlines()] if predictions_path.exists() else []
        queried = {row["example_id"] for row in rows if row.get("attempts", 1) > 0}
        records, _ = read_records(manifest["dataset"])
        for record in records:
            if record["example_id"] not in queried:
                continue
            meta = record["metadata"]
            prior_names.add(meta["task_name"])
            prior_keys.add((meta["suite"], meta["task_index"]))
            prior_specs.add(meta["bddl_sha256"])
    catalogue, selected = [], []
    wanted = {(suite, index) for suite, indices in selection["suites"].items() for index in indices}
    for suite_name in ("libero_goal", "libero_spatial", "libero_object", "libero_10", "libero_90"):
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        for index in range(suite.n_tasks):
            task = suite.get_task(index)
            spec_path = Path(suite.get_task_bddl_file_path(index))
            parsed = parse_bddl(spec_path)
            contract = template_contract({"model_inputs": {"instruction": task.language}, "reference": parsed})
            fresh = ((suite_name, index) not in prior_keys and task.name not in prior_names
                     and file_sha(spec_path) not in prior_specs)
            entry = {"suite": suite_name, "task_index": index, "task_name": task.name,
                     "instruction": task.language, "bddl_sha256": file_sha(spec_path),
                     "new_task_group": fresh, **contract}
            catalogue.append(entry)
            if (suite_name, index) in wanted:
                assert fresh, "Requested task already queried"
                assert contract["supported"] and contract["role_owners"], "Selected contract cannot be graded"
                selected.append(entry)
    assert len(selected) == selection["maximum_calls"] == 20
    ordered = {(task["suite"], task["task_index"]): task for task in selected}
    selected = [ordered[(suite, index)] for suite, indices in selection["suites"].items() for index in indices]
    for number, task in enumerate(selected):
        task["state_index"] = selection["state_indices_alternating"][number % 2]
    coverage = {
        "catalogue_tasks": len(catalogue),
        "new_task_groups": sum(task["new_task_group"] for task in catalogue),
        "supported_structures": sum(task["supported"] for task in catalogue),
        "new_supported_structures": sum(task["supported"] and task["new_task_group"] for task in catalogue),
        "exclusion_reasons": dict(Counter(task["reason"] for task in catalogue if not task["supported"])),
        "scope": "Goal-structure/ownership coverage audit only; not arbitrary-language equivalence or VLM success",
    }
    plan = {"selection": selection, "selection_sha256": file_sha(selection_path),
            "coverage": coverage, "selected": selected, "catalogue": catalogue,
            "selection_frozen_before_rendering_and_inference": True,
            "prior_paid_history": history, "prior_api_attempts": prior_calls,
            "prior_estimated_usage_usd": prior_cost}
    plan_path = ROOT / "results/rgb-role-generalization-plan.json"
    if plan_path.exists():
        raise FileExistsError("Preserve the previously fixed selection plan")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"selected_new_tasks": len(selected), "coverage": coverage,
                      "spending_cap_usd": selection["max_budget_usd"]}), flush=True)

    records = []
    # Extend only this collector process's suite allowlist. The rendering code,
    # image convention, instance references and historical source stay intact.
    original_suites = collector.SUITES
    try:
        collector.SUITES = (*original_suites, "libero_90")
        for task in selected:
            records.extend(collector.collect(
                dataset, [task["suite"]], [task["task_index"]], [task["state_index"]],
                selection["collection_seed"], selection["image_size"], "egl"))
    finally:
        collector.SUITES = original_suites
    (dataset / "records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    validation = validate_dataset(dataset)
    references = freeze_references(dataset, refs_path)
    config = copy.deepcopy(json.loads((ROOT / "configs/rgb_role_points_trial.json").read_text()))
    config.update(example_ids=[record["example_id"] for record in records],
                  frozen_protocol=str(protocol_path.relative_to(ROOT)),
                  references=str(refs_path.relative_to(ROOT)), max_requests=20,
                  max_budget_usd=selection["max_budget_usd"])
    prompt = (ROOT / config["prompt"]).read_text()
    baseline = json.loads((ROOT / "results/rgb-role-points-mini/manifest.json").read_text())
    assert prompt == baseline["prompt_text"]
    for key in ("model", "image_detail", "reasoning_effort", "max_output_tokens", "service_tier"):
        assert config[key] == baseline["config"][key]
    requests = {}
    offline_roles, visibility = {}, []
    for record in records:
        request = build_point_request(record["model_inputs"]["instruction"],
                                      safe_path(dataset, record["model_inputs"]["image"]).read_bytes(), prompt)
        requests[record["example_id"]] = audit_request(request)
        contract = template_contract(record)
        assert contract["supported"] and contract["role_owners"]
        assert references[record["example_id"]]["evaluable"]
        offline_roles[record["example_id"]] = contract["role_owners"]
        visibility.append({"example_id": record["example_id"], "required_owners": len(contract["role_owners"]),
                           "visible_required_owners": sum(owner in record["reference"]["visual"]
                                                          for owner in contract["role_owners"].values())})
    roles_path.write_text(json.dumps({"offline_only": True, "roles": offline_roles}, indent=2) + "\n")
    modules = (
        "rgb_role_generalization_trial.py", "rgb_role_point_trial.py", "rgb_role_points.py",
        "rgb_role_choices.py", "rgb_object_choices.py", "rgb_choices_v2.py", "rgb_choices.py",
        "rgb_evaluation.py", "rgb_grounding.py", "rgb_role_diagnostics.py", "dataset.py",
        "experiment.py", "providers.py", "grounding.py", "libero.py", "bddl.py",
    )
    preserved.extend([ROOT / "src/vlm_tamp/evaluation.py",
                      ROOT / "configs/frozen_rgb_role_points_trial.json",
                      ROOT / "results/rgb-role-points-report/summary.json",
                      ROOT / "results/rgb-role-points-report/index.html"])
    protocol = {
        "config": config, "dataset": str(dataset.relative_to(ROOT)),
        "dataset_sha256": file_sha(dataset / "records.jsonl"),
        "prompt_sha256": digest(prompt.encode()), "references_sha256": file_sha(refs_path),
        "offline_roles_path": str(roles_path.relative_to(ROOT)), "offline_roles_sha256": file_sha(roles_path),
        "selection_plan_path": str(plan_path.relative_to(ROOT)), "selection_plan_sha256": file_sha(plan_path),
        "requests": requests, "coverage": coverage, "required_owner_visibility": visibility,
        "implementation_hashes": {
            name: {"path": "src/vlm_tamp/" + name, "sha256": file_sha(ROOT / "src/vlm_tamp" / name)}
            for name in modules
        },
        "preserved_artifacts": {
            str(path.relative_to(ROOT)): {"path": str(path.relative_to(ROOT)), "sha256": file_sha(path)}
            for path in preserved
        },
        "baseline": "results/rgb-role-points-mini", "prior_api_attempts": prior_calls,
        "prior_estimated_usage_usd": prior_cost, "frozen_before_inference": True,
        "input_image_count": 1, "image_resolution": 512, "segmentation_used": False, "crops_used": False,
        "inventory_provided": False, "task_held_out_from_project_queries": True,
        "comparison_scope": "Twenty preselected task groups disjoint from all earlier paid queries by suite/index, task name and BDDL hash; one new initial arrangement per task. Same raw RGB prompt, role parser, coordinate schema, goal compiler, model/settings and primary semantic grader. Purposive supported-interface sample, not whole-LIBERO accuracy or a matched difficulty comparison with the old ten-scene score.",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"frozen_new_tasks": len(records), "dataset_validation": validation,
                      "visible_roles": sum(item["visible_required_owners"] for item in visibility),
                      "required_roles": sum(item["required_owners"] for item in visibility),
                      "prompt_unchanged": True, "model_calls_made": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
