"""Collect and freeze a bounded task expansion. This script never calls a VLM."""

import hashlib
import json
from pathlib import Path

from vlm_tamp.dataset import build_request, read_records, validate_dataset
from vlm_tamp.experiment import load_config
from vlm_tamp.grounding import strict_output_schema
from vlm_tamp.libero import ROOT, collect


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    selection_path = ROOT / "configs/expanded_task_selection.json"
    selection = json.loads(selection_path.read_text())
    dataset = ROOT / "data/expanded-grounding"
    protocol_path = ROOT / "configs/frozen_expanded_grounding_protocol.json"
    config_path = ROOT / "configs/expanded_grounding.json"
    if dataset.exists() or protocol_path.exists() or config_path.exists():
        raise FileExistsError("Expansion already prepared; preserve it and use its saved configuration")

    history, previous_cost, previous_calls, prior_tasks = [], 0.0, 0, set()
    for path in sorted((ROOT / "results").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        ledger = manifest.get("ledger") or {}
        if not ledger.get("api_attempts") or manifest.get("postprocessing") or manifest.get("offline_mock"):
            continue
        history.append({"manifest": str(path.relative_to(ROOT)), "sha256": sha(path)})
        previous_cost += ledger["estimated_usage_cost_usd"]
        previous_calls += ledger["api_attempts"]
        if manifest.get("dataset"):
            records, _ = read_records(manifest["dataset"])
            selected = set(manifest.get("config", {}).get("example_ids") or [])
            if (path.parent / "predictions.jsonl").exists():
                selected.update(
                    json.loads(line)["example_id"]
                    for line in (path.parent / "predictions.jsonl").read_text().splitlines()
                )
            for record in records:
                if not selected or record["example_id"] in selected:
                    prior_tasks.add((record["metadata"]["suite"], record["metadata"]["task_index"]))
    wanted = {(suite, task) for suite, tasks in selection["suites"].items() for task in tasks}
    if wanted & prior_tasks:
        raise ValueError("Selected task groups already appear in paid inference history")
    if previous_cost + selection["max_budget_usd"] > selection["maximum_cumulative_estimated_usage_usd"]:
        raise ValueError("Expansion exceeds the cumulative estimated-usage bound")

    records = []
    for suite, tasks in selection["suites"].items():
        records.extend(
            collect(
                dataset,
                [suite],
                tasks,
                [selection["state_index"]],
                selection["collection_seed"],
                selection["image_size"],
                "egl",
            )
        )
    (dataset / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    assert len(records) == 10 and len({r["example_id"] for r in records}) == 10
    validation = validate_dataset(dataset)
    config = json.loads((ROOT / "configs/semantic_heldout.json").read_text())
    config.update(
        example_ids=[r["example_id"] for r in records],
        conditions=selection["conditions"],
        max_requests=selection["max_requests"],
        max_budget_usd=selection["max_budget_usd"],
        frozen_protocol=str(protocol_path.relative_to(ROOT)),
    )
    prompt = (ROOT / config["prompt"]).read_text()
    protocol = {
        "schema_version": config["schema_version"],
        "schema_sha256": hashlib.sha256(
            json.dumps(strict_output_schema(config["schema_version"]), sort_keys=True).encode()
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "models": [config["model"]],
        **{
            k: config[k]
            for k in (
                "example_ids",
                "conditions",
                "temperature",
                "image_detail",
                "max_output_tokens",
                "reasoning_effort",
                "max_retries",
            )
        },
        "selection": selection,
        "selection_sha256": sha(selection_path),
        "dataset_sha256": sha(dataset / "records.jsonl"),
        "previously_queried_task_keys": sorted(prior_tasks),
        "prior_paid_history": history,
        "previous_estimated_usage_usd": previous_cost,
        "previous_api_attempts": previous_calls,
        "maximum_inference_attempts": selection["max_requests"],
        "partition_budget_usd": selection["max_budget_usd"],
        "maximum_total_estimated_usage_usd": previous_cost + selection["max_budget_usd"],
        "linker_protocol_sha256": sha(ROOT / "configs/frozen_symbol_linker_protocol.json"),
        "evaluator_sha256": sha(ROOT / "src/vlm_tamp/evaluation.py"),
        "primary_endpoints": [
            "raw and linked schema/request validity with failures retained",
            "strict complete instance-goal agreement",
            "visible whole-object final-goal argument identity",
        ],
        "interpretation": "Purposive task expansion, one initial state each, same frozen schema/prompt/model as v5. New relative to project query history, not a claim about pretraining. Inventory is oracle-assisted. Stage/contact/path accuracy and robot execution remain unassessed.",
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    load_config(config_path)
    audit = []
    for record in records:
        for condition in config["conditions"]:
            request = build_request(record["model_inputs"], dataset, condition, prompt)
            expected = {"instruction"} | ({"candidates"} if condition == "candidates" else set())
            assert set(request.payload) == expected
            assert request.image and request.image_hash == request.source_image_sha256
            audit.append(
                {
                    "example_id": record["example_id"],
                    "condition": condition,
                    "payload_fields": sorted(request.payload),
                    "payload_sha256": hashlib.sha256(request.text.encode()).hexdigest(),
                    "image_sha256": request.image_hash,
                    "oracle_assisted": condition == "candidates",
                }
            )
    (ROOT / "results/payload-audit-expanded-grounding.json").write_text(
        json.dumps(
            {
                "dataset_validation": validation,
                "requests": audit,
                "reference_compartment_sent": False,
                "geometry_sent": False,
                "model_calls_made": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "observations": len(records),
                "planned_calls": len(audit),
                "prior_calls": previous_calls,
                "prior_estimated_usage_usd": previous_cost,
                "new_budget_usd": config["max_budget_usd"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
