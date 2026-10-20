"""Freeze two reasoning-enabled runs against the already saved Mini baseline."""

import hashlib
import json
from pathlib import Path

from vlm_tamp.dataset import build_request, read_records, validate_dataset
from vlm_tamp.experiment import load_config
from vlm_tamp.grounding import strict_output_schema
from vlm_tamp.libero import ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    comparison_path = ROOT / "configs/frozen_reasoning_comparison_protocol.json"
    if comparison_path.exists():
        raise FileExistsError("Reasoning comparison already frozen; preserve its protocol")
    base_path = ROOT / "results/expanded-grounding-raw"
    baseline = json.loads((base_path / "manifest.json").read_text())
    base_config = json.loads((ROOT / "configs/expanded_grounding.json").read_text())
    dataset = ROOT / "data/expanded-grounding"
    records, _ = read_records(dataset)
    assert len(records) == 10 and baseline["offline_mock"] is False
    assert baseline["dataset_sha256"] == sha(dataset / "records.jsonl")
    access = json.loads((ROOT / "results/reasoning-model-access-check.json").read_text())
    assert all(r["metadata_accessible"] for r in access["checks"])
    common = dict(
        base_config,
        temperature=None,
        reasoning_effort="medium",
        service_tier="default",
        capture_usage_details=True,
        stop_on_api_failure=True,
        max_output_tokens=25000,
        timeout_seconds=180,
        max_requests=20,
        max_retries=0,
    )
    arms = {
        "mini-medium": {
            "model": "gpt-5.4-mini-2026-03-17",
            "max_budget_usd": 0.75,
            "pricing": {"input_per_million": 0.75, "output_per_million": 4.5},
            "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
        },
        "astra-medium": {
            "model": "gpt-6-astra",
            "max_budget_usd": 8.0,
            "pricing": {
                "input_per_million": 10.0,
                "cache_write_per_million": 12.5,
                "output_per_million": 50.0,
            },
            "pricing_source": "https://developers.openai.com/api/docs/models/gpt-6-astra",
        },
    }
    prompt = (ROOT / base_config["prompt"]).read_text()
    schema_hash = hashlib.sha256(
        json.dumps(strict_output_schema(base_config["schema_version"]), sort_keys=True).encode()
    ).hexdigest()
    arm_paths = {}
    for arm, settings in arms.items():
        config = dict(common, **settings)
        config["frozen_protocol"] = f"configs/frozen_{arm.replace('-', '_')}_protocol.json"
        config_path = ROOT / f"configs/reasoning_{arm.replace('-', '_')}.json"
        protocol_path = ROOT / config["frozen_protocol"]
        if config_path.exists() or protocol_path.exists():
            raise FileExistsError("Reasoning arm already exists")
        protocol = {
            "schema_version": config["schema_version"],
            "schema_sha256": schema_hash,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "models": [config["model"]],
            **{
                key: config[key]
                for key in (
                    "example_ids",
                    "conditions",
                    "temperature",
                    "image_detail",
                    "max_output_tokens",
                    "reasoning_effort",
                    "service_tier",
                    "capture_usage_details",
                    "stop_on_api_failure",
                    "pricing",
                    "max_requests",
                    "max_retries",
                    "max_budget_usd",
                )
            },
            "dataset_sha256": sha(dataset / "records.jsonl"),
            "evaluator_sha256": sha(ROOT / "src/vlm_tamp/evaluation.py"),
            "linker_protocol_sha256": sha(ROOT / "configs/frozen_symbol_linker_protocol.json"),
            "experiment_sha256": sha(ROOT / "src/vlm_tamp/experiment.py"),
            "provider_sha256": sha(ROOT / "src/vlm_tamp/providers.py"),
            "inference_model_version": "Mini uses a dated snapshot; Astra uses its documented named model ID and records actual returned IDs.",
        }
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        load_config(config_path)
        arm_paths[arm] = str(config_path.relative_to(ROOT))
    previous_cost, previous_calls, history = 0.0, 0, []
    for path in sorted((ROOT / "results").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        ledger = manifest.get("ledger") or {}
        if (
            ledger.get("api_attempts")
            and not manifest.get("postprocessing")
            and not manifest.get("offline_mock")
        ):
            previous_cost += ledger["estimated_usage_cost_usd"]
            previous_calls += ledger["api_attempts"]
            history.append({"manifest": str(path.relative_to(ROOT)), "sha256": sha(path)})
    protocol = {
        "baseline_raw": str(base_path.relative_to(ROOT)),
        "baseline_linked": "results/expanded-grounding-linked",
        "baseline_manifest_sha256": sha(base_path / "manifest.json"),
        "dataset_sha256": sha(dataset / "records.jsonl"),
        "schema_sha256": schema_hash,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "configs": arm_paths,
        "maximum_new_calls": 40,
        "maximum_new_estimated_usage_usd": 8.75,
        "previous_api_attempts": previous_calls,
        "previous_estimated_usage_usd": previous_cost,
        "prior_paid_history": history,
        "primary_endpoints": [
            "raw and linked request-bound schema validity",
            "strict complete ordered instance-goal agreement",
            "visible whole-object goal-role identity",
            "image point-in-mask and visible-target localization",
            "latency, input/output/reasoning tokens and conservative cost",
        ],
        "controlled_backbone_pair": ["mini-medium", "astra-medium"],
        "baseline_comparison_limit": "Saved Mini-none results used temperature 0 and a 1500-token cap. Both new arms use no temperature and a 25000-token cap that includes reasoning. Mini-none versus Mini-medium is a practical configuration comparison, not an isolated reasoning intervention.",
        "interpretation": "Ten previously inspected diagnostic tasks, not a fresh held-out test. One output per task/condition. No prompt, schema, perception, ontology, linker or scoring changes. Inventory remains simulator-assisted. No constraint expert labels or robot execution.",
        "pricing_sources": [v["pricing_source"] for v in arms.values()],
        "pricing_checked": "2026-10-11",
    }
    comparison_path.write_text(json.dumps(protocol, indent=2) + "\n")
    audit = []
    for record in records:
        for condition in common["conditions"]:
            request = build_request(record["model_inputs"], dataset, condition, prompt)
            assert set(request.payload) == (
                {"instruction", "candidates"} if condition == "candidates" else {"instruction"}
            )
            audit.append(
                {
                    "example_id": record["example_id"],
                    "condition": condition,
                    "payload_fields": sorted(request.payload),
                    "payload_sha256": hashlib.sha256(request.text.encode()).hexdigest(),
                    "image_sha256": request.image_hash,
                }
            )
    (ROOT / "results/payload-audit-reasoning-comparison.json").write_text(
        json.dumps(
            {
                "dataset_validation": validate_dataset(dataset),
                "requests_shared_by_both_new_arms": audit,
                "reference_answers_sent": False,
                "private_code_or_credentials_sent_as_content": False,
                "inference_calls_made": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "arms": arm_paths,
                "maximum_new_calls": 40,
                "maximum_new_estimated_usage_usd": 8.75,
                "previous_api_attempts": previous_calls,
                "previous_estimated_usage_usd": previous_cost,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
