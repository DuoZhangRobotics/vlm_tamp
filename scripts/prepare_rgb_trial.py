"""Freeze an RGB-only repair trial and independently test its reference coverage."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_tamp.dataset import build_request, read_records, safe_path
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_evaluation import evaluate_rgb, freeze_references
from vlm_tamp.rgb_grounding import RGBPrediction, rgb_output_schema


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config_path = ROOT / "configs/rgb_semantics_trial.json"
    protocol_path = ROOT / "configs/frozen_rgb_semantics_trial.json"
    references_path = ROOT / "configs/rgb_semantic_references_v1.json"
    if any(p.exists() for p in (config_path, protocol_path, references_path)):
        raise FileExistsError("RGB trial is already prepared; preserve its frozen artifacts")
    dataset = ROOT / "data/expanded-grounding"
    records, root = read_records(dataset)
    references = freeze_references(dataset, references_path)
    controls = []
    for record in records:
        reference = references[record["example_id"]]
        object_ids = sorted({arg[0] for goal in reference["goals"] for arg in goal[1]})
        entities, id_map = [], {}
        for index, identifier in enumerate(object_ids):
            local = "object_" + str(index)
            mask = (
                np.asarray(Image.open(safe_path(root, record["reference"]["visual"][identifier]["mask"]))) > 0
            )
            ys, xs = np.where(mask)
            entities.append(
                {
                    "id": local,
                    "category": "synthetic test object",
                    "description": "Offline synthetic coverage check, never a model prediction",
                    "point": [
                        (float(xs[len(xs) // 2]) + 0.5) / mask.shape[1],
                        (float(ys[len(ys) // 2]) + 0.5) / mask.shape[0],
                    ],
                }
            )
            id_map[identifier, "whole"] = local
        goals = []
        for relation, arguments in reference["goals"]:
            local_args = []
            for identifier, selector in arguments:
                if selector not in ("whole", "interior") and (identifier, selector) not in id_map:
                    local = "part_" + str(len(entities))
                    entities.append(
                        {
                            "id": local,
                            "category": "synthetic part",
                            "description": "Offline semantic-selector coverage check",
                            "kind": "part",
                            "parent": id_map[identifier, "whole"],
                            "selector": selector,
                        }
                    )
                    id_map[identifier, selector] = local
                local_args.append(
                    id_map[identifier, "whole"]
                    if selector in ("whole", "interior")
                    else id_map[identifier, selector]
                )
            goals.append({"relation": relation, "arguments": local_args})
        synthetic = RGBPrediction.model_validate({"entities": entities, "goals": goals})
        result = evaluate_rgb(record, synthetic, reference, root)
        assert result["semantic_goal"]["exact_match"], record["example_id"]
        corrupted = synthetic.model_dump(mode="json")
        corrupted["goals"].append({"relation": "Open", "arguments": [entities[0]["id"]]})
        assert not evaluate_rgb(record, RGBPrediction.model_validate(corrupted), reference, root)[
            "semantic_goal"
        ]["exact_match"]
        controls.append(
            {
                "example_id": record["example_id"],
                "synthetic_correct_goal_is_scorable": True,
                "extra_goal_rejected": True,
                "not_a_model_result": True,
            }
        )
    config = {
        "model": "gpt-5.4-mini-2026-03-17",
        "prompt": "configs/prompts/rgb_semantics_v1.txt",
        "conditions": ["rgb"],
        "example_ids": [r["example_id"] for r in records],
        "reasoning_effort": "medium",
        "temperature": None,
        "service_tier": "default",
        "image_detail": "high",
        "max_output_tokens": 8000,
        "timeout_seconds": 180,
        "max_requests": 10,
        "max_retries": 0,
        "max_budget_usd": 0.25,
        "image_token_allowance": 4096,
        "capture_usage_details": True,
        "pricing": {"input_per_million": 0.75, "output_per_million": 4.5},
        "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
        "pricing_checked": "2026-10-12",
        "references": str(references_path.relative_to(ROOT)),
        "frozen_protocol": str(protocol_path.relative_to(ROOT)),
    }
    prompt = (ROOT / config["prompt"]).read_text()
    history, previous_cost, previous_calls = [], 0.0, 0
    for path in sorted((ROOT / "results").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        ledger = manifest.get("ledger") or {}
        if (
            ledger.get("api_attempts")
            and not manifest.get("postprocessing")
            and not manifest.get("offline_mock")
        ):
            history.append({"manifest": str(path.relative_to(ROOT)), "sha256": sha(path)})
            previous_cost += ledger["estimated_usage_cost_usd"]
            previous_calls += ledger["api_attempts"]
    protocol = {
        **config,
        "dataset_sha256": sha(dataset / "records.jsonl"),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(json.dumps(rgb_output_schema(), sort_keys=True).encode()).hexdigest(),
        "references_sha256": sha(references_path),
        "implementation_hashes": {
            name: {"path": f"src/vlm_tamp/{name}.py", "sha256": sha(ROOT / f"src/vlm_tamp/{name}.py")}
            for name in ("rgb_grounding", "rgb_evaluation", "rgb_trial")
        },
        "previous_cost_usd": previous_cost,
        "previous_calls": previous_calls,
        "prior_paid_history": history,
        "legacy_evaluator_sha256": sha(ROOT / "src/vlm_tamp/evaluation.py"),
        "semantic_metric_changed": True,
        "metric_scope": "Natural-language desired relations with image-identified whole objects and explicit owned-part selectors. Does not require simulator region IDs; does not certify region/part pixel localization, full containment, grasps or robot execution.",
        "baseline_comparison": "Old RGB-only outputs can be separately adapted to this same evaluator. Raw BDDL scores remain unchanged. Prompt and representation changes are coupled. No inventory, masks, IDs, poses or reference answers enter inference.",
        "development_tasks_not_fresh_heldout": True,
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    audit = []
    for record in records:
        request = build_request(record["model_inputs"], dataset, "rgb", prompt)
        assert set(request.payload) == {"instruction"} and request.image_hash == request.source_image_sha256
        audit.append(
            {
                "example_id": record["example_id"],
                "payload_fields": list(request.payload),
                "payload_sha256": hashlib.sha256(request.text.encode()).hexdigest(),
                "image_sha256": request.image_hash,
            }
        )
    (ROOT / "results/rgb-repair-preflight.json").write_text(
        json.dumps(
            {
                "requests": audit,
                "synthetic_coverage_checks": controls,
                "scorable_tasks": len(controls),
                "synthetic_checks_are_not_vlm_successes": True,
                "old_prompt_words": len((ROOT / "configs/prompts/structured_v5.txt").read_text().split()),
                "new_prompt_words": len(prompt.split()),
                "model_calls_made": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "tasks": len(records),
                "input_condition": "rgb only",
                "prompt_words": len(prompt.split()),
                "maximum_new_calls": 10,
                "estimated_budget_usd": 0.25,
                "synthetic_reference_coverage": len(controls),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
