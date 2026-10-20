"""Freeze both repaired Mini partitions before any new model responses."""

import copy
import json

import numpy as np
from PIL import Image

from vlm_tamp.dataset import read_records, safe_path
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_choice_v2_trial import audit_request, file_sha
from vlm_tamp.rgb_choices_v2 import REGION_SETTINGS, build_choice_request, digest
from vlm_tamp.rgb_evaluation import freeze_references


def main():
    artifacts = [
        ROOT / name
        for name in (
            "configs/rgb_choices_v2_dev.json",
            "configs/rgb_choices_v2_validation.json",
            "configs/frozen_rgb_choices_v2_dev.json",
            "configs/frozen_rgb_choices_v2_validation.json",
            "configs/rgb_choices_v2_references.json",
        )
    ]
    if any(path.exists() for path in artifacts):
        raise FileExistsError("Preserve the already frozen v2 experiment")
    dataset = ROOT / "data/rgb-choices-v2"
    records, root = read_records(dataset)
    references = freeze_references(dataset, ROOT / "configs/rgb_choices_v2_references.json")
    baseline_records, _ = read_records(ROOT / "data/expanded-grounding")
    known = {record["example_id"]: record for record in baseline_records}
    queried, history = set(), []
    for path in sorted((ROOT / "results").glob("*/predictions.jsonl")):
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("offline_mock")
            or manifest.get("postprocessing")
            or not (manifest.get("ledger") or {}).get("api_attempts")
        ):
            continue
        history.append({"path": str(manifest_path.relative_to(ROOT)), "sha256": file_sha(manifest_path)})
        queried.update(
            row["example_id"]
            for row in (json.loads(line) for line in path.read_text().splitlines())
            if row.get("attempts", 1)
        )
    validation_ids = {record["example_id"] for record in records if record["metadata"]["state_index"] == 1}
    assert len(validation_ids) == 10 and not validation_ids & queried
    modules = [
        "rgb_choices_v2.py",
        "rgb_choice_v2_trial.py",
        "rgb_choices.py",
        "rgb_evaluation.py",
        "rgb_grounding.py",
        "dataset.py",
        "experiment.py",
        "providers.py",
        "grounding.py",
        "libero.py",
    ]
    base = json.loads((ROOT / "configs/rgb_choices_trial.json").read_text())
    base.update(
        prompt="configs/prompts/rgb_choices_v2.txt",
        conditions=["rgb_choices_v2"],
        references="configs/rgb_choices_v2_references.json",
        max_budget_usd=0.25,
    )
    prompt = (ROOT / base["prompt"]).read_text()
    partitions, coverage = {}, []
    for partition, state in (("dev", 0), ("validation", 1)):
        selected = [record for record in records if record["metadata"]["state_index"] == state]
        assert len(selected) == 10
        config = copy.deepcopy(base)
        config.update(
            example_ids=[record["example_id"] for record in selected],
            frozen_protocol=f"configs/frozen_rgb_choices_v2_{partition}.json",
        )
        requests = {
            record["example_id"]: build_choice_request(
                record["model_inputs"]["instruction"],
                safe_path(root, record["model_inputs"]["image"]).read_bytes(),
                prompt,
            )
            for record in selected
        }
        for record in selected:
            assert record["metadata"]["camera"]["width"] == record["metadata"]["camera"]["height"] == 512
            if partition == "dev":
                assert (
                    record["metadata"]["initial_state_sha256"]
                    == known[record["example_id"]]["metadata"]["initial_state_sha256"]
                )
        protocol = {
            "config": config,
            "partition": partition,
            "dataset_sha256": file_sha(root / "records.jsonl"),
            "prompt_sha256": digest(prompt.encode()),
            "references_sha256": file_sha(ROOT / config["references"]),
            "prompt_words": len(prompt.split()),
            "requests": {identifier: audit_request(request) for identifier, request in requests.items()},
            "implementation_hashes": {
                module: {"path": "src/vlm_tamp/" + module, "sha256": file_sha(ROOT / "src/vlm_tamp" / module)}
                for module in modules
            },
            "region_settings": REGION_SETTINGS,
            "prior_paid_history": history,
            "validation_images_previously_queried": False,
            "same_initial_states_for_development": True,
            "comparison_scope": "Development repeats the ten inspected states with genuine 512 RGB and a revised RGB-only proposal/form pipeline. Validation uses previously unqueried initial states of the same tasks; it is not a task-held-out benchmark.",
            "previous_baseline": "results/rgb-choice-mini: 8/10 at 256 RGB under unchanged semantic grading",
            "prior_api_attempts": 204,
            "prior_estimated_usage_usd": 3.4300996,
            "combined_new_budget_usd": 0.50,
            "legacy_evaluator_sha256": file_sha(ROOT / "src/vlm_tamp/evaluation.py"),
            "previous_mini_manifest_sha256": file_sha(ROOT / "results/rgb-choice-mini/manifest.json"),
            "previous_astra_manifest_sha256": file_sha(ROOT / "results/rgb-choice-astra/manifest.json"),
            "frozen_before_either_partition_inference": True,
        }
        (ROOT / f"configs/rgb_choices_v2_{partition}.json").write_text(json.dumps(config, indent=2) + "\n")
        (ROOT / config["frozen_protocol"]).write_text(json.dumps(protocol, indent=2) + "\n")
        partitions[partition] = (selected, requests)

    # Answer masks are read only after both partitions, all rankings and all
    # inputs have been frozen. They never alter the choices.
    for partition, (selected, requests) in partitions.items():
        for record in selected:
            masks = {
                identifier: np.asarray(Image.open(safe_path(root, visual["mask"]))) > 0
                for identifier, visual in record["reference"]["visual"].items()
                if visual.get("kind", "object") == "object"
            }
            resolved = {}
            for region in requests[record["example_id"]].regions:
                x, y = region["pixel"]
                hits = [identifier for identifier, mask in masks.items() if mask[y, x]]
                resolved[region["choice"]] = hits[0] if len(hits) == 1 else None
            required = sorted(
                {argument[0] for goal in references[record["example_id"]]["goals"] for argument in goal[1]}
            )
            options = {
                identifier: [choice for choice, owner in resolved.items() if owner == identifier]
                for identifier in required
            }
            coverage.append(
                {
                    "example_id": record["example_id"],
                    "partition": partition,
                    "choices": len(requests[record["example_id"]].regions),
                    "all_required_owners_covered": all(options.values()),
                    "required_owners": len(options),
                    "covered_owners": sum(bool(value) for value in options.values()),
                    "required_owner_choices_offline_only": options,
                }
            )
    (ROOT / "results/rgb-choice-v2-preflight.json").write_text(
        json.dumps(
            {
                "choices_frozen_before_coverage": True,
                "coverage_is_not_model_success": True,
                "references_never_sent_to_model": True,
                "tasks": coverage,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "prepared_tasks": len(records),
                "coverage_by_partition": {
                    partition: sum(
                        task["all_required_owners_covered"]
                        for task in coverage
                        if task["partition"] == partition
                    )
                    for partition in partitions
                },
                "prompt_words": len(prompt.split()),
                "combined_new_spending_cap_usd": 0.5,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
