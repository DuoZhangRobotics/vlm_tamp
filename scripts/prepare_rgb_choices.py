"""Freeze image-only choices before measuring offline candidate coverage."""

import json
import cv2
import numpy as np
import PIL
import scipy
from PIL import Image

from vlm_tamp.dataset import read_records, safe_path
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_choice_trial import audit_request, file_sha
from vlm_tamp.rgb_choices import REGION_SETTINGS, build_choice_request, digest
from vlm_tamp.rgb_evaluation import load_references


def main():
    config_path = ROOT / "configs/rgb_choices_trial.json"
    protocol_path = ROOT / "configs/frozen_rgb_choices_trial.json"
    if config_path.exists() or protocol_path.exists():
        raise FileExistsError("Preserve the existing frozen choice trial")
    config = json.loads((ROOT / "configs/rgb_semantics_trial.json").read_text())
    config.update(
        prompt="configs/prompts/rgb_choices_v1.txt",
        conditions=["rgb_choices"],
        frozen_protocol="configs/frozen_rgb_choices_trial.json",
        image_token_allowance=16384,
    )
    dataset = ROOT / "data/expanded-grounding"
    records, root = read_records(dataset)
    by_id = {r["example_id"]: r for r in records}
    records = [by_id[e] for e in config["example_ids"]]
    prompt = (ROOT / config["prompt"]).read_text()
    requests = {
        record["example_id"]: build_choice_request(
            record["model_inputs"]["instruction"],
            safe_path(root, record["model_inputs"]["image"]).read_bytes(),
            prompt,
        )
        for record in records
    }
    names = [
        "rgb_choices.py",
        "rgb_choice_trial.py",
        "rgb_evaluation.py",
        "rgb_grounding.py",
        "dataset.py",
        "experiment.py",
        "providers.py",
        "grounding.py",
    ]
    protocol = {
        "config": config,
        "dataset_sha256": file_sha(root / "records.jsonl"),
        "prompt_sha256": digest(prompt.encode()),
        "prompt_words": len(prompt.split()),
        "references_sha256": file_sha(ROOT / config["references"]),
        "region_settings": REGION_SETTINGS,
        "requests": {identifier: audit_request(request) for identifier, request in requests.items()},
        "implementation_hashes": {
            name: {"path": "src/vlm_tamp/" + name, "sha256": file_sha(ROOT / "src/vlm_tamp" / name)}
            for name in names
        },
        "runtime_versions": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "Pillow": PIL.__version__,
        },
        "baseline": "results/rgb-repair-mini; same model, reasoning, standard service, output allowance and semantic evaluator",
        "comparison_scope": "Form, RGB region proposals and enlarged visual presentation change together; ten inspected development tasks, not a held-out generalization claim",
        "choices_source": "RGB bytes only; no language-conditioned proposal ranking, simulator inventory, masks, categories, poses or answers",
        "metric": "Unchanged rgb-semantic-1.0 final-goal evaluation; full constraints and execution untested",
        "baseline_manifest_sha256": file_sha(ROOT / "results/rgb-repair-mini/manifest.json"),
        "prior_api_attempts": 184,
        "prior_estimated_usage_usd": 2.93177035,
        "legacy_evaluator_sha256": file_sha(ROOT / "src/vlm_tamp/evaluation.py"),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    # These independently measured masks are used ONLY after freezing, to
    # quantify the ceiling imposed by image-generated choices. They do not
    # select, rank, remove, relabel or regenerate a single proposal.
    references = load_references(dataset, ROOT / config["references"])
    coverage = []
    for record in records:
        request = requests[record["example_id"]]
        masks = {
            identifier: np.asarray(Image.open(safe_path(root, ref["mask"]))) > 0
            for identifier, ref in record["reference"]["visual"].items()
            if ref.get("kind", "object") == "object"
        }
        independently_resolved = {}
        for region in request.regions:
            x, y = region["pixel"]
            hits = [identifier for identifier, mask in masks.items() if mask[y, x]]
            independently_resolved[region["choice"]] = hits[0] if len(hits) == 1 else None
        required = sorted({arg[0] for goal in references[record["example_id"]]["goals"] for arg in goal[1]})
        covered = {
            identifier: [
                choice for choice, resolved in independently_resolved.items() if resolved == identifier
            ]
            for identifier in required
        }
        coverage.append(
            {
                "example_id": record["example_id"],
                "choices": len(request.regions),
                "all_required_owners_covered": all(covered.values()),
                "required_owner_choices_offline_only": covered,
                "required_owners": len(required),
                "covered_owners": sum(bool(c) for c in covered.values()),
            }
        )
    output = ROOT / "results/rgb-choice-preflight.json"
    output.write_text(
        json.dumps(
            {
                "protocol_sha256": file_sha(protocol_path),
                "coverage_measured_after_freezing": True,
                "coverage_never_sent_to_model": True,
                "tasks_with_full_candidate_coverage": sum(c["all_required_owners_covered"] for c in coverage),
                "candidate_coverage_is_not_model_success": True,
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
                "tasks_with_full_candidate_coverage": sum(c["all_required_owners_covered"] for c in coverage),
                "prompt_words": len(prompt.split()),
                "estimated_spending_cap_usd": config["max_budget_usd"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
