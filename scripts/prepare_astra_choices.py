"""Freeze an Astra arm with the identical, already-frozen RGB choices."""

import copy
import json

from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_choice_trial import audit_request, file_sha, load_choice_config, request_arguments
from vlm_tamp.rgb_choices import build_choice_request


def main():
    config_path = ROOT / "configs/rgb_choices_astra_trial.json"
    protocol_path = ROOT / "configs/frozen_rgb_choices_astra_trial.json"
    if config_path.exists() or protocol_path.exists():
        raise FileExistsError("Preserve the existing frozen Astra choice arm")
    baseline_config, baseline_protocol, prompt = load_choice_config(ROOT / "configs/rgb_choices_trial.json")
    config = copy.deepcopy(baseline_config)
    config.update(
        model="gpt-6-astra",
        max_output_tokens=4000,
        max_budget_usd=1.0,
        image_token_allowance=4096,
        frozen_protocol="configs/frozen_rgb_choices_astra_trial.json",
        pricing={"input_per_million": 12.5, "cache_write_per_million": 12.5, "output_per_million": 50.0},
        pricing_source="https://developers.openai.com/api/docs/models/gpt-6-astra",
        pricing_checked="2026-10-12",
        accounting_notes="Conservatively charge every input token at the $12.50 cache-write ceiling; ordinary input is $10.00, cached reads $1.00. Output includes reasoning once.",
    )
    baseline = ROOT / "results/rgb-choice-mini"
    equality = []
    for index, identifier in enumerate(config["example_ids"]):
        folder = baseline / f"input-{index:02d}"
        old_args = json.loads((folder / "request.json").read_text())
        payload = json.loads(old_args["input"][0]["content"][0]["text"])
        request = build_choice_request(payload["instruction"], (folder / "original.png").read_bytes(), prompt)
        assert audit_request(request) == baseline_protocol["requests"][identifier]
        new_args = request_arguments(config, request)
        differences = {key for key in new_args if new_args[key] != old_args[key]}
        assert differences == {"model", "max_output_tokens"}
        equality.append(
            {
                "example_id": identifier,
                "input_images_prompt_schema_equal": True,
                "changed_inference_arguments": sorted(differences),
            }
        )
    protocol = copy.deepcopy(baseline_protocol)
    protocol.update(
        config=config,
        baseline="results/rgb-choice-mini: same RGB choices, prompt, schema, reasoning effort, evaluator and references",
        baseline_manifest_sha256=file_sha(baseline / "manifest.json"),
        baseline_protocol_sha256=file_sha(ROOT / "configs/frozen_rgb_choices_trial.json"),
        baseline_artifact="results/rgb-choice-mini",
        public_request_comparison=equality,
        inference_bound_difference={
            "mini_max_output_tokens": 8000,
            "astra_max_output_tokens": 4000,
            "reason": "Bound cost for the short form; any truncation remains a failed trial output",
        },
        comparison_scope="Same ten inspected development tasks, RGB choices, prompt, schema, medium reasoning and evaluator. Astra output bound is 4000 versus Mini 8000. Not held-out accuracy or robot execution.",
        prior_api_attempts=194,
        prior_estimated_usage_usd=3.0156871,
        accounting_note=config["accounting_notes"],
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {
                "model": config["model"],
                "planned_calls": config["max_requests"],
                "estimated_spending_cap_usd": config["max_budget_usd"],
                "identical_input_comparisons": len(equality),
                "max_output_tokens": config["max_output_tokens"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
