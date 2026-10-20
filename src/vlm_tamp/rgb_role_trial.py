"""Version 2 RGB-only small-region and disjoint-state trial; v1 remains frozen."""

import base64
import json
import os
import time
from pathlib import Path

from pydantic import ValidationError

from .dataset import read_records, safe_path
from .experiment import BudgetExceeded, Ledger
from .libero import ROOT
from .providers import ProviderResponse
from .rgb_role_choices import (
    build_choice_request,
    binding_schema,
    compile_bindings,
    validate_bindings,
    digest,
)
from .rgb_evaluation import evaluate_rgb, load_references


def file_sha(path):
    return digest(Path(path).read_bytes())


def request_arguments(config, request):
    # Rebuild from the original RGB and language, not from a stored inventory or
    # any externally supplied mapping. Verify the public finite-choice payload.
    expected = build_choice_request(request.payload["instruction"], request.source_image, request.prompt)
    if (
        set(request.payload) != {"instruction", "visual_choices", "roles"}
        or request.payload != expected.payload
        or request.regions != expected.regions
        or request.scene_image != expected.scene_image
        or request.cards_image != expected.cards_image
    ):
        raise ValueError("Visual choices must be generated from the original RGB only")
    content = [{"type": "input_text", "text": request.text}]
    for title, image in (
        ("Original RGB", request.source_image),
        ("Numbered scene", request.scene_image),
        ("Numbered context crops", request.cards_image),
    ):
        content.extend(
            [
                {"type": "input_text", "text": title},
                {
                    "type": "input_image",
                    "detail": config["image_detail"],
                    "image_url": "data:image/png;base64," + base64.b64encode(image).decode(),
                },
            ]
        )
    return {
        "model": config["model"],
        "instructions": request.prompt,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ChoicePrediction",
                "schema": binding_schema(request.payload["instruction"], len(request.regions)),
                "strict": True,
            }
        },
        "reasoning": {"effort": config["reasoning_effort"]},
        "service_tier": "default",
        "max_output_tokens": config["max_output_tokens"],
        "store": False,
    }


class ChoiceOpenAIProvider:
    def __init__(self, config, client=None):
        from openai import OpenAI

        self.config = config
        self.client = client or OpenAI(timeout=config["timeout_seconds"], max_retries=0)

    def infer(self, request):
        response = self.client.responses.create(**request_arguments(self.config, request))
        return ProviderResponse(
            response.output_text,
            response.model_dump(mode="json"),
            response.usage.input_tokens if response.usage else 0,
            response.usage.output_tokens if response.usage else 0,
        )


def load_choice_config(path):
    import math

    config = json.loads(Path(path).read_text())
    if (
        config["conditions"] != ["rgb_role_choices"]
        or config["max_retries"] != 0
        or config["service_tier"] != "default"
        or config["temperature"] is not None
        or config["reasoning_effort"] != "medium"
    ):
        raise ValueError("Choice trial must use frozen RGB-only standard settings")
    for key in (
        "max_requests",
        "max_output_tokens",
        "max_budget_usd",
        "timeout_seconds",
        "image_token_allowance",
    ):
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Invalid positive choice bound: " + key)
    for key in ("max_requests", "max_output_tokens", "image_token_allowance"):
        if not isinstance(config[key], int):
            raise ValueError("Choice count bounds must be integers")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
        for v in config["pricing"].values()
    ):
        raise ValueError("Paid choice trials require positive prices")
    protocol = json.loads((ROOT / config["frozen_protocol"]).read_text())
    if config != protocol["config"]:
        raise ValueError("Choice settings changed since freezing")
    prompt = (ROOT / config["prompt"]).read_text()
    if digest(prompt.encode()) != protocol["prompt_sha256"]:
        raise ValueError("Choice prompt changed since freezing")
    for artifact in protocol["implementation_hashes"].values():
        if file_sha(ROOT / artifact["path"]) != artifact["sha256"]:
            raise ValueError("Choice implementation changed since freezing")
    if file_sha(ROOT / config["references"]) != protocol["references_sha256"]:
        raise ValueError("Semantic references changed since freezing")
    return config, protocol, prompt


def audit_request(request):
    return {
        "payload_fields": sorted(request.payload),
        "payload_sha256": digest(request.text.encode()),
        "source_image_sha256": digest(request.source_image),
        "scene_image_sha256": digest(request.scene_image),
        "cards_image_sha256": digest(request.cards_image),
        "regions_sha256": digest(json.dumps(request.regions, sort_keys=True).encode()),
        "schema_sha256": digest(
            json.dumps(
                binding_schema(request.payload["instruction"], len(request.regions)), sort_keys=True
            ).encode()
        ),
        "visual_choice_count": len(request.regions),
    }


def run_choice_trial(dataset, config_path, output, allow_paid=False, dry_run=False, provider=None):
    config, protocol, prompt = load_choice_config(config_path)
    records, root = read_records(dataset)
    if file_sha(root / "records.jsonl") != protocol["dataset_sha256"]:
        raise ValueError("Choice dataset changed since freezing")
    by_id = {r["example_id"]: r for r in records}
    records = [by_id[identifier] for identifier in config["example_ids"]]
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new choice trial directory")
    if not dry_run and not allow_paid:
        raise PermissionError("Choice paid calls require explicit --allow-paid")
    if not dry_run and provider is None and not os.environ.get("OPENAI_API_KEY"):
        raise PermissionError("API credentials unavailable; no mock substitution")
    requests = []
    for record in records:
        inputs = record["model_inputs"]
        request = build_choice_request(
            inputs["instruction"], safe_path(root, inputs["image"]).read_bytes(), prompt
        )
        audit = audit_request(request)
        if audit != protocol["requests"][record["example_id"]]:
            raise ValueError("RGB-derived choice inputs changed since freezing")
        requests.append((record, request, audit))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": str(root / "records.jsonl"),
        "dataset_sha256": protocol["dataset_sha256"],
        "config": config,
        "model": config["model"],
        "provider": "openai",
        "offline_mock": provider is not None,
        "dry_run": dry_run,
        "frozen_protocol_sha256": file_sha(ROOT / config["frozen_protocol"]),
        "prompt_text": prompt,
        "model_input_inventory": False,
        "reference_answers_sent": False,
        "choices_source": "RGB-only class-agnostic MobileSAM masks; no task-dependent filtering or category labels",
        "planned_predictions": len(requests),
        "requests": [{"example_id": r["example_id"], **a} for r, _, a in requests],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Save exact public SDK argument dictionaries for review; authentication is
    # managed by the SDK and is absent from these objects.
    for index, (_, request, _) in enumerate(requests):
        folder = output / f"input-{index:02d}"
        folder.mkdir()
        (folder / "request.json").write_text(json.dumps(request_arguments(config, request)) + "\n")
        (folder / "regions.json").write_text(json.dumps(request.regions, indent=2) + "\n")
        (folder / "original.png").write_bytes(request.source_image)
        (folder / "scene.png").write_bytes(request.scene_image)
        (folder / "cards.png").write_bytes(request.cards_image)
    if dry_run:
        return manifest
    provider = provider or ChoiceOpenAIProvider(config)
    ledger, rows = Ledger(config), []
    with (output / "predictions.jsonl").open("w") as stream:
        for index, (record, request, audit) in enumerate(requests):
            input_bound = (
                len(
                    (
                        prompt
                        + request.text
                        + json.dumps(binding_schema(request.payload["instruction"], len(request.regions)))
                    ).encode()
                )
                + config["image_token_allowance"]
                + 512
            )
            reservation = (
                input_bound * config["pricing"]["input_per_million"]
                + config["max_output_tokens"] * config["pricing"]["output_per_million"]
            ) / 1e6
            row = {
                "example_id": record["example_id"],
                "condition": "rgb_role_choices",
                "status": "api_failure",
                "choices": None,
                "prediction": None,
                "attempts": 0,
                "cost_usd": 0.0,
                **audit,
            }
            start = time.perf_counter()
            try:
                ledger.admit(reservation)
                row["attempts"] = 1
                response = provider.infer(request)
                row["cost_usd"] = ledger.account(response, reservation)
                row.update(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    response_model=response.raw.get("model"),
                    response_status=response.raw.get("status"),
                    response_service_tier=response.raw.get("service_tier"),
                    incomplete_details=response.raw.get("incomplete_details"),
                )
                raw_path = output / f"raw-{index:02d}.json"
                raw_path.write_text(json.dumps(response.raw, indent=2) + "\n")
                row["raw_response"] = raw_path.name
                try:
                    choices = validate_bindings(
                        json.loads(response.text), request.payload["instruction"], len(request.regions)
                    )
                    row["choices"] = choices
                    try:
                        prediction = compile_bindings(
                            choices, request.payload["instruction"], request.regions
                        )
                        row.update(status="ok", prediction=prediction.model_dump(mode="json"))
                    except ValueError as error:
                        row.update(status="abstained", compilation_error=str(error))
                except (ValidationError, ValueError) as error:
                    row["status"] = "validation_failure"
                    if isinstance(error, ValidationError):
                        row["validation_errors"] = error.errors(
                            include_input=False, include_url=False, include_context=False
                        )
            except BudgetExceeded:
                row["status"] = "budget_stopped"
            except Exception as error:
                ledger.unknown_cost_reservation += reservation
                row["error"] = {"type": type(error).__name__, "status": getattr(error, "status_code", None)}
            row["latency_seconds"] = time.perf_counter() - start
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            manifest.update(
                ledger=ledger.as_dict(),
                completed_predictions=len(rows),
                successful_predictions=sum(r["status"] == "ok" for r in rows),
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(record["example_id"], row["status"], flush=True)
            if row["status"] in {"api_failure", "budget_stopped"}:
                break
    # Offline answer data is first decoded after all inference has ended.
    references = load_references(dataset, ROOT / config["references"])
    metrics = []
    from .rgb_grounding import RGBPrediction

    for row in rows:
        parsed = RGBPrediction.model_validate(row["prediction"]) if row["prediction"] else None
        metrics.append(
            {
                "example_id": row["example_id"],
                "status": row["status"],
                **evaluate_rgb(by_id[row["example_id"]], parsed, references[row["example_id"]], root),
            }
        )
    (output / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    summary = {
        "planned_tasks": len(records),
        "attempted_calls": ledger.attempts,
        "valid_forms": sum(r["choices"] is not None for r in rows),
        "compiled": sum(r["status"] == "ok" for r in rows),
        "abstained": sum(r["status"] == "abstained" for r in rows),
        "semantic_complete_goals": sum(m["semantic_goal"]["exact_match"] for m in metrics),
        "relation_types_match": sum(m["predicate_types"]["exact_match"] for m in metrics),
        "ledger": ledger.as_dict(),
        "inventory_provided": False,
        "robot_execution_tested": False,
        "metric": "Same frozen RGB semantic final-goal evaluator as the earlier free-form trial",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return manifest
