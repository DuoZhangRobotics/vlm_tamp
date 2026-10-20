"""Audited RGB-only trials, with a compact semantic contract and separate offline scoring."""

import base64
import hashlib
import json
import math
import os
import time
from pathlib import Path

from pydantic import ValidationError

from .dataset import build_request, read_records
from .experiment import BudgetExceeded, Ledger
from .libero import ROOT
from .providers import ProviderResponse
from .rgb_evaluation import evaluate_rgb, load_references
from .rgb_grounding import RGBPrediction, rgb_output_schema


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rgb_config(path):
    config = json.loads(Path(path).read_text())
    for key in (
        "max_budget_usd",
        "max_requests",
        "max_output_tokens",
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
            raise ValueError("Invalid finite positive RGB bound: " + key)
    for key in ("max_requests", "max_output_tokens", "image_token_allowance"):
        if not isinstance(config[key], int):
            raise ValueError("RGB count bounds must be integers")
    if config["conditions"] != ["rgb"] or config["max_retries"] != 0 or config["service_tier"] != "default":
        raise ValueError("RGB trials require RGB only, standard service and no retries")
    if config["temperature"] is not None or config["reasoning_effort"] != "medium":
        raise ValueError("This frozen RGB trial uses medium reasoning with temperature omitted")
    if config["max_requests"] < 1 or config["max_budget_usd"] <= 0 or config["max_output_tokens"] < 16:
        raise ValueError("Invalid RGB budget bounds")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
        for value in config["pricing"].values()
    ):
        raise ValueError("Paid RGB trials require positive pricing")
    protocol = json.loads((ROOT / config["frozen_protocol"]).read_text())
    config["prompt_text"] = (ROOT / config["prompt"]).read_text()
    config["prompt_sha256"] = hashlib.sha256(config["prompt_text"].encode()).hexdigest()
    if (
        config["prompt_sha256"] != protocol["prompt_sha256"]
        or hashlib.sha256(json.dumps(rgb_output_schema(), sort_keys=True).encode()).hexdigest()
        != protocol["schema_sha256"]
    ):
        raise ValueError("RGB prompt/schema differs from its frozen protocol")
    for field in (
        "model",
        "conditions",
        "example_ids",
        "reasoning_effort",
        "temperature",
        "service_tier",
        "image_detail",
        "max_output_tokens",
        "max_requests",
        "max_budget_usd",
        "max_retries",
        "pricing",
    ):
        if config[field] != protocol[field]:
            raise ValueError("RGB settings differ from their frozen protocol: " + field)
    for name, relative in protocol["implementation_hashes"].items():
        if sha(ROOT / relative["path"]) != relative["sha256"]:
            raise ValueError("RGB implementation changed after freezing: " + name)
    if sha(ROOT / config["references"]) != protocol["references_sha256"]:
        raise ValueError("Frozen semantic references changed")
    return config, protocol


class RGBOpenAIProvider:
    def __init__(self, config, client=None):
        from openai import OpenAI

        self.config = config
        self.client = client or OpenAI(timeout=config["timeout_seconds"], max_retries=0)

    def infer(self, request):
        if set(request.payload) != {"instruction"} or request.condition != "rgb" or request.image is None:
            raise ValueError("RGB model input must be instruction and original image only")
        response = self.client.responses.create(
            model=self.config["model"],
            instructions=request.prompt,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": request.text},
                        {
                            "type": "input_image",
                            "detail": self.config["image_detail"],
                            "image_url": "data:image/png;base64," + base64.b64encode(request.image).decode(),
                        },
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "RGBPrediction",
                    "schema": rgb_output_schema(),
                    "strict": True,
                }
            },
            reasoning={"effort": "medium"},
            service_tier="default",
            max_output_tokens=self.config["max_output_tokens"],
            store=False,
        )
        return ProviderResponse(
            response.output_text,
            response.model_dump(mode="json"),
            response.usage.input_tokens if response.usage else 0,
            response.usage.output_tokens if response.usage else 0,
        )


def run_rgb_trial(dataset, config_path, output, allow_paid=False, dry_run=False, provider=None):
    config, protocol = load_rgb_config(config_path)
    records, root = read_records(dataset)
    if sha(root / "records.jsonl") != protocol["dataset_sha256"]:
        raise ValueError("RGB dataset differs from its frozen protocol")
    wanted = set(config["example_ids"])
    records = [r for r in records if r["example_id"] in wanted]
    if {r["example_id"] for r in records} != wanted:
        raise ValueError("Missing RGB observations")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new RGB output directory")
    if not dry_run and not allow_paid:
        raise PermissionError("RGB paid calls require explicit --allow-paid")
    if not dry_run and provider is None and not os.environ.get("OPENAI_API_KEY"):
        raise PermissionError("API credentials unavailable; this trial does not substitute mock results")
    output.mkdir(parents=True, exist_ok=True)
    requests = [(r, build_request(r["model_inputs"], root, "rgb", config["prompt_text"])) for r in records]
    audit = [
        {
            "example_id": r["example_id"],
            "payload_fields": sorted(q.payload),
            "payload_sha256": hashlib.sha256(q.text.encode()).hexdigest(),
            "image_sha256": q.image_hash,
            "image_matches_original": q.image_hash == q.source_image_sha256,
        }
        for r, q in requests
    ]
    assert all(a["payload_fields"] == ["instruction"] and a["image_matches_original"] for a in audit)
    manifest = {
        "dataset": str(root / "records.jsonl"),
        "dataset_sha256": protocol["dataset_sha256"],
        "model": config["model"],
        "provider": "openai",
        "offline_mock": provider is not None,
        "dry_run": dry_run,
        "config": {k: v for k, v in config.items() if k != "prompt_text"},
        "prompt_text": config["prompt_text"],
        "output_schema": rgb_output_schema(),
        "requests": audit,
        "planned_predictions": len(requests),
        "model_input_inventory": False,
        "reference_answers_sent": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if dry_run:
        return manifest
    provider = provider or RGBOpenAIProvider(config)
    ledger = Ledger(config)
    rows = []
    with (output / "predictions.jsonl").open("w") as stream:
        for record, request in requests:
            input_bound = (
                len((request.prompt + request.text + json.dumps(rgb_output_schema())).encode())
                + config["image_token_allowance"]
                + 512
            )
            reservation = (
                input_bound * config["pricing"]["input_per_million"]
                + config["max_output_tokens"] * config["pricing"]["output_per_million"]
            ) / 1e6
            row = {
                "example_id": record["example_id"],
                "condition": "rgb",
                "status": "api_failure",
                "prediction": None,
                "cost_usd": 0.0,
                "attempts": 0,
                **next(a for a in audit if a["example_id"] == record["example_id"]),
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
                    reasoning_tokens=(response.raw.get("usage", {}).get("output_tokens_details") or {}).get(
                        "reasoning_tokens"
                    ),
                    incomplete_details=response.raw.get("incomplete_details"),
                )
                raw_path = output / f"raw-{len(rows):02d}.json"
                raw_path.write_text(json.dumps(response.raw, indent=2) + "\n")
                row["raw_response"] = raw_path.name
                try:
                    prediction = RGBPrediction.model_validate_json(response.text)
                    row.update(status="ok", prediction=prediction.model_dump(mode="json"))
                except (ValidationError, ValueError) as error:
                    row["status"] = "validation_failure"
                    if isinstance(error, ValidationError):
                        row["validation_errors"] = error.errors(
                            include_input=False, include_url=False, include_context=False
                        )
                    try:
                        row["unvalidated_prediction"] = json.loads(response.text)
                    except ValueError:
                        pass
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
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(record["example_id"], row["status"], flush=True)
            if row["status"] in {"api_failure", "budget_stopped"}:
                break
    # Reference loading and evaluation happen after inference, never inside the provider.
    references = load_references(dataset, ROOT / config["references"])
    by_id = {r["example_id"]: r for r in records}
    metrics = []
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
        "valid": sum(r["status"] == "ok" for r in rows),
        "semantic_complete_goals": sum(m["semantic_goal"]["exact_match"] for m in metrics),
        "metric": "Natural-language semantic goals with independently measured image identity and explicit owned-part selectors; not raw BDDL-ID exact agreement or robot task execution.",
        "ledger": ledger.as_dict(),
        "inventory_provided": False,
        "robot_execution_tested": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return manifest
