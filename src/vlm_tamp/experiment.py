"""Serial experiments with auditable retries, caching and paid-call admission."""

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

from pydantic import ValidationError

from .dataset import CONDITIONS, build_request, read_records
from .grounding import CURRENT_SCHEMA_VERSION, GroundingPrediction, strict_output_schema
from .libero import ROOT
from .markers import load_marker_bundle
from .providers import make_provider, retryable


class BudgetExceeded(RuntimeError):
    pass


def load_config(path):
    config = json.loads(Path(path).read_text())
    protocol = None
    if config.get("frozen_protocol"):
        protocol_path = Path(config["frozen_protocol"])
        if not protocol_path.is_absolute():
            protocol_path = ROOT / protocol_path
        protocol = json.loads(protocol_path.read_text())
    config["schema_version"] = config.get(
        "schema_version", (protocol or {}).get("schema_version", CURRENT_SCHEMA_VERSION)
    )
    strict_output_schema(config["schema_version"])
    if config["provider"] not in {"mock", "openai", "gemini"}:
        raise ValueError("Unknown provider")
    for key in (
        "max_budget_usd",
        "max_requests",
        "max_retries",
        "max_output_tokens",
        "timeout_seconds",
        "image_token_allowance",
    ):
        value = config[key]
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid configuration {key}")
    for key in ("max_requests", "max_retries", "max_output_tokens", "image_token_allowance"):
        if not isinstance(config[key], int):
            raise ValueError(f"{key} must be an integer")
    if config["max_output_tokens"] < 16 or config["timeout_seconds"] <= 0:
        raise ValueError("Invalid output-token or timeout bound")
    for price in config["pricing"].values():
        if not isinstance(price, (float, int)) or not math.isfinite(price) or price < 0:
            raise ValueError("Invalid configured pricing")
    if config["provider"] != "mock" and any(
        config["pricing"][key] <= 0 for key in ("input_per_million", "output_per_million")
    ):
        raise ValueError("Paid providers need explicit positive pricing")
    if not all(c in CONDITIONS for c in config["conditions"]) or len(set(config["conditions"])) != len(
        config["conditions"]
    ):
        raise ValueError("Unknown or duplicate condition")
    if config.get("temperature") is not None and not (
        isinstance(config["temperature"], (float, int))
        and math.isfinite(config["temperature"])
        and 0 <= config["temperature"] <= 2
    ):
        raise ValueError("Invalid configured temperature")
    if config.get("image_detail", "high") not in {"high", "low", "auto", "original"}:
        raise ValueError("Invalid configured image_detail")
    if config.get("reasoning_effort") not in {None, "none", "minimal", "low", "medium", "high", "xhigh"}:
        raise ValueError("Invalid configured reasoning_effort")
    if config.get("reasoning_effort") not in {None, "none"} and config.get("temperature") is not None:
        raise ValueError("Do not combine temperature with enabled reasoning")
    if config.get("service_tier") not in {None, "default"}:
        raise ValueError("Only standard service pricing is supported by this runner")
    if not isinstance(config.get("capture_usage_details", False), bool):
        raise ValueError("capture_usage_details must be boolean")
    if not isinstance(config.get("stop_on_api_failure", False), bool):
        raise ValueError("stop_on_api_failure must be boolean")
    prompt_path = Path(config["prompt"])
    if not prompt_path.is_absolute():
        prompt_path = ROOT / prompt_path
    config["prompt_text"] = prompt_path.read_text()
    config["prompt_sha256"] = hashlib.sha256(config["prompt_text"].encode()).hexdigest()
    if "markers" in config["conditions"]:
        if not config.get("marker_bundle"):
            raise ValueError("Markers condition requires marker_bundle configuration")
        marker_path = Path(config["marker_bundle"])
        if not marker_path.is_absolute():
            marker_path = ROOT / marker_path
        if marker_path.is_dir():
            marker_path = marker_path / "markers.json"
        config["marker_bundle"] = str(marker_path.resolve())
        config["marker_bundle_sha256"] = hashlib.sha256(marker_path.read_bytes()).hexdigest()
    if config.get("frozen_protocol"):
        schema_hash = hashlib.sha256(
            json.dumps(strict_output_schema(config["schema_version"]), sort_keys=True).encode()
        ).hexdigest()
        if schema_hash != protocol["schema_sha256"] or config["prompt_sha256"] != protocol["prompt_sha256"]:
            raise ValueError("Schema or prompt differs from the frozen experiment protocol")
        if "markers" in config["conditions"] and config["marker_bundle_sha256"] != protocol.get(
            "marker_bundle_sha256"
        ):
            raise ValueError("Marker inputs differ from the frozen experiment protocol")
        for field in ("example_ids", "conditions", "temperature", "image_detail", "max_output_tokens"):
            if config.get(field) != protocol[field]:
                raise ValueError(f"Configuration {field} differs from the frozen protocol")
        if config["model"] not in protocol["models"]:
            raise ValueError("Model is not part of the frozen comparison")
        for field in (
            "reasoning_effort",
            "service_tier",
            "capture_usage_details",
            "stop_on_api_failure",
            "pricing",
            "max_requests",
            "max_retries",
            "max_budget_usd",
        ):
            if field in protocol and config.get(field) != protocol[field]:
                raise ValueError(f"Configuration {field} differs from the frozen protocol")
        config["frozen_protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    return config


def request_key(config, request):
    content = {
        "provider": config["provider"],
        "model": config["model"],
        "prompt": request.prompt,
        "payload": request.payload,
        "image_sha256": request.image_hash,
        "condition": request.condition,
        "schema": strict_output_schema(config.get("schema_version", CURRENT_SCHEMA_VERSION)),
        "max_output_tokens": config["max_output_tokens"],
        "seed": config["seed"],
        "adapter_version": "1.1" if config.get("schema_version") == "1.1" else "1.2",
        "temperature": config.get("temperature"),
        "reasoning_effort": config.get("reasoning_effort"),
        "image_detail": config.get("image_detail", "high"),
    }
    if "service_tier" in config:
        content["service_tier"] = config["service_tier"]
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def input_price_ceiling(config):
    """Keep old pricing unchanged; cover the non-additive cache-write rate when supplied."""
    return max(
        config["pricing"]["input_per_million"],
        config["pricing"].get("cache_write_per_million", 0),
    )


def request_reservation(config, request):
    # UTF-8 byte count deliberately overestimates typical text token counts.
    # Vision allowance is configurable and is an estimate, not a provider billing guarantee.
    text_bytes = len(
        (
            request.prompt
            + request.text
            + json.dumps(strict_output_schema(config.get("schema_version", CURRENT_SCHEMA_VERSION)))
        ).encode()
    )
    inputs = text_bytes + (config["image_token_allowance"] if request.image else 0) + 512
    cost = (
        inputs * input_price_ceiling(config)
        + config["max_output_tokens"] * config["pricing"]["output_per_million"]
    ) / 1e6
    return inputs, cost


class Ledger:
    def __init__(self, config):
        self.config = config
        self.attempts = 0
        self.actual_cost = 0.0
        self.unknown_cost_reservation = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cached_input_tokens = 0
        self.cache_write_tokens = 0

    def admit(self, reservation):
        if self.attempts >= self.config["max_requests"]:
            raise BudgetExceeded("Maximum actual API request attempts reached")
        if (
            self.actual_cost + self.unknown_cost_reservation + reservation
            > self.config["max_budget_usd"] + 1e-12
        ):
            raise BudgetExceeded("Next attempt exceeds the configured estimated budget")
        self.attempts += 1

    def account(self, response, reservation):
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        if self.config.get("capture_usage_details"):
            usage = response.raw.get("usage") or {}
            inputs = usage.get("input_tokens_details") or {}
            outputs = usage.get("output_tokens_details") or {}
            self.reasoning_tokens += outputs.get("reasoning_tokens") or 0
            self.cached_input_tokens += inputs.get("cached_tokens") or 0
            self.cache_write_tokens += inputs.get("cache_write_tokens") or 0
        cost = (
            response.input_tokens * input_price_ceiling(self.config)
            + response.output_tokens * self.config["pricing"]["output_per_million"]
        ) / 1e6
        self.actual_cost += cost
        if response.input_tokens + response.output_tokens == 0:
            self.unknown_cost_reservation += reservation
        return cost

    def as_dict(self):
        result = {
            "api_attempts": self.attempts,
            "estimated_usage_cost_usd": self.actual_cost,
            "unknown_usage_reserved_usd": self.unknown_cost_reservation,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "budget_usd": self.config["max_budget_usd"],
            "budget_semantics": "conservative estimate; provider invoice is authoritative",
        }
        if self.config.get("capture_usage_details"):
            result.update(
                reasoning_tokens=self.reasoning_tokens,
                cached_input_tokens=self.cached_input_tokens,
                cache_write_tokens=self.cache_write_tokens,
                accounting_input_rate_per_million=input_price_ceiling(self.config),
                accounting_scope="All input billed at the highest configured uncached/cache-write rate; output includes reasoning without double counting.",
            )
        return result


def run_experiment(
    dataset, config_path, output, allow_paid=False, dry_run=False, limit=None, cache=None, provider=None
):
    records, root = read_records(dataset)
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        records = records[:limit]
    config = load_config(config_path)
    if config.get("example_ids"):
        wanted = set(config["example_ids"])
        records = [r for r in records if r["example_id"] in wanted]
        if {r["example_id"] for r in records} != wanted:
            raise ValueError("Configured example IDs are missing from the selected dataset")
    requested_provider = config["provider"]
    paid = config["provider"] != "mock"
    key_name = "OPENAI_API_KEY" if config["provider"] == "openai" else "GEMINI_API_KEY"
    missing_key = (
        paid
        and not os.environ.get(key_name)
        and not (config["provider"] == "gemini" and os.environ.get("GOOGLE_API_KEY"))
    )
    if paid and not dry_run and missing_key and provider is None:
        print(f"{key_name} is not configured: running the explicitly labelled mock backend.")
        config["provider"], config["model"] = "mock", "deterministic-language-baseline-v1"
        config["pricing"] = {"input_per_million": 0, "output_per_million": 0}
        paid = False
    if paid and not dry_run and not allow_paid:
        raise PermissionError("Paid calls require --allow-paid; use --dry-run to inspect cost")
    output = Path(output).resolve()
    if (output / "predictions.jsonl").exists() or (output / "manifest.json").exists():
        raise FileExistsError("Experiment directory already contains a run; select a new --output")
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache or ROOT / "results/cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    marker_samples = (
        load_marker_bundle(config["marker_bundle"], dataset, records)
        if "markers" in config["conditions"]
        else {}
    )
    requests = [
        (
            record,
            condition,
            build_request(
                record["model_inputs"],
                root,
                condition,
                config["prompt_text"],
                marker_samples.get(record["example_id"]) if condition == "markers" else None,
            ),
        )
        for record in records
        for condition in config["conditions"]
    ]
    fresh = [(r, c, q) for r, c, q in requests if not (cache / f"{request_key(config, q)}.json").exists()]
    estimate = {
        "planned_examples": len(records),
        "planned_predictions": len(requests),
        "uncached_predictions": len(fresh),
        "cached_predictions": len(requests) - len(fresh),
        "estimated_cost_without_retries_usd": sum(request_reservation(config, q)[1] for _, _, q in fresh),
        "estimated_cost_with_all_retries_usd": sum(request_reservation(config, q)[1] for _, _, q in fresh)
        * (config["max_retries"] + 1),
        "max_requests_including_retries": config["max_requests"],
        "max_budget_usd": config["max_budget_usd"],
        "note": "Text byte bound + configured image token allowance; estimated admission, not a hard invoice cap.",
    }
    manifest = {
        "requested_provider": requested_provider,
        "provider": config["provider"],
        "model": config["model"],
        "offline_mock": not paid,
        "dry_run": dry_run,
        "dataset": str(root / "records.jsonl"),
        "dataset_sha256": hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest(),
        "config": {k: v for k, v in config.items() if k != "prompt_text"},
        "prompt_text": config["prompt_text"],
        "output_schema": strict_output_schema(config["schema_version"]),
        "estimate": estimate,
        "seed_semantics": "collection/mock determinism; cloud generation need not be deterministic",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(estimate, indent=2), flush=True)
    if dry_run:
        return manifest
    provider = provider or make_provider(config)
    ledger = Ledger(config)
    rng = random.Random(config["seed"])
    rows = []
    try:
        with (output / "predictions.jsonl").open("w") as stream:
            for record, condition, request in requests:
                key = request_key(config, request)
                cache_path = cache / f"{key}.json"
                cached = cache_path.exists()
                row = {
                    "example_id": record["example_id"],
                    "condition": condition,
                    "provider": config["provider"],
                    "model": config["model"],
                    "offline_mock": not paid,
                    "prompt_sha256": config["prompt_sha256"],
                    "request_hash": key,
                    "image_sha256": request.image_hash,
                    "source_image_sha256": request.source_image_sha256,
                    "input_provenance": request.input_provenance,
                    "payload_sha256": hashlib.sha256(request.text.encode()).hexdigest(),
                    "candidate_inventory_sha256": hashlib.sha256(
                        json.dumps(request.payload.get("candidates"), sort_keys=True).encode()
                    ).hexdigest(),
                    "cached": cached,
                    "status": "api_failure",
                    "prediction": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "attempts": 0,
                }
                start = time.perf_counter()
                if cached:
                    saved = json.loads(cache_path.read_text())
                    # Revalidate caches after any schema changes or manual corruption.
                    try:
                        prediction = GroundingPrediction.model_validate(
                            saved["prediction"], context=request.validation_context
                        )
                        if prediction.schema_version != config["schema_version"]:
                            raise ValueError("Cache does not use the configured inference schema")
                        row.update(
                            status="ok",
                            prediction=prediction.inference_dict(),
                            original_input_tokens=saved["input_tokens"],
                            original_output_tokens=saved["output_tokens"],
                            original_cost_usd=saved["cost_usd"],
                        )
                    except (ValueError, KeyError):
                        row.update(status="cache_validation_failure", error="Invalid saved cache record")
                else:
                    reservation = request_reservation(config, request)[1]
                    for attempt in range(config["max_retries"] + 1):
                        try:
                            if paid:
                                ledger.admit(reservation)
                            row["attempts"] += 1
                            response = provider.infer(request)
                            row["response_model"] = response.raw.get(
                                "model", response.raw.get("model_version")
                            )
                            if paid:
                                row["cost_usd"] += ledger.account(response, reservation)
                            row["input_tokens"] += response.input_tokens
                            row["output_tokens"] += response.output_tokens
                            if config.get("capture_usage_details"):
                                usage = response.raw.get("usage") or {}
                                inputs = usage.get("input_tokens_details") or {}
                                outputs = usage.get("output_tokens_details") or {}
                                row.update(
                                    reasoning_tokens=outputs.get("reasoning_tokens"),
                                    cached_input_tokens=inputs.get("cached_tokens"),
                                    cache_write_tokens=inputs.get("cache_write_tokens"),
                                    response_status=response.raw.get("status"),
                                    response_service_tier=response.raw.get("service_tier"),
                                    incomplete_details=response.raw.get("incomplete_details"),
                                )
                            raw_path = output / "raw" / f"{key}-attempt{attempt}.json"
                            raw_path.parent.mkdir(exist_ok=True)
                            raw_path.write_text(json.dumps(response.raw, indent=2))
                            row["raw_response"] = str(raw_path.relative_to(output))
                            try:
                                prediction = GroundingPrediction.model_validate_json(
                                    response.text, context=request.validation_context
                                )
                                if prediction.schema_version != config["schema_version"]:
                                    raise ValueError("New inference must use the configured schema version")
                            except (ValidationError, ValueError) as error:
                                row.update(
                                    status="validation_failure",
                                    error="Response failed GroundingPrediction validation",
                                )
                                if isinstance(error, ValidationError):
                                    row["validation_errors"] = error.errors(
                                        include_input=False, include_url=False, include_context=False
                                    )
                                try:
                                    row["unvalidated_prediction"] = json.loads(response.text)
                                except ValueError:
                                    pass
                                break
                            row.update(status="ok", prediction=prediction.inference_dict())
                            cache_path.write_text(json.dumps(row, indent=2))
                            break
                        except BudgetExceeded as error:
                            row.update(status="budget_stopped", error=str(error))
                            break
                        except Exception as error:
                            if paid:
                                ledger.unknown_cost_reservation += reservation
                            # Log type/status, not messages potentially containing credentials or request data.
                            row["error"] = {
                                "type": type(error).__name__,
                                "status": getattr(error, "status_code", None),
                            }
                            if retryable(error) and attempt < config["max_retries"]:
                                delay = min(30, 2**attempt + rng.random())
                                retry_after = getattr(getattr(error, "response", None), "headers", {}).get(
                                    "retry-after"
                                )
                                if retry_after:
                                    try:
                                        delay = min(30, max(delay, float(retry_after)))
                                    except ValueError:
                                        pass
                                time.sleep(delay)
                                continue
                            break
                row["latency_seconds"] = time.perf_counter() - start
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if row["status"] == "budget_stopped" or (
                    config.get("stop_on_api_failure") and row["status"] == "api_failure"
                ):
                    break
    finally:
        manifest["ledger"] = ledger.as_dict()
        manifest["completed_predictions"] = len(rows)
        manifest["successful_predictions"] = sum(r["status"] == "ok" for r in rows)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
