"""Durable single-attempt Luna requests under the shared two-dollar ceiling."""

import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path


PRIOR_SPEND_USD = 0.154725
BUDGET_USD = 2.0
MODEL = "gpt-6-luna"
INPUT_CEILING_PER_MILLION = 0.125  # Includes the possible cache-write price.
OUTPUT_PER_MILLION = 0.5
OUTPUT_LIMITS = (6000, 12000, 32000)


class ApiError(RuntimeError):
    pass


class BudgetError(ApiError):
    pass


class UncertainRequest(ApiError):
    """The remote request may have been billed; it must not be automatically repeated."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending-write")
    with temporary.open("w") as handle:
        handle.write(_canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_request(arguments, scene_id):
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ApiError("Every request needs a stable scene ID")
    if (arguments.get("model") != MODEL or arguments.get("reasoning") != {"effort": "high"}
            or type(arguments.get("max_output_tokens")) is not int
            or arguments.get("max_output_tokens") not in OUTPUT_LIMITS
            or arguments.get("store") is not False
            or arguments.get("service_tier") != "default"):
        raise ApiError("Only the approved Luna-high request policy is supported")


def request_arguments(numbered_path, prompt, max_output_tokens=6000):
    if type(max_output_tokens) is not int or max_output_tokens not in OUTPUT_LIMITS:
        raise ApiError("Only bounded 6,000-, 12,000- or 32,000-token Luna-high requests are supported")
    image = Path(numbered_path).read_bytes()
    if not image:
        raise ApiError("A nonempty numbered RGB image is required")
    return {"model": MODEL,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "detail": "high",
                 "image_url": "data:image/png;base64," + base64.b64encode(image).decode()},
            ]}],
            "reasoning": {"effort": "high"}, "service_tier": "default",
            "max_output_tokens": max_output_tokens, "store": False}


def reservation_usd(arguments):
    # Preserve the old one-image allowance and reserve separately for every
    # image when the camera views are supplied as independent API inputs.
    content=[item for message in arguments['input'] for item in message['content']]
    text_bytes=sum(len(item['text'].encode('utf-8')) for item in content if item.get('type')=='input_text')
    images=sum(item.get('type')=='input_image' for item in content)
    if images>4:raise ApiError('This runner supports at most four image inputs per request')
    tokens = text_bytes + 512 + max(1,images)*8192
    output_tokens = arguments.get("max_output_tokens")
    if type(output_tokens) is not int or output_tokens not in OUTPUT_LIMITS:
        raise ApiError("Unsupported output-token reservation")
    return (tokens * INPUT_CEILING_PER_MILLION + output_tokens * OUTPUT_PER_MILLION) / 1e6


def _usage_cost(raw, reservation):
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return reservation, False
    inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
    if (type(inputs) is not int or type(outputs) is not int or inputs < 0 or outputs < 0):
        return reservation, False
    return (inputs * INPUT_CEILING_PER_MILLION + outputs * OUTPUT_PER_MILLION) / 1e6, True


class DurableLedger:
    """A serial admission ledger; pending calls retain their entire reservation."""

    def __init__(self, directory, *, prior_spend=PRIOR_SPEND_USD, budget=BUDGET_USD, max_calls=12):
        if (isinstance(prior_spend, bool) or not math.isfinite(prior_spend)
                or abs(prior_spend - PRIOR_SPEND_USD) > 1e-12 or budget != BUDGET_USD
                or type(max_calls) is not int or not 1 <= max_calls <= 12):
            raise BudgetError("Keep the approved prior spending, two-dollar cap and twelve scene maximum")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "ledger.json"
        self.prior_spend, self.budget, self.max_calls = prior_spend, budget, max_calls

    def _read(self):
        if not self.path.exists():
            return {"version": "rekep-libero-luna-1", "model": MODEL, "reasoning_effort": "high",
                    "budget_usd": self.budget, "prior_conservative_spend_usd": self.prior_spend,
                    "max_scene_requests": self.max_calls, "requests": {}}
        saved = json.loads(self.path.read_text())
        if (saved.get("version") != "rekep-libero-luna-1" or saved.get("model") != MODEL
                or saved.get("budget_usd") != self.budget
                or saved.get("prior_conservative_spend_usd") != self.prior_spend
                or saved.get("max_scene_requests") != self.max_calls):
            raise BudgetError("Existing durable ledger does not match the approved cost policy")
        return saved

    def _save(self, saved):
        accounted = sum(entry.get("cost_usd", entry["reservation_usd"])
                        for entry in saved["requests"].values())
        saved.update(api_attempts=len(saved["requests"]),
                     experiment_conservative_spend_usd=accounted,
                     cumulative_conservative_spend_usd=self.prior_spend + accounted,
                     remaining_usd=max(0.0, self.budget - self.prior_spend - accounted))
        _atomic(self.path, saved)

    def snapshot(self):
        with (self.directory / "ledger.lock").open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            saved = self._read()
            self._save(saved)
            return saved

    def perform(self, arguments, scene_id, *, allow_paid=False, client=None):
        validate_request(arguments, scene_id)
        digest = hashlib.sha256(_canonical(arguments).encode()).hexdigest()
        response_path = self.directory / f"response-{digest}.json"
        request_path = self.directory / f"request-{digest}.json"
        with (self.directory / "ledger.lock").open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            saved = self._read()
            entries = saved["requests"]
            prior = entries.get(scene_id) or next(
                (entry for entry in entries.values() if scene_id in entry.get("scene_aliases", [])), None)
            if prior is None:
                prior = next((entry for entry in entries.values() if entry["request_sha256"] == digest), None)
                if prior is not None:
                    prior.setdefault("scene_aliases", []).append(scene_id)
            if prior is not None:
                if prior["request_sha256"] != digest:
                    raise ApiError("This scene already has one attempted request with different inputs")
                if response_path.exists():
                    raw = json.loads(response_path.read_text())
                    cost, usage_known = _usage_cost(raw, prior["reservation_usd"])
                    prior.update(status="completed", cost_usd=cost, usage_known=usage_known,
                                 response_status=raw.get("status"))
                    self._save(saved)
                    return raw
                prior["status"] = "uncertain"
                self._save(saved)
                raise UncertainRequest("An earlier request has no durable response; its budget remains reserved")
            if not allow_paid:
                raise ApiError("Paid inference was not enabled by the runner")
            reservation = reservation_usd(arguments)
            spent = self.prior_spend + sum(entry.get("cost_usd", entry["reservation_usd"])
                                            for entry in entries.values())
            if len(entries) >= self.max_calls or spent + reservation > self.budget:
                raise BudgetError("This ledger's API budget cannot admit another bounded request")
            entries[scene_id] = {"request_sha256": digest, "status": "pending",
                                 "reservation_usd": reservation, "response": response_path.name}
            _atomic(request_path, arguments)
            self._save(saved)  # Admission is durable before the remote call begins.
            try:
                if client is None:
                    from openai import OpenAI
                    client = OpenAI(base_url="https://api.openai.com/v1", timeout=240, max_retries=0)
                response = client.responses.create(**arguments)
                raw = response.model_dump(mode="json")
                _atomic(response_path, raw)  # Enables crash recovery without another API call.
            except Exception as error:
                entries[scene_id].update(status="uncertain", error_type=type(error).__name__)
                self._save(saved)
                raise UncertainRequest("API attempt failed; no automatic retry and full cost reserved") from error
            cost, usage_known = _usage_cost(raw, reservation)
            entries[scene_id].update(status="completed", cost_usd=cost, usage_known=usage_known,
                                     response_status=raw.get("status"))
            self._save(saved)
            return raw


def request(numbered_path, prompt, outdir, config):
    """Root-runner entry point. Credentials are loaded locally, never serialized."""
    from ..cli import load_credentials
    directory = config.get("ledger_dir", Path(__file__).resolve().parents[3] / "results/rekep-libero-api")
    ledger = DurableLedger(directory, prior_spend=config.get("prior_spend_usd", PRIOR_SPEND_USD),
                           budget=config.get("budget_usd", BUDGET_USD), max_calls=config.get("max_requests", 12))
    if config.get("allow_paid", False):
        load_credentials(config.get("env_file", "/home/duo/.codex/keys/vlm-tamp.env"))
    arguments = request_arguments(numbered_path, prompt, config.get("max_output_tokens", 6000))
    raw = ledger.perform(arguments, config.get("scene_id", str(Path(outdir).resolve())),
                         allow_paid=config.get("allow_paid", False))
    _atomic(Path(outdir) / "response.json", raw)
    return raw
