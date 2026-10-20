"""A $2 budget per experiment, shared across its workers and resumptions.

Historical ledgers remain read-only. Completed exact requests can be reused
without transferring their charge to a new run. Unknown outcomes stay reserved
in the original run and are never automatically retried in another run.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .rekep_libero.api import (
    ApiError, BudgetError, DurableLedger, UncertainRequest, MODEL, BUDGET_USD,
    _atomic, _canonical, validate_request,
)

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "results/api-runs"
LEGACY_LEDGER = ROOT / "results/articulated-api-v1/ledger.json"
_RUN_ID = ContextVar("vlm_tamp_api_run_id", default=None)
VERSION = "luna-per-experiment-1"


def validate_run_id(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", run_id):
        raise BudgetError("Use a stable experiment ID containing letters, digits, dots, dashes or underscores")
    return run_id


def current_run_id():
    value = _RUN_ID.get() or os.environ.get("VLM_TAMP_API_RUN_ID")
    return validate_run_id(value) if value is not None else None


@contextmanager
def api_run_scope(run_id):
    token = _RUN_ID.set(validate_run_id(run_id))
    try:
        yield
    finally:
        _RUN_ID.reset(token)


def _read(path):
    return json.loads(Path(path).read_text())


def _spend(entries):
    values = [entry.get("cost_usd", entry["reservation_usd"]) for entry in entries.values()]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise BudgetError("Invalid API charge or reservation")
    return sum(values)


class RunLedger(DurableLedger):
    """Reuse durable admission, charging only this experiment's new requests."""

    def __init__(self, run_id, *, root=None, legacy_path=None):
        self.run_id = validate_run_id(run_id)
        self.root = Path(root if root is not None else RUN_ROOT)
        self.legacy_path = Path(legacy_path if legacy_path is not None else LEGACY_LEDGER)
        self.directory = self.root / self.run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "ledger.json"
        self.prior_spend, self.budget, self.max_calls = 0.0, BUDGET_USD, 10000

    def _read(self):
        expected = {"version": VERSION, "run_id": self.run_id, "model": MODEL,
                    "reasoning_effort": "high", "budget_scope": "experiment_run",
                    "budget_usd": self.budget, "prior_conservative_spend_usd": 0.0,
                    "max_scene_requests": self.max_calls}
        if not self.path.exists():
            return {**expected, "requests": {}, "cache_hits": {}}
        saved = _read(self.path)
        if any(saved.get(k) != v for k, v in expected.items()):
            raise BudgetError("The saved experiment budget or identity changed")
        _spend(saved["requests"])
        return saved

    def _save(self, saved):
        spend = _spend(saved["requests"])
        saved.update(api_attempts=len(saved["requests"]), run_conservative_spend_usd=spend,
                     experiment_conservative_spend_usd=spend,
                     remaining_usd=max(0.0, self.budget - spend),
                     remaining_run_budget_usd=max(0.0, self.budget - spend))
        _atomic(self.path, saved)

    def perform(self, arguments, scene_id, *, allow_paid=False, client=None):
        validate_request(arguments, scene_id)
        digest = hashlib.sha256(_canonical(arguments).encode()).hexdigest()
        # Lock ordering is cache then run. Admission stays serialized even when
        # separate processes use the same experiment ID or the same request.
        with (self.root / "request-cache.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            with (self.directory / "ledger.lock").open("a") as run_lock:
                fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX)
                saved = self._read()
                cached = saved["cache_hits"].get(scene_id)
                if cached is not None and cached["request_sha256"] != digest:
                    raise ApiError("This scene already has one cached request with different inputs")
                own = any(key == scene_id or scene_id in entry.get("scene_aliases", [])
                          or entry["request_sha256"] == digest
                          for key, entry in saved["requests"].items())
                if not own:
                    candidates = ([self.legacy_path] if self.legacy_path.exists() else [])
                    candidates += sorted(p for p in self.root.glob("*/ledger.json") if p != self.path)
                    for path in candidates:
                        source = _read(path)
                        for entry in source["requests"].values():
                            if entry["request_sha256"] != digest:
                                continue
                            response = path.parent / f"response-{digest}.json"
                            if not response.exists():
                                raise UncertainRequest("This exact request has an earlier uncertain attempt; no automatic retry")
                            raw = _read(response)
                            saved["cache_hits"][scene_id] = {"request_sha256": digest,
                                "source_ledger": str(path), "response": str(response), "charged_usd": 0.0}
                            self._save(saved)
                            return raw
                    if cached is not None:
                        raise ApiError("The recorded cached response is unavailable; no automatic paid replacement")
            # The original implementation reserves cost durably before HTTP,
            # recovers stored responses and retains uncertain-call reservations.
            return super().perform(arguments, scene_id, allow_paid=allow_paid, client=client)


def ledger_for_output(output, run_id=None):
    """Bind an experiment root, never an episode, to a durable shared budget."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = run_id or current_run_id()
    if selected is not None:
        validate_run_id(selected)
    with (output / "api-run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = output / "api-run.json"
        if path.exists():
            saved = _read(path)
            if saved.get("budget_scope") != "experiment_run" or saved.get("budget_usd") != BUDGET_USD:
                raise BudgetError("Existing experiment budget policy changed")
            if selected is not None and selected != saved["run_id"]:
                raise BudgetError("A resumed experiment must keep its existing API run ID and balance")
            selected = saved["run_id"]
        else:
            if selected is None:
                slug = re.sub(r"[^A-Za-z0-9_.-]", "-", output.name).strip(".-_") or "experiment"
                selected = slug[:95] + "-" + hashlib.sha256(str(output).encode()).hexdigest()[:12]
            _atomic(path, {"run_id": selected, "budget_scope": "experiment_run", "budget_usd": BUDGET_USD,
                           "shared_by": ["episodes", "variants", "parallel_workers", "resumptions"]})
    return RunLedger(selected)


def snapshot(ledger=None, *, legacy_path=None, root=None):
    """Report project history separately from the only enforced cap: this run."""
    root = Path(root if root is not None else ledger.root if ledger is not None else RUN_ROOT)
    history_path = Path(legacy_path if legacy_path is not None else
                        ledger.legacy_path if ledger is not None else LEGACY_LEDGER)
    if ledger is None and current_run_id() is not None:
        ledger = RunLedger(current_run_id(), root=root, legacy_path=history_path)
    historical = _read(history_path).get("cumulative_conservative_spend_usd", 0.0) if history_path.exists() else 0.0
    if not isinstance(historical, (int, float)) or isinstance(historical, bool) or not math.isfinite(historical) or historical < 0:
        raise BudgetError("Historical API total must be a finite nonnegative number")
    current = ledger.snapshot() if ledger is not None else {
        "run_id": None, "budget_scope": "experiment_run", "budget_usd": BUDGET_USD,
        "run_conservative_spend_usd": None, "remaining_run_budget_usd": None,
        "remaining_usd": None, "requests": {}, "api_attempts": 0, "max_scene_requests": 10000}
    total = historical + sum(_spend(_read(p)["requests"]) for p in root.glob("*/ledger.json"))
    return {**current, "historical_conservative_spend_usd": historical,
            "cumulative_conservative_spend_usd": total, "project_total_is_informational": True}
