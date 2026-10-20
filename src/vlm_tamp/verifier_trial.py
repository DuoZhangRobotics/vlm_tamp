"""Bounded proposal/check/feedback trial, with physics and benchmark scoring separated."""

import hashlib
import html
import json
import os
from pathlib import Path

from pydantic import ValidationError

from .bddl import normalize_predicate
from .dataset import ModelRequest, build_request, read_records, safe_path
from .experiment import Ledger, load_config, request_key, request_reservation
from .grounding import GroundingPrediction, strict_output_schema
from .libero import ROOT
from .linking import link_prediction
from .providers import make_provider
from .verification import verifier_feedback, verify_proposal


def _benchmark_comparison(record, prediction):
    """Called only after both passes; its output is never used as model feedback."""
    if prediction is None:
        return {"evaluable": True, "exact_match": False, "scope": "benchmark scoring only"}
    entities = {e["id"]: e for e in prediction["entities"]}
    generated = {
        (normalize_predicate(g["name"]), tuple(entities[a].get("reference_id") for a in g["arguments"]))
        for g in prediction["goals"]
    }
    expected = {
        (normalize_predicate(g["name"]), tuple(g["arguments"]))
        for g in record["reference"]["goal_predicates"]
    }
    resolved = prediction["goal_logic"] == "and" and all(
        g.get("intent") != "uncertain" for g in prediction["goals"]
    )
    return {
        "evaluable": record["reference"]["coverage"]["symbolic_goals"],
        "exact_match": generated == expected and resolved,
        "scope": "benchmark scoring only; not physical execution",
        "feedback_used": False,
    }


def run_verifier_trial(dataset, config_path, output, allow_paid=False, dry_run=False, provider=None):
    config = load_config(config_path)
    if config["max_requests"] > 2 or config["max_retries"] != 0:
        raise ValueError("This trial permits at most two attempts and no retries")
    records, root = read_records(dataset)
    selected = [r for r in records if r["example_id"] in config["example_ids"]]
    if len(selected) != 1:
        raise ValueError("The verifier trial requires exactly one selected observation")
    record = selected[0]
    facts_path = Path(config["physical_facts"])
    if not facts_path.is_absolute():
        facts_path = ROOT / facts_path
    facts = json.loads(facts_path.read_text())
    protocol_path = Path(config["frozen_protocol"])
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    protocol = json.loads(protocol_path.read_text())
    if hashlib.sha256(facts_path.read_bytes()).hexdigest() != protocol["physical_facts_sha256"]:
        raise ValueError("Physical evidence differs from the frozen protocol")
    for filename, field in (
        ("verification.py", "verifier_sha256"),
        ("verifier_trial.py", "trial_runner_sha256"),
    ):
        if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != protocol[field]:
            raise ValueError("Verifier implementation differs from the frozen protocol")
    if (
        facts["example_id"] != record["example_id"]
        or facts["initial_state_sha256"] != record["metadata"]["initial_state_sha256"]
    ):
        raise ValueError("Physical evidence does not belong to this observation")
    original = safe_path(root, record["model_inputs"]["image"]).read_bytes()
    if hashlib.sha256(original).hexdigest() != facts["image_sha256"]:
        raise ValueError("Observation image changed after the physical probes")
    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("Trial output already exists; use a new --output")
    output.mkdir(parents=True, exist_ok=True)
    base = build_request(record["model_inputs"], root, "candidates", config["prompt_text"])
    estimate = {
        "maximum_api_attempts": config["max_requests"],
        "maximum_budget_usd": config["max_budget_usd"],
        "first_request_reservation_usd": request_reservation(config, base)[1],
        "correction": "Second input adds original proposal and scoped independent simulator facts; its exact reservation is checked before admission.",
        "benchmark_answers_excluded": True,
    }
    manifest = {
        "model": config["model"],
        "provider": config["provider"],
        "offline_mock": config["provider"] == "mock",
        "dry_run": dry_run,
        "config": {k: v for k, v in config.items() if k != "prompt_text"},
        "prompt_text": config["prompt_text"],
        "output_schema": strict_output_schema(config["schema_version"]),
        "dataset_sha256": hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest(),
        "example_id": record["example_id"],
        "physical_facts": str(facts_path),
        "physical_facts_sha256": protocol["physical_facts_sha256"],
        "estimate": estimate,
        "human_review_performed": False,
        "robot_execution_attempted": False,
    }
    if not dry_run and config["provider"] != "mock":
        if not allow_paid:
            raise PermissionError("Paid trial requires --allow-paid")
        if provider is None and not os.environ.get("OPENAI_API_KEY"):
            raise PermissionError("Authorized API credential is not configured")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(estimate, indent=2), flush=True)
    if dry_run:
        return manifest
    ledger = Ledger(config)
    backend = provider or make_provider(config)
    phases = []
    feedback = None
    try:
        for index in range(config["max_requests"]):
            if index == 0:
                request = base
            else:
                payload = {**base.payload, "verification_feedback": feedback}
                request = ModelRequest(
                    prompt=base.prompt,
                    payload=payload,
                    image=base.image,
                    condition="verifier_feedback_oracle_assisted",
                    source_image_sha256=base.source_image_sha256,
                )
            reservation = request_reservation(config, request)[1]
            phase = {
                "phase": "proposal" if index == 0 else "verifier_feedback_correction",
                "oracle_assistance": "scene inventory"
                if index == 0
                else "scene inventory and independently checked mechanical facts",
                "request_hash": request_key(config, request),
                "request_payload": request.payload,
                "image_sha256": request.image_hash,
                "prediction": None,
                "raw_prediction": None,
                "raw_schema_valid": False,
                "linked_schema_valid": False,
                "link_events": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
            if config["provider"] != "mock":
                ledger.admit(reservation)
            try:
                response = backend.infer(request)
            except Exception as error:
                if config["provider"] != "mock":
                    ledger.unknown_cost_reservation += reservation
                phase["api_error"] = {
                    "type": type(error).__name__,
                    "status": getattr(error, "status_code", None),
                }
                phases.append(phase)
                break
            if config["provider"] != "mock":
                phase["cost_usd"] = ledger.account(response, reservation)
            phase.update(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                returned_model=response.raw.get("model", response.raw.get("model_version")),
            )
            (output / f"raw-{index + 1}.json").write_text(json.dumps(response.raw, indent=2) + "\n")
            try:
                raw = json.loads(response.text)
                if not isinstance(raw, dict):
                    raise ValueError("Model output is not a JSON object")
                phase["raw_prediction"] = raw
            except ValueError:
                phase["validation_errors"] = [
                    {"type": "invalid_json", "msg": "Model did not return a JSON object"}
                ]
                phases.append(phase)
                break
            try:
                GroundingPrediction.model_validate(raw, context=base.validation_context)
                phase["raw_schema_valid"] = True
            except ValidationError as error:
                phase["raw_validation_errors"] = error.errors(
                    include_input=False, include_url=False, include_context=False
                )
            linked, events = link_prediction(raw, base.validation_context)
            phase["link_events"] = events
            errors = None
            try:
                prediction = GroundingPrediction.model_validate(linked, context=base.validation_context)
                if prediction.schema_version != config["schema_version"]:
                    raise ValueError("Model returned a different schema version")
                phase.update(prediction=prediction.inference_dict(), linked_schema_valid=True)
            except (ValidationError, ValueError) as error:
                errors = (
                    error.errors(include_input=False, include_url=False, include_context=False)
                    if isinstance(error, ValidationError)
                    else [{"type": "schema_version", "msg": "Wrong schema version"}]
                )
                phase["validation_errors"] = errors
            report = verify_proposal(phase["prediction"] or linked, facts, errors)
            phase["verification"] = report
            feedback = verifier_feedback(phase["prediction"] or linked, report, facts)
            phases.append(phase)
            (output / "phases.json").write_text(json.dumps(phases, indent=2) + "\n")
            print(f"Completed {phase['phase']}: {report['counts']}", flush=True)
    finally:
        # Benchmark answers enter only this final assessment, after feedback generation/calls.
        for phase in phases:
            phase["benchmark_assessment"] = _benchmark_comparison(record, phase["prediction"])
        manifest.update(ledger=ledger.as_dict(), completed_phases=len(phases))
        (output / "phases.json").write_text(json.dumps(phases, indent=2) + "\n")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if phases:
        final = phases[-1]
        grounded = {
            "schema_version": "1.0",
            "candidate_specification": final["prediction"],
            "verified_mechanical_bindings": final.get("verification", {}).get("verified_bindings", []),
            "claims": final.get("verification", {}).get("claims", []),
            "ready_for_robot_execution": False,
            "benchmark_answers_used_for_generation_or_correction": False,
        }
        (output / "grounded-specification.json").write_text(json.dumps(grounded, indent=2) + "\n")
        write_trial_report(output, manifest, phases, facts, record["model_inputs"]["instruction"])
    return manifest


def write_trial_report(output, manifest, phases, facts, instruction):
    output = Path(output)
    images = output / "images"
    images.mkdir(exist_ok=True)
    facts_root = Path(manifest["physical_facts"]).parent
    for name in ("original.png", "closed-probe.png", "open-probe.png"):
        (images / name).write_bytes((facts_root / name).read_bytes())
    title = (
        "OFFLINE MOCK verification workflow"
        if manifest["offline_mock"]
        else "Drawer insertion: VLM proposal and independent checks"
    )
    rows = []
    for phase in phases:
        for c in phase.get("verification", {}).get("claims", []):
            rows.append(
                f"<tr><td>{html.escape(phase['phase'])}</td><td class='{c['status']}'>{c['status'].upper()}</td><td>{html.escape(c['subject'])}<br><small>{html.escape(c['kind'])}</small></td><td>{html.escape(c['reason'])}<br><small>{html.escape(c['scope'])}</small></td></tr>"
            )
    body = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        "<style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:24px;color:#18263a;background:#f4f6fa}section{background:white;padding:20px;border-radius:12px;margin:20px 0}.images{display:flex;flex-wrap:wrap;gap:20px}img{width:256px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd4df;text-align:left}small{color:#57677a}.verified{color:#13724a}.contradicted{color:#ab253a}.unresolved{color:#8a6100}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>",
        f"<h1>{title}</h1><p>{html.escape(instruction)}</p>",
        "<section><h2>What was tested</h2><p>The VLM proposes a task specification. An independent simulator checks joint mechanics and finite static placement samples. Feedback contains measured facts and scoped checks, with benchmark answers reserved for final assessment.</p><p><strong>No robot grasp, trajectory or task execution was performed.</strong> Verified statuses apply only to their stated scope.</p></section>",
        '<section><h2>Static physics probes</h2><p>Objects are repositioned directly for these tests. These images are not an execution demonstration.</p><div class="images">',
        '<div><img src="images/original.png" alt="Original scene"><p>Original scene</p></div><div><img src="images/closed-probe.png" alt="Static pose with closed drawer"><p>Closed-drawer pose probe</p></div><div><img src="images/open-probe.png" alt="Static pose with open drawer"><p>Open-drawer pose probe</p></div></div>',
        f"<p>Closed: {facts['placement_probes']['closed']['fully_contained_collision_box_count']} / {facts['placement_probes']['closed']['tested']} samples pass full collision-box containment and penetration tolerance. Open: {facts['placement_probes']['open']['fully_contained_collision_box_count']} / {facts['placement_probes']['open']['tested']}.</p>",
        "<p>LIBERO site containment checks a reference point. Full box-vertex containment and collision checks add geometric evidence; neither establishes a feasible insertion path or stable grasp.</p></section>",
        "<section><h2>Verified, contradicted and unresolved claims</h2><table><tr><th>Pass</th><th>Status</th><th>Claim</th><th>Evidence and scope</th></tr>",
        *rows,
        "</table></section>",
    ]
    for phase in phases:
        body.append(
            f"<section><h2>{html.escape(phase['phase'])}</h2><p>Raw schema valid: {phase['raw_schema_valid']}. Linked schema valid: {phase['linked_schema_valid']}. Assistance: {html.escape(phase['oracle_assistance'])}.</p><details><summary>Generated task specification</summary><pre>{html.escape(json.dumps(phase['prediction'] or phase['raw_prediction'], indent=2))}</pre></details><details><summary>Verification and final benchmark assessment</summary><pre>{html.escape(json.dumps({'verification': phase.get('verification'), 'benchmark': phase['benchmark_assessment']}, indent=2))}</pre></details></section>"
        )
    body.append(
        f"<p>Model: {html.escape(manifest['model'])}. Actual attempts: {manifest['ledger']['api_attempts']}. Estimated usage: ${manifest['ledger']['estimated_usage_cost_usd']:.8f}. This trial does not create human-reviewed labels.</p></html>"
    )
    (output / "report.html").write_text("\n".join(body))
    lines = [
        f"# {title}",
        "",
        instruction,
        "",
        "The VLM proposal is checked against independent physics. Verified claims have explicit scope; a complete robot task has not been executed.",
        "",
        "| Pass | Raw valid | Linked valid | Verified | Contradicted | Unresolved | Final benchmark agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in phases:
        counts = phase.get("verification", {}).get("counts", {})
        lines.append(
            f"| {phase['phase']} | {phase['raw_schema_valid']} | {phase['linked_schema_valid']} | {counts.get('verified', 0)} | {counts.get('contradicted', 0)} | {counts.get('unresolved', 0)} | {phase['benchmark_assessment']['exact_match']} |"
        )
    lines += [
        "",
        f"New API attempts: {manifest['ledger']['api_attempts']}; estimated usage: ${manifest['ledger']['estimated_usage_cost_usd']:.8f}.",
        "",
        "Correction uses simulator facts and is explicitly oracle-assisted. Benchmark answers are assessed only after both passes and never enter feedback. Contact strategy, necessity of intermediate steps, grasp stability and executable motion remain unverified.",
        "",
        "See report.html for original/probe images, every claim's status and scope, and the generated specifications.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
