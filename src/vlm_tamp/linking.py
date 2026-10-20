"""Auditable structural linking of declared symbols, without benchmark goal access."""

import copy
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from .dataset import build_request, read_records
from .grounding import GroundingPrediction
from .markers import load_marker_bundle


def link_prediction(payload, context):
    """Return a copy and explicit events; no new identities, predicates or geometry.

    Geometry-owner and local/reference aliases come from the prediction itself.
    Ownership corrections use only the supplied candidate inventory. Conflicting
    ambiguity/identity claims become abstentions, never forced resolutions.
    """
    result = copy.deepcopy(payload)
    if result.get("schema_version") != "1.2":
        return result, []
    events = []
    entities = result.get("entities", [])
    by_id = {e["id"]: e for e in entities}
    candidates = {e["identifier"]: e for e in context.get("candidates") or []}
    for entity in entities:
        if entity.get("ambiguous") and entity.get("reference_id") is not None:
            events.append(
                {
                    "rule": "abstain_conflicting_identity",
                    "entity": entity["id"],
                    "before": entity["reference_id"],
                    "after": None,
                }
            )
            entity["reference_id"] = None
            result.setdefault("abstentions", []).append(
                "Structural linker retained declared ambiguity and removed a conflicting identity claim."
            )
    references = {}
    for entity in entities:
        if entity.get("reference_id"):
            references.setdefault(entity["reference_id"], []).append(entity["id"])
    aliases = {ref: ids[0] for ref, ids in references.items() if len(ids) == 1 and ref not in by_id}
    geometry = {
        g["id"]: g["entity"]
        for g in result.get("geometry_targets", [])
        if g["entity"] in by_id and g["id"] not in by_id
    }

    def linked(identifier):
        return geometry.get(identifier, aliases.get(identifier, identifier))

    for entity in entities:
        before = entity.get("parent")
        after = aliases.get(before, before)
        candidate = candidates.get(entity.get("reference_id"))
        if candidate is not None:
            expected = candidate["parent"]
            if expected is None:
                after = None
            elif after not in by_id or by_id[after].get("reference_id") != expected:
                matches = references.get(expected, [])
                if len(matches) == 1:
                    after = matches[0]
        if after != before:
            entity["parent"] = after
            events.append(
                {
                    "rule": "bind_supplied_parent"
                    if candidate is not None
                    else "bind_parent_reference_alias",
                    "entity": entity["id"],
                    "before": before,
                    "after": after,
                }
            )
    for namespace in ("stages", "constraints"):
        for item in result.get(namespace, []):
            before = item.get("entities", [])
            after = [linked(identifier) for identifier in before]
            if after != before:
                item["entities"] = after
                events.append(
                    {
                        "rule": "bind_declared_entity_alias",
                        "namespace": namespace,
                        "id": item["id"],
                        "before": before,
                        "after": after,
                    }
                )
            if namespace == "stages":
                for subgoal in item.get("subgoals", []):
                    before = subgoal["arguments"]
                    after = [linked(identifier) for identifier in before]
                    if after != before:
                        subgoal["arguments"] = after
                        events.append(
                            {
                                "rule": "bind_intermediate_subgoal_alias",
                                "stage": item["id"],
                                "before": before,
                                "after": after,
                            }
                        )
            elif item.get("articulated_part") is not None:
                before = item["articulated_part"]
                after = linked(before)
                if after != before:
                    item["articulated_part"] = after
                    events.append(
                        {
                            "rule": "bind_articulated_entity_alias",
                            "constraint": item["id"],
                            "before": before,
                            "after": after,
                        }
                    )
    assert result.get("goals") == payload.get("goals")
    assert result.get("geometry_targets") == payload.get("geometry_targets")
    assert len(result.get("entities", [])) == len(payload.get("entities", []))
    return result, events


def link_saved_outputs(dataset, experiment, output, protocol):
    records, root = read_records(dataset)
    by_id = {r["example_id"]: r for r in records}
    experiment = Path(experiment).resolve()
    output = Path(output).resolve()
    protocol = Path(protocol).resolve()
    frozen = json.loads(protocol.read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != frozen["linker_sha256"]:
        raise ValueError("Structural linker differs from its frozen protocol")
    manifest = json.loads((experiment / "manifest.json").read_text())
    if hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Dataset changed since inference")
    if (output / "manifest.json").exists() or (output / "predictions.jsonl").exists():
        raise FileExistsError("Linked experiment already exists; use a new --output")
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (experiment / "predictions.jsonl").read_text().splitlines() if line]
    selected = [by_id[eid] for eid in {r["example_id"] for r in rows}]
    samples = (
        load_marker_bundle(manifest["config"]["marker_bundle"], dataset, selected)
        if "markers" in manifest["config"]["conditions"]
        else {}
    )
    derived = []
    for source in rows:
        row = copy.deepcopy(source)
        row.update(
            derived=True,
            source_status=source["status"],
            source_cost_usd=source["cost_usd"],
            source_input_tokens=source["input_tokens"],
            source_output_tokens=source["output_tokens"],
            cost_usd=0.0,
            input_tokens=0,
            output_tokens=0,
            attempts=0,
            cached=False,
            latency_seconds=0.0,
        )
        row.pop("raw_response", None)
        row["source_raw_response"] = (
            str(experiment / source["raw_response"]) if source.get("raw_response") else None
        )
        payload = source.get("prediction") or source.get("unvalidated_prediction")
        if payload is not None and payload.get("schema_version") == "1.2":
            record = by_id[source["example_id"]]
            request = build_request(
                record["model_inputs"],
                root,
                source["condition"],
                manifest["prompt_text"],
                samples.get(source["example_id"]) if source["condition"] == "markers" else None,
            )
            linked, events = link_prediction(payload, request.validation_context)
            row["link_events"] = events
            row["unvalidated_prediction"] = linked
            try:
                prediction = GroundingPrediction.model_validate(linked, context=request.validation_context)
                row.update(status="ok", prediction=prediction.inference_dict())
                row.pop("validation_errors", None)
                row.pop("error", None)
            except ValidationError as error:
                row.update(
                    status="validation_failure",
                    prediction=None,
                    validation_errors=error.errors(
                        include_input=False, include_url=False, include_context=False
                    ),
                )
        derived.append(row)
    manifest = copy.deepcopy(manifest)
    manifest.update(
        source_experiment=str(experiment),
        source_ledger=manifest.get("ledger"),
        postprocessing={
            "method": "deterministic_declared_symbol_linking",
            "protocol": str(protocol),
            "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "linker_sha256": frozen["linker_sha256"],
            "benchmark_goals_accessed": False,
            "final_goal_declarations_changed": False,
            "human_review_performed": False,
        },
        ledger={
            "api_attempts": 0,
            "estimated_usage_cost_usd": 0.0,
            "unknown_usage_reserved_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "budget_usd": 0.0,
        },
        completed_predictions=len(derived),
        successful_predictions=sum(r["status"] == "ok" for r in derived),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in derived))
    return manifest
