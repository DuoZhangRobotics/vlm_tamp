"""Two-stage public evidence packets for a bounded Luna decision diagnostic.

Nothing here runs a simulator or learns a policy. Before and after packets are
separate so the observed result cannot enter the prediction turn. Private case
selection, grading criteria and native measurements belong to the runner.
"""
import json
from pathlib import Path
import shutil

from .checkpoint_diagnostic import (
    ATTACHED_CAMERAS, build_packet, prefix_events, read_json, sha256,
    verify_packet, write_json,
)
from .compact_robot_interface import compact_response


COMMON_BEFORE = """Assess a recorded robot episode immediately before one recorded request.
The task is in inputs/before/checkpoint.json. Read its current observation,
inputs/before/pending-request.json, and inputs/before/history-compact.jsonl.
Full replies, earlier public observations, tools/full.txt, tools/<operation>.txt,
and experience/guidelines.json are available in inputs/before/. The provisional
guide is prior experience, not a guaranteed recipe. Files referenced as
robot_data/... are relative to the packet directory. The attached original
images show the BEFORE state, in order: frontview, agentview, wrist, arm_side.

This is offline. No robot service is running. The request is recorded, not a
request you can change or execute. The result will be revealed in a later turn.
You may inspect public files/images and run local NumPy/SciPy/Pillow analysis.
Do not obtain project code, other episodes, simulator data or external services.
Do not modify inputs. Keep your public note concise, not a reasoning transcript.
Write before.json with exactly: note (string), evidence (list of references),
uncertainties (list of strings). Then end this turn.
"""

BEFORE_INSTRUCTIONS = {
    "baseline": "Summarize the current situation and the recorded request. Record the most relevant evidence and uncertainties in before.json.\n",
    "predict-inspect-revise": "Before seeing the result, briefly predict the observable effect of the recorded request. Name the evidence you would inspect to check that prediction and what remains uncertain. Put this compact expectation in before.json.\n",
}

COMMON_AFTER = """The recorded request has returned. Its original public result is in
inputs/after/latest-result.json; the compact result is latest-result-compact.json.
The full history and public assets through this result are in inputs/after/.
Attached images are BEFORE frontview, AFTER frontview, BEFORE agentview,
AFTER agentview, BEFORE wrist, AFTER wrist, BEFORE arm_side, AFTER arm_side.
These are original pixels, not annotated interpretations. When a request did
not return a new observation, the same last sensor snapshot appears twice;
that is not a new capture. Check observation IDs and the returned result.

Assess the observed result and choose one next step. Do not execute any robot
request or invent its outcome. Read a relevant tool contract if proposing a
robot request. Finish with decision.json using the exact schema in
inputs/after/decision-schema.json. Enumerated labels are reporting categories,
not hints about this case. Cite accessible evidence; report uncertainty where
the observations do not establish an answer. Keep text fields concise and
record only conclusions and supporting evidence, not a reasoning transcript.
"""

AFTER_INSTRUCTIONS = {
    "baseline": "Use the available evidence to assess the current state and choose the next step.\n",
    "predict-inspect-revise": "Compare the before/after evidence with your recorded expectation in before.json. Briefly state what matched or differed, revise your current working assumption as needed, and choose the next step from that updated assessment.\n",
}


def enum(*values):
    return {"type": "string", "enum": list(values)}


DECISION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "motion_assessment": enum("moved", "no_visible_motion", "uncertain", "not_executed"),
        "pose_update_basis": enum("measured_rgbd", "hand_prediction", "none"),
        "contact_assessment": enum("retained", "lost", "uncertain", "not_assessed"),
        "task_success": {"type": "boolean"},
        "next_step": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "kind": enum("inspect_existing", "request_observation", "reestimate", "recover_contact",
                             "continue_motion", "replan", "stop_success", "stop_budget"),
                "request": {"anyOf": [
                    {"type": "null"},
                    {"type": "object", "additionalProperties": False, "properties": {
                        "operation": {"type": "string", "minLength": 1},
                        "arguments": {"type": "object"}}, "required": ["operation", "arguments"]}]},
                "description": {"type": "string", "minLength": 1},
            }, "required": ["kind", "request", "description"],
        },
        "observed_changes": {"type": "string", "minLength": 1},
        "evidence": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
        "uncertainty": {"type": "string", "minLength": 1},
        "working_note": {"type": "string", "minLength": 1},
    },
}
DECISION_SCHEMA["required"] = list(DECISION_SCHEMA["properties"])


def refresh_manifest(packet, manifest):
    manifest["packet_sha256"] = {
        str(p.relative_to(packet)): sha256(p) for p in sorted(packet.rglob("*")) if p.is_file()
    }
    return manifest


def build_stages(source, event, destination, *, shared_sources):
    """Return private provenance, keeping it outside both public stages."""
    source, destination = Path(source), Path(destination)
    if type(event) is not int or event < 2:
        raise ValueError("Two-stage diagnostics require an event after reset")
    manifests = {}
    for stage, cutoff in [("before", event - 1), ("after", event)]:
        packet = destination / stage
        manifest = build_packet(source, cutoff, packet, shared_sources=shared_sources)
        # The original index refers to these manuals. Both conditions receive
        # the same robot documentation; no live RPC client is supplied.
        manuals = Path(shared_sources["tools.txt"]).parent / "tools"
        shutil.copytree(manuals, packet / "tools")
        rows = prefix_events(source / "tool-events.jsonl", cutoff)
        with (packet / "history-compact.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps({"sequence": row["sequence"], "request": row["request"],
                                          "response": compact_response(row["response"])}) + "\n")
        manifests[stage] = manifest
    target = prefix_events(source / "tool-events.jsonl", event)[-1]
    write_json(destination / "before/pending-request.json", target["request"])
    write_json(destination / "after/latest-result.json", target)
    write_json(destination / "after/latest-result-compact.json", {
        "sequence": event, "request": target["request"], "response": compact_response(target["response"])})
    write_json(destination / "after/decision-schema.json", DECISION_SCHEMA)
    for stage, manifest in manifests.items():
        refresh_manifest(destination / stage, manifest)
        verify_packet(destination / stage, manifest)
    return manifests


def attached_images(packet_root, manifests, stage):
    packet_root = Path(packet_root)
    if stage == "before":
        return [packet_root / "before" / p for p in manifests["before"]["current_images"]]
    if stage != "after":
        raise ValueError("Unknown stage")
    return [packet_root / side / manifests[side]["current_images"][i]
            for i in range(len(ATTACHED_CAMERAS)) for side in ["before", "after"]]


def validate_note(path):
    value = read_json(path)
    if not isinstance(value, dict) or set(value) != {"note", "evidence", "uncertainties"}:
        raise ValueError("Invalid before.json fields")
    if not isinstance(value["note"], str) or not value["note"].strip():
        raise ValueError("Missing before note")
    for key in ["evidence", "uncertainties"]:
        if not isinstance(value[key], list) or not all(isinstance(x, str) for x in value[key]):
            raise ValueError("Invalid note list: " + key)
    return value


def validate_decision(path):
    import jsonschema
    value = read_json(path)
    jsonschema.validate(value, DECISION_SCHEMA)
    return value


def score_labels(decision, expected):
    """Registered label checks, separate from evidence/next-step human review.

    'Uncertain' remains a permitted response, but does not earn a successful
    motion discrimination. Contact loss is never forced from private contacts.
    """
    checks = {
        "motion_discrimination": decision["motion_assessment"] in expected["motion_labels"],
        "pose_provenance": decision["pose_update_basis"] == expected["pose_update_basis"],
        "success_status": decision["task_success"] == expected["task_success"],
        "next_step_compatible": decision["next_step"]["kind"] in expected["next_step_kinds"],
    }
    if expected.get("contact_loss_case"):
        checks["contact_not_assumed_retained"] = decision["contact_assessment"] in ["lost", "uncertain"]
    if expected["task_success"]:
        checks["no_request_after_success"] = decision["next_step"]["request"] is None
    return {"checks": checks, "all_label_checks_passed": all(checks.values()),
            "scope": "Structured-label screen only. Evidence support and action feasibility require separate review."}
