"""Scope-aware checks of VLM proposals against independent physical evidence."""

import hashlib
import json

import numpy as np

from .bddl import normalize_predicate


AXIS_NAMES = {"axis", "joint_axis", "axis_world", "axis_local"}
LIMIT_NAMES = {"limits", "joint_limits", "joint_range"}


def claim(identifier, subject, kind, status, reason, scope, evidence=None):
    return {
        "id": identifier,
        "subject": subject,
        "kind": kind,
        "status": status,
        "reason": reason,
        "scope": scope,
        "evidence": evidence,
    }


def axis_check(parameter, fact, tolerance_degrees=5.0):
    value = parameter.get("value")
    if parameter.get("status") != "known" or not isinstance(value, list) or len(value) != 3:
        return "unresolved", "No numerical 3D axis was claimed.", None
    frame = parameter.get("frame")
    if frame == "world":
        expected = np.asarray(fact["axis_world"])
    elif frame in {"body_local", fact["body"]}:
        expected = np.asarray(fact["axis_local"])
    else:
        return "unresolved", "The axis frame is not comparable with the independently measured frames.", None
    proposed = np.asarray(value, dtype=float)
    length = np.linalg.norm(proposed)
    if not np.isfinite(proposed).all() or length < 1e-10:
        return "contradicted", "The claimed axis is not a finite nonzero vector.", None
    if parameter.get("units") == "unit_vector" and not np.isclose(length, 1.0, atol=1e-6):
        return "contradicted", "A claimed unit vector must have unit length.", {"length": float(length)}
    cosine = abs(float(np.dot(proposed / length, expected / np.linalg.norm(expected))))
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return (
        "verified" if angle <= tolerance_degrees else "contradicted",
        f"Sign-invariant axis error is {angle:.3f} degrees.",
        {"error_degrees": angle, "frame": frame},
    )


def limits_check(parameter, fact, tolerance=1e-6):
    value = parameter.get("value")
    if parameter.get("status") != "known" or not isinstance(value, list) or len(value) != 2:
        return "unresolved", "No lower/upper numerical joint limits were claimed.", None
    allowed_units = {"m", "meters"} if fact["coordinate_units"] == "m" else {"rad", "radians"}
    if parameter.get("units") not in allowed_units or parameter.get("frame") not in {
        "joint",
        "joint_coordinate",
        fact["joint_id"],
    }:
        return (
            "unresolved",
            "Joint limits require matching coordinate units and a joint-coordinate frame.",
            None,
        )
    proposed = np.asarray(value, dtype=float)
    valid = np.isfinite(proposed).all() and np.allclose(proposed, fact["limits"], atol=tolerance, rtol=0)
    return (
        "verified" if valid else "contradicted",
        "Compared with the simulator's actual limited-joint range.",
        {"expected_limits": fact["limits"], "units": fact["coordinate_units"]},
    )


def verify_proposal(prediction, facts, structural_errors=None):
    """Benchmark goals are intentionally not accepted as an argument."""
    checks, bindings = [], []
    if structural_errors:
        checks.append(
            claim(
                "structure",
                "proposal",
                "schema",
                "contradicted",
                "The proposal has unresolved structural/input-bound errors.",
                "typed input-only validation",
                structural_errors,
            )
        )
    else:
        checks.append(
            claim(
                "structure",
                "proposal",
                "schema",
                "verified",
                "The schema and supplied-identity relationships passed validation.",
                "structure only; not semantic or physical correctness",
            )
        )
    entities = {e["id"]: e for e in prediction.get("entities", [])}
    for index, goal in enumerate(prediction.get("goals", [])):
        args = [entities[a].get("reference_id") for a in goal["arguments"] if a in entities]
        name = normalize_predicate(goal["name"])
        if name == "in" and args == [facts["moving_object"], facts["target_region"]]:
            probes = facts["placement_probes"]
            tested = probes["open"]
            count = tested["nonpenetrating_in_count"]
            status = "verified" if count else "unresolved"
            checks.append(
                claim(
                    f"goal-{index}-placement",
                    f"{goal['name']}{args}",
                    "sampled_placement",
                    status,
                    f"{count} sampled poses satisfy the official reference-point In predicate and pass the penetration tolerance.",
                    "finite static poses at the tested orientation; not whole-object containment, reachable trajectory or dynamic stability",
                    {
                        "tested": tested["tested"],
                        "joint_q": tested["joint_q"],
                        "tolerance_m": facts["collision_penetration_tolerance_m"],
                    },
                )
            )
            full_count = tested["fully_contained_collision_box_count"]
            checks.append(
                claim(
                    f"goal-{index}-whole-volume",
                    f"{goal['name']}{args}",
                    "whole_object_site_proxy",
                    "verified" if full_count else "unresolved",
                    f"{full_count} sampled nonpenetrating poses also place every collision-box vertex inside the semantic site box.",
                    "conservative semantic-site proxy, not an exact drawer-volume model; no sample found is not proof of impossibility",
                )
            )
        elif name in {"open", "close"} and len(args) == 1 and args[0] in facts["joints"]:
            fact = facts["joints"][args[0]]
            reachable = any(s["open"] == (name == "open") for s in fact["sweep"])
            checks.append(
                claim(
                    f"goal-{index}-joint-state",
                    str(goal["arguments"]),
                    "joint_state",
                    "verified" if reachable else "unresolved",
                    "The forward-kinematic sweep includes the proposed joint state.",
                    "joint-state satisfiability only; terminal intent is not established",
                )
            )
        else:
            checks.append(
                claim(
                    f"goal-{index}-coverage",
                    str(goal["arguments"]),
                    "goal_coverage",
                    "unresolved",
                    "This goal/identity tuple is outside the trial's measured placement/state coverage.",
                    "no semantic correction from benchmark answers",
                )
            )
        if name in {"open", "close"} or goal.get("intent") == "uncertain":
            checks.append(
                claim(
                    f"goal-{index}-intent",
                    goal["name"],
                    "terminal_intent",
                    "unresolved",
                    "Mechanics cannot prove whether this state must persist at task completion.",
                    "instruction intent; requires interpretation or an ordinary desired-outcome clarification",
                )
            )
    for constraint in prediction.get("constraints", []):
        cid = constraint["id"]
        if constraint["category"] == "articulation":
            entity = entities.get(constraint.get("articulated_part"), {})
            fact = facts["joints"].get(entity.get("reference_id"))
            if fact is None:
                checks.append(
                    claim(
                        cid,
                        str(constraint.get("articulated_part")),
                        "mechanism",
                        "unresolved",
                        "The articulated entity has no independently measured joint binding.",
                        "identity must be grounded first",
                    )
                )
                continue
            kind = constraint.get("joint_type")
            status = (
                "unresolved"
                if kind in {None, "unknown"}
                else "verified"
                if kind == fact["joint_type"]
                else "contradicted"
            )
            checks.append(
                claim(
                    f"{cid}-type",
                    fact["entity_id"],
                    "joint_type",
                    status,
                    f"Simulator joint mechanism is {fact['joint_type']}.",
                    "kinematic mechanism; not task necessity",
                    {"joint_id": fact["joint_id"]},
                )
            )
            sweep_pass = fact["open_toggled_in_sweep"] and (
                fact["joint_type"] != "prismatic"
                or fact["translation_axis_residual_m"] < 1e-7
                and fact["rotation_residual"] < 1e-7
            )
            checks.append(
                claim(
                    f"{cid}-motion",
                    fact["entity_id"],
                    "mechanism_sweep",
                    "verified" if sweep_pass else "contradicted",
                    "Independent sweep checks body motion and the official Open predicate.",
                    "forward kinematics; no gripper path executed",
                    {
                        "axis_residual_m": fact["translation_axis_residual_m"],
                        "rotation_residual": fact["rotation_residual"],
                    },
                )
            )
            bindings.append(
                {
                    "constraint": cid,
                    "entity_id": fact["entity_id"],
                    "joint_id": fact["joint_id"],
                    "joint_type": fact["joint_type"],
                    "axis_world": fact["axis_world"],
                    "axis_local": fact["axis_local"],
                    "axis_point_world": fact["axis_point_world"],
                    "limits": fact["limits"],
                    "coordinate_units": fact["coordinate_units"],
                    "status": "verified",
                    "scope": "measured model parameters, independent of model confidence",
                }
            )
            parameters = {p["name"]: p for p in constraint.get("parameters", [])}
            axes = [(name, p) for name, p in parameters.items() if name in AXIS_NAMES]
            ranges = [(name, p) for name, p in parameters.items() if name in LIMIT_NAMES]
            for name, parameter in axes or [("axis", {})]:
                status, reason, evidence = axis_check(parameter, fact)
                checks.append(
                    claim(
                        f"{cid}-{name}",
                        fact["entity_id"],
                        "claimed_axis",
                        status,
                        reason,
                        "same-frame, sign-invariant axis comparison",
                        evidence,
                    )
                )
            for name, parameter in ranges or [("joint_limits", {})]:
                status, reason, evidence = limits_check(parameter, fact)
                checks.append(
                    claim(
                        f"{cid}-{name}",
                        fact["entity_id"],
                        "claimed_limits",
                        status,
                        reason,
                        "joint coordinate and units",
                        evidence,
                    )
                )
        elif constraint["category"] == "goal":
            matches = [
                g
                for g in prediction.get("goals", [])
                if normalize_predicate(g["name"]) == normalize_predicate(constraint["type"])
                and g["arguments"] == constraint["entities"]
            ]
            checks.append(
                claim(
                    cid,
                    constraint["type"],
                    "goal_declaration_link",
                    "verified" if matches else "unresolved",
                    "Compared the constraint's predicate/entity declaration with the generated final goals.",
                    "internal declaration consistency only; not benchmark truth or physical completion",
                )
            )
        else:
            checks.append(
                claim(
                    cid,
                    constraint["type"],
                    "strategy_or_path",
                    "unresolved",
                    "Static geometry and joint sweeps do not establish a grasp, contact strategy or executable robot path.",
                    "no task necessity or motion-feasibility proof",
                )
            )
    closed, opened = (facts["placement_probes"][name] for name in ("closed", "open"))
    checks.append(
        claim(
            "access-order",
            "opening before insertion",
            "access_requirement",
            "unresolved",
            f"Static-pose tests passed in {closed['fully_contained_collision_box_count']} closed and {opened['fully_contained_collision_box_count']} open samples. End-state pose tests do not prove insertion-path access or universal necessity.",
            "finite pose sampling is not a trajectory planner",
        )
    )
    checks.append(
        claim(
            "robot-execution",
            "complete task",
            "execution",
            "unresolved",
            "Grasp stability, contact forces, collision-free robot motion and completion were not tested.",
            "robot execution not attempted",
        )
    )
    counts = {
        status: sum(c["status"] == status for c in checks)
        for status in ("verified", "contradicted", "unresolved")
    }
    return {
        "schema_version": "1.0",
        "proposal_sha256": hashlib.sha256(json.dumps(prediction, sort_keys=True).encode()).hexdigest(),
        "facts_sha256": hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest(),
        "benchmark_goal_answers_used": False,
        "claims": checks,
        "counts": counts,
        "verified_bindings": bindings,
        "ready_for_robot_execution": False,
        "scope": "Every status is limited to its stated evidence scope; verified kinematics do not certify a complete manipulation plan.",
    }


def verifier_feedback(prediction, report, facts):
    """An explicit oracle-assisted feedback boundary; no evaluator answers."""
    target = facts["joints"][facts["target_region"]]
    return {
        "source": "independent_simulator_verification; oracle-assisted correction",
        "prior_proposal": prediction,
        "checks": [
            {key: c[key] for key in ("subject", "kind", "status", "reason", "scope")}
            for c in report["claims"]
        ],
        "measured_joint": {
            key: target[key]
            for key in (
                "entity_id",
                "parent_id",
                "joint_id",
                "body",
                "joint_type",
                "axis_local",
                "axis_world",
                "axis_point_world",
                "limits",
                "coordinate_units",
            )
        },
        "scope": "These are geometry/kinematic facts, not required task strategies or benchmark goal answers. Keep every generated constraint verified=false.",
    }
