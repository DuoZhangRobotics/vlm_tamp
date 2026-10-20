"""Track A and reviewed Track B scoring with explicit coverage denominators."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .annotations import load_annotations
from .bddl import normalize_predicate
from .dataset import ORACLE_CONDITIONS, read_records, safe_path
from .grounding import GroundingPrediction
from .references import attach_reference, load_simulator_references


EVALUATION_VERSION = "1.4"
ARTICULATION_TYPE_ALIASES = {
    "prismatic": "prismatic",
    "prismatic_joint": "prismatic",
    "follow_prismatic_joint": "prismatic",
    "revolute": "revolute",
    "revolute_joint": "revolute",
    "follow_revolute_joint": "revolute",
}


# Closed, documented category aliases. No unrestricted semantic or substring match.
CATEGORY_ALIASES = {
    "akita_black_bowl": "bowl",
    "black_bowl": "bowl",
    "bowl": "bowl",
    "wine_bottle": "bottle",
    "bottle": "bottle",
    "wine bottle": "bottle",
    "wooden_cabinet": "cabinet",
    "cabinet": "cabinet",
    "drawer": "drawer",
    "flat_stove": "stove",
    "stove": "stove",
    "plate": "plate",
    "microwave": "microwave",
    "white_bowl": "bowl",
    "white_mug": "mug",
    "mug": "mug",
    "red_mug": "mug",
    "yellow_and_white_mug": "mug",
}


def score_sets(predicted, reference, complete=True):
    p, r = set(predicted), set(reference)
    tp, fp, fn = len(p & r), len(p - r), len(r - p)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not r else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp if complete else None,
        "fn": fn,
        "precision": precision if complete else None,
        "recall": recall,
        "f1": f1 if complete else None,
        "exact_match": p == r if complete else None,
        "unjudged_predictions": 0 if complete else fp,
    }


def reference_category(identifier, entities):
    entity = entities.get(identifier)
    if not entity:
        return "unresolved"
    category = entity["category"]
    if (
        category == "region"
        and entity.get("parent")
        and "cabinet" in entity["parent"]
        and any(identifier.endswith(f"_{x}_region") for x in ("top", "middle", "bottom"))
    ):
        return "drawer"
    return CATEGORY_ALIASES.get(category, category)


def bbox_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def axis_error_degrees(predicted, reference, predicted_frame, reference_frame, sign_invariant=True):
    if predicted_frame is None or predicted_frame != reference_frame:
        return None
    a, b = np.asarray(predicted, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return None
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    if sign_invariant:
        cosine = abs(cosine)
    return math.degrees(math.acos(np.clip(cosine, -1, 1)))


def reachability(edges):
    reached = set(edges)
    changed = True
    while changed:
        expanded = reached | {(a, d) for a, b in reached for c, d in reached if b == c}
        changed = expanded != reached
        reached = expanded
    return reached


def evaluate_track_b(prediction, annotation, resolved_entities=None):
    if annotation is None or annotation.status != "reviewed":
        return {
            "evaluable": False,
            "reason": "expert annotations not reviewed",
            "unjudged_constraints": len(prediction.constraints),
        }
    entity_map = {
        e.id: (resolved_entities.get(e.id) if resolved_entities is not None else e.reference_id)
        or f"unresolved:{e.id}"
        for e in prediction.entities
    }

    def stage_sig(stage, mapping):
        return stage.action, tuple(mapping.get(e, e) for e in stage.entities)

    pred_stages = {s.id: stage_sig(s, entity_map) for s in prediction.stages}
    scores = []
    for plan in annotation.alternatives:
        ref_stages = {s.id: stage_sig(s, {}) for s in plan.stages}
        entry = {"alternative": plan.id, "evaluable": True}
        if annotation.coverage.stages:
            entry["stages"] = score_sets(
                pred_stages.values(), ref_stages.values(), annotation.coverage.stages_complete
            )
        if annotation.coverage.dependencies:
            p = reachability(
                [(d.before, d.after) for d in prediction.dependencies if d.requirement != "optional"]
            )
            r = reachability([(d.before, d.after) for d in plan.dependencies if d.requirement != "optional"])
            p = {(pred_stages[a], pred_stages[b]) for a, b in p}
            r = {(ref_stages[a], ref_stages[b]) for a, b in r}
            entry["dependencies"] = score_sets(p, r, annotation.coverage.dependencies_complete)

        def constraint_sig(c, mapping, stages):
            return (
                c.category,
                c.type,
                tuple(mapping.get(e, e) for e in c.entities),
                c.scope,
                c.requirement,
                tuple(sorted(stages[s] for s in c.stages)),
            )

        if annotation.coverage.constraints:
            categories = set(annotation.coverage.constraint_categories)
            p = [c for c in prediction.constraints if not categories or c.category in categories]
            r = [c for c in plan.constraints if not categories or c.category in categories]
            entry["constraints"] = score_sets(
                [constraint_sig(c, entity_map, pred_stages) for c in p],
                [constraint_sig(c, {}, ref_stages) for c in r],
                annotation.coverage.constraints_complete,
            )
            entry["by_category"] = {
                category: score_sets(
                    [constraint_sig(c, entity_map, pred_stages) for c in p if c.category == category],
                    [constraint_sig(c, {}, ref_stages) for c in r if c.category == category],
                    annotation.coverage.constraints_complete,
                )
                for category in sorted({c.category for c in r + p})
            }
            axis_errors = []
            for pc in p:
                matching = [
                    rc
                    for rc in r
                    if constraint_sig(pc, entity_map, pred_stages) == constraint_sig(rc, {}, ref_stages)
                ]
                for rc in matching:
                    for param in pc.parameters:
                        rp = next((x for x in rc.parameters if x.name == param.name), None)
                        if (
                            rp
                            and param.name == "axis"
                            and isinstance(param.value, list)
                            and isinstance(rp.value, list)
                        ):
                            value = axis_error_degrees(
                                param.value, rp.value, param.frame, rp.frame, annotation.axis_sign_invariant
                            )
                            if value is not None:
                                axis_errors.append(value)
            entry["axis_angular_errors_degrees"] = axis_errors
        scores.append(entry)
    if not scores:
        return {
            "evaluable": False,
            "reason": "no reference alternatives",
            "unjudged_constraints": len(prediction.constraints),
        }
    return max(
        scores,
        key=lambda x: sum(
            x[k].get("f1") if x[k].get("f1") is not None else x[k]["recall"]
            for k in ("stages", "dependencies", "constraints")
            if k in x
        ),
    )


def evaluate_articulation_facts(prediction, reference, resolved_entities):
    if reference is None or not reference.parts:
        return {"evaluable": False, "reference_parts": 0, "goal_parts": 0}
    parts = {p.entity_id: p for p in reference.parts}
    goals = {p.entity_id for p in reference.parts if p.goal_related}
    identified = {
        resolved_entities[e.id]
        for e in prediction.entities
        if e.kind in {"part", "region"} and resolved_entities[e.id] in parts
    }
    typed, type_claims, matched_types, correct_types = set(), 0, 0, 0
    joint_claims, correct_joints, incompatible_frames, conflicting_types = 0, 0, 0, 0
    axis_errors, unclassified = [], 0
    explicit_known, explicit_unknown = 0, 0
    for constraint in prediction.constraints:
        if constraint.category != "articulation":
            continue
        parameters = {p.name: p for p in constraint.parameters}
        claimed = set()
        if constraint.joint_type in {"prismatic", "revolute"}:
            claimed.add(constraint.joint_type)
            explicit_known += 1
        elif constraint.joint_type == "unknown":
            explicit_unknown += 1
        if prediction.schema_version == "1.0" and constraint.type.lower() in ARTICULATION_TYPE_ALIASES:
            claimed.add(ARTICULATION_TYPE_ALIASES[constraint.type.lower()])
        for key in ("joint_type", "articulation_type") if prediction.schema_version == "1.0" else ():
            param = parameters.get(key)
            if param and isinstance(param.value, str) and param.value.lower() in ARTICULATION_TYPE_ALIASES:
                claimed.add(ARTICULATION_TYPE_ALIASES[param.value.lower()])
        if not claimed:
            unclassified += 1
        if len(claimed) > 1:
            conflicting_types += 1
        type_claims += bool(claimed)
        entity_ids = (
            [constraint.articulated_part] if constraint.articulated_part is not None else constraint.entities
        )
        targets = {resolved_entities[e] for e in entity_ids if resolved_entities[e] in parts}
        for target in targets:
            fact = parts[target]
            if claimed:
                matched_types += 1
                if claimed == {fact.joint_type}:
                    correct_types += 1
                    typed.add(target)
            joint = parameters.get("joint_id") or parameters.get("joint")
            if joint and isinstance(joint.value, str):
                joint_claims += 1
                correct_joints += joint.value == fact.joint_id
            axis = parameters.get("axis")
            if axis and isinstance(axis.value, list):
                target_axis = (
                    fact.axis_world
                    if axis.frame == "world"
                    else fact.axis_local
                    if axis.frame == fact.body
                    else None
                )
                if target_axis is None:
                    incompatible_frames += 1
                    continue
                angle = axis_error_degrees(
                    axis.value, target_axis, axis.frame, axis.frame, fact.axis_sign_invariant
                )
                if angle is not None:
                    axis_errors.append(angle)
    return {
        "evaluable": True,
        "reference_parts": len(parts),
        "visible_parts": sum(p.visible_pixels > 0 for p in reference.parts),
        "goal_parts": len(goals),
        "identified_goal_parts": len(identified & goals),
        "goal_part_identification_recall": len(identified & goals) / len(goals) if goals else None,
        "joint_type_claims": type_claims,
        "explicit_known_joint_types": explicit_known,
        "explicit_unknown_joint_types": explicit_unknown,
        "matched_type_claims": matched_types,
        "correct_type_claims": correct_types,
        "conditional_joint_type_accuracy": correct_types / matched_types if matched_types else None,
        "goal_part_joint_type_recall": len(typed & goals) / len(goals) if goals else None,
        "unclassified_articulation_declarations": unclassified,
        "conflicting_joint_types": conflicting_types,
        "joint_identity_claims": joint_claims,
        "correct_joint_identities": correct_joints,
        "joint_identity_accuracy": correct_joints / joint_claims if joint_claims else None,
        "axis_angular_errors_degrees": axis_errors,
        "incompatible_axis_frames": incompatible_frames,
        "note": "Kinematic facts do not establish required stages, grasp/contact strategy, or path constraints.",
    }


def evaluate_example(record, row, root, annotation=None, simulator_reference=None):
    if simulator_reference is not None:
        record = attach_reference(record, root, simulator_reference)
    prediction = (
        GroundingPrediction.model_validate(row["prediction"])
        if row.get("prediction")
        else GroundingPrediction()
    )
    entities = {e["identifier"]: e for e in record["reference"]["entities"]}
    predicted = {e.id: e for e in prediction.entities}
    goals = record["reference"]["goal_predicates"]
    p = [
        (
            normalize_predicate(g.name),
            tuple(predicted[a].reference_id or f"unresolved:{a}" for a in g.arguments),
        )
        for g in prediction.goals
    ]
    r = [(normalize_predicate(g["name"]), tuple(g["arguments"])) for g in goals]
    evaluable = record["reference"]["coverage"]["symbolic_goals"]
    intent_resolved = prediction.schema_version != "1.2" or all(
        g.intent != "uncertain" for g in prediction.goals
    )
    symbolic = score_sets(p, r) if evaluable else None
    if symbolic is not None and (prediction.goal_logic != "and" or not intent_resolved):
        symbolic = score_sets([], r)
    pc = [
        (
            normalize_predicate(g.name),
            tuple(CATEGORY_ALIASES.get(predicted[a].category, predicted[a].category) for a in g.arguments),
        )
        for g in prediction.goals
    ]
    rc = [
        (normalize_predicate(g["name"]), tuple(reference_category(a, entities) for a in g["arguments"]))
        for g in goals
    ]
    category_goals = score_sets(pc, rc) if evaluable else None
    types = (
        score_sets(
            [normalize_predicate(g.name) for g in prediction.goals],
            [normalize_predicate(g["name"]) for g in goals],
        )
        if evaluable
        else None
    )
    # Category accuracy pairs only predicates with identical names and arities.
    available = list(rc)
    correct_categories, category_denominator = 0, sum(len(a) for _, a in rc)
    for name, args in pc:
        candidates = [
            (i, sum(a == b for a, b in zip(args, ra)))
            for i, (rn, ra) in enumerate(available)
            if rn == name and len(ra) == len(args)
        ]
        if candidates:
            idx, correct = max(candidates, key=lambda x: x[1])
            correct_categories += correct
            available.pop(idx)
    reference_ids = {a for g in goals for a in g["arguments"]}
    visible = record["reference"]["visual"]
    # Unique role correspondence permits pixel scoring without exposing identities to the model.
    expected_roles = defaultdict(set)
    for pg in prediction.goals:
        matching = [
            rg
            for rg in goals
            if normalize_predicate(rg["name"]) == normalize_predicate(pg.name)
            and len(rg["arguments"]) == len(pg.arguments)
        ]
        if len(matching) == 1:
            for semantic_id, reference_id in zip(pg.arguments, matching[0]["arguments"]):
                expected_roles[semantic_id].add(reference_id)
    oracle = row["condition"] in ORACLE_CONDITIONS
    resolved_entities = {e.id: e.reference_id if oracle else None for e in prediction.entities}
    ious, point_hits, localized, loc_details = [], [], set(), []
    for entity in prediction.entities:
        refid = entity.reference_id if oracle else None
        localized_id = None
        if row["condition"] == "language":
            continue
        if entity.camera not in {None, record["metadata"]["camera"]["name"]}:
            loc_details.append({"entity": entity.id, "status": "incompatible_camera"})
            continue
        if refid in visible:
            reference = visible[refid]
            if entity.bbox is not None:
                iou = bbox_iou(entity.bbox, reference["bbox"])
                ious.append(iou)
                if iou >= 0.5:
                    localized_id = refid
            if entity.point is not None:
                mask = np.array(Image.open(safe_path(root, reference["mask"]))) > 0
                x = min(mask.shape[1] - 1, int(entity.point[0] * mask.shape[1]))
                y = min(mask.shape[0] - 1, int(entity.point[1] * mask.shape[0]))
                hit = bool(mask[y, x])
                point_hits.append(int(hit))
                if hit:
                    localized_id = refid
        elif refid is None and (entity.point is not None or entity.bbox is not None):
            expected = expected_roles[entity.id]
            if len(expected) == 1 and next(iter(expected)) in visible:
                reference = visible[next(iter(expected))]
                if entity.bbox is not None:
                    ious.append(bbox_iou(entity.bbox, reference["bbox"]))
                if entity.point is not None:
                    mask = np.array(Image.open(safe_path(root, reference["mask"]))) > 0
                    x, y = (
                        min(mask.shape[1] - 1, int(entity.point[0] * mask.shape[1])),
                        min(mask.shape[0] - 1, int(entity.point[1] * mask.shape[0])),
                    )
                    point_hits.append(int(mask[y, x]))
            # Resolve by independently measured localization, never by category alone.
            matches = []
            for identifier, reference in visible.items():
                reference_kind = reference.get("kind", "object")
                if entity.kind == "object" and reference_kind != "object":
                    continue
                if entity.kind in {"part", "region"} and reference_kind != "part":
                    continue
                if entity.point is not None:
                    mask = np.array(Image.open(safe_path(root, reference["mask"]))) > 0
                    x, y = (
                        min(mask.shape[1] - 1, int(entity.point[0] * mask.shape[1])),
                        min(mask.shape[0] - 1, int(entity.point[1] * mask.shape[0])),
                    )
                    if mask[y, x]:
                        matches.append(identifier)
                elif bbox_iou(entity.bbox, reference["bbox"]) >= 0.5:
                    matches.append(identifier)
            if len(matches) == 1:
                localized_id = matches[0]
        if localized_id:
            localized.add(localized_id)
            if refid is None:
                resolved_entities[entity.id] = localized_id
        if entity.point is not None or entity.bbox is not None:
            loc_details.append({"entity": entity.id, "resolved_by_pixels": localized_id})
    p = [
        (normalize_predicate(g.name), tuple(resolved_entities[a] or f"unresolved:{a}" for a in g.arguments))
        for g in prediction.goals
    ]
    if evaluable:
        symbolic = score_sets(p if prediction.goal_logic == "and" and intent_resolved else [], r)
    # Exploratory diagnostic: preserve strict logic-gated scoring above.
    symbolic_content = score_sets(p, r) if evaluable else None
    target_visible = reference_ids & set(visible)
    object_roles = []
    for name, args in r:
        for position, identifier in enumerate(args):
            if identifier not in visible or entities[identifier]["kind"] != "object":
                continue
            matches = [pa[position] for pn, pa in p if pn == name and len(pa) == len(args)]
            object_roles.append(
                {
                    "predicate": name,
                    "argument_index": position,
                    "reference_id": identifier,
                    "predicted_ids": matches,
                    "correct": identifier in matches,
                }
            )
    relevant_annotation = (
        annotation is not None
        and annotation.status == "reviewed"
        and annotation.relevant_entities is not None
    )
    reliability = {
        "hallucinated_instance_ids": sum(
            e.reference_id is not None and e.reference_id not in entities for e in prediction.entities
        ),
        "invalid_predicates": sum(
            normalize_predicate(g.name)
            not in {"on", "in", "open", "close", "turnon", "turnoff", "at", "stack"}
            for g in prediction.goals
        ),
        "ambiguous_entities": sum(e.ambiguous for e in prediction.entities),
        "identity_abstentions": sum(e.reference_id is None for e in prediction.entities),
        "declared_abstentions": len(prediction.abstentions),
        "logical_status_abstention": prediction.goal_logic == "unresolved",
        "terminal_intent_abstentions": sum(g.intent == "uncertain" for g in prediction.goals),
        "geometry_descriptions": len(prediction.geometry_targets),
        "unsupported_identity_claims": sum(e.reference_id is not None for e in prediction.entities)
        if not oracle
        else 0,
        "unsupported_language_localization": sum(
            e.point is not None or e.bbox is not None for e in prediction.entities
        )
        if row["condition"] == "language"
        else 0,
        "dependency_cycles": sum(
            a == b for a, b in reachability([(d.before, d.after) for d in prediction.dependencies])
        ),
        "contradictory_goals": sum(
            (other, args) in set(p)
            for name, args in set(p)
            for other in ({"open": ["close"], "turnon": ["turnoff"]}.get(name, []))
        ),
        "unverified_numerical_parameters": sum(
            param.value is not None and isinstance(param.value, (float, list))
            for c in prediction.constraints
            for param in c.parameters
        ),
        "validation_failure": row["status"] in {"validation_failure", "cache_validation_failure"},
        "api_failure": row["status"] == "api_failure",
        "budget_stopped": row["status"] == "budget_stopped",
    }
    visual = {
        "setting": "oracle_assisted" if oracle else "unconstrained",
        "visible_whole_object_goal_roles": {
            "correct": sum(role["correct"] for role in object_roles),
            "denominator": len(object_roles),
            "accuracy": sum(role["correct"] for role in object_roles) / len(object_roles)
            if object_roles
            else None,
            "details": object_roles,
            "scope": "ordered final-goal object arguments; identity only, independent of logical-status flag",
        },
        "bbox_iou_mean": float(np.mean(ious)) if ious else None,
        "bbox_evaluable": len(ious),
        "point_in_mask_accuracy": float(np.mean(point_hits)) if point_hits else None,
        "points_evaluable": len(point_hits),
        "localized_target_recall": len(localized & target_visible) / len(target_visible)
        if target_visible
        else None,
        "visible_target_denominator": len(target_visible),
        "localization_attempts": len(loc_details),
        "localized_entities": sorted(localized),
        "localization_details": loc_details,
        "instance_identification": score_sets(
            [e.reference_id for e in prediction.entities if e.reference_id], annotation.relevant_entities
        )
        if relevant_annotation
        else None,
        "part_identification": score_sets(
            [e.reference_id for e in prediction.entities if e.kind == "part" and e.reference_id],
            annotation.manipulable_parts,
        )
        if annotation and annotation.status == "reviewed" and annotation.manipulable_parts is not None
        else None,
    }
    return {
        "example_id": record["example_id"],
        "suite": record["metadata"]["suite"],
        "task_index": record["metadata"]["task_index"],
        "state_index": record["metadata"]["state_index"],
        "condition": row["condition"],
        "model": row["model"],
        "provider": row["provider"],
        "offline_mock": row["offline_mock"],
        "status": row["status"],
        "symbolic": symbolic,
        "symbolic_content": symbolic_content,
        "goal_logic": prediction.goal_logic,
        "terminal_intent_resolved": intent_resolved,
        "predicate_types": types,
        "category_goals": category_goals,
        "object_category_correct": correct_categories,
        "object_category_denominator": category_denominator,
        "visual": visual,
        "reliability": reliability,
        "evaluation_version": EVALUATION_VERSION,
        "track_b": evaluate_track_b(prediction, annotation, resolved_entities),
        "articulation_facts": evaluate_articulation_facts(prediction, simulator_reference, resolved_entities),
        "latency_seconds": row["latency_seconds"],
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "cost_usd": row["cost_usd"],
        "cached": row["cached"],
    }


def aggregate(rows, per_task=False):
    groups = defaultdict(list)
    for row in rows:
        key = (row["provider"], row["model"], row["suite"], row["condition"], row["offline_mock"])
        if per_task:
            key += (row["task_index"],)
        groups[key].append(row)
    summaries = []
    for key, examples in groups.items():
        provider, model, suite, condition, offline = key[:5]
        scored = [e for e in examples if e["symbolic"] is not None]
        tp, fp, fn = (sum(e["symbolic"][k] for e in scored) for k in ("tp", "fp", "fn"))
        precision, recall = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
        cat_den = sum(e["object_category_denominator"] for e in scored)
        bbox_n = sum(e["visual"]["bbox_evaluable"] for e in examples)
        point_n = sum(e["visual"]["points_evaluable"] for e in examples)
        target_n = sum(e["visual"]["visible_target_denominator"] for e in examples)
        object_role_n = sum(e["visual"]["visible_whole_object_goal_roles"]["denominator"] for e in examples)
        object_role_correct = sum(e["visual"]["visible_whole_object_goal_roles"]["correct"] for e in examples)
        summary = {
            "provider": provider,
            "model": model,
            "suite": suite,
            "condition": condition,
            "offline_mock": offline,
            "examples": len(examples),
            "successful": sum(e["status"] == "ok" for e in examples),
            "symbolic_evaluable": len(scored),
            "predicate_precision": precision,
            "predicate_recall": recall,
            "predicate_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "goal_exact_match": sum(e["symbolic"]["exact_match"] for e in scored) / len(scored)
            if scored
            else None,
            "predicate_content_exact_match": sum(e["symbolic_content"]["exact_match"] for e in scored)
            / len(scored)
            if scored
            else None,
            "predicate_type_exact_match": sum(e["predicate_types"]["exact_match"] for e in scored)
            / len(scored)
            if scored
            else None,
            "category_goal_exact_match": sum(e["category_goals"]["exact_match"] for e in scored) / len(scored)
            if scored
            else None,
            "object_category_accuracy": sum(e["object_category_correct"] for e in scored) / cat_den
            if cat_den
            else None,
            "bbox_iou_mean": sum(
                (e["visual"]["bbox_iou_mean"] or 0) * e["visual"]["bbox_evaluable"] for e in examples
            )
            / bbox_n
            if bbox_n
            else None,
            "bbox_evaluable": bbox_n,
            "point_in_mask_accuracy": sum(
                (e["visual"]["point_in_mask_accuracy"] or 0) * e["visual"]["points_evaluable"]
                for e in examples
            )
            / point_n
            if point_n
            else None,
            "points_evaluable": point_n,
            "localized_target_recall": sum(
                (e["visual"]["localized_target_recall"] or 0) * e["visual"]["visible_target_denominator"]
                for e in examples
            )
            / target_n
            if target_n
            else None,
            "visible_target_denominator": target_n,
            "visible_whole_object_goal_roles_correct": object_role_correct,
            "visible_whole_object_goal_roles_denominator": object_role_n,
            "visible_whole_object_goal_role_accuracy": object_role_correct / object_role_n
            if object_role_n
            else None,
            "track_b_evaluable": sum(e["track_b"]["evaluable"] for e in examples),
            "evaluation_version": EVALUATION_VERSION,
            "simulator_fact_examples": sum(e["articulation_facts"]["evaluable"] for e in examples),
            "reference_goal_parts": sum(e["articulation_facts"].get("goal_parts", 0) for e in examples),
            "identified_goal_parts": sum(
                e["articulation_facts"].get("identified_goal_parts", 0) for e in examples
            ),
            "matched_joint_type_claims": sum(
                e["articulation_facts"].get("matched_type_claims", 0) for e in examples
            ),
            "correct_joint_type_claims": sum(
                e["articulation_facts"].get("correct_type_claims", 0) for e in examples
            ),
            "explicit_known_joint_types": sum(
                e["articulation_facts"].get("explicit_known_joint_types", 0) for e in examples
            ),
            "explicit_unknown_joint_types": sum(
                e["articulation_facts"].get("explicit_unknown_joint_types", 0) for e in examples
            ),
            "latency_seconds_mean": float(np.mean([e["latency_seconds"] for e in examples])),
            "input_tokens": sum(e["input_tokens"] for e in examples),
            "output_tokens": sum(e["output_tokens"] for e in examples),
            "cost_usd": sum(e["cost_usd"] for e in examples),
            "cached": sum(e["cached"] for e in examples),
            "reliability": {
                k: sum(e["reliability"][k] for e in examples) for k in examples[0]["reliability"]
            },
        }
        if not scored:
            for field in ("predicate_precision", "predicate_recall", "predicate_f1"):
                summary[field] = None
        if per_task:
            summary["task_index"] = key[5]
        summaries.append(summary)
    return summaries


def evaluate_saved(dataset, experiment, output=None, annotations=None, simulator_references=None):
    records, root = read_records(dataset)
    by_id = {r["example_id"]: r for r in records}
    experiment = Path(experiment).resolve()
    output = Path(output or experiment).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((experiment / "manifest.json").read_text())
    if str(root / "records.jsonl") != manifest["dataset"]:
        raise ValueError("Experiment dataset path differs from evaluation dataset")
    import hashlib

    if hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Dataset changed since inference; evaluation must use the original references")
    predictions = [
        json.loads(line) for line in (experiment / "predictions.jsonl").read_text().splitlines() if line
    ]
    annotation_map = load_annotations(annotations)
    simulator_map = load_simulator_references(simulator_references)
    marker_samples = {}
    if "markers" in manifest["config"]["conditions"]:
        from .markers import load_marker_bundle

        marker_path = Path(manifest["config"]["marker_bundle"])
        if hashlib.sha256(marker_path.read_bytes()).hexdigest() != manifest["config"]["marker_bundle_sha256"]:
            raise ValueError("Marker bundle changed since inference")
        selected_ids = {row["example_id"] for row in predictions if row["condition"] == "markers"}
        marker_samples = load_marker_bundle(
            marker_path, dataset, [r for r in records if r["example_id"] in selected_ids]
        )
    seen = set()
    rows = []
    for row in predictions:
        identity = row["example_id"], row["condition"]
        if identity in seen or row["example_id"] not in by_id:
            raise ValueError("Duplicate prediction or unknown example")
        seen.add(identity)
        if row.get("image_sha256") is not None:
            image_path = safe_path(root, by_id[row["example_id"]]["model_inputs"]["image"])
            expected_source = row.get("source_image_sha256") or row["image_sha256"]
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != expected_source:
                raise ValueError("Observation image changed since inference")
            if row["condition"] != "markers" and row["image_sha256"] != expected_source:
                raise ValueError("Model-visible observation image hash mismatch")
            if (
                row["condition"] == "markers"
                and marker_samples[row["example_id"]]["image_sha256"] != row["image_sha256"]
            ):
                raise ValueError("Model-visible marker image changed since inference")
        rows.append(
            evaluate_example(
                by_id[row["example_id"]],
                row,
                root,
                annotation_map.get(row["example_id"]),
                simulator_map.get(row["example_id"]),
            )
        )
    summary = aggregate(rows)
    (output / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "evaluation_version": EVALUATION_VERSION,
                "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "dataset_sha256": manifest["dataset_sha256"],
                "simulator_references_sha256": hashlib.sha256(
                    json.dumps(
                        {k: v.model_dump(mode="json") for k, v in simulator_map.items()}, sort_keys=True
                    ).encode()
                ).hexdigest(),
                "expert_annotations_sha256": hashlib.sha256(
                    json.dumps(
                        {k: v.model_dump(mode="json") for k, v in annotation_map.items()}, sort_keys=True
                    ).encode()
                ).hexdigest(),
                "experiment": str(experiment),
            },
            indent=2,
        )
    )
    task_summary = aggregate(rows, per_task=True)
    (output / "per_task_summary.json").write_text(json.dumps(task_summary, indent=2))
    (output / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (output / "summary.json").write_text(
        json.dumps(
            {
                "offline_mock": manifest["offline_mock"],
                "groups": summary,
                "planned_predictions": manifest["estimate"]["planned_predictions"],
                "evaluated_predictions": len(rows),
                "ledger": manifest.get("ledger"),
            },
            indent=2,
        )
    )
    fields = [k for k in summary[0] if k != "reliability"] if summary else ["model", "suite", "condition"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    with (output / "per_example.csv").open("w", newline="") as stream:
        fields = [
            "example_id",
            "suite",
            "model",
            "condition",
            "status",
            "goal_exact_match",
            "predicate_f1",
            "cost_usd",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: row.get(k) for k in fields if k not in {"goal_exact_match", "predicate_f1"}}
                | {
                    "goal_exact_match": row["symbolic"]["exact_match"] if row["symbolic"] else None,
                    "predicate_f1": row["symbolic"]["f1"] if row["symbolic"] else None,
                }
            )
    return rows, summary
