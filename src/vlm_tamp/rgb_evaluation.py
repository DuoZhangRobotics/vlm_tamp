"""Evaluate RGB referents and natural-language task meaning, without hidden-ID requirements.

References stay offline. Region ownership/roles are obtained from the scene's whole
region inventory, not from a predicted answer. Exact legacy BDDL scores are untouched.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .bddl import normalize_predicate
from .dataset import read_records, safe_path
from .evaluation import bbox_iou, score_sets
from .rgb_grounding import RGBPrediction


VERSION = "rgb-semantic-1.0"


def scene_roles(record):
    """Decode every declared scene region through a generic naming/ownership contract."""
    entities = {e["identifier"]: e for e in record["reference"]["entities"]}
    roles = {identifier: (identifier, "whole") for identifier, e in entities.items() if e["kind"] == "object"}
    for identifier, entity in entities.items():
        if entity["kind"] == "object":
            continue
        parent = entity["parent"]
        suffix = identifier[len(parent) + 1 :] if identifier.startswith(parent + "_") else identifier
        level = next(
            (
                x
                for x in ("top", "middle", "bottom")
                if suffix == x + "_region" and "cabinet" in entities[parent]["category"]
            ),
            None,
        )
        if level:
            roles[identifier] = (parent, level + "_drawer")
        elif any(word in suffix for word in ("contain", "heating", "interior")):
            roles[identifier] = (parent, "interior")
        elif suffix == "stove_front_region":
            anchors = [
                key for key, e in entities.items() if e["kind"] == "object" and "stove" in e["category"]
            ]
            if len(anchors) == 1:
                roles[identifier] = (anchors[0], "front_area")
        elif any(word in suffix for word in ("cooking", "support", "top_side")):
            roles[identifier] = (parent, "support_surface")
    return roles


def normalize_goal(relation, arguments):
    """Task-independent physical relation/owner normalization, used on both sides."""
    relation = {
        "on": "On",
        "in": "In",
        "open": "Open",
        "close": "Close",
        "turnon": "Turnon",
        "turnoff": "Turnoff",
        "infrontof": "InFrontOf",
        "at": "At",
        "stack": "Stack",
    }.get(normalize_predicate(relation).lower(), relation)
    args = list(arguments)
    if len(args) == 2 and relation == "On" and args[1][1] == "front_area":
        relation = "InFrontOf"
        args[1] = (args[1][0], "whole")
    elif len(args) == 2 and relation == "InFrontOf":
        args[1] = (args[1][0], "whole")
    elif len(args) == 2 and relation == "In" and args[1][1] == "whole":
        args[1] = (args[1][0], "interior")
    elif len(args) == 2 and relation in {"On", "Stack"} and args[1][1] == "support_surface":
        args[1] = (args[1][0], "whole")
    if len(args) == 1 and relation in {"Open", "Close"} and args[0][1] == "door":
        args[0] = (args[0][0], "whole")
    return relation, tuple(args)


def make_reference(record):
    roles = scene_roles(record)
    goals, notes = [], []
    instruction = record["model_inputs"]["instruction"].lower()
    entities = {e["identifier"]: e for e in record["reference"]["entities"]}
    for goal in record["reference"]["goal_predicates"]:
        if not all(a in roles for a in goal["arguments"]):
            return {"evaluable": False, "reason": "Unsupported scene-region semantic contract", "goals": []}
        relation = goal["name"]
        args = [roles[a] for a in goal["arguments"]]
        # The published natural-language request establishes containment intent.
        # BDDL's bowl-support predicate alone does not distinguish inside from on.
        if (
            normalize_predicate(relation) == "on"
            and len(args) == 2
            and "bowl" in entities[args[1][0]]["category"]
            and re.search(r"\b(?:in|inside)\s+(?:the\s+)?bowl\b", instruction)
        ):
            if len(record["reference"]["goal_predicates"]) != 1:
                return {
                    "evaluable": False,
                    "reason": "Multiple clauses need independently reviewed containment intent",
                    "goals": [],
                }
            relation = "In"
            notes.append(
                "Containment relation is sourced from the published instruction; legacy BDDL On is retained separately."
            )
        goals.append(normalize_goal(relation, args))
    return {
        "evaluable": not record["reference"]["unsupported_goal_expressions"]
        if "unsupported_goal_expressions" in record["reference"]
        else True,
        "goals": goals,
        "notes": notes,
        "source": "official instruction and BDDL instance labels, normalized through frozen task-independent scene-role contract",
        "human_reviewed_constraints": False,
    }


def freeze_references(dataset, output):
    records, root = read_records(dataset)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Semantic references already frozen")
    references = {r["example_id"]: make_reference(r) for r in records}
    if not all(r["evaluable"] for r in references.values()):
        raise ValueError(
            "The selected RGB tasks need complete independently specified semantic reference coverage"
        )
    output.write_text(
        json.dumps(
            {
                "version": VERSION,
                "dataset_sha256": hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest(),
                "reference_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "goal_encoding_changed_from_raw_bddl": True,
                "references": references,
            },
            indent=2,
        )
        + "\n"
    )
    return references


def load_references(dataset, path):
    _, root = read_records(dataset)
    saved = json.loads(Path(path).read_text())
    if saved["dataset_sha256"] != hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest():
        raise ValueError("Reference dataset changed")
    if saved["reference_builder_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Semantic evaluator changed since references were frozen")
    return saved["references"]


def resolve_entities(prediction: RGBPrediction, record, root):
    """Match image evidence against ALL object masks; never use reference task goals."""
    visible = {
        key: ref
        for key, ref in record["reference"]["visual"].items()
        if ref.get("kind", "object") == "object"
    }
    resolved, diagnostics = {}, []
    for entity in prediction.entities:
        if entity.kind != "object":
            continue
        point_matches, box_matches = [], []
        for identifier, reference in visible.items():
            if entity.point is not None:
                mask = np.asarray(Image.open(safe_path(root, reference["mask"]))) > 0
                x = min(mask.shape[1] - 1, int(entity.point[0] * mask.shape[1]))
                y = min(mask.shape[0] - 1, int(entity.point[1] * mask.shape[0]))
                if mask[y, x]:
                    point_matches.append(identifier)
            if entity.bbox is not None and bbox_iou(entity.bbox, reference["bbox"]) >= 0.5:
                box_matches.append(identifier)
        point_id = point_matches[0] if len(point_matches) == 1 else None
        box_id = box_matches[0] if len(box_matches) == 1 else None
        conflict = point_id and box_id and point_id != box_id
        chosen = None if conflict else point_id or box_id
        resolved[entity.id] = (chosen, "whole") if chosen else ("unresolved:" + entity.id, "whole")
        diagnostics.append(
            {
                "entity": entity.id,
                "point_matches": point_matches,
                "box_matches": box_matches,
                "conflict": bool(conflict),
                "resolved": chosen,
            }
        )
    by_id = {e.id: e for e in prediction.entities}

    def part(entity):
        if entity.id in resolved:
            return resolved[entity.id]
        owner = by_id[entity.parent]
        owner_ref = part(owner)
        resolved[entity.id] = (owner_ref[0], entity.selector)
        return resolved[entity.id]

    for entity in prediction.entities:
        part(entity)
    return resolved, diagnostics


def evaluate_rgb(record, prediction, reference, root):
    if prediction is None:
        return {
            "semantic_goal": score_sets([], [tuple_goal(g) for g in reference["goals"]]),
            "identity_details": [],
            "predicate_types": score_sets([], [g[0] for g in reference["goals"]]),
            "part_pixel_accuracy": None,
            "constraint_accuracy": None,
        }
    resolved, diagnostics = resolve_entities(prediction, record, root)
    predicted = [normalize_goal(g.relation, [resolved[a] for a in g.arguments]) for g in prediction.goals]
    expected = [tuple_goal(g) for g in reference["goals"]]
    score = score_sets(predicted, expected)
    return {
        "semantic_goal": score,
        "predicted_goal_signatures": predicted,
        "reference_goal_signatures": expected,
        "identity_details": diagnostics,
        "predicate_types": score_sets([g[0] for g in predicted], [g[0] for g in expected]),
        "part_selector_scope": "Owner image identity and semantic part selector; part-level image localization remains separately unverified.",
        "part_pixel_accuracy": None,
        "constraint_accuracy": None,
        "robot_execution_tested": False,
    }


def tuple_goal(goal):
    return goal[0], tuple(tuple(argument) for argument in goal[1])
