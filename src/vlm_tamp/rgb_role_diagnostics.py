"""Offline language coverage and pixel diagnostics; never used for VLM inputs."""

from collections import Counter

import numpy as np
from PIL import Image
from scipy import ndimage

from .dataset import safe_path
from .rgb_choices_v2 import STATES
from .rgb_evaluation import make_reference, normalize_goal
from .rgb_role_choices import instruction_template


def goal_structure(goal):
    return goal[0], tuple(argument[1] for argument in goal[1])


def template_contract(record):
    """Audit representable goal structure against an independent task reference.

    This is a coverage audit, not a model score or an instruction parser repair.
    It does not establish arbitrary-language semantic equivalence.
    """
    try:
        template = instruction_template(record["model_inputs"]["instruction"])
    except ValueError:
        return {"supported": False, "reason": "instruction_grammar_rejected", "role_owners": {}}
    reference = make_reference(record)
    if not reference["evaluable"]:
        return {"supported": False, "reason": "reference_contract_unsupported", "role_owners": {}}
    declared = []
    for goal in template["goals"]:
        args = [(goal["object_slot"], goal["object_part"])]
        if goal["target_slot"] is not None:
            args.append((goal["target_slot"], goal["target_part"]))
        declared.append(normalize_goal(STATES[goal["state"]], args))
    if Counter(map(goal_structure, declared)) != Counter(map(goal_structure, reference["goals"])):
        return {"supported": False, "reason": "goal_structure_mismatch", "role_owners": {}}
    expected, used = {}, set()
    for goal in declared:
        matches = [index for index, ref in enumerate(reference["goals"])
                   if index not in used and goal_structure(ref) == goal_structure(goal)]
        if len(matches) != 1:
            return {"supported": True, "reason": "ambiguous_offline_role_alignment", "role_owners": {}}
        index = matches[0]
        used.add(index)
        for argument, ref_argument in zip(goal[1], reference["goals"][index][1]):
            slot, owner = argument[0], ref_argument[0]
            if slot in expected and expected[slot] != owner:
                return {"supported": False, "reason": "conflicting_role_alignment", "role_owners": {}}
            expected[slot] = owner
    if set(expected) != set(template["roles"]):
        return {"supported": False, "reason": "incomplete_role_alignment", "role_owners": {}}
    return {"supported": True, "reason": "supported_structure", "role_owners": expected}


def diagnose_role_points(record, points, role_owners, root):
    """Describe wrong-instance, background and unknown points without rescuing them."""
    visible = {identifier: visual for identifier, visual in record["reference"]["visual"].items()
               if visual.get("kind", "object") == "object"}
    masks = {identifier: np.asarray(Image.open(safe_path(root, visual["mask"]))) > 0
             for identifier, visual in visible.items()}
    height, width = record["metadata"]["camera"]["height"], record["metadata"]["camera"]["width"]
    rows = []
    for slot, owner in role_owners.items():
        point = points.get(slot) if points is not None else None
        row = {"slot": slot, "expected_owner_offline_only": owner, "point": point,
               "target_visible": owner in masks, "mask_matches": [],
               "pixel_distance_to_expected_mask": None, "point_inside_expected_bbox": False}
        if points is None:
            row["diagnostic"] = "invalid_or_missing_response"
        elif point is None:
            row["diagnostic"] = "abstained"
        else:
            x, y = min(width - 1, int(point[0] * width)), min(height - 1, int(point[1] * height))
            row["mask_matches"] = [identifier for identifier, mask in masks.items() if mask[y, x]]
            if owner in masks:
                distance = ndimage.distance_transform_edt(~masks[owner])
                row["pixel_distance_to_expected_mask"] = float(distance[y, x])
                box = visible[owner]["bbox"]
                row["point_inside_expected_bbox"] = box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]
            if len(row["mask_matches"]) == 1:
                row["diagnostic"] = "correct_instance" if row["mask_matches"][0] == owner else "wrong_instance"
            elif row["mask_matches"]:
                row["diagnostic"] = "ambiguous_point"
            else:
                row["diagnostic"] = "background_or_occluder"
        rows.append(row)
    return rows
