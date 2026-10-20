"""Held-back candidate coverage and selection checks; never an inference input.

Whole-object masks establish instance membership only. Candidate coverage here
does not establish handle, contact, support, axis, or grasp affordance coverage.
No candidate is moved, added, filtered, or assigned an identity for inference.
"""

import numpy as np
from PIL import Image

from .dataset import safe_path
from .rgb_role_diagnostics import template_contract
from .rgbd_landmark_choices import ID_FIELDS, resolve_blanks, validate_candidates
from .rgbd_numeric_evaluation import grade_case
from .rgbd_numericals import POINT_FIELDS


def _reference(record, root):
    contract = template_contract(record)
    if not contract["supported"] or len(contract["role_owners"]) != 2:
        raise ValueError("Candidate evaluation requires two independently aligned goal roles")
    source, target = contract["role_owners"].values()
    masks = {}
    for owner, visual in record["reference"]["visual"].items():
        if visual.get("kind", "object") == "object":
            with Image.open(safe_path(root, visual["mask"])) as image:
                masks[owner] = np.asarray(image) > 0
    if not masks:
        raise ValueError("Whole-object reference masks are required for offline coverage")
    shape = next(iter(masks.values())).shape
    if len(shape) != 2 or any(mask.shape != shape for mask in masks.values()):
        raise ValueError("Reference masks must share one two-dimensional image grid")
    return source, target, masks, shape


def proposal_coverage(record, root, candidates):
    """Grade immutable candidate pixels against all held-back whole-object masks.

    ``available_candidate_ids`` requires unique expected-instance membership.
    ``expected_object_hit_ids`` also includes shared/ambiguous pixels, making
    missing-proposal diagnostics distinct from strict usable-role coverage.
    """
    source, target, masks, shape = _reference(record, root)
    validate_candidates(candidates, shape)
    rows = []
    for candidate in candidates:
        x, y = candidate["pixel_xy"]
        matches = [owner for owner, mask in masks.items() if mask[y, x]]
        diagnostic = ("background" if not matches else "shared_candidate" if len(matches) > 1
                      else "expected_instance" if matches[0] in {source, target} else "wrong_instance")
        rows.append({"id": candidate["id"], "pixel_xy": list(candidate["pixel_xy"]),
                     "mask_matches": matches, "diagnostic": diagnostic})
    roles = {}
    for role, owner in (("source", source), ("target", target)):
        available = [row["id"] for row in rows if row["mask_matches"] == [owner]]
        hits = [row["id"] for row in rows if owner in row["mask_matches"]]
        roles[role] = {"expected_owner_offline_only": owner,
                       "available_candidate_ids": available, "expected_object_hit_ids": hits,
                       "shared_candidate_ids": [identifier for identifier in hits if identifier not in available],
                       "whole_object_covered": bool(available), "expected_object_hit": bool(hits)}
    fields = []
    for field in POINT_FIELDS:
        role = "target" if field == "target_uv" else "source"
        fields.append({"field": field, "role": role, **roles[role]})
    return {
        "scope": "offline whole-object instance coverage only; no part/contact/axis affordance labels",
        "roles": roles, "fields": fields, "candidate_diagnostics": rows,
        "candidate_count": len(rows),
        "required_role_count": 2,
        "required_roles_covered": sum(role["whole_object_covered"] for role in roles.values()),
        "complete_required_role_coverage": all(role["whole_object_covered"] for role in roles.values()),
        "required_field_count": len(POINT_FIELDS),
        "required_fields_covered": sum(field["whole_object_covered"] for field in fields),
        "shared_candidate_ids": [row["id"] for row in rows if row["diagnostic"] == "shared_candidate"],
        "background_candidate_ids": [row["id"] for row in rows if row["diagnostic"] == "background"],
        "wrong_instance_candidate_ids": [row["id"] for row in rows if row["diagnostic"] == "wrong_instance"],
        "part_coverage_verified": False, "contact_coverage_verified": False,
        "axis_coverage_verified": False, "grasp_coverage_verified": False,
        "inference_identity_rescue": False,
    }


def grade_choices(record, root, case, choices, candidates, numericals):
    """Resolve exact IDs and delegate accepted forms to the unchanged grader.

    Missing or invalid forms remain failed cases. For null/absent selections,
    ``proposal_missing`` means no pixel hits the expected whole-object mask;
    otherwise it is ``selection_abstention``. Non-null wrong or invalid
    selections are ``selection_error``. These are offline diagnostics and do
    not establish causal error attribution or semantic part correctness.
    """
    coverage = proposal_coverage(record, root, candidates)
    _, _, _, shape = _reference(record, root)
    resolved, error = None, None
    try:
        resolved = resolve_blanks(choices, candidates, shape)
    except ValueError as exc:
        error = str(exc)
    grade = grade_case(record, root, case, {
        "blanks": resolved, "numericals": numericals if resolved is not None else None,
    })
    by_field = {row["field"]: row for row in coverage["fields"]}
    diagnostics = []
    for id_field, anchor in zip(ID_FIELDS, grade["anchors"]):
        selected = choices.get(id_field) if isinstance(choices, dict) else None
        available = by_field[anchor["field"]]
        if selected is None:
            diagnostic = "selection_abstention" if available["expected_object_hit"] else "proposal_missing"
        elif resolved is None or anchor["diagnostic"] != "correct_instance":
            diagnostic = "selection_error"
        else:
            diagnostic = "correct_selection"
        diagnostics.append({"field": anchor["field"], "id_field": id_field,
                            "selected_id": selected, "diagnostic": diagnostic,
                            "expected_owner_offline_only": anchor["expected_owner_offline_only"],
                            "available_candidate_ids": list(available["available_candidate_ids"]),
                            "expected_object_hit_ids": list(available["expected_object_hit_ids"]),
                            "selected_point_diagnostic": anchor["diagnostic"],
                            "part_contact_axis_semantics_verified": False})
    return {**grade, "choices_valid": resolved is not None, "choice_validation_error": error,
            "resolved_blanks": resolved, "proposal_coverage": coverage,
            "selection_diagnostics": diagnostics, "case_in_denominator": True}
