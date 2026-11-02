"""A targeted push must follow measured target motion before its first stroke."""

from typing import Any

import numpy as np
import pytest

import vlm_tamp.contact_skill_service as module
from vlm_tamp.contact_skills import SkillStop
from test_retained_push_contact import contact_service as contact_fixture, call

contact_service = contact_fixture


def fresh_approaches(
    service: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    shifts: list[list[float]],
    blocked: bool = False,
    search_error: str | None = None,
) -> tuple[list[np.ndarray], list[str], list[list[float]]]:
    """Move the fixture target after approaches and record where strokes begin."""
    original_plan = module.plan_engagement
    original_issue = service.issue
    goals = []
    labels = []
    stroke_starts = []

    def plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return a direct test path to the requested material point."""
        point = np.asarray(args[4]["point_world_m"], float)
        goals.append(point.copy())
        if len(goals) > 1 and search_error:
            raise SkillStop(search_error, search_complete=False)
        result = original_plan(*args, **kwargs)
        q = np.asarray(service.state["metadata"]["joint_positions"])
        target = q.copy()
        target[:3] = point
        result.update(
            reuse_existing_contact=False,
            path=np.vstack([q, target]),
            references=target[None],
            free_references=target[None],
            contact_world=point,
        )
        return result

    def issue(refs: np.ndarray, gripper: int, label: str, **kwargs: Any) -> dict[str, Any]:
        """Separate simulated acquisition motion from stroke motion."""
        labels.append(label)
        approach = label == "free-approach" or label.startswith("contact-reacquisition-")
        service.object_fraction = 0.0 if approach else 1.0
        if not approach:
            stroke_starts.append(service.state["metadata"]["joint_positions"][:3])
        result = original_issue(refs, gripper, label, **kwargs)
        if approach:
            index = sum(x == "free-approach" or x.startswith("contact-reacquisition-") for x in labels) - 1
            if index < len(shifts):
                service.object_position += np.asarray(shifts[index])
            if blocked and index > 0:
                result["execution"]["arrival"]["goal_reached"] = False
        return result

    monkeypatch.setattr(module, "plan_engagement", plan)
    monkeypatch.setattr(service, "issue", issue)
    return goals, labels, stroke_starts


def test_reacquires_moved_material_point_before_starting_requested_stroke(
    contact_service: tuple[Any, dict[str, int], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An observed displacement changes acquisition, not the requested stroke length."""
    service, counts, raw = contact_service
    goals, labels, starts = fresh_approaches(service, monkeypatch, shifts=[[0.1, 0, 0]])
    result = call(service, distance_m=0.01)
    assert labels == ["free-approach", "contact-reacquisition-1", "push-stroke-1"]
    np.testing.assert_allclose(goals, [[0, 0, 0], [0.1, 0, 0]])
    np.testing.assert_allclose(starts, [[0.1, 0, 0]])
    assert result["stages"][-1]["material_point_offset_m"] == pytest.approx(0.0)
    assert result["observed_progress"] == pytest.approx(0.01)  # excludes acquisition motion
    assert result["measured_contact_progress_m"] == pytest.approx(0.01)
    assert result["contact_handoff"]["status"] == "current_target_reached"
    assert counts["builds"] == counts["trackers"] == 1 and counts["searches"] == 2
    np.testing.assert_array_equal(service.active_push_contact["scene"].origin, raw.origin)
    assert service.active_push_contact["scene"].static_points is raw.static_points


def test_tracks_motion_during_reacquisition_without_new_contact_guesses(
    contact_service: tuple[Any, dict[str, int], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Further motion updates the same contact without rebuilding its scene."""
    service, counts, _ = contact_service
    goals, labels, starts = fresh_approaches(service, monkeypatch, shifts=[[0.1, 0, 0], [0, 0.02, 0]])
    result = call(service)
    np.testing.assert_allclose(goals, [[0, 0, 0], [0.1, 0, 0], [0.1, 0.02, 0]])
    np.testing.assert_allclose(starts, [[0.1, 0.02, 0]])
    assert result["contact_handoff"]["reacquisitions"] == 2
    assert counts["builds"] == 1 and labels[-1] == "push-stroke-1"


def test_failed_reacquisition_does_not_push_from_the_obsolete_contact(
    contact_service: tuple[Any, dict[str, int], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreached replacement approach cannot issue the following stroke."""
    service, counts, _ = contact_service
    _, labels, starts = fresh_approaches(service, monkeypatch, shifts=[[0.1, 0, 0]], blocked=True)
    result = call(service)
    assert result["stop_reason"] == "approach_not_reached"
    assert labels == ["free-approach", "contact-reacquisition-1"] and not starts
    assert counts["strokes"] == 0 and not result["continue_push_available"]


def test_reacquisition_search_uses_existing_budget_and_reports_incomplete_search(
    contact_service: tuple[Any, dict[str, int], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Search expiration remains distinct from physical infeasibility."""
    service, counts, _ = contact_service
    _, labels, _ = fresh_approaches(
        service, monkeypatch, shifts=[[0.1, 0, 0]], search_error="budget_exhausted"
    )
    result = call(service, stroke_plan_seconds=5.0)
    assert labels == ["free-approach"] and counts["strokes"] == 0
    assert result["stop_reason"] == "contact_reacquisition_budget_exhausted"
    assert result["diagnostic"]["search_complete"] is False
    assert result["diagnostic"]["physical_infeasibility_proved"] is False
    assert result["planning_stages"][-1]["requested_seconds"] == 5.0
    assert result["contact_handoff"]["status"] == "acquisition_stopped"


def test_reacquisition_cannot_extend_episode_control_budget(
    contact_service: tuple[Any, dict[str, int], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated measured target changes still share the episode control cap."""
    service, counts, _ = contact_service
    service.state["maximum_control_steps"] = 62
    _, labels, _ = fresh_approaches(service, monkeypatch, shifts=[[0.1, 0, 0], [0, 0.02, 0]])
    result = call(service)
    assert labels == ["free-approach", "contact-reacquisition-1"] and counts["strokes"] == 0
    assert result["stop_reason"] == "control_budget_exhausted"
    assert service.state["control_steps"] == 62
    assert not result["continue_push_available"]


def test_unchanged_target_does_not_trigger_ik_residual_polishing(
    contact_service: tuple[Any, dict[str, int], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retarget only measured object changes, not compounded solver/controller error."""
    service, counts, _ = contact_service
    _, labels, _ = fresh_approaches(service, monkeypatch, shifts=[])
    original_plan = module.plan_engagement
    original_issue = service.issue

    def solver_reference(*args: Any, **kwargs: Any) -> dict:
        """Represent a native IK endpoint with three millimetres of residual."""
        result = original_plan(*args, **kwargs)
        result["free_references"][-1, 0] = .003
        result["path"][-1, 0] = .003
        return result

    def arrived_reference(refs: np.ndarray, gripper: int, label: str, **kwargs: Any) -> dict:
        """Represent a reached native reference with residual to the original target."""
        result = original_issue(refs, gripper, label, **kwargs)
        if label == "free-approach":
            service.state["metadata"]["joint_positions"][0] = 0.006
            service.state["metadata"]["eef_pose_world_xyzw"][0] = 0.006
        return result

    monkeypatch.setattr(service, "issue", arrived_reference)
    monkeypatch.setattr(module, "plan_engagement", solver_reference)
    result = call(service)
    assert labels == ["free-approach", "push-stroke-1"]
    assert counts["searches"] == 1
    assert result["contact_handoff"]["status"] == "reference_reached_no_retarget_needed"
    assert result["contact_handoff"]["approaches"][0]["current_target_error_m"] == pytest.approx(0.006)


@pytest.mark.parametrize("tracking_known", [True, False])
def test_no_reacquisition_for_stationary_or_unresolved_target(
    contact_service: tuple[Any, dict[str, int], Any], monkeypatch: pytest.MonkeyPatch, tracking_known: bool
) -> None:
    """No extra motion is inferred from absent or stationary tracking evidence."""
    service, counts, _ = contact_service
    service.fail_tracking = not tracking_known
    _, labels, _ = fresh_approaches(service, monkeypatch, shifts=[])
    result = call(service)
    assert labels == ["free-approach", "push-stroke-1"] and counts["searches"] == 1
    assert result["contact_handoff"]["status"] == (
        "current_target_reached" if tracking_known else "object_pose_unresolved"
    )
    assert not result["contact_handoff"]["physical_contact_verified"]
    if not tracking_known:
        assert result["contact_handoff"]["approaches"][0]["current_target_error_m"] is None
