"""Surface sides, native solver criteria and execution use the same contact model."""

from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vlm_tamp.contact_orientation import (
    checked_surface_normal,
    contact_orientation_report,
    frame_with_z,
    joint_push_prediction,
)
from vlm_tamp.contact_orientation import selection_geometry
from vlm_tamp.timed_execution import pose_arrival
from vlm_tamp.robot_contracts import validate_operation
from test_curobo_arc_worker import worker_scene as worker_fixture
from test_retained_push_contact import contact_service as service_fixture, call


@pytest.mark.parametrize('normal',[[1.,0,0],[-1.,0,0],[0,0,1.],[0,1.,0]])
def test_approach_uses_the_minimum_hand_plane_translation(normal: list[float]) -> None:
    """Every sphere is outside, and any shorter translation intersects the plane."""
    from vlm_tamp.contact_orientation import approach_geometry
    hand=np.array([[0,0,-.08,.02],[0,0,-.01,.012]])
    point=np.array([.3,-.5,1.1])
    rotation=frame_with_z(-np.asarray(normal))
    result=approach_geometry(point,normal,hand,[0,0,0],rotation)
    assert result['approach_distance_m']==pytest.approx(.002)
    np.testing.assert_allclose(result['precontact_point_world_m'],point+.002*np.asarray(normal))
    gaps=hand[:,:3]@rotation.T@np.asarray(normal)-hand[:,3]+result['approach_distance_m']
    assert min(gaps)==pytest.approx(0.,abs=1e-12)
    assert min(gaps-.00001)<0
    np.testing.assert_allclose(result['approach_direction_world'],-np.asarray(normal))
    assert result['collision_padding_m']==0
    assert not result['task_direction_verified']


def test_approach_adds_no_standoff_when_the_hand_already_clears_the_plane() -> None:
    """There is no minimum travel, tolerance reserve or full-hand-length buffer."""
    from vlm_tamp.contact_orientation import approach_geometry
    result=approach_geometry([0,0,0],[0,0,1],[[0,0,.02,.01]],[0,0,0],np.eye(3))
    assert result['approach_distance_m']==0.

worker_scene = worker_fixture
contact_service = service_fixture


@pytest.mark.parametrize("normal", [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0.2, -0.3, 0.8]])
def test_contact_frame_preserves_the_selected_signed_side(normal: list[float]) -> None:
    """Both faces remain distinct; the coordinate basis is orthonormal."""
    frame = frame_with_z(normal)
    np.testing.assert_allclose(frame[:, 2], np.asarray(normal) / np.linalg.norm(normal), atol=1e-12)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0)
    np.testing.assert_allclose(frame_with_z(-np.asarray(normal))[:, 2], -frame[:, 2])


def test_normal_model_does_not_choose_opening_closing_or_motion_direction() -> None:
    """Opposite contact forces predict opposite joint signs, not semantic goals."""
    a = joint_push_prediction([1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0])
    b = joint_push_prediction([1, 0, 0], [0, 0, 0], [0, 0, 1], [0, -1, 0])
    assert (a["signed_rotation_tendency"], b["signed_rotation_tendency"]) == (1, -1)
    assert a["open_or_close"] == b["open_or_close"]
    report = contact_orientation_report(
        np.eye(3),
        [0, 0, 1],
        [0, 0, -1],
        [0, 0, -1],
        current_contact_point=[0, 0, 0.035],
        target_point=[0, 0, 0],
    )
    assert report["normal_error_degrees"] == pytest.approx(0.0)
    assert report["requested_motion_dot_outward_normal"] == pytest.approx(1.0)
    assert report["current_contact_signed_plane_offset_m"] == pytest.approx(-0.035)
    assert report["task_direction_verified"] is False


def test_model_requires_an_explicit_normal_instead_of_inventing_one() -> None:
    """Missing or contradictory arguments are reported before numerical work."""
    assert checked_surface_normal("point", None) is None
    with pytest.raises(ValueError, match="signed outward"):
        checked_surface_normal("surface_normal", None)
    with pytest.raises(ValueError, match="Select contact_constraint"):
        checked_surface_normal("point", [0, 0, 1])
    with pytest.raises(ValueError, match="finite nonzero"):
        checked_surface_normal("surface_normal", [0, 0, 0])


def test_view_plane_sides_do_not_depend_on_an_arbitrary_normal_flip() -> None:
    """Candidate views describe geometry, rather than guessing an opening side."""
    front = np.eye(4)
    front[2, 3] = 1.0
    back = np.eye(4)
    back[2, 3] = -1.0
    cameras = {"front": {"camera_to_world": front}, "back": {"camera_to_world": back}}
    for normal in ([0, 0, 1], [0, 0, -1]):
        report = selection_geometry([0, 0, 0], normal, [0, 0, -0.035], [0, 0, 1], cameras)
        assert report["same_plane_side_camera_candidates"] == ["back"]


def test_compact_search_diagnostic_keeps_native_blockers() -> None:
    """A rejected palm endpoint is not reduced to an unhelpful unknown count."""
    from vlm_tamp.compact_robot_interface import contact_search_summary

    diagnostic = {
        "attempts": [
            {
                "joints": [0.0] * 7,
                "planning": {
                    "status": "invalid_endpoint",
                    "endpoints": {
                        "start": {"accepted": True},
                        "goal": {
                            "accepted": False,
                            "blocking_robot_part": "gripper0_hand_collision",
                            "blocking_surface": "observed stationary geometry",
                            "blocking_sphere_world_m": [0.0, 0.0, 1.0, 0.028],
                        },
                    },
                },
            }
        ],
        "search_complete": True,
        "global_infeasibility_proved": False,
    }
    result = contact_search_summary(diagnostic)
    assert result["rejection_counts"] == {"goal: gripper0_hand_collision / observed stationary geometry": 1}
    assert result["blocking_examples"][0]["blocking_sphere_world_m"] == [0.0, 0.0, 1.0, 0.028]
    assert not result["global_infeasibility_proved"]


def test_contact_plan_defaults_to_its_planned_aperture() -> None:
    """Generic execution must not silently close a contact-planned gripper."""
    from types import SimpleNamespace
    from serve_subscription_robot import RobotService

    plan = {
        "kind": "free",
        "control_steps": 60,
        "epoch": 0,
        "path": np.zeros((2, 7)),
        "extra": {
            "required_gripper_command": 0,
            "arrival": {
                "contact_local_m": [0.02, 0, 0],
                "contact_normal_local": [0, 0, 1],
                "maximum_wait_steps": 20,
            },
        },
    }
    sent = []
    service = SimpleNamespace(
        plans={"p": plan},
        state={"control_steps": 60},
        epoch=0,
        issue=lambda refs, gripper, label, **kwargs: sent.append((gripper, kwargs)),
    )
    RobotService.execute_plan(service, "p")
    assert sent[0][0] == 0
    assert sent[0][1]["arrival"]["contact_normal_local"] == [0, 0, 1]
    with pytest.raises(ValueError, match="current aperture"):
        RobotService.execute_plan(service, "p", gripper=1)
    assert len(sent) == 1


def test_normal_arrival_allows_twist_but_rejects_face_tilt() -> None:
    """Execution matches point-plus-normal, not an arbitrary full wrist quaternion."""
    goal = [0, 0, 0, 0, 0, 0, 1]
    for yaw in [30, 90, 180]:
        actual = [0, 0, 0, *Rotation.from_euler("z", yaw, degrees=True).as_quat()]
        assert pose_arrival(goal, actual, [0, 0, 0], [0, 0, 1])["reached"]
    tilted = [0, 0, 0, *Rotation.from_euler("x", 20, degrees=True).as_quat()]
    assert not pose_arrival(goal, tilted, [0, 0, 0], [0, 0, 1])["reached"]
    assert pose_arrival(goal, tilted, [0, 0, 0])["reached"]
    rotation = Rotation.from_euler("z", 90, degrees=True)
    offset = np.array([0.05, 0, 0])
    shifted_tcp = offset - rotation.apply(offset)
    actual = [*shifted_tcp, *rotation.as_quat()]
    assert pose_arrival(goal, actual, offset, [0, 0, 1])["reached"]


def test_continuation_transports_the_same_face_using_measured_part_motion(
    contact_service: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No camera-facing reflip or new contact search replaces the selected face."""
    import vlm_tamp.contact_skill_service as service_module

    service, counts, scene = contact_service
    engage = service_module.plan_engagement
    observe = service_module.ContactTracker.observe
    plan = service_module.plan_contact_stroke
    rotation = Rotation.from_euler("y", 40, degrees=True).as_matrix()
    planned_normals = []

    def contact(*args: Any, **kwargs: Any) -> dict:
        """Supply the configured face for the fixture's existing contact."""
        result = engage(*args, **kwargs)
        result["contact_normal_local"] = np.array([0.0, 0.0, 1.0])
        result["contact_constraint"] = "surface_normal"
        return result

    def tracked(tracker: Any, *args: Any) -> dict:
        """Simulate an independently measured part rotation."""
        result = observe(tracker, *args)
        transform = np.asarray(result["transform"])
        transform[:3, :3] = rotation
        result["transform"] = transform.tolist()
        return result

    def stroke(*args: Any, **kwargs: Any) -> tuple:
        """Record the normal delivered to the existing numerical path service."""
        planned_normals.append(args[1].skill_surface_normal_world.copy())
        return plan(*args, **kwargs)

    monkeypatch.setattr(service_module, "plan_engagement", contact)
    monkeypatch.setattr(service_module.ContactTracker, "observe", tracked)
    monkeypatch.setattr(service_module, "plan_contact_stroke", stroke)
    first = call(service, contact_constraint="surface_normal", surface_normal_world=[0, 0, -1])
    np.testing.assert_allclose(service.active_push_contact["surface_normal_world"], rotation @ [0, 0, -1])
    second = call(service, continued=True)
    np.testing.assert_allclose(planned_normals, [[0, 0, -1], rotation @ [0, 0, -1]], atol=1e-12)
    assert first["contact_constraint"] == second["contact_constraint"] == "surface_normal"
    assert counts["searches"] == 1
    np.testing.assert_array_equal(service.active_push_contact["scene"].axis, scene.axis)


def test_native_criteria_constrain_the_contact_frame_and_keep_both_solvers_warm(worker_scene: Any) -> None:
    """Native partial pose criteria are selected explicitly, without a point fallback."""
    worker, _, meshes, solvers = worker_scene
    args = dict(
        point=[0.5, 0.1, 0.2],
        contact_local=[0.04, 0, 0],
        reference_quaternion_xyzw=[0, 0, 0, 1],
        seed=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    point = worker.solve_contact(**args, local=False)
    normal = worker.solve_contact(
        **args, local=False, contact_normal_local=[1, 0, 0], surface_normal_world=[0, 1, 0]
    )
    assert not meshes and point["contact_constraint"] == "point"
    assert normal["contact_constraint"] == "surface_normal"
    criteria = solvers[1].criteria["contact_point"]
    assert criteria.project_distance_to_goal is True
    assert criteria.terminal_pose_axes_weight_factor == [1, 1, 1, 1, 1, 0]
    assert "orientation_tolerance" not in solvers[1].options
    again = worker.solve_contact(
        **args, local=False, contact_normal_local=[1, 0, 0], surface_normal_world=[0, -1, 0]
    )
    original = worker.solve_contact(**args, local=False)
    assert not again["solver_created"] and not original["solver_created"]
    assert all(s.destroy_count == 0 for s in solvers)
    np.testing.assert_allclose(normal["solutions"][0], args["seed"])


def test_callable_schemas_expose_normal_model_without_inferred_motion() -> None:
    """The actual callable tools retain an explicit motion request."""
    args = {
        "observation_id": 1,
        "camera": "agentview",
        "contact_pixel_xy": [5, 5],
        "region_xyxy": [1, 1, 10, 10],
        "contact_name": "gripper0_finger1_pad_collision_distal_face",
        "contact_constraint": "surface_normal",
        "surface_normal_world": [0, 0, 1],
        "direction_world": [1, 0, 0],
    }
    validate_operation("push", args)
    validate_operation(
        "plan_contact_pose",
        {
            "point_world_m": [0, 0, 1],
            "contact_name": args["contact_name"],
            "contact_constraint": "surface_normal",
            "surface_normal_world": [0, 0, 1],
        },
    )
    del args["direction_world"]
    with pytest.raises(ValueError):
        validate_operation("push", args)
