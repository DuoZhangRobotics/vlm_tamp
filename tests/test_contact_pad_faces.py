"""Expose executable robot contact geometry without changing collision physics."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from vlm_tamp.contact_skills import contact_landmarks
from vlm_tamp.rekep_libero.geometry import pose_matrix, transform_points
from vlm_tamp.rekep_libero.kinematics import PandaIK
from vlm_tamp.surface_contact_tools import contact_pose_candidates


def test_distal_pad_face_uses_calibrated_box_and_preserves_inner_contact() -> None:
    """A rotated robot still exposes the actual pad face and its active mask."""
    pose = [0.2, -0.1, 1.0, *Rotation.from_euler("xyz", [25, 45, 60], degrees=True).as_quat()]
    vertices = np.array(
        [[x, y, z] for x in [0.02, 0.028] for y in [-0.008, 0.008] for z in [-0.0116, 0.0044]]
    )
    metadata = {
        "eef_pose_world_xyzw": pose,
        "gripper_qpos": [0.04, -0.04],
        "robot_collision_volumes_world": [
            {
                "robot_geom_name": "gripper0_finger2_pad_collision",
                "vertices_world": transform_points(vertices, pose_matrix(pose)),
                "physical_collision_masks": {"contype": 1, "conaffinity": 1},
            }
        ],
    }
    contacts = {row["name"]: row for row in contact_landmarks(metadata, "fingertips")}
    distal = contacts["gripper0_finger2_pad_collision_distal_face"]
    np.testing.assert_allclose(distal["point"], [0.024, 0.0, 0.0044], atol=1e-12)
    np.testing.assert_array_equal(distal["normal"], [0, 0, 1])
    np.testing.assert_allclose(
        contacts["gripper0_finger2_pad_collision"]["point"], [0.02, 0.0, -0.0036], atol=1e-12
    )
    assert distal["initiates_simulator_contacts"] is True
    assert len(contact_landmarks(metadata, "fingertips", hook=True)) == 1
    goal = [0.3, 0.2, 1.1]
    candidate = contact_pose_candidates(metadata, goal, pose[3:], distal["name"])[0]
    tcp = pose_matrix(candidate["tcp_pose_world_xyzw"])
    np.testing.assert_allclose(tcp[:3, 3] + tcp[:3, :3] @ distal["point"], goal)
    assert candidate["robot_collision_geom"] == "gripper0_finger2_pad_collision"


def test_passive_body_tip_remains_available_but_is_labelled() -> None:
    """Legacy point names do not falsely imply active physical contact."""
    metadata = {
        "eef_pose_world_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "robot_collision_volumes_world": [
            {
                "robot_geom_name": "gripper0_finger2_collision",
                "vertices_world": [[0.02, 0.0, 0.0], [0.025, 0.0, 0.01], [0.03, 0.0, 0.01]],
                "physical_collision_masks": {"contype": 0, "conaffinity": 1},
            }
        ],
    }
    tip = contact_landmarks(metadata, "fingertips")[0]
    assert tip["name"] == "gripper0_finger2_collision_distal_tip"
    assert tip["initiates_simulator_contacts"] is False
    assert tip["physical_collision_masks"] == {"contype": 0, "conaffinity": 1}


def test_collision_export_reports_masks_without_modifying_them() -> None:
    """Only configured robot volumes contribute masks to the public metadata."""
    solver = PandaIK.__new__(PandaIK)
    solver.model = SimpleNamespace(geom_contype=np.array([0, 1]), geom_conaffinity=np.array([1, 1]),
                                   geom_bodyid=np.array([0,1]))
    solver.data = SimpleNamespace(
        geom_xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)), geom_xpos=np.zeros((2, 3)),
        xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)),xpos=np.array([[.1,0,0],[.2,0,0]])
    )
    solver._collision_volumes_local = [
        {
            "geom_id": index,
            "type": "sphere",
            "radius": 0.01,
            "fixed": False,
            "robot_geom_name": f"robot_{index}",
        }
        for index in range(2)
    ]
    volumes = solver.collision_volumes()
    assert volumes[0]['body_pose_world_xyzw']==[.1,0.,0.,0.,0.,0.,1.]
    assert [row["physical_collision_masks"]["contype"] for row in volumes] == [0, 1]
    np.testing.assert_array_equal(solver.model.geom_contype, [0, 1])
