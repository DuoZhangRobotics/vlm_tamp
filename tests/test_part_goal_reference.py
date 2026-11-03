"""Whole-part fits must resist single-point drift without substituting stale poses."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from refine_part_goal_references import observed_reference_orbit, orbit_distance

from vlm_tamp.part_goal_reference import (
    carry_reference,
    part_reference_gap,
)


def test_occluded_reference_is_carried_by_other_part_landmarks() -> None:
    rng = np.random.default_rng(5)
    source = rng.uniform([-0.15, -0.1, -0.1], [0.15, 0.1, 0.1], (64, 3))
    rotation = Rotation.from_rotvec([0.12, -0.08, 0.25]).as_matrix()
    translation = np.array([0.02, 0.01, -0.03])
    target = source @ rotation.T + translation
    target[:15] = rng.uniform(-0.5, 0.5, (15, 3))
    target[15:20] = np.nan
    anchor = np.array([0.17, 0.02, 0.04])  # Not one of the fitted landmarks.
    result = carry_reference(source, target, anchor)
    assert result["fit"]["accepted"]
    np.testing.assert_allclose(
        result["estimated_reference_world_m"], rotation @ anchor + translation, atol=1e-9
    )
    assert not result["is_direct_reference_depth_measurement"]
    assert all(i >= 20 for i in result["fit"]["inlier_indices"])


def test_no_supported_part_motion_does_not_reuse_the_reference_pose() -> None:
    source = np.zeros((7, 3))
    result = carry_reference(source, source, [0.1, 0.2, 0.3])
    assert result["estimated_reference_world_m"] is None
    assert result["fit"]["reason"] == "insufficient_visible_matches"
    assert part_reference_gap(result, result)["estimated_gap_m"] is None


def test_gap_uses_both_estimated_part_positions_without_a_preferred_direction() -> None:
    moving = {"estimated_reference_world_m": [0, 0, 0], "fit": {"accepted": True}}
    fixed = {"estimated_reference_world_m": [0.3, 0.4, 0], "fit": {"accepted": True}}
    assert part_reference_gap(moving, fixed)["estimated_gap_m"] == pytest.approx(0.5)


def test_depth_lifting_accepts_geometry_without_an_unused_rgb_payload() -> None:
    from vlm_tamp.part_motion import lift_tracks

    view = {
        "world_points": np.ones((4, 5, 3)),
        "depth": np.ones((4, 5)),
        "known_robot_depth_prior": np.full((4, 5), np.nan),
    }
    points, valid = lift_tracks([view], np.array([[[2.0, 3.0]]]), np.array([[True]]))
    assert valid.tolist() == [[True]]
    np.testing.assert_array_equal(points[0, 0], [1, 1, 1])


def test_textureless_part_queries_cover_the_mask_instead_of_its_top_rows() -> None:
    from vlm_tamp.part_motion import query_points

    view = {
        "rgb": np.zeros((200, 300, 3), np.uint8),
        "world_points": np.ones((200, 300, 3)),
        "depth": np.ones((200, 300)),
        "known_robot_depth_prior": np.full((200, 300), np.nan),
    }
    mask = np.zeros((200, 300), bool)
    mask[20:180, 30:270] = True
    points = query_points(view, mask, maximum=64)
    assert len(points) == 64
    for left, top in [(False, False), (False, True), (True, False), (True, True)]:
        assert np.sum(((points[:, 0] < 150) == left) & ((points[:, 1] < 100) == top)) >= 10
    assert np.all(mask[points[:, 1].astype(int), points[:, 0].astype(int)])


def test_observed_orbit_uses_arbitrary_measured_axis_and_never_selects_a_goal() -> None:
    axis = np.array([1.0, 2.0, 0.0])
    axis /= np.linalg.norm(axis)
    center = np.array([0.1, 0.3, 0.5])
    anchor = np.array([0.3, 0.4, 0.7])
    poses = []
    for angle in [-0.2, 0.1, 0.3]:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        pose[:3, 3] = center - pose[:3, :3] @ center
        poses.append(pose)
    orbit = observed_reference_orbit(poses, anchor)
    target = Rotation.from_rotvec(axis * 1.4).apply(anchor - center) + center
    assert orbit_distance(orbit, target)["minimum_distance_to_orbit_m"] < 1e-9
    assert orbit_distance(orbit, target + 0.04 * axis)[
        "minimum_distance_to_orbit_m"
    ] == pytest.approx(0.04)
    assert not orbit["goal_angle_selected"]
    assert not observed_reference_orbit([np.eye(4)], anchor)["supported"]
