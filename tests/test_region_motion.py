"""Two-camera fusion preserves evidence provenance and does not assign task semantics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vlm_tamp.region_motion import fit_camera_groups, region_proximity, transformed_region


def test_two_views_jointly_resolve_a_pose_that_neither_view_can_fit_alone() -> None:
    rng = np.random.default_rng(7)
    source = {name: rng.uniform(-0.15, 0.15, (5, 3)) for name in ["agentview", "wrist"]}
    rotation = Rotation.from_rotvec([0.1, 0.2, -0.3]).as_matrix()
    translation = np.array([0.01, -0.02, 0.03])
    target = {name: points @ rotation.T + translation for name, points in source.items()}
    for name in source:
        assert not fit_camera_groups({name: source[name]}, {name: target[name]})["accepted"]
    result = fit_camera_groups(source, target)
    assert result["accepted"]
    assert result["camera_support"]["agentview"]["inliers"] == 5
    assert result["camera_support"]["wrist"]["inliers"] == 5
    np.testing.assert_allclose(
        transformed_region(source["wrist"], result), target["wrist"], atol=1e-10
    )


def test_unobserved_depth_does_not_become_a_stationary_pose_or_closed_label() -> None:
    source = {"agentview": np.ones((8, 3)), "wrist": np.ones((8, 3))}
    result = fit_camera_groups(
        source, {name: np.full_like(points, np.nan) for name, points in source.items()}
    )
    assert not result["accepted"]
    assert transformed_region(source["agentview"], result) is None
    proximity = region_proximity(None, source["wrist"])
    assert not proximity["available"] and proximity["closing_label"] is None


def test_region_proximity_is_symmetric_and_invariant_to_camera_world_orientation() -> None:
    first = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    second = first + [0.2, 0, 0]
    result = region_proximity(first, second)
    assert result["symmetric_mean_m"] == pytest.approx(0.2)
    rotation = Rotation.from_rotvec([1.0, -0.3, 0.4]).as_matrix()
    changed = region_proximity(first @ rotation.T + [1, 2, 3], second @ rotation.T + [1, 2, 3])
    assert changed["symmetric_mean_m"] == pytest.approx(result["symmetric_mean_m"])
    assert result["closing_label"] is None
    assert not result["semantic_match_verified"]
