"""A measured gap must not be repaired using motion predictions or robot pixels."""
from __future__ import annotations

import numpy as np
import pytest

from vlm_tamp.goal_gap import landmark_sample, measured_gap


def view() -> dict:
    world = np.zeros((4, 5, 3), dtype=float)
    world[..., 2] = 1
    world[1, 1] = [.1, .2, 1.]
    world[2, 3] = [.4, .6, 1.]
    return {'world_points': world, 'depth': np.ones((4, 5)),
            'known_robot_depth_prior': np.full((4, 5), np.nan)}


def test_gap_uses_observed_world_points_without_an_axis_or_direction() -> None:
    data = view()
    a = landmark_sample(data, [1, 1])
    b = landmark_sample(data, [3, 2])
    assert measured_gap(a, b)['gap_m'] == pytest.approx(.5)
    data['world_points'][2, 3] = [.1, .6, 1.]
    moved_fixed = landmark_sample(data, [3, 2])
    assert measured_gap(a, moved_fixed)['gap_m'] == pytest.approx(.4)


def test_occluded_prediction_does_not_become_a_measured_goal_gap() -> None:
    data = view()
    a = landmark_sample(data, [1, 1], visible=False)
    b = landmark_sample(data, [3, 2])
    result = measured_gap(a, b)
    assert result['gap_m'] is None
    assert result['raw_sample_distance_m'] == pytest.approx(.5)
    assert result['reason']['moving'] == 'tracker_not_visible'


def test_robot_depth_is_retained_as_raw_evidence_not_a_supported_object_point() -> None:
    data = view()
    data['known_robot_depth_prior'][1, 1] = 1.
    a = landmark_sample(data, [1, 1])
    assert a['raw_world_m'] == [.1, .2, 1.]
    assert not a['supported'] and a['reason'] == 'known_robot_depth'


def test_invalid_and_outside_samples_are_not_snapped() -> None:
    data = view()
    assert landmark_sample(data, [5, 2])['reason'] == 'outside_image'
    data['world_points'][1, 1] = np.nan
    assert landmark_sample(data, [1, 1])['raw_world_m'] is None


def test_missing_archived_depth_is_not_reconstructed_from_a_track() -> None:
    sample = landmark_sample(None, [1.2, 3.4], visible=True)
    assert sample['pixel_xy'] == [1.2, 3.4]
    assert sample['reason'] == 'recorded_depth_not_retained'
    assert sample['raw_world_m'] is None and not sample['supported']
