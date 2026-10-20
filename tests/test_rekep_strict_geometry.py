"""Unknown admission regressions independent of generated constraints."""

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from vlm_tamp.rekep_libero.geometry import ObservedSDF
from vlm_tamp.rekep_libero.strict_geometry import StrictObservedSDF, audit_frozen_bowl_grasps


def mixed_grid(sdf_type=ObservedSDF):
    free = np.ones((3, 3, 3), bool)
    free[1, 1, 1] = False
    signed = (distance_transform_edt(~free) - distance_transform_edt(free)) * .01
    return sdf_type(np.zeros(3), np.full(3, .02), np.zeros(free.shape, bool), free, signed, np.full(3, .01))


def test_mixed_corner_unknown_cannot_pass_interpolated_clearance():
    frozen, strict = mixed_grid(), mixed_grid(StrictObservedSDF)
    point = np.array([[.0149, .0149, .0149]])
    assert frozen.states(point)['unknown'][0]
    assert frozen.query(point)[0] < -.005  # Seven free corners hide the unknown voxel.
    assert strict.query(point)[0] > 0
    old, _ = frozen.collision_values(point)
    new, _ = strict.collision_values(point)
    assert old[0] <= 0 and new[0] > 0


def test_finger_contact_does_not_excuse_unknown_target_support():
    strict = mixed_grid(StrictObservedSDF)
    point = np.array([[.0149, .0149, .0149]])
    values, allowed = strict.collision_values(point, finger_flags=[True], target_points=point)
    assert values[0] > 0 and not allowed[0]


def test_observed_target_contact_and_known_robot_prior_remain_supported():
    free = np.ones((3, 3, 3), bool)
    free[1, 1, 1] = False
    occupied = ~free
    signed = (distance_transform_edt(~free) - distance_transform_edt(free)) * .01
    strict = StrictObservedSDF(np.zeros(3), np.full(3, .02), occupied, free, signed, np.full(3, .01))
    point = np.array([[.01, .01, .01]])
    values, allowed = strict.collision_values(point, finger_flags=[True], target_points=point)
    assert allowed[0] and values[0] < 0
    prior = np.zeros(free.shape, bool)
    prior[0, 0, 0] = True
    strict.known_robot_prior = prior
    assert strict.query([[0, 0, 0]])[0] < 0  # Configured robot prior is known free.


def test_outside_nonfinite_and_negative_margin_fail_closed():
    strict = mixed_grid(StrictObservedSDF)
    assert (strict.query([[1, 1, 1], [np.nan, 0, 0]]) > 0).all()
    with pytest.raises(ValueError, match='nonnegative'):
        strict.collision_values([[0, 0, 0]], margin=-.1)


def test_from_rgbd_constructs_strict_subclass():
    camera = {'depth': np.ones((8, 8)), 'intrinsic': [[4, 0, 4], [0, 4, 4], [0, 0, 1]],
              'camera_to_world': np.eye(4)}
    strict = StrictObservedSDF.from_rgbd([camera], [-.1, -.1, .5], [.1, .1, 1.5], .1)
    assert isinstance(strict, StrictObservedSDF)
    assert strict.query([[0, 0, 1.3]])[0] > 0


def test_real_frozen_500_grasps_still_rejected_and_three_alias_samples_blocked():
    # This optional artifact regression does not invoke any model or simulator.
    root = Path(__file__).resolve().parents[1] / 'results/rekep-libero-v1'
    if not (root / 'libero_goal-08-state041-seed0/stage1-grasp-sdf.npz').is_file():
        pytest.skip('Frozen evaluation artifacts are unavailable')
    result = audit_frozen_bowl_grasps(root)
    assert result['episodes'] == 5 and result['retained_candidates'] == 500
    assert result['unknown_samples_previously_admitted'] == 3
    assert result['additional_blocked_tool_samples'] == 3
    assert result['candidate_outcome_changes'] == 0 and result['tool_admission_changes'] == 0
    assert result['strict_accepted'] == 0
    affected = result['rows'][-1]['affected_candidates']
    assert [(row['index'], row['alias_samples']) for row in affected] == [(52, 1), (141, 2)]
    assert result['new_api_calls'] == result['new_model_inference_calls'] == result['new_simulator_calls'] == 0
