import numpy as np
import pytest

from vlm_tamp.landmark_evidence import summarize_pairs


def test_stationary_selected_points_are_retained_as_evidence():
    a = np.array([[0., 0., 0.], [.1, .1, .1], [.2, .2, .2]])
    r = summarize_pairs(a, a)
    assert r['finite_pairs'] == r['near_stationary_xy_pairs'] == 3
    assert r['pairs_entering_existing_hinge_fit'] == 0
    assert r['xy_displacement_mm']['median'] == 0


def test_motion_is_not_lost_and_z_drift_is_reported_separately():
    a = np.zeros((3, 3)); b = a.copy()
    b[:, 0] = [.01, .02, .2]; b[1, 2] = .005
    r = summarize_pairs(a, b)
    assert r['near_stationary_xy_pairs'] == 0
    assert r['pairs_entering_existing_hinge_fit'] == 1
    assert r['vertical_drift_at_least_4mm_pairs'] == 1
    assert r['xy_displacement_mm']['median'] == 20


def test_empty_or_invalid_observations_do_not_become_stationarity_claims():
    r = summarize_pairs(np.array([[np.nan, 0, 0]]), np.zeros((1, 3)))
    assert r['finite_pairs'] == 0 and r['nonfinite_pairs_excluded'] == 1
    assert r['xy_displacement_mm']['median'] is None
    assert 'motion_class' not in r and 'contact_retained' not in r
    with pytest.raises(ValueError):
        summarize_pairs(np.zeros((2, 3)), np.zeros((3, 3)))
