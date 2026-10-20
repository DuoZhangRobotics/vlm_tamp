"""A selected plate's rim is contact geometry; the table and other objects are not."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from vlm_tamp.rekep_libero.contact_policy_v8 import ContactPolicy
from vlm_tamp.rekep_libero.scene_state_v9 import expand_selected_support, connected_source_mask


def support_fixture():
    xy = np.array([[x, y] for x in np.arange(-.025, .026, .002)
                   for y in np.arange(-.025, .026, .002) if x*x + y*y <= .025**2])
    r = np.linalg.norm(xy, axis=1)
    plate = np.c_[xy, 1.01 + .004 * (r / .025) ** 6]
    seed = plate[r <= .012]
    floor = np.array([[x, y, 1.] for x in np.arange(-.04, .041, .002) for y in np.arange(-.04, .041, .002)])
    other = np.array([[x, y, 1.015] for x in [.032, .033, .034] for y in [.025, .026, .027]])
    return np.vstack([plate, floor, other]), seed, plate, floor, other


def test_support_growth_recovers_rim_but_never_table_or_disconnected_object():
    points, seed, plate, floor, other = support_fixture()
    selected, evidence = expand_selected_support(points, seed, 1.)
    assert len(selected) > len(seed) * 3
    assert cKDTree(selected).query(plate)[0].max() < 1e-9
    assert cKDTree(selected).query(floor)[0].min() > .005
    assert cKDTree(selected).query(other)[0].min() > .008
    assert evidence['minimum_selected_height_world_m'] > 1.002


def test_full_plate_contact_permission_remains_phase_specific():
    points, seed, plate, floor, other = support_fixture()
    selected, _ = expand_selected_support(points, seed, 1.)
    policy = ContactPolicy([{'robot_geom_name': 'gripper0_finger1_collision'}], {'task_type': 'pick_place'}, {})
    rim = plate[np.linalg.norm(plate[:, :2], axis=1) > .022]
    assert policy.support_allowed(rim, [], selected, 'place').all()
    assert not policy.support_allowed(rim, [], selected, 'carry').any()
    assert not policy.support_allowed(floor, [], selected, 'place').any()
    assert not policy.support_allowed(other, [], selected, 'place').any()


def test_support_without_selected_seed_evidence_is_rejected():
    points, seed, *_ = support_fixture()
    with pytest.raises(ValueError, match='no observed surface|lacks connected'):
        expand_selected_support(points + [.2, 0, 0], seed, 1.)


def test_current_source_component_can_include_newly_seen_connected_surfaces():
    points, seed, source, floor, other = support_fixture()
    low, high = np.array([-.035, -.035, 1.007]), np.array([.035, .035, 1.035])
    mask, evidence = connected_source_mask(points, seed, low, high)
    selected = points[mask]
    assert cKDTree(selected).query(source)[0].max() < 1e-9
    assert cKDTree(selected).query(other)[0].min() > .008
    assert cKDTree(selected).query(floor)[0].min() > .005
    assert evidence['current_component_ownership']
