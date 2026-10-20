"""A coarse support estimate must not turn the tabletop into payload geometry."""

import numpy as np
import pytest

from vlm_tamp.rekep_libero.scene_state_v10 import separate_source_from_support


def test_coarse_low_floor_estimate_does_not_absorb_higher_table_observations():
    table = np.array([[x, y, 1.0012 + .0003 * x / .15] for x in np.arange(-.15, .151, .003)
                      for y in np.arange(-.15, .151, .003)])
    body = np.array([[x, y, z] for x in np.arange(-.024, .025, .003)
                     for y in np.arange(-.024, .025, .003) for z in [1.007, 1.01, 1.013]])
    seed = body[(np.abs(body[:, 0]) < .016) & (np.abs(body[:, 1]) < .016)]
    points = np.vstack([table, body])
    mask, floor, evidence = separate_source_from_support(points, seed, .9999)
    assert not mask[:len(table)].any()
    assert mask[len(table):].all()
    assert floor > 1.001
    assert not evidence['table_removed_from_world_collision_map']


def test_missing_observed_support_is_a_recorded_geometry_failure():
    body = np.array([[x, y, z] for x in [-.01, 0, .01] for y in [-.01, 0, .01] for z in [1.01, 1.02]])
    with pytest.raises(ValueError, match='lacks observed local support'):
        separate_source_from_support(body, body, 1.)
