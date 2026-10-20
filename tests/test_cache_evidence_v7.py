"""Cache reuse accepts tiny render changes and rejects changed visual geometry."""

import numpy as np
import pytest

from vlm_tamp.rekep_libero.cache_evidence_v7 import grasp_proposal_agreement, region_agreement


CONFIG = {'cache_world_tolerance_m': .0002, 'cache_rgb_mean_channel_tolerance': .25,
          'cache_rgb_max_channel_tolerance': 16}


def evidence():
    rgb = np.full((100, 3), 120, np.uint8)
    xyz = np.c_[np.linspace(0., .1, 100), np.zeros(100), np.ones(100)]
    return rgb, xyz


def test_tiny_rgbd_capture_change_is_recorded_not_called_identical():
    rgb, xyz = evidence()
    new = rgb.copy()
    new[3] += 12
    result = region_agreement(rgb, new, xyz, xyz + [0, 0, .0001], CONFIG)
    assert result['accepted']
    assert not result['rgb_byte_exact']
    assert not result['world_points_byte_exact']


@pytest.mark.parametrize('change', ['recolored', 'moved', 'occluded'])
def test_changed_semantic_or_spatial_evidence_rejects_cache(change):
    rgb, xyz = evidence()
    other_rgb, other_xyz = rgb.copy(), xyz.copy()
    if change == 'recolored':
        other_rgb += 30
    elif change == 'moved':
        other_xyz += [0, 0, .003]
    else:
        other_xyz[3] = np.nan
    assert not region_agreement(rgb, other_rgb, xyz, other_xyz, CONFIG)['accepted']


def test_grasp_reuse_is_a_proposal_and_rejects_larger_motion():
    _, xyz = evidence()
    small = grasp_proposal_agreement(xyz + [.0001, 0, 0], xyz, CONFIG)
    assert small['accepted'] and not small['input_cloud_identical']
    assert small['current_geometry_admission_required']
    assert not grasp_proposal_agreement(xyz + [.002, 0, 0], xyz, CONFIG)['accepted']
