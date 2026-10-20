import numpy as np
import pytest

from vlm_tamp.rgbd_keypoint_candidates import propose_candidates, validated_settings
from vlm_tamp.rgbd_landmark_choices import validate_candidates
from vlm_tamp.rgbd_numericals import pixel_index


def scene(height=8, width=10):
    yy, xx = np.mgrid[:height, :width]
    points = np.stack([xx*.01, yy*.01, np.ones_like(xx)], axis=-1)
    features = np.stack([xx, yy, xx+yy, xx-yy], axis=-1).astype(float)
    return features, points


def test_background_and_invalid_geometry_features_bounds_and_depth_are_excluded():
    features, points = scene()
    labels = np.zeros((8, 10), dtype=int)
    labels[1:7, 1:9] = 4
    features[1, 1] = np.nan
    points[2, 2] = np.inf
    points[3, 3, 0] = 5.
    depth = np.ones((8, 10))
    depth[4, 4] = 6.
    depth[4, 5] = 0.
    settings = {"bounds_min": [-1., -1., .5], "bounds_max": [1., 1., 2.],
                "minimum_depth_m": .1, "maximum_depth_m": 5.}
    result = propose_candidates(features, points, labels, settings, depth_m=depth)
    assert result
    for candidate in result:
        x, y = candidate["pixel_xy"]
        assert candidate["predicted_region_id"] == labels[y, x] == 4
        assert (x, y) not in {(1, 1), (2, 2), (3, 3), (4, 4), (5, 4)}
        assert np.isfinite(candidate["position_world_m"]).all()
        assert np.array_equal(candidate["position_world_m"], points[y, x])


def test_constant_features_and_planar_coordinates_remain_finite_and_small_masks_reduce_clusters():
    features = np.ones((3, 4, 8))
    yy, xx = np.mgrid[:3, :4]
    points = np.stack([xx*.01, yy*.01, np.ones_like(xx)], axis=-1)
    result = propose_candidates(features, points, np.ones((3, 4), dtype=int),
                                {"candidates_per_region": 20})
    assert 0 < len(result) <= 12
    assert all(np.isfinite(item["position_world_m"]).all() for item in result)
    coincident = propose_candidates(features, np.ones((3, 4, 3)), np.ones((3, 4), dtype=int))
    assert len(coincident) == 1


def test_tiny_and_all_invalid_regions_and_empty_mask_list_return_empty():
    features, points = scene()
    labels = np.zeros((8, 10), dtype=int)
    labels[0, :11] = 1
    assert propose_candidates(features, points, labels) == []
    assert propose_candidates(features, np.full_like(points, np.nan), np.ones((8, 10), dtype=int)) == []
    assert propose_candidates(features, points, []) == []


def test_repeat_calls_preserve_candidates_ids_original_pixels_and_non_square_uvs():
    features, points = scene(9, 13)
    labels = np.zeros((9, 13), dtype=int)
    labels[:, :6], labels[:, 6:] = 2, 5
    first = propose_candidates(features, points, labels)
    assert first == propose_candidates(features, points, labels)
    assert validate_candidates(first, (9, 13)) == list(range(len(first)))
    for item in first:
        assert pixel_index(item["uv"], (9, 13)) == tuple(item["pixel_xy"])
        x, y = item["pixel_xy"]
        assert item["position_world_m"] == points[y, x].tolist()
        assert set(item) == {"id", "pixel_xy", "uv", "position_world_m", "predicted_region_id"}


def test_deduplication_never_merges_distinct_regions_even_at_identical_3d_points():
    features = np.ones((6, 8, 4))
    points = np.ones((6, 8, 3))
    labels = np.ones((6, 8), dtype=int)
    labels[:, 4:] = 2
    result = propose_candidates(features, points, labels, {"min_separation_m": .1})
    assert len(result) == 2
    assert {item["predicted_region_id"] for item in result} == {1, 2}
    assert result[0]["position_world_m"] == result[1]["position_world_m"]


def test_overlapping_rgb_masks_are_independent_and_region_cap_uses_declared_order():
    features, points = scene()
    mask = np.ones((8, 10), dtype=bool)
    result = propose_candidates(features, points, [mask, mask], {"candidates_per_region": 1})
    assert len(result) == 2
    assert result[0]["pixel_xy"] == result[1]["pixel_xy"]
    capped = propose_candidates(features, points, [mask, mask],
                                 {"candidates_per_region": 1, "max_regions": 1})
    assert len(capped) == 1 and capped[0]["predicted_region_id"] == 1


def test_world_z_is_not_substituted_for_missing_optical_depth():
    features, points = scene()
    with pytest.raises(ValueError, match="aligned depth_m"):
        propose_candidates(features, points, np.ones((8, 10), dtype=int), {"maximum_depth_m": 5.})


@pytest.mark.parametrize("settings", [
    {"candidates_per_region": True}, {"max_regions": 0}, {"min_separation_m": np.nan},
    {"bounds_min": [0, 0, 0]}, {"bounds_min": [1, 1, 1], "bounds_max": [0, 0, 0]},
    {"minimum_depth_m": 5., "maximum_depth_m": 1.}, {"maximum_depth_m": -1},
    {"reference_mask": "forbidden"},
])
def test_invalid_and_undeclared_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        validated_settings(settings)


def test_feature_and_geometry_grids_must_match_original_mask_and_depth_grid():
    features, points = scene()
    with pytest.raises(ValueError):
        propose_candidates(features[:-1], points, np.ones((8, 10), dtype=int))
    with pytest.raises(ValueError):
        propose_candidates(features, points, np.ones((8, 9), dtype=int))
    with pytest.raises(ValueError):
        propose_candidates(features, points, [np.ones((8, 9), dtype=bool)])
    with pytest.raises(ValueError):
        propose_candidates(features, points, np.ones((8, 10), dtype=float))
    with pytest.raises(ValueError):
        propose_candidates(features, points, np.ones((8, 10), dtype=int), depth_m=np.ones((8, 9)))
