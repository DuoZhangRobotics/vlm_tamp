"""RGB target transfer rejects occlusion and geometric table contamination."""

import numpy as np
import pytest

from vlm_tamp.rekep_libero.target_region_v3 import (
    infer_multiview_target_cloud, visible_projected_points,
)


def camera(depth=None):
    depth = np.ones((20, 20)) if depth is None else np.asarray(depth, float)
    y, x = np.indices(depth.shape)
    points = np.stack([(x - 10) * depth / 80, (y - 10) * depth / 80, depth], axis=-1)
    return {'depth': depth, 'world_points': points, 'rgb': np.zeros((20, 20, 3), np.uint8),
            'intrinsic': [[80, 0, 10], [0, 80, 10], [0, 0, 1]], 'camera_to_world': np.eye(4)}


def mask():
    value = np.zeros((20, 20), bool)
    value[8:13, 8:13] = True
    return value


class Predictor:
    def __init__(self, masks, scores=None):
        self.masks = np.asarray(masks)
        self.scores = np.ones(len(masks)) * .95 if scores is None else np.asarray(scores)
        self.calls = []

    def set_image(self, rgb):
        assert rgb.shape == (20, 20, 3)

    def predict(self, point_coords, point_labels, multimask_output):
        assert np.all(point_labels == 1) and multimask_output
        self.calls.append(np.asarray(point_coords).copy())
        return self.masks, self.scores, None


def bundle(other):
    return {'main_camera': 'main', 'cameras': {'main': camera(), 'other': other}}


def test_visible_projection_never_uses_occluded_seeds():
    target = camera()['world_points'][mask()]
    depth = np.ones((20, 20))
    depth[:, :10] = .98
    visible, indices, diagnostics = visible_projected_points(target, camera(depth))
    assert len(visible) == 15 and np.all(visible[:, 0] >= 10)
    assert diagnostics['occluded_source_points'] == 10
    assert np.all(target[indices, 0] >= 0)


def test_only_depth_visible_object_points_prompt_the_rgb_model(tmp_path):
    depth = np.ones((20, 20))
    depth[:, :10] = .98
    visible_mask = mask()
    visible_mask[:, :10] = False
    predictor = Predictor([visible_mask])
    cloud, masks, report = infer_multiview_target_cloud(bundle(camera(depth)), mask(), predictor, tmp_path,
        minimum_mask_points=5)
    assert len(predictor.calls) == 1 and np.all(predictor.calls[0][:, 0] >= 10)
    assert report['accepted_added_views'] == 1
    assert np.array_equal(cloud[:25], camera()['world_points'][mask()])
    assert not masks['other'][:, :10].any()
    assert (tmp_path / 'quality.json').is_file()


def test_fully_occluded_view_never_calls_the_rgb_predictor():
    predictor = Predictor([mask()])
    cloud, masks, report = infer_multiview_target_cloud(bundle(camera(np.full((20, 20), .9))),
        mask(), predictor, None, minimum_mask_points=5)
    assert not predictor.calls and len(cloud) == 25 and 'other' not in masks
    assert report['views'][0]['reason'] == 'insufficient_depth_visible_object_seeds'


def test_nearby_table_points_are_not_added_even_if_rgb_mask_leaks_one_pixel():
    depth = np.ones((20, 20))
    depth[12, 12] = .994  # Table lies only 6mm below the cached object surface.
    depth[10, 10] = 1.002  # Different visible object surface is within measured bounds.
    predictor = Predictor([mask()])
    cloud, masks, report = infer_multiview_target_cloud(bundle(camera(depth)), mask(), predictor, None,
        minimum_mask_points=5)
    assert report['accepted_added_views'] == 1
    assert not masks['other'][12, 12]
    assert np.min(cloud[:, 2]) >= 1
    assert np.max(cloud[:, 2]) <= 1.003
    assert np.array_equal(cloud[:25], camera()['world_points'][mask()])


def test_rgb_region_extending_far_beyond_reference_object_is_rejected():
    broad = np.zeros((20, 20), bool)
    broad[7:14, 7:14] = True
    predictor = Predictor([broad])
    cloud, masks, report = infer_multiview_target_cloud(bundle(camera()), mask(), predictor, None,
        minimum_mask_points=5)
    assert report['accepted_added_views'] == 0 and len(cloud) == 25 and 'other' not in masks
    assert report['views'][0]['candidate_masks'][0]['reason'] == 'rgb_mask_extends_outside_measured_object_extent'


def test_giant_tabletop_mask_is_rejected_before_geometric_clipping():
    predictor = Predictor([np.ones((20, 20), bool)])
    _, masks, report = infer_multiview_target_cloud(bundle(camera()), mask(), predictor, None,
        minimum_mask_points=5)
    assert 'other' not in masks
    assert report['views'][0]['candidate_masks'][0]['reason'] == 'oversized_rgb_region'


def test_mask_must_cover_depth_visible_object_seeds():
    wrong = mask()
    wrong[:, 8:11] = False
    predictor = Predictor([wrong])
    _, masks, report = infer_multiview_target_cloud(bundle(camera()), mask(), predictor, None,
        minimum_mask_points=5)
    assert 'other' not in masks
    assert report['views'][0]['candidate_masks'][0]['reason'] == 'rgb_mask_does_not_cover_visible_object_seeds'


def test_projection_bounds_invalid_depth_and_calibration_validation():
    source = np.asarray([[0, 0, 1], [20, 0, 1], [0, 0, -1]])
    observed = camera()
    observed['depth'][10, 10] = np.nan
    pixels, _, report = visible_projected_points(source, observed)
    assert len(pixels) == 0 and report['projected_in_frustum'] == 1
    with pytest.raises(ValueError, match='at most 4mm'):
        visible_projected_points(source, camera(), depth_tolerance=.006)


def test_existing_target_report_is_preserved(tmp_path):
    (tmp_path / 'quality.json').write_text('existing')
    with pytest.raises(FileExistsError, match='Preserve'):
        infer_multiview_target_cloud(bundle(camera()), mask(), Predictor([mask()]), tmp_path,
            minimum_mask_points=5)
