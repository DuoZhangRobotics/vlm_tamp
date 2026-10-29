import numpy as np
from vlm_tamp.point_cloud_components import largest_component


def test_disconnected_background_removed_without_moving_coordinates():
    rng = np.random.default_rng(1)
    target = rng.uniform([0, 0, .8], [.05, .05, .85], (1500, 3))
    far = rng.uniform([0, 0, 3.], [.05, .05, 3.05], (200, 3))
    points = np.concatenate([target, far])
    mask, info = largest_component(points)
    np.testing.assert_array_equal(points[mask], target)
    assert info["components"] == 2


def test_component_selection_uses_pixel_support_not_voxel_count():
    a = np.repeat([[0., 0., 1.]], 100, axis=0)
    b = np.array([[1 + i * .005, 0, 1] for i in range(20)])
    mask, _ = largest_component(np.concatenate([a, b]))
    assert mask[:100].all() and not mask[100:].any()
