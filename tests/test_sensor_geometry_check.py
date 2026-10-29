import numpy as np
import pytest

from vlm_tamp.sensor_geometry_check import triangulate_vertical_hinge


def camera(eye):
    eye = np.array(eye, float)
    z = np.array([.2, -.1, 1.])-eye; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4); t[:3, :3] = np.c_[x, y, z]; t[:3, 3] = eye
    return {'intrinsic': np.array([[500., 0., 256.], [0., 500., 256.], [0., 0., 1.]]), 'camera_to_world': t}


def pixels(view, xy):
    p = np.array([[*xy, .9], [*xy, 1.2]])
    t = view['camera_to_world']; q = (p-t[:3, 3]) @ t[:3, :3]
    q = q @ view['intrinsic'].T
    return q[:, :2] / q[:, 2:3]


def test_calibrated_lines_recover_xy_and_check_unfitted_view():
    views = {'a': camera([1, 0, 1.7]), 'b': camera([0, -1, 1.5]), 'check': camera([-1, .3, 1.6])}
    annotations = {name: pixels(view, [.2, -.1]) for name, view in views.items()}
    result = triangulate_vertical_hinge(views, annotations, fit_cameras=['a', 'b'], reference_z_m=1.07)
    np.testing.assert_allclose(result['origin_world_m'], [.2, -.1, 1.07], atol=1e-12)
    assert not result['views']['check']['used_for_fit']
    assert result['views']['check']['hinge_line_reprojection_error_px'] < 1e-10
    # Height along a vertical axis is a gauge choice, not a new native estimate.
    other = triangulate_vertical_hinge(views, annotations, fit_cameras=['a', 'b'], reference_z_m=.95)
    np.testing.assert_allclose(other['origin_world_m'][:2], [.2, -.1], atol=1e-12)


def test_parallel_view_planes_are_not_reported_as_verified():
    view = camera([1, 0, 1.7]); points = pixels(view, [.2, -.1])
    with pytest.raises(ValueError, match='insufficiently independent'):
        triangulate_vertical_hinge({'a': view, 'b': view}, {'a': points, 'b': points}, fit_cameras=['a', 'b'], reference_z_m=1.)


def test_withheld_wrong_edge_is_visible_in_reprojection_residual():
    views = {'a': camera([1, 0, 1.7]), 'b': camera([0, -1, 1.5]), 'check': camera([-1, .3, 1.6])}
    annotations = {name: pixels(view, [.2, -.1]) for name, view in views.items()}
    annotations['check'] = pixels(views['check'], [.2, .15])
    result = triangulate_vertical_hinge(views, annotations, fit_cameras=['a', 'b'], reference_z_m=1.)
    assert result['views']['check']['hinge_line_reprojection_error_px'] > 20
