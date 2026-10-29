import numpy as np
import pytest

from vlm_tamp.roartnet_adapter import OPTICAL_TO_MODEL, model_to_optical, optical_to_world, validate_cloud


def test_author_camera_rotation_and_inverse():
    optical = np.array([[.2, -.3, 1.4], [-.8, .1, .9]])
    model = optical @ OPTICAL_TO_MODEL.T
    np.testing.assert_allclose(model, [[1.4, -.2, .3], [.9, .8, -.1]])
    np.testing.assert_allclose(model_to_optical(model), optical)
    assert np.linalg.det(OPTICAL_TO_MODEL) == pytest.approx(1.)


def test_world_translation_does_not_affect_axis():
    transform = np.array([[0., -1., 0., 3.], [1., 0., 0., -2.], [0., 0., 1., .8], [0., 0., 0., 1.]])
    np.testing.assert_allclose(optical_to_world([.2, .3, 1.], transform), [2.7, -1.8, 1.8])
    np.testing.assert_allclose(optical_to_world([1., 0., 0.], transform, direction=True), [0., 1., 0.])


@pytest.mark.parametrize("bad", [np.diag([2., 1., 1., 1.]), np.diag([-1., 1., 1., 1.]), np.zeros((4, 4))])
def test_reject_scaled_or_reflected_calibration(bad):
    with pytest.raises(ValueError, match="preserve"):
        optical_to_world([1., 2., 3.], bad)


def test_invalid_depth_is_not_silently_used():
    cloud = np.random.default_rng(0).uniform([-.1, -.1, .7], [.1, .1, 1.], size=(200, 3))
    assert validate_cloud(cloud).dtype == np.float32
    cloud[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_cloud(cloud)
    cloud[0, 2] = -.1
    with pytest.raises(ValueError, match="positive"):
        validate_cloud(cloud)


def test_author_raw_fixture_exception_is_explicit_and_still_requires_finite_points():
    cloud = np.random.default_rng(1).uniform([-.1, -.1, .7], [.1, .1, 1.], size=(200, 3))
    cloud[0] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_cloud(cloud)
    assert np.array_equal(validate_cloud(cloud, author_raw_fixture=True)[0], [0, 0, 0])
    cloud[0, 0] = 1.
    assert np.array_equal(validate_cloud(cloud, author_raw_fixture=True)[0], [1, 0, 0])
    cloud[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_cloud(cloud, author_raw_fixture=True)
