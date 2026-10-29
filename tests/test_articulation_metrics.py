import numpy as np
import pytest
from vlm_tamp.articulation_metrics import axis_errors


def test_parallel_displacement_and_origin_gauge():
    for origin in [[.03, .04, 0.], [.03, .04, 100.]]:
        result = axis_errors(origin, [0, 0, -1], [0, 0, .8], [0, 0, 1])
        assert result["axis_angle_deg"] == pytest.approx(0.)
        assert result["shortest_line_distance_mm"] == pytest.approx(50.)
        assert result["reference_plane_error_mm"] == pytest.approx(50.)


def test_shortest_distance_alone_hides_tilt():
    result = axis_errors([0, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1])
    assert result["shortest_line_distance_mm"] == pytest.approx(0.)
    assert result["reference_plane_error_mm"] == pytest.approx(1000.)
    assert result["axis_angle_deg"] == pytest.approx(45.)
    moved = axis_errors([2, 0, 2], [1, 0, 1], [0, 0, 1], [0, 0, 1])
    assert moved["reference_plane_error_mm"] == pytest.approx(result["reference_plane_error_mm"])


def test_perpendicular_axis_has_no_unique_reference_plane_intersection():
    result = axis_errors([0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1])
    assert result["axis_angle_deg"] == pytest.approx(90.)
    assert result["reference_plane_error_mm"] is None


@pytest.mark.parametrize("axis", [[0, 0, 0], [np.nan, 0, 1]])
def test_invalid_axes_rejected(axis):
    with pytest.raises(ValueError):
        axis_errors([0, 0, 0], axis, [0, 0, 0], [0, 0, 1])
