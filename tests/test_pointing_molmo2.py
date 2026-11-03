"""The model's image index and coordinate scale must survive the adapter."""
from vlm_tamp.pointing_molmo2 import parse_multi_image_points


def test_multi_image_coordinates_use_own_resolution() -> None:
    result = parse_multi_image_points(
        '<points coords="1 1 500 250;2 1 125 750" alt="door"/>',
        [(512, 512), (640, 480)])
    assert result['status'] == 'parsed'
    assert result['points'][0]['pixel_xy'] == [256, 128]
    assert result['points'][1]['image_index'] == 2
    assert result['points'][1]['pixel_xy'] == [80, 360]


def test_multiple_points_are_not_silently_reduced_to_one() -> None:
    result = parse_multi_image_points(
        '<points coords="2 1 300 400 2 600 700"/>', [(512, 512)] * 2)
    assert len(result['points']) == 2
    assert result['points'][1]['point_id'] == 2


def test_bad_coordinates_are_not_clipped() -> None:
    result = parse_multi_image_points('<points coords="2 1 1001 500"/>', [(512, 512)] * 2)
    assert result['status'] == 'coordinates_out_of_range'
    assert result['points'] == []


def test_wrong_image_is_not_reassigned() -> None:
    result = parse_multi_image_points('<points coords="3 1 500 500"/>', [(512, 512)] * 2)
    assert result['status'] == 'invalid_image_index'


def test_no_point_is_an_explicit_model_outcome() -> None:
    result = parse_multi_image_points('No visible contact.', [(512, 512)] * 2)
    assert result['status'] == 'no_point_tag'
    assert result['raw_text'] == 'No visible contact.'


def test_partial_parse_does_not_hide_malformed_tail() -> None:
    result = parse_multi_image_points('<points coords="1 1 500 500;garbled"/>', [(512, 512)] * 2)
    assert result['status'] == 'invalid_point_format'
    assert result['points'] == []
