import pytest

from vlm_tamp.robopoint_baseline import parse_points


def test_native_point_lists_are_preserved_without_mask_based_selection():
    assert parse_points("[(0.9, 0.1), (0.2, 0.8)]") == [[0.9, 0.1], [0.2, 0.8]]
    assert parse_points("```python\n[(0.1, 0.2)]\n```") == [[0.1, 0.2]]
    assert parse_points("(0.1, 0.2)") == [[0.1, 0.2]]


@pytest.mark.parametrize("text", ["[]", "[(1.1, 0.2)]", "[(True, 0.2)]", "[(0.2,)]",
                                  "__import__('os').system('false')", "Here is [(0.1, 0.2)]"])
def test_malformed_or_executable_outputs_are_rejected(text):
    with pytest.raises(ValueError):
        parse_points(text)
