"""Blocked preprocessing must preserve native GraspGen filtering without an N-squared allocation."""

import pytest
import torch

from vlm_tamp.pointcloud_graspgen import filter_graspgen_points


@pytest.mark.parametrize('rows_per_block', [1, 17, 128])
def test_same_filter_as_dense_native_rule_with_duplicate_and_isolated_points(rows_per_block: int) -> None:
    """Point ordering, duplicate neighbours and outlier membership match the native calculation."""
    generator = torch.Generator().manual_seed(42)
    points = torch.rand(96, 3, generator=generator)*.02
    points[5:10] = points[0]
    points[-4:] += 1.
    distances = torch.cdist(points, points, p=1)
    distances.fill_diagonal_(float('inf'))
    keep = distances.topk(20, dim=1, largest=False).values.mean(1) < .014
    actual, record = filter_graspgen_points(points, rows_per_block=rows_per_block)
    assert keep.any() and (~keep).any()
    torch.testing.assert_close(actual, points[keep], rtol=0, atol=0)
    assert record['input_points'] == len(points)
    assert record['retained_points'] == int(keep.sum())
    assert not record['point_subsampling']


def test_distance_queries_bound_storage_but_keep_all_neighbours(monkeypatch: pytest.MonkeyPatch) -> None:
    """All original points remain eligible neighbours in every bounded CUDA-compatible query."""
    original = torch.cdist
    shapes = []

    def recorded(first: torch.Tensor, second: torch.Tensor, *, p: int) -> torch.Tensor:
        """Record the allocation dimensions while preserving the actual calculation."""
        shapes.append((len(first), len(second), p))
        return original(first, second, p=p)

    monkeypatch.setattr(torch, 'cdist', recorded)
    filter_graspgen_points(torch.zeros(73, 3), rows_per_block=16)
    assert shapes == [(16, 73, 1)]*4+[(9, 73, 1)]
