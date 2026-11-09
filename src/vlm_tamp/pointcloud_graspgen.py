"""GraspGen's existing exact outlier rule with bounded pairwise-distance storage."""

from __future__ import annotations

import torch


def filter_graspgen_points(
    points: torch.Tensor, *, threshold: float = .014, neighbours: int = 20,
    rows_per_block: int = 1024,
) -> tuple[torch.Tensor, dict]:
    """Apply the native mean-K-neighbour L1 rule without constructing an N-by-N tensor.

    Every point is compared with every other point on the input device. Only its
    own diagonal entry is excluded; coincident distinct points remain neighbours.
    This preserves GraspGen's threshold and metric, including the strict cutoff.
    Blocking changes storage/launch size, not the cloud or neighbour search.
    """
    points = points.float()
    retained = torch.empty(len(points), dtype=torch.bool, device=points.device)
    for start in range(0, len(points), rows_per_block):
        stop = min(start+rows_per_block, len(points))
        distances = torch.cdist(points[start:stop], points, p=1)
        local = torch.arange(stop-start, device=points.device)
        distances[local, local+start] = float('inf')
        nearest = torch.topk(distances, neighbours, dim=1, largest=False).values
        retained[start:stop] = nearest.mean(dim=1) < threshold
    filtered = points[retained]
    return filtered, {
        'method': 'native GraspGen exact mean-K-neighbour L1 filter, row-blocked',
        'input_points': len(points), 'retained_points': len(filtered),
        'removed_points': len(points)-len(filtered), 'neighbours': neighbours,
        'mean_distance_threshold_m': threshold, 'rows_per_block': rows_per_block,
        'point_subsampling': False, 'device': str(points.device),
    }
