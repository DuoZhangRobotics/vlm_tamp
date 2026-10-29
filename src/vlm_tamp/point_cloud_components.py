"""Metric connected-component cleanup for a selected sensor point cloud."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def largest_component(points, *, voxel_m=.005, connection_m=.015):
    """Keep the component with most original points, using sparse voxel centers.

    This removes disconnected background; it is not object recognition and
    cannot separate an object from a touching table or reconstruct occlusions.
    The caller must first select a target region. No joint estimate is used.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected a nonempty finite XYZ cloud")
    if not 0 < voxel_m < connection_m:
        raise ValueError("Require positive voxel size smaller than connection radius")
    cells, inverse, weights = np.unique(np.floor(points / voxel_m).astype(np.int64), axis=0,
                                       return_inverse=True, return_counts=True)
    pairs = cKDTree((cells + .5) * voxel_m).query_pairs(connection_m, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells)))
    count, labels = connected_components(graph.tocsr(), directed=False)
    support = np.bincount(labels, weights=weights).astype(int)
    selected = int(np.argmax(support))
    mask = labels[inverse] == selected
    return mask, {"method": "largest connected component by original-point count",
                  "voxel_m": voxel_m, "connection_m": connection_m, "components": int(count),
                  "input_points": len(points), "retained_points": int(mask.sum()),
                  "component_point_counts_descending": sorted(support.tolist(), reverse=True),
                  "joint_information_used": False}
