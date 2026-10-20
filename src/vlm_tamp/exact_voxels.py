"""Faster voxel deduplication with identical lexicographic/first-point output."""
import math
import numpy as np
from .rekep_libero.scene_state_v7 import voxel_points as baseline_voxel_points


def voxel_points(points,spacing=.002):
    values=np.asarray(points,float).reshape(-1,3)
    values=values[np.isfinite(values).all(axis=1)]
    if not len(values):return values
    if not np.isfinite(spacing) or spacing<=0:return baseline_voxel_points(values,spacing)
    cells=np.floor(values/spacing).astype(np.int64)
    low=cells.min(axis=0);high=cells.max(axis=0)
    widths=[int(b)-int(a)+1 for a,b in zip(low,high)]
    # A monotone mixed-radix key has exactly the same order as the original
    # three-column integer lexicographic sort. Overflow falls back unchanged.
    if math.prod(widths)>np.iinfo(np.int64).max:return baseline_voxel_points(values,spacing)
    offset=cells-low
    keys=(offset[:,0]*widths[1]+offset[:,1])*widths[2]+offset[:,2]
    return values[np.unique(keys,return_index=True)[1]]
