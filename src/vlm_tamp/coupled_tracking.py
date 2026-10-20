"""Arrival checks for constrained trajectories under loaded joint tracking."""

import numpy as np


def phase_arrival(path, joints, progress, *, joint_tolerance=.006,
                  minimum_progress_fraction=.9):
    """A small joint target alone must not consume a short correction phase."""
    path=np.asarray(path,float);joints=np.asarray(joints,float)
    if (path.ndim!=2 or path.shape[1]!=8 or len(path)<2 or joints.shape!=(7,)
            or not np.isfinite(path).all() or not np.isfinite(joints).all()
            or not np.isfinite(progress)):
        raise ValueError('Finite path, seven joints and progress are required')
    if joint_tolerance<=0 or not 0<minimum_progress_fraction<=1:
        raise ValueError('Positive arrival tolerance and a progress fraction in (0,1] are required')
    span=float(path[-1,7]-path[0,7])
    if abs(span)<1e-10:raise ValueError('A contact phase needs nonzero angular progress')
    fraction=float((progress-path[0,7])/span)
    error=float(np.max(np.abs(joints-path[-1,:7])))
    return {'arrived':bool(error<joint_tolerance and fraction>=minimum_progress_fraction),
        'joint_target_reached':bool(error<joint_tolerance),'joint_error_rad':error,
        'directed_progress_fraction':fraction,'minimum_progress_fraction':minimum_progress_fraction,
        'joint_tolerance_rad':joint_tolerance}
