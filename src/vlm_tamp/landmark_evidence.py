"""Compact descriptive evidence from public tracked 3-D point pairs.

This does not classify target motion, infer contact, choose a hinge, or change
the fitter's acceptance rules. The selection mask can contain other surfaces.
"""
import numpy as np


def summarize_pairs(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.ndim != 2 or first.shape[1:] != (3,) or second.shape != first.shape:
        raise ValueError('Require equally sized N-by-3 world-coordinate arrays')
    finite = np.isfinite(first).all(1) & np.isfinite(second).all(1)
    a, b = first[finite], second[finite]
    xy = np.linalg.norm(b[:, :2] - a[:, :2], axis=1)
    dz = np.abs(b[:, 2] - a[:, 2])
    return {
        'source': 'Public RGB-D optical-flow correspondences in the previously selected moving geometry',
        'units': 'millimetres for displacement summaries; world-coordinate XY plane',
        'finite_pairs': len(a), 'nonfinite_pairs_excluded': int((~finite).sum()),
        'near_stationary_xy_pairs': int((xy <= .0015).sum()),
        'near_stationary_threshold_mm': 1.5,
        'xy_displacement_mm': {name: float(np.quantile(xy, fraction) * 1000) if len(xy) else None
                               for name, fraction in [('p10', .1), ('median', .5), ('p90', .9)]},
        'vertical_drift_at_least_4mm_pairs': int((dz >= .004).sum()),
        'pairs_entering_existing_hinge_fit': int(((xy > .0015) & (xy < .15) & (dz < .004)).sum()),
        'interpretation': 'These are measured point displacements before the hinge fitter removes near-stationary matches. Selection may include background, and correspondence errors or occlusion may occur. Counts do not certify target identity, contact retention, hinge accuracy, or task success. Inspect them alongside the original images and public fit result.',
        'simulator_object_state_used': False,
    }
