"""Sensor-only diagnostics that distinguish fit quality from model agreement."""
import numpy as np
from .rigid_hinge_tracking import estimate_rigid_hinge


def hinge_fit_diagnostics(reference_world, current_world, prior, conditioned_fit):
    unconditioned=estimate_rigid_hinge(reference_world,current_world,prior=None)
    candidate=unconditioned.get('hinge_world_m')
    distance=(float(np.linalg.norm(np.asarray(candidate)[:2]-np.asarray(prior)[:2]))
              if candidate is not None else None)
    return {
        'source':'Same measured RGB-D correspondences, fitting without the initial pivot proximity condition',
        'sensor_fit_without_initial_pivot_prior':unconditioned,
        'initial_pivot_world_m':np.asarray(prior,float).tolist(),
        'candidate_distance_from_initial_pivot_xy_m':distance,
        'initial_pivot_proximity_tolerance_m':.10,
        'initial_pivot_proximity_passed':bool(distance<.10) if distance is not None else None,
        'sensor_fit_passed_but_initial_pivot_proximity_failed':bool(
            unconditioned['accepted'] and not conditioned_fit['accepted']
            and distance is not None and distance>=.10),
        'original_measurement_and_planner_model_unchanged':True,
        'simulator_articulation_used':False,
    }
