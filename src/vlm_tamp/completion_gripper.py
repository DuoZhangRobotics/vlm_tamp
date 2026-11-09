"""Measured mechanical closure, separate from object contact and retention."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# A finger moving less than 0.1 mm during one 20 Hz control interval is settled
# at this reporting resolution. This does not inflate any collision geometry.
FINGER_TRAVEL_TOLERANCE_M = .0001
FINGER_SPEED_TOLERANCE_M_S = .002
DEFAULT_CLOSE_CONTROL_LIMIT = 40


def closure_sample(previous_positions: ArrayLike, positions: ArrayLike,
                   velocities: ArrayLike, previous_reference: ArrayLike,
                   reference: ArrayLike) -> dict:
    """Require a completed command ramp and small measured motion of both fingers.

    A stopped finger may be touching an object or its own closed limit. No force,
    object identity, or simulator contact is used to infer retention.
    """
    arrays = [np.asarray(value, float) for value in
              (previous_positions, positions, velocities, previous_reference, reference)]
    if any(value.shape != (2,) or not np.isfinite(value).all() for value in arrays):
        raise ValueError('Two finite finger positions, velocities and controller references required')
    before, current, velocity, old_reference, new_reference = arrays
    travel = float(np.max(np.abs(current-before)))
    speed = float(np.max(np.abs(velocity)))
    reference_finished = bool(np.array_equal(old_reference, new_reference))
    return {
        'settled': reference_finished and travel <= FINGER_TRAVEL_TOLERANCE_M
                   and speed <= FINGER_SPEED_TOLERANCE_M_S,
        'controller_reference_stopped_changing': reference_finished,
        'maximum_finger_travel_m': travel, 'maximum_finger_speed_m_s': speed,
        'travel_tolerance_m': FINGER_TRAVEL_TOLERANCE_M,
        'speed_tolerance_m_s': FINGER_SPEED_TOLERANCE_M_S,
        'positions_m': current.tolist(), 'velocities_m_s': velocity.tolist(),
        'source': 'finger joint proprioception and closing command reference',
        'simulator_contacts_used': False,
    }
