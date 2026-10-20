"""Permit leaving a selected support without bypassing unrelated collisions."""

import numpy as np
import pytest

from vlm_tamp.rekep_libero.contact_policy_v11 import ContactPolicy


SPEC = {'task_type': 'pick_place', 'object_surface': {'center_xy_m': [0., 0.], 'low_z_m': 1.}}


@pytest.mark.parametrize('offset,expected', [([0, 0, 0], True), ([0, 0, .04], False), ([.04, 0, 0], False)])
def test_selected_plate_pickup_permission_expires_after_lifting_or_transport(offset, expected):
    policy = ContactPolicy([{'robot_geom_name': 'gripper0_finger1_collision'}], SPEC, {})
    moving = np.array([[-.01, 0, 1.], [.01, 0, 1.02]]) + offset
    plate = np.array([[.01, 0, 1.001]])
    assert bool(policy.support_allowed(plate, moving, plate, 'carry')[0]) == expected


def test_pickup_permission_does_not_allow_other_objects_or_approach_contacts():
    policy = ContactPolicy([{'robot_geom_name': 'gripper0_finger1_collision'}], SPEC, {})
    moving = np.array([[-.01, 0, 1.], [.01, 0, 1.02]])
    plate, other = np.array([[.01, 0, 1.001]]), np.array([[.05, 0, 1.001]])
    assert not policy.support_allowed(other, moving, plate, 'carry').any()
    assert not policy.support_allowed(plate, moving, plate, 'approach').any()
    assert policy.support_allowed(plate, moving + [.1, 0, .04], plate, 'place').all()
