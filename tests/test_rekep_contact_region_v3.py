"""Semantic target contacts do not excuse an adjacent table surface."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vlm_tamp.rekep_libero.contact_region_v3 import RGBTargetContactField
from vlm_tamp.rekep_libero.grasp_v3 import AnalyticRobotSDF


class Field:
    def __init__(self, state='occupied', allowed=True):
        self.state, self.allowed = state, allowed
        self.bounds_min = np.zeros(3)

    def states(self, points):
        return {name: np.full(len(points), name == self.state, bool) for name in ['occupied', 'free', 'unknown']}

    def query(self, points):
        return np.full(len(points), -.01 if self.state == 'free' else .003)

    def collision_values(self, points, margin, finger_flags, target_points, contact_radius):
        allowed = np.full(len(points), self.allowed, bool)
        return np.where(allowed, -margin, self.query(points) + margin), allowed


def observed_object():
    object_point = np.asarray([0., 0., .9048])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('y', np.pi / 3).as_matrix()
    transform[:3, 3] = object_point - transform[:3, :3] @ np.asarray([0., 0., 1.])
    world = np.broadcast_to(object_point, (9, 9, 3)).copy()
    camera = {'depth': np.ones((9, 9)), 'world_points': world,
              'intrinsic': [[40, 0, 4], [0, 40, 4], [0, 0, 1]], 'camera_to_world': transform}
    mask = np.zeros((9, 9), bool)
    mask[4, 4] = True
    return object_point, {'rgb_object_view': camera}, {'rgb_object_view': mask}


def test_oblique_table_point_passes_optical_tolerance_but_fails_world_surface_support():
    object_point, cameras, masks = observed_object()
    wrapper = RGBTargetContactField(Field(), cameras, masks)
    table_point = object_point.copy()
    table_point[2] -= .0034
    camera = cameras['rgb_object_view']
    inverse = np.linalg.inv(camera['camera_to_world'])
    optical_table = inverse[:3, :3] @ table_point + inverse[:3, 3]
    assert abs(optical_table[2] - 1) < .003  # Optical-only gating would wrongly allow the table.
    values, allowed = wrapper.collision_values([table_point], finger_flags=[True], target_points=[object_point])
    assert not allowed[0] and values[0] > 0
    assert wrapper.policy_metadata()['unknown_exemptions'] is False


def test_genuine_occupied_object_pad_contact_remains_allowed():
    point, cameras, masks = observed_object()
    wrapper = RGBTargetContactField(Field(), cameras, masks)
    values, allowed = wrapper.collision_values([point], finger_flags=[True], target_points=[point])
    assert allowed[0] and values[0] < 0


def test_rgb_mask_is_required_even_at_exact_measured_surface():
    point, cameras, masks = observed_object()
    masks['rgb_object_view'][:] = False
    values, allowed = RGBTargetContactField(Field(), cameras, masks).collision_values(
        [point], finger_flags=[True], target_points=[point])
    assert not allowed[0] and values[0] > 0


def test_wrapper_never_promotes_underlying_non_contact_or_unknown_to_allowed():
    point, cameras, masks = observed_object()
    no_contact = RGBTargetContactField(Field(allowed=False), cameras, masks)
    assert not no_contact.collision_values([point], finger_flags=[False], target_points=[point])[1][0]
    unknown = RGBTargetContactField(Field(state='unknown'), cameras, masks)
    values, allowed = unknown.collision_values([point], finger_flags=[True], target_points=[point])
    assert not allowed[0] and values[0] > 0


def test_exact_known_robot_free_prior_remains_unchanged():
    point, cameras, masks = observed_object()
    wrapper = RGBTargetContactField(Field(state='free', allowed=False), cameras, masks)
    values, allowed = wrapper.collision_values([point], finger_flags=[False])
    assert values[0] == pytest.approx(-.005) and not allowed[0]
    assert np.array_equal(wrapper.bounds_min, np.zeros(3))


def test_real_field_occupied_object_contact_and_unknown_neighborhood():
    point, cameras, masks = observed_object()
    shape = (3, 3, 3)
    occupied = np.zeros(shape, bool)
    occupied[1, 1, 1] = True
    field = AnalyticRobotSDF(point - .01, point + .01, occupied, np.zeros(shape, bool),
                             np.full(shape, .01), np.full(3, .01))
    wrapper = RGBTargetContactField(field, cameras, masks)
    points = np.asarray([point, point + [.01, 0, 0]])
    values, allowed = wrapper.collision_values(points, finger_flags=[True, True], target_points=points)
    assert allowed.tolist() == [True, False] and values[0] < 0 and values[1] > 0


def test_excessive_contact_tolerance_and_uncalibrated_masks_rejected():
    _, cameras, masks = observed_object()
    with pytest.raises(ValueError, match='at most 3mm'):
        RGBTargetContactField(Field(), cameras, masks, .005)
    with pytest.raises(ValueError, match='calibrated camera'):
        RGBTargetContactField(Field(), {}, masks)
