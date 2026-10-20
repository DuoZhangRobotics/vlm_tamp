"""Known robot depth self filtering retains strict occlusion behavior."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from vlm_tamp.rekep_libero.grasp_v3 import AnalyticRobotSDF
from vlm_tamp.rekep_libero.sensor_filter_v3 import (
    DEPTH_PRIOR_SOURCE, VISUAL_SOURCE, build_sensor_field, robot_endpoint_cameras,
)


def camera(depth=1.):
    return {'depth': np.full((3, 3), depth), 'intrinsic': [[10, 0, 1], [0, 10, 1], [0, 0, 1]],
            'camera_to_world': np.eye(4)}


def visual_hull():
    vertices = np.asarray([[x, y, z] for x in [-.2, .2] for y in [-.2, .2] for z in [.999, 1.001]])
    return {'type': 'convex', 'source': VISUAL_SOURCE, 'render_group': 1,
            'robot_geom_name': 'robot0_link6_visual', 'vertices_world': vertices.tolist(),
            'planes_world': ConvexHull(vertices).equations.tolist()}


def test_robot_endpoints_stop_being_occupied_without_becoming_free():
    observed = camera()
    filtered, metadata = robot_endpoint_cameras([observed], [visual_hull()])
    assert metadata['camera_counts'][0]['matched_robot_endpoints'] == 9
    assert np.isfinite(observed['depth']).all()
    assert np.isfinite(filtered[0]['depth']).all()
    assert np.isnan(filtered[0]['occupancy_depth']).all()
    field = build_sensor_field(AnalyticRobotSDF, filtered, [-.05, -.05, .5], [.05, .05, 1.3], .01)
    assert field.states([[0, 0, .8]])['free'][0]
    assert field.states([[0, 0, 1.]])['unknown'][0]
    assert field.states([[0, 0, 1.2]])['unknown'][0]
    assert field.collision_values([[0, 0, 1.], [0, 0, 1.2]])[0].min() > 0
    assert metadata['visual_hulls_clear_unknown'] is False


def test_another_camera_real_scene_endpoint_still_overrides_a_free_ray():
    filtered, _ = robot_endpoint_cameras([camera(), camera(.99)], [visual_hull()])
    assert np.isnan(filtered[0]['occupancy_depth']).all()
    assert np.isfinite(filtered[1]['occupancy_depth']).all()
    field = build_sensor_field(AnalyticRobotSDF, filtered, [-.05, -.05, .5], [.05, .05, 1.3], .01)
    assert field.states([[0, 0, .99]])['occupied'][0]
    assert field.collision_values([[0, 0, .99]])[0][0] > 0


def test_known_robot_before_hit_ray_is_free_but_occluded_tail_stays_unknown():
    filtered, metadata = robot_endpoint_cameras([camera()], [visual_hull()])
    field = build_sensor_field(AnalyticRobotSDF, filtered, [-.01, -.01, .996], [.01, .01, 1.106], .01)
    assert field.states([[0, 0, .996]])['free'][0]  # 4mm before a matched robot hit.
    assert field.collision_values([[0, 0, .996]])[0][0] <= 0
    assert field.states([[0, 0, 1.006]])['unknown'][0]
    assert field.collision_values([[0, 0, 1.006]])[0][0] > 0
    assert metadata['matched_robot_free_ray_buffer_m'] == 1e-5
    assert 'simulation' in metadata['robot_buffer_basis']


def test_unmatched_environment_pixels_retain_original_surface_and_ray_buffer():
    filtered, _ = robot_endpoint_cameras([camera(.98)], [visual_hull()])
    assert not filtered[0]['robot_endpoint_mask'].any()
    field = build_sensor_field(AnalyticRobotSDF, filtered, [-.01, -.01, .956], [.01, .01, 1.086], .01)
    assert field.states([[0, 0, .976]])['occupied'][0]  # 4mm before real scene hit stays in the band.
    assert field.collision_values([[0, 0, .976]])[0][0] > 0


def test_no_self_filter_matches_the_frozen_sensor_fusion():
    views = [camera()]
    original = AnalyticRobotSDF.from_rgbd(views, [-.05, -.05, .5], [.05, .05, 1.3], .02)
    filtered, metadata = robot_endpoint_cameras(views, [])
    adapted = build_sensor_field(AnalyticRobotSDF, filtered, [-.05, -.05, .5], [.05, .05, 1.3], .02)
    assert not metadata['enabled']
    assert np.array_equal(original.free, adapted.free)
    assert np.array_equal(original.occupied, adapted.occupied)
    assert np.array_equal(original.sdf_voxels, adapted.sdf_voxels)


def test_visual_hull_never_clears_unknown_until_another_depth_view_sees_it_free():
    reverse = camera()
    reverse['camera_to_world'] = np.diag([1., -1., -1., 1.])
    reverse['camera_to_world'][2, 3] = 2
    filtered, _ = robot_endpoint_cameras([camera(), reverse], [visual_hull()])
    field = build_sensor_field(AnalyticRobotSDF, filtered, [-.05, -.05, .5], [.05, .05, 1.5], .01)
    assert field.states([[0, 0, 1.2]])['free'][0]  # Seen by reverse depth, not inferred from a hull.
    assert field.states([[0, 0, 1.]])['unknown'][0]


def test_scene_geometry_cannot_enter_robot_endpoint_filter():
    volume = visual_hull()
    volume['source'] = 'simulator scene mesh'
    with pytest.raises(ValueError, match='configured current robot visual'):
        robot_endpoint_cameras([camera()], [volume])


def test_known_robot_render_prior_matches_only_same_depth_and_preserves_closer_scene():
    observed = camera()
    observed['depth'][0, 0] = .98  # Genuine foreground scene point must remain occupied.
    observed['known_robot_depth_prior'] = np.ones((3, 3))
    observed['known_robot_depth_prior'][2, 2] = np.nan  # No known robot projected here.
    observed['robot_depth_prior'] = {'source': DEPTH_PRIOR_SOURCE, 'scene_geometry_used': False,
                                     'units': 'meters', 'depth_convention': 'optical-axis z'}
    filtered, metadata = robot_endpoint_cameras([observed], [])
    assert metadata['enabled']
    assert np.isnan(filtered[0]['occupancy_depth']).sum() == 7
    assert filtered[0]['occupancy_depth'][0, 0] == .98
    assert filtered[0]['occupancy_depth'][2, 2] == 1
    assert metadata['camera_counts'][0]['closer_scene_endpoints_preserved'] == 1
    assert 'same-pixel' in metadata['camera_counts'][0]['self_identification']


def test_render_prior_does_not_require_a_convex_hull_or_lifted_endpoint_match():
    observed = camera()
    observed['known_robot_depth_prior'] = observed['depth'].copy()
    observed['robot_depth_prior'] = {'source': DEPTH_PRIOR_SOURCE, 'scene_geometry_used': False,
                                     'units': 'meters', 'depth_convention': 'optical-axis z'}
    observed['world_points'] = np.full((3, 3, 3), 10.)  # Deliberately outside all current visual hulls.
    filtered, _ = robot_endpoint_cameras([observed], [visual_hull()])
    assert np.isnan(filtered[0]['occupancy_depth']).all()


def test_render_prior_provenance_cannot_include_scene_geometry():
    observed = camera()
    observed['known_robot_depth_prior'] = observed['depth'].copy()
    observed['robot_depth_prior'] = {'source': DEPTH_PRIOR_SOURCE, 'scene_geometry_used': True,
                                     'units': 'meters', 'depth_convention': 'optical-axis z'}
    with pytest.raises(ValueError, match='only from configured current robot'):
        robot_endpoint_cameras([observed], [])
