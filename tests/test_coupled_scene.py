import unittest
import copy
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from vlm_tamp.coupled_scene import (RobotVolumes, Scene, moving_mask, observed_mount_mask,
                                     bounded_panel_completion)
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.rekep_libero.geometry import transform_points
from vlm_tamp.rekep_libero.scene_state_v7 import bounded_points


def chain():
    return RobotChain({'reference_joints': [0.]*7, 'joint_axes_world': [[0., 0., 1.]]*7,
        'joint_anchors_world_m': [[0., 0., 0.]]*7, 'reference_tcp_world': np.eye(4),
        'joint_limits': [[-3., 3.]]*7})


def sphere(name, center, radius=.02, fixed=False):
    return {'robot_geom_name': name, 'type': 'sphere', 'center_world': np.asarray(center),
            'radius_m': radius, 'fixed': fixed}


def spec():
    return {'joint_origin_world_m': [0., 0., 0.], 'joint_axis_world': [0., 0., 1.],
            'front_plane': {'point_world_m': [1., 0., 1.]},
            'panel_normal_world': [1., 0., 0.],
            'measured_panel_surface_half_band_m': .03, 'motion': 'close'}


def panel_frame():
    side, height = np.meshgrid(np.linspace(-.12, .12, 61), np.linspace(.90, 1.10, 51))
    border = (np.abs(side) > .10) | (height < .92) | (height > 1.08)
    return np.c_[np.ones(border.sum()), side[border], height[border]]


class TestPanelCompletion(unittest.TestCase):
    def test_handle_support_cannot_expand_selected_panel_footprint(self):
        panel=panel_frame()
        # A dense observed handle at a higher level must remain collision data,
        # but cannot turn empty space between it and the panel into a slab.
        bar=np.c_[np.full(500,1.02),np.zeros(500),np.linspace(1.18,1.30,500)]
        observed=np.vstack([panel,bar]);static=[[1.3,0.,1.]]
        scene=Scene.from_points(static,observed,bar,spec(),
            config={'complete_planar_panel':True},panel_support=panel)
        np.testing.assert_array_equal(scene.moving_reference,observed)
        np.testing.assert_array_equal(scene.static_points,static)
        self.assertLessEqual(scene.inferred_panel_reference[:,2].max(),1.100001)
        self.assertEqual(scene.panel_completion['footprint_source'],'selected RGB-D plane pixels')
        # The unobserved interior of the supported panel is still occupied.
        hit=scene.check_volumes([sphere('gripper0_hand_collision',[1.,0.,1.],.01)],0.)
        self.assertFalse(hit['accepted'])
        self.assertGreater(hit['collisions'][0]['blocked_inferred_panel_points'],0)

    def test_persisted_panel_prior_transports_rigidly_without_rebuilding(self):
        observed = panel_frame()
        contact = np.c_[np.full(21, 1.05), np.zeros(21), np.linspace(.95, 1.05, 21)]
        original = Scene.from_points([], observed, contact, spec(),
            config={'complete_planar_panel': True})
        rotation = Rotation.from_rotvec([0., 0., .6]).as_matrix()
        translation = np.array([.1, -.02, .001])
        move = lambda points: np.asarray(points)@rotation.T+translation
        current_spec = spec()
        current_spec['front_plane'] = {'point_world_m': move([1., 0., 1.]).tolist()}
        current_spec['panel_normal_world'] = (rotation@np.array([1., 0., 0.])).tolist()
        current_handle = move(contact)
        prior = original.panel_prior_for(current_spec, current_handle)
        expected = move(original.inferred_panel_reference)
        np.testing.assert_allclose(prior['points_world'], expected, atol=1e-12)
        self.assertFalse(prior['points_world'].flags.writeable)
        np.testing.assert_allclose(np.asarray(prior['record']['normal_axis_world']),
                                   current_spec['panel_normal_world'], atol=1e-12)
        # Much larger newly visible support cannot enlarge the existing grid.
        fresh_observed = move(observed*np.array([1., 1.4, 1.]))
        static = [[3., .1, 1.]]
        with patch('vlm_tamp.coupled_scene.bounded_panel_completion',
                   side_effect=AssertionError('Prior reuse must not regenerate geometry')):
            current = Scene.from_points(static, fresh_observed, current_handle, current_spec,
                config={'complete_planar_panel': True}, panel_completion_prior=prior)
        np.testing.assert_array_equal(current.moving_reference, fresh_observed)
        np.testing.assert_array_equal(current.static_points, static)
        np.testing.assert_allclose(current.inferred_panel_reference, expected, atol=1e-12)
        self.assertEqual(len(current.inferred_panel_reference), len(original.inferred_panel_reference))
        self.assertEqual(current.panel_completion['spacing_at_most_m'], original.panel_completion['spacing_at_most_m'])
        self.assertTrue(current.panel_completion['rigid_prior_reused'])
        self.assertFalse(current.panel_completion['regenerated_from_fresh_view'])
        self.assertFalse(current.inferred_panel_reference.flags.writeable)
        self.assertFalse(np.shares_memory(current.inferred_panel_reference, prior['points_world']))

    def test_changed_observed_plane_origin_does_not_shift_existing_solid(self):
        contact = [[1.05, 0., .95], [1.05, 0., 1.05]]
        previous = Scene.from_points([], panel_frame(), contact, spec(),
            config={'complete_planar_panel': True})
        fitted = spec()
        fitted['front_plane'] = {'point_world_m': [1.02, 0., 1.]}
        prior = previous.panel_prior_for(fitted, contact)
        np.testing.assert_allclose(prior['points_world'], previous.inferred_panel_reference, atol=1e-12)
        np.testing.assert_allclose(prior['record']['origin_world_m'], [1., 0., 1.], atol=1e-12)
        self.assertEqual(prior['record']['current_observed_panel_origin_world_m'], [1.02, 0., 1.])

    def test_invalid_or_oversized_panel_prior_is_rejected(self):
        contact = [[1.05, 0., .95], [1.05, 0., 1.05]]
        previous = Scene.from_points([], panel_frame(), contact, spec(),
            config={'complete_planar_panel': True})
        prior = previous.panel_prior_for(spec(), contact)
        config = {'complete_planar_panel': True}
        bad = copy.deepcopy(prior)
        bad['points_world'] = np.array([[np.nan, 0., 1.]])
        with self.assertRaisesRegex(ValueError, 'finite Nx3'):
            Scene.from_points([], panel_frame(), contact, spec(), config=config, panel_completion_prior=bad)
        bad = copy.deepcopy(prior)
        bad['record']['inferred_points'] += 1
        with self.assertRaisesRegex(ValueError, 'point count'):
            Scene.from_points([], panel_frame(), contact, spec(), config=config, panel_completion_prior=bad)
        with self.assertRaisesRegex(ValueError, 'count limit'):
            Scene.from_points([], panel_frame(), contact, spec(),
                config={**config, 'panel_completion_maximum_points': 100}, panel_completion_prior=prior)
        bad = copy.deepcopy(prior)
        bad['record']['origin_world_m'] = [10., 10., 10.]
        with self.assertRaisesRegex(ValueError, 'finite bounds'):
            Scene.from_points([], panel_frame(), contact, spec(), config=config, panel_completion_prior=bad)
        with self.assertRaisesRegex(ValueError, 'complete_planar_panel'):
            Scene.from_points([], panel_frame(), contact, spec(), panel_completion_prior=prior)

    def test_missing_window_blocked_only_after_explicit_completion(self):
        observed = panel_frame()
        contact = [[1.05, 0., 1.]]
        old = Scene.from_points([], observed, contact, spec())
        completed = Scene.from_points([], observed, contact, spec(), config={'complete_planar_panel': True,
            'allow_palm_panel_brace': False, 'allow_palm_handle_contact': False})
        hand = sphere('gripper0_hand_collision', [1., 0., 1.], .01)
        self.assertTrue(old.check_volumes([hand], 0.)['accepted'])
        result = completed.check_volumes([hand], 0.)
        self.assertFalse(result['accepted'])
        self.assertGreater(result['collisions'][0]['blocked_inferred_panel_points'], 0)
        np.testing.assert_array_equal(completed.moving_reference, observed)
        self.assertEqual(completed.record()['moving_observed_points'], len(observed))
        self.assertGreater(completed.record()['inferred_panel_points'], 0)
        self.assertLessEqual(completed.record()['inferred_panel_points'], 50000)
        self.assertFalse(result['sampled_observed_surfaces_only'])

    def test_finite_extents_do_not_fill_outliers_and_rotate_with_door(self):
        observed = np.vstack([panel_frame(), [1., 2., 1.]])
        scene = Scene.from_points([], observed, [[1.05, 0., 1.]], spec(),
            config={'complete_planar_panel': True})
        self.assertLessEqual(np.abs(scene.inferred_panel_reference[:, 1]).max(), .120001)
        outside = sphere('robot0_link6_collision', [1., .4, 1.], .01)
        self.assertTrue(scene.check_volumes([outside], 0.)['accepted'])
        rotated_window = sphere('robot0_link6_collision', [0., 1., 1.], .01)
        self.assertTrue(scene.check_volumes([rotated_window], 0.)['accepted'])
        self.assertFalse(scene.check_volumes([rotated_window], np.pi/2)['accepted'])

    def test_completion_never_erases_or_relabels_static_obstacles(self):
        static = [[1., .4, 1.]]
        scene = Scene.from_points(static, panel_frame(), [[1.05, 0., 1.]], spec(),
            config={'complete_planar_panel': True})
        np.testing.assert_array_equal(scene.static_points, static)
        result = scene.check_volumes([sphere('gripper0_finger1_collision', static[0])], .6)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['collisions'][0]['scene_part'], 'static')

    def test_completion_fails_explicitly_for_unsupported_or_excessive_grid(self):
        with self.assertRaisesRegex(ValueError, '75 tagged'):
            bounded_panel_completion([[1., 0., 1.]], [1., 0., 1.], [1., 0., 0.], .03)
        with self.assertRaisesRegex(ValueError, 'two-dimensional'):
            bounded_panel_completion(np.c_[np.ones(100), np.zeros(100), np.linspace(.9, 1.1, 100)],
                                     [1., 0., 1.], [1., 0., 0.], .03)
        with self.assertRaisesRegex(ValueError, 'exceeding'):
            bounded_panel_completion(panel_frame(), [1., 0., 1.], [1., 0., 0.], .03,
                                     maximum_points=100)

    def test_target_finger_contact_allows_touch_but_limits_penetration(self):
        scene = Scene.from_points([], [[1., 0., 1.]], [[1., 0., 1.]], spec(),
            config={'maximum_target_contact_penetration_m': .003})
        shallow = sphere('gripper0_finger1_collision', [1.019, 0., 1.], .02)
        deep = sphere('gripper0_finger1_collision', [1.010, 0., 1.], .02)
        self.assertTrue(scene.check_volumes([shallow], 0.)['accepted'])
        self.assertFalse(scene.check_volumes([deep], 0.)['accepted'])

    def test_maintained_pad_bar_pair_exempts_only_selected_measured_contact(self):
        location = [1., 0., 1.]
        cfg = {'maximum_target_contact_penetration_m': .003}
        old = Scene.from_points([], [location], [location], spec(), config=cfg)
        maintained = Scene.from_points([], [location], [location], spec(),
            config={**cfg, 'allow_pad_handle_contact': True})
        pad = sphere('gripper0_finger1_pad_collision', location, .003001764)
        self.assertFalse(old.check_volumes([pad], 0.)['accepted'])
        self.assertTrue(maintained.check_volumes([pad], 0.)['accepted'])
        outer_finger = sphere('gripper0_finger1_collision', location, .003001764)
        self.assertFalse(maintained.check_volumes([outer_finger], 0.)['accepted'])
        # A moving point outside the selected measured bar surface is still
        # subject to the generic penetration limit even for a named pad.
        other = Scene.from_points([], [[1.012, 0., 1.]], [location], spec(),
            config={**cfg, 'allow_pad_handle_contact': True})
        self.assertFalse(other.check_volumes([sphere('gripper0_finger1_pad_collision',
                                                   [1.012, 0., 1.], .004)], 0.)['accepted'])
        static = Scene.from_points([location], [location], [location], spec(),
            config={**cfg, 'allow_pad_handle_contact': True})
        self.assertFalse(static.check_volumes([pad], 0.)['accepted'])

    def test_pad_exemption_never_applies_to_inferred_panel_samples(self):
        # The central missing-window sample is inferred, even though it lies
        # at the same coordinate as this selected contact reference fixture.
        scene = Scene.from_points([], panel_frame(), [[1., 0., 1.]], spec(),
            config={'complete_planar_panel': True, 'allow_pad_handle_contact': True,
                    'maximum_target_contact_penetration_m': .003})
        point = scene.inferred_panel_reference[np.argmin(np.linalg.norm(
            scene.inferred_panel_reference-[1., 0., 1.], axis=1))]
        result = scene.check_volumes([sphere('gripper0_finger1_pad_collision', point, .003001764)], 0.)
        self.assertFalse(result['accepted'])
        self.assertGreater(result['collisions'][0]['blocked_inferred_panel_points'], 0)

    def test_maintained_finger_assembly_matches_selected_handle_pair_only(self):
        point = [1., 0., 1.]
        cfg = {'maximum_target_contact_penetration_m': .003,
               'allow_finger_handle_contact': True}
        scene = Scene.from_points([], [point], [point], spec(), config=cfg)
        for name in ['gripper0_finger1_collision', 'gripper0_finger1_pad_collision',
                     'gripper0_finger2_collision', 'gripper0_finger2_pad_collision']:
            self.assertTrue(scene.check_volumes([sphere(name, point, .010)], 0.)['accepted'])
        # Enabling the semantic finger assembly never includes the palm.
        self.assertFalse(scene.check_volumes([sphere('gripper0_hand_collision', point, .010)], 0.)['accepted'])
        nonhandle = Scene.from_points([], [[1.012, 0., 1.]], [point], spec(), config=cfg)
        finger = sphere('gripper0_finger1_collision', [1.012, 0., 1.], .010)
        self.assertFalse(nonhandle.check_volumes([finger], 0.)['accepted'])
        static = Scene.from_points([point], [point], [point], spec(), config=cfg)
        self.assertFalse(static.check_volumes([sphere('gripper0_finger1_collision', point, .010)], 0.)['accepted'])

    def test_finger_assembly_exemption_never_applies_to_inferred_panel(self):
        scene = Scene.from_points([], panel_frame(), [[1., 0., 1.]], spec(),
            config={'complete_planar_panel': True, 'allow_finger_handle_contact': True,
                    'maximum_target_contact_penetration_m': .003})
        point = scene.inferred_panel_reference[np.argmin(np.linalg.norm(
            scene.inferred_panel_reference-[1., 0., 1.], axis=1))]
        for name in ['gripper0_finger1_collision', 'gripper0_finger1_pad_collision']:
            result = scene.check_volumes([sphere(name, point, .003025507)], 0.)
            self.assertFalse(result['accepted'])
            self.assertGreater(result['collisions'][0]['blocked_inferred_panel_points'], 0)

    def test_configured_palm_permissions_are_independently_disabled(self):
        scene = Scene.from_points([], [[1., 0., 1.]], [[1., 0., 1.]], spec(),
            config={'allow_palm_handle_contact': False, 'allow_palm_panel_brace': False})
        self.assertFalse(scene.check_volumes([sphere('gripper0_hand_collision', [1., 0., 1.], .002)], 0.)['accepted'])
        # This point is outside handle proximity but inside the legacy brace.
        legacy = Scene.from_points([], [[1., .05, 1.]], [[1., 0., 1.]], spec())
        no_brace = Scene.from_points([], [[1., .05, 1.]], [[1., 0., 1.]], spec(),
            config={'allow_palm_panel_brace': False})
        hand = sphere('gripper0_hand_collision', [1., .05, 1.], .002)
        self.assertTrue(legacy.check_volumes([hand], 0.)['accepted'])
        self.assertFalse(no_brace.check_volumes([hand], 0.)['accepted'])


class TestRobotVolumes(unittest.TestCase):
    def test_prefix_attachment_and_fixed_base(self):
        robot = RobotVolumes(chain(), [sphere('robot0_link0_collision', [1., 0., 0.]),
            sphere('robot0_link2_collision', [1., 0., 0.]),
            sphere('gripper0_hand_collision', [1., 0., 0.])])
        q = np.array([.2, .3, .1, 0., 0., 0., 0.])
        result = robot.forward(q)
        np.testing.assert_allclose(result[0]['center_world'], [1., 0., 0.])
        np.testing.assert_allclose(result[1]['center_world'], [np.cos(.5), np.sin(.5), 0.])
        np.testing.assert_allclose(result[2]['center_world'], [np.cos(.6), np.sin(.6), 0.])

    def test_reference_volumes_need_not_be_at_chain_reference(self):
        q = np.full(7, .1)
        center = Rotation.from_rotvec([0., 0., .7]).apply([1., 0., 0.])
        robot = RobotVolumes(chain(), [sphere('gripper0_hand_collision', center)], q)
        np.testing.assert_allclose(robot.forward(np.zeros(7))[0]['center_world'], [1., 0., 0.], atol=1e-12)

    def test_convex_planes_and_vertices_move_consistently(self):
        vertices = np.array([[1., -.01, -.01], [1., .01, -.01], [1., 0., .01], [1.02, 0., 0.]])
        planes = np.array([[1., 0., 0., -1.02], [-1., 0., 0., 1.]])
        robot = RobotVolumes(chain(), [{'robot_geom_name': 'robot0_link1_collision',
            'type': 'convex', 'vertices_world': vertices, 'planes_world': planes}])
        q = np.array([np.pi/2, 0., 0., 0., 0., 0., 0.])
        volume = robot.forward(q)[0]
        np.testing.assert_allclose(volume['vertices_world'][:, 1], vertices[:, 0], atol=1e-12)
        np.testing.assert_allclose(volume['planes_world'][:, :3], [[0., 1., 0.], [0., -1., 0.]], atol=1e-12)

    def test_unknown_attachment_fails(self):
        with self.assertRaisesRegex(ValueError, 'attachment'):
            RobotVolumes(chain(), [sphere('mystery_collision', [1., 0., 0.])])


class TestTaggedScene(unittest.TestCase):
    def test_execution_guard_accepts_positive_gap_below_planning_buffer(self):
        cfg = {'observed_clearance_margin_m': .003, 'noncontact_tool_clearance_m': .002}
        scene = Scene.from_points([[1.0226, 0., 1.]], [[2., 0., 1.]], [[2., 0., 1.]],
                                  spec(), config=cfg)
        arm = sphere('robot0_link6_collision', [1., 0., 1.], .02)
        planning = scene.check_volumes([arm], 0.)
        execution = scene.check_volumes([arm], 0., execution=True)
        self.assertFalse(planning['accepted'])
        self.assertTrue(execution['accepted'])
        self.assertEqual(planning['checking_mode'], 'planning')
        self.assertEqual(planning['clearance_margin_m_by_robot_part']['robot0_link6_collision'], .003)
        self.assertEqual(execution['checking_mode'], 'execution_guard')
        self.assertEqual(execution['clearance_margin_m_by_robot_part']['robot0_link6_collision'], .001)
        self.assertFalse(execution['unknown_space_validated'])

    def test_execution_guard_does_not_permit_static_penetration_or_inferred_panel(self):
        cfg = {'observed_clearance_margin_m': .003}
        scene = Scene.from_points([[1.0199, 0., 1.]], [[2., 0., 1.]], [[2., 0., 1.]],
                                  spec(), config=cfg)
        arm = sphere('robot0_link6_collision', [1., 0., 1.], .02)
        self.assertFalse(scene.check_volumes([arm], 0.)['accepted'])
        self.assertFalse(scene.check_volumes([arm], 0., execution=True)['accepted'])
        panel = Scene.from_points([], panel_frame(), [[1.05, 0., 1.]], spec(), config={
            'complete_planar_panel': True, 'allow_palm_handle_contact': False,
            'allow_palm_panel_brace': False, 'noncontact_tool_clearance_m': .002})
        palm = sphere('gripper0_hand_collision', [1., 0., 1.], .01)
        self.assertFalse(panel.check_volumes([palm], 0.)['accepted'])
        self.assertFalse(panel.check_volumes([palm], 0., execution=True)['accepted'])
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            scene.config['execution_clearance_margin_m'] = -.001
            scene.check_volumes([arm], 0., execution=True)

    def test_reference_frame_queries_equal_world_queries_at_nonzero_hinge_angles(self):
        rng = np.random.default_rng(4)
        moving = np.array([1., 0., 1.])+rng.uniform([-.02, -.08, -.08], [.02, .08, .08], (400, 3))
        contact_points = np.c_[np.ones(30), np.zeros(30), np.linspace(.96, 1.04, 30)]
        static = np.array([[.8, .2, 1.], [.78, .22, 1.02]])
        for axis in [[0., 0., 1.], [.3, .1, .95]]:
            model = spec();model['joint_axis_world'] = axis
            scene = Scene.from_points(static, moving, contact_points, model)
            for theta in [.4, -.7]:
                transform = scene.transform(theta)
                centre = transform_points([[1., 0., 1.]], transform)[0]
                volumes = [sphere('gripper0_finger1_collision', centre, .045),
                           sphere('gripper0_hand_collision', centre, .07),
                           sphere('robot0_link6_collision', centre, .05)]
                half = np.array([.035, .055, .03])
                vertices = centre+np.stack(np.meshgrid(*[[-v, v] for v in half]), axis=-1).reshape(-1, 3)
                normals = np.r_[np.eye(3), -np.eye(3)]
                volumes.append({'type': 'convex', 'robot_geom_name': 'robot0_link5_collision',
                    'vertices_world': vertices,
                    'planes_world': np.c_[normals, -normals@centre-np.r_[half, half]]})
                # An independently implemented world-frame query is the
                # previous checker, before the reference-frame optimization.
                world_moving = transform_points(moving, transform)
                world_contacts = transform_points(contact_points, transform)
                contact_tree = cKDTree(world_contacts)
                panel_origin = transform_points([scene.panel_origin], transform)[0]
                panel_normal = transform[:3, :3]@scene.panel_normal
                low, high = np.quantile(world_contacts[:, 2], [.01, .99])
                expected, exemptions = {}, 0
                for volume in volumes:
                    name = volume['robot_geom_name']
                    finger, hand = 'finger' in name, name.endswith('hand_collision')
                    margin = 0. if finger or hand else .001
                    for tag, cloud in [('static', static), ('moving', world_moving)]:
                        ids, distance = bounded_points(cloud, cKDTree(cloud), volume, max(.001, margin))
                        hit = cloud[ids]
                        allowed = np.zeros(len(hit), bool)
                        if tag == 'moving' and (finger or hand):
                            selected_distance = contact_tree.query(hit)[0]
                            allowed |= selected_distance < .028
                            reach = .065 if finger else scene.tool_panel_radius
                            allowed |= ((selected_distance < reach)
                                        & (np.abs((hit-panel_origin)@panel_normal) < scene.panel_band)
                                        & (hit[:, 2] >= low-.015) & (hit[:, 2] <= high+.015))
                        exemptions += int(allowed.sum())
                        blocked = (distance <= margin)&~allowed
                        if blocked.any():
                            expected[name, tag] = int(blocked.sum()), float(distance[blocked].min())
                actual = scene.check_volumes(volumes, theta)
                self.assertEqual(actual['accepted'], not expected)
                self.assertEqual(actual['local_contact_exemptions'], exemptions)
                self.assertEqual(len(actual['collisions']), len(expected))
                for row in actual['collisions']:
                    count, minimum = expected[row['robot_geom_name'], row['scene_part']]
                    self.assertEqual(row['blocked_points'], count)
                    self.assertAlmostEqual(row['minimum_signed_point_clearance_m'], minimum, places=12)
                    if row['scene_part'] == 'moving':
                        self.assertLess(cKDTree(world_moving).query(row['example_points_world_m'])[0].max(), 1e-12)

    def test_static_mug_is_never_an_allowed_handle_contact(self):
        location = [1., 0., 1.]
        scene = Scene.from_points([location], [location], [location], spec())
        result = scene.check_volumes([sphere('gripper0_finger1_collision', location)], 0.)
        self.assertFalse(result['accepted'])
        self.assertEqual([row['scene_part'] for row in result['collisions']], ['static'])
        self.assertEqual(result['local_contact_exemptions'], 1)

    def test_optional_tool_clearance_rejects_static_near_contact_only(self):
        location = [1., 0., 1.]
        moving = [location]
        static = [[1.021, 0., 1.]]
        tool = sphere('gripper0_finger1_collision', location, .02)
        default = Scene.from_points(static, moving, moving, spec())
        careful = Scene.from_points(static, moving, moving, spec(),
            config={'noncontact_tool_clearance_m': .002})
        self.assertTrue(default.check_volumes([tool], 0.)['accepted'])
        result = careful.check_volumes([tool], 0.)
        self.assertFalse(result['accepted'])
        self.assertEqual([row['scene_part'] for row in result['collisions']], ['static'])
        self.assertEqual(result['local_contact_exemptions'], 1)

    def test_hand_contact_does_not_exempt_arm(self):
        location = [1., 0., 1.]
        scene = Scene.from_points([], [location], [location], spec())
        self.assertTrue(scene.check_volumes([sphere('gripper0_hand_collision', location)], 0.)['accepted'])
        self.assertFalse(scene.check_volumes([sphere('robot0_link6_collision', location)], 0.)['accepted'])
        self.assertFalse(scene.check_volumes([sphere('gripper0_hand_collision', location)], 0., contact=False)['accepted'])

    def test_transport_never_erases_or_moves_static_points(self):
        scene = Scene.from_points([[0., 1., 1.]], [[1., 0., 1.]], [[1., 0., 1.]], spec())
        before = scene.static_points.copy()
        moved = scene.moved(np.pi/2)[0]
        np.testing.assert_allclose(moved, [[0., 1., 1.]], atol=1e-12)
        np.testing.assert_array_equal(scene.static_points, before)
        self.assertFalse(scene.static_points.flags.writeable)
        result = scene.check_volumes([sphere('gripper0_hand_collision', [0., 1., 1.])], np.pi/2)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['collisions'][0]['scene_part'], 'static')

    def test_replanning_retains_an_occluded_static_obstacle(self):
        location=[.8,.2,1.]
        before=Scene.from_points([location],[[1.,0.,1.]],[[1.,0.,1.]],spec())
        current=Scene.from_points([],[[1.,0.,1.]],[[1.,0.,1.]],spec())
        current.retain_static(before)
        result=current.check_volumes([sphere('gripper0_hand_collision',location)],0.)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['collisions'][0]['scene_part'],'static')
        self.assertTrue(current.record()['prior_static_observations_retained'])
        self.assertFalse(current.static_points.flags.writeable)

    def test_newly_visible_bar_side_moves_while_nearby_object_stays_static(self):
        yy,xx=np.indices((10,10))
        camera={'world_points':np.stack([xx*.01,yy*.01,np.ones_like(xx)],axis=-1),
                'camera_to_world':np.eye(4),'intrinsic':np.diag([100.,100.,1.])}
        model=spec();model['front_plane']={'point_world_m':[0.,.02,1.]}
        model['panel_normal_world']=[0.,1.,0.];model['source_anchor_world_m']=[.06,.06,1.]
        partial=np.c_[np.full(100,.07),np.full(100,.06),np.linspace(.98,1.02,100)]
        points=np.array([[.05,.06,1.],[.20,.06,1.]])
        selected=moving_mask(points,camera,model,partial,np.ones((10,10),bool),
                             config={'extend_observed_panel':False})
        np.testing.assert_array_equal(selected,[True,False])

    def test_observed_mount_owns_initial_and_rotated_blockers_only(self):
        model = spec()
        model['joint_origin_world_m'] = [-.1881895, .2823369, .96183]
        model['front_plane'] = {'point_world_m': [-.1639175, .1314838, .986611]}
        model['panel_normal_world'] = [-.98730185, -.15885546, 0.]
        model['source_anchor_world_m'] = [-.1971686, .0257473, 1.06566]
        anchor = np.asarray(model['source_anchor_world_m'])
        normal = np.asarray(model['panel_normal_world'])
        handle = np.c_[np.full(100, anchor[0]-.010), np.full(100, anchor[1]),
                       np.linspace(.944, 1.091, 100)]
        blocker = np.array([-.178, .035, 1.09])
        seed = blocker+np.array([.002, 0., 0.])
        # A nearby object is outside the finite lateral radius. An endpoint
        # inside the capsule but not connected to the observed part also stays
        # stationary; geometrical plausibility alone does not establish it.
        off_axis = blocker+np.array([0., .030, 0.])
        unsupported = blocker+np.array([0., 0., -.060])
        points = np.vstack([blocker, seed, off_axis, unsupported])
        original = observed_mount_mask(points, model, handle, [False, True, False, False])
        np.testing.assert_array_equal(original, [True, True, False, False])
        rotation = Rotation.from_rotvec([0., 0., .45]).as_matrix()
        pivot = np.asarray(model['joint_origin_world_m'])
        move = lambda cloud: (np.asarray(cloud)-pivot)@rotation.T+pivot
        rotated = dict(model)
        rotated['source_anchor_world_m'] = move(anchor).tolist()
        rotated['front_plane'] = {'point_world_m': move(model['front_plane']['point_world_m']).tolist()}
        rotated['panel_normal_world'] = (rotation@normal).tolist()
        transformed = observed_mount_mask(move(points), rotated, move(handle), [False, True, False, False])
        np.testing.assert_array_equal(transformed, original)

    def test_mount_prior_rejects_non_upright_or_unbounded_gap(self):
        model = spec();model['source_anchor_world_m'] = [1.05, 0., 1.]
        handle = np.c_[np.full(20, 1.06), np.zeros(20), np.linspace(.95, 1.05, 20)]
        model['joint_axis_world'] = [1., 0., 0.]
        self.assertFalse(observed_mount_mask([[1.04, 0., 1.]], model, handle, [True]).any())
        model['joint_axis_world'] = [0., 0., 1.]
        model['front_plane'] = {'point_world_m': [.5, 0., 1.]}
        self.assertFalse(observed_mount_mask([[1.04, 0., 1.]], model, handle, [True]).any())

    def test_missing_robot_parts_fail(self):
        volume = sphere('robot0_link6_collision', [1., 0., 1.])
        scene = Scene.from_points([], [[1., 0., 1.]], [[1., 0., 1.]], spec(), reference_volumes=[volume])
        with self.assertRaisesRegex(ValueError, 'Full configured robot'):
            scene.check_volumes([], 0.)

    def test_unknown_volume_and_space_not_certified(self):
        scene = Scene.from_points([], [[1., 0., 1.]], [[1., 0., 1.]], spec())
        self.assertFalse(scene.record()['door_environment_collision_checked'])
        self.assertFalse(scene.record()['unknown_space_validated'])

    def test_panel_partition_leaves_unrelated_depth_static(self):
        yy, xx = np.indices((10, 10))
        world = np.stack([xx*.01, yy*.01, np.ones_like(xx)], axis=-1)
        camera = {'world_points': world, 'camera_to_world': np.eye(4),
                  'intrinsic': np.diag([100., 100., 1.])}
        local_spec = spec()
        local_spec['front_plane'] = {'point_world_m': [0., 0., 1.]}
        local_spec['panel_normal_world'] = [0., 1., 0.]
        panel_mask = np.ones((10, 10), bool)
        # Narrow observed slab around y=0.02; the second endpoint is an
        # unrelated object projecting into the same RGB region but far deeper.
        local_spec['front_plane'] = {'point_world_m': [0., .02, 1.]}
        points = np.array([[.04, .02, 1.], [.04, .08, 1.]])
        result = moving_mask(points, camera, local_spec, [[.04, .02, 1.]], panel_mask,
            choices={'moving_panel_box_xyxy': [0., 0., 9., 9.]})
        np.testing.assert_array_equal(result, [True, False])


if __name__ == '__main__':
    unittest.main()
