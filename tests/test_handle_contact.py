import unittest
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from vlm_tamp.handle_contact import require_arrival, upright_plane, fit_upright_hinge, ObservedRobotScene, axis_from_bulk
from vlm_tamp.handle_contact import contact_pose_variants
from vlm_tamp.handle_contact import local_panel_band


class HandleContactTests(unittest.TestCase):
    def contact_fixture(self, name, cloud, *, allow_tool=False, panel=None):
        scene=ObservedRobotScene.__new__(ObservedRobotScene)
        scene.points=np.array([[0.,0.,0.]])
        scene.tree=cKDTree(scene.points)
        scene.contact=np.asarray(cloud)
        scene.contact_tree=cKDTree(scene.contact)
        scene.required=set();scene.allow_tool_contact=allow_tool;scene.panel=panel
        scene.tool_panel_contact=False;scene.tool_panel_brace_radius=.15
        scene.panel_half_band_m=.010
        volume={'robot_geom_name':name,'type':'sphere','center_world':[0.,0.,0.],
                'radius_m':.005,'fixed':False}
        return scene,volume

    def test_palm_contact_requires_capture_and_nearby_selected_handle(self):
        scene,palm=self.contact_fixture('gripper0_hand_collision',[[0.,0.,0.]])
        self.assertFalse(scene.check([palm],contact=True)['accepted'])
        scene.allow_tool_contact=True
        self.assertTrue(scene.check([palm],contact=True)['accepted'])
        arm={**palm,'robot_geom_name':'robot0_link6_collision'}
        self.assertFalse(scene.check([arm],contact=True)['accepted'])
        scene.contact_tree=cKDTree([[.05,0.,0.]])
        self.assertFalse(scene.check([palm],contact=True)['accepted'])

    def test_finger_brace_is_local_to_the_observed_panel(self):
        scene,finger=self.contact_fixture('gripper0_finger1_collision',[[.05,0.,0.]],
                                         panel=(np.zeros(3),np.array([0.,0.,1.])))
        self.assertTrue(scene.check([finger],contact=True)['accepted'])
        self.assertFalse(scene.check([finger],contact=False)['accepted'])
        scene.panel=(np.array([0.,0.,.05]),np.array([0.,0.,1.]))
        self.assertFalse(scene.check([finger],contact=True)['accepted'])

    def test_postcapture_finger_can_brace_other_face_of_measured_door_slab(self):
        scene,finger=self.contact_fixture('gripper0_finger1_collision',[[.05,0.,0.]],
             panel=(np.array([0.,0.,.024]),np.array([0.,0.,1.])))
        self.assertFalse(scene.check([finger],contact=True)['accepted'])
        scene.panel_half_band_m=.032
        self.assertTrue(scene.check([finger],contact=True)['accepted'])
        arm={**finger,'robot_geom_name':'robot0_link6_collision'}
        self.assertFalse(scene.check([arm],contact=True)['accepted'])

    def test_closing_palm_brace_stays_on_local_selected_panel_and_excludes_arm(self):
        scene,palm=self.contact_fixture('gripper0_hand_collision',[[.10,0.,0.]],
                                       allow_tool=True,panel=(np.zeros(3),np.array([0.,0.,1.])))
        self.assertFalse(scene.check([palm],contact=True)['accepted'])
        scene.tool_panel_contact=True
        self.assertTrue(scene.check([palm],contact=True)['accepted'])
        self.assertFalse(scene.check([palm],contact=False)['accepted'])
        self.assertFalse(scene.check([{**palm,'robot_geom_name':'robot0_link6_collision'}],contact=True)['accepted'])
        scene.panel=(np.array([0.,0.,.05]),np.array([0.,0.,1.]))
        self.assertFalse(scene.check([palm],contact=True)['accepted'])
        scene.panel=(np.zeros(3),np.array([0.,0.,1.]))
        scene.contact_tree=cKDTree([[.20,0.,0.]])
        self.assertFalse(scene.check([palm],contact=True)['accepted'])

    def test_postcapture_palm_accepts_positive_gap_but_still_rejects_non_target_penetration(self):
        scene,palm=self.contact_fixture('gripper0_hand_collision',[[.10,0.,0.]])
        scene.points=np.array([[.0055,0.,0.]])
        scene.tree=cKDTree(scene.points)
        self.assertFalse(scene.check([palm],contact=True)['accepted'])
        scene.allow_tool_contact=True
        self.assertTrue(scene.check([palm],contact=True)['accepted'])
        self.assertFalse(scene.check([{**palm,'robot_geom_name':'robot0_link6_collision'}],contact=True)['accepted'])
        scene.points=np.array([[.0045,0.,0.]])
        scene.tree=cKDTree(scene.points)
        self.assertFalse(scene.check([palm],contact=True)['accepted'])

    def test_current_panel_band_tracks_measured_face_and_excludes_table_and_distant_object(self):
        rng=np.random.default_rng(4)
        handle=np.c_[np.zeros(200),np.full(200,.05),np.linspace(.95,1.10,200)]
        panel=np.c_[rng.uniform(-.10,.10,500),np.full(500,.030),rng.uniform(.96,1.09,500)]
        table=np.c_[rng.uniform(-.1,.1,500),rng.uniform(-.03,.03,500),np.full(500,.90)]
        unrelated=np.c_[rng.uniform(-.1,.1,500),np.full(500,.12),rng.uniform(.96,1.09,500)]
        band,evidence=local_panel_band(np.vstack([panel,table,unrelated]),handle,
                                     (np.zeros(3),np.array([0.,1.,0.])),.15,.015)
        self.assertTrue(evidence['measured'])
        self.assertAlmostEqual(band,.036,places=6)
        self.assertEqual(evidence['support_points'],len(panel))

    def test_closing_moves_toward_observed_cabinet_and_opening_away(self):
        anchor=np.array([0.,-.24,1.]);pivot=np.array([0.,0.,1.]);bulk=np.array([.20,0.,1.])
        close,_=axis_from_bulk(anchor,pivot,bulk,'close')
        opened,_=axis_from_bulk(anchor,pivot,bulk,'open')
        np.testing.assert_allclose(close,[0,0,1])
        np.testing.assert_allclose(opened,[0,0,-1])

    def test_missing_robot_render_prior_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'not attached'):
            ObservedRobotScene({'main': {'robot_depth_prior': {'source': 'configured robot'}}}, {}, [], {})

    def test_missed_approach_cannot_authorize_closure(self):
        goal = [0., 0., 1., 0., 0., 0., 1.]
        for miss in [.053, .091]:
            actual = goal.copy(); actual[2] += miss
            with self.assertRaisesRegex(ValueError, 'refusing closure'):
                require_arrival(goal, actual)

    def test_arrival_checks_orientation_too(self):
        with self.assertRaises(ValueError):
            require_arrival([0,0,1,0,0,0,1], [0,0,1,0,0,1,0])
        self.assertLess(require_arrival([0,0,1,0,0,0,1], [.002,0,1,0,0,0,1])['position_error_m'], .006)

    def test_wrist_alternatives_preserve_contact_position_and_bound_rotation(self):
        target=np.r_[[.12,-.08,1.06],Rotation.from_euler('xyz',[.2,-.4,.8]).as_quat()]
        measured=np.r_[[.11,-.09,1.06],Rotation.from_euler('xyz',[.1,-.3,.7]).as_quat()]
        original=target.copy()
        options=contact_pose_variants(target,measured,[0.,0.,-1.])
        nominal=Rotation.from_quat(target[3:])
        for option in options:
            pose=option['pose_world_xyzw']
            np.testing.assert_array_equal(pose[:3],target[:3])
            self.assertLessEqual((Rotation.from_quat(pose[3:])*nominal.inv()).magnitude(),.300001)
        np.testing.assert_array_equal(target,original)

    def test_table_majority_does_not_replace_upright_door(self):
        rng = np.random.default_rng(7)
        table = np.c_[rng.uniform(-.3,.3,8000), rng.uniform(-.3,.3,8000), np.full(8000,.90)]
        door = np.c_[rng.uniform(-.15,.15,2000), rng.normal(.05,.0003,2000), rng.uniform(.94,1.12,2000)]
        result = upright_plane(np.vstack([table,door]), [0.,1.,1.2])
        self.assertGreater(result['normal']@[0,1,0], .99)
        self.assertAlmostEqual(result['point'][1], .05, delta=.002)

    def test_observed_arc_recovers_hinge(self):
        angle = np.linspace(0,.55,14)
        points = np.c_[.12+.24*np.cos(angle), -.08+.24*np.sin(angle), np.full(14,1.01)]
        fitted = fit_upright_hinge(points)
        self.assertTrue(fitted['accepted'])
        np.testing.assert_allclose(fitted['hinge_world_m'][:2], [.12,-.08], atol=1e-9)

    def test_short_arc_does_not_claim_identified_hinge(self):
        angle = np.linspace(0,.02,5)
        points = np.c_[.24*np.cos(angle), .24*np.sin(angle), np.ones(5)]
        self.assertFalse(fit_upright_hinge(points)['accepted'])


if __name__ == '__main__':
    unittest.main()
