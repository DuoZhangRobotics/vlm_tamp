import unittest
from unittest.mock import patch

import numpy as np

from vlm_tamp.coupled_pipeline import terminal_angle,supported_handle_anchors,closing_normal_residual,goal_pose_menu


class CoupledGoalTests(unittest.TestCase):
    def test_goal_roll_menu_preserves_contact_point_and_horizontal_jaws(self):
        from types import SimpleNamespace
        from scipy.spatial.transform import Rotation
        reference=np.eye(4);reference[:3,3]=[.3,.2,1.]
        rotation=Rotation.from_rotvec([0.,0.,.7]).as_matrix()
        transform=np.eye(4);transform[:3,:3]=rotation
        base=transform@reference;pad=np.array([.002,.001,-.0036])
        problem=SimpleNamespace(pose=lambda angle:base,transform=lambda angle:transform)
        candidates=list(goal_pose_menu(problem,.7,reference,pad,[0,0,1]))
        self.assertEqual(len(candidates),20)
        expected=rotation@(reference[:3,3]+pad)
        for pose,angles in candidates:
            np.testing.assert_allclose(pose[:3,3]+pose[:3,:3]@pad,expected,atol=1e-12)
            self.assertAlmostEqual(pose[2,0],0.,places=12)
        self.assertGreater(max(abs(a['roll_about_jaw_rad']) for _,a in candidates),1.5)

    def test_final_closing_residual_uses_observed_plane_orientation(self):
        angle=np.deg2rad(.5)
        normal=[np.cos(angle),-np.sin(angle),0.]
        self.assertAlmostEqual(closing_normal_residual(normal,[1.,0.,0.],[0.,0.,1.]),angle)
        self.assertAlmostEqual(closing_normal_residual([1.,0.,0.],[1.,0.,0.],[0.,0.,1.]),0.)
        self.assertLess(closing_normal_residual([np.cos(angle),np.sin(angle),0.],[1.,0.,0.],[0.,0.,1.]),0.)

    @staticmethod
    def pad_metadata():
        from itertools import product
        return {'eef_pose_world_xyzw':[0.,0.,0.,0.,0.,0.,1.],
            'robot_collision_volumes_world':[
                {'robot_geom_name':f'finger{i}_pad_collision',
                 'vertices_world':np.array(list(product([x-.002,x+.002],[-.008,.008],[-.0046,-.0026]))).tolist()}
                for i,x in enumerate([-.03,.03])]}

    def test_finite_pads_exclude_observed_handle_tip(self):
        cloud=np.c_[np.zeros(1001),np.zeros(1001),np.linspace(.94,1.10,1001)]
        kept,record=supported_handle_anchors([[0.,0.,1.02],[0.,0.,1.086],[0.,0.,.954]],cloud,self.pad_metadata())
        self.assertEqual(len(kept),1)
        self.assertAlmostEqual(kept[0][2],1.02)
        self.assertAlmostEqual(record['pad_vertical_span_bound_m'],np.hypot(.008,.001))
        self.assertFalse(record['simulator_object_geometry_used'])

    def test_no_full_pad_support_is_recorded_as_infeasible(self):
        cloud=np.c_[np.zeros(100),np.zeros(100),np.linspace(1.,1.01,100)]
        with self.assertRaisesRegex(ValueError,'No observed handle interval'):
            supported_handle_anchors([[0.,0.,1.005]],cloud,self.pad_metadata())

    def test_closing_endpoint_comes_from_stationary_depth_face(self):
        rng=np.random.default_rng(6)
        cabinet=np.c_[rng.uniform(.02,.25,1500),rng.normal(0.,.00005,1500),rng.uniform(.96,1.09,1500)]
        table=np.c_[rng.uniform(-.2,.2,2000),rng.uniform(-.2,.2,2000),np.full(2000,.90)]
        angle=np.deg2rad(80.)
        normal=np.array([-np.sin(angle),-np.cos(angle),0.])
        spec={'motion':'close','joint_origin_world_m':[0.,0.,1.],
              'joint_axis_world':[0.,0.,1.],'panel_normal_world':normal.tolist(),
              'front_plane':{'point_world_m':[0.,0.,1.]},'camera':'main'}
        cloud=np.c_[np.full(200,0.),np.full(200,-.24),np.linspace(.95,1.10,200)]
        cameras={'main':{'camera_to_world':np.eye(4)}}
        with patch('vlm_tamp.coupled_pipeline.observed_points',return_value=(np.vstack([cabinet,table]),{})):
            actual,evidence=terminal_angle(cameras,{},spec,cloud)
        self.assertAlmostEqual(actual,angle,delta=.001)
        self.assertFalse(evidence['simulator_scene_state_used'])

    def test_unobserved_closing_endpoint_is_not_silently_assumed(self):
        spec={'motion':'close','joint_origin_world_m':[0.,0.,1.],
              'joint_axis_world':[0.,0.,1.],'panel_normal_world':[-1.,0.,0.],
              'front_plane':{'point_world_m':[0.,0.,1.]},'camera':'main'}
        cloud=np.c_[np.zeros(200),np.full(200,-.24),np.linspace(.95,1.10,200)]
        with patch('vlm_tamp.coupled_pipeline.observed_points',return_value=(np.empty((0,3)),{})):
            with self.assertRaisesRegex(ValueError,'No observed stationary'):
                terminal_angle({'main':{'camera_to_world':np.eye(4)}},{},spec,cloud)


if __name__=='__main__':unittest.main()
