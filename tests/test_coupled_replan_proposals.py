"""Prospective path repair must not enlarge the measured binding allowance."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_coupled_door as runner
from try_rekep_bowl_v3 import save


class JawChain:
    limits=np.tile([-2.,2.],(7,1))

    def forward(self,q,jacobian=False):
        pose=np.r_[[0.,0.,1.],Rotation.from_rotvec([0.,q[0],0.]).as_quat()]
        jac=np.zeros((6,7));jac[4,0]=1.
        return (pose,jac) if jacobian else pose


class ClearScene:
    def __init__(self,*args,**kwargs):pass
    def check(self,*args,**kwargs):return {'accepted':True}
    def record(self):return {}


class ReplanProposalTests(unittest.TestCase):
    def test_adaptive_horizons_never_overshoot_a_short_remaining_goal(self):
        config={'contact_segment_angle_rad':.4,'contact_fallback_segment_angle_rad':.2}
        self.assertEqual(runner.contact_horizons(.1,config),[.1])
        self.assertEqual(runner.contact_horizons(.4,config),[.4,.2])
        self.assertEqual(runner.contact_horizons(.8,config),[.8,.4,.2])

    def test_invalid_horizons_cannot_reach_the_planning_dispatch(self):
        for goal,maximum,fallback in [(0.,.4,.2),(.8,.4,.5),(.8,.4,float('nan'))]:
            with self.assertRaises(ValueError):
                runner.contact_horizons(goal,{'contact_segment_angle_rad':maximum,
                                             'contact_fallback_segment_angle_rad':fallback})

    def call(self,folder,jaw_angle,dispatcher,warm):
        chain=JawChain();q=np.zeros(7);q[0]=jaw_angle
        metadata={'eef_pose_world_xyzw':chain.forward(q).tolist(),'joint_positions':q.tolist(),
                  'robot_collision_volumes_world':[]}
        spec={'joint_origin_world_m':[0.,0.,0.],'joint_axis_world':[0.,0.,1.],
              'source_anchor_world_m':[0.,0.,1.]}
        ik=SimpleNamespace(chain=chain,self_valid=lambda q:True)
        with patch.object(runner,'RobotVolumes',return_value=object()), \
             patch.object(runner,'Scene',ClearScene), \
             patch.object(runner,'pad_local',return_value=np.zeros(3)), \
             patch.object(runner,'world_pad',return_value=np.array([0.,0.,1.])), \
             patch.object(runner,'supported_handle_anchors',return_value=([],{'accepted':True})), \
             patch.object(runner,'observed_pad_capture',return_value={'maximum_axis_distance_m':.03}), \
             patch.object(runner,'isolated_plan',side_effect=dispatcher) as native:
            result=runner.fixed_contact_plan(ik,{'metadata':metadata},{},spec,np.array([[0.,0.,1.]]),
                np.ones((1,1),bool),{},.2,folder,{},warm_path=warm)
        return result,native

    def test_measured_start_keeps_original_point_zero_four_radian_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'plan'
            with self.assertRaisesRegex(ValueError,'Measured grasp cannot bind'):
                self.call(folder,.06,lambda *a,**k:self.fail('No planner may run for invalid measured binding'),None)
            binding=json.loads((folder/'binding.json').read_text())
            self.assertFalse(binding['local_projection']['accepted'])
            self.assertEqual(binding['local_projection']['maximum_allowed_joint_correction_rad'],.04)
            self.assertLessEqual(binding['measured_to_projected_start_max_joint_correction_rad'],.04+1e-12)

    def test_larger_hypothetical_goal_repair_still_requires_strict_validated_planning(self):
        dispatched=[]
        def plan(problem,start,goal,scene,folder,writer,**kwargs):
            dispatched.append(kwargs)
            if kwargs['strategy']=='warmstart':return np.empty((0,8)),{'exact_solution':False}
            self.assertEqual(kwargs['strategy'],'atlas')
            self.assertTrue(problem.satisfied(goal))
            return np.vstack([start,goal]),{'exact_solution':True}
        warm=np.zeros((2,8));warm[:,2]=.0;warm[-1,0]=.1;warm[-1,7]=.2
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'plan'
            result,_=self.call(folder,0.,plan,warm)
            reuse=json.loads((folder/'goal-reuse.json').read_text())
            binding=json.loads((folder/'binding.json').read_text())
            self.assertTrue(reuse['valid'])
            self.assertGreater(reuse['projection']['maximum_joint_correction_rad'],.04)
            self.assertLessEqual(reuse['projection']['maximum_joint_correction_rad'],.2)
            self.assertEqual(binding['measured_to_projected_start_max_joint_correction_rad'],0.)
            self.assertEqual(dispatched[0]['warm_maximum_joint_correction_rad'],.2)
            self.assertEqual(dispatched[1]['seconds'],60.)


if __name__=='__main__':unittest.main()
