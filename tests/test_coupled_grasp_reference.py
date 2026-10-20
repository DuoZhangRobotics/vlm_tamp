"""Validated capture retains its measured offset instead of recentering the hand."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from vlm_tamp.coupled_pipeline import validated_grasp_reference
from vlm_tamp.coupled_manifold import CoupledHandlePoint
from test_coupled_manifold import fixture


class GraspReferenceTests(unittest.TestCase):
    def test_near_axis_offset_does_not_request_a_new_hand_translation(self):
        chain,rigid=fixture()
        pad_local=np.array([0.,0.,-.0036])
        tcp=rigid.reference_tcp
        pad=tcp[:3,3]+tcp[:3,:3]@pad_local
        observed=pad+[-.0015,.0004,.05]
        reference,record=validated_grasp_reference(pad,observed)
        problem=CoupledHandlePoint(chain,rigid.origin,rigid.axis,tcp,pad_local=pad_local,
            reference_anchor=reference,orientation_cone=np.pi/2,progress_bounds=(-.1,.5))
        np.testing.assert_array_equal(reference,pad)
        np.testing.assert_allclose(problem.evaluate(np.zeros(8))[0],0.,atol=1e-14)
        self.assertAlmostEqual(record['observed_pad_to_handle_distance_m'],np.hypot(.0015,.0004))
        self.assertTrue(record['measured_grasp_offset_preserved'])
        reference[0]+=1.
        self.assertNotEqual(reference[0],pad[0])  # The caller's measured array is not aliased.

    def test_measured_offset_rotates_with_the_hinge(self):
        chain,rigid=fixture()
        pad_local=np.array([0.,0.,-.0036])
        pad=rigid.reference_tcp[:3,3]+rigid.reference_tcp[:3,:3]@pad_local
        observed=pad+[-.0015,.0004,.05]
        reference,record=validated_grasp_reference(pad,observed)
        problem=CoupledHandlePoint(chain,rigid.origin,rigid.axis,rigid.reference_tcp,
            pad_local=pad_local,reference_anchor=reference,orientation_cone=np.pi/2,
            progress_bounds=(-.1,.5))
        angle=.4;rotation=Rotation.from_rotvec(angle*rigid.axis).as_matrix()
        bar=np.asarray(record['observed_axis_point_world_m'])
        moved_bar=rigid.origin+rotation@(bar-rigid.origin)
        expected=moved_bar+rotation@np.asarray(record['observed_axis_to_pad_offset_world_m'])
        np.testing.assert_allclose(problem.target_point(angle),expected,atol=1e-14)
        self.assertGreater(np.linalg.norm(rotation@(pad-bar)-(pad-bar)),.0005)

    def test_far_hand_and_nonfinite_points_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'too far'):
            validated_grasp_reference([.02,0.,1.],[0.,0.,1.])
        for bad in ([np.nan,0.,1.],[np.inf,0.,1.],[0.,1.]):
            with self.assertRaises(ValueError):
                validated_grasp_reference(bad,[0.,0.,1.])
        with self.assertRaises(ValueError):
            validated_grasp_reference([0.,0.,1.],[0.,0.,1.],maximum_distance=np.nan)


if __name__=='__main__':unittest.main()


def test_point_contact_keeps_jaw_tilt_as_an_inequality():
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from vlm_tamp.coupled_local_projection import project_nearby
    def forward(q,jacobian=False):
        pose=np.r_[[0.,0.,1.],Rotation.from_rotvec([0.,q[0],0.]).as_quat()]
        jac=np.zeros((6,7));jac[4,0]=1.
        return (pose,jac) if jacobian else pose
    chain=SimpleNamespace(limits=np.tile([-2.,2.],(7,1)),forward=forward)
    reference=np.eye(4);reference[2,3]=1.
    problem=CoupledHandlePoint(chain,[0,0,0],[0,0,1],reference,pad_local=[0,0,0],
        orientation_cone=np.pi/2,jaw_tilt_limit=np.pi/3,progress_bounds=(0,1))
    accepted=np.zeros(8);accepted[0]=np.pi/4
    rejected=accepted.copy();rejected[0]=np.deg2rad(70)
    assert problem.codimension==3
    assert problem.satisfied(accepted)
    assert not problem.satisfied(rejected)
    assert not project_nearby(problem,rejected,fixed_progress=0.)['accepted']
