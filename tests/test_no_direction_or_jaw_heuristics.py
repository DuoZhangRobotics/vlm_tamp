"""Task intent is caller-owned; a point constraint has no hidden wrist rule."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_sensor_geometry_check import camera,pixels
from vlm_tamp.handle_contact import door_model
from vlm_tamp.rekep_libero.mechanism_geometry import project
from vlm_tamp.partial_pose_arc import PartialPoseArc


def ground(hint,occupied,*,motion='close',fixed=None):
    view=camera([1,0,1.7]);pivot=np.array([.4,-.1,1.]);anchor=np.array([.4,.1,1.])
    choices={'camera':'a','hinge_pixels_xy':pixels(view,pivot[:2]).tolist(),
        'initial_image_direction_xy':hint,'motion':motion}
    face={'point':anchor,'rms':.001}
    cloud=np.c_[np.full(20,.4),np.full(20,.1),np.linspace(.95,1.05,20)]
    spec={'joint_family':'revolute','target_progress':1.,'motion':motion}
    return door_model({'a':view},choices,spec,{},view,face,np.array([1.,0.,0.]),
        np.ones((2,2),bool),cloud,[anchor],fixed_joint=fixed,occupied=occupied)


@pytest.mark.parametrize('sign',[-1,1])
def test_surrounding_depth_cannot_veto_or_reverse_caller_direction(sign):
    view=camera([1,0,1.7]);pivot=np.array([.4,-.1,1.]);anchor=np.array([.4,.1,1.])
    moved=pivot+Rotation.from_rotvec([0,0,.1*sign]).apply(anchor-pivot)
    hint=(project([moved],view)-project([anchor],view))[0].tolist()
    for points in [np.empty((0,3)),np.tile([.7,.1,1.],(200,1)),np.tile([.1,.1,1.],(200,1))]:
        mechanism,evidence,_,_=ground(hint,points)
        np.testing.assert_allclose(mechanism.axis,[0,0,sign])
        assert evidence['task_direction_status']=='caller_specified'
        assert not evidence['coordinate_convention_only']
        assert 'depth_prior_axis' not in evidence


@pytest.mark.parametrize('motion',['open','close'])
def test_missing_hint_returns_geometry_without_guessing_task_direction(motion):
    mechanism,evidence,_,_=ground(None,np.empty((0,3)),motion=motion)
    assert evidence['task_direction_status']=='unknown'
    assert evidence['coordinate_convention_only']
    assert mechanism.axis.tolist()==[0,0,1]
    assert not evidence['task_direction_verified']


def test_explicit_fixed_axis_is_retained_during_contact_refresh():
    fixed={'joint_origin_world_m':[.4,-.1,1.],'joint_axis_world':[0,0,-1],
        'direction_source':{'task_direction_status':'caller_specified','coordinate_convention_only':False}}
    mechanism,evidence,_,_=ground(None,np.tile([.7,.1,1.],(200,1)),fixed=fixed)
    assert mechanism.axis.tolist()==[0,0,-1]
    assert evidence['fixed_joint_preserved'] and evidence['task_direction_status']=='caller_specified'


def orientation_chain():
    def forward(q,jacobian=False):
        r=Rotation.from_rotvec([0.,q[0],0.]);pose=np.r_[[.2,0,1.],r.as_quat()]
        J=np.zeros((6,7));J[4,0]=1.
        return (pose,J) if jacobian else pose
    return SimpleNamespace(forward=forward,limits=np.tile([-3.,3.],(7,1)))


def test_point_target_has_no_jaw_tilt_or_wrist_orientation_constraint():
    model=PartialPoseArc(orientation_chain(),[0,0,1],[0,0,1],np.zeros(7),np.zeros(3))
    # A 90-degree wrist rotation makes the closing axis parallel to the hinge.
    q=np.zeros(7);q[0]=np.pi/2
    error,jac=model.residual(q,0.,jacobian=True)
    np.testing.assert_array_equal(error,np.zeros(3))
    assert jac.shape==(3,7)
    rigid=PartialPoseArc(orientation_chain(),[0,0,1],[0,0,1],np.zeros(7),np.zeros(3),grasp_constraint='rigid')
    assert np.linalg.norm(rigid.residual(q,0.)[3:])>0
