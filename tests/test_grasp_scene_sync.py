import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vlm_tamp.coupled_scene import Scene
from vlm_tamp.scene_pose_sync import transport_scene
from vlm_tamp.partial_pose_arc import PartialPoseArc
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain


def scene():
    return Scene.from_points([[3, 2, 1]], [[1, 0, 1], [1, .1, 1]], [[1.1, 0, 1]],
        {'joint_origin_world_m':[0,0,0], 'joint_axis_world':[0,0,1],
         'front_plane':{'point_world_m':[1,0,1]}, 'panel_normal_world':[1,0,0],
         'source_anchor_world_m':[1.1,0,1]})


def test_scene_pose_accumulates_between_queries_without_moving_static_obstacles():
    original=scene(); first=transport_scene(original,original.transform(.25),
        source='hand_motion_prediction',reference_step=20)
    second=transport_scene(first,first.transform(.25),
        source='hand_motion_prediction',reference_step=40)
    expected=Rotation.from_rotvec([0,0,.5]).apply(original.moving_reference)
    np.testing.assert_allclose(second.moving_reference,expected,atol=1e-12)
    np.testing.assert_array_equal(second.static_points,original.static_points)
    np.testing.assert_array_equal(original.moving_reference,[[1,0,1],[1,.1,1]])
    np.testing.assert_allclose(second.spec['joint_origin_world_m'],[0,0,0])
    assert second.pose_update['assumes_contact_retention']
    assert not second.pose_update['hinge_refined']
    assert not second.moving_reference.flags.writeable


def test_observed_transport_is_distinguished_from_prediction():
    original=scene()
    current=transport_scene(original,original.transform(.2),
        source='rgbd_rigid_motion',reference_step=20,fit={'pose_accepted':True})
    assert not current.record()['pose_update']['assumes_contact_retention']
    bad=np.eye(4);bad[0,0]=2
    with pytest.raises(ValueError):transport_scene(original,bad,source='rgbd_rigid_motion',reference_step=1)


@pytest.mark.parametrize('mode',['rigid','jaw_axis'])
def test_grasp_orientation_jacobian_and_projection(mode):
    tcp=np.eye(4);tcp[:3,3]=[1,0,1]
    chain=RobotChain({'reference_joints':[0.]*7,'joint_axes_world':[[0,0,1]]*7,
        'joint_anchors_world_m':[[0,0,0]]*7,'reference_tcp_world':tcp,
        'joint_limits':[[-3.,3.]]*7})
    model=PartialPoseArc(chain,[0,0,0],[0,0,1],np.zeros(7),[.03,.02,0],grasp_constraint=mode)
    q=np.array([.01,.03,-.01,.02,0.,.01,.02])
    residual,J=model.residual(q,.3,jacobian=True)
    numeric=np.column_stack([(model.residual(q+np.eye(7)[j]*1e-6,.3)
                             -model.residual(q-np.eye(7)[j]*1e-6,.3))/2e-6 for j in range(7)])
    np.testing.assert_allclose(J,numeric,atol=1e-7)
    solved,info=model.project(q,.6)
    assert info['accepted']
    assert np.linalg.norm(model.residual(solved,.6))<1e-6
