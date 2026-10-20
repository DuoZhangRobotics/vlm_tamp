import numpy as np

from vlm_tamp.rekep_libero.object_motion_v3 import verify_observed_transport


def cloud():
    rng=np.random.default_rng(3)
    return rng.uniform([-.04,-.03,.92],[.04,.03,.96],size=(1200,3))


def test_stationary_object_is_not_grasped_by_a_moving_tool():
    reference=cloud();delta=np.eye(4);delta[2,3]=.05
    result=verify_observed_transport(reference,reference,delta,[.012,-.012])
    assert not result['verified']


def test_observed_object_follows_tool_with_nonempty_closure():
    reference=cloud();delta=np.eye(4);delta[2,3]=.05
    result=verify_observed_transport(reference,reference+[0,0,.05],delta,[.012,-.012])
    assert result['verified']
    assert result['following_better']


def test_empty_closed_fingers_do_not_establish_retention():
    reference=cloud();delta=np.eye(4);delta[2,3]=.05
    result=verify_observed_transport(reference,reference+[0,0,.05],delta,[.0005,-.0005])
    assert not result['verified']
