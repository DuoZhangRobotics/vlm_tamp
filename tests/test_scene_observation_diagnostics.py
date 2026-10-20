import numpy as np

from vlm_tamp.scene_observation_diagnostics import depth_support


def camera(depth):
    return {'camera_to_world':np.eye(4),'intrinsic':np.eye(3),
        'depth':np.full((4,4),depth),'rgb':np.zeros((4,4,3),np.uint8)}


def test_depth_support_distinguishes_supported_free_and_occluded_points():
    points=np.array([[0,0,2.],[0,0,1.],[0,0,3.]])
    result,_,status=depth_support(points,{'a':camera(2.),'b':camera(2.)})
    assert status.tolist()==[1,2,0]
    assert result['supported_fraction']==1/3
    assert result['multi_view_free_space_conflict_fraction']==1/3
    assert not result['execution_veto'] and not result['model_changed']


def test_one_view_disagreement_is_uncertain_and_any_support_is_preserved():
    point=np.array([[0,0,2.]])
    _,_,status=depth_support(point,{'a':camera(3.),'b':camera(1.)})
    assert status[0]==0
    _,_,status=depth_support(point,{'a':camera(3.),'b':camera(2.)})
    assert status[0]==1
    _,_,status=depth_support(point,{'a':camera(float('nan')),'b':camera(3.)})
    assert status[0]==0
