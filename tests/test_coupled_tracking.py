"""Arrival under load must distinguish reaching joints from moving the door."""
import numpy as np
import pytest
from vlm_tamp.coupled_tracking import phase_arrival


def test_small_correction_is_not_consumed_by_absolute_joint_tolerance():
    path=np.zeros((2,8));path[-1,0]=.005;path[-1,7]=.012
    result=phase_arrival(path,np.zeros(7),.002)
    assert result['joint_target_reached']
    assert not result['arrived']
    assert result['directed_progress_fraction']<.2


def test_loaded_joint_residual_is_allowed_after_meaningful_phase_motion():
    path=np.zeros((2,8));path[-1,0]=.4;path[-1,7]=.8
    joints=path[-1,:7].copy();joints[0]-=.0036
    assert phase_arrival(path,joints,.795)['arrived']


def test_progress_check_respects_negative_motion_direction():
    path=np.zeros((2,8));path[:,7]=[.4,-.2]
    assert phase_arrival(path,np.zeros(7),-.18)['arrived']
    assert not phase_arrival(path,np.zeros(7),.45)['arrived']


def test_degenerate_phase_is_rejected():
    with pytest.raises(ValueError,match='nonzero'):
        phase_arrival(np.zeros((2,8)),np.zeros(7),0.)
